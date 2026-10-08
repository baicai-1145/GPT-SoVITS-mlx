"""MLX optimizers for GPT-SoVITS training: torch-exact AdamW and the k2 ScaledAdam.

AdamW
    Torch-update-math-exact AdamW (decoupled weight decay applied FIRST as
    ``p *= (1 - lr*wd)`` using the pre-update parameter, bias-corrected
    moments, fp32 m/v states). Gradients are treated as ALREADY unscaled:
    the GradScaler divides fp32 grads before ``apply_group``.

ScaledAdam
    Port of the official AR trainer's optimizer
    (GPT-SoVITS-cuda_graph_accel_v5/GPT_SoVITS/AR/modules/optim.py, Apache-2.0,
    (c) 2022 Xiaomi Corp., authors: Daniel Povey). Only the MATH is ported:
    the torch class's shape-batched stacking is a GPU kernel-launch
    optimization with no numerical effect (verified: per-tensor state, no
    stacking, reproduces the reference bit-for-bit modulo fp32 rounding —
    see tests/test_train_core.py::test_scaled_adam_golden).

    Reference quirks that are replicated faithfully (verified against the
    torch source line by line; deviating from "what the code obviously
    meant" would break golden parity):

    * clipping_scale (gradient-norm clipping vs a running median) multiplies
      the grad ONLY where it enters ``scale_grads`` (the per-tensor size/
      scale update). ``_step()`` re-reads ``p.grad`` — the UNCLIPPED grad —
      for exp_avg_sq and the momentum delta, so the main Adam-style update
      is never actually clipped. Upstream bug faithfully preserved.
    * the clipping ring buffer + threshold live per PARAM GROUP (torch calls
      _get_clipping_scale once per group); the ring is stored in the state
      of the group's first tensor.
    * clipping quartile "median": sorted_norms[min(period-1, (period//4)*2)]
      — index 2*period/4 of a sorted ascending window, with the last slot
      reachable.
    * ``_step()`` divides exp_avg_sq by bias_correction2 only when
      bias_correction2 < 0.99 (an approximation switch, not a numerical-
      stability guard).
    * scalar tensors (numel==1) take the regular-Adam path with lr scaled by
      scalar_lr_scale=0.1, and the parameter is clamped to ±scalar_max BEFORE
      adding delta.
    * param_rms is recomputed only every size_update_period steps (and only
      refreshed at steps where step % period == period-1); the alpha used in
      the main update is param_rms.clamp(min=param_min_rms).

    Defaults match the official s1 trainer: lr=1e-2, betas=(0.9, 0.98),
    size_update_period=4, clipping_scale=2.0 with clipping_update_period=1000.

WarmupCosineLRSchedule lives in schedule.py.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

__all__ = ["AdamW", "ScaledAdam"]


# ---------------------------------------------------------------------------
# AdamW (torch-exact)
# ---------------------------------------------------------------------------

class AdamW:
    """AdamW with param_groups, matching torch.optim.AdamW update math.

    Per-param state is kept in fp32 (m, v, step) regardless of the dtype of
    the parameters being optimized (the mixed-precision Trainer keeps fp32
    master params anyway).

    Update order (torch 2.x single-tensor path, ``decoupled_weight_decay``):
        1. step += 1
        2. p *= (1 - lr * weight_decay)               # decoupled wd FIRST,
                                                       # uses current p
        3. m  = beta1 * m + (1 - beta1) * g
        4. v  = beta2 * v + (1 - beta2) * g * g
        5. bc1 = 1 - beta1 ** step
        6. bc2 = 1 - beta2 ** step
        7. denom = sqrt(v) / sqrt(bc2) + eps
        8. p  -=  (lr / bc1) * m / denom

    ``param_groups``: list of dicts ``{"params": {name: mx.array}, "lr": ...,
    "weight_decay": ..., "grads": {name: mx.array}}``. All math runs in fp32;
    params are cast to fp32 for the update and cast back on store.
    """

    def __init__(
        self,
        params: dict | list,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.8, 0.99),
        eps: float = 1e-9,
        weight_decay: float = 0.01,
    ):
        self.defaults = dict(lr=lr, betas=tuple(betas), eps=eps,
                             weight_decay=weight_decay)
        if isinstance(params, dict):
            params = [params]
        self.param_groups = []
        for group in params:
            if isinstance(group, dict) and "params" in group:
                g = dict(self.defaults)
                g.update(group)
                g["params"] = dict(group["params"])
            else:
                g = dict(self.defaults)
                g["params"] = dict(group)
            g.setdefault("grads", {})
            g.setdefault("lr", lr)
            g.setdefault("weight_decay", weight_decay)
            self.param_groups.append(g)
        self._states = [dict() for _ in self.param_groups]  # name -> {m, v, step}
        self._step = 0

    # -- gradient plumbing ---------------------------------------------------
    def set_grads(self, group_index: int, grads: dict) -> None:
        """Attach fp32 grads {name: array} to a param group."""
        self.param_groups[group_index]["grads"] = grads

    # -- stepping -------------------------------------------------------------
    def apply_group(self, group_index: int, grads: dict) -> None:
        """Apply one update to the group's params from fp32 grads (unscaled)."""
        group = self.param_groups[group_index]
        states = self._states[group_index]
        lr = float(group["lr"])
        beta1, beta2 = group["betas"]
        eps = float(group["eps"])
        wd = float(group["weight_decay"])
        params = group["params"]
        updates = {}
        for name, g in grads.items():
            p32 = params[name].astype(mx.float32)
            st = states.get(name)
            if st is None:
                st = dict(m=mx.zeros_like(p32), v=mx.zeros_like(p32), step=0)
                states[name] = st
            st["step"] += 1
            step = st["step"]
            # decoupled weight decay first, on current p
            if wd != 0.0:
                p32 = p32 * (1.0 - lr * wd)
            m = st["m"] * beta1 + g.astype(mx.float32) * (1.0 - beta1)
            v = st["v"] * beta2 + g.astype(mx.float32) * g.astype(mx.float32) * (1.0 - beta2)
            st["m"], st["v"] = m, v
            bc1 = 1.0 - beta1 ** step
            bc2 = 1.0 - beta2 ** step
            denom = mx.sqrt(v) / math.sqrt(bc2) + eps
            upd = m * (lr / bc1) / denom
            new_p = p32 - upd
            orig_dtype = params[name].dtype
            updates[name] = new_p.astype(orig_dtype)
        params.update(updates)
        mx.eval(list(updates.values()), st["m"], st["v"])

    def step(self) -> None:
        """Apply updates for every group that has grads attached."""
        for gi, group in enumerate(self.param_groups):
            if group.get("grads"):
                self.apply_group(gi, group["grads"])

    # -- state io -------------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "step": self._step,
            "groups": [
                {
                    "lr": g["lr"],
                    "weight_decay": g["weight_decay"],
                    "states": {k: dict(m=v["m"], v=v["v"], step=v["step"])
                               for k, v in self._states[gi].items()},
                }
                for gi, g in enumerate(self.param_groups)
            ],
        }

    def load_state_dict(self, sd: dict) -> None:
        self._step = sd.get("step", 0)
        for gi, gsd in enumerate(sd["groups"]):
            self.param_groups[gi]["lr"] = gsd["lr"]
            self.param_groups[gi]["weight_decay"] = gsd["weight_decay"]
            self._states[gi] = {k: dict(m=v["m"], v=v["v"], step=v["step"])
                                for k, v in gsd["states"].items()}


# ---------------------------------------------------------------------------
# ScaledAdam (k2 port)
# ---------------------------------------------------------------------------

class ScaledAdam:
    """Math port of the official s1 ScaledAdam (see module docstring).

    ``param_groups``: list of dicts ``{"params": {name: mx.array}, "lr": ..,
    "grads": {name: mx.array}, ...}``. Constructor mirrors the torch one but
    params is a flat {name: array} dict per group (no nn.Parameter).

    All state is per-tensor; scalars (size-1 tensors) use the regular-Adam
    path. Params and states are fp32.
    """

    def __init__(
        self,
        params: dict | list,
        lr: float = 1e-2,
        betas: tuple[float, float] = (0.9, 0.98),
        scalar_lr_scale: float = 0.1,
        eps: float = 1e-8,
        param_min_rms: float = 1e-5,
        param_max_rms: float = 3.0,
        scalar_max: float = 10.0,
        size_update_period: int = 4,
        clipping_scale: float | None = 2.0,
        clipping_update_period: int = 1000,
    ):
        self.defaults = dict(
            lr=lr, betas=tuple(betas), scalar_lr_scale=scalar_lr_scale, eps=eps,
            param_min_rms=param_min_rms, param_max_rms=param_max_rms,
            scalar_max=scalar_max, size_update_period=size_update_period,
            clipping_scale=clipping_scale,
            clipping_update_period=clipping_update_period)
        if isinstance(params, dict):
            params = [params]
        self.param_groups = []
        for group in params:
            if isinstance(group, dict) and "params" in group:
                g = dict(self.defaults)
                g.update(group)
                g["params"] = dict(group["params"])
            else:
                g = dict(self.defaults)
                g["params"] = dict(group)
            g.setdefault("grads", {})
            self.param_groups.append(g)
        self._states = [dict() for _ in self.param_groups]
        self._group_clipping = [dict(model_norms=None, threshold=None,
                                     num_clipped=0)
                                for _ in self.param_groups]

    # -- gradient plumbing -----------------------------------------------------
    def set_grads(self, group_index: int, grads: dict) -> None:
        self.param_groups[group_index]["grads"] = grads

    def _init_state(self, group, p: mx.array) -> dict:
        """Init per-tensor state (fp32). p is the fp32 param view."""
        state = dict(step=0,
                     delta=mx.zeros_like(p),
                     exp_avg_sq=mx.zeros_like(p))
        if p.size > 1:
            param_rms = mx.sqrt(mx.mean(p * p)).reshape(1)
            state["param_rms"] = param_rms
            state["scale_exp_avg_sq"] = mx.zeros((1,))
            state["scale_grads"] = mx.zeros((group["size_update_period"], 1))
        return state

    # -- clipping -----------------------------------------------------------------
    def _get_clipping_scale(self, gi: int) -> float:
        """Replicates torch _get_clipping_scale for one param group.

        * returns 1.0 while any tensor in the group is uninitialized or the
          group's first-tensor step counter is 0 (torch checks the first
          batch's state before the states exist);
        * accumulates tot_sumsq = sum over tensors of (grad*param_rms)^2
          (raw grad^2 for size-1 tensors) into a per-group ring buffer of
          length clipping_update_period; every time step % period == 0 the
          threshold is refreshed to clipping_scale * "median" (sorted index
          min(period-1, period//4*2)) and num_clipped resets;
        * while step < period: no clipping;
        * else ans = min(1, threshold / (tot_norm + 1e-20)).
        """
        group = self.param_groups[gi]
        clipping_scale = group["clipping_scale"]
        states = self._states[gi]
        params = group["params"]
        grads = group["grads"]
        if not states:
            return 1.0
        first_name = next(iter(states))
        step = states[first_name]["step"]
        if clipping_scale is None or step == 0:
            return 1.0
        period = group["clipping_update_period"]
        clip = self._group_clipping[gi]
        tot = 0.0
        for name, st in states.items():
            g = grads[name]
            if params[name].size == 1:
                tot += float((g.astype(mx.float32) * g.astype(mx.float32)).sum())
            else:
                grms = g * st["param_rms"]
                tot += float((grms.astype(mx.float32) * grms.astype(mx.float32)).sum())
        tot_norm = float(np.sqrt(tot))
        if clip["model_norms"] is None:
            clip["model_norms"] = np.zeros(period)
        clip["model_norms"][step % period] = tot_norm
        if step % period == 0:
            srt = np.sort(clip["model_norms"])
            idx = min(period - 1, (period // 4) * 2)
            clip["threshold"] = clipping_scale * float(srt[idx])
            clip["num_clipped"] = 0
        if step < period:
            return 1.0
        ans = min(1.0, clip["threshold"] / (tot_norm + 1e-20))
        if ans < 1.0:
            clip["num_clipped"] += 1
        return ans

    # -- stepping -------------------------------------------------------------------
    def apply_group(self, group_index: int, grads: dict) -> None:
        gi = group_index
        group = self.param_groups[gi]
        group["grads"] = grads
        states = self._states[gi]
        params = group["params"]
        if not states or any(states[n]["step"] == 0 for n in params):
            clipping_scale = 1.0
        else:
            clipping_scale = self._get_clipping_scale(gi)
        updates = {}
        evals = []
        for name, p in params.items():
            g = grads[name].astype(mx.float32)
            st = states.get(name)
            if st is None:
                st = self._init_state(group, p.astype(mx.float32))
                states[name] = st
            new_p, new_st = self._step_one(group, p.astype(mx.float32), g, st,
                                           clipping_scale)
            updates[name] = new_p.astype(p.dtype)
            states[name] = new_st
            evals.extend([new_p, new_st["delta"], new_st["exp_avg_sq"]])
        params.update(updates)
        mx.eval(*evals)

    def step(self) -> None:
        for gi, group in enumerate(self.param_groups):
            if group.get("grads"):
                self.apply_group(gi, group["grads"])

    def _step_one(self, group, p, grad, state, clipping_scale):
        """One tensor's update. Mirrors torch _step_one_batch + _step/_step_scalar.

        Returns (new_param, new_state) without mutating inputs (functional
        style keeps MLX graphs sane). NOTE: like the reference, `grad` enters
        scale_grads multiplied by clipping_scale but the main update uses the
        UNCLIPPED grad (upstream quirk, see class docstring).
        """
        lr = group["lr"]
        size_update_period = group["size_update_period"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]

        step = state["step"]
        delta = state["delta"] * beta1

        if p.size > 1:
            scale_grads = list(state["scale_grads"])
            scale_grads[step % size_update_period] = \
                (p * grad * clipping_scale).sum().reshape(1)
            scale_grads = mx.stack(scale_grads, axis=0)
            if step % size_update_period == size_update_period - 1:
                param_rms = mx.sqrt(mx.mean(p * p)).reshape(1)
                if step > 0:
                    delta, scale_exp_avg_sq = self._size_update(
                        group, scale_grads, p, delta, state)
                else:
                    scale_exp_avg_sq = state["scale_exp_avg_sq"]
            else:
                param_rms = state["param_rms"]
                scale_exp_avg_sq = state["scale_exp_avg_sq"]
        else:
            param_rms = None
            scale_grads = None
            scale_exp_avg_sq = None

        if p.size == 1:
            p, delta, exp_avg_sq = self._step_scalar(group, p, grad, delta,
                                                     state)
        else:
            p, delta, exp_avg_sq, param_rms_used = self._step(
                group, p, grad, delta, state, param_rms)

        new_state = dict(step=step + 1, delta=delta, exp_avg_sq=exp_avg_sq)
        if p.size > 1:
            new_state["param_rms"] = param_rms
            new_state["scale_exp_avg_sq"] = scale_exp_avg_sq
            new_state["scale_grads"] = scale_grads
        return p, new_state

    def _size_update(self, group, scale_grads, p, delta, state):
        """Learn the per-tensor scale (torch _size_update). Returns
        (delta, scale_exp_avg_sq)."""
        beta1, beta2 = group["betas"]
        size_lr = group["lr"] * group["scalar_lr_scale"]
        param_min_rms = group["param_min_rms"]
        param_max_rms = group["param_max_rms"]
        eps = group["eps"]
        step = state["step"]
        size_update_period = scale_grads.shape[0]
        param_rms = mx.sqrt(mx.mean(p * p)).reshape(1)  # freshly recomputed

        beta2_corr = beta2 ** size_update_period
        mean_sq = mx.mean(scale_grads * scale_grads)
        scale_exp_avg_sq = state["scale_exp_avg_sq"] * beta2_corr + \
            mean_sq * (1.0 - beta2_corr)
        size_step = (step + 1) // size_update_period
        bias_correction2 = 1.0 - beta2_corr ** size_step
        denom = mx.sqrt(scale_exp_avg_sq) + eps
        scale_step = -size_lr * math.sqrt(bias_correction2) * \
            mx.sum(scale_grads) / denom
        is_too_small = param_rms < param_min_rms
        is_too_large = param_rms > param_max_rms
        scale_step = mx.where(is_too_small, mx.array(0.0), scale_step)
        scale_step = mx.where(is_too_large,
                              mx.array(-size_lr * size_update_period),
                              scale_step)
        delta = delta + p * scale_step * (1.0 - beta1)
        return delta, scale_exp_avg_sq

    def _step(self, group, p, grad, delta, state, param_rms):
        """Core update for numel>1 tensors (torch _step)."""
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        param_min_rms = group["param_min_rms"]

        exp_avg_sq = state["exp_avg_sq"] * beta2 + grad * grad * (1.0 - beta2)

        this_step = state["step"]
        bias_correction2 = 1.0 - beta2 ** (this_step + 1)
        if bias_correction2 < 0.99:
            exp_avg_sq_use = exp_avg_sq * (1.0 / bias_correction2)
        else:
            exp_avg_sq_use = exp_avg_sq
        denom = mx.sqrt(exp_avg_sq_use) + eps
        grad_n = grad / denom
        alpha = -lr * (1.0 - beta1) * mx.maximum(param_rms, param_min_rms)
        delta = delta + grad_n * alpha
        p = p + delta
        return p, delta, exp_avg_sq, param_rms

    def _step_scalar(self, group, p, grad, delta, state):
        """Scalar path (torch _step_scalar): regular Adam + ±scalar_max clamp
        applied to p BEFORE adding delta."""
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        scalar_max = group["scalar_max"]
        lr = group["lr"] * group["scalar_lr_scale"]

        exp_avg_sq = state["exp_avg_sq"] * beta2 + grad * grad * (1.0 - beta2)
        bias_correction2 = 1.0 - beta2 ** (state["step"] + 1)
        denom = mx.sqrt(exp_avg_sq / bias_correction2) + eps
        delta = delta + (grad / denom) * (-lr * (1.0 - beta1))
        p = mx.clip(p, -scalar_max, scalar_max)
        p = p + delta
        return p, delta, exp_avg_sq

    # -- state io ------------------------------------------------------------
    def state_dict(self) -> dict:
        def ser_state(st):
            out = dict(step=st["step"], delta=st["delta"],
                       exp_avg_sq=st["exp_avg_sq"])
            if "param_rms" in st:
                out["param_rms"] = st["param_rms"]
                out["scale_exp_avg_sq"] = st["scale_exp_avg_sq"]
                out["scale_grads"] = st["scale_grads"]
            return out

        groups = []
        for gi, g in enumerate(self.param_groups):
            groups.append({
                k: g[k] for k in
                ("lr", "betas", "scalar_lr_scale", "eps", "param_min_rms",
                 "param_max_rms", "scalar_max", "size_update_period",
                 "clipping_scale", "clipping_update_period")}
            | {"states": {k: ser_state(v) for k, v in self._states[gi].items()},
               "clipping": {k: (v.tolist() if v is not None and hasattr(v, "tolist") else v)
                            for k, v in self._group_clipping[gi].items()}})
        return {"groups": groups}

    def load_state_dict(self, sd: dict) -> None:
        for gi, gsd in enumerate(sd["groups"]):
            for k in ("lr", "betas", "scalar_lr_scale", "eps", "param_min_rms",
                      "param_max_rms", "scalar_max", "size_update_period",
                      "clipping_scale", "clipping_update_period"):
                if k in gsd:
                    self.param_groups[gi][k] = gsd[k]
            self._states[gi] = {k: dict(v) for k, v in gsd["states"].items()}
            clip = gsd.get("clipping", {})
            self._group_clipping[gi] = dict(
                model_norms=np.asarray(clip["model_norms"])
                if clip.get("model_norms") is not None else None,
                threshold=clip.get("threshold"),
                num_clipped=clip.get("num_clipped", 0))
