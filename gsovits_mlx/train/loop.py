"""Training-loop skeleton: mixed-precision (fp16 compute / fp32 master) Trainer
with JSONL loss logging and a whole-process footprint sampler.

fp16 policy (chosen: fp32 MASTER WEIGHTS + per-forward cast):
    The Trainer owns the authoritative parameters in fp32 (``master`` dicts,
    one per optimizer param group). For each forward pass it materializes a
    fp16 working copy (``cast_master_to_fp16``), runs the model's forward on
    those, and after every optimizer application copies the updated fp32
    values back — the fp16 copy is rebuilt from master each step, so there is
    exactly ONE source of truth and resume-from-checkpoint parity is trivial
    (checkpoints store the fp32 masters + optimizer states + step/epoch).

    Rationale vs the alternative (train directly on fp16 params, copy into
    fp32 after step): training on fp16 params accumulates rounding in the
    parameter values themselves and loses small updates (lr*m/denom can be
    below fp16 resolution ~6e-8·|p|), which official 16-mixed training never
    does — torch keeps fp32 params and casts per forward under autocast.
    This layout matches that semantic on MLX.

Memory accounting (per run, JSON ``memory_report()``):
    weights            -- analytic bytes of fp32 master params
    optimizer_states   -- analytic bytes from param shapes (AdamW: 2x param
                          bytes fp32; ScaledAdam: delta+exp_avg_sq+param_rms+
                          scale_exp_avg_sq+scale_grads)
    activations_peak   -- measured delta of mx.metal.get_active_memory across
                          forward/backward (0 when running on CPU)
    gpu_wired_peak     -- peak /usr/bin/footprint phys_footprint of OUR pid
                          sampled every 0.3 s by a background thread (the
                          closest public proxy for GPU-wired+resident; MLX
                          wired buffers show up in phys_footprint)
    total_peak_footprint -- the same sampler's whole-process peak

The sampler thread is daemon; stop it with ``stop_footprint_sampler()``.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time

import mlx.core as mx

__all__ = ["Trainer", "FootprintSampler", "analytic_bytes"]


# ---------------------------------------------------------------------------
# memory helpers
# ---------------------------------------------------------------------------

_DTYPE_BYTES = {
    mx.float32: 4, mx.float16: 2, mx.bfloat16: 2,
    mx.uint32: 4, mx.int32: 4, mx.uint8: 1, mx.bool_: 1,
}


def analytic_bytes(params: dict) -> int:
    """Total bytes of a {name: mx.array} dict (weights or optimizer states)."""
    total = 0
    for p in params.values():
        total += _DTYPE_BYTES.get(p.dtype, 4) * p.size
    return total


def optimizer_state_bytes(optim) -> int:
    """Analytic optimizer-state bytes from param shapes.

    AdamW: m + v fp32 per param. ScaledAdam: delta + exp_avg_sq fp32 plus,
    for numel>1 tensors, param_rms + scale_exp_avg_sq (1 float each) and
    scale_grads (size_update_period floats).
    """
    from .optim import AdamW, ScaledAdam

    total = 0
    for gi, group in enumerate(optim.param_groups):
        for name, p in group["params"].items():
            n = p.size
            if isinstance(optim, ScaledAdam):
                total += 2 * 4 * n
                if n > 1:
                    total += 4 * (1 + 1 + group["size_update_period"])
            elif isinstance(optim, AdamW):
                total += 2 * 4 * n
            else:
                total += 2 * 4 * n  # assume adam-like
    return total


class FootprintSampler(threading.Thread):
    """Sample own-pid /usr/bin/footprint every interval; track peak."""

    def __init__(self, interval: float = 0.3, pid: int | None = None):
        super().__init__(daemon=True)
        self.interval = interval
        self.pid = pid or os.getpid()
        self.peak = 0
        self.samples: list[int] = []
        self._stop = threading.Event()

    @staticmethod
    def read_phys_footprint(pid: int) -> int:
        """phys_footprint (bytes) of pid via /usr/bin/footprint; 0 on failure."""
        import re
        try:
            out = subprocess.run(["/usr/bin/footprint", "-j", "/dev/stdout", str(pid)],
                                 capture_output=True, text=True,
                                 timeout=10).stdout
        except Exception:
            return 0
        # JSON goes to the -j file; stdout holds the human table. Use the
        # human table header instead ("Footprint: 309 MB"): simpler + same value.
        try:
            out2 = subprocess.run(["/usr/bin/footprint", str(pid)],
                                  capture_output=True, text=True,
                                  timeout=10).stdout
        except Exception:
            return 0
        m = re.search(r"phys_footprint:\s*(\d+)", out)
        if not m:
            mm = re.search(r"Footprint:\s+([\d.]+)\s+(B|KB|MB|GB)\b", out2)
            if not mm:
                return 0
            unit = {"B": 1, "KB": 2**10, "MB": 2**20, "GB": 2**30}[mm.group(2)]
            return int(float(mm.group(1)) * unit)
        return int(m.group(1))

    def run(self):
        while not self._stop.is_set():
            fp = self.read_phys_footprint(self.pid)
            if fp:
                self.samples.append(fp)
                if fp > self.peak:
                    self.peak = fp
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()
        self.join(timeout=5)


def metal_active_memory() -> int:
    """mx.metal.get_active_memory() if available and on GPU, else 0."""
    try:
        return int(mx.metal.get_active_memory())
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Epoch/step driver around a (model-forward, optimizer, scheduler, scaler)
    tuple. Sub-classes / callers provide a ``forward_fn(params_fp16, batch)``
    returning an fp32 loss (already averaged over grad_accum micro-batches
    handled by the caller; this class divides by grad_accum_steps itself).

    checkpoint hooks: ``on_checkpoint(step, epoch)`` called every
    ``checkpoint_every`` steps — implementers call ckpt.save_resume(...).
    """

    def __init__(
        self,
        model,
        optimizers,
        scheduler=None,
        scaler=None,
        grad_accum_steps: int = 1,
        log_interval: int = 10,
        checkpoint_every: int = 1000,
        on_checkpoint=None,
        loss_log_path: str | None = None,
        sample_footprint: bool = True,
        footprint_interval: float = 0.3,
    ):
        self.model = model
        self.optimizers = list(optimizers) if isinstance(optimizers, (list, tuple)) \
            else [optimizers]
        self.scheduler = scheduler
        self.scaler = scaler
        self.grad_accum_steps = max(1, int(grad_accum_steps))
        self.log_interval = log_interval
        self.checkpoint_every = checkpoint_every
        self.on_checkpoint = on_checkpoint
        self.step = 0
        self.epoch = 0
        self.loss_log_path = loss_log_path
        self._losses: list[dict] = []
        self._fp_sampler = None
        self._act_peak = 0
        self._weights_bytes = sum(analytic_bytes(g["params"])
                                  for o in self.optimizers
                                  for g in o.param_groups)
        self._opt_bytes = sum(optimizer_state_bytes(o) for o in self.optimizers)
        if sample_footprint:
            self.start_footprint_sampler(footprint_interval)

    # -- mixed-precision plumbing -------------------------------------------
    @staticmethod
    def cast_master_to_fp16(params: dict) -> dict:
        return {k: v.astype(mx.float16) for k, v in params.items()}

    def sync_masters(self) -> None:
        """Optimizer param_groups already hold the fp32 masters (AdamW/ScaledAdam
        update them in place); nothing to copy. Kept as the explicit sync point
        for alternative trainer layouts."""
        pass

    # -- footprint -------------------------------------------------------------
    def start_footprint_sampler(self, interval: float = 0.3) -> None:
        if self._fp_sampler is not None:
            return
        self._fp_sampler = FootprintSampler(interval=interval)
        self._fp_sampler.start()

    def stop_footprint_sampler(self) -> None:
        if self._fp_sampler is not None:
            self._fp_sampler.stop()
            self._fp_sampler = None

    def memory_report(self) -> dict:
        wired = self._fp_sampler.peak if self._fp_sampler else 0
        return {
            "weights": self._weights_bytes,
            "optimizer_states": self._opt_bytes,
            "activations_peak": self._act_peak,
            "gpu_wired_peak": wired,
            "total_peak_footprint": wired,
        }

    def dump_memory_json(self, path: str) -> dict:
        rep = self.memory_report()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(rep, f, indent=1)
        return rep

    # -- step -----------------------------------------------------------------
    def train_step(self, batch, forward_fn, backward_fn) -> float:
        """One optimizer step over grad_accum_steps micro-batches.

        forward_fn(params16, batch) -> loss (scalar mx.array, fp32 expected)
        backward_fn(params, grads_dict, loss) -> grads {name: fp32 array}
        (implementation-specific; typically mx.grad over the flat param dict).

        Returns the unscaled loss value (python float).
        """
        loss_acc = 0.0
        grad_sums = None
        self._act_before = metal_active_memory()
        for _ in range(self.grad_accum_steps):
            params16_by_group = [self.cast_master_to_fp16(g["params"])
                                 for g in
                                 (g for o in self.optimizers for g in o.param_groups)]
            # one flat dict across all groups for the forward
            flat16 = {}
            for d in params16_by_group:
                flat16.update(d)
            loss = forward_fn(flat16, batch)
            if self.scaler is not None:
                loss = self.scaler.scale(loss)
            grads = backward_fn(flat16, loss)
            mx.eval(loss)
            floss = float(loss)
            if self.scaler is not None:
                floss /= self.scaler.get_scale()
            loss_acc += floss / self.grad_accum_steps
            g32 = {k: (v.astype(mx.float32) / self.grad_accum_steps) for k, v in grads.items()}
            if grad_sums is None:
                grad_sums = g32
            else:
                grad_sums = {k: grad_sums[k] + g32[k] for k in grad_sums}
        self._act_peak = max(self._act_peak,
                             metal_active_memory() - self._act_before)

        # route grads into groups (by param membership)
        all_params = {}
        for o in self.optimizers:
            for gi, g in enumerate(o.param_groups):
                for name in g["params"]:
                    all_params[name] = (o, gi)
        for name, (o, gi) in all_params.items():
            if name not in grad_sums:
                raise KeyError(f"missing grad for param {name}")
        for o in self.optimizers:
            for gi, g in enumerate(o.param_groups):
                g["grads"] = {k: grad_sums[k] for k in g["params"]}

        if self.scaler is not None:
            self.scaler.unscale_(self.optimizers[0])
            # unscale remaining optimizers with the same scale
            for o in self.optimizers[1:]:
                for gi in range(len(o.param_groups)):
                    group = o.param_groups[gi]
                    inv = 1.0 / self.scaler.get_scale()
                    finite = all(bool(mx.all(mx.isfinite(x))) for x in group["grads"].values())
                    if finite:
                        group["grads"] = {k: v * inv for k, v in group["grads"].items()}
            stepped_any = False
            for o in self.optimizers:
                for gi in range(len(o.param_groups)):
                    group = o.param_groups[gi]
                    inf = False
                    if o is self.optimizers[0]:
                        inf = self.scaler.found_inf.get(gi, False)
                    else:
                        inf = not all(bool(mx.all(mx.isfinite(x)))
                                      for x in group["grads"].values())
                    if not inf:
                        o.apply_group(gi, group["grads"])
                        stepped_any = True
            self.scaler.update()
        else:
            for o in self.optimizers:
                o.step()

        if self.scheduler is not None:
            self.scheduler.step()
        self.step += 1
        self._log(loss_acc)
        if self.checkpoint_every and self.step % self.checkpoint_every == 0 \
                and self.on_checkpoint:
            self.on_checkpoint(self.step, self.epoch)
        return loss_acc

    def train_epoch(self, dataloader, forward_fn, backward_fn, epoch=None):
        if epoch is not None:
            self.epoch = epoch
        for batch in dataloader:
            self.train_step(batch, forward_fn, backward_fn)
        self.epoch += 1
        return self.epoch

    # -- logging ----------------------------------------------------------------
    def _log(self, loss: float) -> None:
        rec = {"step": self.step, "epoch": self.epoch, "loss": loss}
        if self.step % self.log_interval == 0 or self.step == 1:
            self._losses.append(rec)
            if self.loss_log_path:
                os.makedirs(os.path.dirname(self.loss_log_path) or ".",
                            exist_ok=True)
                with open(self.loss_log_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
