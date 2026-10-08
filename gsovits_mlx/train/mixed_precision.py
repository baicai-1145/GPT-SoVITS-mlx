"""GradScaler for MLX: torch.amp.GradScaler("16-mixed") semantics.

Implements the official dynamic-loss-scaling state machine on top of MLX
graphs:

* ``scale(loss)`` multiplies the loss (and therefore every grad flowing from
  it) by a power-of-two scale so fp16 activations/grads stay representable.
* ``unscale_(optimizer)`` divides the (already-evaluated) fp32 grads of every
  param in ``optimizer.param_groups`` by the scale and records, per param
  group, whether any grad became non-finite (inf/nan) — ``mx.allfinite``-style
  check via ``mx.isfinite`` + ``mx.all`` (mlx 0.32.2 has no ``allfinite``).
* ``step(optimizer)`` applies the optimizer step only for groups whose grads
  were all finite; groups with inf/nan grads are skipped entirely (their
  optimizer state must NOT advance — matches torch, where found_inf per device
  gates the whole per-device step; here the per-param-group granularity is the
  natural MLX analogue because each group is stepped independently).
* ``update()`` maintains the growth/backoff counters exactly like
  ``torch.amp.GradScaler``:
    - every consecutive finite step increments ``_growth_tracker``; once it
      reaches ``growth_interval`` the scale is multiplied by
      ``growth_factor`` (typically x2) and the tracker resets;
    - any inf step multiplies the scale by ``backoff_factor`` (typically x0.5)
      immediately and resets the tracker.

Usage:
    scaler = GradScaler()
    loss = model(...)
    scaler.unscale_(opt)               # or let scaler.step do it
    scaler.step(opt, apply_fn=lambda o, grads: o.apply_grads(grads, params))
    scaler.update()

NOTE MLX grads are ordinary arrays here (trainers keep a flat ``{name: array}``
param/grad dict); ``unscale_`` mutates the grad dict in place.
"""

from __future__ import annotations

import mlx.core as mx

__all__ = ["GradScaler"]


def _all_finite(grads: dict) -> bool:
    for g in grads.values():
        if g.size == 0:
            continue
        if not bool(mx.all(mx.isfinite(g))):
            return False
    return True


class GradScaler:
    def __init__(
        self,
        init_scale: float = 65536.0,
        growth_interval: int = 2000,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
    ):
        self._scale = float(init_scale)
        self.growth_interval = int(growth_interval)
        self.growth_factor = float(growth_factor)
        self.backoff_factor = float(backoff_factor)
        self._growth_tracker = 0
        # per param-group found_inf flags from the last unscale_
        self.found_inf: dict[int, bool] = {}
        self._unscaled_for_step = False  # reset by update() each iteration

    # -- introspection -----------------------------------------------------
    def get_scale(self) -> float:
        return self._scale

    @property
    def scale(self) -> float:
        return self._scale

    # -- core ---------------------------------------------------------------
    def scale(self, loss: mx.array) -> mx.array:
        return loss * self._scale

    def unscale_(self, optimizer) -> None:
        """Divide grads by the current scale; set found_inf per param group.

        ``optimizer`` must expose ``param_groups``: a list of dicts with key
        ``grads`` (a ``{name: mx.array}`` dict, fp32) — the layout used by
        :class:`gsovits_mlx.train.optim.AdamW` / :class:`ScaledAdam`.
        """
        self.found_inf = {}
        for gi, group in enumerate(optimizer.param_groups):
            grads = group["grads"]
            if not grads:
                self.found_inf[gi] = False
                continue
            finite = _all_finite(grads)
            self.found_inf[gi] = not finite
            if finite:
                inv = 1.0 / self._scale
                for k in grads:
                    grads[k] = grads[k] * inv

    def step(self, optimizer, apply_fn=None, **apply_kwargs) -> bool:
        """Unscale (if not yet done) then step all groups with finite grads.

        ``apply_fn(optimizer, grads, **apply_kwargs)`` performs the actual
        parameter application; default expects the optimizer to expose
        ``apply_group(gi, grads)``. Returns True if any group stepped.
        Skipped groups keep their optimizer state untouched (no step
        increment, no moment updates).
        """
        if not self._unscaled_for_step:
            self.unscale_(optimizer)
            self._unscaled_for_step = True
        stepped = False
        for gi, group in enumerate(optimizer.param_groups):
            if self.found_inf.get(gi, False):
                continue
            grads = group["grads"]
            if apply_fn is not None:
                apply_fn(optimizer, grads, **apply_kwargs)
            else:
                optimizer.apply_group(gi, grads)
            stepped = True
        return stepped

    def update(self, new_scale: float | None = None) -> None:
        """Growth/backoff bookkeeping; call once after every ``step``.

        Pass ``new_scale`` to force a scale (torch's ``scaler.update(new_scale)``
        debugging path).
        """
        if new_scale is not None:
            self._scale = float(new_scale)
            self._growth_tracker = 0
            self._unscaled_for_step = False
            return
        self._unscaled_for_step = False
        if any(self.found_inf.values()):
            self._scale *= self.backoff_factor
            self._growth_tracker = 0
        else:
            self._growth_tracker += 1
            if self._growth_tracker >= self.growth_interval:
                self._scale *= self.growth_factor
                self._growth_tracker = 0
