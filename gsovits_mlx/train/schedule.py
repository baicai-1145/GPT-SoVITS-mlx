"""Warmup+cosine LR schedule with the OFFICIAL locked-LR quirk.

Port of AR/modules/lr_schedulers.py WarmupCosineLRSchedule. The official
file hard-locks the lr (self.lr = lr = self.end_lr = 0.002 — "锁定用线性")
and every step() writes lr = 0.002 into every param group regardless of
configured init/peak/end LRs or warmup/cosine windows; the scheduled
value never escapes. Port default lr_locked=True keeps that byte-for-byte
(lr() always 0.002 after the first step). lr_locked=False enables the
intended math (linear warmup init→peak, cosine peak→end, constant after)
for experiments — NOT what official training used. step() writes
optimizer.param_groups[i]["lr"] like torch.
"""

from __future__ import annotations

import math

__all__ = ["WarmupCosineLRSchedule", "LOCKED_LR"]


LOCKED_LR = 0.002  # official s1 hard-lock ("不听话，直接锁定！")


class WarmupCosineLRSchedule:
    def __init__(
        self,
        optimizer,
        init_lr: float = 1e-6,
        peak_lr: float = 1e-4,
        end_lr: float = 1e-6,
        warmup_steps: int = 10000,
        total_steps: int = 400000,
        current_step: int = 0,
        lr_locked: bool = True,
    ):
        self.init_lr = init_lr
        self.peak_lr = peak_lr
        self.end_lr = end_lr
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self._warmup_rate = (peak_lr - init_lr) / warmup_steps if warmup_steps else 0.0
        self._current_step = current_step
        self.lr = init_lr
        self.lr_locked = lr_locked
        self._last_lr = [self.lr]

    def set_lr(self, lr: float) -> None:
        """Mirrors the official quirk: writes ``end_lr`` (which step() has just
        set to the locked 0.002), NOT the passed-in lr."""
        for g in self.optimizer.param_groups:
            g["lr"] = self.end_lr if self.lr_locked else lr
        self._last_lr = [g["lr"] for g in self.optimizer.param_groups]

    def step(self) -> float:
        if self._current_step < self.warmup_steps:
            lr = self.init_lr + self._warmup_rate * self._current_step
        elif self._current_step > self.total_steps:
            lr = self.end_lr
        else:
            decay_ratio = (self._current_step - self.warmup_steps) / \
                (self.total_steps - self.warmup_steps)
            if decay_ratio < 0.0 or decay_ratio > 1.0:
                raise RuntimeError(
                    "Decay ratio must be in [0.0, 1.0]. Fix LR scheduler settings.")
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            lr = self.end_lr + coeff * (self.peak_lr - self.end_lr)
        if self.lr_locked:
            self.lr = lr = self.end_lr = LOCKED_LR
        else:
            self.lr = lr
        self.set_lr(lr)
        self.lr = lr
        self._current_step += 1
        return self.lr
