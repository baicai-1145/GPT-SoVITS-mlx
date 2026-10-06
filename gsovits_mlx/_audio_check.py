"""Post-run audio sanity gate: a silent/zero wav must never be written.

Shared by every tools/e2e_v*.py call site (and bench.py children inherit it
through the scripts they spawn). Phase-2 bug this prevents: the CPU-pinned
device gate let the v3-family CFM path silently produce all-zero audio —
the run "succeeded" and wrote a valid-looking 309 s silent file. Cheap
numerical checks, hard error on failure.
"""

from __future__ import annotations

import sys

import numpy as np


def assert_audible(wav: np.ndarray, context: str = "") -> None:
    """Raise SystemExit if `wav` is silent/zero/degenerate.

    Gates (tuned for TTS output, speech-band energy):
      * non-empty and finite
      * peak amplitude > 0.01 (all-zero/near-silent render)
      * RMS > 1e-4
    """
    a = np.asarray(wav)
    if a.ndim > 1:
        a = a.mean(axis=1)
    ctx = f" ({context})" if context else ""
    if a.size == 0:
        raise SystemExit(f"[audio-check] EMPTY wav{ctx} — refusing to write")
    if not np.isfinite(a).all():
        raise SystemExit(
            f"[audio-check] NON-FINITE samples in wav{ctx} — refusing to write "
            "(NaN/Inf; check the decode path/device)")
    peak = float(np.abs(a).max())
    rms = float(np.sqrt(np.mean(a.astype(np.float64) ** 2)))
    if peak <= 0.01:
        raise SystemExit(
            f"[audio-check] SILENT wav (peak={peak:.2e}){ctx} — refusing to write. "
            "Likely cause: pipeline ran on a device that silently produced zeros "
            "(CPU is NOT supported for AR/CFM decode paths — run with --gpu "
            "under the lock, or --frontend-only for front-end smoke tests).")
    if rms <= 1e-4:
        raise SystemExit(
            f"[audio-check] DEGENERATE wav (peak={peak:.3f} but rms={rms:.2e}){ctx} "
            "— refusing to write.")
