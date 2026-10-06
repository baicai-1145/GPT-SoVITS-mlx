"""Process-wide MLX device gate (GPU-lock discipline).

Contract (single source of truth for tools/e2e_v*.py and tools/bench.py):

* DEFAULT-DENY: entry scripts pin mlx to the CPU device at main() start.
  Nothing in the process issues Metal commands unless the run explicitly
  opted in AND the gpu.lock is held.
* Opt-in is either the --gpu flag or env GSOVITS_GPU_LOCK_OK=1, and in both
  cases .tmp/gpu.lock.d must exist with a fresh (<=45 min) owner file. When
  env GSOVITS_RUN_TAG is set, the owner file must contain it; otherwise any
  non-empty owner passes (the opt-in itself asserts "the lock is mine").
* CPU-pinned and GPU runs are numerically identical on this machine (MLX
  kernels are device-deterministic for these ops); the lock exists to keep
  Metal memory/time serial across agents, not because results differ.

Call resolve_device() exactly once, before creating any mx arrays.
"""

from __future__ import annotations

import os
import time

LOCK_STALE_SEC = 45 * 60


def _find_lock() -> tuple[str, str] | None:
    """Return (owner_text, lock_dir) for the first gpu.lock.d found walking
    up from CWD (covers the main checkout and any worktrees inside it)."""
    d = os.getcwd()
    while True:
        lock = os.path.join(d, ".tmp", "gpu.lock.d")
        if os.path.isdir(lock):
            try:
                owner = open(os.path.join(lock, "owner")).read().strip()
            except OSError:
                owner = ""
            return owner, lock
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


def lock_status() -> tuple[bool, str]:
    """(held_and_fresh, owner_text) — diagnostic helper."""
    found = _find_lock()
    if not found:
        return False, ""
    owner, lock = found
    try:
        fresh = (time.time() - os.path.getmtime(os.path.join(lock, "owner"))) < LOCK_STALE_SEC
    except OSError:
        fresh = False
    return (bool(owner) and fresh), owner


def resolve_device(flag_gpu: bool = False, verbose: bool = False) -> str:
    """Returns "gpu" only when opt-in + a fresh lock are both present;
    otherwise pins mlx to CPU and returns "cpu"."""
    opt_in = flag_gpu or os.environ.get("GSOVITS_GPU_LOCK_OK") == "1"
    held, owner = lock_status()
    run_tag = os.environ.get("GSOVITS_RUN_TAG", "")
    tag_ok = (run_tag in owner) if run_tag else bool(owner)
    if opt_in and held and tag_ok:
        if verbose:
            print(f"[device] gpu (lock owner: {owner})", flush=True)
        return "gpu"
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    if verbose:
        why = "no --gpu/GSOVITS_GPU_LOCK_OK opt-in" if not opt_in else \
              (f"lock not fresh/ours (owner={owner!r})" if owner else "no .tmp/gpu.lock.d")
        print(f"[device] cpu ({why})", flush=True)
    return "cpu"


def require_gpu_for_pipeline(device: str, cpu_ok_flag: bool = False) -> None:
    """Hard-stop CPU pipeline runs (phase-2 bug: CPU-pinned v3 CFM silently
    produced a 309 s all-zero wav at 100x slowdown instead of failing fast).

    CPU is not a supported compute path for AR/CFM/decode — for ANY version
    (v1/v2 render slowly-but-validly, which is a trap, not a feature).
    Front-end-only smoke stays CPU-legal; call sites pass cpu_ok_flag=True
    only for explicit --frontend-only / --cpu-i-know-broken runs.
    """
    if device != "cpu" or cpu_ok_flag:
        return
    raise SystemExit(
        "[device] REFUSING full-pipeline run on CPU.\n"
        "  CPU is not a supported compute path for AR/CFM/decode (v3-family"
        " CFM silently yields all-zero audio; v1/v2 render 30-100x slow).\n"
        "  * real synthesis: take the gpu.lock and pass --gpu (or set"
        " GSOVITS_GPU_LOCK_OK=1)\n"
        "  * front-end smoke only: pass --frontend-only\n"
        "  * truly force it anyway: pass --cpu-i-know-broken")
