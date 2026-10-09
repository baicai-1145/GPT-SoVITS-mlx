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

LOCK_STALE_SEC = 15 * 60


def _repo_root() -> str:
    """Main-repo root for THIS checkout, even inside a git worktree.

    All agents (including teammates in .pi/herdr-team/*/worktrees/*, which
    physically live INSIDE the main checkout's tree) must serialize on ONE
    lock: the main checkout's .tmp/gpu.lock.d. A worktree-local lock is
    invisible to processes launched from the main repo and has caused a
    3-way concurrent-GPU OOM + watchdog reboot (2026-10-09) — never create
    or honor one.

    Resolution: walk up from this file's location (source of truth, cwd
    independent) to the outermost directory containing a .git marker. For
    the main checkout that is the repo root; for a worktree copy of this
    package it is the worktree root, whose PARENT CHAIN still leads to the
    main root — so we keep walking to the topmost .git-bearing ancestor,
    which for in-tree worktrees is the MAIN repo root.
    """
    best = os.path.dirname(os.path.abspath(__file__))
    d = best
    while True:
        parent = os.path.dirname(d)
        if os.path.exists(os.path.join(parent, ".git")):
            best = parent  # remember topmost .git-bearing ancestor
        if parent == d:
            return best
        d = parent


def lock_dir() -> str:
    """Canonical lock dir: <main-repo-root>/.tmp/gpu.lock.d (single lock)."""
    return os.path.join(_repo_root(), ".tmp", "gpu.lock.d")


def _find_lock() -> tuple[str, str] | None:
    """Return (owner_text, lock_dir) for the CANONICAL main-repo lock.

    Historical note: this used to walk up from CWD and accept the first
    gpu.lock.d found, which let worktree-local locks authorize GPU use —
    mutual exclusion broke (concurrent trainers, machine OOM/watchdog
    reboot). Now only the single main-repo lock counts; CWD is irrelevant.
    """
    lock = lock_dir()
    if not os.path.isdir(lock):
        return None
    try:
        owner = open(os.path.join(lock, "owner")).read().strip()
    except OSError:
        owner = ""
    return owner, lock


def acquire_lock(owner_text: str) -> str:
    """Create the canonical main-repo lock; SystemExit if already held.

    Every GPU run MUST acquire via this helper (atomic mkdir; owner file
    names tool+task+agent+timestamp). Raises SystemExit with the current
    owner if the lock exists — never work around it by making a local lock.
    """
    lock = lock_dir()
    try:
        os.makedirs(lock, exist_ok=False)
    except FileExistsError:
        try:
            cur = open(os.path.join(lock, "owner")).read().strip()
        except OSError:
            cur = "<unreadable>"
        raise SystemExit(
            f"[gpu.lock] held by {cur!r}; ONE GPU process at a time "
            "(AGENTS.md). Wait, or take over only if stale (>15 min, "
            "then rm -rf the dir and re-acquire).")
    with open(os.path.join(lock, "owner"), "w") as f:
        f.write(f"{owner_text} {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    return lock


def refresh_lock() -> None:
    """Touch the owner file (long tasks: refresh at least every 15 min)."""
    os.utime(os.path.join(lock_dir(), "owner"))


def release_lock() -> None:
    """Remove the lock (no-op when absent). Only the owner releases."""
    import shutil
    shutil.rmtree(lock_dir(), ignore_errors=True)


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
