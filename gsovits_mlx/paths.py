"""Centralized path resolution for gsovits-mlx.

Ensures that model weights, cache directories, and front-end resources
resolve cleanly across git worktrees, standard checkouts, and pip installations.
"""
from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> str | None:
    """Return the git repository root if running inside a source checkout."""
    cand = Path(__file__).resolve().parent.parent
    if (cand / ".git").exists() or (cand / "models_local").exists():
        return str(cand)
    # Check parent chain
    cur = cand
    while cur.parent != cur:
        if (cur / ".git").exists():
            return str(cur)
        cur = cur.parent
    return None


def default_models_root() -> str:
    """Resolve the default directory for model weights.

    Precedence:
    1. GSOVITS_MODELS_ROOT env var
    2. Local models_local/ directory in repo or cwd
    3. User cache directory: ~/.cache/gsovits/models
    4. Legacy external disk: /Volumes/2T/gpt-sovits-models/mlx (if attached)
    """
    if env := os.environ.get("GSOVITS_MODELS_ROOT"):
        return os.path.abspath(env)

    if root := repo_root():
        local_cand = os.path.join(root, "models_local")
        if os.path.isdir(local_cand):
            return local_cand

    cwd_models = os.path.abspath("models_local")
    if os.path.isdir(cwd_models):
        return cwd_models

    legacy = "/Volumes/2T/gpt-sovits-models/mlx"
    if os.path.isdir(legacy):
        return legacy

    cache_dir = os.path.expanduser("~/.cache/gsovits/models")
    return cache_dir
