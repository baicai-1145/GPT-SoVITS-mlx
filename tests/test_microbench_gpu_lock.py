"""CPU-only regression tests for the shared microbench GPU-lock path."""

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("linked_worktree", [False, True])
@pytest.mark.parametrize("absolute_common_dir", [False, True])
def test_repo_root_resolves_common_dir_against_git_cwd(
    monkeypatch, tmp_path, linked_worktree, absolute_common_dir
):
    source = Path(__file__).resolve().parents[1] / "tools" / "microbench.py"
    spec = importlib.util.spec_from_file_location("microbench_lock_test", source)
    microbench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(microbench)

    checkout = tmp_path / "checkout"
    script_dir = (
        checkout / ".pi" / "worktrees" / "worker" / "tools"
        if linked_worktree
        else checkout / "tools"
    )
    common_dir = checkout / ".git"
    git_output = (
        str(common_dir)
        if absolute_common_dir
        else os.path.relpath(common_dir, script_dir)
    )
    calls = []

    def git_run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(stdout=git_output, returncode=0)

    monkeypatch.setattr(microbench, "__file__", str(script_dir / "microbench.py"))
    monkeypatch.setattr(microbench.subprocess, "run", git_run)
    monkeypatch.chdir(tmp_path)

    assert microbench._repo_root() == str(checkout)
    assert len(calls) == 1
    assert calls[0][0] == ["git", "rev-parse", "--git-common-dir"]
    assert calls[0][1]["cwd"] == str(script_dir)
