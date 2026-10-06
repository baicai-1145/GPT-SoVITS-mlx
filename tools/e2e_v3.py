"""Thin wrapper: exec tools/e2e.py with v3 pinned.

Kept for compatibility with existing docs/scripts and tools/bench.py
(E2E_SCRIPTS spawns these paths); identical behavior and byte-identical
outputs to `tools/e2e.py --version v3`.
"""
import os
import subprocess
import sys

_EXTRA = []

if __name__ == "__main__":
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd = [sys.executable, os.path.join(repo, "tools", "e2e.py"),
           "--version", 'v3'] + _EXTRA + sys.argv[1:]
    sys.exit(subprocess.run(cmd).returncode)
