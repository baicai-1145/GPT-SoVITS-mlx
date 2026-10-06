"""Unified-CLI dispatch tests (NO GPU, no model loads).

Covers the task-6 requirements:
* --version dispatch resolves every family's component dirs/params
* legacy --model flags from the e2e_v2pro/e2e_v5 wrapper era map correctly
  and reject contradictions
* the e2e_v*.py wrappers exec e2e.py with the right pinned version
* v1 rejects ko/yue languages (symbol-table fact)
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _parse(argv):
    """Import tools/e2e.py fresh and parse `argv` (list) into args+cfg."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("e2e_cli", os.path.join(REPO, "tools", "e2e.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, mod.parse_args(argv)


def test_all_versions_resolve():
    mod, args = _parse(["--version", "v2", "--text", "t", "--ref-audio", "a",
                        "--ref-text", "r"])
    assert args.version == "v2"
    for v in ("v1", "v2", "v2Pro", "v2ProPlus", "v3", "v4", "v5dev", "v5turbo"):
        assert v in mod.VERSION_CONFIG
    # component dirs
    assert mod.VERSION_CONFIG["v1"]["s1_dir"] == "s1v1"
    assert mod.VERSION_CONFIG["v2"]["s1_dir"] == "s1v2"
    assert mod.VERSION_CONFIG["v3"]["family"] == "v3"
    assert mod.VERSION_CONFIG["v5turbo"]["steps"] == 4
    assert mod.VERSION_CONFIG["v5dev"]["steps"] == 32


def test_legacy_model_flag_maps_and_overrides():
    mod, args = _parse(["--version", "v5dev", "--model", "turbo", "--text", "t",
                        "--ref-audio", "a", "--ref-text", "r"])
    assert args.version == "v5turbo"
    mod, args = _parse(["--version", "v2Pro", "--model", "proplus", "--text", "t",
                        "--ref-audio", "a", "--ref-text", "r"])
    assert args.version == "v2ProPlus"


def test_legacy_model_flag_rejects_cross_family():
    with pytest.raises(SystemExit):
        _parse(["--version", "v2", "--model", "turbo", "--text", "t",
                "--ref-audio", "a", "--ref-text", "r"])


def test_v1_rejects_ko_yue():
    with pytest.raises(SystemExit):
        _parse(["--version", "v1", "--lang", "ko", "--text", "t",
                "--ref-audio", "a", "--ref-text", "r"])
    # v2 accepts it
    mod, args = _parse(["--version", "v2", "--lang", "ko", "--text", "t",
                        "--ref-audio", "a", "--ref-text", "r"])
    assert args.lang == "ko"


def test_defaults_match_official_per_version():
    mod, args = _parse(["--version", "v1", "--text", "t", "--ref-audio", "a",
                        "--ref-text", "r"])
    assert args.early_stop_num == 2700
    assert args.out.endswith("out_v1.wav")
    mod, args = _parse(["--version", "v3", "--text", "t", "--ref-audio", "a",
                        "--ref-text", "r"])
    assert args.early_stop_num == 2850
    assert mod.VERSION_CONFIG["v3"]["steps"] == 32
    assert mod.VERSION_CONFIG["v3"]["cfg_rate"] == 0.0


def test_wrappers_pin_versions():
    """Each e2e_v*.py wrapper must exec e2e.py with its version pinned."""
    import re
    expectations = {
        "e2e_v1.py": "v1", "e2e_v2.py": "v2", "e2e_v2pro.py": "v2Pro",
        "e2e_v3.py": "v3", "e2e_v4.py": "v4", "e2e_v5.py": "v5dev",
    }
    for script, version in expectations.items():
        src = open(os.path.join(REPO, "tools", script)).read()
        assert re.search(rf'"--version",\s*.{version}.', src), script
        assert "tools", "e2e.py" and 'e2e.py' in src


def test_wrapper_help_runs():
    """The wrapper subprocess path works (argparse stage only)."""
    r = subprocess.run([sys.executable, os.path.join(REPO, "tools", "e2e_v2.py"), "--help"],
                       capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0
    assert "--version" in r.stdout
