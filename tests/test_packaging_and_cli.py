"""Verification test for PyPI packaging and CLI entry points."""
import subprocess
import sys
import pytest

from gsovits_mlx.paths import default_models_root


def test_default_models_root_valid():
    root = default_models_root()
    assert isinstance(root, str)
    assert len(root) > 0


def test_gsovits_cli_help():
    r = subprocess.run([sys.executable, "-m", "gsovits_mlx.cli.main", "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0
    assert "infer" in r.stdout
    assert "train-s1" in r.stdout
    assert "train-s2" in r.stdout
    assert "download" in r.stdout


def test_download_cli_check():
    r = subprocess.run([sys.executable, "-m", "gsovits_mlx.download", "base", "--check"],
                       capture_output=True, text=True)
    assert r.returncode == 0
    assert "Destination:" in r.stdout


def test_subcommand_dispatch():
    r = subprocess.run([sys.executable, "-m", "gsovits_mlx.cli.main", "download", "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0
    assert "Download GPT-SoVITS-mlx weights" in r.stdout


def test_forwarder_tools_train_s1():
    r = subprocess.run([sys.executable, "tools/train_s1.py", "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0
    assert "--exp-dir" in r.stdout


def test_forwarder_tools_train_s2():
    r = subprocess.run([sys.executable, "tools/train_s2.py", "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0
    assert "--exp-dir" in r.stdout


def test_forwarder_tools_prepare():
    r = subprocess.run([sys.executable, "tools/prepare_data.py", "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0
    assert "--exp-dir" in r.stdout
