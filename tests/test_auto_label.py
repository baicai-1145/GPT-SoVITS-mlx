"""Stage-0 auto-label tests: text parsing, list emission, skip semantics.

GPU-dependent paths (separation/ASR) are heavy-marked elsewhere; these
tests cover the pure-Python logic. subprocess.run is monkeypatched via
pytest's tmp-safe monkey fixture (a bare assignment here once leaked the
fake into the whole suite and broke 5 resource-sensitive tests).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gsovits_mlx.train.auto_label import _parse_asr_text, transcribe  # noqa: E402


def test_parse_asr_text():
    assert _parse_asr_text("language Chinese<asr_text>你好。") == ("Chinese", "你好。")
    assert _parse_asr_text("language English<asr_text>Hmm.") == ("English", "Hmm.")
    assert _parse_asr_text("no tag") == ("", "no tag")


def test_transcribe_lang_flip_kept(tmp_path, monkeypatch):
    """force_lang mismatch keeps the transcript (sub-1s clip flip)."""
    import gsovits_mlx.train.auto_label as al

    class Fake:
        returncode = 0
        stdout = '{"text": "language English<asr_text>Hmm."}'
        stderr = ""

    monkeypatch.setattr(al.subprocess, "run", lambda *a, **k: Fake())
    assert transcribe(str(tmp_path / "x.wav"), "m", force_lang="zh") == "Hmm."


def test_transcribe_error_raises(tmp_path, monkeypatch):
    import gsovits_mlx.train.auto_label as al

    class FakeErr:
        returncode = 1
        stdout = ""
        stderr = "boom"

    monkeypatch.setattr(al.subprocess, "run", lambda *a, **k: FakeErr())
    try:
        transcribe(str(tmp_path / "x.wav"), "m")
        assert False, "should raise"
    except RuntimeError as e:
        assert "asr failed" in str(e)
