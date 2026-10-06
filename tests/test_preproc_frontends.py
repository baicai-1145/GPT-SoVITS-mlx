"""Static parity/regression tests for the official text front-end.

Two layers:

* test_static_regression_cells — digest regression against
  tests/refs/frontend_static_refs.json (captured from the official CPUFast
  side by .tmp/parity_frontend_edge.py --official). Needs the refenv python
  (jieba_fast, pypinyin, ...) and the G2PW safetensors export; skips when
  they are absent.
* test_lang_segmenter_equivalence — per-run equivalence of our
  LangSegmenter re-port vs the official module when BOTH can be imported
  (needs fast_langdetect; skips otherwise).

GPU is never touched (mx.set_default_device(mx.cpu)).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

MODELS_ROOT = os.environ.get("GSOVITS_MODELS_ROOT",
                             "/Volumes/2T/gpt-sovits-models/mlx")
CPUFAST = os.environ.get("GPT_SOVITS_CPUFAST",
                         "/Users/baicai1145/repos/gpt-sovits/GPT-SoVITS-CPUFast")
G2PW_ENV = Path(__file__).parent.parent / ".tmp" / "refenv" / "bin" / "python"
CORPUS = [
    ("zh", "你好，欢迎来到各自的旅程。今天我们聊聊机器学习。", "zh", "v2"),
    ("en", "Hello, welcome to our journey. Today let's talk about machine learning.", "en", "v2"),
    ("ja", "こんにちは、私たちの旅へようこそ。今日は機械学習について話しましょう。", "ja", "v2"),
    ("ko", "안녕하세요, 우리의 여정에 오신 것을 환영합니다. 오늘은 기계 학습에 대해 이야기해 봅시다.", "ko", "v2"),
    ("yue", "你好，歡迎嚟到我哋嘅旅程。今日我哋傾下機器學習。", "yue", "v2"),
    ("mix_ze", "今天天气不错，let's go shopping 吧。", "zh", "v2"),
    ("mix4", "今天我们来 discuss AI、機械学習と 한국어 mixing 一起。", "auto", "v2"),
    ("digits", "在2024年GPT-SoVITS达到了0.15的RTF", "all_zh", "v2"),
    ("short", "你好。", "zh", "v2"),
    ("prompt", "希望你以后能够做得比我还好哟。", "zh", "v2"),
]


def _digest(phones, bert) -> str:
    return hashlib.sha256(
        np.asarray(phones, np.int32).tobytes()
        + np.asarray(bert, np.float32).tobytes()).hexdigest()[:16]


def _front_fe():
    from gsovits_mlx.text.preproc import TextFrontend, bootstrap

    bootstrap(models_root=MODELS_ROOT)
    return TextFrontend(models_root=MODELS_ROOT, device="cpu")


def _refenv_available() -> bool:
    if not os.path.exists(G2PW_ENV):
        return False
    if not (os.path.isdir(CPUFAST)
            and os.path.exists(os.path.join(CPUFAST, "GPT_SoVITS", "text",
                                            "G2PWModel", "g2pw.pth"))):
        return False
    try:
        import mlx.core  # noqa: F401
        import jieba_fast  # noqa: F401
        import pypinyin  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.static
def test_static_regression_cells():
    """Digest regression vs tests/refs/frontend_static_refs.json (captured
    from the official CPUFast side)."""
    refs_path = Path(__file__).parent / "refs" / "frontend_static_refs.json"
    if not refs_path.exists():
        pytest.skip("static refs not captured")
    if not _refenv_available():
        pytest.skip("refenv python / CPUFast / G2PW assets not available")
    refs = json.loads(refs_path.read_text())
    fe = _front_fe()
    for cell_json, want in refs["cells"].items():
        cell = json.loads(cell_json)
        if cell.get("kind") == "prompt":
            r = fe.segment_prompt(cell["text"], cell["lang"], cell["version"])
        else:
            r = fe.get_phones_and_bert(cell["text"], cell["lang"], cell["version"])
        assert _digest(r.phones, r.bert) == want, f"cell {cell} drifted"


@pytest.mark.static
def test_lang_segmenter_equivalence():
    """Our getTexts vs official module (when fast_langdetect is importable)."""
    try:
        import fast_langdetect  # noqa: F401
    except ImportError:
        pytest.skip("fast_langdetect not installed")
    if not os.path.isdir(CPUFAST):
        pytest.skip("CPUFast checkout missing")

    import sys

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from gsovits_mlx.text.lang_segmenter import get_texts as ours

    sys.path.insert(0, os.path.join(CPUFAST, "GPT_SoVITS"))
    sys.path.insert(0, CPUFAST)
    try:
        from text.LangSegmenter import LangSegmenter  # noqa: F401
    except ImportError:
        pytest.skip("official LangSegmenter import failed (missing deps)")

    texts = [t for _, t, _, _ in CORPUS] + [
        "MyGO?,你也喜欢まいご吗？",
        "当时ThinkPad T60刚刚发布，一同推出的还有一款名为Advanced Dock的扩展坞配件。",
        "ねえ、知ってる？最近、僕は天文学を勉強してるんだ。",
    ]
    for t in texts:
        for dl in ("", "zh", "ja", "ko"):
            assert ours(t, dl) == LangSegmenter.getTexts(t, dl), (t, dl)


def test_non_zh_bert_is_zero():
    """Official get_bert_inf semantics: only zh gets BERT features."""
    if not _refenv_available():
        pytest.skip("refenv python / CPUFast / G2PW assets not available")
    fe = _front_fe()
    r = fe.get_phones_and_bert("Hello world, this is a test.", "en", "v2")
    assert not r.bert.any()
    assert r.bert.shape[1] == len(r.phones)


def test_prompt_and_target_paths_differ():
    """'。'-prefix is pre_seg_text(target)-only; prompt path pads the tail."""
    if not _refenv_available():
        pytest.skip("refenv python / CPUFast / G2PW assets not available")
    fe = _front_fe()
    # target <4 chars first segment gets '。' prefix (preprocess ->
    # pre_seg_text); clean_text normalizes it to '.' in norm_text
    tgt = fe.preprocess("你好。", "zh", "cut0", "v2")
    assert tgt[0].norm_text == ".你好.", tgt[0].norm_text
    # prompt keeps its text, pads trailing separator instead
    pr = fe.segment_prompt("希望你以后能够做得比我还好哟。", "zh", "v2")
    assert pr.norm_text == "希望你以后能够做得比我还好哟."
    assert len(pr.phones) == 29  # stage-1 anchor (75/75 combined parity)


def test_v1_rejects_ko_yue():
    if not _refenv_available():
        pytest.skip("refenv python / CPUFast / G2PW assets not available")
    fe = _front_fe()
    with pytest.raises(ValueError):
        fe.get_phones_and_bert("안녕하세요.", "ko", "v1")
    with pytest.raises(ValueError):
        fe.get_phones_and_bert("你好，歡迎嚟到我哋嘅旅程。", "yue", "v1")
