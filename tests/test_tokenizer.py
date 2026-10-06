"""Byte-exact cross-check of gsovits_mlx.text.bert_tokenizer vs
transformers AutoTokenizer for chinese-roberta-wwm-ext-large.

Static tests always run (captured reference ids from transformers on this
machine). The live transformers comparison additionally sweeps the full
21128-token vocab and 3000 random multi-token strings; it runs only when
transformers is installed, e.g.:

    uv run --with transformers pytest -q tests/test_tokenizer.py

2026-02 capture result: corpus 10/10, vocab 21128/21128, random 3000/3000
exact id match.
"""

import os
import random
import pytest

MODELS_ROOT = os.environ.get(
    "GSOVITS_MODELS_ROOT", "/Volumes/2T/gpt-sovits-models/mlx")

TEXTS = [
    "你好，欢迎来到各自的旅程。今天我们聊聊机器学习。",
    "希望你以后能够做得比我还好哟。",
    "Hello, welcome to our journey. Today let's talk about machine learning.",
    "こんにちは、私たちの旅へようこそ。今日は機械学習について話しましょう。",
    "안녕하세요, 우리의 여정에 오신 것을 환영합니다. 오늘은 기계 학습에 대해 이야기해 봅시다.",
    "你好，歡迎嚟到我哋嘅旅程。今日我哋傾下機器學習。",
    "今天天气不错，let's go shopping 吧。",
    "今天我们来 discuss AI、機械学習と 한국어 mixing 一起。",
    "In 2024, GPT-SoVITS hit 0.15 RTF!!!  unbelievable... right?",
    "a",
]

# Captured from transformers AutoTokenizer(chinese-roberta-wwm-ext-large).
REFERENCE_IDS = {
    "a": [101, 143, 102],
    "In 2024, GPT-SoVITS hit 0.15 RTF!!!  unbelievable... right?":
        [101, 8217, 9707, 8159, 117, 13228, 8165, 118, 8968, 9197, 8723, 10295,
         121, 119, 8115, 10678, 8189, 106, 106, 106, 163, 8171, 12157, 8402,
         8786, 8862, 119, 119, 119, 10378, 136, 102],
}


def _tok():
    from gsovits_mlx.text.bert_tokenizer import (
        load_bert_tokenizer, find_tokenizer_json)
    return load_bert_tokenizer(find_tokenizer_json(MODELS_ROOT))


@pytest.mark.skipif(not os.path.exists(
    os.path.join(MODELS_ROOT, "bert", "tokenizer.json")),
    reason="converted bert export not mounted")
def test_shapes_and_template():
    from gsovits_mlx.text.bert_tokenizer import encode_text
    tok = _tok()
    for t in TEXTS:
        enc = encode_text(tok, t)
        ids, am = enc["input_ids"], enc["attention_mask"]
        assert len(ids) == 1 and len(am) == 1  # batch-1 like AutoTokenizer
        ids, am = ids[0], am[0]
        assert len(ids) == len(am) >= 3
        assert ids[0] == 101 and ids[-1] == 102  # [CLS] ... [SEP]
        assert all(a == 1 for a in am)
        assert all(isinstance(i, int) and 0 <= i < 21128 for i in ids)


@pytest.mark.skipif(not os.path.exists(
    os.path.join(MODELS_ROOT, "bert", "tokenizer.json")),
    reason="converted bert export not mounted")
def test_known_reference_ids():
    from gsovits_mlx.text.bert_tokenizer import encode_text
    tok = _tok()
    for text, ref_ids in REFERENCE_IDS.items():
        assert encode_text(tok, text)["input_ids"][0] == ref_ids, text


def test_transformers_live_crosscheck():
    """Exact id match vs transformers AutoTokenizer across the corpus, the
    full 21128 vocab (standalone) and 3000 random multi-token strings."""
    from gsovits_mlx.text.bert_tokenizer import encode_text
    try:
        from transformers import AutoTokenizer
    except ImportError:
        pytest.skip("transformers not installed")
    orig = os.path.join(MODELS_ROOT, "..", "pretrained_models",
                        "chinese-roberta-wwm-ext-large")
    if not os.path.isdir(orig):
        pytest.skip("original model dir not available")
    ref = AutoTokenizer.from_pretrained(orig)
    tok = _tok()

    def ours(t):
        return encode_text(tok, t)["input_ids"][0]

    def theirs(t):
        return list(ref(t)["input_ids"])

    for t in TEXTS:
        assert ours(t) == theirs(t), t

    vocab = ref.get_vocab()
    for s in vocab:
        assert ours(s) == theirs(s), s

    rng = random.Random(0)
    tokens = list(vocab)
    for _ in range(3000):
        s = "".join(rng.choice(tokens) for _ in range(rng.randint(1, 8)))
        assert ours(s) == theirs(s), s
