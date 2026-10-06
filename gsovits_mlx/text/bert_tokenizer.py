"""BERT tokenizer via the `tokenizers` library (no transformers dependency).

The official pipeline loads chinese-roberta-wwm-ext-large through
transformers.AutoTokenizer, which is a thin wrapper around the SAME Rust
implementation (`tokenizers.Tokenizer`) serialized in tokenizer.json. Loading
that file directly is byte-identical for single-sentence encoding: same
BertNormalizer / BertPreTokenizer / WordPiece(21128) / [CLS]..[SEP] template.

The export lives next to the BERT weights: <models_root>/bert/tokenizer.json.
Verified against transformers AutoTokenizer on the full 21128 vocab and real
multi-language texts (tests/test_tokenizer.py) — exact id match.
"""

from __future__ import annotations

import functools
import os


@functools.lru_cache(maxsize=4)
def load_bert_tokenizer(tokenizer_json: str):
    """Load and cache a BertTokenizer from tokenizer.json. Returns HF BatchEncoding-
    like dicts via encode_text()."""
    from tokenizers import Tokenizer

    return Tokenizer.from_file(tokenizer_json)


def find_tokenizer_json(models_root: str) -> str:
    p = os.path.join(models_root, "bert", "tokenizer.json")
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"tokenizer.json not found at {p}; it ships with the converted "
            "bert export (copy from the original chinese-roberta-wwm-ext-large)")
    return p


def encode_text(tokenizer, text: str) -> dict:
    """Encode one sentence -> {"input_ids": [[int]], "attention_mask": [[int]]}.

    Mirrors AutoTokenizer(norm_text, return_tensors="np") for single inputs:
    batch-1 arrays, [CLS] ... [SEP], attention over all real tokens, no
    padding/truncation.
    """
    enc = tokenizer.encode(text)
    return {"input_ids": [list(enc.ids)], "attention_mask": [list(enc.attention_mask)]}
