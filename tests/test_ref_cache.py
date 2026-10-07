"""Ref-prompt cache unit tests (fake files, mtime+size keying, env fingerprint).

No GPU, no model loads — pure cache-contract tests.
"""

from __future__ import annotations

import json
import os
import tempfile
from time import time

import numpy as np
import pytest

from gsovits_mlx.text import ref_cache


@pytest.fixture()
def cache_dir(tmp_path):
    return str(tmp_path / "cache")


@pytest.fixture()
def ref_wav(tmp_path):
    p = tmp_path / "ref.wav"
    p.write_bytes(b"RIFFfake")
    return str(p)


def test_roundtrip(cache_dir, ref_wav):
    key = ref_cache.cache_key(ref_wav, "你好", "zh", "v2")
    meta, arrays, hit = ref_cache.cached_call(
        cache_dir, key, lambda: {"ref_text": "你好"},
        lambda: {"p_ids": np.arange(5, dtype=np.int32),
                 "p_bert": np.ones((4, 5), np.float32)})
    assert hit is False
    meta, arrays, hit = ref_cache.cached_call(
        cache_dir, key, lambda: {"ref_text": "should-not-run"},
        lambda: pytest.fail("compute_fn must not run on a cache hit"))
    assert hit is True
    assert arrays["p_ids"].tolist() == [0, 1, 2, 3, 4]
    assert meta["ref_text"] == "你好"


def test_mtime_key_changes(cache_dir, ref_wav):
    k1 = ref_cache.cache_key(ref_wav, "你好", "zh", "v2")
    os.utime(ref_wav, (time() + 5, time() + 5))
    k2 = ref_cache.cache_key(ref_wav, "你好", "zh", "v2")
    assert k1 != k2


def test_text_and_lang_key_changes(cache_dir, ref_wav):
    assert ref_cache.cache_key(ref_wav, "a", "zh", "v2") != \
        ref_cache.cache_key(ref_wav, "b", "zh", "v2")
    assert ref_cache.cache_key(ref_wav, "a", "zh", "v2") != \
        ref_cache.cache_key(ref_wav, "a", "en", "v2")


def test_env_fingerprint_in_key_and_meta(cache_dir, ref_wav):
    fp = ref_cache.env_fingerprint()
    assert "mlx" in fp and "numpy" in fp or "/" in fp  # py/mlx/mlx-metal/numpy
    key = ref_cache.cache_key(ref_wav, "x", "zh", "v2")
    ref_cache.put_cached(cache_dir, key, {"x": 1}, {"v": np.zeros(1, np.float32)})
    meta, _ = ref_cache.get_cached(cache_dir, key)
    assert meta["env"] == fp
    assert meta["format"] == ref_cache.FORMAT_TAG


def test_v1_format_rejected(cache_dir, ref_wav):
    key = ref_cache.cache_key(ref_wav, "x", "zh", "v2")
    ref_cache.put_cached(cache_dir, key, {"x": 1}, {"v": np.zeros(1, np.float32)})
    meta_path = os.path.join(cache_dir, "prompt", key + ".json")
    meta = json.load(open(meta_path))
    meta["format"] = "gsovits-mlx/refcache-v1"
    json.dump(meta, open(meta_path, "w"))
    assert ref_cache.get_cached(cache_dir, key) is None


def test_clear(cache_dir, ref_wav):
    key = ref_cache.cache_key(ref_wav, "x", "zh", "v2")
    ref_cache.put_cached(cache_dir, key, {}, {"v": np.zeros(1, np.float32)})
    assert ref_cache.clear(cache_dir) == 1
    assert ref_cache.get_cached(cache_dir, key) is None

