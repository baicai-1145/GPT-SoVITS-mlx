"""Memory-hygiene loader tests: lazy safetensors mapping + eval_tree/release.

gsovits_mlx.io used to materialize whole weight files into a dict before the
model loaders copied arrays onto modules (double residency). The loader now
returns a lazy mmap-backed Mapping; these tests pin its dict-compatible API
and the eval/release helpers. Runs on tiny in-memory tensors — no weights,
no GPU workload.
"""
from __future__ import annotations

import os
import tempfile

import mlx.core as mx
import numpy as np
import pytest

from gsovits_mlx.io import MappedSafetensors, eval_tree, load_mlx_safetensors, release


@pytest.fixture()
def st_path(tmp_path):
    path = str(tmp_path / "t.safetensors")
    mx.save_safetensors(path, {"a.weight": mx.array(np.random.rand(64, 32).astype(np.float32)),
                               "a.bias": mx.array(np.random.rand(64).astype(np.float16)),
                               "b.embed": mx.array(np.random.rand(10, 8).astype(np.float32))})
    return path


def test_lazy_mapping_protocol(st_path):
    m = load_mlx_safetensors(st_path)
    assert isinstance(m, MappedSafetensors)
    assert set(m.keys()) == {"a.weight", "a.bias", "b.embed"}
    assert len(m) == 3
    assert "a.weight" in m and "zzz" not in m
    assert sorted(iter(m)) == ["a.bias", "a.weight", "b.embed"]


def test_lazy_get_returns_array(st_path):
    m = load_mlx_safetensors(st_path)
    w = m.get("a.weight")
    assert isinstance(w, mx.array) and w.shape == (64, 32) and w.dtype == mx.float32
    assert np.allclose(np.array(w), mx.load(st_path)["a.weight"])
    assert m.get("zzz") is None
    assert m.get("zzz", 5) == 5
    # re-get yields an equal array (values are not cached)
    assert np.array_equal(np.array(w), np.array(m.get("a.weight")))


def test_eval_tree_realizes_nested_holders(st_path):
    class Holder:
        pass

    h = Holder()
    h.w = mx.array(np.zeros(4, np.float32)) + 2
    h.nested = {"x": [mx.array(np.ones(3)) * 3, None], "s": "str"}
    sub = Holder()
    sub.deep = (mx.array(np.zeros(2)) + 5,)
    h.sub = sub
    m = load_mlx_safetensors(st_path)
    eval_tree(h, m["b.embed"], {"k": [mx.array(np.ones(2)) * 9]})
    assert float(h.w[0]) == 2.0
    assert float(h.nested["x"][0][1]) == 3.0
    assert float(sub.deep[0][0]) == 5.0
    release(m, h)


def test_release_after_del_does_not_raise(st_path):
    m = load_mlx_safetensors(st_path)
    arr = mx.array(np.ones(8).astype(np.float32)) * 2
    eval_tree(arr)
    assert float(arr[3]) == 2.0
    del m, arr
    release()
