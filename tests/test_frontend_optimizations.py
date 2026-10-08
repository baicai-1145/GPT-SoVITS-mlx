"""CPU-only frontend query mapping and fused-normalization fallbacks."""
import mlx.core as mx
import numpy as np
import pytest

from gsovits_mlx.text.fused_ln import dense_residual_norm
from gsovits_mlx.text.g2pw_mlx import _forward, _NUM_POS
import gsovits_mlx.sovits.dit as dit


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    monkeypatch.setenv("GSOVITS_BERT_FUSED_LN", "1")
    monkeypatch.setenv("GSOVITS_DIT_FUSED_ADALN", "1")
    monkeypatch.setenv("GSOVITS_DIT_FAST_LN", "1")
    yield
    mx.set_default_device(previous)


def toy_classifier():
    rng = np.random.default_rng(27)
    shapes = {
        "bert.embeddings.word_embeddings.weight": (8, 12),
        "bert.embeddings.position_embeddings.weight": (5, 12),
        "bert.embeddings.token_type_embeddings.weight": (2, 12),
        "bert.embeddings.LayerNorm.weight": (12,),
        "bert.embeddings.LayerNorm.bias": (12,),
        "pos_classifier.weight": (_NUM_POS, 12),
        "pos_classifier.bias": (_NUM_POS,),
        "descriptor_bias.weight": (1, 4),
        "char_descriptor.weight": (3, 4),
        "second_order_descriptor.weight": (3 * _NUM_POS, 4),
        "classifier.weight": (4, 12),
        "classifier.bias": (4,),
    }
    return dict(n_layers=0, arrays={k: mx.array(rng.standard_normal(s).astype(np.float32))
                                    for k, s in shapes.items()})


def evaluate(w, ids, positions, chars, batches=None):
    return np.asarray(_forward(w, ids, mx.zeros_like(ids), mx.ones(ids.shape),
                               mx.ones((len(positions), 4)), mx.array(chars, mx.int32),
                               mx.array(positions, mx.int32),
                               mx.array(batches, mx.int32) if batches is not None else None))


def test_one_query_per_context_does_not_form_cartesian_product():
    w = toy_classifier()
    ids = mx.array([[1, 2, 3], [4, 5, 6]], mx.int32)
    got = evaluate(w, ids, [0, 2], [1, 2])
    separate = np.concatenate([evaluate(w, ids[b:b + 1], [pos], [char])
                               for b, pos, char in ((0, 0, 1), (1, 2, 2))])
    assert got.shape == (2, 4)
    np.testing.assert_array_equal(got, separate)


def test_multiple_queries_share_context_without_changing_order():
    w = toy_classifier()
    ids = mx.array([[1, 2, 3], [4, 5, 6]], mx.int32)
    got = evaluate(w, ids, [2, 0, 1], [1, 0, 2], [1, 0, 1])
    separate = np.concatenate([evaluate(w, ids[b:b + 1], [pos], [char])
                               for b, pos, char in ((1, 2, 1), (0, 0, 0), (1, 1, 2))])
    np.testing.assert_array_equal(got, separate)


@pytest.mark.parametrize("dtype", [mx.float16, mx.float32])
def test_dense_residual_fallback_preserves_left_associated_rounding(dtype):
    rng = np.random.default_rng(14)
    mm, residual = [mx.array(rng.standard_normal((1, 3, 768)).astype(np.float32), dtype)
                    for _ in range(2)]
    bias, weight, beta = [mx.array(rng.standard_normal(768).astype(np.float32), dtype)
                          for _ in range(3)]
    expected = mx.fast.layer_norm((mm + bias) + residual, weight, beta, 1e-12)
    got = dense_residual_norm(mm, bias, residual, weight, beta, 1e-12)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))


@pytest.mark.parametrize("dtype", [mx.float16, mx.float32])
def test_modulated_norm_cpu_fallback_never_launches_metal(monkeypatch, dtype):
    def forbidden(*args):
        raise AssertionError("Metal kernel launched on CPU")
    monkeypatch.setattr(dit, "fused_modulated_norm", forbidden)
    x = mx.array(np.random.default_rng(1).standard_normal((1, 3, 1024)).astype(np.float32))
    scale, shift = mx.full((1, 1024), 0.2), mx.full((1, 1024), -0.1)
    expected = (dit._ln_noaffine(x) * (1 + scale[:, None]) + shift[:, None]).astype(dtype)
    got = dit._modulated_norm(x, scale, shift, dtype)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))
