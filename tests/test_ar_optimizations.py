"""Small CPU contract tests; real AR parity and timing use GPU captures."""
import mlx.core as mx
import numpy as np
import pytest

from gsovits_mlx.gpt.t2s import T2SBlock, _sample
from gsovits_mlx.gpt.mixed_gemv import mixed_gemv, supports
from gsovits_mlx.sovits.models_v1v2 import Generator


@pytest.fixture(autouse=True)
def cpu_device(monkeypatch):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    monkeypatch.setenv("GSOVITS_AR_PROMOTE_CACHE", "1")
    monkeypatch.setenv("GSOVITS_AR_MIXED_GEMV", "1")
    yield
    mx.set_default_device(previous)


def make_block(dim=16):
    rng = np.random.default_rng(7)
    shapes = {"qkv_w": (3 * dim, dim), "qkv_b": (3 * dim,),
              "out_w": (dim, dim), "out_b": (dim,),
              "mlp1.w": (2 * dim, dim), "mlp1.b": (2 * dim,),
              "mlp2.w": (dim, 2 * dim), "mlp2.b": (dim,),
              "norm1.g": (dim,), "norm1.b": (dim,),
              "norm2.g": (dim,), "norm2.b": (dim,)}
    params = {"block.0." + k: mx.array((rng.standard_normal(s) * 0.1).astype(np.float16))
              for k, s in shapes.items()}
    return T2SBlock(2, dim).weights(params, 0), params


def test_promoted_projection_cache_and_reload():
    block, params = make_block()
    x = mx.array(np.random.default_rng(8).standard_normal((1, 3, 16)).astype(np.float32))
    want = x @ block.qkv_w.T + block.qkv_b
    got = block._linear(x, "qkv_w", "qkv_b")
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    cached = block._projection_cache["qkv_w", mx.float32]
    block._linear(x, "qkv_w", "qkv_b")
    assert block._projection_cache["qkv_w", mx.float32] is cached
    block.weights(params, 0)
    assert not block._projection_cache


def test_projection_epilogue_fallback():
    block, _ = make_block()
    x = mx.ones((1, 1, 16), mx.float32)
    h = mx.maximum(x @ block.linear1_w.T + block.linear1_b, 0)
    want = (h @ block.linear2_w.T + block.linear2_b) + x
    got = block._mlp_residual(x)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    half = x.astype(mx.float16)
    assert block._linear(half, "qkv_w", "qkv_b").dtype == mx.float16


@pytest.mark.parametrize("dtype", [mx.float16, mx.float32])
@pytest.mark.parametrize("top_p", [1.0, 0.8])
def test_sample_host_info_preserves_random_draw_and_raw_stop(dtype, top_p):
    logits = mx.array([[-3.0, 4.0, 0.5, 3.0]], dtype)
    previous = mx.array([[1, 1, 3]], mx.int32)
    np.random.seed(3)
    original = _sample(logits, previous, 3, top_p, 0.8, 2.0, None)
    np.random.seed(3)
    got, sample_id, raw_argmax = _sample(logits, previous, 3, top_p, 0.8, 2.0, None,
                                       _return_info=True)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(original))
    assert sample_id == int(np.asarray(original)[0, 0])
    assert raw_argmax == 1


def test_mixed_gemv_never_launches_on_cpu():
    x, w, b = mx.ones((1, 128)), mx.ones((16, 128), mx.float16), mx.zeros((16,))
    assert not supports(x, w, b)
    with pytest.raises(ValueError, match="supported GPU"):
        mixed_gemv(x, w, b)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("trim", [False, True])
def test_compiled_vocoder_contract(monkeypatch, enabled, trim):
    gen = Generator(2, "1", [3], [[1]], [2], 16, [4])
    gen.conv_pre.weight = gen.conv_pre.weight.astype(mx.float16)
    compiled, seen = [], []

    def compile_fn(model):
        compiled.append(model)
        def run(x, g):
            seen.append((x.dtype, g.dtype if g is not None else None))
            return x
        return run

    monkeypatch.setattr(mx, "compile", compile_fn)
    def eager(model, x, g):
        seen.append((x.dtype, g.dtype if g is not None else None))
        return x
    monkeypatch.setattr(Generator, "__call__", eager)
    monkeypatch.setenv("GSOVITS_VOCODER_FAST", str(int(enabled)))
    monkeypatch.setenv("GSOVITS_HIFIGAN_STAGE_TRIM", str(int(trim)))
    x = mx.array([[[0.123456]]], mx.float32)
    for _ in range(2):
        out = gen.infer(x, g=x)
        assert out.dtype == mx.float32
        np.testing.assert_array_equal(np.asarray(out), np.asarray(x).astype(np.float16).astype(np.float32))
    assert compiled == ([gen] if enabled and not trim else [])
    assert seen == [(mx.float16, mx.float16)] * 2
