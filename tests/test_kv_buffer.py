"""KV state contracts; CPU tests do not establish inference audio parity."""
import mlx.core as mx
import numpy as np
import pytest

from gsovits_mlx.gpt.kv_cache import KVBuffer


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


@pytest.mark.parametrize("dtype", [mx.float16, mx.float32])
def test_buffer_matches_concat_across_capacity_growth(dtype):
    prefix = mx.arange(6, dtype=mx.float32).reshape(1, 3, 2).astype(dtype)
    state = KVBuffer(prefix, minimum_capacity=4)
    expected = prefix
    mx.eval(state.buffer)
    for step in range(12):
        row = mx.full((1, 1, 2), step + 0.5, dtype=dtype)
        state.append(row)
        expected = mx.concatenate((expected, row), axis=1)
        mx.eval(state.buffer, expected)
        assert state.used == expected.shape[1]
        assert state.buffer.shape[1] >= state.used
        assert np.array_equal(np.asarray(state.active()), np.asarray(expected))
    assert state.buffer.shape[1] == 16


def test_buffer_preserves_concat_dtype_promotion():
    prefix = mx.array([[[1.25, 2.5]]], dtype=mx.float16)
    row = mx.array([[[1.0001, 2.0001]]], dtype=mx.float32)
    state = KVBuffer(prefix, minimum_capacity=4)
    state.append(row)
    expected = mx.concatenate((prefix, row), axis=1)
    mx.eval(state.buffer, expected)
    assert state.buffer.dtype == expected.dtype == mx.float32
    assert np.array_equal(np.asarray(state.active()), np.asarray(expected))


@pytest.mark.parametrize("buffered_key", [False, True])
def test_attention_rejects_asymmetric_cache_types(buffered_key):
    from gsovits_mlx.gpt.t2s import T2SBlock
    block = T2SBlock(num_heads=2, hidden_dim=4)
    values = mx.ones((1, 1, 4))
    marker = KVBuffer(values)
    key, value = (marker, values) if buffered_key else (values, marker)
    with pytest.raises(AssertionError, match="K/V cache types must match"):
        block._attn(values, key, value, None)

