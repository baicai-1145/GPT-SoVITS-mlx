"""Actual SynthesizerTrn decoder entry must keep trim barriers outside compile."""
from types import SimpleNamespace
import mlx.core as mx
import pytest

from gsovits_mlx.sovits.models_v1v2 import Generator, SynthesizerTrn


@pytest.mark.parametrize("existing_graph", [False, True])
def test_decode_stage_trim_bypasses_compiled_decoder(monkeypatch, existing_graph):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        decoder = Generator(2, "1", [3], [[1, 3, 5]], [2], 4, [4])
        def stale_graph(*args):
            raise AssertionError("trim mode invoked an existing compiled decoder")
        z = mx.ones((1, 2, 2))
        mask = mx.ones((1, 1, 2))
        owner = SimpleNamespace(
            get_ge=lambda *a: None, is_v2pro=False, semantic_frame_rate="50hz",
            quantizer=SimpleNamespace(decode=lambda codes: z),
            enc_p=lambda *a: (z, z, mx.zeros_like(z), mask),
            flow=lambda values, *a, **kw: values,
            dec=decoder, _dec_fast=stale_graph if existing_graph else None,
        )
        monkeypatch.setenv("GSOVITS_HIFIGAN_FAST", "1")
        monkeypatch.setenv("GSOVITS_HIFIGAN_STAGE_TRIM", "1")
        out, _ = SynthesizerTrn.decode(owner, mx.zeros((1, 1, 1), mx.int32),
                                      mx.zeros((1, 2), mx.int32), z, noise_scale=0,
                                      key=mx.random.key(0))
        mx.eval(out)
        assert out.dtype == mx.float32
    finally:
        mx.set_default_device(previous)
