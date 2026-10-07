"""AdaLN prefold parity through the production DiT and AdaLN call paths."""

from types import SimpleNamespace

import mlx.core as mx
from mlx.utils import tree_map
import pytest

from gsovits_mlx.sovits.dit import AdaLayerNormZero, DiT, DiTBlock


@pytest.fixture(autouse=True)
def cpu_device(monkeypatch):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    monkeypatch.setenv("GSOVITS_DIT_PREFOLD", "0")
    monkeypatch.setenv("GSOVITS_DIT_FP16_BLOCKS", "0")
    yield
    mx.set_default_device(previous)


class ModulationProbe:
    """Observe modulation without duplicating the reference activation."""

    def __init__(self, dim, dtype, seed):
        self.attn = SimpleNamespace(to_q_w=mx.zeros((dim, dim), dtype=mx.float16))
        self.attn_norm = AdaLayerNormZero(dim)
        self.attn_norm.linear_w = (
            mx.random.normal((6 * dim, dim), key=mx.random.key(seed)) * 0.2
        ).astype(dtype)
        self.attn_norm.linear_b = (
            mx.random.normal((6 * dim,), key=mx.random.key(seed + 1)) * 0.1
        ).astype(dtype)
        self.gate = None

    def __call__(self, x, t, mask, rope, precomputed_mods=None):
        if precomputed_mods is None:
            _, self.gate, _, _, _ = self.attn_norm(x, t)
        else:
            dim = x.shape[-1]
            self.gate = precomputed_mods[:, 2 * dim:3 * dim]
        return x


@pytest.mark.parametrize("weight_dtype", [mx.float32, mx.float16],
                         ids=["fp32_weights", "fp16_weights"])
@pytest.mark.parametrize("fp16_stream", [False, True],
                         ids=["fp32_stream", "fp16_stream"])
def test_prefold_matches_production_adaln(monkeypatch, weight_dtype, fp16_stream):
    dim, batch, length = 16, 3, 9
    model = DiT(dim=dim, depth=2, heads=2, dim_head=8, mel_dim=4,
                text_dim=4, conv_layers=0, use_step_embedding=False)
    probes = [ModulationProbe(dim, weight_dtype, seed) for seed in (7, 19)]
    model.transformer_blocks = probes
    embeddings = mx.stack([
        mx.linspace(-2 + offset, 2 + offset, dim)
        for offset in (-0.5, 0.0, 0.5)
    ])
    model.time_embed = lambda time: embeddings
    monkeypatch.setenv("GSOVITS_DIT_FP16_BLOCKS", "1" if fp16_stream else "0")
    x = mx.zeros((batch, 4, length))
    lengths = mx.full((batch,), length, dtype=mx.int32)
    time = mx.full((batch,), 0.25)

    model(x, x, lengths, time, None, x, infer=True)
    reference = [probe.gate for probe in probes]
    mx.eval(*reference)
    monkeypatch.setenv("GSOVITS_DIT_PREFOLD", "1")
    model(x, x, lengths, time, None, x, infer=True)
    actual = [probe.gate for probe in probes]
    mx.eval(*actual)

    stream_dtype = mx.float16 if fp16_stream else mx.float32
    tolerance = 2e-3 if weight_dtype == mx.float16 or fp16_stream else 1e-6
    for folded, naive in zip(actual, reference):
        assert folded.dtype == naive.dtype == stream_dtype
        if weight_dtype == mx.float16:
            # AdaLN rounds the projection and bias in fp16 before the stream cast.
            round_trip = folded.astype(mx.float16).astype(stream_dtype)
            assert bool(mx.array_equal(folded, round_trip)), "lost fp16 modulation rounding"
        delta = float(mx.max(mx.abs(folded - naive)).item())
        assert delta <= tolerance, f"prefold modulation maxdiff={delta}"


class RecordingBlock(DiTBlock):
    def __call__(self, *args, **kwargs):
        self.observed = super().__call__(*args, **kwargs)
        return self.observed


@pytest.mark.parametrize("weight_dtype,fp16_stream", [
    (mx.float32, False), (mx.float16, False), (mx.float16, True),
], ids=["fp32", "mixed_precision", "fp16_stream"])
def test_prefold_real_blocks_match_reference(monkeypatch, weight_dtype, fp16_stream):
    dim, batch, length = 16, 3, 9
    model = DiT(dim=dim, depth=2, heads=2, dim_head=8, mel_dim=4,
                text_dim=4, conv_layers=0, use_step_embedding=False)
    blocks = [RecordingBlock(dim, heads=2, dim_head=8) for _ in range(2)]
    seed = 31

    def initialize(tensor):
        nonlocal seed
        seed += 1
        return (mx.random.normal(tensor.shape, key=mx.random.key(seed))
                * 0.05).astype(weight_dtype)

    for block in blocks:
        block.update(tree_map(initialize, block.parameters()))
    model.transformer_blocks = blocks
    embeddings = mx.stack([
        mx.linspace(-2 + offset, 2 + offset, dim)
        for offset in (-0.5, 0.0, 0.5)
    ])
    model.time_embed = lambda time: embeddings
    monkeypatch.setenv("GSOVITS_DIT_FP16_BLOCKS", "1" if fp16_stream else "0")
    x = mx.zeros((batch, 4, length))
    lengths = mx.full((batch,), length, dtype=mx.int32)
    time = mx.full((batch,), 0.25)

    model(x, x, lengths, time, None, x, infer=True)
    reference = [block.observed for block in blocks]
    mx.eval(*reference)
    monkeypatch.setenv("GSOVITS_DIT_PREFOLD", "1")
    model(x, x, lengths, time, None, x, infer=True)
    actual = [block.observed for block in blocks]
    mx.eval(*actual)

    stream_dtype = weight_dtype if fp16_stream else mx.float32
    tolerance = 2e-3 if weight_dtype == mx.float16 else 1e-6
    for folded, naive in zip(actual, reference):
        assert folded.dtype == naive.dtype == stream_dtype
        assert bool(mx.all(mx.isfinite(folded)))
        delta = float(mx.max(mx.abs(folded - naive)).item())
        assert delta <= tolerance, f"prefold block output maxdiff={delta}"
