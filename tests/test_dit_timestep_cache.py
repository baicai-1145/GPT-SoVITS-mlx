"""CPU toy-model contracts; production time-table parity is GPU-gated."""
import mlx.core as mx
from mlx.utils import tree_map
import numpy as np
import pytest

from gsovits_mlx.sovits.dit import DiT, load_dit_params


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    for name in ("GSOVITS_DIT_PREFOLD", "GSOVITS_DIT_FP16_BLOCKS", "GSOVITS_DIT_FAST_LN"):
        monkeypatch.setenv(name, "0")
    yield
    mx.set_default_device(previous)


def model_and_inputs():
    model = DiT(dim=16, depth=2, heads=2, dim_head=8, mel_dim=4, text_dim=4,
                ff_mult=2, use_step_embedding=False)
    rng = np.random.default_rng(17)
    model.update(tree_map(lambda a: mx.array((rng.standard_normal(a.shape) * 0.05).astype(np.float16)),
                          model.parameters()))
    x, cond, text = [mx.array(rng.standard_normal((1, 4, 7)).astype(np.float32)) for _ in range(3)]
    return model, (x, cond, mx.array([7], mx.int32), text)


def checkpoint(model):
    fields = [("dit.time_embed.0", model.time_embed, "time_mlp_0"),
              ("dit.time_embed.2", model.time_embed, "time_mlp_2"),
              ("dit.in.proj", model.input_embed, "proj"),
              ("dit.in.conv_pos_0", model.input_embed, "conv_pos_0"),
              ("dit.in.conv_pos_1", model.input_embed, "conv_pos_1"),
              ("dit.norm_out", model.norm_out, "linear"),
              ("dit.proj_out", model, "proj_out")]
    for i, block in enumerate(model.transformer_blocks):
        prefix = f"dit.blocks.{i}"
        fields.append((prefix + ".attn_norm", block.attn_norm, "linear"))
        for key, attr in (("to_q", "to_q"), ("to_k", "to_k"), ("to_v", "to_v"), ("to_out", "to_out_0")):
            fields.append((prefix + ".attn." + key, block.attn, attr))
        fields.extend([(prefix + ".ff.0", block.ff, "ff_0_0"), (prefix + ".ff.2", block.ff, "ff_2")])
    return {key + "." + suffix: getattr(obj, attr + "_" + suffix)
            for key, obj, attr in fields for suffix in ("w", "b")}


def test_timestep_replay_is_bit_identical_and_reused():
    model, (x, cond, lengths, text) = model_and_inputs()
    table = model.prepare_timestep_cache(4, 1, mx.float16, mx.float32)
    assert model.prepare_timestep_cache(4, 1, mx.float16, mx.float32) is table
    for i, entry in enumerate(table):
        time = mx.array([i / 4], mx.float16)
        ordinary = model(x, cond, lengths, time, None, text, infer=True)[0]
        cached = model(x, cond, lengths, time, None, text, infer=True, timestep_cache=entry)[0]
        np.testing.assert_array_equal(np.asarray(ordinary), np.asarray(cached))
    other = model.prepare_timestep_cache(4, 1, mx.float32, mx.float32)
    assert other is not table
    assert not np.array_equal(np.asarray(table[0]["time"]), np.asarray(table[1]["time"]))


def test_timestep_cache_rejects_step_embedding_and_wrong_stream():
    model, (x, cond, lengths, text) = model_and_inputs()
    bad = model.prepare_timestep_cache(4, 1, mx.float16, mx.float16)[0]
    with pytest.raises(ValueError, match="residual stream"):
        model(x, cond, lengths, mx.array([0.0]), None, text, infer=True, timestep_cache=bad)
    with pytest.raises(ValueError, match="no-step-embedding"):
        DiT(dim=16, depth=1).prepare_timestep_cache(4, 1, mx.float16, mx.float32)


def test_checkpoint_reload_invalidates_time_modulation():
    model, _ = model_and_inputs()
    table = model.prepare_timestep_cache(4, 1, mx.float16, mx.float32)
    old = np.asarray(table[1]["blocks"][0]).copy()
    params = checkpoint(model)
    params["dit.blocks.0.attn_norm.w"] = params["dit.blocks.0.attn_norm.w"] + mx.array(0.05, mx.float16)
    load_dit_params(model, params, depth=2, text_blocks=0, has_d_embed=False)
    assert not model._timestep_cache
    new = model.prepare_timestep_cache(4, 1, mx.float16, mx.float32)
    assert new is not table
    assert not np.array_equal(old, np.asarray(new[1]["blocks"][0]))
