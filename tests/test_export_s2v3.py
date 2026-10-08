"""Heavy (needs models_local + prep exp_dir): export round-trip gate.

Trains nothing; verifies the full export path: LoRA adapters (perturbed to
simulate training) merge into exactly the 88 DiT attention projections,
the exported safetensors is bit-identical to the base checkpoint everywhere
else, and pipeline.load_sovits_v3 reloads it.
"""

import os
import random

import numpy as np
import pytest

import mlx.core as mx
from mlx.utils import tree_map

from gsovits_mlx.io import load_mlx_safetensors
from gsovits_mlx.pipeline import load_sovits_v3
from gsovits_mlx.train.ckpt import save_s2v3_merged
from gsovits_mlx.train.lora import inject_lora, merged_training_weights

MODELS = pytest.importorskip("os").environ.get(
    "GSOVITS_MODELS_ROOT",
    "/Users/baicai1145/repos/gpt-sovits/GPT-SoVITS-mlx/models_local")


@pytest.mark.heavy
@pytest.mark.parametrize("version", ["v3", "v4", "v5dev", "v5turbo"])
def test_export_roundtrip(tmp_path, version):
    if not os.path.isdir(f"{MODELS}/{version}"):  # noqa: F821
        pytest.skip("converted weights unavailable")
    model, meta = load_sovits_v3(f"{MODELS}/{version}", version)
    model.update(tree_map(lambda v: v.astype(mx.float32)
                          if hasattr(v, "dtype") else v, model.parameters()))
    adapters = inject_lora(model.cfm.estimator, rank=32, seed=1234)
    for a in adapters:  # simulate trained adapters (nonzero B)
        a.lora_B = mx.random.normal(a.lora_B.shape, key=mx.random.key(5)) * 0.01
    out = str(tmp_path / version)
    merged = merged_training_weights(model.cfm.estimator, adapters)
    save_s2v3_merged(out, model, merged, dict(meta["model_hps"]), version)

    exp = load_mlx_safetensors(f"{out}/sovits.safetensors")
    base = load_mlx_safetensors(f"{MODELS}/{version}/sovits.safetensors")
    changed, drift, n = [], 0.0, 0
    for k in base:
        if k not in exp:
            continue
        n += 1
        af = np.array(base[k].astype(mx.float32))
        bf = np.array(exp[k].astype(mx.float32))
        delta = float(np.max(np.abs(af - bf))) if af.size else 0.0
        if delta > 5e-4:
            changed.append(k)
        else:
            drift = max(drift, delta)
    assert n == len(base)
    assert len(changed) == 22 * 4
    assert all("attn.to_" in k for k in changed)
    assert drift == 0.0
    # reload through the real inference loader
    m2, meta2 = load_sovits_v3(out, version)
    assert meta2["dit"]["depth"] == 22
