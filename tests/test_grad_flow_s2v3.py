"""Grad-flow boundary tests for S2V3TrainModel (CPU, tiny shapes).

Official LoRA trainer semantics (s2_train_v3_lora.py + set_no_grad):
  TRAINABLE: ref_enc.*, bridge.*, wns1.*, linear_mel.*, DiT LoRA A/B
  FROZEN:    ssl_proj.*, quantizer.*, enc_p.*, DiT base weights
"""

import os
import random
import sys

import numpy as np
import pytest

import mlx.core as mx

mx.set_default_device(mx.cpu)

from gsovits_mlx.train.lora import inject_lora
from gsovits_mlx.train.s2_cfm import S2V3TrainModel, upcast_training_model  # noqa: E402


_WEIGHTS = os.environ.get(
    "GSOVITS_MODELS_ROOT",
    "/Users/baicai1145/repos/gpt-sovits/GPT-SoVITS-mlx/models_local") + "/v3"


def _real_batch():
    """A real deterministic batch (dumped by .tmp/dump_ref_batches.py)."""
    import glob
    files = sorted(glob.glob(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ".tmp/ref_batches/batch_000.npz")))
    if not files:
        pytest.skip("no deterministic batch dump available")
    b = np.load(files[0])
    return dict(
        ssl=mx.array(b["ssl"]), spec=mx.array(b["spec"]),
        mel=mx.array(b["mel"]), ssl_lengths=mx.array(b["ssl_lengths"]),
        spec_lengths=mx.array(b["spec_lengths"]),
        text=mx.array(b["text"].astype(np.int32)),
        text_lengths=mx.array(b["text_lengths"]),
        mel_lengths=mx.array(b["mel_lengths"].astype(np.float32)),
    )


def _load_tiny_model():
    """Synthesize a minimal SynthesizerTrnV3 without touching weights dirs."""
    from gsovits_mlx.sovits.models_v3 import SynthesizerTrnV3
    m = SynthesizerTrnV3(version="v3")
    # randomize trainable-side params a bit so grads are nonzero-able
    m.update(mx.compile(lambda m: m))
    return m


@pytest.mark.skipif(not os.path.exists(_WEIGHTS), reason="weights unavailable")
def test_lora_grad_flow_boundaries():
    from gsovits_mlx.pipeline import load_sovits_v3
    model, _ = load_sovits_v3(_WEIGHTS, "v3")
    upcast_training_model(model, dit_fp16=True)
    adapters = inject_lora(model.cfm.estimator, rank=8, seed=0)
    tm = S2V3TrainModel(model, "v3", lora_adapters=adapters,
                        rng=random.Random(0))
    masters = {k: v.astype(mx.float32) for k, v in tm.trainable_params().items()}

    b = _real_batch()

    def loss_of(p):
        tm.apply_params(p)
        loss, _ = tm.forward(b["ssl"], b["spec"], b["mel"], b["ssl_lengths"],
                             b["spec_lengths"], b["text"], b["text_lengths"],
                             b["mel_lengths"])
        return loss

    _, grads = mx.value_and_grad(loss_of)(masters)
    assert grads, "no grads at all"
    has = {}
    for k, g in grads.items():
        has[k] = float(mx.abs(g).max())
    # LoRA adapters receive grads (88 projections × A,B = 176 tensors)
    lora_keys = [k for k in grads if ".lora_" in k]
    assert len(lora_keys) == 176, len(lora_keys)
    assert all(has[k] > 0 for k in lora_keys), \
        {k: has[k] for k in lora_keys[:4]}
    # trainable trunk receives grads (ref_enc / bridge / wns1 / linear_mel)
    for prefix in ("ref_enc.", "bridge_0.", "wns1.", "linear_mel."):
        ks = [k for k in grads if k.startswith(prefix)]
        assert ks, f"{prefix} missing from grads"
        assert any(has[k] > 0 for k in ks), \
            f"{prefix} all-zero grads — stop_gradient cut too wide?"



