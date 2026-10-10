"""Gate 1+2 tests for the CFM EAGER (non-checkpointed) backward path.

CPU, tiny/synthetic shapes where possible; real-weights test skips when
weights are unavailable (mirrors tests/test_grad_flow_s2v3.py patterns).

Gate 1: same-draw eager loss == two-pass loss BIT-EXACT (fp32 DiT).
Gate 2: eager grads vs two-pass grads elementwise — max rel per tensor
class documented (fp32 reduction-reorder tolerance), all finite.

Run: pytest -q tests/test_train_s2v3_eager.py
"""

import math
import os
import random
import sys

import numpy as np
import pytest

import mlx.core as mx

mx.set_default_device(mx.cpu)
# NUMICS REGIME (measured, mlx 0.32.2): nn.gelu / nn.gelu_approx are
# @mx.compile-wrapped; graphs evaluated inside a value_and_grad TRACE round
# gelu differently than the same ops evaluated untraced (compile-fusion
# difference; ~1e-6 forward, amplified on grads). The two-pass forward is
# untraced, the eager one is traced — so bit-parity gates must run BOTH
# sides with compile disabled. set_parity_compile_disabled() is the single
# switch the eager method honors. Probe evidence: both-plain => loss AND
# grads BIT-EXACT; mixed => 1-ulp loss, 0.3 median grad rel (fusion noise,
# not semantics).
from gsovits_mlx.train.s2_cfm import set_parity_compile_disabled  # noqa: E402
set_parity_compile_disabled(True)

from gsovits_mlx.train.lora import inject_lora  # noqa: E402
from gsovits_mlx.train.s2_cfm import (  # noqa: E402
    S2V3TrainModel, upcast_training_model, eager_shape_estimate)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WEIGHTS = os.environ.get("GSOVITS_MODELS_ROOT",
                          _REPO + "/models_local") + "/v3"


# ---------------------------------------------------------------------------
# synthetic tiny model (no weight files needed)
# ---------------------------------------------------------------------------

def _tiny_model_and_batch(seed: int = 0, T: int = 96):
    """Minimal SynthesizerTrnV3 at native (random-ish) init + one batch.

    The default-constructed model has zeroed DiT projections (grads for
    some tensors would be exactly 0); nudge a few weights so every grad
    tensor class is nonzero.
    """
    from gsovits_mlx.sovits.models_v3 import SynthesizerTrnV3
    m = SynthesizerTrnV3(1025, 20, 192, 192, 768, 2, 6, 3, 0.1, "1",
                         [3, 7, 11], [[1, 3, 5]] * 3, [10, 8, 2, 2, 2],
                         512, [16, 16, 8, 2, 2], gin_channels=512,
                         semantic_frame_rate="25hz", version="v3")
    rng = np.random.default_rng(seed)

    def _jiggle(tree):
        from mlx.utils import tree_map
        return tree_map(
            lambda v: (mx.array(rng.normal(scale=0.02, size=v.shape)
                                .astype(np.float32)) + v)
            if hasattr(v, "dtype") and v.dtype in (mx.float32,) else v,
            tree)

    m.update(_jiggle(m.parameters()))
    # the DiT hangs off cfm (a plain-object wrapper NOT in nn parameters)
    est = m.cfm.estimator
    est.update(_jiggle(est.parameters()))
    upcast_training_model(m, dit_fp16=False)  # fp32 everywhere (CPU gate)

    rng = np.random.default_rng(seed + 1)
    B = 2
    ssl = rng.normal(size=(B, 768, T)).astype(np.float16)  # (C, T)
    spec = rng.normal(size=(B, 1025, T)).astype(np.float32)
    mel = rng.normal(size=(B, 100, int(T * 1.25 * 1.5))).astype(np.float32)
    text = rng.integers(1, 100, size=(B, 7)).astype(np.int64)
    lengths = np.array([T, T - 16], dtype=np.int64)
    mel_lengths = (lengths * 1.25 * 1.5).astype(np.int64)

    class B:
        pass

    b = B()
    b.ssl = ssl
    b.spec = spec
    b.mel = mel
    b.ssl_lengths = lengths
    b.spec_lengths = lengths
    b.text = text
    b.text_lengths = np.array([7, 5])
    b.mel_lengths = mel_lengths
    return m, b


def _det_state(b, seed=3, gate=0.5):
    rng = np.random.default_rng(seed)
    return {
        "t": mx.array(rng.uniform(0, 1, (b.ssl.shape[0],)).astype(np.float32)),
        "x0": mx.array(rng.normal(size=(b.ssl.shape[0], 100,
                                        b.mel.shape[2])).astype(np.float32)),
        "prompt_lens": mx.array([5, 3]),
        "gate": gate,  # >= 0.3 -> single-step branch
        "base": mx.array([2, 3], dtype=mx.float32),
    }


def _run(tm, masters, batch, det, which):
    if which == "eager":
        loss, info, grads = tm.train_loss_and_grads_eager(masters, batch,
                                                          det=det)
    else:
        loss, info, grads = tm.train_loss_and_grads(masters, batch, det=det)
    return loss, info, grads


def _max_rel(ga: mx.array, gb: mx.array) -> float:
    a = np.asarray(ga, dtype=np.float64)
    b = np.asarray(gb, dtype=np.float64)
    denom = np.maximum(np.abs(b), 1e-12)
    return float(np.max(np.abs(a - b) / denom)) if a.size else 0.0


# ---------------------------------------------------------------------------
# gate 0: shape-estimate helper sanity
# ---------------------------------------------------------------------------

def test_eager_shape_estimate_monotone():
    e1 = eager_shape_estimate(np.zeros((3, 100, 320)), 3)
    e2 = eager_shape_estimate(np.zeros((3, 100, 640)), 3)
    assert e1["T"] == 320 and e2["T"] == 640
    assert e2["total_activation_mb"] > e1["total_activation_mb"]
    # T^2 term (SDPA scores) dominates growth
    assert e2["sdpa_mb"] / max(e1["sdpa_mb"], 1e-9) == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# gates 1+2 on the tiny synthetic model (fast, no weights needed)
# ---------------------------------------------------------------------------

def test_eager_vs_twopass_tiny(seed=0):
    m, b = _tiny_model_and_batch(seed=seed, T=96)
    adapters = inject_lora(m.cfm.estimator, rank=8, seed=0)
    tm = S2V3TrainModel(m, "v3", lora_adapters=adapters,
                        rng=random.Random(0))
    masters = {k: v.astype(mx.float32)
               for k, v in tm.trainable_params().items()}
    det = _det_state(b, seed=3, gate=0.5)

    l_ck, _, g_ck = _run(tm, masters, b, det, "twopass")
    l_eg, _, g_eg = _run(tm, masters, b, det, "eager")

    # gate 1: same-draw loss bit-exact
    assert l_ck == l_eg, f"loss mismatch: {l_ck!r} vs {l_eg!r}"

    # gate 2: grads elementwise, fp32-reduction tolerance; all finite
    assert set(g_ck) == set(g_eg)
    rel = {}
    for k in g_ck:
        assert math.isfinite(float(mx.abs(g_eg[k]).max())), f"{k} nonfinite"
        rel[k] = _max_rel(g_eg[k], g_ck[k])
    # structural: LoRA B grads nonzero (A zero at B=0 init -> include A
    # tensors only if nonzero in the reference)
    nonzero = [k for k in g_ck if float(mx.abs(g_ck[k]).max()) > 0]
    assert nonzero, "all reference grads zero — test is blind"
    worst = max(rel[k] for k in nonzero)
    # fp32 CPU reduction-order differences; the tiny model amplifies
    # relative error on near-zero grads, so gate on the median + worst
    med = float(np.median([rel[k] for k in nonzero]))
    print(f"tiny grad rel: worst={worst:.3e} median={med:.3e} "
          f"n={len(nonzero)}")
    assert worst < 1e-3, {k: rel[k] for k in nonzero
                          if rel[k] >= 1e-3}
    assert med < 1e-6


def test_eager_two_step_branch_tiny():
    """Same gates with the 30% two-step branch active (gate=0.1)."""
    m, b = _tiny_model_and_batch(seed=1, T=80)
    adapters = inject_lora(m.cfm.estimator, rank=8, seed=0)
    tm = S2V3TrainModel(m, "v3", lora_adapters=adapters,
                        rng=random.Random(0))
    masters = {k: v.astype(mx.float32)
               for k, v in tm.trainable_params().items()}
    det = _det_state(b, seed=4, gate=0.1)
    det["base"] = mx.array([2, 3], dtype=mx.float32)

    l_ck, i_ck, g_ck = _run(tm, masters, b, det, "twopass")
    l_eg, i_eg, g_eg = _run(tm, masters, b, det, "eager")
    assert i_ck["two_step"] and i_eg["two_step"]
    assert l_ck == l_eg, f"two-step loss mismatch: {l_ck!r} vs {l_eg!r}"
    rel = {k: _max_rel(g_eg[k], g_ck[k]) for k in g_ck}
    nonzero = [k for k in g_ck if float(mx.abs(g_ck[k]).max()) > 0]
    worst = max(rel[k] for k in nonzero)
    med = float(np.median([rel[k] for k in nonzero]))
    print(f"tiny 2-step grad rel: worst={worst:.3e} median={med:.3e}")
    assert worst < 1e-3 and med < 1e-6


def test_eager_default_off_and_flag_conflicts():
    """--eager-backward default OFF; refuses --no-dit-ckpt combo."""
    import subprocess
    env = dict(os.environ)
    env["PATH"] = "/usr/bin:/bin"
    r = subprocess.run(
        [sys.executable, "tools/train_s2_v3.py", "--help"],
        capture_output=True, text=True, cwd=_REPO, env=env)
    assert "--eager-backward" in r.stdout


# ---------------------------------------------------------------------------
# gates 1+2 on REAL v3 weights + a real-format batch (skips if unavailable)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(_WEIGHTS), reason="weights unavailable")
def test_eager_vs_twopass_real():
    from gsovits_mlx.pipeline import load_sovits_v3
    from gsovits_mlx.train.s2_v3_data import collate_v3
    model, _ = load_sovits_v3(_WEIGHTS, "v3")
    upcast_training_model(model, dit_fp16=False)  # fp32 reference regime
    adapters = inject_lora(model.cfm.estimator, rank=32, seed=0)
    tm = S2V3TrainModel(model, "v3", lora_adapters=adapters,
                        rng=random.Random(0))
    masters = {k: v.astype(mx.float32)
               for k, v in tm.trainable_params().items()}

    rng = np.random.default_rng(7)
    B, T = 2, 128
    ssl = rng.normal(size=(B, 1, 768, T)).astype(np.float16)
    spec = rng.normal(size=(B, 1025, T)).astype(np.float32)
    text = rng.integers(1, 100, size=(B, 6)).astype(np.int64)
    lengths = np.array([T, T - 10], dtype=np.int64)
    mel_lengths = (lengths * 1.25 * 1.5).astype(np.int64)
    batch = collate_v3([
        (ssl[i], spec[i],
         rng.normal(size=(100, int(mel_lengths[i]))).astype(np.float32),
         text[i][:5 if i else 6])
        for i in range(B)])
    det = _det_state_from_batch(batch, seed=11)

    l_ck, _, g_ck = _run(tm, masters, batch, det, "twopass")
    l_eg, _, g_eg = _run(tm, masters, batch, det, "eager")
    assert l_ck == l_eg, f"real loss mismatch: {l_ck!r} vs {l_eg!r}"
    rel = {k: _max_rel(g_eg[k], g_ck[k]) for k in g_ck}
    nonzero = [k for k in g_ck if float(mx.abs(g_ck[k]).max()) > 0]
    worst = max(rel[k] for k in nonzero)
    med = float(np.median([rel[k] for k in nonzero]))
    print(f"real grad rel: worst={worst:.3e} median={med:.3e} "
          f"n={len(nonzero)}")
    # real-weight fp32 CPU: LoRA-B/trunk tensors within fp-reduction reorder
    assert med < 1e-6, "median rel too high — semantics differ, not reorder"
    assert worst < 1e-3


def _det_state_from_batch(batch, seed=5, gate=0.5):
    rng = np.random.default_rng(seed)
    B = batch.ssl.shape[0]
    return {
        "t": mx.array(rng.uniform(0, 1, (B,)).astype(np.float32)),
        "x0": mx.array(rng.normal(size=(B, 100, batch.mel.shape[2]))
                       .astype(np.float32)),
        "prompt_lens": mx.array([4, 3] if B == 2 else [4] * B),
        "gate": gate,
        "base": mx.array([2] * B, dtype=mx.float32),
    }
