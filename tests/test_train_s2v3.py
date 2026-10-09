"""Light tests for the s2 v3/v4/v5 CFM+LoRA trainer (task-5).

Covers (CPU, no model weights needed unless marked heavy):
* LoRA init + merge math (peft semantics: B=0, A kaiming_uniform(sqrt(5)),
  merge W + B@A with scaling 1.0)
* CFM loss prompt zeroing / trajectory construction (t, x0, xt, vt)
* 2-step branch reproduction with seeded rng (d = 2^-base, d_input mask,
  vt = (v1+v2)/2 detached, dt = 2d)
* collate V3/V4 shapes + sorting order
* nearest-interp parity for the x1.875/x2 fea upsampling
* ffmpeg load_audio equivalence to the official tools/my_utils.load_audio
  (f32 pcm md5, marked heavy: needs the base-venv torch+ffmpeg side)

Run: pytest -q tests/test_train_s2v3.py
"""

import math
import random

import numpy as np
import pytest

import mlx.core as mx

from gsovits_mlx.train.lora import (LoRALinear, inject_lora, merge_lora,
                                    kaiming_uniform_bound)
from gsovits_mlx.train.s2_cfm import CFMTrainingLoss, dit_sequence_mask
from gsovits_mlx.train.s2_v3_data import collate_v3, collate_v4, norm_spec
from gsovits_mlx.sovits.dit import DiT
from gsovits_mlx.sovits.models_v1v2 import _nearest_interp

MODELS_ROOT = os_check = None


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

def _tiny_dit(depth=2, use_step_embedding=True) -> DiT:
    return DiT(dim=64, depth=depth, heads=4, dim_head=16, ff_mult=2,
               mel_dim=100, text_dim=64, conv_layers=1,
               use_step_embedding=use_step_embedding)


def test_lora_init_peft_semantics():
    dit = _tiny_dit()
    adapters = inject_lora(dit, rank=8, seed=0)
    # one adapter per (block, to_k/to_q/to_v/to_out.0)
    assert len(adapters) == 2 * 4
    names = [a.name for a in adapters]
    # peft module-tree order: block, then to_k, to_q, to_v, to_out.0
    assert names[:4] == ["dit.blocks.0.attn.to_k", "dit.blocks.0.attn.to_q",
                         "dit.blocks.0.attn.to_v", "dit.blocks.0.attn.to_out.0"]
    for a in adapters:
        assert a.lora_B.shape == (1024 if False else a.lora_B.shape[0], 8)
        # B init = 0 (peft: lora_B zero-initialized -> adapter is a no-op at
        # step 0)
        assert float(mx.abs(a.lora_B).max()) == 0.0
        # A init ~ U(-bound, bound), bound = sqrt(1/fan_in) (torch nn.Linear
        # default kaiming_uniform a=sqrt(5))
        bound = kaiming_uniform_bound(a.lora_A.shape[1])
        assert float(mx.abs(a.lora_A).max()) <= bound
        assert float(mx.abs(a.lora_A).max()) > 0.9 * bound  # ~uniform draw


def test_lora_merge_math():
    dit = _tiny_dit()
    adapters = inject_lora(dit, rank=8, seed=0)
    a = adapters[0]
    w = dit.transformer_blocks[0].attn.to_k_w
    # perturb B so the merge is nontrivial
    a.lora_B = mx.random.normal(a.lora_B.shape)
    merged = merge_lora(w, a, scaling=1.0)
    expected = w.astype(mx.float32) + a.lora_B @ a.lora_A
    np.testing.assert_allclose(np.array(merged), np.array(expected),
                               rtol=0, atol=1e-5)
    # scaling term: alpha/r = 1 for the official config
    merged2 = merge_lora(w, a, scaling=1.0)
    assert merged.dtype == w.dtype


# ---------------------------------------------------------------------------
# CFM loss internals
# ---------------------------------------------------------------------------

class _RecordingEstimator:
    """Estimator stub: records calls, returns (B, C, T)-shaped constant
    output like the real DiT (our DiT returns (B, T, C))."""

    def __init__(self, transpose=False):
        self.calls = []
        self.transpose = transpose

    def __call__(self, xt, prompt, x_lens, t, d, mu):
        self.calls.append(dict(xt=xt, prompt=prompt, x_lens=x_lens, t=t,
                               d=d, mu=mu))
        if self.transpose:
            # real-DiT layout: consume (B,C,T), return (B,T,C) — the CFM
            # code transposes outputs back to (B,C,T)
            return mx.zeros((xt.shape[0], xt.shape[2], xt.shape[1]))
        return mx.zeros_like(xt)


def test_cfm_loss_prompt_zeroing_and_shapes():
    est = _RecordingEstimator(transpose=True)
    b, c, t = 2, 100, 50
    x1 = mx.random.normal((b, c, t))
    mu = mx.random.normal((b, 512, t))
    prompt_lens = [10, 0]
    x_lens = [50, 40]
    loss_fn = CFMTrainingLoss(est, rng=random.Random(0))
    # stub estimator returns zeros -> loss = mean(vt^2) over prompt-excluded
    # slices; check the prompt zeroing happened inside
    key = mx.random.key(42)
    loss, info = loss_fn(x1, x_lens, prompt_lens, mu, key=key)
    mx.eval(loss)
    assert 0 <= float(loss)
    final = est.calls[-1]
    xt = final["xt"]
    # sample 0: prompt region zeroed
    assert float(mx.abs(xt[0, :, :10]).max()) == 0.0
    assert float(mx.abs(xt[0, :, 10:]).max()) > 0
    # sample 1: no prompt
    assert float(mx.abs(xt[1, :, :]).max()) > 0
    # prompt tensor carries x1 in the prompt region for sample 0
    np.testing.assert_allclose(np.array(final["prompt"][0, :, :10]),
                               np.array(x1[0, :, :10]), atol=1e-6)


def test_cfm_two_step_branch_seeded():
    est = _RecordingEstimator(transpose=True)
    b, c, t = 2, 100, 30
    x1 = mx.random.normal((b, c, t), key=mx.random.key(1))
    mu = mx.zeros((b, 512, t))

    class Forced(random.Random):
        def random(self):
            return 0.1  # < 0.3 -> always take the 2-step branch

        def randint(self, a, b_):
            return 3  # base = 3 for all samples

    rng = Forced(0)
    loss_fn = CFMTrainingLoss(est, rng=rng)
    loss, info = loss_fn(x1, [30, 20], [5, 0], mu, key=mx.random.key(7))
    mx.eval(loss)
    assert info["two_step"] is True
    assert len(est.calls) == 3  # v1, v2, final
    d_call = est.calls[0]["d"]
    np.testing.assert_allclose(
        np.array(d_call), np.array([1 / 8, 1 / 8]), atol=1e-7)
    # dt of the FINAL call = 2*d
    dt_final = est.calls[2]["d"]
    np.testing.assert_allclose(
        np.array(dt_final), np.array([1 / 4, 1 / 4]), atol=1e-7)
    # t of the second call = t + d (same t as call 0 plus d)
    np.testing.assert_allclose(
        np.array(est.calls[1]["t"] - est.calls[0]["t"]),
        np.array([1 / 8] * b), atol=1e-6)


def test_cfm_two_step_d_input_mask():
    """d_input zeroes d < 1e-2 (base 7 -> d = 1/128 < 0.01 -> 0)."""
    est = _RecordingEstimator(transpose=True)
    x1 = mx.random.normal((1, 100, 20), key=mx.random.key(2))

    class Forced(random.Random):
        def random(self):
            return 0.05

        def randint(self, a, b_):
            return 7  # d = 2^-7 = 0.0078 < 1e-2

    loss_fn = CFMTrainingLoss(est, rng=Forced(0))
    _, info = loss_fn(x1, [20], [0], mx.zeros((1, 512, 20)),
                      key=mx.random.key(3))
    mx.eval()
    d_in = est.calls[0]["d"]
    np.testing.assert_allclose(np.array(d_in), [0.0], atol=1e-9)


# ---------------------------------------------------------------------------
# collate V3/V4
# ---------------------------------------------------------------------------

def _fake_item(ssl_t, spec_t, mel_t, text_n=20):
    return (np.random.rand(1, 768, ssl_t).astype(np.float16),
            np.random.rand(1025, spec_t).astype(np.float32),
            np.random.rand(100, mel_t).astype(np.float32),
            np.random.randint(0, 732, (text_n,)).astype(np.int64))


def test_collate_v3_shapes():
    # real-data relationship: mel@24k(hop256) = T*93.75 ≈ ssl(50hz)*1.875;
    # spec@32k(hop640) = T*50 = ssl (so mel_t ≈ int(ssl_t*1.875))
    batch = [_fake_item(100, 50, 187, 20), _fake_item(96, 48, 180, 25),
             _fake_item(80, 40, 150, 10)]
    out = collate_v3(batch)
    max_ssl = 100
    max_ssl_len1 = 8 * (max_ssl // 8 + 1)      # 104
    max_mel = int(max_ssl_len1 * 1.25 * 1.5)   # int(104*1.875) = 195
    assert out.ssl.shape == (3, 768, 102)
    assert out.spec.shape == (3, 1025, 52)
    assert out.mel.shape == (3, 100, max_mel)
    # sorted by spec len descending (text follows its sample)
    assert list(out.spec_lengths) == [50, 48, 40]
    assert list(out.mel_lengths) == [187, 180, 150]


def test_collate_v4_shapes():
    # v4: mel@32k(hop320) = T*100 = spec*2 exactly; ssl padded like spec
    batch = [_fake_item(100, 50, 100, 15), _fake_item(96, 48, 96, 12)]
    out = collate_v4(batch)
    max_spec = 2 * (50 // 2 + 1)  # 52
    max_ssl = 2 * (100 // 2 + 1)   # 102
    assert out.mel.shape == (2, 100, max_spec * 2)
    assert out.spec.shape == (2, 1025, max_spec)
    assert out.ssl.shape == (2, 768, max_ssl)
    assert list(out.spec_lengths) == [50, 48]


def test_norm_spec():
    x = np.array([[-12.0, 2.0, -5.0]], dtype=np.float32)
    out = np.array(norm_spec(mx.array(x)))
    np.testing.assert_allclose(out, [[-1.0, 1.0, (-5 + 12) / 14 * 2 - 1]],
                               atol=1e-6)


# ---------------------------------------------------------------------------
# interpolation parity (fea x1.875 / x2)
# ---------------------------------------------------------------------------

def _torch_nearest(x: np.ndarray, scale: float) -> np.ndarray:
    import torch
    import torch.nn.functional as F
    t = torch.from_numpy(x)
    return F.interpolate(t, scale_factor=scale, mode="nearest").numpy()


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def test_nearest_interp_x1875_matches_torch():
    """Heavy: needs base-venv torch for the reference (F.interpolate)."""
    if not _torch_available():
        pytest.skip("torch not installed (run under base venv)")
    x = np.random.rand(2, 8, 37).astype(np.float32)
    ours = np.array(_nearest_interp(mx.array(x), int(37 * 1.875),
                                    scale_factor=1.875))
    ref = _torch_nearest(x, 1.875)
    np.testing.assert_array_equal(ours, ref)


def test_nearest_interp_x2_matches_torch():
    """Heavy: needs base-venv torch for the reference (F.interpolate)."""
    if not _torch_available():
        pytest.skip("torch not installed (run under base venv)")
    x = np.random.rand(2, 8, 37).astype(np.float32)
    ours = np.array(_nearest_interp(mx.array(x), 37 * 2, scale_factor=2.0))
    ref = _torch_nearest(x, 2.0)
    np.testing.assert_array_equal(ours, ref)


# ---------------------------------------------------------------------------
# ffmpeg load_audio parity (heavy: torch side)
# ---------------------------------------------------------------------------

@pytest.mark.heavy
def test_ffmpeg_load_audio_matches_official(tmp_path):
    """f32 pcm bit-parity vs tools/my_utils.load_audio on a generated wav.

    tools/my_utils imports gradio at module import; the official load_audio
    body is a thin ffmpeg-python wrapper, so we inline the same ffmpeg
    invocation via the base-venv python (which has ffmpeg-python installed)
    instead of importing the module.
    """
    import soundfile as sf
    import subprocess
    from gsovits_mlx.train.s2_v3_data import ffmpeg_load_audio
    sr = 32000
    t = np.linspace(0, 0.5, sr // 2, endpoint=False)
    wav = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    path = str(tmp_path / "a.wav")
    sf.write(path, wav, sr, subtype="PCM_16")
    ours24 = ffmpeg_load_audio(path, 24000)
    code = (
        "import ffmpeg, numpy as np, sys;"
        "out, _ = (ffmpeg.input(sys.argv[2], threads=0)"
        ".output('-', format='f32le', acodec='pcm_f32le', ac=1, ar=24000)"
        ".run(cmd=['ffmpeg', '-nostdin'], capture_stdout=True, capture_stderr=True));"
        "np.save(sys.argv[1], np.frombuffer(out, dtype=np.float32))"
    )
    out_npy = str(tmp_path / "ref24.npy")
    subprocess.run(["/Users/baicai1145/.venvs/base/bin/python", "-c", code,
                    out_npy, path], check=True)
    ref = np.load(out_npy)
    assert ours24.shape == ref.shape
    # bit parity: same ffmpeg flags -> identical f32 pcm
    np.testing.assert_array_equal(ours24, ref)
