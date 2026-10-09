"""Tests for the s2 (SoVITS v1/v2/v2Pro/v2ProPlus) GAN trainer.

CPU-safe subset (runs under `pytest -q -m 'not heavy'`, MLX CPU device):
  * bucket sampler: batch-composition determinism per epoch, coverage,
    bucket-boundary placement, pad-to-batch-multiple repeats.
  * collate: sort-by-spec_len-desc, even max lengths (2*((max//2)+1)),
    zero padding, Pro sv_embs handling.
  * weight-norm effective-weight closed forms (Conv1d out-channel /
    Conv2d out-channel / ConvTranspose1d in-channel).
  * export fuse helpers vs numpy golden.
  * frozen-quantizer kl_ssl semantics (exactly 0.0 when frozen; mse commit
    value when not — official quantizer.eval() semantics).

Torch goldens (self-skip when torch is absent in the running venv; re-run
under /Users/baicai1145/.venvs/base via subprocess):
  * spec parity: MLX stft_magnitude vs torch.stft-based spectrogram_torch
    semantics on 20 random signals + a real wav (max diff <= 1e-5).
  * weight-norm GRAD parity vs torch.nn.utils.weight_norm autograd
    (Conv1d / Conv2d / ConvTranspose1d): output, input-grad and (g,v)-grads
    all <= 1e-5.
  * loss parity vs the official module/losses.py.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from gsovits_mlx.train.s2_data import BucketSampler, TextAudioSpeakerCollate  # noqa: E402

BASE_VENV_PY = "/Users/baicai1145/.venvs/base/bin/python"
OFFICIAL = ("/Users/baicai1145/repos/gpt-sovits/GPT-SoVITS-cuda_graph_accel_v5/"
            "GPT_SoVITS")
REF_WAV = os.path.join(repo_root, "models_local/ref_zh_3.5s.wav")
HAVE_TORCH = importlib.util.find_spec("torch") is not None


# ---------------------------------------------------------------------------
# bucket sampler (official DistributedBucketSampler, num_replicas=1)
# ---------------------------------------------------------------------------

def test_bucket_sampler_determinism_and_coverage():
    rng = np.random.default_rng(7)
    lengths = rng.integers(35, 1950, size=97).tolist()
    bs = 6
    s = BucketSampler(lengths, bs)
    s.set_epoch(1)
    b1 = [list(b) for b in s]
    s.set_epoch(1)
    b1b = [list(b) for b in s]
    assert b1 == b1b, "same epoch must give identical batch composition/order"
    s.set_epoch(2)
    b2 = [list(b) for b in s]
    counts = np.bincount([x for b in b1 for x in b], minlength=97)
    assert (counts >= 1).all(), "every sample must appear at least once"
    assert sum(counts) == s.num_samples  # bucket pad-to-multiple accounting
    assert all(len(b) == bs for b in b1)
    for b in b1:
        for i in b:
            # each member inside one bucket interval (with repeats allowed)
            ls = [lengths[j] for j in b]
            k = np.searchsorted(np.asarray(s.boundaries), lengths[i], side="left")
            assert 0 < k < len(s.boundaries)


def test_bucket_sampler_tiny_bucket_repeat():
    lengths = [33] * 5 + [1000] * 7  # bucket pads: 5->6, 7->12
    s = BucketSampler(lengths, 6)
    assert sorted(s.num_samples_per_bucket) == [6, 12]
    s.set_epoch(0)
    b1 = list(s)
    assert sum(len(b) for b in b1) == 18
    assert all(len(b) == 6 for b in b1)


# ---------------------------------------------------------------------------
# collate
# ---------------------------------------------------------------------------

def _fake_item(spec_t, wav_n, text_len):
    ssl = np.zeros((1, 768, spec_t), np.float32)
    spec = np.arange(spec_t, dtype=np.float32)[None].repeat(1025, 0) / spec_t
    wav = np.zeros(wav_n, np.float32)
    text = np.arange(text_len, dtype=np.int64) % 322
    return (ssl, spec, wav, text)


def test_collate_shapes_and_sort():
    items = [_fake_item(301, 301 * 640, 20),
             _fake_item(400, 400 * 640, 15),
             _fake_item(299, 299 * 640, 10)]
    out = TextAudioSpeakerCollate()(items)
    ssl, ssl_len, spec, spec_len, wav, wav_len, text, text_len = out
    n = 3
    assert ssl.shape == (n, 768, 2 * ((400 // 2) + 1))
    assert spec.shape == (n, 1025, 2 * ((400 // 2) + 1))
    assert wav.shape == (n, 1, 400 * 640)
    assert text.shape == (n, 20)
    assert spec_len.tolist() == [400, 301, 299]
    assert spec[0, :, :400].sum() > 0 and spec[0, :, 400:].sum() == 0
    assert text_len.tolist() == [15, 20, 10]


def test_collate_pro_sv():
    items = [(*_fake_item(300, 300 * 640, 9), np.full((1, 20480), 0.5, np.float32)),
             (*_fake_item(302, 302 * 640, 11), np.full((1, 20480), -0.25, np.float32))]
    out = TextAudioSpeakerCollate(version="v2Pro")(items)
    assert len(out) == 9
    sv = out[8]
    assert sv.shape == (2, 20480)
    assert np.allclose(sv[0], 0.5) and np.allclose(sv[1], -0.25)


# ---------------------------------------------------------------------------
# weight-norm math (closed form, torch semantics)
# ---------------------------------------------------------------------------

def test_weight_norm_effective_closed_form():
    from gsovits_mlx.train.s2_discriminator import WeightNormConv1d, WeightNormConv2d
    from gsovits_mlx.utils.layers import Conv1d, ConvTranspose1d

    rng = np.random.default_rng(3)
    c1 = WeightNormConv1d(8, 16, 5)
    c1.weight_v = mx.array(rng.normal(size=(16, 5, 8)).astype(np.float32))
    c1.weight_g = mx.array(rng.normal(size=(16,)).astype(np.float32))
    w = np.asarray(c1.weight)
    v = np.asarray(c1.weight_v)
    exp = (np.asarray(c1.weight_g)[:, None, None]
           * v / np.linalg.norm(v, axis=(1, 2), keepdims=True))
    assert np.abs(w - exp).max() < 1e-6

    c2 = WeightNormConv2d(8, 16, (5, 1))
    c2.weight_v = mx.array(rng.normal(size=(16, 8, 5, 1)).astype(np.float32))
    c2.weight_g = mx.array(rng.normal(size=(16,)).astype(np.float32))
    w2 = np.asarray(c2.weight)
    v2 = np.asarray(c2.weight_v)
    exp2 = (np.asarray(c2.weight_g)[:, None, None, None]
            * v2 / np.linalg.norm(v2, axis=(1, 2, 3), keepdims=True))
    assert np.abs(w2 - exp2).max() < 1e-6

    ct = ConvTranspose1d(16, 8, 5)
    ct.weight_v = mx.array(rng.normal(size=(8, 5, 16)).astype(np.float32))
    ct.weight_g = mx.array(rng.normal(size=(16,)).astype(np.float32))
    w3 = np.asarray(ct.effective_weight(mx.float32))  # (out,k,in)
    v3 = np.asarray(ct.weight_v)
    g3 = np.asarray(ct.weight_g)
    exp3 = v3 / np.linalg.norm(v3, axis=(0, 1), keepdims=True) * g3[None, None, :]
    assert np.abs(w3 - exp3).max() < 1e-6


def test_export_fuse_golden():
    from tools.train_s2_export import fuse_wn, fuse_wn_t
    rng = np.random.default_rng(11)
    g = mx.array(rng.normal(size=(16,)).astype(np.float32))
    v = mx.array(rng.normal(size=(16, 5, 8)).astype(np.float32))
    w = np.asarray(fuse_wn(g, v))
    v_np, g_np = np.asarray(v), np.asarray(g)
    exp = g_np[:, None, None] * v_np / np.linalg.norm(v_np, axis=(1, 2), keepdims=True)
    assert np.abs(w - exp).max() < 1e-6
    gt = mx.array(rng.normal(size=(8,)).astype(np.float32))
    vt = mx.array(rng.normal(size=(16, 5, 8)).astype(np.float32))  # (out,k,in)
    wt = np.asarray(fuse_wn_t(gt, vt))
    vt_np, gt_np = np.asarray(vt), np.asarray(gt)
    expt = vt_np / np.linalg.norm(vt_np, axis=(0, 1), keepdims=True) * gt_np[None, None, :]
    assert np.abs(wt - expt).max() < 1e-6


def test_frozen_quantizer_kl_ssl_zero():
    from gsovits_mlx.train import s2_gan as G
    rng = np.random.default_rng(5)
    m = G.SynthesizerTrnTrain(version="v2", segment_size=32)
    m.quantizer_embed = mx.array(
        (rng.normal(size=(1024, 768)) * 0.1).astype(np.float32))
    ssl = mx.array((rng.normal(size=(2, 768, 40)) * 0.1).astype(np.float32))
    q, kl = m.quantize_ssl(ssl)
    assert float(kl.sum()) == 0.0, "frozen quantizer -> kl_ssl exactly 0"
    assert q.shape == (2, 768, 20)
    q2, commit = m._quantize_ssl(ssl, with_commit=True)
    assert np.allclose(np.asarray(q), np.asarray(q2), atol=0)
    proj = np.asarray(m.ssl_proj(ssl.astype(mx.float32)))
    exp = float(np.mean((np.asarray(q2) - proj) ** 2))
    assert abs(float(commit.sum()) - exp) < 1e-6


def test_losses_closed_form():
    from gsovits_mlx.train import s2_gan as G
    d_r = [mx.array(np.full((1, 4), 0.5))]
    d_g = [mx.array(np.zeros((1, 4)))]
    ld, rl, gl = G.discriminator_loss(d_r, d_g)
    assert abs(float(ld) - 0.25) < 1e-6  # (1-0.5)^2 + 0^2
    lg, _ = G.generator_loss(d_g)
    assert abs(float(lg) - 1.0) < 1e-6  # (1-0)^2
    fmap_r = [[mx.array(np.full((1, 4, 5), 1.0))]]
    fmap_g = [[mx.array(np.zeros((1, 4, 5)))]]
    lf = G.feature_loss(fmap_r, fmap_g)
    assert abs(float(lf) - 2.0) < 1e-6  # 2 * mean|1-0| = 2


# ---------------------------------------------------------------------------
# torch goldens (single subprocess script; torch + mlx live in base venv)
# ---------------------------------------------------------------------------

_GOLDEN = r'''
import os, sys
sys.path.insert(0, %(repo)r)
import numpy as np
import torch
import mlx.core as mx
mx.set_default_device(mx.cpu)

FAIL = []

def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    if not cond:
        FAIL.append(name)

# ---- 1. spec parity: stft_magnitude vs spectrogram_torch -------------------
from gsovits_mlx.text.mel_frontend import stft_magnitude

def torch_spec(y, n_fft, hop, win):
    spec = torch.stft(
        torch.from_numpy(y), n_fft, hop_length=hop, win_length=win,
        window=torch.hann_window(win, dtype=torch.float32),
        center=False, pad_mode="reflect", normalized=False, onesided=True,
        return_complex=True).abs()
    # official spectrogram_torch: sqrt(|stft|^2) (energy->magnitude)
    return spec.pow(2).sqrt().numpy()

rng = np.random.default_rng(0)
worst = 0.0
for i in range(20):
    n = int(rng.integers(8000, 40000))
    y = (rng.normal(size=n) * 0.1).astype(np.float32)
    t = torch_spec(y[None], 2048, 640, 2048)[0]
    m = np.asarray(stft_magnitude(mx.array(y[None]), 2048, 640, 2048))[0]
    assert t.shape == m.shape, (t.shape, m.shape)
    worst = max(worst, np.abs(m - t).max())
check("spec_parity_random20 (20 files, max %.3e <= 1e-5)" % worst, worst <= 1e-5)

import soundfile as sf
if os.path.exists(%(wav)r):
    audio, sr = sf.read(%(wav)r, dtype="float32")
    worst_w = 0.0
    for start in range(0, min(len(audio) - 40000, 120000), 20000):
        seg = audio[start:start + 40000]
        t = torch_spec(seg[None], 2048, 640, 2048)[0]
        m = np.asarray(stft_magnitude(mx.array(seg[None]), 2048, 640, 2048))[0]
        worst_w = max(worst_w, np.abs(m - t).max())
    check("spec_parity_wav (real ref audio, max %.3e <= 1e-5)" % worst_w,
          worst_w <= 1e-5)

# ---- 2. weight-norm GRAD parity vs torch autograd ---------------------------
from gsovits_mlx.train.s2_discriminator import WeightNormConv1d, WeightNormConv2d
from gsovits_mlx.utils.layers import ConvTranspose1d as MLXConvT1d

def run_case(name, mlx_layer, torch_layer, x_np, v_layout, is_t=False):
    """v_layout maps torch (g,v) into the MLX-layer storage layout."""
    rng = np.random.default_rng(abs(hash(name)) %% 1000)
    g_t = (rng.normal(size=torch_layer.weight_g.shape).astype(np.float32))
    with torch.no_grad():
        torch_layer.weight_g.copy_(torch.from_numpy(g_t))
    v_t = torch_layer.weight_v.detach().numpy()
    v_m = v_layout(v_t)
    mlx_layer.weight_v = mx.array(v_m)
    if is_t:
        mlx_layer.weight_g = mx.array(g_t.reshape(-1))
    else:
        mlx_layer.weight_g = mx.array(g_t.reshape(g_t.shape[0]))
    if getattr(mlx_layer, "bias", None) is not None:
        mlx_layer.bias = mx.array(torch_layer.bias.detach().numpy())

    x = mx.array(x_np)
    xt = torch.from_numpy(x_np).requires_grad_(True)
    o = mlx_layer(x)
    lt = torch_layer(xt)
    od = np.abs(np.asarray(o.astype(mx.float32)) - lt.detach().numpy()).max()
    check(name + ": forward <= 1e-5 (%.2e)" % od, od <= 1e-5)

    # grads wrt (v, g) together and wrt input
    def fn(v_, g_, x_):
        mlx_layer.weight_v, mlx_layer.weight_g = v_, g_
        out = mlx_layer(x_)
        return (out * out).sum() / out.size
    (dv, dg, dx) = mx.grad(fn, argnums=(0, 1, 2))(
        mlx_layer.weight_v, mlx_layer.weight_g, x)
    lt2 = (lt * lt).sum() / lt.numel()
    torch_layer.zero_grad(set_to_none=True)
    if xt.grad is not None:
        xt.grad = None
    lt2.backward()
    tv = torch_layer.weight_v.grad.detach().numpy()
    tg = torch_layer.weight_g.grad.detach().numpy().reshape(-1)
    tx = xt.grad.detach().numpy()
    tv_m = v_layout(tv)
    dvd = np.abs(np.asarray(dv.astype(mx.float32)) - tv_m).max()
    dgd = np.abs(np.asarray(dg.astype(mx.float32)) - tg).max()
    dxd = np.abs(np.asarray(dx.astype(mx.float32)) - tx).max()
    ok = dvd <= 1e-5 and dgd <= 1e-5 and dxd <= 1e-5
    check(name + ": grads(v %.2e g %.2e x %.2e) <= 1e-5" % (dvd, dgd, dxd), ok)

torch.manual_seed(1)
B, C, T = 2, 8, 37
x1 = np.random.default_rng(10).normal(size=(B, C, T)).astype(np.float32)
tc1 = torch.nn.Conv1d(C, 16, 5, padding=2, bias=True)
torch.nn.utils.weight_norm(tc1, dim=0)
run_case("wn_conv1d", WeightNormConv1d(C, 16, 5, padding=2), tc1, x1,
         lambda v: np.ascontiguousarray(v.transpose(0, 2, 1)))

x2 = np.random.default_rng(11).normal(size=(B, C, 10, 4)).astype(np.float32)
tc2 = torch.nn.Conv2d(C, 16, (5, 1), stride=(3, 1), padding=(2, 0), bias=True)
torch.nn.utils.weight_norm(tc2, dim=0)
run_case("wn_conv2d", WeightNormConv2d(C, 16, (5, 1), (3, 1), (2, 0)), tc2, x2,
         lambda v: v)

x3 = np.random.default_rng(12).normal(size=(B, C, T)).astype(np.float32)
tc3 = torch.nn.ConvTranspose1d(C, 16, 5, stride=3, padding=2, bias=True)
torch.nn.utils.weight_norm(tc3, dim=0)
mlxt = MLXConvT1d(C, 16, 5, stride=3, padding=2)
run_case("wn_convT1d", mlxt, tc3, x3,
         lambda v: np.ascontiguousarray(v.transpose(1, 2, 0)), is_t=True)

# ---- 3. losses parity vs official module/losses.py ---------------------------
off = %(official)r
sys.path.insert(0, off)
os.chdir(off)
import module.losses as tloss
from gsovits_mlx.train import s2_gan as G

rng = np.random.default_rng(21)
def mk(shape):
    return (rng.normal(size=shape) * 1.0).astype(np.float32)
fmap_r = [[mx.array(mk((2, 4, 17)))] for _ in range(6)]
fmap_g = [[mx.array(mk((2, 4, 17)))] for _ in range(6)]
tf_r = [[torch.from_numpy(np.asarray(t)) for t in sub] for sub in fmap_r]
tf_g = [[torch.from_numpy(np.asarray(t)) for t in sub] for sub in fmap_g]
d = abs(float(G.feature_loss(fmap_r, fmap_g)) - float(tloss.feature_loss(tf_r, tf_g)))
check("feature_loss (%.2e <= 1e-6)" % d, d <= 1e-6)

dr = [mx.array(mk((2, 33))) for _ in range(6)]
dg = [mx.array(mk((2, 33))) for _ in range(6)]
ldm, _, _ = G.discriminator_loss(dr, dg)
ldt, _, _ = tloss.discriminator_loss(
    [torch.from_numpy(np.asarray(t)) for t in dr],
    [torch.from_numpy(np.asarray(t)) for t in dg])
check("discriminator_loss", abs(float(ldm) - float(ldt)) <= 1e-6)

lgm, _ = G.generator_loss(dg)
lgt, _ = tloss.generator_loss([torch.from_numpy(np.asarray(t)) for t in dg])
check("generator_loss", abs(float(lgm) - float(lgt)) <= 1e-6)

z, lq, mp, lp = mk((2, 8, 30)), mk((2, 8, 30)), mk((2, 8, 30)), mk((2, 8, 30))
klm = G.kl_loss(mx.array(z), mx.array(lq), mx.array(mp), mx.array(lp),
                mx.ones((2, 1, 30)))
klt = tloss.kl_loss(torch.from_numpy(z), torch.from_numpy(lq),
                    torch.from_numpy(mp), torch.from_numpy(lp),
                    torch.ones(2, 1, 30))
check("kl_loss", abs(float(klm) - float(klt)) <= 1e-6)

print("SUMMARY_FAIL=" + str(len(FAIL)))
sys.exit(1 if FAIL else 0)
'''


def _run_golden():
    r = subprocess.run(
        [BASE_VENV_PY, "-c", _GOLDEN % {
            "repo": repo_root, "official": OFFICIAL, "wav": REF_WAV}],
        capture_output=True, text=True, timeout=1800)
    return r


@pytest.mark.skipif(not os.path.isfile(BASE_VENV_PY), reason="base venv absent")
@pytest.mark.skipif(HAVE_TORCH, reason="torch importable here; goldens run in-venv")
def test_golden_via_base_venv():
    r = _run_golden()
    print(r.stdout)
    assert r.returncode == 0, r.stdout + "\n---stderr---\n" + r.stderr


@pytest.mark.skipif(not HAVE_TORCH, reason="torch not in this venv")
def test_golden_torch_inplace():
    # same script; when torch exists in the running venv it is re-executed
    # under the base interpreter anyway for a pinned torch (2.x) — still
    # counts as running in-place when the venv IS base.
    r = _run_golden()
    print(r.stdout)
    assert r.returncode == 0, r.stdout + "\n---stderr---\n" + r.stderr
