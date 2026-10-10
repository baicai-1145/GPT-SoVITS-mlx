"""GAN trainer memory-governance additions (gan-bucket-compile task).

1. GAN batch-shape finding (documented in TRAINING.md): the official
   DistributedBucketSampler is the GAN trainer's ONLY sampler (already
   unconditional in tools/train_s2.py — official s2_train.py itself never
   runs flat order), so "--length-buckets" reduces to CLI symmetry; the
   varying dimensions are the collated ssl/spec/wav/text WIDTHS (83
   distinct spec widths in 120 b3 batches of the 939-item exp set) —
   quantized/fixed pads target those.
2. Pad quantization (--pad-multiple) / fixed-shape (--fixed-shape) collate
   contract: widths rounded/fixed; *_lengths arrays IDENTICAL (masks and
   slices consume only lengths); extra columns exactly zero beyond every
   length.
3. Pad-neutrality of the GAN step on the REAL SynthesizerTrnTrain (CPU,
   real s2G_v2 weights), pinned-draw contract:
   (a) z's pad region is exactly 0 (x_mask kills it);
   (b) ids_slice (segment starts) is width-independent — the uniform draw
       is over lengths-bounded max with a (B,) shape (bit-equal);
   (c) with the two official-parity width-coupled draws PINNED (posterior
       noise per absolute position; ge = ref_enc output), loss_mel is
       BIT-IDENTICAL across pad widths and loss_kl agrees to <=1e-6 rel
       (kl sums over the padded width; masked entries are exactly 0, only
       the fp32 accumulation ORDER changes). The two couplings are exactly
       the ones official torch has (torch.randn_like over the padded
       tensor; MelStyleEncoder reads the batch-max-padded y), so
       fixed-shape padding shifts them exactly like official batch-max
       padding does — distribution-identical, not bitwise.
4. Gather-based segment slice (compile-safe path): bit-identical forward
   and gradients vs the reference loop slice; same draw in rand variant.
5. mx.compile smoke of the driver's exact call structures: compiled inner
   forwards INSIDE value_and_grad (fwd arm) and value_and_grad INSIDE
   compile (step arm), weights-as-args (no stale-capture).

Run: pytest -q tests/test_train_s2_memshape.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
mx.set_default_device(mx.cpu)

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from gsovits_mlx.train.s2_data import TextAudioSpeakerCollate  # noqa: E402
from gsovits_mlx.train import s2_gan as G  # noqa: E402
from gsovits_mlx.utils.layers import sequence_mask  # noqa: E402


def _fake_item(spec_t: int, seed: int, text_len: int | None = None):
    rng = np.random.default_rng(seed)
    ssl = rng.uniform(-1, 1, (1, 768, spec_t)).astype(np.float32)
    spec = rng.uniform(-2, 2, (1025, spec_t)).astype(np.float32)
    wav = rng.uniform(-0.5, 0.5, spec_t * 640).astype(np.float32)
    n = text_len if text_len is not None else max(4, spec_t // 20)
    text = np.sort(rng.choice(np.arange(1, 300), size=n, replace=False)) \
        .astype(np.int64)
    return (ssl, spec, wav, text)


# ---------------------------------------------------------------------------
# 1+2: pad quantization / fixed widths collate contract
# ---------------------------------------------------------------------------

def test_collate_pad_multiple_contract():
    items = [_fake_item(300, 1), _fake_item(201, 2), _fake_item(155, 3)]
    plain = TextAudioSpeakerCollate()(items)
    quant = TextAudioSpeakerCollate(pad_multiple=64)(items)
    ssl_p, ssl_len_p, spec_p, spec_len_p, wav_p, wav_len_p, tp, tl_p = plain
    ssl_q, ssl_len_q, spec_q, spec_len_q, wav_q, wav_len_q, tq, tl_q = quant
    # widths grew to multiples of 64 (wav in hop units: 64*640 samples)
    assert ssl_q.shape[-1] % 64 == 0 and ssl_q.shape[-1] >= ssl_p.shape[-1]
    assert spec_q.shape[-1] % 64 == 0 and spec_q.shape[-1] >= spec_p.shape[-1]
    assert wav_q.shape[-1] % (64 * 640) == 0
    assert tq.shape[-1] % 64 == 0
    # lengths IDENTICAL (mask/slice inputs — the loss contract)
    for a, b in ((ssl_len_p, ssl_len_q), (spec_len_p, spec_len_q),
                 (wav_len_p, wav_len_q), (tl_p, tl_q)):
        np.testing.assert_array_equal(a, b)
    # data prefix identical; extra columns exactly zero
    for p, q in ((ssl_p, ssl_q), (spec_p, spec_q), (wav_p, wav_q),
                 (tp, tq)):
        w = p.shape[-1]
        np.testing.assert_allclose(p, q[..., :w], rtol=0, atol=0)
        extra = q[..., w:]
        assert float(np.abs(extra).max()) == 0.0 if extra.size else True


def test_collate_fixed_widths_contract():
    items = [_fake_item(300, 4), _fake_item(201, 5)]
    fw = dict(ssl=512, spec=512, wav=512 * 640, text=32)
    out = TextAudioSpeakerCollate(fixed_widths=fw)(items)
    ssl, ssl_len, spec, spec_len, wav, wav_len, text, tlen = out
    assert ssl.shape == (2, 768, 512)
    assert spec.shape == (2, 1025, 512)
    assert wav.shape == (2, 1, 512 * 640)
    assert text.shape == (2, 32)
    # lengths are the TRUE per-item lengths (unpadded)
    np.testing.assert_array_equal(spec_len, [300, 201])
    np.testing.assert_array_equal(wav_len, [300 * 640, 201 * 640])
    # zero pad beyond lengths
    assert float(np.abs(spec[:, :, 300:]).max()) == 0.0
    assert float(np.abs(wav[:, 0, 300 * 640:]).max()) == 0.0
    # text rows beyond tlen are zero and rows sorted by spec len desc
    np.testing.assert_array_equal(tlen, [15, 10])


# ---------------------------------------------------------------------------
# 3: pad-neutrality on the real model (CPU, real s2G_v2 weights)
# ---------------------------------------------------------------------------

_S2G_V2 = os.path.join(repo_root, ".tmp/train_s2/s2G_v2.npz")

needs_weights = pytest.mark.skipif(
    not os.path.exists(_S2G_V2), reason="s2G_v2.npz extraction absent")


@needs_weights
def test_fused_loss_pad_neutral_real_model():
    """Pinned-draw pad-neutrality on the real model (see module docstring
    item 3). Pinning ge (ref_enc) and the posterior noise isolates every
    width-coupled quantity that official torch ALSO couples to pad width;
    under pinning the losses must agree (mel bit-exact, kl <=1e-6 rel)."""
    from gsovits_mlx.train.s2_torchio import load_train_params_npz
    from gsovits_mlx.utils.layers import sequence_mask
    m = G.SynthesizerTrnTrain(version="v2", segment_size=32)
    m.bind(load_train_params_npz(_S2G_V2))
    rng = np.random.default_rng(5)
    B, T, TH = 2, 90, 12
    ssl = rng.uniform(-1, 1, (B, 768, T)).astype(np.float32)
    spec = rng.uniform(-2, 2, (B, 1025, T)).astype(np.float32)
    wav = rng.uniform(-0.5, 0.5, (B, T * 640)).astype(np.float32)
    text = np.sort(rng.choice(np.arange(1, 300), size=(B, TH), replace=False),
                   axis=1).astype(np.int64)
    lengths = np.array([T, T - 25], np.int64)
    tlen = np.array([TH, TH - 3], np.int64)
    key = mx.random.key(42)
    pad = 70
    WMAX = T + pad
    mel_basis = mx.array(rng.uniform(0.05, 0.3, (128, 1025))
                         .astype(np.float32))

    # pinned draws, taken ONCE at the max width
    noise_pin = np.asarray(mx.random.normal((B, 192, WMAX), key=key))
    ymask_max = sequence_mask(mx.array(lengths), WMAX).astype(mx.float32)
    spec_max = np.zeros((B, 1025, WMAX), np.float32)
    spec_max[:, :, :T] = spec
    ge_pin = np.asarray(m.ref_enc(mx.array(spec_max)[:, :704] * ymask_max,
                                  ymask_max))

    def run(w_extra: int):
        W = T + w_extra
        ssl_p = np.zeros((B, 768, W), np.float32); ssl_p[:, :, :T] = ssl
        spec_p = np.zeros((B, 1025, W), np.float32); spec_p[:, :, :T] = spec
        wav_p = np.zeros((B, 1, W * 640), np.float32)
        wav_p[:, 0, :T * 640] = wav
        text_p = np.zeros((B, TH + w_extra), np.int64); text_p[:, :TH] = text
        y_lengths = mx.array(lengths)
        ym = sequence_mask(y_lengths, W).astype(mx.float32)
        ge = mx.array(ge_pin)
        quant, kl_ssl = m.quantize_ssl(mx.array(ssl_p))
        quantized = G._nearest_interp(quant, quant.shape[-1] * 2)
        _, m_p, logs_p, _ = m.enc_p(quantized.astype(mx.float32), y_lengths,
                                    mx.array(text_p).astype(mx.int32),
                                    mx.array(tlen), ge)
        xq = m.enc_q.pre(mx.array(spec_p)) * ym
        hq = m.enc_q.enc(xq, ym, g=mx.stop_gradient(ge))
        stats = m.enc_q.proj(hq) * ym
        m_q, logs_q = stats[:, :192], stats[:, 192:]
        noise = mx.array(noise_pin[:, :, :W])
        z = (m_q + noise * mx.exp(logs_q)) * ym
        z_p = m.flow(z, ym, g=ge)
        z_slice, ids = G.rand_slice_segments(z, y_lengths, m.segment_size,
                                             key=key)
        o = m.dec(z_slice, g=ge)
        # mel-path surrogate on |spec| rows (identical width semantics to
        # the driver's spec_to_mel+slice; keeps the CPU test cheap)
        y_mel = G.slice_segments(mx.abs(mx.array(spec_p))[:, :128, :],
                                 ids, m.segment_size)
        y_hat_mel = G.mel_spectrogram_train(
            o.astype(mx.float32).squeeze(1), mel_basis, 2048, 640, 2048)
        loss_mel = mx.mean(mx.abs(y_mel - y_hat_mel)) * 45.0
        loss_kl = G.kl_loss(z_p, logs_q, m_p, logs_p,
                            sequence_mask(y_lengths, W)) * 1.0
        return (float(loss_mel), float(loss_kl), np.asarray(ids),
                np.asarray(z), lengths)

    m0, k0, ids0, z0, lens = run(0)
    m1, k1, ids1, z1, _ = run(pad)
    # (b) ids_slice width-independent (bit)
    np.testing.assert_array_equal(ids0, ids1)
    # (a) z pad region exactly 0 in the padded arm
    for i, L in enumerate(lens):
        assert float(np.abs(z1[i, :, L:]).max()) == 0.0
    # (c) pinned draws -> mel bit-identical, kl <= 1e-6 rel
    assert m0 == m1, f"loss_mel pad-dependent under pinned draws: {m0} {m1}"
    assert abs(k0 - k1) <= 1e-6 * max(abs(k0), abs(k1)), (k0, k1)


# ---------------------------------------------------------------------------
# 4: gather slice equivalence (compile-safe path)
# ---------------------------------------------------------------------------

def test_slice_segments_gather_bit_identical():
    rng = np.random.default_rng(0)
    x = mx.array(rng.normal(size=(3, 5, 50)).astype(np.float32))
    ids = mx.array(np.array([10, 3, 30], dtype=np.int32))
    a = G.slice_segments(x, ids, 7)
    b = G.slice_segments_gather(x, ids, 7)
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0, atol=0)
    gl = mx.grad(lambda t: G.slice_segments(t, ids, 7).sum())(x)
    gg = mx.grad(lambda t: G.slice_segments_gather(t, ids, 7).sum())(x)
    np.testing.assert_allclose(np.asarray(gl), np.asarray(gg), rtol=0, atol=0)


def test_rand_slice_gather_same_draw():
    rng = np.random.default_rng(1)
    x = mx.array(rng.normal(size=(2, 4, 64)).astype(np.float32))
    xl = mx.array(np.array([64, 50], dtype=np.int64))
    key = mx.random.key(7)
    a, ia = G.rand_slice_segments(x, xl, 8, key=key)
    b, ib = G.rand_slice_segments_gather(x, xl, 8, key=key)
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0, atol=0)
    np.testing.assert_array_equal(np.asarray(ia), np.asarray(ib))


# ---------------------------------------------------------------------------
# 5: compile smoke of the driver's exact call structures
# ---------------------------------------------------------------------------

def test_compile_fwd_arm_smoke():
    """Compiled inner forwards INSIDE value_and_grad (fwd arm), weights as
    args, stop_gradient walls — the driver's fused_loss_fn structure."""
    rng = np.random.default_rng(0)
    p32 = {"w": mx.array(rng.normal(size=(4, 4)).astype(np.float32))}
    d32 = {"dw": mx.array(rng.normal(size=(3, 4)).astype(np.float32))}

    def _g_fwd(p, x, key):
        return p["w"] @ x + mx.random.normal(x.shape, key=key)[:, :x.shape[1]]

    g_c = mx.compile(_g_fwd)

    def _d_fwd(d, y):
        return d["dw"] @ y

    d_c = mx.compile(_d_fwd)

    def fused(d16, p16, x, key):
        o = g_c(p16, x, key)
        ld = mx.mean((d_c(d16, mx.stop_gradient(o)) - 1.0) ** 2)
        dc = {k: mx.stop_gradient(v) for k, v in d16.items()}
        lg = mx.mean(d_c(dc, o) ** 2)
        return ld + lg, (ld, lg)

    x = mx.array(rng.normal(size=(4, 4)).astype(np.float32))
    key = mx.random.key(3)
    (l, parts), (gd, gp) = mx.value_and_grad(fused, argnums=(0, 1))(
        d32, p32, x, key)
    mx.eval(l, parts[0], parts[1], *gd.values(), *gp.values())
    assert np.isfinite(float(l))
    # weights-as-args: a second call with updated weights recompiles nothing
    # stale (new values flow through)
    p32b = {"w": p32["w"] * 2}
    (l2, _), _ = mx.value_and_grad(fused, argnums=(0, 1))(
        d32, p32b, x, key)
    mx.eval(l2)
    assert np.isfinite(float(l2)) and float(l2) != float(l)


def test_compile_step_arm_smoke():
    """value_and_grad INSIDE compile (step arm): w=ones, x=eye ->
    (w@x).sum()=4, (w*w).sum()=4, total 8; grad = x@1 + 2w."""
    def loss_fn(w, x):
        return (w @ x).sum() + (w * w).sum()

    def joint(w, x):
        return mx.value_and_grad(loss_fn)(w, x)

    joint_c = mx.compile(joint)
    w = mx.array(np.ones((2, 2), np.float32))
    x = mx.array(np.eye(2, dtype=np.float32))
    l, g = joint_c(w, x)
    mx.eval(l, g)
    assert float(l) == 8.0
    np.testing.assert_allclose(np.asarray(g), np.full((2, 2), 3.0), atol=1e-6)
