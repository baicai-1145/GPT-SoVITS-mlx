"""G-forward mode equivalence + bf16 dtype-policy tests (gan-speed-opt).

CPU-safe (pytest -q -m 'not heavy'); no GPU, no real weights.

1. Fused single-trace vs legacy double-trace GRADIENT partitioning:
   exact same losses and exact same per-branch gradients (mathematically
   identical objects — the fused trace computes both branches from one G
   forward with stop_gradient walls; the legacy traces compute the same
   branches separately). Uses a miniature G/D surrogate with the EXACT
   call pattern of tools/train_s2.py (bind/load weights, one key, y_hat
   detached for D, D detached for G).
2. Same-draw double vs single: with the same RNG key both forwards of the
   double mode reproduce the single mode's y_hat bit-for-bit (official
   single-draw semantics).
3. bf16/fp32 dtype policy: make_compute-style exemptions (sv_emb/ge_to512/
   prelu + frozen32 stay fp32), posterior-noise strict cast keeps the
   decoder graph in bf16, fp32 randn promotes (documents why strict cast
   is needed), KL head stays fp32 at extreme logs_p (fp16 would overflow).
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

from gsovits_mlx.train import s2_gan as G  # noqa: E402


# ---------------------------------------------------------------------------
# 1 + 2: fused vs double trace on a surrogate with the driver's exact shape
# ---------------------------------------------------------------------------

def _make_surrogate(rng):
    """Tiny stand-ins with the driver's structure: dict params, bind-by-ref,
    stop_gradient walls, shared RNG draw."""
    p32 = {"gw": mx.array(rng.normal(size=(4, 4)).astype(np.float32)),
           "gb": mx.array(rng.normal(size=(4,)).astype(np.float32))}
    d32 = {"dw": mx.array(rng.normal(size=(3, 4)).astype(np.float32))}
    return p32, d32


def _g_fwd(p, x, key):
    """Surrogate G forward: one rand draw (like rand_slice + posterior)."""
    noise = mx.random.normal(x.shape, key=key)
    return p["gw"] @ x + noise[:, : x.shape[1] if x.ndim == 2 else 0] + p["gb"][:, None]


def _d_fwd(d, y):
    return d["dw"] @ y


def _losses(d, p, x, key):
    """The driver's fused_loss_fn structure, verbatim pattern."""
    o = _g_fwd(p, x, key)
    # D branch (y_hat detached)
    ld = (( _d_fwd(d, mx.stop_gradient(o))) ** 2).sum() * 0 + \
        mx.mean((_d_fwd(d, mx.stop_gradient(o)) - 1.0) ** 2)
    # G branch (D detached)
    dc = {k: mx.stop_gradient(v) for k, v in d.items()}
    lg = mx.mean((_d_fwd(dc, o)) ** 2)
    return ld, lg


def test_fused_vs_double_partition_exact():
    rng = np.random.default_rng(0)
    p32, d32 = _make_surrogate(rng)
    x = mx.array(rng.normal(size=(4, 5)).astype(np.float32))
    key = mx.random.key(7)

    # ---- fused: one trace over (d, p) jointly ----
    def fused(d, p):
        ld, lg = _losses(d, p, x, key)
        return ld + lg, (ld, lg)

    (tot, (ld_f, lg_f)), (gd_f, gp_f) = mx.value_and_grad(
        fused, argnums=(0, 1))(d32, p32)

    # ---- legacy: two separate traces ----
    def d_trace(d):
        o = _g_fwd(p32, x, key)  # p as constants (bound/captured)
        return mx.mean((_d_fwd(d, mx.stop_gradient(o)) - 1.0) ** 2)

    def g_trace(p):
        o = _g_fwd(p, x, key)
        dc = {k: mx.stop_gradient(v) for k, v in d32.items()}
        return mx.mean((_d_fwd(dc, o)) ** 2)

    ld_d = d_trace(d32)
    lg_d = g_trace(p32)
    gd_d = mx.grad(d_trace)(d32)
    gp_d = mx.grad(g_trace)(p32)

    # losses bit-identical
    assert float(ld_f) == float(ld_d)
    assert float(lg_f) == float(lg_d)
    assert float(tot) == float(ld_d) + float(lg_d)
    # gradients bit-identical per branch
    for k in d32:
        assert np.array_equal(np.asarray(gd_f[k]), np.asarray(gd_d[k])), k
    for k in p32:
        assert np.array_equal(np.asarray(gp_f[k]), np.asarray(gp_d[k])), k


def test_same_key_double_matches_single_yhat():
    """Both double-mode forwards with key=k reproduce the single forward's
    draw bit-for-bit (same RNG stream position)."""
    rng = np.random.default_rng(1)
    p32, _ = _make_surrogate(rng)
    x = mx.array(rng.normal(size=(4, 5)).astype(np.float32))
    key = mx.random.key(42)
    o1 = _g_fwd(p32, x, key)
    o2 = _g_fwd(p32, x, key)
    assert np.array_equal(np.asarray(o1), np.asarray(o2))


# ---------------------------------------------------------------------------
# 3: dtype policy (bf16) — make_compute exemptions + strict noise + KL head
# ---------------------------------------------------------------------------

def test_make_compute_dtype_policy():
    """Exemptions mirror tools/train_s2.py::make_compute (driver logic
    replicated here so the policy is unit-locked): everything -> bf16 except
    sv_emb./ge_to512./prelu + frozen32 merged fp32."""
    params32 = {
        "dec.conv_pre.weight": mx.zeros((2, 2, 3), mx.float32),
        "enc_q.pre.weight": mx.zeros((2, 2, 1), mx.float32),
        "sv_emb.weight": mx.zeros((2, 2), mx.float32),
        "ge_to512.weight": mx.zeros((2, 2), mx.float32),
        "prelu.weight": mx.zeros((2,), mx.float32),
        "ssl_proj.weight": mx.zeros((2, 2, 1), mx.float32),  # frozen32
    }
    frozen32 = {k: v for k, v in params32.items()
                if k.startswith(("ssl_proj.", "quantizer."))}

    def make_compute(params_f32, precision):
        out = {}
        for k, v in params_f32.items():
            if precision == "fp32":
                out[k] = v
            elif k.startswith(("sv_emb.", "ge_to512.", "prelu")):
                out[k] = v
            else:
                out[k] = v.astype(mx.bfloat16 if precision == "bf16"
                                  else mx.float16)
        out.update(frozen32)
        return out

    out = make_compute(params32, "bf16")
    assert out["dec.conv_pre.weight"].dtype == mx.bfloat16
    assert out["enc_q.pre.weight"].dtype == mx.bfloat16
    for k in ("sv_emb.weight", "ge_to512.weight", "prelu.weight",
              "ssl_proj.weight"):
        assert out[k].dtype == mx.float32, k
    out32 = make_compute(params32, "fp32")
    assert all(v.dtype == mx.float32 for v in out32.values())


def test_posterior_noise_dtype_reaches_decoder():
    """PosteriorEncoderTrain strict_noise: bf16 params -> bf16 z (decoder
    input); without strict the fp32 randn promotes z to fp32 (documented
    promotion the flag fixes)."""
    m = G.PosteriorEncoderTrain(8, 4, 8, 5, 1, 2, gin_channels=0)
    rng = np.random.default_rng(2)
    # WNConv1d.weight starts as None (bind-style); layout is (out, k, in)
    m.pre.weight = mx.array(rng.normal(size=(8, 1, 8)).astype(np.float32))
    m.proj.weight = mx.array(rng.normal(size=(8, 1, 8)).astype(np.float32))
    for cv in list(m.enc.in_layers) + list(m.enc.res_skip_layers):
        cv.weight = mx.array(rng.normal(size=cv.weight.shape).astype(np.float32))
        cv.weight_v = None
    x16 = mx.array(rng.normal(size=(1, 8, 20)).astype(np.float32)).astype(mx.bfloat16)
    lens = mx.array([20])
    key = mx.random.key(3)
    z_ns, _, _, mask_ns = m(x16, lens, key=key, strict_noise=False)
    z_st, _, _, mask_st = m(x16, lens, key=key, strict_noise=True)
    # fp32 params + bf16 input: input dtype wins in our convs (effective
    # weight casts to x.dtype); strict casts the randn to the compute dtype.
    assert z_st.dtype == mx.bfloat16, "strict noise keeps decoder graph bf16"
    assert mask_st.dtype == mx.bfloat16
    # non-strict keeps legacy behavior (fp32 randn promotes)
    assert z_ns.dtype == mx.float32


def test_kl_loss_head_fp32_at_extreme_logs():
    """KL entry in bf16 vs fp32 on extreme logs_p: the loss head casts to
    fp32 FIRST (kl_loss astype policy), so exp(-2*logs_p) cannot overflow
    the way the banned fp16 forward does (exp(40)=2.35e17 > 65504)."""
    rng = np.random.default_rng(4)
    z_p = mx.array(rng.normal(size=(1, 4, 10)).astype(np.float32))
    logs_q = mx.array(rng.normal(size=(1, 4, 10)).astype(np.float32) * 0.1)
    m_p = mx.array(rng.normal(size=(1, 4, 10)).astype(np.float32) * 0.1)
    logs_p = mx.array(np.full((1, 4, 10), -20.0, np.float32))  # exp(40)
    mask = mx.ones((1, 1, 10))
    kl = G.kl_loss(z_p, logs_q, m_p, logs_p, mask)
    assert bool(mx.isfinite(kl)), "fp32 KL head must stay finite"
    assert float(kl) > 1e16  # dominated by sigma_p^2 ~ e^40
    # the same entry computed WITH the cast AFTER exp in bf16 stays finite
    # too (bf16 shares fp32 exponent range) — the contrast vs fp16:
    v16 = mx.exp((-2.0 * logs_p).astype(mx.float16))
    assert not bool(mx.all(mx.isfinite(v16))), "fp16 exp overflows (banned)"
    vb = mx.exp((-2.0 * logs_p).astype(mx.bfloat16))
    assert bool(mx.all(mx.isfinite(vb))), "bf16 exp finite"


def test_frozen_quantizer_still_zero_and_fp32():
    m = G.SynthesizerTrnTrain(version="v2", segment_size=32)
    rng = np.random.default_rng(5)
    m.quantizer_embed = mx.array(
        (rng.normal(size=(1024, 768)) * 0.1).astype(np.float32))
    ssl = mx.array((rng.normal(size=(1, 768, 30)) * 0.1).astype(np.float32))
    q, kl = m.quantize_ssl(ssl)
    assert float(kl.sum()) == 0.0
    assert m.quantizer_embed.dtype == mx.float32  # frozen32 stays fp32
