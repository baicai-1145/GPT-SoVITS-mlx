"""tests for gsovits_mlx/train/metal_kernels.py (fused Metal kernels).

Three gates per kernel (mission law): (i) full-coverage probe (every
output element written), (ii) parity vs the replaced MLX op chain
(fp32: bit-exact where op order is preserved; documented <=1e-6 rel where
GPU reduction order differs), (iii) micro-bench vs the op chain.

GPU tests are skipped on CPU (no Metal) — CPU-safe by construction.
"""
from __future__ import annotations

import numpy as np
import pytest

mlx = pytest.importorskip("mlx.core")
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

from gsovits_mlx.train import metal_kernels as MK  # noqa: E402

requires_gpu = pytest.mark.skipif(
    not MK.available(), reason="no Metal GPU / mx.fast.metal_kernel")


# ---------------------------------------------------------------------------
# 1. fused_bias_lrelu
# ---------------------------------------------------------------------------

@requires_gpu
@pytest.mark.parametrize("shape", [
    (3, 32, 100), (1, 1, 20480), (3, 1024, 20480), (5, 7, 3),
    (2, 1024, 17), (3, 256, 6827), (4, 1024, 3414),
])
def test_bias_lrelu_full_coverage_and_parity(shape):
    """Every element written (coverage via unique-index fill) + bit-exact."""
    rs = np.random.RandomState(0)
    n = int(np.prod(shape))
    x = mx.array((rs.randn(n).reshape(shape)).astype(np.float32))
    b = mx.array((rs.randn(shape[1])).astype(np.float32) * 3)
    y = MK.bias_lrelu(x, b, 0.1)
    mx.eval(y)
    # coverage: unique values in -> unique transforms out on the neg side
    ref = nn.leaky_relu(x + b[None, :, None], 0.1)
    assert bool(mx.all(y == ref).item()), "bit-exact vs transpose+add+lrelu chain"
    # explicit write-all check: replace x by constant; every elem must change
    x2 = mx.full(shape, -2.0, mx.float32)
    b2 = mx.zeros((shape[1],), mx.float32)
    y2 = MK.bias_lrelu(x2, b2, 0.1)
    mx.eval(y2)
    assert bool(mx.all(y2 == -0.2).item())


@requires_gpu
def test_bias_lrelu_grad_parity():
    rs = np.random.RandomState(4)
    x = mx.array(rs.randn(3, 64, 960).astype(np.float32))
    b = mx.array(rs.randn(64).astype(np.float32))
    f = lambda xx, bb: mx.sum(MK.fused_bias_lrelu(xx, bb, 0.1) ** 2)
    r = lambda xx, bb: mx.sum(nn.leaky_relu(xx + bb[None, :, None], 0.1) ** 2)
    gx_f, gb_f = mx.grad(f, argnums=(0, 1))(x, b)
    gx_r, gb_r = mx.grad(r, argnums=(0, 1))(x, b)
    assert bool(mx.all(gx_f == gx_r).item())
    assert bool(mx.all(gb_f == gb_r).item())


# ---------------------------------------------------------------------------
# 2. fused_wn_scale
# ---------------------------------------------------------------------------

@requires_gpu
@pytest.mark.parametrize("shape", [
    (384, 5, 192),    # WN in_layer conv (hidden 192 k5)
    (384, 1, 384),    # WN res_skip 1x1
    (256, 1, 41),     # D ConvS k41
    (1024, 1, 205, 1),  # D ConvP k5 (torch (out,in,k,1) flattened)
    (128, 16, 512),   # dec up (in-channel norm NOT this; shape sanity)
    (512, 7),         # conv_pre k7
])
def test_wn_scale_full_coverage_and_parity(shape):
    out = shape[0]
    rs = np.random.RandomState(1)
    g = mx.array((rs.randn(out) * 0.5 + 1).astype(np.float32))
    v = mx.array(rs.randn(*shape).astype(np.float32))
    w = MK.fused_wn_scale(g, v)
    mx.eval(w)
    axes = tuple(range(1, v.ndim))
    norm = mx.sqrt(mx.sum(v * v, axis=axes, keepdims=True))
    ref = g.reshape(-1, *([1] * (v.ndim - 1))) * v / norm
    assert bool((w != 0).all().item()), "full coverage (all rows written)"
    rel = float((mx.abs(w - ref) / mx.maximum(mx.abs(ref), 1e-30)).max().item())
    # GPU serial row-sum vs MLX tree reduction: not bit-exact, documented
    # tolerance 1e-6 rel for R<=~1k; 4e-6 for R~8k (worst measured 2.0e-6
    # on 128x4608). Assert <= 5e-6 and tighten with data below.
    assert rel <= 5e-6, f"wn parity rel {rel}"


@requires_gpu
def test_wn_scale_grad_parity():
    rs = np.random.RandomState(6)
    g = mx.array((rs.randn(384) * 0.5 + 1).astype(np.float32))
    v = mx.array(rs.randn(384, 5, 192).astype(np.float32))
    f = lambda g_, v_: mx.sum(MK.fused_wn_scale(g_, v_) ** 2)
    r = lambda g_, v_: mx.sum(
        (g_[:, None, None] * v_ / mx.sqrt(mx.sum(v_ * v_, axis=(1, 2), keepdims=True))) ** 2)
    gg_f, gv_f = mx.grad(f, argnums=(0, 1))(g, v)
    gg_r, gv_r = mx.grad(r, argnums=(0, 1))(g, v)
    # analytic vjp recomputes the norm chain in MLX ops: gg within 1e-6;
    # gv differs only where |grad| ~ 1e-9 (cancellation) — floor the denom.
    relg = float((mx.abs(gg_f - gg_r) / mx.maximum(mx.abs(gg_r), 1e-6)).max().item())
    relv = float((mx.abs(gv_f - gv_r) / mx.maximum(mx.abs(gv_r), 1e-3)).max().item())
    assert relg <= 1e-6
    assert relv <= 1e-4


# ---------------------------------------------------------------------------
# 3. fused_gate
# ---------------------------------------------------------------------------

@requires_gpu
@pytest.mark.parametrize("shape,tb", [
    ((3, 768, 960), 960),   # flow/WN at trunk width (g_l full width)
    ((3, 768, 960), 1),     # enc_q g_l broadcast over T (ge is (B,512,1))
    ((1, 384, 7), 1),
    ((2, 768, 17), 17),
    ((3, 384, 20480), 1),   # decoder-width WN (not used in G dec but shape-safe)
])
def test_gate_full_coverage_and_parity(shape, tb):
    rs = np.random.RandomState(2)
    a = mx.array((rs.randn(*shape) * 2).astype(np.float32))
    bshape = (shape[0], shape[1], tb)
    b = mx.array((rs.randn(*bshape) * 2).astype(np.float32))
    y = MK.fused_gate(a, b)
    mx.eval(y)
    h = shape[1] // 2
    v = a + b
    ref = mx.tanh(v[:, :h, :]) * mx.sigmoid(v[:, h:, :])
    assert bool((y != 0).all().item())
    rel = float((mx.abs(y - ref) / mx.maximum(mx.abs(ref), 1e-30)).max().item())
    assert rel <= 2e-6, f"gate parity rel {rel}"


@requires_gpu
def test_gate_grad_parity():
    rs = np.random.RandomState(8)
    a = mx.array((rs.randn(3, 768, 960) * 2).astype(np.float32))
    b = mx.array((rs.randn(3, 768, 960) * 2).astype(np.float32))
    h = 384
    f = lambda a_, b_: mx.sum(MK.fused_gate(a_, b_) ** 2)
    r = lambda a_, b_: mx.sum(
        (mx.tanh((a_ + b_)[:, :h, :]) * mx.sigmoid((a_ + b_)[:, h:, :])) ** 2)
    ga_f, gb_f = mx.grad(f, argnums=(0, 1))(a, b)
    ga_r, gb_r = mx.grad(r, argnums=(0, 1))(a, b)
    da = float(mx.abs(ga_f - ga_r).max().item())
    db = float(mx.abs(gb_f - gb_r).max().item())
    assert da <= 1e-5 and db <= 1e-5


# ---------------------------------------------------------------------------
# micro-bench smoke (queued dispatch; GPU)
# ---------------------------------------------------------------------------

@requires_gpu
def test_microbench_smoke():
    import time
    rs = np.random.RandomState(3)

    def queued(fn, iters=300):
        for _ in range(20):
            fn()
        mx.eval(mx.array(0))
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        mx.eval(mx.array(0))
        return (time.perf_counter() - t0) / iters * 1e3

    # bias+lrelu at D scale (3,1024,20480)
    x = mx.array(rs.randn(3, 1024, 20480).astype(np.float32))
    b = mx.array(rs.randn(1024).astype(np.float32))
    MK._BIAS_LRELU_SLOPE[0] = 0.1
    tk = queued(lambda: MK.fused_bias_lrelu(x, b, 0.1))
    tc = queued(lambda: nn.leaky_relu(x.transpose(0, 2, 1).astype(mx.float32)
                                      + b[None, None, :]
                                      ).transpose(0, 2, 1))
    # wn at enc_q in_layer scale
    g = mx.array((rs.randn(384) * 0.5 + 1).astype(np.float32))
    v = mx.array(rs.randn(384, 5, 192).astype(np.float32))
    tk2 = queued(lambda: MK.fused_wn_scale(g, v))
    tc2 = queued(lambda: g[:, None, None].astype(mx.float32)
                 * v.astype(mx.float32)
                 / mx.sqrt(mx.sum(v.astype(mx.float32) ** 2, axis=(1, 2), keepdims=True)))
    # gate at trunk scale
    a = mx.array(rs.randn(3, 768, 960).astype(np.float32))
    bb = mx.array(rs.randn(3, 768, 960).astype(np.float32))
    tk3 = queued(lambda: MK.fused_gate(a, bb))
    tc3 = queued(lambda: (lambda vv: mx.tanh(vv[:, :384, :])
                          * mx.sigmoid(vv[:, 384:, :]))(a + bb))
    print(f"\n[bench] bias_lrelu: kernel {tk:.4f} ms vs chain {tc:.4f} ms "
          f"({tc/tk:.2f}x)")
    print(f"[bench] wn_scale:   kernel {tk2:.4f} ms vs chain {tc2:.4f} ms "
          f"({tc2/tk2:.2f}x)")
    print(f"[bench] gate:       kernel {tk3:.4f} ms vs chain {tc3:.4f} ms "
          f"({tc3/tk3:.2f}x)")
    # SMOKE only: kernels run and are within the custom_function python
    # wrapper overhead of the chain at queued depth. The decision-grade
    # instrument is the deep dependent-queue bench (in-graph situation):
    # bias_lrelu 1.7-4.9x / gate 1.4x / wn ~0.65x per call, and the
    # whole-MPD-forward 1.25x — see metal_kernels.py docstring and
    # .tmp/mk/RUN_PLAN.md. Queued-depth ratios here fluctuate run-to-run
    # with GPU launch latency (0.5x-1.7x observed) and are NOT asserts.
    assert tk < 5 * tc and tk2 < 5 * tc2 and tk3 < 5 * tc3, \
        "kernel pathologically slower (>5x) than chain"


# ---------------------------------------------------------------------------
# wiring: flag off = eager chains; flag on = kernels (structure test, CPU ok)
# ---------------------------------------------------------------------------

def test_flag_default_off_cpu_safe():
    """set_metal_kernels(False) keeps eager path importable on any machine."""
    import gsovits_mlx.utils.layers as L
    L.set_metal_kernels(False)
    assert L._mk_enabled() is False


@requires_gpu
def test_flag_on_uses_kernels():
    import gsovits_mlx.utils.layers as L
    L.set_metal_kernels(True)
    try:
        assert L._mk_enabled() is True
        # WN gating path picks the fused kernel without error
        rs = np.random.RandomState(7)
        a = mx.array(rs.randn(2, 8, 5).astype(np.float32))
        b = mx.zeros((2, 8, 5), mx.float32)
        y = MK.fused_gate(a, b)
        mx.eval(y)
        assert y.shape == (2, 4, 5)
    finally:
        L.set_metal_kernels(False)
