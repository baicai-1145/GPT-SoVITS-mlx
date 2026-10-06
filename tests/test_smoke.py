"""Smoke tests: converter stage-name mapping + lightweight pipeline pieces.

Heavy e2e runs need converted weights on /Volumes/2T and are marked `heavy`
(deselect with `pytest -m 'not heavy'`).
"""
from __future__ import annotations

import numpy as np
import pytest


def test_bench_dispatch_covers_all_versions():
    from tools.bench import E2E_SCRIPTS

    assert set(E2E_SCRIPTS) == {"v1", "v2", "v2Pro", "v2ProPlus", "v3", "v4",
                                "v5dev", "v5turbo"}
    for name, (script, _extra, _v2args) in E2E_SCRIPTS.items():
        assert script.startswith("e2e_"), name


def test_kaldi_fbank_shape_and_finiteness():
    from gsovits_mlx.sovits.kaldi_fbank import fbank

    wav = np.random.RandomState(0).randn(16000, ).astype(np.float32)  # 1 s @16k
    feat = fbank(wav)
    assert feat.ndim == 2 and feat.shape[1] == 80
    assert feat.shape[0] == 1 + (16000 - 400) // 160  # snip_edges frame count
    assert np.isfinite(feat).all()


def test_mlx_conv2d_matches_manual_1x1():
    import mlx.core as mx

    from gsovits_mlx.sovits.sv_encoder import Conv2d

    rng = np.random.RandomState(1)
    conv = Conv2d(8, 4, 1, 1, 0)
    conv.weight = mx.array(rng.randn(4, 1, 1, 8).astype(np.float32))
    conv.bias = mx.array(rng.randn(4).astype(np.float32))
    x = mx.array(rng.randn(1, 6, 5, 8).astype(np.float32))
    y = np.array(conv(x))[0]  # (H, W, C)
    w = np.array(conv.weight)[:, 0, 0]  # (4, 8)
    b = np.array(conv.bias)
    for h in range(6):
        for wi in range(5):
            ref = w @ x[0, h, wi] + b
            assert np.allclose(y[h, wi], ref, atol=1e-5)


@pytest.mark.heavy
def test_sv_encoder_parity_reference():
    """Full-precision check vs the captured torch reference (needs 2T models)."""
    import os

    import mlx.core as mx

    from gsovits_mlx.pipeline import load_sv_encoder

    models = "/Volumes/2T/gpt-sovits-models/mlx/sv"
    ref = os.path.join(os.path.dirname(__file__), "..", ".tmp", "sv_ref.npz")
    if not (os.path.isdir(models) and os.path.isfile(ref)):
        pytest.skip("converted sv model / torch reference not available")
    m, _ = load_sv_encoder(models)
    feat = np.load(ref)["feat"]
    emb = np.array(m.forward3(mx.array(feat[None])))
    assert np.abs(emb - np.load(ref)["emb"]).max() < 1e-4
