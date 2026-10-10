"""Light CPU tests for the CFM-trainer memory optimizations (task cfm-mem-opt).

Covers, without GPU or model weights:
1. LengthBucketSampler determinism: same (seed, epoch) -> identical batches;
   different epochs differ; batch composition comes from ONE boundary bucket
   (similar lengths); last-batch padding never exceeds batch_size; every
   index appears (possibly repeated) — official DistributedBucketSampler
   invariants (s2_data.py lineage).
2. Pad-quantization loss-neutrality: collate with pad_multiple=64 widens
   ssl/spec/mel rows by zero columns ONLY beyond every length; all lengths
   arrays and the un-padded data prefix are IDENTICAL to the official
   collate — which is exactly the information the training loss consumes
   (loss slices [lo:mel_len_i]; masks key off *_lengths). A CPU CFM-loss
   check on a stub estimator verifies the numeric claim end-to-end.
3. collate_for version dispatch keeps official semantics when pad_multiple
   is None (default-off contract).

Run: pytest -q tests/test_train_s2v3_memopt.py
"""

import math
import os
import random
import sys

import numpy as np
import pytest

import mlx.core as mx

mx.set_default_device(mx.cpu)

from gsovits_mlx.train.s2_cfm import CFMTrainingLoss  # noqa: E402
from gsovits_mlx.train.s2_v3_data import (  # noqa: E402
    LengthBucketSampler, collate_for, collate_v3, collate_v4)


# ---------------------------------------------------------------------------
# synthetic samples: (ssl (1,768,T), spec (1025,S), mel (100,M), text ids)
# ---------------------------------------------------------------------------

def _sample(t: int, seed: int):
    rng = np.random.default_rng(seed)
    ssl = rng.uniform(-1, 1, (1, 768, t)).astype(np.float16)
    spec = rng.uniform(-2, 2, (1025, t)).astype(np.float32)
    mel = rng.uniform(-1, 1, (100, t)).astype(np.float32)
    text = rng.choice(np.arange(1, 700), size=min(t, 8), replace=False) \
        .astype(np.int64)
    return ssl, spec, mel, text


def _make_dataset(n=40, seed=7):
    rng = np.random.default_rng(seed)
    lengths = rng.integers(40, 950, size=n).tolist()
    samples = [_sample(t, seed * 1000 + i) for i, t in enumerate(lengths)]
    return lengths, samples


# ---------------------------------------------------------------------------
# 1. LengthBucketSampler
# ---------------------------------------------------------------------------

def test_length_bucket_sampler_determinism():
    lengths, _ = _make_dataset()
    s = LengthBucketSampler(lengths, batch_size=3, seed=1234)
    b1 = s.batch_indices(epoch=0)
    b2 = s.batch_indices(epoch=0)
    assert b1 == b2, "same (seed, epoch) must give identical batches"
    assert s.batch_indices(epoch=1) != b1 or s.batch_indices(epoch=2) != b1, \
        "epoch reshuffle expected for this dataset"
    # fresh sampler, same seed -> identical (no hidden state)
    s2 = LengthBucketSampler(lengths, batch_size=3, seed=1234)
    assert s2.batch_indices(epoch=0) == b1
    # different seed -> (almost surely) different order
    s3 = LengthBucketSampler(lengths, batch_size=3, seed=99)
    assert s3.batch_indices(epoch=0) != b1


def test_length_bucket_sampler_official_invariants():
    lengths, _ = _make_dataset(n=37, seed=11)
    bs = 3
    s = LengthBucketSampler(lengths, batch_size=bs, seed=0)
    batches = s.batch_indices(epoch=3)
    assert len(batches) == len(s)  # __len__ contract
    bounds = s.boundaries
    for b in batches:
        assert 1 <= len(b) <= bs, "batch size must never exceed batch_size"
        # all members of a batch come from ONE (lo, hi] bucket —
        # the similar-lengths property the memory win depends on
        first = lengths[b[0]]
        for i in b:
            assert _bucket_of(lengths[i], bounds) == _bucket_of(first, bounds)
    # coverage: every dataset index appears at least once
    seen = set(i for b in batches for i in b)
    assert seen == set(range(len(lengths)))


def _bucket_of(x, bounds):
    for i in range(len(bounds) - 1):
        if bounds[i] < x <= bounds[i + 1]:
            return i
    return -1


# ---------------------------------------------------------------------------
# 2. pad-quantization loss-neutrality
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("collate,ver", [(collate_v3, "v3"), (collate_v4, "v4")])
def test_pad_quantization_loss_neutral(collate, ver):
    lengths, samples = _make_dataset(n=6, seed=3)
    batch = samples[:4]
    plain = collate(batch)
    quant = collate(batch, pad_multiple=64)
    # widths grew to multiples of 64 (or stayed)
    for name in ("ssl", "spec", "mel"):
        w_p, w_q = plain.__getattribute__(name).shape[-1], \
            quant.__getattribute__(name).shape[-1]
        assert w_q >= w_p
        assert w_q % 64 == 0
    # lengths arrays are IDENTICAL (the loss/mask inputs)
    for name in ("ssl_lengths", "spec_lengths", "mel_lengths",
                 "text_lengths", "text"):
        np.testing.assert_array_equal(plain.__getattribute__(name),
                                      quant.__getattribute__(name))
    # data prefix identical; extra columns are exactly zero
    for name in ("ssl", "spec", "mel"):
        a, b = plain.__getattribute__(name), quant.__getattribute__(name)
        w = a.shape[-1]
        np.testing.assert_allclose(a, b[..., :w], rtol=0, atol=0)
        assert float(np.abs(b[..., w:]).max()) == 0.0


def test_pad_quantization_cfm_loss_identical():
    """End-to-end numeric claim: the CFM loss (stub estimator) is unchanged
    by pad quantization, because every consumption path is bounded by
    x_lens/mel_lengths slices."""
    lengths, samples = _make_dataset(n=4, seed=5)
    batch = samples[:3]
    prompt_lens = [5, 0, 3]

    class _Est:
        """Deterministic stub: consumes (B,C,T) -> (B,T,C)."""
        def __call__(self, xt, prompt, x_lens, t, d, mu):
            seed = float(mx.sum(xt))
            key = mx.random.key(int(abs(seed)) % (2**31))
            return mx.random.normal((xt.shape[0], xt.shape[2], xt.shape[1]),
                                    key=key)

    class _PinnedEst:
        """Same random draw regardless of padded width: seed from the
        UNPADDED region only (slices by x_lens), so padding cannot change
        the estimator output on the loss region."""
        def __init__(self, inner):
            self.inner = inner
        def __call__(self, xt, prompt, x_lens, t, d, mu):
            xl = [int(v) for v in x_lens]
            outs = []
            for i, L in enumerate(xl):
                sub = xt[i:i+1, :, :L]
                o = self.inner(sub, prompt[i:i+1, :, :L],
                               [L], t[i:i+1], d[i:i+1], mu[i:i+1, :, :L])
                # (1, T, C) -> scatter into the full padded width
                pad = xt.shape[2] - L
                if pad > 0:
                    o = mx.concatenate(
                        [o, mx.zeros((1, pad, xt.shape[1]))], axis=1)
                outs.append(o)
            return mx.concatenate(outs, axis=0)

    losses = {}
    # draw-fixed comparison: det state fixes t/x0 for both runs. x0 is drawn
    # ONCE at the PADDED batch width and sliced to each run's mel width
    # (both batches share the unpadded prefix because padding only appends
    # zero columns beyond every length).
    mel_q = mx.array(collate_v3(batch, pad_multiple=64).mel)
    det = {"t": mx.array([0.1, 0.5, 0.9]),
           "x0": mx.random.normal(mel_q.shape, key=mx.random.key(9)),
           "prompt_lens": mx.array(prompt_lens),
           "gate": 0.5}  # >= 0.3 -> no two-step branch
    for tag, batch_obj in (("plain", collate_v3(batch)),
                           ("pad64", collate_v3(batch, pad_multiple=64))):
        est = _PinnedEst(_Est())
        mel = mx.array(batch_obj.mel)
        x_lens = mx.array(batch_obj.mel_lengths, dtype=mx.float32)
        mu = mx.array(np.zeros((batch_obj.mel.shape[0], 512,
                                batch_obj.mel.shape[-1]), np.float32))
        loss, _ = CFMTrainingLoss(est, rng=random.Random(42))(
            mel, x_lens, prompt_lens, mu, key=mx.random.key(123), det=det)
        losses[tag] = float(loss)
    assert math.isclose(losses["plain"], losses["pad64"], rel_tol=1e-9), \
        f"pad quantization changed CFM loss: {losses}"


def test_collate_for_default_official():
    """Default (no pad_multiple) keeps the exact official collates."""
    lengths, samples = _make_dataset(n=4, seed=13)
    b = samples[:3]
    assert collate_for("v3") is collate_v3
    assert collate_for("v4") is collate_v4
    out = collate_for("v3")(b)
    assert out.mel.shape[-1] == out.ssl.shape[-1] * 0 + \
        int(8 * (max(s[1].shape[1] for s in b) // 8 + 1) * 1.25 * 1.5)
    out4 = collate_for("v4")(b)
    assert out4.mel.shape[-1] == out4.spec.shape[-1] * 2
    # opt-in path wraps with the multiple
    q = collate_for("v3", pad_multiple=64)(b)
    assert q.mel.shape[-1] % 64 == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
