"""RVQ quantizer (module/quantize.py, inference path) — pure MLX."""

from __future__ import annotations

import mlx.core as mx


class ResidualVectorQuantizer:
    """n_q=1 codebook wrapper for SoVITS ssl quantizer.

    Checkpoint layout: quantizer.vq.layers.0._codebook.embed (1024, 768).
    encode: nearest neighbour in L2; decode: table lookup.
    """

    def __init__(self, embed: mx.array):
        # (K, C)
        self.embed = embed

    def encode(self, x: mx.array) -> mx.array:
        """x: (B, C, T) -> codes (B, 1, T)."""
        xb = mx.transpose(x, (0, 2, 1))  # (B, T, C)
        d = (mx.sum(xb**2, axis=-1, keepdims=True)
             + mx.sum(self.embed**2, axis=-1)[None, None, :]
             - 2.0 * xb @ self.embed.T)
        codes = mx.argmin(d, axis=-1)[:, None, :]  # (B, 1, T)
        return codes.astype(mx.int32)

    def decode(self, codes: mx.array) -> mx.array:
        """codes: (B, 1, T) -> (B, C, T)."""
        return mx.transpose(self.embed[codes[:, 0].astype(mx.int32)], (0, 2, 1))
