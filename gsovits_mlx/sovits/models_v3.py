"""SynthesizerTrnV3 for v3/v4/v5dev/v5turbo. Pure MLX.

Reference: GPT-SoVITS-CPUFast module/models.py SynthesizerTrnV3 (inference path:
decode_encp / extract_latent only; training forward omitted).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ..utils.layers import Conv1d, LayerNormChannels, WN, sequence_mask
from .attentions import Encoder
from .cfm import CFM, CFMV5
from .dit import DiT
from .models_v1v2 import _nearest_interp
from .quantizer import ResidualVectorQuantizer


class WNEncoder(nn.Module):
    """models.Encoder (models.py:397) as used for wns1: pre -> WN(gin) -> proj.
    Checkpoint has no final norm; returns (out, x_mask)."""

    def __init__(self, in_channels, out_channels, hidden_channels, kernel_size,
                 dilation_rate, n_layers, gin_channels=0):
        super().__init__()
        self.pre = Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = Conv1d(hidden_channels, out_channels, 1)

    def __call__(self, x, x_lengths, g=None):
        x_mask = sequence_mask(x_lengths, int(x.shape[2]))
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        return self.proj(x) * x_mask, x_mask


class SynthesizerTrnV3(nn.Module):
    """v3 / v4 / v5dev / v5turbo inference model (no enc_q / flow / dec)."""

    def __init__(self, spec_channels, segment_size, inter_channels, hidden_channels,
                 filter_channels, n_heads, n_layers, kernel_size, p_dropout,
                 resblock, resblock_kernel_sizes, resblock_dilation_sizes,
                 upsample_rates, upsample_initial_channel, upsample_kernel_sizes,
                 n_speakers=0, gin_channels=0, use_sdp=True, semantic_frame_rate=None,
                 freeze_quantizer=None, version="v3", **kwargs):
        super().__init__()
        self.version = version
        self.model_dim = 512
        self.semantic_frame_rate = semantic_frame_rate or "25hz"

        n_symbols = 732
        from .models_v1v2 import TextEncoder
        self.enc_p = TextEncoder(inter_channels, hidden_channels, filter_channels, n_heads,
                                 n_layers, kernel_size, p_dropout, n_symbols=n_symbols)
        self.ref_enc = MelStyleEncoderLocal(704, style_vector_dim=gin_channels)

        ssl_dim = 768
        if semantic_frame_rate == "25hz":
            self.ssl_proj = Conv1d(ssl_dim, ssl_dim, 2, stride=2)
        else:
            self.ssl_proj = Conv1d(ssl_dim, ssl_dim, 1, stride=1)

        inter_channels2 = 512
        self.bridge_0 = Conv1d(inter_channels, inter_channels2, 1)
        self.wns1 = WNEncoder(inter_channels2, inter_channels2, inter_channels2, 5, 1, 8,
                              gin_channels=gin_channels)
        self.linear_mel = Conv1d(inter_channels2, 100, 1)
        dit = DiT(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=inter_channels2,
                  conv_layers=4, use_step_embedding=version not in {"v5", "v5dev", "v5turbo"})
        if version in {"v5", "v5dev", "v5turbo"}:
            self.cfm = CFMV5(100, dit)
        else:
            self.cfm = CFM(100, dit)
        self.quantizer = ResidualVectorQuantizer(mx.zeros((1024, ssl_dim)))

    # -- shared encoder trunk -----------------------------------------------
    def decode_encp(self, codes, text, refer=None, ge=None, speed=1.0):
        """codes: (B,1,T); text: (B,Tph). Returns (fea, ge). fea is 50hz * extra factor."""
        if ge is None:
            refer_mask = mx.ones((refer.shape[0], 1, refer.shape[2]), dtype=refer.dtype)
            ge = self.ref_enc(refer[:, :704] * refer_mask, refer_mask)
        y_lengths = mx.array([int(codes.shape[2] * 2)])
        if speed == 1:
            sizee = int(codes.shape[2] * (3.875 if self.version == "v3" else 4))
        else:
            sizee = int(codes.shape[2] * (3.875 if self.version == "v3" else 4) / speed) + 1
        y_lengths1 = mx.array([sizee])
        text_lengths = mx.array([text.shape[-1]])

        quantized = self.quantizer.decode(codes)
        if self.semantic_frame_rate == "25hz":
            quantized = _nearest_interp(quantized, quantized.shape[-1] * 2)
        x, m_p, logs_p, y_mask = self.enc_p(quantized, y_lengths, text, text_lengths, ge, speed)
        fea = nn.leaky_relu(self.bridge_0(x), 0.01)
        sc = 1.875 if self.version == "v3" else 2
        fea = _nearest_interp(fea, int(fea.shape[-1] * sc), scale_factor=sc)
        fea, _ = self.wns1(fea, y_lengths1, ge)
        return fea, ge

    def extract_latent(self, x: mx.array) -> mx.array:
        """ssl (B, 768, T) -> codes (B, T) int32. Official SynthesizerTrnV3.extract_latent
        returns codes.transpose(0, 1) of shape (B, 1, T); callers index [0, 0],
        folded here into a (B, T) return."""
        ssl = self.ssl_proj(x)
        codes = self.quantizer.encode(ssl)  # (B, 1, T)
        return codes[:, 0]

    def forward(self, *a, **kw):
        raise NotImplementedError("SynthesizerTrnV3 is inference-only; use decode_encp")


class MelStyleEncoderLocal:
    """ref_enc for v3+: modules.MelStyleEncoder(704) via the shared implementation."""

    def __new__(cls, *a, **kw):
        from .mrte import MelStyleEncoder
        return MelStyleEncoder(*a, **kw)
