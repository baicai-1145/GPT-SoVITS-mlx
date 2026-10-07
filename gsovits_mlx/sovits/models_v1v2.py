"""SoVITS v1/v2 SynthesizerTrn (inference-only) + v2Pro additions. Pure MLX."""

from __future__ import annotations

import math
import os

import mlx.core as mx
import mlx.nn as nn

from ..utils.layers import (
    Conv1d,
    ConvTranspose1d,
    LayerNormChannels,
    ResBlock1,
    ResBlock2,
    WN,
    sequence_mask,
)
from .attentions import Encoder
from .mrte import MRTE, MelStyleEncoder
from .quantizer import ResidualVectorQuantizer


class TextEncoder(nn.Module):
    def __init__(self, out_channels, hidden_channels, filter_channels, n_heads, n_layers,
                 kernel_size, p_dropout, latent_channels=192, version="v2", n_symbols=732):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.version = version

        self.ssl_proj = Conv1d(768, hidden_channels, 1)
        self.encoder_ssl = Encoder(hidden_channels, filter_channels, n_heads, n_layers // 2,
                                   kernel_size, p_dropout)
        self.encoder_text = Encoder(hidden_channels, filter_channels, n_heads, n_layers,
                                    kernel_size, p_dropout)
        self.text_embedding = mx.zeros((n_symbols, hidden_channels))
        self.mrte = MRTE()
        self.encoder2 = Encoder(hidden_channels, filter_channels, n_heads, n_layers // 2,
                                kernel_size, p_dropout)
        self.proj = Conv1d(hidden_channels, out_channels * 2, 1)

    def __call__(self, y, y_lengths, text, text_lengths, ge, speed=1.0):
        y_mask = sequence_mask(y_lengths, int(y.shape[2]))
        y = self.ssl_proj(y * y_mask) * y_mask
        y = self.encoder_ssl(y * y_mask, y_mask)

        text_mask = sequence_mask(text_lengths, int(text.shape[1]))
        text = mx.transpose(self.text_embedding[text], (0, 2, 1))
        text = self.encoder_text(text * text_mask, text_mask)
        y = self.mrte(y, y_mask, text, text_mask, ge)

        y = self.encoder2(y * y_mask, y_mask)

        if speed != 1:
            new_len = int(y.shape[-1] / speed) + 1
            y = _linear_interp(y, new_len)
            y_mask = _nearest_interp(y_mask, new_len)
        stats = self.proj(y) * y_mask
        m, logs = stats[:, : self.out_channels], stats[:, self.out_channels :]
        return y, m, logs, y_mask


def _linear_interp(x: mx.array, size: int) -> mx.array:
    """F.interpolate(mode='linear') along last axis."""
    t_in = x.shape[-1]
    if t_in == size:
        return x
    pos = mx.arange(size, dtype=mx.float32) * ((t_in - 1) / max(size - 1, 1))
    i0 = mx.floor(pos).astype(mx.int32)
    i1 = mx.minimum(i0 + 1, t_in - 1)
    frac = (pos - i0)[None, None, :]
    a = x[:, :, i0]
    b = x[:, :, i1]
    return a * (1 - frac) + b * frac


def _nearest_interp(x: mx.array, size: int, scale_factor: float | None = None) -> mx.array:
    """torch F.interpolate(mode='nearest') exact semantics.

    With scale_factor, torch indexes src = floor(dst / scale_factor) (fp32 division);
    with size, src = floor(dst * t_in / size). These differ for non-integer scales (v3 1.875).
    """
    t_in = x.shape[-1]
    if t_in == size:
        return x
    if scale_factor is not None:
        idx = (mx.arange(size, dtype=mx.float32) / scale_factor).astype(mx.int32)
    else:
        idx = (mx.arange(size, dtype=mx.float32) * (t_in / size)).astype(mx.int32)
    idx = mx.clip(idx, 0, t_in - 1)
    return x[:, :, idx]


class Generator(nn.Module):
    """HiFi-GAN generator (SoVITS decoder), gin conditioning optional."""

    LRELU_SLOPE = 0.1

    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes,
                 upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=0):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        block_cls = ResBlock1 if resblock == "1" else ResBlock2
        self.ups = [
            ConvTranspose1d(upsample_initial_channel // (2**i),
                            upsample_initial_channel // (2 ** (i + 1)),
                            k, u, padding=(k - u) // 2)
            for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes))
        ]
        self.resblocks = []
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(block_cls(ch, k, d))
        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3)
        if gin_channels != 0:
            self.cond = Conv1d(gin_channels, upsample_initial_channel, 1)

    def __call__(self, x: mx.array, g: mx.array | None = None) -> mx.array:
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)
        # GSOVITS_HIFIGAN_STAGE_TRIM=1: eval + clear_cache per upsample stage.
        # Same disease/cure as the BigVGAN fix (6124273): freed transients
        # otherwise pile into MLX's buffer cache (measured ~5 GB on the
        # v2ProPlus 8.8 s stream). Placement-only; off by default pending
        # conv-fp16's task-8 A/B (timing + parity).
        trim = os.environ.get("GSOVITS_HIFIGAN_STAGE_TRIM") == "1"
        for i in range(self.num_upsamples):
            x = nn.leaky_relu(x, self.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = 0
            for j in range(self.num_kernels):
                xs = xs + self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
            if trim:
                mx.eval(x)
                try:
                    mx.clear_cache()
                except AttributeError:
                    mx.metal.clear_cache()
        x = nn.leaky_relu(x)
        x = self.conv_post(x)
        return mx.tanh(x)


class ResidualCouplingLayer(nn.Module):
    """modules.ResidualCouplingLayer, mean_only=True (all GPT-SoVITS ckpts)."""

    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate, n_layers,
                 gin_channels=0, mean_only=True):
        super().__init__()
        assert channels % 2 == 0
        self.half_channels = channels // 2
        self.mean_only = mean_only
        self.pre = Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.post = Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)

    def __call__(self, x, x_mask, g=None, reverse=False):
        x0, x1 = x[:, : self.half_channels], x[:, self.half_channels :]
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = stats[:, : self.half_channels], stats[:, self.half_channels :]
        else:
            m, logs = stats, mx.zeros_like(m := stats)

        if not reverse:
            x1 = m + x1 * mx.exp(logs) * x_mask
            return mx.concatenate([x0, x1], 1)
        else:
            x1 = (x1 - m) * mx.exp(-logs) * x_mask
            return mx.concatenate([x0, x1], 1)


class Flip(nn.Module):
    def __call__(self, x, *args, reverse=False, **kwargs):
        return x[:, ::-1, :]


class ResidualCouplingBlock(nn.Module):
    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate, n_layers,
                 n_flows=4, gin_channels=0):
        super().__init__()
        self.flows = []
        for _ in range(n_flows):
            self.flows.append(ResidualCouplingLayer(channels, hidden_channels, kernel_size,
                                                    dilation_rate, n_layers, gin_channels=gin_channels,
                                                    mean_only=True))
            self.flows.append(Flip())

    def __call__(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class PosteriorEncoder(nn.Module):
    """Training-only; kept for weight-compat tests. Not used at inference."""

    def __init__(self, in_channels, out_channels, hidden_channels, kernel_size, dilation_rate, n_layers,
                 gin_channels=0):
        super().__init__()
        self.pre = Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = Conv1d(hidden_channels, out_channels * 2, 1)

    def __call__(self, x, x_lengths, g=None):
        x_mask = sequence_mask(x_lengths, int(x.shape[2]))
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        return stats, x_mask


class RVQLayer:
    """ResidualVectorQuantizer inference pieces (n_q=1): nearest-codebook lookup + decode."""

    def __init__(self, codebook: mx.array):
        # codebook: (1, 1024, 768)
        self.codebook = codebook

    def decode(self, codes: mx.array) -> mx.array:
        # codes: (B, 1, T) int -> quantized (B, 768, T)
        cb = self.codebook[0]  # (K, C)
        return mx.transpose(cb[codes[:, 0]], (0, 2, 1))


class SynthesizerTrn(nn.Module):
    """v1 / v2 / v2Pro / v2ProPlus inference SynthesizerTrn."""

    def __init__(self, spec_channels, segment_size, inter_channels, hidden_channels,
                 filter_channels, n_heads, n_layers, kernel_size, p_dropout,
                 resblock, resblock_kernel_sizes, resblock_dilation_sizes,
                 upsample_rates, upsample_initial_channel, upsample_kernel_sizes,
                 n_speakers=0, gin_channels=0, use_sdp=True, semantic_frame_rate=None,
                 freeze_quantizer=None, version="v2", **kwargs):
        super().__init__()
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.version = version
        self._dec_fast = None  # compiled decoder closure (GSOVITS_HIFIGAN_FAST)
        self.upsample_rates = upsample_rates
        self.segment_size = segment_size
        self.gin_channels = gin_channels

        n_symbols = 322 if version == "v1" else 732
        self.enc_p = TextEncoder(inter_channels, hidden_channels, filter_channels, n_heads,
                                 n_layers, kernel_size, p_dropout, version=version,
                                 n_symbols=n_symbols)
        self.dec = Generator(inter_channels, resblock, resblock_kernel_sizes,
                             resblock_dilation_sizes, upsample_rates, upsample_initial_channel,
                             upsample_kernel_sizes, gin_channels=gin_channels)
        self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16,
                                      gin_channels=gin_channels)
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4,
                                          gin_channels=gin_channels)
        if version == "v1":
            self.ref_enc = MelStyleEncoder(spec_channels, style_vector_dim=gin_channels)
        else:
            self.ref_enc = MelStyleEncoder(704, style_vector_dim=gin_channels)

        ssl_dim = 768
        self.semantic_frame_rate = semantic_frame_rate
        if semantic_frame_rate == "25hz":
            self.ssl_proj = Conv1d(ssl_dim, ssl_dim, 2, stride=2)
        else:
            self.ssl_proj = Conv1d(ssl_dim, ssl_dim, 1, stride=1)

        self.is_v2pro = version in ("v2Pro", "v2ProPlus")
        if self.is_v2pro:
            self.sv_emb = ConvlessLinear(20480, gin_channels)
            self.ge_to512 = ConvlessLinear(gin_channels, 512)
            self.prelu_weight = mx.ones((1, gin_channels, 1))
        self.quantizer = ResidualVectorQuantizer(mx.zeros((1024, ssl_dim)))

    # -- reference embedding ------------------------------------------------
    def get_ge(self, refer: mx.array, sv_emb_raw: mx.array | None = None) -> mx.array:
        refer_mask = mx.ones((refer.shape[0], 1, refer.shape[2]), dtype=refer.dtype)
        if self.version == "v1":
            ge = self.ref_enc(refer * refer_mask, refer_mask)
        else:
            ge = self.ref_enc(refer[:, :704] * refer_mask, refer_mask)
        if self.is_v2pro:
            sv = self.sv_emb(sv_emb_raw)  # (B, gin)
            ge = ge + sv[:, :, None]
            ge = nn.prelu(ge, self.prelu_weight)
        return ge

    # -- main decode (used by pipeline) --------------------------------------
    def decode(self, codes: mx.array, text: mx.array, refer: mx.array,
               noise_scale: float = 0.5, speed: float = 1.0,
               sv_emb_raw: mx.array | None = None, key: mx.array | None = None):
        """codes: (B, 1, T) int32; text: (B, T_ph); refer: (B, C, T)."""
        ge = self.get_ge(refer, sv_emb_raw)
        ge512 = self.ge_to512(mx.transpose(ge, (0, 2, 1))).transpose(0, 2, 1) if self.is_v2pro else ge

        y_lengths = mx.array([codes.shape[2] * 2])
        text_lengths = mx.array([text.shape[-1]])

        quantized = self.quantizer.decode(codes)
        if self.semantic_frame_rate == "25hz":
            quantized = _nearest_interp(quantized, quantized.shape[-1] * 2)
        x, m_p, logs_p, y_mask = self.enc_p(quantized, y_lengths, text, text_lengths,
                                            ge512 if self.is_v2pro else ge, speed)
        z_p = m_p + mx.random.normal(m_p.shape, key=key) * mx.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=ge, reverse=True)
        zin = z * y_mask
        # GSOVITS_HIFIGAN_FAST=1 (task-8): cast the decoder input to the
        # weight dtype (fp16 exports) and use a compiled decoder closure.
        # The flow emits fp32; without the cast every conv re-promotes to
        # fp32 (measured: v2 decode 645ms compiled-fp16-in vs 681ms
        # fp32-in; fp16 w with fp32 in = 0.999x). Compiled warm is a
        # further ~5% over eager fp16; gates: corr 0.9995+ vs seed-0.
        if os.environ.get("GSOVITS_HIFIGAN_FAST") == "1":
            if self._dec_fast is None:
                self._dec_fast = mx.compile(self.dec)
            zin = zin.astype(self.dec.conv_pre.weight.dtype)
            gin_fast = ge.astype(self.dec.conv_pre.weight.dtype) if ge is not None else None
            o = self._dec_fast(zin, gin_fast)
        else:
            o = self.dec(zin, g=ge)
        return o, y_mask

    def extract_latent(self, x: mx.array) -> mx.array:
        ssl = self.ssl_proj(x)
        codes = self.quantizer.encode(ssl)
        return mx.transpose(codes, (0, 2, 1)) if codes.ndim == 3 and codes.shape[1] == 1 else codes


class ConvlessLinear(nn.Module):
    """nn.Linear equivalent kept simple (weight (out, in), bias)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = mx.zeros((out_features, in_features))
        self.bias = mx.zeros((out_features,)) if bias else None

    def __call__(self, x: mx.array) -> mx.array:
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out
