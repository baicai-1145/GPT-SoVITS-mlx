"""MRTE (multi-reference timbre encoder) + MelStyleEncoder. Pure MLX."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from ..utils.layers import Conv1d, Conv1dGLU, LayerNormChannels, LinearNorm, Mish
from .attentions import MultiHeadAttention


class MRTE(nn.Module):
    def __init__(self, content_enc_channels=192, hidden_size=512, out_channels=192,
                 kernel_size=5, n_heads=4, ge_layer=2):
        super().__init__()
        self.cross_attention = MultiHeadAttention(hidden_size, hidden_size, n_heads)
        self.c_pre = Conv1d(content_enc_channels, hidden_size, 1)
        self.text_pre = Conv1d(content_enc_channels, hidden_size, 1)
        self.c_post = Conv1d(hidden_size, out_channels, 1)

    def __call__(self, ssl_enc, ssl_mask, text, text_mask, ge):
        attn_mask = text_mask[:, :, None, :] * ssl_mask[:, :, :, None]
        ssl_enc = self.c_pre(ssl_enc * ssl_mask)
        text_enc = self.text_pre(text * text_mask)
        x = self.cross_attention(ssl_enc * ssl_mask, text_enc * text_mask, attn_mask) + ssl_enc + ge
        return self.c_post(x * ssl_mask)


class MelStyleEncoder(nn.Module):
    """modules.MelStyleEncoder.

    torch uses Sequential(LinearNorm, Mish, Dropout, LinearNorm, Mish, Dropout) named
    `spectral` -> weights spectral.0.fc / spectral.3.fc; here we expose nested modules
    with identical parameter names via a small shim (converted checkpoints are remapped,
    so we simply use two LinearNorm layers named via dict for weight loading).
    """

    def __init__(self, n_mel_channels=80, style_hidden=128, style_vector_dim=256,
                 style_kernel_size=5, style_head=2, dropout=0.1):
        super().__init__()
        self.in_dim = n_mel_channels
        self.hidden_dim = style_hidden
        self.out_dim = style_vector_dim
        self.n_head = style_head

        self.spectral_0 = LinearNorm(n_mel_channels, style_hidden)
        self.spectral_1 = LinearNorm(style_hidden, style_hidden)
        self.mish = Mish()

        self.temporal_0 = Conv1dGLU(style_hidden, style_hidden, style_kernel_size, dropout)
        self.temporal_1 = Conv1dGLU(style_hidden, style_hidden, style_kernel_size, dropout)

        self.slf_attn = ScaledMultiHeadAttention(
            style_head, style_hidden, style_hidden // style_head,
            style_hidden // style_head, dropout=0.0,
        )
        self.fc = LinearNorm(style_hidden, style_vector_dim)

    def _temporal_avg_pool(self, x: mx.array, mask: mx.array | None) -> mx.array:
        # x: (B, T, C); mask: (B, 1, T) with 1 = masked(pad)
        if mask is None:
            return mx.mean(x, axis=1)
        keep = 1.0 - mask[:, 0, :]  # (B, T) 1 = keep
        x = x * keep[:, :, None]
        len_ = mx.maximum(mx.sum(keep, axis=1, keepdims=True), 1.0)
        return mx.sum(x, axis=1) / len_  # len_ already (B,1) broadcastable over (B,C)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        # x: (B, C, T); mask: (B, 1, T) as produced by sequence_mask (1 = keep)
        x = mx.transpose(x, (0, 2, 1))  # (B, T, C)
        pad = None
        if mask is not None:
            pad = ((mask == 0)[:, 0, :]).astype(x.dtype)  # (B, T) 1 = pad

        x = self.mish(self.spectral_0(x))
        x = self.mish(self.spectral_1(x))

        x = mx.transpose(x, (0, 2, 1))  # (B, C, T)
        x = self.temporal_0(x)
        x = self.temporal_1(x)
        x = mx.transpose(x, (0, 2, 1))  # (B, T, C)

        if mask is not None:
            x = x * (1.0 - pad[:, :, None].astype(x.dtype))
        x = self.slf_attn(x, pad)
        x = self.fc(x)
        w = self._temporal_avg_pool(x, pad[:, None, :])  # pool expects (B,1,T); 1 = pad
        return w[:, :, None]  # (B, C, 1)


class ScaledMultiHeadAttention(nn.Module):
    """modules.ScaledDotProductAttention + MultiHeadAttention wrapper (MelStyleEncoder style).

    Parameter names w_qs/w_ks/w_vs/fc match the checkpoint.
    NOTE on scaling: torch's ScaledDotProductAttention here divides by
    sqrt(d_model) (temperature=np.power(d_model, 0.5) in MultiHeadAttention),
    NOT the usual sqrt(d_k). Verified against torch on the v1/v2 refs.
    """

    def __init__(self, n_head: int, d_model: int, d_k: int, d_v: int, dropout: float = 0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = LinearNorm(d_model, n_head * d_k)
        self.w_ks = LinearNorm(d_model, n_head * d_k)
        self.w_vs = LinearNorm(d_model, n_head * d_v)
        self.fc = LinearNorm(n_head * d_v, d_model)
        self.scale = 1.0 / math.sqrt(d_model)  # official temperature = sqrt(d_model)
        self.d_model = d_model

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        # x: (B, T, C); mask: (B, T) float/bool with 1 = pad (key-side masking)
        b, t, _ = x.shape
        q = mx.reshape(self.w_qs(x), (b, t, self.n_head, self.d_k))
        k = mx.reshape(self.w_ks(x), (b, t, self.n_head, self.d_k))
        v = mx.reshape(self.w_vs(x), (b, t, self.n_head, self.d_v))
        q, k, v = (mx.transpose(a, (0, 2, 1, 3)) for a in (q, k, v))

        scores = (q * self.scale) @ mx.swapaxes(k, -1, -2)  # (b, h, t, t)
        if mask is not None:
            neg = mx.array(-1e4, scores.dtype)
            scores = mx.where(mask[:, None, None, :] > 0, neg, scores)
        attn = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        out = attn @ v  # (b, h, t, dv)
        out = mx.reshape(mx.transpose(out, (0, 2, 1, 3)), (b, t, self.n_head * self.d_v))
        out = self.fc(out)
        return x + out  # reference adds residual after fc
