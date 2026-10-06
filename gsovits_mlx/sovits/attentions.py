"""Attention building blocks for SoVITS TextEncoder (attentions.Encoder path).

Pure MLX. The isflow/TransformerCouplingLayer variant is not used by any GPT-SoVITS
checkpoint (v1/v2 flow uses modules.WN); omitted intentionally.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from ..utils.layers import Conv1d, LayerNormChannels, convert_pad_shape, fused_add_tanh_sigmoid_multiply


class MultiHeadAttention(nn.Module):
    """module.attentions.MultiHeadAttention (1x1-conv projections + relative positional emb)."""

    def __init__(self, channels: int, out_channels: int, n_heads: int,
                 p_dropout: float = 0.0, window_size: int | None = None):
        super().__init__()
        assert channels % n_heads == 0
        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.k_channels = channels // n_heads
        self.window_size = window_size
        self.scale = 1.0 / math.sqrt(self.k_channels)

        self.conv_q = Conv1d(channels, channels, 1)
        self.conv_k = Conv1d(channels, channels, 1)
        self.conv_v = Conv1d(channels, channels, 1)
        self.conv_o = Conv1d(channels, out_channels, 1)
        if window_size is not None:
            rel_stddev = self.k_channels**-0.5
            self.emb_rel_k = mx.random.normal((1, window_size * 2 + 1, self.k_channels)) * rel_stddev
            self.emb_rel_v = mx.random.normal((1, window_size * 2 + 1, self.k_channels)) * rel_stddev

    def __call__(self, x: mx.array, c: mx.array, attn_mask: mx.array | None = None) -> mx.array:
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)
        out = self._attention(q, k, v, attn_mask)
        return self.conv_o(out)

    def _attention(self, query, key, value, mask=None):
        b, d, t_t = query.shape
        t_s = key.shape[2]

        # (b, c, t) -> (b, h, t, dk)
        query = mx.transpose(mx.reshape(query, (b, self.n_heads, self.k_channels, t_t)), (0, 1, 3, 2))
        key = mx.transpose(mx.reshape(key, (b, self.n_heads, self.k_channels, t_s)), (0, 1, 3, 2))
        value = mx.transpose(mx.reshape(value, (b, self.n_heads, self.k_channels, t_s)), (0, 1, 3, 2))

        scores = (query * self.scale) @ mx.swapaxes(key, -1, -2)
        if self.window_size is not None:
            assert t_s == t_t, "relative attention is self-attention only"
            key_rel = self._get_relative_embeddings(self.emb_rel_k, t_s)[0]   # (2l-1, dk)
            rel_logits = (query * self.scale) @ mx.swapaxes(key_rel, -1, -2)  # (b,h,l,2l-1)
            scores = scores + self._relative_to_absolute(rel_logits)
        if mask is not None:
            scores = mx.where(mask == 0, mx.array(-1e4, scores.dtype), scores)
        p_attn = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        output = p_attn @ value
        if self.window_size is not None:
            relative_weights = self._absolute_to_relative(p_attn)             # (b,h,l,2l-1)
            value_rel = self._get_relative_embeddings(self.emb_rel_v, t_s)[0]  # (2l-1, dk)
            output = output + (relative_weights @ value_rel)                  # (b,h,l,dk)
        output = mx.reshape(mx.transpose(output, (0, 1, 3, 2)), (b, d, t_t))
        return output

    def _get_relative_embeddings(self, relative_embeddings: mx.array, length: int) -> mx.array:
        max_relative_position = 2 * self.window_size + 1
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start = max((self.window_size + 1) - length, 0)
        slice_end = slice_start + 2 * length - 1
        if pad_length > 0:
            relative_embeddings = mx.pad(relative_embeddings, convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]))
        return relative_embeddings[:, slice_start:slice_end]

    def _relative_to_absolute(self, x: mx.array) -> mx.array:
        batch, heads, length, _ = x.shape
        x = mx.pad(x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, 1]]))
        x_flat = mx.reshape(x, [batch, heads, length * 2 * length])
        x_flat = mx.pad(x_flat, convert_pad_shape([[0, 0], [0, 0], [0, length - 1]]))
        return mx.reshape(x_flat, [batch, heads, length + 1, 2 * length - 1])[:, :, :length, length - 1:]

    def _absolute_to_relative(self, x: mx.array) -> mx.array:
        batch, heads, length, _ = x.shape
        x = mx.pad(x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length - 1]]))
        x_flat = mx.reshape(x, [batch, heads, length**2 + length * (length - 1)])
        x_flat = mx.pad(x_flat, convert_pad_shape([[0, 0], [0, 0], [length, 0]]))
        return mx.reshape(x_flat, [batch, heads, length, 2 * length])[:, :, :, 1:]


class FFN(nn.Module):
    """module.attentions.FFN."""

    def __init__(self, in_channels: int, out_channels: int, filter_channels: int,
                 kernel_size: int, p_dropout: float = 0.0, activation: str | None = None,
                 causal: bool = False):
        super().__init__()
        self.kernel_size = kernel_size
        self.causal = causal
        self.activation = activation
        self.conv_1 = Conv1d(in_channels, filter_channels, kernel_size)
        self.conv_2 = Conv1d(filter_channels, out_channels, kernel_size)

    def _pad(self, x: mx.array) -> mx.array:
        if self.kernel_size == 1:
            return x
        if self.causal:
            return mx.pad(x, [(0, 0), (0, 0), (self.kernel_size - 1, 0)])
        pad_l = (self.kernel_size - 1) // 2
        pad_r = self.kernel_size // 2
        return mx.pad(x, [(0, 0), (0, 0), (pad_l, pad_r)])

    def __call__(self, x: mx.array, x_mask: mx.array) -> mx.array:
        x = self.conv_1(self._pad(x * x_mask))
        if self.activation == "gelu":
            x = x * mx.sigmoid(1.702 * x)
        else:
            x = nn_relu(x)
        x = self.conv_2(self._pad(x * x_mask))
        return x * x_mask


def nn_relu(x):
    return mx.maximum(x, 0)


class Encoder(nn.Module):
    """attentions.Encoder: self-attention (window_size=4) + FFN stack, no gin."""

    def __init__(self, hidden_channels: int, filter_channels: int, n_heads: int, n_layers: int,
                 kernel_size: int = 1, p_dropout: float = 0.0, window_size: int = 4,
                 isflow: bool = False, **kwargs):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.window_size = window_size
        self.attn_layers = [
            MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout,
                               window_size=window_size)
            for _ in range(n_layers)
        ]
        self.norm_layers_1 = [LayerNormChannels(hidden_channels) for _ in range(n_layers)]
        self.ffn_layers = [
            FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout)
            for _ in range(n_layers)
        ]
        self.norm_layers_2 = [LayerNormChannels(hidden_channels) for _ in range(n_layers)]

    def __call__(self, x: mx.array, x_mask: mx.array, g: mx.array | None = None) -> mx.array:
        attn_mask = x_mask[:, :, None, :] * x_mask[:, :, :, None]
        x = x * x_mask
        for i in range(self.n_layers):
            y = self.attn_layers[i](x, x, attn_mask)
            x = self.norm_layers_1[i](x + y)
            y = self.ffn_layers[i](x, x_mask)
            x = self.norm_layers_2[i](x + y)
        return x * x_mask
