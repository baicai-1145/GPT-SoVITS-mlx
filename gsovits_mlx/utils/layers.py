"""Basic layers / helpers shared by all modules. Pure MLX.

Conventions:
- MLX conv1d/conv_transpose1d operate on channels-LAST input (N, L, C) with weights
  (out, k, in)/(out, k, in). Our layers accept (B, C, T) tensors (matching the torch
  reference) and internally transpose: x -> (B, T, C) for conv, then back.
- Linear weights use torch layout (out, in); matmul: x @ W.T on last-axis features.
- Weight-norm convs from checkpoints are pre-fused at conversion time.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# functional helpers
# ---------------------------------------------------------------------------

def sequence_mask(length: mx.array, max_length: int | None = None) -> mx.array:
    """length: int array [B] (or scalar tensor) -> float mask [B, 1, T]."""
    if max_length is None:
        max_length = int(length.max().item())
    ids = mx.arange(max_length)
    mask = (ids[None, :] < length[:, None]).astype(mx.float32)
    return mask[:, None, :]


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size - 1) * dilation // 2


def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def kl_divergence(m_p, logs_p, m_q, logs_q):
    """All (B, C, T); returns (B, T)."""
    m_q = mx.transpose(m_q, (0, 2, 1))
    logs_q = mx.transpose(logs_q, (0, 2, 1))
    m_p = mx.transpose(m_p, (0, 2, 1))
    logs_p = mx.transpose(logs_p, (0, 2, 1))
    return mx.sum(
        logs_q - logs_p - 0.5
        + 0.5 * (mx.exp(2 * logs_p) + (m_p - m_q) ** 2) * mx.exp(-2 * logs_q),
        axis=1,
    )


def fused_add_tanh_sigmoid_multiply(input_a: mx.array, input_b: mx.array, n_channels: int) -> mx.array:
    """torch commons.fused_add_tanh_sigmoid_multiply: SUM inputs, then tanh/sigmoid halves."""
    in_act = input_a + input_b
    t_act = mx.tanh(in_act[:, :n_channels, :])
    s_act = mx.sigmoid(in_act[:, n_channels:, :])
    return t_act * s_act


def convert_pad_shape(pad_shape) -> list[tuple[int, int]]:
    """torch-style per-axis pad widths, passed through in the SAME axis order.

    (torch's commons.convert_pad_shape reverses the list because F.pad wants
    reversed order; mx.pad wants natural order.)
    """
    return [(int(a), int(b)) for a, b in pad_shape]


def generate_path(duration: mx.array, mask: mx.array) -> mx.array:
    """duration: (B, 1, T_x); mask: (B, 1, T_y, T_x) -> path (B, 1, T_y, T_x)."""
    b, _, t_y, t_x = mask.shape
    cum_duration = mx.cumsum(duration, axis=-1)
    cum_duration_flat = mx.reshape(cum_duration, (b * t_y,))
    path = sequence_mask(cum_duration_flat, t_x)
    path = mx.reshape(path, (b, t_y, t_x))
    path = path.astype(mx.uint8)[:, None, :, :] * mask.astype(mx.uint8)
    path = path.astype(mask.dtype)
    # left-to-right exclusive cumsum along T_x
    return mx.cumsum(path, axis=-1) - path


# ---------------------------------------------------------------------------
# layers
# ---------------------------------------------------------------------------

class LayerNormChannels(nn.Module):
    """LayerNorm over channel dim for (B, C, T) I/O (module.modules.LayerNorm)."""

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = mx.ones((channels,))
        self.beta = mx.zeros((channels,))

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.transpose(x, (0, 2, 1))
        x = mx.fast.layer_norm(x, self.gamma, self.beta, self.eps)
        return mx.transpose(x, (0, 2, 1))


class LinearNorm(nn.Module):
    """modules.LinearNorm (nn.Linear wrapper, bias=True). Accepts (B, T, C)."""

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True, w_init_gain: str = "linear"):
        super().__init__()
        limit = 1.0 / math.sqrt(in_dim)
        self.weight = mx.random.uniform(-limit, limit, (out_dim, in_dim))
        self.bias = mx.zeros((out_dim,)) if bias else None

    def __call__(self, x: mx.array) -> mx.array:
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out


class Mish(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return x * mx.tanh(mx.log1p(mx.exp(mx.minimum(x, mx.array(30.0, x.dtype)))))


class Conv1d(nn.Module):
    """Conv1d, torch weight layout (out, in, k), applied to (B, C, T).

    MLX conv is channels-last, so we transpose input/output (zero-copy views in MLX).
    """

    def __init__(self, in_ch: int, out_ch: int, k: int, stride: int = 1, padding: int = 0,
                 dilation: int = 1, groups: int = 1, bias: bool = True):
        super().__init__()
        assert in_ch % groups == 0 and out_ch % groups == 0
        limit = 1.0 / math.sqrt(in_ch * k / groups)
        self.weight = mx.random.uniform(-limit, limit, (out_ch, k, in_ch // groups))
        self.bias = mx.zeros((out_ch,)) if bias else None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.transpose(x, (0, 2, 1))  # B C T -> B T C
        if self.padding:
            x = mx.pad(x, [(0, 0), (self.padding, self.padding), (0, 0)])
        out = mx.conv1d(x, self.weight, stride=self.stride, padding=0,
                        dilation=self.dilation, groups=self.groups)
        out = mx.transpose(out, (0, 2, 1))
        if self.bias is not None:
            out = out + self.bias[None, :, None]
        return out


class ConvTranspose1d(nn.Module):
    """ConvTranspose1d, torch weight layout (in, out, k), applied to (B, C, T)."""

    def __init__(self, in_ch: int, out_ch: int, k: int, stride: int = 1, padding: int = 0, bias: bool = True):
        super().__init__()
        limit = 1.0 / math.sqrt(in_ch * k)
        self.weight = mx.random.uniform(-limit, limit, (out_ch, k, in_ch))
        self.bias = mx.zeros((out_ch,)) if bias else None
        self.stride = stride
        self.padding = padding

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.transpose(x, (0, 2, 1))
        out = mx.conv_transpose1d(x, self.weight, stride=self.stride, padding=self.padding)
        out = mx.transpose(out, (0, 2, 1))
        if self.bias is not None:
            out = out + self.bias[None, :, None]
        return out


class Conv1dGLU(nn.Module):
    """modules.Conv1dGLU: conv1d + GLU + residual, I/O (B, C, T), channels-first.

    Checkpoint param path: conv1.conv.weight (2C, C, k) / conv1.conv.bias.
    Converted to MLX layout (2C, k, C).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float = 0.1):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.w_1 = mx.zeros((out_channels * 2, kernel_size, in_channels))
        self.b_1 = mx.zeros((out_channels * 2,))

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, C, T) -> channels-last for MLX conv
        residual = x
        xc = mx.transpose(x, (0, 2, 1))
        pad = self.kernel_size // 2
        xc = mx.pad(xc, [(0, 0), (pad, pad), (0, 0)])
        h = mx.conv1d(xc, self.w_1)  # stored MLX layout (2C, k, C)
        h = mx.transpose(h, (0, 2, 1)) + self.b_1[None, :, None]
        a, b = h[:, : self.out_channels], h[:, self.out_channels :]
        h = a * mx.sigmoid(b)
        return residual + h


class ResBlock1(nn.Module):
    """HiFi-GAN ResBlock1 (weight-norm convs; weights pre-fused at convert time)."""

    def __init__(self, channels: int, kernel_size: int = 3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = [Conv1d(channels, channels, kernel_size, 1,
                              dilation=d, padding=get_padding(kernel_size, d)) for d in dilation]
        self.convs2 = [Conv1d(channels, channels, kernel_size, 1,
                              dilation=1, padding=get_padding(kernel_size, 1)) for _ in dilation]

    def __call__(self, x: mx.array, x_mask=None) -> mx.array:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = nn.leaky_relu(x, 0.1)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c1(xt)
            xt = nn.leaky_relu(xt, 0.1)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c2(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x


class ResBlock2(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation=(1, 3)):
        super().__init__()
        self.convs = [Conv1d(channels, channels, kernel_size, 1,
                             dilation=d, padding=get_padding(kernel_size, d)) for d in dilation]

    def __call__(self, x: mx.array, x_mask=None) -> mx.array:
        for c in self.convs:
            xt = nn.leaky_relu(x, 0.1)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x


class WN(nn.Module):
    """modules.WN (non-conv, no gin) and mrte_model.WN share this layout when gin_channels=0.

    Supports gin conditioning via cond_layer (weight-norm 1x1 conv, pre-fused).
    """

    def __init__(self, hidden_channels: int, kernel_size: int, dilation_rate: int, n_layers: int,
                 gin_channels: int = 0, p_dropout: float = 0.0):
        super().__init__()
        assert kernel_size % 2 == 1
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        if gin_channels != 0:
            self.cond_layer = Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)
        self.in_layers = []
        self.res_skip_layers = []
        for i in range(n_layers):
            dilation = dilation_rate**i
            padding = int((kernel_size * dilation - dilation) / 2)
            self.in_layers.append(
                Conv1d(hidden_channels, 2 * hidden_channels, kernel_size, dilation=dilation, padding=padding))
            res_skip = 2 * hidden_channels if i < n_layers - 1 else hidden_channels
            self.res_skip_layers.append(Conv1d(hidden_channels, res_skip, 1))

    def __call__(self, x: mx.array, x_mask: mx.array, g: mx.array | None = None) -> mx.array:
        output = mx.zeros_like(x)
        if g is not None:
            g = self.cond_layer(g)
        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None:
                g_l = g[:, i * 2 * self.hidden_channels : (i + 1) * 2 * self.hidden_channels, :]
            else:
                g_l = mx.zeros_like(x_in)
            acts = fused_add_tanh_sigmoid_multiply(x_in, g_l, self.hidden_channels)
            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, : self.hidden_channels, :]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels :, :]
            else:
                output = output + res_skip_acts
        return output * x_mask
