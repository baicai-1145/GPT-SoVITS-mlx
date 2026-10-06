"""BigVGAN v2 (v3 vocoder) and HiFi-GAN Generator (v4/v5 vocoder). Pure MLX.

- Snake/SnakeBeta activations with alias-free Activation1d (kaiser window up/downsample).
- Weight-norm convs are pre-fused at conversion time (weight = weight_v * weight_g).
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from ..utils.layers import Conv1d, ConvTranspose1d, get_padding


# ---------------------------------------------------------------------------
# kaiser resample (alias_free_activation)
# ---------------------------------------------------------------------------

def kaiser_sinc_filter1d(cutoff: float, half_width: float, kernel_size: int) -> mx.array:
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2
    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.0:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.0:
        beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21.0)
    else:
        beta = 0.0
    # torch.kaiser_window(periodic=False) == numpy.kaiser
    window = mx.array(_np_kaiser(kernel_size, beta))
    if even:
        time = mx.arange(-half_size, half_size, dtype=mx.float32) + 0.5
    else:
        time = mx.arange(kernel_size, dtype=mx.float32) - half_size
    if cutoff == 0:
        filter_ = mx.zeros_like(time)
    else:
        filter_ = 2 * cutoff * window * _sinc(2 * cutoff * time)
        filter_ = filter_ / mx.sum(filter_)
    return filter_[None, None]


def _np_kaiser(n: int, beta: float) -> "list[float]":
    import numpy as np
    return np.kaiser(n, beta).tolist()


def _sinc(x: mx.array) -> mx.array:
    pix = math.pi * x
    return mx.where(mx.abs(x) < 1e-12, mx.array(1.0, x.dtype), mx.sin(pix) / mx.maximum(mx.abs(pix), 1e-12) * mx.sign(pix + 1e-30) * mx.sign(x + 1e-30))


def _sinc_safe(x: mx.array) -> mx.array:
    out = mx.sin(math.pi * x) / (math.pi * x)
    return mx.where(mx.abs(x) < 1e-6, mx.array(1.0, x.dtype), out)


class LowPassFilter1d(nn.Module):
    def __init__(self, cutoff=0.5, half_width=0.6, stride: int = 1, padding: bool = True,
                 padding_mode: str = "replicate", kernel_size: int = 12):
        super().__init__()
        self.even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.filter = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, C, T) -> MLX channels-last depthwise: weight (1, k, 1), groups=C
        _, C, _ = x.shape
        if self.padding:
            x = _pad_replicate(x, self.pad_left, self.pad_right)
        x = mx.transpose(x, (0, 2, 1))
        f = mx.broadcast_to(self.filter.transpose(0, 2, 1), (C, self.filter.shape[-1], 1))
        out = mx.conv1d(x, f, stride=self.stride, groups=C)
        return mx.transpose(out, (0, 2, 1))


def _pad_replicate(x: mx.array, left: int, right: int) -> mx.array:
    if left:
        x = mx.concatenate([mx.repeat(x[:, :, :1], left, axis=2), x], axis=2)
    if right:
        x = mx.concatenate([x, mx.repeat(x[:, :, -1:], right, axis=2)], axis=2)
    return x


class UpSample1d(nn.Module):
    def __init__(self, ratio: int = 2, kernel_size: int | None = None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        self.filter = kaiser_sinc_filter1d(0.5 / ratio, 0.6 / ratio, self.kernel_size)

    def __call__(self, x: mx.array) -> mx.array:
        _, C, _ = x.shape
        x = _pad_replicate(x, self.pad, self.pad)
        x = mx.transpose(x, (0, 2, 1))
        f = mx.broadcast_to(self.filter.transpose(0, 2, 1), (C, self.kernel_size, 1))
        x = self.ratio * mx.conv_transpose1d(x, f, stride=self.stride, groups=C)
        x = mx.transpose(x, (0, 2, 1))
        return x[:, :, self.pad_left : -self.pad_right]


class DownSample1d(nn.Module):
    def __init__(self, ratio: int = 2, kernel_size: int | None = None):
        super().__init__()
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.lowpass = LowPassFilter1d(cutoff=0.5 / ratio, half_width=0.6 / ratio,
                                       stride=ratio, kernel_size=self.kernel_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.lowpass(x)


class Activation1d(nn.Module):
    def __init__(self, activation, up_ratio: int = 2, down_ratio: int = 2,
                 up_kernel_size: int = 12, down_kernel_size: int = 12):
        super().__init__()
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.upsample(x)
        x = self.act(x)
        x = self.downsample(x)
        return x


# ---------------------------------------------------------------------------
# Snake activations
# ---------------------------------------------------------------------------

class Snake(nn.Module):
    def __init__(self, in_features: int, alpha: float = 1.0, alpha_logscale: bool = False):
        super().__init__()
        self.alpha_logscale = alpha_logscale
        self.alpha = (mx.zeros((in_features,)) if alpha_logscale else mx.ones((in_features,))) * alpha

    def __call__(self, x: mx.array) -> mx.array:
        alpha = self.alpha[None, :, None]
        if self.alpha_logscale:
            alpha = mx.exp(alpha)
        a = mx.sin(x * alpha)
        return x + a * a / (alpha + 1e-9)


class SnakeBeta(nn.Module):
    def __init__(self, in_features: int, alpha: float = 1.0, alpha_logscale: bool = False):
        super().__init__()
        self.alpha_logscale = alpha_logscale
        init = mx.zeros((in_features,)) if alpha_logscale else mx.ones((in_features,))
        self.alpha = init * alpha
        self.beta = init * alpha

    def __call__(self, x: mx.array) -> mx.array:
        alpha, beta = self.alpha[None, :, None], self.beta[None, :, None]
        if self.alpha_logscale:
            alpha, beta = mx.exp(alpha), mx.exp(beta)
        a = mx.sin(x * alpha)
        return x + a * a / (beta + 1e-9)


# ---------------------------------------------------------------------------
# BigVGAN
# ---------------------------------------------------------------------------

class AMPBlock1(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation=(1, 3, 5),
                 activation: str = "snakebeta", snake_logscale: bool = True):
        super().__init__()
        self.convs1 = [Conv1d(channels, channels, kernel_size, 1, dilation=d,
                              padding=get_padding(kernel_size, d)) for d in dilation]
        self.convs2 = [Conv1d(channels, channels, kernel_size, 1, dilation=1,
                              padding=get_padding(kernel_size, 1)) for _ in dilation]
        num_layers = len(self.convs1) + len(self.convs2)
        act_cls = Snake if activation == "snake" else SnakeBeta
        self.activations = [
            Activation1d(act_cls(channels, alpha_logscale=snake_logscale)) for _ in range(num_layers)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x
        return x


class BigVGAN(nn.Module):
    def __init__(self, h: dict):
        super().__init__()
        self.num_kernels = len(h["resblock_kernel_sizes"])
        self.num_upsamples = len(h["upsample_rates"])
        self.use_bias_at_final = h.get("use_bias_at_final", True)
        self.use_tanh_at_final = h.get("use_tanh_at_final", True)

        self.conv_pre = Conv1d(h["num_mels"], h["upsample_initial_channel"], 7, 1, padding=3)

        self.ups = []
        for i, (u, k) in enumerate(zip(h["upsample_rates"], h["upsample_kernel_sizes"])):
            self.ups.append(ConvTranspose1d(h["upsample_initial_channel"] // (2**i),
                                            h["upsample_initial_channel"] // (2 ** (i + 1)),
                                            k, u, padding=(k - u) // 2))
        self.resblocks = []
        for i in range(len(self.ups)):
            ch = h["upsample_initial_channel"] // (2 ** (i + 1))
            for k, d in zip(h["resblock_kernel_sizes"], h["resblock_dilation_sizes"]):
                self.resblocks.append(AMPBlock1(ch, k, d, activation=h["activation"],
                                                snake_logscale=h["snake_logscale"]))
        ch = h["upsample_initial_channel"] // (2 ** len(self.ups))
        act = SnakeBeta(ch, alpha_logscale=h["snake_logscale"]) if h["activation"] == "snakebeta" \
            else Snake(ch, alpha_logscale=h["snake_logscale"])
        self.activation_post = Activation1d(act)
        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=self.use_bias_at_final)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = self.ups[i](x)
            xs = 0
            for j in range(self.num_kernels):
                xs = xs + self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = self.activation_post(x)
        x = self.conv_post(x)
        return mx.tanh(x) if self.use_tanh_at_final else mx.clip(x, -1.0, 1.0)


# ---------------------------------------------------------------------------
# HiFi-GAN Generator (v4/v5 vocoder)
# ---------------------------------------------------------------------------

class GeneratorVocoder(nn.Module):
    """module.models.Generator used as v4/v5 vocoder (no gin)."""

    def __init__(self, initial_channel=100, resblock="1", resblock_kernel_sizes=(3, 7, 11),
                 resblock_dilation_sizes=((1, 3, 5),) * 3, upsample_rates=(10, 6, 2, 2, 2),
                 upsample_initial_channel=512, upsample_kernel_sizes=(20, 12, 4, 4, 4),
                 gin_channels=0, is_bias=True):
        super().__init__()
        from ..utils.layers import ResBlock1, ResBlock2
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        block_cls = ResBlock1 if resblock == "1" else ResBlock2
        self.ups = []
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(ConvTranspose1d(upsample_initial_channel // (2**i),
                                            upsample_initial_channel // (2 ** (i + 1)),
                                            k, u, padding=(k - u) // 2))
        self.resblocks = []
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(block_cls(ch, k, d))
        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=is_bias)

    def __call__(self, x: mx.array, g=None) -> mx.array:
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = nn.leaky_relu(x, 0.1)
            x = self.ups[i](x)
            xs = 0
            for j in range(self.num_kernels):
                xs = xs + self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = nn.leaky_relu(x)
        x = self.conv_post(x)
        return mx.tanh(x)
