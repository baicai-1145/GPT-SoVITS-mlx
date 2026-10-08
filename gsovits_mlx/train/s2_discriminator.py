"""MultiPeriodDiscriminator (module/models.py) in pure MLX for s2 GAN training.

Structure (official, use_spectral_norm=False everywhere in GPT-SoVITS):
  discriminators[0] = DiscriminatorS: 1-D convs (15/41x4 groups) all weight_norm
  discriminators[1:] = DiscriminatorP(period): 2-D convs (k,1)/(stride,1)
      periods [2,3,5,7,11] (v1/v2) or [2,3,5,7,11,17,23] (v2Pro/Plus)

Weight norm is implemented as a live parameterization (NOT pre-fused like the
inference exports): each conv stores (weight_g, weight_v) with
w = g * v / ||v||_2  (norm over all dims except dim 0 = out-channel), exactly
torch.nn.utils.weight_norm(dim=0 default). Gradients flow into g and v; the
norm-computation happens inside the forward graph so mx.grad handles it.

Weight naming matches the official s2D checkpoints:
  discriminators.<i>.convs.<j>.weight_g / weight_v / bias
  discriminators.<i>.conv_post.weight_g / weight_v / bias
(DiscriminatorS convs are Conv1d: weight (out,in,k); DiscriminatorP convs are
Conv2d: weight (out,in,kh,kw) with kw=1.)

Losses (module/losses.py) are in gsovits_mlx.train.s2_gan; this module is
only the networks.
"""

from __future__ import annotations

import mlx.core as mx

from ..utils.layers import get_padding

LRELU_SLOPE = 0.1


# ---------------------------------------------------------------------------
# weight-norm convolution layers (live parameterization)
# ---------------------------------------------------------------------------

class WeightNormConv1d:
    """weight_norm(Conv1d(...)) equivalent.

    Params: weight_g (out,), weight_v (out, in, k), bias (out,).
    Effective weight = g * v / ||v||_per-out-channel (fp32 norms).
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int = 1,
                 padding: int = 0, dilation: int = 1, groups: int = 1):
        limit = 0.01  # torch weight_norm keeps the underlying init; small
        # values only matter pre-load; checkpoints overwrite g/v anyway
        self.weight_v = mx.random.normal((out_ch, kernel_size, in_ch // groups)) * limit
        self.weight_g = mx.ones((out_ch,))
        self.bias = mx.zeros((out_ch,))
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    @property
    def weight(self) -> mx.array:
        v32 = self.weight_v.astype(mx.float32)
        norm = mx.sqrt(mx.sum(v32 * v32, axis=(1, 2), keepdims=True))
        w = self.weight_g[:, None, None] * v32 / norm
        return w

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.transpose(x, (0, 2, 1))  # (B,C,T) -> (B,T,C)
        if self.padding:
            x = mx.pad(x, [(0, 0), (self.padding, self.padding), (0, 0)])
        w = self.weight.astype(x.dtype)
        out = mx.conv1d(x, w, stride=self.stride, padding=0,
                        dilation=self.dilation, groups=self.groups)
        out = mx.transpose(out, (0, 2, 1))
        return out + self.bias[None, :, None]


class WeightNormConv2d:
    """weight_norm(Conv2d(...)) equivalent; weight (out, in, kh, kw)."""

    def __init__(self, in_ch, out_ch, kernel, stride=(1, 1), padding=(0, 0)):
        limit = 0.01
        self.weight_v = mx.random.normal((out_ch, in_ch, kernel[0], kernel[1])) * limit
        self.weight_g = mx.ones((out_ch,))
        self.bias = mx.zeros((out_ch,))
        self.stride = stride
        self.padding = padding

    @property
    def weight(self) -> mx.array:
        v32 = self.weight_v.astype(mx.float32)
        norm = mx.sqrt(mx.sum(v32 * v32, axis=(1, 2, 3), keepdims=True))
        return self.weight_g[:, None, None, None] * v32 / norm

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, C, H, W) -> (B, H, W, C) channels-last for mx.conv2d
        x = mx.transpose(x, (0, 2, 3, 1))
        if any(self.padding):
            x = mx.pad(x, [(0, 0), (self.padding[0], self.padding[0]),
                           (self.padding[1], self.padding[1]), (0, 0)])
        w = self.weight.astype(x.dtype)
        out = mx.conv2d(x, w, stride=self.stride, padding=(0, 0))
        out = mx.transpose(out, (0, 3, 1, 2))
        return out + self.bias[None, :, None, None]


# ---------------------------------------------------------------------------
# discriminators
# ---------------------------------------------------------------------------

class DiscriminatorS:
    def __init__(self):
        self.convs = [
            WeightNormConv1d(1, 16, 15, 1, padding=7),
            WeightNormConv1d(16, 64, 41, 4, groups=4, padding=20),
            WeightNormConv1d(64, 256, 41, 4, groups=16, padding=20),
            WeightNormConv1d(256, 1024, 41, 4, groups=64, padding=20),
            WeightNormConv1d(1024, 1024, 41, 4, groups=256, padding=20),
            WeightNormConv1d(1024, 1024, 5, 1, padding=2),
        ]
        self.conv_post = WeightNormConv1d(1024, 1, 3, 1, padding=1)

    def __call__(self, x: mx.array):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = mx.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = mx.reshape(x, (x.shape[0], -1))
        return x, fmap


class DiscriminatorP:
    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3):
        self.period = period
        self.convs = [
            WeightNormConv2d(1, 32, (kernel_size, 1), (stride, 1),
                             (get_padding(kernel_size, 1), 0)),
            WeightNormConv2d(32, 128, (kernel_size, 1), (stride, 1),
                             (get_padding(kernel_size, 1), 0)),
            WeightNormConv2d(128, 512, (kernel_size, 1), (stride, 1),
                             (get_padding(kernel_size, 1), 0)),
            WeightNormConv2d(512, 1024, (kernel_size, 1), (stride, 1),
                             (get_padding(kernel_size, 1), 0)),
            WeightNormConv2d(1024, 1024, (kernel_size, 1), (1, 1),
                             (get_padding(kernel_size, 1), 0)),
        ]
        self.conv_post = WeightNormConv2d(1024, 1, (3, 1), (1, 1), (1, 0))

    def __call__(self, x: mx.array):
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = _reflect_pad_1d(x, n_pad)
            t = t + n_pad
        x = mx.reshape(x, (b, c, t // self.period, self.period))
        for l in self.convs:
            x = l(x)
            x = mx.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = mx.reshape(x, (x.shape[0], -1))
        return x, fmap


def _reflect_pad_1d(x: mx.array, n_pad: int) -> mx.array:
    """F.pad(x, (0, n_pad), 'reflect') on (B, C, T)."""
    if n_pad == 0:
        return x
    # reflect needs the pad width <= T-1; training segments are long, but be safe
    t = x.shape[-1]
    if n_pad > t - 1:
        # torch raises; replicate its behavior by repeated reflection
        parts = [x]
        rem = n_pad
        while rem > 0:
            take = min(rem, t - 1)
            src = parts[-1]
            parts.append(src[..., 2 * take - 1 : t - 1][..., ::-1] if False else
                         mx.take(src, mx.arange(t - 1, t - 1 - take, -1), axis=-1))
            rem -= take
        return mx.concatenate(parts, axis=-1)
    idx = mx.arange(t - 1, t - 1 - n_pad, -1)
    return mx.concatenate([x, mx.take(x, idx, axis=-1)], axis=-1)


class MultiPeriodDiscriminator:
    def __init__(self, periods=None):
        if periods is None:
            periods = [2, 3, 5, 7, 11]
        self.discriminators = [DiscriminatorS()] + \
            [DiscriminatorP(p) for p in periods]

    def __call__(self, y: mx.array, y_hat: mx.array):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

    # -- parameter plumbing ---------------------------------------------------
    def parameters(self) -> dict:
        out = {}
        for i, d in enumerate(self.discriminators):
            for j, conv in enumerate(d.convs):
                out[f"discriminators.{i}.convs.{j}.weight_g"] = conv.weight_g
                out[f"discriminators.{i}.convs.{j}.weight_v"] = conv.weight_v
                out[f"discriminators.{i}.convs.{j}.bias"] = conv.bias
            out[f"discriminators.{i}.conv_post.weight_g"] = d.conv_post.weight_g
            out[f"discriminators.{i}.conv_post.weight_v"] = d.conv_post.weight_v
            out[f"discriminators.{i}.conv_post.bias"] = d.conv_post.bias
        return out


def load_mpd_weights(mpd: MultiPeriodDiscriminator, arrays: dict) -> list:
    """Load official s2D checkpoint arrays (torch layout) into mpd.

    torch Conv1d weight (out,in,k) -> MLX (out,k,in); Conv2d unchanged layout.
    Returns the official-style missing/unexpected key report.
    """
    src = dict(arrays)
    expected = set(mpd.parameters().keys())
    loaded = []
    for name in expected:
        if name in src:
            a = src.pop(name)
            if a.ndim == 3 and ".convs." in name:  # Conv1d weight_g/v
                a = mx.transpose(a, (0, 2, 1))
            target = mpd.parameters()
            # assign into the right module attribute
            _assign(mpd, name, a)
            loaded.append(name)
    missing = sorted(expected - set(loaded))
    unexpected = sorted(src.keys())
    return missing, unexpected


def _assign(mpd: MultiPeriodDiscriminator, name: str, a: mx.array) -> None:
    parts = name.split(".")
    i, sub, j = int(parts[1]), parts[2], int(parts[4]) if parts[3] == "convs" else None
    d = mpd.discriminators[i]
    if parts[3] == "convs":
        conv = d.convs[j]
    else:
        conv = d.conv_post
    field = parts[-1]
    if field == "bias":
        conv.bias = a
    elif field == "weight_g":
        conv.weight_g = a.reshape(-1)
    else:
        conv.weight_v = a
