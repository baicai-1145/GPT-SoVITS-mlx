"""ERes2NetV2 speaker-verification encoder (v2Pro/v2ProPlus sv embedding).

Reference: GPT_SoVITS/eres2net/ERes2NetV2.py (3D-Speaker). Only forward3 is needed at
inference: (B, T, 80) kaldi-fbank -> (B, 2048) mean over time of flattened (C,T) fusion map.

Checkpoint: pretrained_eres2netv2w24s4ep4.ckpt, baseWidth=24 scale=4 expansion=4.
BatchNorm in eval mode folds to per-channel scale/shift (pre-computed at conversion).
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


def _relu(x: mx.array) -> mx.array:
    return nn.relu(mx.minimum(mx.maximum(x, 0.0), 20.0))  # Hardtanh(0, 20) == ReLU here


class BN2d:
    """Eval-mode batch norm over (B, C, H, W)."""

    def __init__(self, num_features: int):
        self.weight = mx.ones((num_features,))
        self.bias = mx.zeros((num_features,))
        self.running_mean = mx.zeros((num_features,))
        self.running_var = mx.ones((num_features,))
        self.eps = 1e-5

    def __call__(self, x: mx.array) -> mx.array:
        # input NHWC (B, H, W, C): channel is LAST
        shape = (1, 1, 1, -1)
        return (x - self.running_mean.reshape(shape)) / mx.sqrt(
            self.running_var.reshape(shape) + self.eps) * self.weight.reshape(shape) + self.bias.reshape(shape)


class Conv2d:
    """torch layout weight (out, in, kh, kw) -> MLX (out, kh, kw, in); input NHWC.

    The sv ckpt convs carry biases (official fusion.py AFF uses bias=True); loaded
    from safetensors by load_sv_encoder."""

    def __init__(self, in_c: int, out_c: int, k: int, stride: int = 1, padding: int = 0):
        self.weight = mx.zeros((out_c, k, k, in_c))
        self.bias = None  # set by loader when the ckpt conv carries one
        self.stride = stride
        self.padding = padding

    def __call__(self, x: mx.array) -> mx.array:
        if self.padding:
            x = mx.pad(x, [(0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)])
        y = mx.conv2d(x, self.weight, stride=[self.stride, self.stride])
        if self.bias is None:
            return y
        return y + self.bias.reshape(1, 1, 1, -1)


class AFF:
    """Attentional feature fusion."""

    def __init__(self, channels: int, r: int = 4):
        inter = channels // r
        self.c0 = Conv2d(channels * 2, inter, 1)
        self.bn0 = BN2d(inter)
        self.c1 = Conv2d(inter, channels, 1)
        self.bn1 = BN2d(channels)

    def __call__(self, x: mx.array, ds_y: mx.array) -> mx.array:
        xa = mx.concatenate([x, ds_y], axis=-1)  # channel-last fusion of (B,H,W,2C)
        att = self.bn1(self.c1(nn.silu(self.bn0(self.c0(xa)))))
        att = 1.0 + mx.tanh(att)
        return x * att + ds_y * (2.0 - att)


class BasicResBlock:
    """BasicBlockERes2NetV2 (sequential split-add)."""

    def __init__(self, in_planes: int, planes: int, stride: int = 1,
                 base_width: int = 26, scale: int = 2, expansion: int = 2):
        width = int(math.floor(planes * (base_width / 64.0)))
        self.width = width
        self.scale = scale
        self.expansion = expansion
        self.conv1 = Conv2d(in_planes, width * scale, 1, stride=stride)
        self.bn1 = BN2d(width * scale)
        self.convs = [Conv2d(width, width, 3, 1, 1) for _ in range(scale)]
        self.bns = [BN2d(width) for _ in range(scale)]
        self.conv3 = Conv2d(width * scale, planes * expansion, 1)
        self.bn3 = BN2d(planes * expansion)
        if stride != 1 or in_planes != expansion * planes:
            self.sc_conv = Conv2d(in_planes, expansion * planes, 1, stride=stride)
            self.sc_bn = BN2d(expansion * planes)
        else:
            self.sc_conv = None

    def __call__(self, x: mx.array) -> mx.array:
        # x: NHWC (B, H, W, C)
        out = _relu(self.bn1(self.conv1(x)))
        # split channels -> move to channel dim for the sequential pattern
        outs = []
        sp = out[..., : self.width]
        sp = _relu(self.bns[0](self.convs[0](sp)))
        acc = sp
        outs.append(acc)
        for i in range(1, self.scale):
            sp = out[..., i * self.width:(i + 1) * self.width]
            sp = sp + acc  # sequential: sp_i = sp_{i-1} + split_i
            sp = _relu(self.bns[i](self.convs[i](sp)))
            acc = sp
            outs.append(acc)
        out = mx.concatenate(outs, axis=-1)
        out = self.bn3(self.conv3(out))
        res = x if self.sc_conv is None else self.sc_bn(self.sc_conv(x))
        return _relu(out + res)


class BasicAFFBlock:
    """BasicBlockERes2NetV2AFF (AFF fusion between splits)."""

    def __init__(self, in_planes: int, planes: int, stride: int = 1,
                 base_width: int = 26, scale: int = 2, expansion: int = 2):
        width = int(math.floor(planes * (base_width / 64.0)))
        self.width = width
        self.scale = scale
        self.expansion = expansion
        self.conv1 = Conv2d(in_planes, width * scale, 1, stride=stride)
        self.bn1 = BN2d(width * scale)
        self.convs = [Conv2d(width, width, 3, 1, 1) for _ in range(scale)]
        self.bns = [BN2d(width) for _ in range(scale)]
        self.fuses = [AFF(width, 4) for _ in range(scale - 1)]
        self.conv3 = Conv2d(width * scale, planes * expansion, 1)
        self.bn3 = BN2d(planes * expansion)
        if stride != 1 or in_planes != expansion * planes:
            self.sc_conv = Conv2d(in_planes, expansion * planes, 1, stride=stride)
            self.sc_bn = BN2d(expansion * planes)
        else:
            self.sc_conv = None

    def __call__(self, x: mx.array) -> mx.array:
        out = _relu(self.bn1(self.conv1(x)))
        acc = _relu(self.bns[0](self.convs[0](out[..., : self.width])))
        outs = [acc]
        for i in range(1, self.scale):
            sp = self.fuses[i - 1](acc, out[..., i * self.width:(i + 1) * self.width])
            sp = _relu(self.bns[i](self.convs[i](sp)))
            acc = sp
            outs.append(acc)
        out = mx.concatenate(outs, axis=-1)
        out = self.bn3(self.conv3(out))
        res = x if self.sc_conv is None else self.sc_bn(self.sc_conv(x))
        return _relu(out + res)


class ERes2NetV2:
    def __init__(self, base_width: int = 24, scale: int = 4, expansion: int = 4,
                 num_blocks=(3, 4, 6, 3), m_channels: int = 64):
        self.expansion = expansion
        self.conv1 = Conv2d(1, m_channels, 3, 1, 1)
        self.bn1 = BN2d(m_channels)
        in_p = m_channels
        self.layer1 = self._make(BasicResBlock, in_p, m_channels, num_blocks[0], 1,
                                 base_width, scale, expansion)
        in_p = m_channels * expansion
        self.layer2 = self._make(BasicResBlock, in_p, m_channels * 2, num_blocks[1], 2,
                                 base_width, scale, expansion)
        in_p = m_channels * 2 * expansion
        self.layer3 = self._make(BasicAFFBlock, in_p, m_channels * 4, num_blocks[2], 2,
                                 base_width, scale, expansion)
        in_p = m_channels * 4 * expansion
        self.layer4 = self._make(BasicAFFBlock, in_p, m_channels * 8, num_blocks[3], 2,
                                 base_width, scale, expansion)
        self.layer3_ds = Conv2d(m_channels * 4 * expansion, m_channels * 8 * expansion, 3, 2, 1)
        self.fuse34 = AFF(m_channels * 8 * expansion, 4)

    @staticmethod
    def _make(cls, in_planes: int, planes: int, n: int, stride: int,
              bw: int, scale: int, expansion: int):
        layers = []
        strides = [stride] + [1] * (n - 1)
        cur_in = in_planes
        for s in strides:
            layers.append(cls(cur_in, planes, s, base_width=bw, scale=scale, expansion=expansion))
            cur_in = planes * expansion
        return layers

    def forward3(self, x: mx.array) -> mx.array:
        # x: (B, T, 80) kaldi fbank. torch does (B,T,F)->(B,F,T)->(B,1,F,T) NCHW;
        # MLX NHWC equivalent is (B, F, T, C=1).
        xc = mx.transpose(x, (0, 2, 1))[:, :, :, None]  # (B, 80, T, 1)
        out = _relu(self.bn1(self.conv1(xc)))
        for blk in self.layer1:
            out = blk(out)
        for blk in self.layer2:
            out = blk(out)
        out3 = out
        for blk in self.layer3:
            out3 = blk(out3)
        out4 = out3
        for blk in self.layer4:
            out4 = blk(out4)
        out3_ds = self.layer3_ds(out3)
        fused = self.fuse34(out4, out3_ds)  # NHWC (B, H, W, C)
        # torch: (B,C,H,W).flatten(1,2).mean(-1) -> (B, C*H); NHWC: transpose to BCHW then reshape
        b = mx.transpose(fused, (0, 3, 1, 2))  # (B, C, H, W)
        return mx.reshape(b, (b.shape[0], -1, b.shape[3])).mean(axis=2)  # (B, C*H)
