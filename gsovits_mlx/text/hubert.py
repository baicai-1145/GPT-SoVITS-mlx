"""MLX Chinese-HuBERT (chinese-hubert-base) for GPT-SoVITS semantic tokens.

HF HubertModel, group-norm conv feature extractor (7 convs), no stable layer norm,
pos_conv with fused weight-norm (pre-fused at conversion).

Audio pipeline: 16 kHz mono float -> conv feature extractor (320x downsample) ->
projection -> + pos conv emb -> 12 transformer layers -> last_hidden_state (768).
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


def _gelu(x: mx.array) -> mx.array:
    return nn.gelu(x)


class GroupNorm1d:
    """torch GroupNorm over channels; input (B, T, C)."""

    def __init__(self, num_groups: int, dim: int, eps: float = 1e-5):
        self.groups = num_groups
        self.dim = dim
        self.eps = eps
        self.weight = mx.ones((dim,))
        self.bias = mx.zeros((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        b, t, c = x.shape
        g = self.groups
        xr = x.reshape(b, t, g, c // g)
        mean = xr.mean(axis=(1, 3), keepdims=True)
        var = xr.var(axis=(1, 3), keepdims=True)
        xr = (xr - mean) / mx.sqrt(var + self.eps)
        xr = xr.reshape(b, t, c)
        return xr * self.weight + self.bias


class HubertLayer:
    def __init__(self, arrays, i: int, dim: int, heads: int, inter: int, eps: float):
        g = arrays.get
        p = f"encoder.layers.{i}."
        self.eps = eps
        self.heads = heads
        self.head_dim = dim // heads
        self.q_w, self.q_b = g(p + "attention.q_proj.weight"), g(p + "attention.q_proj.bias")
        self.k_w, self.k_b = g(p + "attention.k_proj.weight"), g(p + "attention.k_proj.bias")
        self.v_w, self.v_b = g(p + "attention.v_proj.weight"), g(p + "attention.v_proj.bias")
        self.o_w, self.o_b = g(p + "attention.out_proj.weight"), g(p + "attention.out_proj.bias")
        # HF wav2vec2/Hubert layer: attn WITHOUT inner norm; post-attn LayerNorm; ff; post-ff LayerNorm
        self.a_ln_w, self.a_ln_b = g(p + "layer_norm.weight"), g(p + "layer_norm.bias")
        self.fc1_w, self.fc1_b = g(p + "feed_forward.intermediate_dense.weight"), g(p + "feed_forward.intermediate_dense.bias")
        self.fc2_w, self.fc2_b = g(p + "feed_forward.output_dense.weight"), g(p + "feed_forward.output_dense.bias")
        self.f_ln_w, self.f_ln_b = g(p + "final_layer_norm.weight"), g(p + "final_layer_norm.bias")

    def _attn(self, x: mx.array, mask) -> mx.array:
        b, t, d = x.shape
        q = (x @ self.q_w.T + self.q_b).reshape(b, t, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        k = (x @ self.k_w.T + self.k_b).reshape(b, t, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        v = (x @ self.v_w.T + self.v_b).reshape(b, t, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(self.head_dim),
                                                 mask=mask)
        return o.transpose(0, 2, 1, 3).reshape(b, t, d) @ self.o_w.T + self.o_b

    def __call__(self, x: mx.array, mask) -> mx.array:
        x = x + self._attn(x, mask)          # attention on raw x (no pre-norm)
        x = mx.fast.layer_norm(x, self.a_ln_w, self.a_ln_b, self.eps)
        h = _gelu(x @ self.fc1_w.T + self.fc1_b) @ self.fc2_w.T + self.fc2_b
        x = mx.fast.layer_norm(x + h, self.f_ln_w, self.f_ln_b, self.eps)
        return x


class HubertModel:
    def __init__(self, arrays: dict, config: dict):
        g = arrays.get
        self.dim = config["hidden_size"]
        self.eps = config.get("layer_norm_eps", 1e-5)
        # conv feature extractor: group norm per conv (feat_extract_norm=group), bias=False
        conv_dims = config["conv_dim"]
        conv_kernels = config["conv_kernel"]
        conv_strides = config["conv_stride"]
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.convs = []
        self.convs_ln = []
        c_in = 1
        for i, (d, k, s) in enumerate(zip(conv_dims, conv_kernels, conv_strides)):
            w = g(f"feature_extractor.conv_layers.{i}.conv.weight")
            self.convs.append((w, s))  # weight (d, k, c_in) MLX layout
            if i == 0:
                ln_w = g(f"feature_extractor.conv_layers.{i}.layer_norm.weight")
                ln_b = g(f"feature_extractor.conv_layers.{i}.layer_norm.bias")
                self.convs_ln.append((ln_w, ln_b))
            else:
                self.convs_ln.append(None)
            c_in = d
        self.feat_ln_w = g("feature_projection.layer_norm.weight")
        self.feat_ln_b = g("feature_projection.layer_norm.bias")
        self.proj_w = g("feature_projection.projection.weight")
        self.proj_b = g("feature_projection.projection.bias")
        # pos conv: torch grouped (768, 48, 128) groups=16 -> MLX (768, 128, 48); pad 63 both sides, stride 1
        pw = g("encoder.pos_conv_embed.conv.weight")
        self.pos_weight = mx.transpose(pw, (0, 2, 1)) if pw.shape[1] != pw.shape[2] else pw
        self.pos_groups = 16
        self.pos_bias = g("encoder.pos_conv_embed.conv.bias")
        self.enc_ln_w = g("encoder.layer_norm.weight")
        self.enc_ln_b = g("encoder.layer_norm.bias")
        self.layers = []
        n = config["num_hidden_layers"]
        for i in range(n):
            self.layers.append(HubertLayer(arrays, i, self.dim,
                                           config["num_attention_heads"],
                                           config["intermediate_size"], self.eps))
        self.mask_emb = arrays.get("masked_spec_embed")

    def _conv_front(self, wav: mx.array) -> mx.array:
        # wav: (B, T) -> (B, T', C)
        x = wav[:, :, None]  # (B, T, 1)
        for i, ((w, s), ln) in enumerate(zip(self.convs, self.convs_ln)):
            x = mx.conv1d(x, w, stride=s)  # HF hubert: valid conv, no padding
            if ln is not None:
                x = GroupNormFx(ln, x)
            x = nn.gelu(x)
        return x

    def __call__(self, wav: mx.array, attention_mask=None) -> mx.array:
        """wav (B, T) float16/32 -> last_hidden_state (B, T', 768)."""
        x = self._conv_front(wav)
        x = mx.fast.layer_norm(x, self.feat_ln_w, self.feat_ln_b, self.eps)
        x = x @ self.proj_w.T + self.proj_b
        # pos conv
        xp = mx.pad(x, [(0, 0), (64, 64), (0, 0)])
        pe = mx.conv1d(xp, self.pos_weight, stride=1, groups=self.pos_groups)
        pe = pe[:, :-1, :]  # HubertSamePadLayer trims last frame (even kernel)
        pe = pe + self.pos_bias[None, None, :]
        x = x + _gelu(pe)
        x = mx.fast.layer_norm(x, self.enc_ln_w, self.enc_ln_b, self.eps)
        mask = None
        if attention_mask is not None:
            mask = (1.0 - attention_mask[:, None, None, :].astype(mx.float32)) * -1e9
        for layer in self.layers:
            x = layer(x, mask)
        return x


def GroupNormFx(ln_params, x: mx.array) -> mx.array:
    """First conv layer group norm (groups=512, one channel each) on (B, T, C)."""
    w, b = ln_params
    mean = x.mean(axis=1, keepdims=True)
    var = x.var(axis=1, keepdims=True)
    return (x - mean) / mx.sqrt(var + 1e-5) * w + b
