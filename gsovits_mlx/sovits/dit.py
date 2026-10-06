"""DiT backbone (f5_tts) + CFM / CFMV5 flow matching. Pure MLX."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


def sequence_mask_2d(lengths, max_length) -> mx.array:
    ids = mx.arange(max_length)
    return (ids[None, :] < lengths[:, None]).astype(mx.float32)  # (B, T)


class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def __call__(self, x: mx.array, scale: float = 1000.0) -> mx.array:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = mx.exp(mx.arange(half_dim, dtype=mx.float32) * -emb)
        emb = scale * x[:, None] * emb[None, :]
        return mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)


class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int, freq_embed_dim: int = 256):
        super().__init__()
        self.freq_embed_dim = freq_embed_dim
        self.time_mlp_0_w = mx.zeros((dim, freq_embed_dim))
        self.time_mlp_0_b = mx.zeros((dim,))
        self.time_mlp_2_w = mx.zeros((dim, dim))
        self.time_mlp_2_b = mx.zeros((dim,))

    def __call__(self, timestep: mx.array) -> mx.array:
        half = self.freq_embed_dim // 2
        emb = math.log(10000) / (half - 1)
        emb = mx.exp(mx.arange(half, dtype=mx.float32) * -emb)
        emb = timestep[:, None].astype(mx.float32) * emb[None, :] * 1000.0
        emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)
        x = nn.silu(emb @ self.time_mlp_0_w.T + self.time_mlp_0_b)
        return x @ self.time_mlp_2_w.T + self.time_mlp_2_b


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0,
                         theta_rescale_factor: float = 1.0) -> mx.array:
    theta *= theta_rescale_factor ** (dim / (dim - 2))
    freqs = 1.0 / (theta ** (mx.arange(0, dim, 2)[: dim // 2].astype(mx.float32) / dim))
    t = mx.arange(end, dtype=mx.float32)
    freqs = t[:, None] * freqs[None, :]
    return mx.concatenate([mx.cos(freqs), mx.sin(freqs)], axis=-1)  # (end, dim)


def get_pos_embed_indices(start: int, length: int, max_pos: int, scale: float = 1.0) -> mx.array:
    pos = (start + mx.arange(length, dtype=mx.float32) * scale).astype(mx.int32)
    return mx.minimum(pos, max_pos - 1)


class ConvNeXtV2Block(nn.Module):
    """f5_tts ConvNeXtV2Block: dwconv -> LN(affine) -> pwconv1 -> GELU -> GRN -> pwconv2 + residual.

    I/O (B, T, C) channels-last. Checkpoint names map:
      dwconv.weight (C,1,k)->dw, norm.{weight,bias}, pwconv1/pwconv2, grn.{gamma,beta}.
    """

    def __init__(self, dim: int, intermediate_dim: int, kernel_size: int = 7):
        super().__init__()
        self.kernel_size = kernel_size
        self.dw = mx.zeros((dim, kernel_size, 1))
        self.dw_b = mx.zeros((dim,))
        self.norm_w = mx.ones((dim,))
        self.norm_b = mx.zeros((dim,))
        self.pwconv1_w = mx.zeros((intermediate_dim, dim))
        self.pwconv1_b = mx.zeros((intermediate_dim,))
        self.pwconv2_w = mx.zeros((dim, intermediate_dim))
        self.pwconv2_b = mx.zeros((dim,))
        self.grn_beta = mx.zeros((intermediate_dim,))
        self.grn_gamma = mx.zeros((intermediate_dim,))

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        # depthwise conv over time
        pad = self.kernel_size // 2
        xc = mx.pad(x, [(0, 0), (pad, pad), (0, 0)])
        xc = mx.conv1d(xc, self.dw, groups=x.shape[-1]) + self.dw_b[None, None, :]
        x = mx.fast.layer_norm(xc, self.norm_w, self.norm_b, 1e-6)
        x = nn.gelu(x @ self.pwconv1_w.T + self.pwconv1_b)
        # GRN (dim=intermediate): norm across time per channel
        g = mx.sqrt(mx.sum(x**2, axis=1, keepdims=True))
        nx = g / (mx.mean(g, axis=-1, keepdims=True) + 1e-6)
        x = self.grn_gamma[None, None, :] * (x * nx) + self.grn_beta[None, None, :] + x
        x = x @ self.pwconv2_w.T + self.pwconv2_b
        return residual + x


def rope_rotate_half(x: mx.array) -> mx.array:
    """x-transformers rotate_half: freqs are packed as INTERLEAVED pairs
    (stack((f, f), -1).reshape), so 'pairs' are adjacent elements
    (x[2i], x[2i+1]); rotate_half swaps them -> (-x2, x1) interleaved.
    """
    x1, x2 = x[..., ::2], x[..., 1::2]
    rot = mx.concatenate([-x2[..., None], x1[..., None]], axis=-1)
    return mx.reshape(rot, x.shape)


def apply_rope(t: mx.array, freqs: mx.array) -> mx.array:
    """t: (B, H, T, D); freqs: (T, D/2) raw angles -> rotated.

    torch packs freqs interleaved ([f0, f0, f1, f1, ...]); we repeat cos/sin
    to the same layout instead of materialising the interleaved angles.
    """
    cosv = mx.repeat(mx.cos(freqs), 2, axis=-1)
    sinv = mx.repeat(mx.sin(freqs), 2, axis=-1)
    return t * cosv[None, None] + rope_rotate_half(t) * sinv[None, None]


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        inner = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.to_q_w = mx.zeros((inner, dim))
        self.to_q_b = mx.zeros((inner,))
        self.to_k_w = mx.zeros((inner, dim))
        self.to_k_b = mx.zeros((inner,))
        self.to_v_w = mx.zeros((inner, dim))
        self.to_v_b = mx.zeros((inner,))
        self.to_out_0_w = mx.zeros((dim, inner))
        self.to_out_0_b = mx.zeros((dim,))

    def __call__(self, x: mx.array, mask: mx.array | None, rope: mx.array) -> mx.array:
        b, t, _ = x.shape
        q = x @ self.to_q_w.T + self.to_q_b
        k = x @ self.to_k_w.T + self.to_k_b
        v = x @ self.to_v_w.T + self.to_v_b
        # torch applies apply_rotary_pos_emb BEFORE the head split on the (B, T, inner)
        # tensor; rot_dim = freqs.shape[-1] = dim_head, so only the FIRST dim_head
        # channels (= head 0 after the split) get rotated; heads 1..N-1 are unrotated.
        # Replicate exactly: rotate the head-0 slice only.
        q_h = q.reshape(b, t, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        k_h = k.reshape(b, t, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        v = v.reshape(b, t, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        q_h = mx.concatenate([apply_rope(q_h[:, :1], rope), q_h[:, 1:]], axis=1)
        k_h = mx.concatenate([apply_rope(k_h[:, :1], rope), k_h[:, 1:]], axis=1)

        if mask is not None:
            attn_mask = mask[:, None, None, :].astype(mx.float32)  # (B,1,1,T)
            neg = mx.array(-1e9, mx.float32)
        out = mx.fast.scaled_dot_product_attention(q_h, k_h, v, scale=1.0 / math.sqrt(self.dim_head),
                                                   mask=mask[:, None, None, :] if mask is not None else None)
        out = out.transpose(0, 2, 1, 3).reshape(b, t, self.heads * self.dim_head)
        out = out @ self.to_out_0_w.T + self.to_out_0_b
        if mask is not None:
            out = out * mask[:, :, None]
        return out


class FeedForward(nn.Module):
    def __init__(self, dim: int, dim_out=None, mult: int = 4, approximate: str = "tanh"):
        super().__init__()
        inner = int(dim * mult)
        dim_out = dim_out or dim
        self.ff_0_0_w = mx.zeros((inner, dim))
        self.ff_0_0_b = mx.zeros((inner,))
        self.ff_2_w = mx.zeros((dim_out, inner))
        self.ff_2_b = mx.zeros((dim_out,))

    def __call__(self, x: mx.array) -> mx.array:
        h = x @ self.ff_0_0_w.T + self.ff_0_0_b
        h = nn.gelu_approx(h)
        return h @ self.ff_2_w.T + self.ff_2_b


class AdaLayerNormZero(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear_w = mx.zeros((dim * 6, dim))
        self.linear_b = mx.zeros((dim * 6,))

    def __call__(self, x: mx.array, emb: mx.array):
        # x: (B, T, D); emb: (B, D)
        emb = nn.silu(emb) @ self.linear_w.T + self.linear_b
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = [emb[:, i*x.shape[-1]:(i+1)*x.shape[-1]] for i in range(6)]
        norm = _ln_noaffine(x)
        x = norm * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormZero_Final(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear_w = mx.zeros((dim * 2, dim))
        self.linear_b = mx.zeros((dim * 2,))

    def __call__(self, x: mx.array, emb: mx.array) -> mx.array:
        emb = nn.silu(emb) @ self.linear_w.T + self.linear_b
        scale, shift = emb[:, : x.shape[-1]], emb[:, x.shape[-1]:]
        return _ln_noaffine(x) * (1 + scale[:, None]) + shift[:, None]


def _ln_noaffine(x: mx.array) -> mx.array:
    mu = mx.mean(x, axis=-1, keepdims=True)
    var = mx.mean((x - mu) ** 2, axis=-1, keepdims=True)
    return (x - mu) / mx.sqrt(var + 1e-6)


class DiTBlock(nn.Module):
    """f5_tts DiTBlock: AdaLayerNormZero pre-norm attention + AdaLayerNorm pre-norm FFN.

    ff_norm here is a plain AdaLayerNorm (scale/shift from t)."""

    def __init__(self, dim: int, heads: int, dim_head: int, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn_norm = AdaLayerNormZero(dim)
        self.attn = Attention(dim, heads, dim_head, dropout)
        self.ff = FeedForward(dim=dim, mult=ff_mult, approximate="tanh")

    def __call__(self, x: mx.array, t: mx.array, mask: mx.array | None, rope: mx.array) -> mx.array:
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, t)
        attn_output = self.attn(norm, mask, rope)
        x = x + gate_msa[:, None] * attn_output
        norm = _ln_noaffine(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm)
        x = x + gate_mlp[:, None] * ff_output
        return x


class TextEmbedding(nn.Module):
    """conv_layers=4 ConvNeXt text conditioner (plus rope-free sinus pos add)."""

    def __init__(self, text_dim: int, conv_layers: int = 0, conv_mult: int = 2):
        super().__init__()
        self.conv_layers = conv_layers
        if conv_layers > 0:
            self.precompute_max_pos = 4096
            # torch freqs_cis is a non-persistent buffer: rebuild at init (fp32)
            self._freqs = precompute_freqs_cis(text_dim, self.precompute_max_pos)
            self.text_blocks = [ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)]

    def __call__(self, text: mx.array, seq_len: int, drop_text: bool = False) -> mx.array:
        if drop_text:
            text = mx.zeros_like(text)
        if self.conv_layers > 0:
            idx = get_pos_embed_indices(0, seq_len, self.precompute_max_pos)
            text = text + self._freqs[idx][None]
            for blk in self.text_blocks:
                text = blk(text)
        return text


class InputEmbedding(nn.Module):
    """proj + grouped-conv (groups=16) positional embedding with Mish."""

    def __init__(self, mel_dim: int, text_dim: int, out_dim: int):
        super().__init__()
        self.proj_w = mx.zeros((out_dim, mel_dim * 2 + text_dim))
        self.proj_b = mx.zeros((out_dim,))
        # torch conv1d weight (1024, 64, 31) groups=16 -> MLX (1024, 31, 64)
        self.conv_pos_0_w = mx.zeros((out_dim, 31, out_dim // 16))
        self.conv_pos_0_b = mx.zeros((out_dim,))
        self.conv_pos_1_w = mx.zeros((out_dim, 31, out_dim // 16))
        self.conv_pos_1_b = mx.zeros((out_dim,))
        self.groups = 16

    def _conv_pos(self, x: mx.array) -> mx.array:
        # x: (B, T, C); MLX conv1d is channels-last: out (B, T', C), bias over C
        xc = mx.pad(x, [(0, 0), (15, 15), (0, 0)])
        y = mx.conv1d(xc, self.conv_pos_0_w, groups=self.groups) + self.conv_pos_0_b[None, None, :]
        y = nn_mish(y)
        yc = mx.pad(y, [(0, 0), (15, 15), (0, 0)])
        y = mx.conv1d(yc, self.conv_pos_1_w, groups=self.groups) + self.conv_pos_1_b[None, None, :]
        return nn_mish(y)

    def __call__(self, x: mx.array, cond: mx.array, text_embed: mx.array,
                 drop_audio_cond: bool = False) -> mx.array:
        if drop_audio_cond:
            cond = mx.zeros_like(cond)
        x = mx.concatenate([x, cond, text_embed], axis=-1) @ self.proj_w.T + self.proj_b
        return self._conv_pos(x) + x


def nn_mish(x):
    return x * mx.tanh(mx.log1p(mx.exp(mx.minimum(x, mx.array(30.0, x.dtype)))))


class DiT(nn.Module):
    def __init__(self, *, dim, depth=8, heads=8, dim_head=64, dropout=0.1, ff_mult=4,
                 mel_dim=100, text_dim=None, conv_layers=0, long_skip_connection=False,
                 use_step_embedding=True):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.use_step_embedding = use_step_embedding
        text_dim = text_dim if text_dim is not None else mel_dim

        self.time_embed = TimestepEmbedding(dim)
        if use_step_embedding:
            self.d_embed = TimestepEmbedding(dim)
        self.text_embed = TextEmbedding(text_dim, conv_layers=conv_layers)
        self.input_embed = InputEmbedding(mel_dim, text_dim, dim)
        self.heads = heads
        self.dim_head = dim_head

        self.transformer_blocks = [
            DiTBlock(dim, heads, dim_head, ff_mult=ff_mult, dropout=dropout) for _ in range(depth)
        ]
        self.long_skip_connection = long_skip_connection
        if long_skip_connection:
            self.long_skip_w = mx.zeros((dim, dim * 2))

        self.norm_out = AdaLayerNormZero_Final(dim)
        self.proj_out_w = mx.zeros((mel_dim, dim))
        self.proj_out_b = mx.zeros((mel_dim,))

    def _rope(self, seq_len: int, dtype) -> mx.array:
        # cached per seq_len; RAW angles (T, dim_head/2) — apply_rope derives
        # interleaved cos/sin. (torch RotaryEmbedding: inv_freq = theta**-arange(0,d,2)/d,
        # freqs = t[:, None] * inv_freq[None, :], then stack((f,f),-1).reshape.)
        cache = getattr(self, "_rope_cache", None)
        if cache is None or cache.shape[0] < seq_len or cache.dtype != dtype:
            half = self.dim_head // 2
            inv_freq = 1.0 / (10000.0 ** (mx.arange(0, half, dtype=mx.float32) / half))
            t = mx.arange(max(seq_len, 4096), dtype=mx.float32)
            cache = (t[:, None] * inv_freq[None, :]).astype(dtype)
            self._rope_cache = cache
        return cache[:seq_len]

    def prepare_static_cache(self, cond0: mx.array, x_lens, text0: mx.array):
        """Precompute text embedding, static input projection and rope (v5 path)."""
        text = mx.transpose(text0, (0, 2, 1))
        cond = mx.transpose(cond0, (0, 2, 1))
        seq_len = cond.shape[1]
        text_embed = self.text_embed(text, seq_len, drop_text=False)
        mel_dim = cond.shape[-1]
        static = (mx.concatenate([cond, text_embed], axis=-1) @ self.input_embed.proj_w[:, mel_dim:].T
                  + self.input_embed.proj_b)
        negative_static = (mx.concatenate([mx.zeros_like(cond), text_embed], axis=-1)
                           @ self.input_embed.proj_w[:, mel_dim:].T + self.input_embed.proj_b)
        mask = sequence_mask_2d(x_lens, seq_len)
        return {"condition": static, "negative_condition": negative_static, "mask": mask,
                "rope": self._rope(seq_len, static.dtype)}

    def __call__(self, x0, cond0, x_lens, time, dt_base_bootstrap, text0,
                 use_grad_ckpt=False, drop_audio_cond=False, drop_text=False,
                 infer=False, text_cache=None, dt_cache=None, static_cache=None):
        x = mx.transpose(x0, (0, 2, 1))
        cond = mx.transpose(cond0, (0, 2, 1))
        text = mx.transpose(text0, (0, 2, 1))
        if static_cache is not None:
            mask = static_cache["mask"]
        else:
            mask = sequence_mask_2d(x_lens, x.shape[1])

        batch, seq_len = x.shape[0], x.shape[1]
        # torch: t = time_embed(time); dt = d_embed(dt_base) (first call) or dt_cache;
        #        t += dt; returns dt (the d_embed OUTPUT, not time+d) as the cache value.
        t = self.time_embed(time.astype(mx.float32))
        dt_out = None
        if self.use_step_embedding:
            if infer and dt_cache is not None:
                t = t + dt_cache
            else:
                dt_out = self.d_embed(dt_base_bootstrap.astype(mx.float32))
                t = t + dt_out

        if static_cache is not None:
            static = static_cache["negative_condition"] if drop_audio_cond else static_cache["condition"]
            x = x @ self.input_embed.proj_w[:, : x.shape[-1]].T + static
            x = self.input_embed._conv_pos(x) + x
        else:
            text_embed = text_cache if (infer and text_cache is not None) else self.text_embed(
                text, seq_len, drop_text=drop_text)
            x = self.input_embed(x, cond, text_embed, drop_audio_cond=drop_audio_cond)

        rope = static_cache["rope"] if static_cache is not None else self._rope(seq_len, x.dtype)

        residual = x
        for block in self.transformer_blocks:
            x = block(x, t, mask, rope)
        if self.long_skip_connection:
            x = mx.concatenate([x, residual], axis=-1) @ self.long_skip_w.T

        x = self.norm_out(x, t)
        out = x @ self.proj_out_w.T + self.proj_out_b
        return (out, text_embed if static_cache is None else None, dt_out) if infer else out


def load_dit_params(dit: DiT, arrays: dict, depth: int, text_blocks: int, has_d_embed: bool):
    """Map converted safetensors arrays onto the DiT structures."""
    ge = arrays.get
    dit.time_embed.time_mlp_0_w = ge("dit.time_embed.0.w")
    dit.time_embed.time_mlp_0_b = ge("dit.time_embed.0.b")
    dit.time_embed.time_mlp_2_w = ge("dit.time_embed.2.w")
    dit.time_embed.time_mlp_2_b = ge("dit.time_embed.2.b")
    if has_d_embed:
        dit.d_embed.time_mlp_0_w = ge("dit.d_embed.0.w")
        dit.d_embed.time_mlp_0_b = ge("dit.d_embed.0.b")
        dit.d_embed.time_mlp_2_w = ge("dit.d_embed.2.w")
        dit.d_embed.time_mlp_2_b = ge("dit.d_embed.2.b")
    for i in range(text_blocks):
        blk = dit.text_embed.text_blocks[i]
        blk.dw = ge(f"dit.text.{i}.dw")
        blk.dw_b = ge(f"dit.text.{i}.dw_b")
        blk.norm_w = ge(f"dit.text.{i}.norm_w")
        blk.norm_b = ge(f"dit.text.{i}.norm_b")
        blk.pwconv1_w = ge(f"dit.text.{i}.pw1.w")
        blk.pwconv1_b = ge(f"dit.text.{i}.pw1.b")
        blk.pwconv2_w = ge(f"dit.text.{i}.pw2.w")
        blk.pwconv2_b = ge(f"dit.text.{i}.pw2.b")
        blk.grn_gamma = ge(f"dit.text.{i}.grn.gamma")
        blk.grn_beta = ge(f"dit.text.{i}.grn.beta")
    ie = dit.input_embed
    ie.proj_w = ge("dit.in.proj.w")
    ie.proj_b = ge("dit.in.proj.b")
    ie.conv_pos_0_w = ge("dit.in.conv_pos_0.w")
    ie.conv_pos_0_b = ge("dit.in.conv_pos_0.b")
    ie.conv_pos_1_w = ge("dit.in.conv_pos_1.w")
    ie.conv_pos_1_b = ge("dit.in.conv_pos_1.b")
    for i in range(depth):
        blk = dit.transformer_blocks[i]
        blk.attn_norm.linear_w = ge(f"dit.blocks.{i}.attn_norm.w")
        blk.attn_norm.linear_b = ge(f"dit.blocks.{i}.attn_norm.b")
        blk.attn.to_q_w = ge(f"dit.blocks.{i}.attn.to_q.w")
        blk.attn.to_q_b = ge(f"dit.blocks.{i}.attn.to_q.b")
        blk.attn.to_k_w = ge(f"dit.blocks.{i}.attn.to_k.w")
        blk.attn.to_k_b = ge(f"dit.blocks.{i}.attn.to_k.b")
        blk.attn.to_v_w = ge(f"dit.blocks.{i}.attn.to_v.w")
        blk.attn.to_v_b = ge(f"dit.blocks.{i}.attn.to_v.b")
        blk.attn.to_out_0_w = ge(f"dit.blocks.{i}.attn.to_out.w")
        blk.attn.to_out_0_b = ge(f"dit.blocks.{i}.attn.to_out.b")
        blk.ff.ff_0_0_w = ge(f"dit.blocks.{i}.ff.0.w")
        blk.ff.ff_0_0_b = ge(f"dit.blocks.{i}.ff.0.b")
        blk.ff.ff_2_w = ge(f"dit.blocks.{i}.ff.2.w")
        blk.ff.ff_2_b = ge(f"dit.blocks.{i}.ff.2.b")
    if "dit.long_skip" in arrays:
        dit.long_skip_w = ge("dit.long_skip")
    dit.norm_out.linear_w = ge("dit.norm_out.w")
    dit.norm_out.linear_b = ge("dit.norm_out.b")
    dit.proj_out_w = ge("dit.proj_out.w")
    dit.proj_out_b = ge("dit.proj_out.b")
