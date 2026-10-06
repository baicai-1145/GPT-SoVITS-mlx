"""AR GPT t2s transformer (s1v3.ckpt / s1bert25hz ckpts). Pure MLX, batch=1.

Checkpoint architecture (AR/models/t2s_model.py Text2SemanticDecoder, norm_first=False):
- POST-norm transformer: x = norm1(x + attn); x = norm2(x + mlp(relu))
- plain LayerNorm (no adaptive conditioning)
- packed in_proj qkv, learned sine positional embeddings with learned alpha scale
- inference: full-sequence recompute per step is wasteful; we use KV cache with the
  post-norm property that each position's output depends only on itself + past (valid).

Fallback: if a checkpoint carries norm1.project_layer weights (AdaptiveLayerNorm,
norm_first=True variant), loading raises and the adaptive path is used instead.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import mlx.nn as nn

NEG_INF = -1e9


class SinePositionalEmbedding:
    """AR/modules/embedding.py SinePositionalEmbedding (scale=False, alpha=True)."""

    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.x_scale = 1.0
        self.alpha = mx.ones((1,))
        position = mx.arange(0, 4096, dtype=mx.float32)[:, None]
        div_term = mx.exp(mx.arange(0, embedding_dim, 2, dtype=mx.float32)
                          * -(math.log(10000.0) / embedding_dim))
        pe = mx.zeros((4096, embedding_dim))
        pe[:, 0::2] = mx.sin(position * div_term)
        pe[:, 1::2] = mx.cos(position * div_term)
        self.pe = pe[None]

    def full(self, x: mx.array) -> mx.array:
        return x * self.x_scale + self.alpha * self.pe[:, : x.shape[1]].astype(x.dtype)

    def step(self, emb: mx.array, pos: int) -> mx.array:
        return emb * self.x_scale + self.alpha * self.pe[:, pos : pos + 1].astype(emb.dtype)


class T2SBlock:
    def __init__(self, num_heads: int, hidden_dim: int, adaptive: bool = False):
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        self.norm_eps = 1e-5
        self.adaptive = adaptive

    def weights(self, p: dict, i: int):
        g = p.__getitem__
        pre = f"block.{i}."
        self.qkv_w = g(pre + "qkv_w")
        self.qkv_b = g(pre + "qkv_b")
        self.out_w = g(pre + "out_w")
        self.out_b = g(pre + "out_b")
        self.linear1_w = g(pre + "mlp1.w")
        self.linear1_b = g(pre + "mlp1.b")
        self.linear2_w = g(pre + "mlp2.w")
        self.linear2_b = g(pre + "mlp2.b")
        if pre + "norm1.w" in p:
            self.adaptive = True
            self.norm1_w = g(pre + "norm1.w")
            self.norm1_b = g(pre + "norm1.b")
            self.norm2_w = g(pre + "norm2.w")
            self.norm2_b = g(pre + "norm2.b")
        else:
            self.norm1_g = g(pre + "norm1.g")
            self.norm1_b = g(pre + "norm1.b")
            self.norm2_g = g(pre + "norm2.g")
            self.norm2_b = g(pre + "norm2.b")
        return self

    def _ln(self, x: mx.array, which: int, emb: mx.array | None) -> mx.array:
        if self.adaptive:
            w_b = emb @ (self.norm1_w if which == 1 else self.norm2_w).T
            w_b = w_b + (self.norm1_b if which == 1 else self.norm2_b)
            d = x.shape[-1]
            weight, bias = w_b[..., :d], w_b[..., d:]
            return weight * mx.fast.layer_norm(x, None, None, self.norm_eps) + bias
        if which == 1:
            return mx.fast.layer_norm(x, self.norm1_g, self.norm1_b, self.norm_eps)
        return mx.fast.layer_norm(x, self.norm2_g, self.norm2_b, self.norm_eps)

    def _mlp(self, x: mx.array) -> mx.array:
        h = x @ self.linear1_w.T + self.linear1_b
        h = nn.relu(h)
        return h @ self.linear2_w.T + self.linear2_b

    def _attn(self, x: mx.array, k_cache: mx.array | None, v_cache: mx.array | None,
              mask: mx.array | None):
        b, t, d = x.shape
        qkv = x @ self.qkv_w.T + self.qkv_b
        q, k, v = qkv[..., :d], qkv[..., d : 2 * d], qkv[..., 2 * d :]
        if k_cache is not None:
            k_cache = mx.concatenate([k_cache, k], axis=1)
            v_cache = mx.concatenate([v_cache, v], axis=1)
        else:
            k_cache, v_cache = k, v

        def split_heads(t4: mx.array) -> mx.array:
            return t4.reshape(b, t4.shape[1], self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        qh = split_heads(q)
        kh = split_heads(k_cache)
        vh = split_heads(v_cache)
        out = mx.fast.scaled_dot_product_attention(
            qh, kh, vh, scale=1.0 / math.sqrt(self.head_dim),
            mask=mask if mask is not None else None)
        out = out.transpose(0, 2, 1, 3).reshape(b, t, d)
        out = out @ self.out_w.T + self.out_b
        return out, k_cache, v_cache

    def run(self, x: mx.array, emb_cond: mx.array | None, mask: mx.array | None,
            k_cache, v_cache, prefill: bool):
        if prefill:
            attn, k_out, v_out = self._attn(x, None, None, mask)
            k_cache = k_out  # (b, t, d) for this block
            v_cache = v_out
        else:
            attn, k_cache, v_cache = self._attn(x, k_cache, v_cache, None)
        # post-norm
        x = self._ln(x + attn, 1, emb_cond)
        x = self._ln(x + self._mlp(x), 2, emb_cond)
        return x, k_cache, v_cache


class Text2SemanticDecoder:
    def __init__(self, config: dict):
        m = config["model"]
        self.model_dim = m["hidden_dim"]
        self.embedding_dim = m["embedding_dim"]
        self.num_head = m["head"]
        self.num_layers = m["n_layer"]
        self.vocab_size = m["vocab_size"]
        self.phoneme_vocab_size = m["phoneme_vocab_size"]
        self.EOS = m["EOS"]
        self.max_sec = config["data"]["max_sec"]
        self.early_stop_num = 50 * self.max_sec
        self.blocks = [T2SBlock(self.num_head, self.model_dim) for _ in range(self.num_layers)]
        self.ar_text_position = SinePositionalEmbedding(self.embedding_dim)
        self.ar_audio_position = SinePositionalEmbedding(self.embedding_dim)

    def load(self, params: dict):
        self.params = params
        for i, block in enumerate(self.blocks):
            block.weights(params, i)
        self.ar_text_embedding = params["ar_text_embedding"]
        self.ar_audio_embedding = params["ar_audio_embedding"]
        self.bert_proj_w = params["bert_proj.weight"]
        self.bert_proj_b = params["bert_proj.bias"]
        self.ar_text_position.alpha = params["ar_text_position.alpha"]
        self.ar_audio_position.alpha = params["ar_audio_position.alpha"]
        self.ar_predict_layer_w = params["ar_predict_layer.weight"]
        return self

    def _text_embed(self, phones: mx.array, bert_feature: mx.array) -> mx.array:
        x = self.ar_text_embedding[phones]
        x = x + (mx.transpose(bert_feature, (0, 2, 1)) @ self.bert_proj_w.T + self.bert_proj_b)
        return self.ar_text_position.full(x)

    def infer(self, phones: mx.array, bert_feature: mx.array, prompt: mx.array,
              top_k: int = 15, top_p: float = 1.0, temperature: float = 1.0,
              repetition_penalty: float = 1.35, early_stop_num: int | None = None,
              key: mx.array | None = None) -> mx.array:
        """Batch=1 AR decode with KV cache. Returns generated semantic tokens (1, T)."""
        early_stop_num = early_stop_num if early_stop_num is not None else self.early_stop_num
        if key is None:
            key = mx.random.key(0)
        x = self._text_embed(phones, bert_feature)
        x_len = x.shape[1]
        y = prompt
        prefix_len = y.shape[1]

        y_emb = self.ar_audio_embedding[y]
        y_pos = self.ar_audio_position.full(y_emb)
        out = mx.concatenate([x, y_pos], axis=1)

        src_len = x_len + prefix_len
        # Mask semantics (torch infer_panel_naive):
        #   x rows: attend x only (y columns blocked)
        #   y rows: x columns always visible; causal only within the y block
        x_rows = mx.concatenate(
            [mx.zeros((x_len, x_len), mx.float32),
             mx.full((x_len, prefix_len), NEG_INF, mx.float32)], axis=1)
        y_causal = mx.triu(mx.full((prefix_len, prefix_len), NEG_INF, mx.float32), k=1)
        y_rows = mx.concatenate([mx.zeros((prefix_len, x_len), mx.float32), y_causal], axis=1)
        causal = mx.concatenate([x_rows, y_rows], axis=0)[None, None]

        k_cache: list = [None] * len(self.blocks)
        v_cache: list = [None] * len(self.blocks)
        emb_cond = None
        for idx in range(1500):
            for bi, block in enumerate(self.blocks):
                if idx == 0:
                    out, k_cache[bi], v_cache[bi] = block.run(
                        out, emb_cond, causal, k_cache[bi], v_cache[bi], prefill=True)
                else:
                    out, k_cache[bi], v_cache[bi] = block.run(
                        out, emb_cond, None, k_cache[bi], v_cache[bi], prefill=False)
            logits = out[:, -1] @ self.ar_predict_layer_w.T

            if idx < 11:
                logits = logits[:, :-1]  # EOS forbidden for first 10 tokens

            key, step_key = mx.random.split(key)
            sample = _sample(logits, y, top_k, top_p, temperature, repetition_penalty, step_key)
            y = mx.concatenate([y, sample], axis=1)

            # torch: stop when argmax(logits)==EOS OR sampled token==EOS
            amax = int(mx.argmax(logits[0]))
            if amax == self.EOS or int(sample[0, 0]) == self.EOS:
                if y.shape[1] == prefix_len:
                    y = mx.concatenate([y, mx.zeros((1, 1), mx.int32)], axis=1)
                break
            if (y.shape[1] - prefix_len) > early_stop_num:
                break

            emb = self.ar_audio_embedding[sample]
            emb_cond = self.ar_audio_position.step(emb, prefix_len + idx)
            out = emb_cond

        return y[:, prefix_len:]


def _sample(logits: mx.array, previous_tokens: mx.array, top_k: int, top_p: float,
            temperature: float, repetition_penalty: float, key: mx.array | None) -> mx.array:
    """AR/models/utils.py sample() for batch=1. Returns (1, 1) int32."""
    logits = logits.astype(mx.float32)
    if repetition_penalty != 1.0:
        prev = previous_tokens[0].astype(mx.int32)
        score = mx.take_along_axis(logits, prev[None, :], axis=-1)
        score = mx.where(score < 0, score * repetition_penalty, score / repetition_penalty)
        logits = mx.put_along_axis(logits, prev[None, :], score, axis=-1)
    if top_p is not None and top_p < 1.0:
        order = mx.argsort(-logits, axis=-1)
        sorted_logits = mx.take_along_axis(logits, order, axis=-1)
        cum = mx.cumsum(mx.softmax(sorted_logits, axis=-1), axis=-1)
        remove_sorted = cum > top_p
        remove_sorted[:, 0] = False
        remove = mx.zeros_like(remove_sorted)
        remove = mx.put_along_axis(remove, order, remove_sorted, axis=-1)
        logits = mx.where(remove, NEG_INF, logits)
    logits = logits / max(temperature, 1e-5)
    if top_k is not None and top_k > 0:
        kth = mx.sort(logits, axis=-1)[:, -top_k][:, None]
        logits = mx.where(logits < kth, NEG_INF, logits)
    probs = mx.softmax(logits, axis=-1)
    # One uniform per row (batch), not per column. Inverse-CDF over the row distribution.
    cdf = mx.cumsum(probs, axis=-1)
    u = np.random.random((probs.shape[0],))
    idx_np = np.empty((probs.shape[0],), dtype=np.int64)
    cdf_np = np.asarray(cdf)
    for b in range(probs.shape[0]):
        idx_np[b] = np.searchsorted(cdf_np[b], u[b], side='right')
        idx_np[b] = min(idx_np[b], probs.shape[-1] - 1)
    return mx.array(idx_np)[:, None].astype(mx.int32)
