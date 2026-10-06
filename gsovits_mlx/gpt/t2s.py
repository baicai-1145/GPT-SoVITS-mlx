"""AR GPT t2s transformer (v1/v2 s1 ckpts & s1v3.ckpt). Pure MLX, batch=1 optimized.

Architecture per t2s_model.py + AR/modules/transformer.py:
- pre-norm transformer, AdaptiveLayerNorm (weight/bias conditioned on LAST token embedding)
- packed in_proj qkv, learned sine positional embeddings with learned alpha scale
- inference: prompt prefill (mask-aware), then per-token decode with KV cache
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

NEG_INF = -1e9


class SinePositionalEmbedding:
    """AR/modules/embedding.py SinePositionalEmbedding (scale=False, alpha=True)."""

    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.x_scale = 1.0
        self.alpha = mx.ones((1,))
        position = mx.arange(0, 4000, dtype=mx.float32)[:, None]
        div_term = mx.exp(mx.arange(0, embedding_dim, 2, dtype=mx.float32)
                          * -(math.log(10000.0) / embedding_dim))
        pe = mx.zeros((4000, embedding_dim))
        pe[:, 0::2] = mx.sin(position * div_term)
        pe[:, 1::2] = mx.cos(position * div_term)
        self.pe = pe[None]  # (1, 4000, D) fp32

    def full(self, x: mx.array) -> mx.array:
        """x: (B, T, D) -> positional add."""
        return x * self.x_scale + self.alpha * self.pe[:, : x.shape[1]].astype(x.dtype)

    def step(self, emb: mx.array, pos: int) -> mx.array:
        """emb: (B, 1, D) for decode step at absolute position `pos`."""
        return emb * self.x_scale + self.alpha * self.pe[:, pos : pos + 1].astype(emb.dtype)


class AdaptiveLayerNorm(nn.Module):
    """norm1/norm2 in s1 ckpts: weight & bias predicted from last token hidden."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.project_layer_w = mx.zeros((2 * d_model, d_model))
        self.project_layer_b = mx.zeros((2 * d_model,))
        self.eps = eps

    def __call__(self, x: mx.array, embedding: mx.array) -> mx.array:
        # embedding: (B, 1, D) conditioning (last token of previous output)
        wb = embedding @ self.project_layer_w.T + self.project_layer_b  # (B,1,2D)
        weight, bias = wb[..., : x.shape[-1]], wb[..., x.shape[-1]:]
        return weight * mx.fast.layer_norm(x, None, None, self.eps) + bias


class T2SBlock:
    """Functional block holding transposed-friendly weights."""

    def __init__(self, num_heads: int, hidden_dim: int):
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        self.norm_eps = 1e-5

    def weights(self, p: dict, i: int):
        get = p.__getitem__
        self.qkv_w = get(f"model.h.layers.{i}.self_attn.in_proj_weight")   # (3D, D)
        self.qkv_b = get(f"model.h.layers.{i}.self_attn.in_proj_bias")
        self.out_w = get(f"model.h.layers.{i}.self_attn.out_proj.weight")  # (D, D)
        self.out_b = get(f"model.h.layers.{i}.self_attn.out_proj.bias")
        self.norm1 = AdaptiveLayerNorm(self.hidden_dim)
        self.norm1.project_layer_w = get(f"model.h.layers.{i}.norm1.project_layer.weight")
        self.norm1.project_layer_b = get(f"model.h.layers.{i}.norm1.project_layer.bias")
        self.norm2 = AdaptiveLayerNorm(self.hidden_dim)
        self.norm2.project_layer_w = get(f"model.h.layers.{i}.norm2.project_layer.weight")
        self.norm2.project_layer_b = get(f"model.h.layers.{i}.norm2.project_layer.bias")
        self.linear1_w = get(f"model.h.layers.{i}.linear1.weight")
        self.linear1_b = get(f"model.h.layers.{i}.linear1.bias")
        self.linear2_w = get(f"model.h.layers.{i}.linear2.weight")
        self.linear2_b = get(f"model.h.layers.{i}.linear2.bias")
        return self

    def _mlp(self, x: mx.array) -> mx.array:
        h = x @ self.linear1_w.T + self.linear1_b
        h = nn.relu(h)
        return h @ self.linear2_w.T + self.linear2_b

    def _attn(self, x: mx.array, k_cache: mx.array, v_cache: mx.array,
              mask: mx.array | None) -> tuple[mx.array, mx.array, mx.array]:
        b, t, d = x.shape
        qkv = x @ self.qkv_w.T + self.qkv_b
        q, k, v = qkv[..., :d], qkv[..., d : 2 * d], qkv[..., 2 * d :]
        if k_cache is not None:
            k_cache = mx.concatenate([k_cache, k], axis=1)
            v_cache = mx.concatenate([v_cache, v], axis=1)
        else:
            k_cache, v_cache = k, v
        kv_len = k_cache.shape[1]

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

    def prefill(self, x: mx.array, emb_cond: mx.array, mask: mx.array | None,
                k_cache=None, v_cache=None):
        attn, k_cache, v_cache = self._attn(x, k_cache, v_cache, mask)
        x = x + attn
        x = self.norm1(x, emb_cond)
        x = x + self._mlp(x)
        x = self.norm2(x, emb_cond)
        return x, k_cache, v_cache


class Text2SemanticDecoder:
    """Weight holder + inference graph. Params loaded flat into a dict."""

    def __init__(self, config: dict):
        m = config["model"]
        self.model_dim = hidden = m["hidden_dim"]
        self.embedding_dim = m["embedding_dim"]
        self.num_head = m["head"]
        self.num_layers = m["n_layer"]
        self.vocab_size = m["vocab_size"]
        self.phoneme_vocab_size = m["phoneme_vocab_size"]
        self.EOS = m["EOS"]
        self.max_sec = config["data"]["max_sec"]
        self.early_stop_num = 50 * self.max_sec

        self.blocks = [T2SBlock(self.num_head, hidden) for _ in range(self.num_layers)]
        self.ar_text_position = SinePositionalEmbedding(self.embedding_dim)
        self.ar_audio_position = SinePositionalEmbedding(self.embedding_dim)

    def load(self, params: dict):
        """params: flat dict with 'model.'-prefixed keys (fp16 mlx arrays)."""
        self.params = params
        for i, block in enumerate(self.blocks):
            block.weights(params, i)
        self.ar_text_embedding = params["model.ar_text_embedding.word_embeddings.weight"]
        self.ar_audio_embedding = params["model.ar_audio_embedding.word_embeddings.weight"]
        self.bert_proj_w = params["model.bert_proj.weight"]
        self.bert_proj_b = params["model.bert_proj.bias"]
        self.ar_text_position.alpha = params["model.ar_text_position.alpha"]
        self.ar_audio_position.alpha = params["model.ar_audio_position.alpha"]
        self.ar_predict_layer_w = params["model.ar_predict_layer.weight"]
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
        x = self._text_embed(phones, bert_feature)  # (1, T_ph, D)
        x_len = x.shape[1]
        y = prompt  # (1, T_pr)
        prefix_len = y.shape[1]

        y_emb = self.ar_audio_embedding[y]
        y_pos = self.ar_audio_position.full(y_emb)
        xy_pos = mx.concatenate([x, y_pos], axis=1)

        # causal mask: (1, 1, src, src); text block fully visible, prompt causal
        src_len = x_len + prefix_len
        causal = mx.triu(mx.full((src_len, src_len), NEG_INF, mx.float32), k=1)[None, None]
        causal[:, :, :x_len, :x_len] = 0

        k_caches: list = []
        v_caches: list = []
        out = xy_pos
        emb_cond = None
        for idx in range(1500):
            for li, block in enumerate(self.blocks):
                cond = emb_cond if emb_cond is not None else out[:, -1:]
                if idx == 0:
                    out, k_caches, v_caches = _run_block(block, out, cond, causal,
                                                         k_caches, v_caches, li, prefill=True)
                else:
                    out, k_caches, v_caches = _run_block(block, out, cond, None,
                                                         k_caches, v_caches, li, prefill=False)
            logits = out[:, -1] @ self.ar_predict_layer_w.T  # (1, vocab)

            if idx < 11:
                logits = logits[:, :-1]  # EOS forbidden for first 10 tokens

            sample = _sample(logits, y, top_k, top_p, temperature, repetition_penalty, key)
            y = mx.concatenate([y, sample], axis=1)

            if int(mx.argmax(logits, axis=-1)[0]) == self.EOS or int(sample[0, 0]) == self.EOS:
                if y.shape[1] == prefix_len:
                    y = mx.concatenate([y, mx.zeros((1, 1), mx.int32)], axis=1)
                break
            if (y.shape[1] - prefix_len) > early_stop_num:
                break

            emb = self.ar_audio_embedding[sample]
            pos_idx = prefix_len + idx
            emb_cond = self.ar_audio_position.step(emb, pos_idx)  # input for next step
            out = emb_cond

        return y[:, prefix_len:]

def _run_block(block: T2SBlock, x: mx.array, emb_cond: mx.array, mask: mx.array | None,
               k_cache, v_cache, layer: int, prefill: bool):
    if prefill:
        attn, k_out, v_out = block._attn(x, None, None, mask)
        k_cache = list(k_cache) if k_cache else []
        v_cache = list(v_cache) if v_cache else []
        k_cache.append(k_out)
        v_cache.append(v_out)
    else:
        attn, k_new, v_new = block._attn(x, k_cache[layer], v_cache[layer], None)
        k_cache[layer] = k_new
        v_cache[layer] = v_new
    x = x + attn
    x = block.norm1(x, emb_cond)
    x = x + block._mlp(x)
    x = block.norm2(x, emb_cond)
    return x, k_cache, v_cache


def _sample(logits: mx.array, previous_tokens: mx.array, top_k: int, top_p: float,
            temperature: float, repetition_penalty: float, key: mx.array | None) -> mx.array:
    """AR/models/utils.py sample() for batch=1. Returns (1, 1) int32."""
    logits = logits.astype(mx.float32)
    if repetition_penalty != 1.0:
        prev = previous_tokens[0].astype(mx.int32)  # (T,) with duplicates
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
    q = mx.random.uniform(shape=probs.shape, key=key)
    cdf = mx.cumsum(probs, axis=-1)
    idx = mx.argmax((cdf > q).astype(mx.int32), axis=-1)  # (1,)
    return idx[:, None].astype(mx.int32)
