"""MLX BERT encoder (chinese-roberta-wwm-ext-large) for GPT-SoVITS bert features.

Only the encoder path is needed: embeddings -> 24 layers -> hidden_states.
We reproduce transformers' BertModel forward for the -3 hidden state (index -3 of
hidden_states tuple = output of layer 21 for a 24-layer model... note transformers
returns (embeddings, layer1..24); [-3] = layer 22 output). Actually hidden_states[-3:-2]
selects layer 22 output (second to last non-embedding... for 24 layers: [-1]=24, [-2]=23,
[-3]=22). GPT-SoVITS uses hidden_states[-3:-2] => output of encoder layer 22.

Weights loaded from converted safetensors (keys prefixed "bert.").
Tokenizer: HF fast tokenizer via tokenizers library (pure rust, no torch).
"""

from __future__ import annotations

import json
import math

import mlx.core as mx
import mlx.nn as nn


class BertSelfAttention:
    def __init__(self, dim: int, heads: int):
        self.heads = heads
        self.head_dim = dim // heads

    def __call__(self, q_w, q_b, k_w, k_b, v_w, v_b, x, ext_mask):
        b, t, d = x.shape
        q = (x @ q_w.T + q_b).reshape(b, t, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        k = (x @ k_w.T + k_b).reshape(b, t, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        v = (x @ v_w.T + v_b).reshape(b, t, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=1.0 / math.sqrt(self.head_dim), mask=ext_mask)
        return out.transpose(0, 2, 1, 3).reshape(b, t, d)


class BertLayer:
    def __init__(self, arrays, i: int, dim: int, heads: int, inter: int, act: str, eps: float):
        g = arrays.get
        p = f"bert.encoder.layer.{i}."
        self.attention = BertSelfAttention(dim, heads)
        self.q_w, self.q_b = g(p + "attention.self.query.weight"), g(p + "attention.self.query.bias")
        self.k_w, self.k_b = g(p + "attention.self.key.weight"), g(p + "attention.self.key.bias")
        self.v_w, self.v_b = g(p + "attention.self.value.weight"), g(p + "attention.self.value.bias")
        self.o_w, self.o_b = g(p + "attention.output.dense.weight"), g(p + "attention.output.dense.bias")
        self.a_ln_w = g(p + "attention.output.LayerNorm.weight")
        self.a_ln_b = g(p + "attention.output.LayerNorm.bias")
        self.i_w, self.i_b = g(p + "intermediate.dense.weight"), g(p + "intermediate.dense.bias")
        self.o2_w, self.o2_b = g(p + "output.dense.weight"), g(p + "output.dense.bias")
        self.o_ln_w = g(p + "output.LayerNorm.weight")
        self.o_ln_b = g(p + "output.LayerNorm.bias")
        self.act = act
        self.eps = eps
        self.inter = inter

    def __call__(self, x: mx.array, ext_mask) -> mx.array:
        a = self.attention(self.q_w, self.q_b, self.k_w, self.k_b, self.v_w, self.v_b,
                           x, ext_mask)
        x = mx.fast.layer_norm(x @ self.o_w.T + self.o_b + x, self.a_ln_w, self.a_ln_b, self.eps)
        h = x @ self.i_w.T + self.i_b
        h = nn.gelu_approx(h) if self.act == "gelu" else nn.gelu(h)
        x = mx.fast.layer_norm(h @ self.o2_w.T + self.o2_b + x, self.o_ln_w, self.o_ln_b, self.eps)
        return x


class BertModel:
    def __init__(self, arrays: dict, config: dict):
        g = arrays.get
        self.eps = config.get("layer_norm_eps", 1e-12)
        dim = config["hidden_size"]
        self.dim = dim
        self.layers = [
            BertLayer(arrays, i, dim, config["num_attention_heads"],
                      config["intermediate_size"], config.get("hidden_act", "gelu"), self.eps)
            for i in range(config["num_hidden_layers"])
        ]
        self.word_emb = g("bert.embeddings.word_embeddings.weight")
        self.pos_emb = g("bert.embeddings.position_embeddings.weight")
        self.tok_type_emb = g("bert.embeddings.token_type_embeddings.weight")
        self.ln_w = g("bert.embeddings.LayerNorm.weight")
        self.ln_b = g("bert.embeddings.LayerNorm.bias")

    def embed(self, input_ids: mx.array, token_type_ids: mx.array | None = None) -> mx.array:
        b, t = input_ids.shape
        if token_type_ids is None:
            token_type_ids = mx.zeros((b, t), mx.int32)
        x = (self.word_emb[input_ids] + self.pos_emb[:t][None]
             + self.tok_type_emb[token_type_ids])
        return mx.fast.layer_norm(x, self.ln_w, self.ln_b, self.eps)

    def hidden_states(self, input_ids: mx.array, attention_mask: mx.array) -> list:
        """Returns per-layer outputs [emb, l1, ..., lN]."""
        x = self.embed(input_ids)
        # additive mask: 0 keep, -inf pad (broadcast over heads/queries)
        m = (1.0 - attention_mask[:, None, None, :].astype(mx.float32)) * -1e9
        states = [x]
        for layer in self.layers:
            x = layer(x, m)
            states.append(x)
        return states

    def get_bert_feature(self, input_ids: mx.array, attention_mask: mx.array,
                         word2ph: list[int]) -> mx.array:
        """TextPreprocessor.get_bert_feature: hidden_states[-3] (skip CLS/SEP), repeat by word2ph.

        Returns (1024, sum(word2ph)) phone-level feature.
        """
        states = self.hidden_states(input_ids, attention_mask)
        res = states[-3][0][1:-1]  # drop [CLS], [SEP]
        feats = []
        for i, n in enumerate(word2ph):
            if n > 0:
                feats.append(mx.repeat(res[i][None], n, axis=0))
        return mx.transpose(mx.concatenate(feats, axis=0), (1, 0))
