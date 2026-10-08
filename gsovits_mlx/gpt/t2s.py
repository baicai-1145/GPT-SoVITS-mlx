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

from .kv_cache import KVBuffer

import math
import os

import mlx.core as mx
import numpy as np
import mlx.nn as nn

from .mixed_gemv import _launch as launch_mixed_gemv, supports as supports_mixed_gemv

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
        self._projection_cache = {}
        self._cache_promoted_weights = os.environ.get("GSOVITS_AR_PROMOTE_CACHE", "1") != "0"
        self._use_mixed_gemv = os.environ.get("GSOVITS_AR_MIXED_GEMV", "1") == "1"

    def weights(self, p: dict, i: int):
        self._projection_cache.clear()
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

    def _linear(self, x: mx.array, weight: str, bias: str, *,
                relu: bool = False, residual=None) -> mx.array:
        w, b = getattr(self, weight), getattr(self, bias)
        if self._use_mixed_gemv and supports_mixed_gemv(x, w, b, residual):
            return launch_mixed_gemv(x, w, b, relu=relu, residual=residual)
        dtype = mx.result_type(x.dtype, w.dtype)
        if self._cache_promoted_weights and w.dtype != dtype:
            key = (weight, dtype)
            if key not in self._projection_cache:
                # Mixed streams otherwise recast the full matrix at every token.
                self._projection_cache[key] = w.astype(dtype)
            w = self._projection_cache[key]
        out = x @ w.T + b
        if residual is not None:
            out = out + residual
        return nn.relu(out) if relu else out

    def _mlp(self, x: mx.array) -> mx.array:
        h = self._linear(x, "linear1_w", "linear1_b", relu=True)
        return self._linear(h, "linear2_w", "linear2_b")

    def _mlp_residual(self, x: mx.array) -> mx.array:
        h = self._linear(x, "linear1_w", "linear1_b", relu=True)
        return self._linear(h, "linear2_w", "linear2_b", residual=x)

    def _attn(self, x: mx.array, k_cache: mx.array | KVBuffer | None,
              v_cache: mx.array | KVBuffer | None, mask: mx.array | None):
        b, t, d = x.shape
        buffered = isinstance(k_cache, KVBuffer)
        assert isinstance(v_cache, KVBuffer) == buffered, "K/V cache types must match"
        qkv = self._linear(x, "qkv_w", "qkv_b")
        q, k, v = qkv[..., :d], qkv[..., d : 2 * d], qkv[..., 2 * d :]
        if buffered:
            k_cache.append(k)
            v_cache.append(v)
            active_k, active_v = k_cache.active(), v_cache.active()
        elif k_cache is not None:
            k_cache = mx.concatenate([k_cache, k], axis=1)
            v_cache = mx.concatenate([v_cache, v], axis=1)
            active_k, active_v = k_cache, v_cache
        else:
            k_cache, v_cache = k, v
            active_k, active_v = k_cache, v_cache

        def split_heads(t4: mx.array) -> mx.array:
            return t4.reshape(b, t4.shape[1], self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        qh = split_heads(q)
        kh = split_heads(active_k)
        vh = split_heads(active_v)
        out = mx.fast.scaled_dot_product_attention(
            qh, kh, vh, scale=1.0 / math.sqrt(self.head_dim),
            mask=mask if mask is not None else None)
        out = out.transpose(0, 2, 1, 3).reshape(b, t, d)
        out = self._linear(out, "out_w", "out_b")
        return out, k_cache, v_cache

    def _attn_fastcache(self, x: mx.array, cache: dict | None, mask: mx.array | None,
                        pos: int):
        """Attention against a PREALLOCATED (B, H, T_max, D) KV buffer.

        Identical math to _attn with an incrementally built cache: the buffer
        holds the same values in the same order — new keys/values are written
        at slot ``pos`` (slice update, no history copy) and attention reads
        the lazy slice [.., :pos+1, ..]. The legacy path copies the whole
        history per step (concatenate) and re-splits heads per step; this one
        writes one slot and never re-touches history. Returns the same
        attention output plus the (unchanged except slot pos) buffer dict.
        """
        b, t, d = x.shape
        qkv = self._linear(x, "qkv_w", "qkv_b")
        q, k, v = qkv[..., :d], qkv[..., d : 2 * d], qkv[..., 2 * d :]

        def split_heads(t4: mx.array) -> mx.array:
            return t4.reshape(b, t4.shape[1], self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        if cache is None:
            t_total = t
            kh, vh = split_heads(k), split_heads(v)
        else:
            kh, vh = split_heads(k), split_heads(v)
            cache["k"][:, :, pos : pos + t, :] = kh
            cache["v"][:, :, pos : pos + t, :] = vh
            kh = cache["k"][:, :, : pos + t, :]
            vh = cache["v"][:, :, : pos + t, :]
        qh = split_heads(q)
        out = mx.fast.scaled_dot_product_attention(
            qh, kh, vh, scale=1.0 / math.sqrt(self.head_dim),
            mask=mask if mask is not None else None)
        out = out.transpose(0, 2, 1, 3).reshape(b, t, d)
        out = self._linear(out, "out_w", "out_b")
        if cache is None:
            return out, kh, vh
        return out, cache, cache

    def run(self, x: mx.array, emb_cond: mx.array | None, mask: mx.array | None,
            k_cache: mx.array | KVBuffer | None, v_cache: mx.array | KVBuffer | None,
            prefill: bool):
        if prefill:
            attn, k_out, v_out = self._attn(x, None, None, mask)
            k_cache = k_out  # (b, t, d) for this block
            v_cache = v_out
        else:
            attn, k_cache, v_cache = self._attn(x, k_cache, v_cache, None)
        # post-norm
        x = self._ln(x + attn, 1, emb_cond)
        x = self._ln(self._mlp_residual(x), 2, emb_cond)
        return x, k_cache, v_cache

    def run_fastcache(self, x: mx.array, emb_cond: mx.array | None,
                      cache: dict | None, pos: int, mask: mx.array | None = None):
        """Prefill (cache=None) or one decode step against a preallocated KV
        buffer (see _attn_fastcache). Returns (x, cache_a, cache_b): prefill
        yields the head-split kh/vh to be wrapped into padded buffers; a
        decode step yields the updated cache dict in both slots."""
        attn, c1, c2 = self._attn_fastcache(x, cache, mask, pos)
        x = self._ln(x + attn, 1, emb_cond)
        x = self._ln(x + self._mlp(x), 2, emb_cond)
        return x, c1, c2


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
              key: mx.array | None = None, fast_cache: bool = False) -> mx.array:
        """Batch=1 AR decode with KV cache. Returns generated semantic tokens (1, T).

        fast_cache=False (default) is the concat-cache path, bit-identical to
        the verified streams (anchors/parity gates). fast_cache=True uses
        preallocated buffers: strictly less memory traffic (wins at long T)
        but SDPA over non-contiguous views takes a different fp16 kernel path,
        which flips inverse-CDF near-ties -> different (equally valid) token
        stream. Never enable it for anchor/parity runs.
        """
        early_stop_num = early_stop_num if early_stop_num is not None else self.early_stop_num
        if key is None:
            key = mx.random.key(0)
        x = self._text_embed(phones, bert_feature)
        x_len = x.shape[1]
        y = prompt
        prefix_len = y.shape[1]
        cpu_sample_info = os.environ.get("GSOVITS_AR_CPU_SAMPLE_INFO", "1") == "1"
        buffered_kv = not fast_cache and os.environ.get("GSOVITS_AR_TMAJOR_CACHE", "1") == "1"
        predict_weights = {}

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
        caches: list = [None] * len(self.blocks)
        t_max = src_len + 1500  # decode loop is hard-capped at 1500 steps
        emb_cond = None
        for idx in range(1500):
            if fast_cache:
                if idx == 0:
                    kh_list, vh_list = [], []
                    for block in self.blocks:
                        out, kh_i, vh_i = block.run_fastcache(
                            out, emb_cond, None, 0, mask=causal)
                        kh_list.append(kh_i)
                        vh_list.append(vh_i)
                    # one-time wrap into preallocated buffers (dtype follows
                    # the computed k/v; slice writes afterwards, no copies)
                    for bi in range(len(self.blocks)):
                        kh, vh = kh_list[bi], vh_list[bi]
                        kb = mx.zeros((1, kh.shape[1], t_max, kh.shape[3]), dtype=kh.dtype)
                        kb[:, :, :src_len, :] = kh
                        vb = mx.zeros((1, vh.shape[1], t_max, vh.shape[3]), dtype=vh.dtype)
                        vb[:, :, :src_len, :] = vh
                        caches[bi] = {"k": kb, "v": vb}
                else:
                    pos = src_len + idx - 1
                    for bi, block in enumerate(self.blocks):
                        out, caches[bi], _ = block.run_fastcache(
                            out, emb_cond, caches[bi], pos)
            else:
                for bi, block in enumerate(self.blocks):
                    if idx == 0:
                        out, k_cache[bi], v_cache[bi] = block.run(
                            out, emb_cond, causal, k_cache[bi], v_cache[bi], prefill=True)
                        if buffered_kv:
                            k_cache[bi], v_cache[bi] = KVBuffer(k_cache[bi]), KVBuffer(v_cache[bi])
                    else:
                        out, k_cache[bi], v_cache[bi] = block.run(
                            out, emb_cond, None, k_cache[bi], v_cache[bi], prefill=False)
            dtype = mx.result_type(out.dtype, self.ar_predict_layer_w.dtype)
            if dtype not in predict_weights:
                predict_weights[dtype] = self.ar_predict_layer_w.astype(dtype)
            logits = out[:, -1] @ predict_weights[dtype].T

            if idx < 11:
                logits = logits[:, :-1]  # EOS forbidden for first 10 tokens

            key, step_key = mx.random.split(key)
            sample_result = _sample(logits, y, top_k, top_p, temperature,
                                    repetition_penalty, step_key,
                                    _return_info=cpu_sample_info)
            if cpu_sample_info:
                sample, sample_id, amax = sample_result
            else:
                sample = sample_result
            y = mx.concatenate([y, sample], axis=1)
            # Realize per-step state (whole-process footprint fix, 2026-10-08):
            # without this the concat-cache path keeps every step's k/v concat
            # and y-concat graphs lazy until the final decode eval — the graph
            # piles up O(T^2) and materializes as a 3-8GB end-of-run spike
            # (measured: v2 6.6GB, v5turbo 10.0GB whole-process phys_footprint).
            # Bit-identical outputs (eval is placement-only).
            state = [y, sample, logits, out]
            if not fast_cache:
                state.extend(c.buffer if isinstance(c, KVBuffer) else c for c in k_cache)
                state.extend(c.buffer if isinstance(c, KVBuffer) else c for c in v_cache)
            mx.eval(*state)

            # torch: stop when argmax(logits)==EOS OR sampled token==EOS
            if not cpu_sample_info:
                amax = int(mx.argmax(logits[0]))
                sample_id = int(sample[0, 0])
            if amax == self.EOS or sample_id == self.EOS:
                if y.shape[1] == prefix_len:
                    y = mx.concatenate([y, mx.zeros((1, 1), mx.int32)], axis=1)
                break
            if (y.shape[1] - prefix_len) > early_stop_num:
                break

            emb = self.ar_audio_embedding[sample]
            emb_cond = self.ar_audio_position.step(emb, prefix_len + idx)
            out = emb_cond

        return y[:, prefix_len:]

    def infer_batch(self, inputs: list, top_k: int = 15, top_p: float = 1.0,
                    temperature: float = 1.0, repetition_penalty: float = 1.35,
                    early_stop_num: int | None = None,
                    uniforms: list | None = None) -> list:
        """Batched AR decode over B independent segments (production-structure).

        Each element of ``inputs`` is an ``(phones, bert_feature, prompt)``
        tuple, exactly what :meth:`infer` takes per segment. Rows must share
        equal (x_len + prefix_len); the prefill mask is built from the first
        row. Per-row semantics mirror the serial loop: same op family/order
        (fp16 matmul _linear fallback, SDPA, layer_norm), same uniform
        consumption when ``uniforms[r][i]`` replays the serial per-row stream
        (token-equality gate), and per-row EOS freeze (frozen rows pad their
        prev with the last real token; their KV slot keeps receiving writes
        that are never read back). Eval discipline matches fast_cache:
        [y, sample, logits, out] only — KV buffers stay lazy.
        Returns per-row token arrays like infer().
        """
        early_stop_num = early_stop_num if early_stop_num is not None else self.early_stop_num
        B = len(inputs)
        if B == 1:
            ph, bt, pr = inputs[0]
            return [self.infer(ph, bt, pr, top_k, top_p, temperature,
                               repetition_penalty, early_stop_num=early_stop_num)]

        xs = []
        ys_np = []
        x_lens = []
        prefix_len = inputs[0][2].shape[1]
        for ph, bt, pr in inputs:
            x = self._text_embed(ph, bt)
            assert pr.shape[1] == prefix_len, "infer_batch requires equal prompt lengths"
            xs.append(x)
            x_lens.append(x.shape[1])
            ys_np.append(list(np.asarray(pr)[0]))

        # ---- left-pad ragged rows to a common x_len (official semantics) ----
        # Official infer_panel_batch_infer pads the EMBEDDED x on the LEFT
        # with zeros; the padding mask hides pad columns from attention, so
        # this is mathematically equivalent to per-row unpadded decode.
        x_len = max(x_lens)
        src_len = x_len + prefix_len
        t_max = src_len + 1500
        caches: list = [None] * len(self.blocks)
        outs = []
        for si, (ph, bt, pr) in enumerate(inputs):
            x = xs[si]
            xl = x_lens[si]
            pad = x_len - xl
            if pad:
                x = mx.concatenate(
                    [mx.zeros((1, pad, x.shape[2]), dtype=x.dtype), x], axis=1)
            y_emb = self.ar_audio_embedding[pr]
            y_pos = self.ar_audio_position.full(y_emb)
            out = mx.concatenate([x, y_pos], axis=1)
            # additive mask, official layout: pad columns invisible, causal
            # within the visible (x, y) region. pad rows are computed but
            # never read (same NaN-avoidance rationale as the official code).
            x_rows = mx.concatenate(
                [mx.zeros((x_len, x_len), mx.float32),
                 mx.full((x_len, prefix_len), NEG_INF, mx.float32)], axis=1)
            y_causal = mx.triu(mx.full((prefix_len, prefix_len), NEG_INF, mx.float32), k=1)
            y_rows = mx.concatenate(
                [mx.zeros((prefix_len, x_len), mx.float32), y_causal], axis=1)
            causal = mx.concatenate([x_rows, y_rows], axis=0)[None, None]
            if pad:
                # mask out pad COLUMNS: queries may not attend left-pad slots
                # (pad occupies the FIRST `pad` positions after left-padding)
                col_block = mx.full((src_len, pad), NEG_INF, mx.float32)
                rest = mx.zeros((src_len, src_len - pad), mx.float32)
                pad_cols = mx.concatenate([col_block, rest], axis=1)
                causal = mx.minimum(causal, pad_cols[None, None])
            kh_list, vh_list = [], []
            for block in self.blocks:
                out, kh_i, vh_i = block.run_fastcache(out, None, None, 0, mask=causal)
                kh_list.append(kh_i); vh_list.append(vh_i)
            outs.append(out[:, -1:])
            if si == 0:
                for bi in range(len(self.blocks)):
                    kh, vh = kh_list[bi], vh_list[bi]
                    kb = mx.zeros((B, kh.shape[1], t_max, kh.shape[3]), dtype=kh.dtype)
                    vb = mx.zeros((B, vh.shape[1], t_max, vh.shape[3]), dtype=vh.dtype)
                    caches[bi] = {"k": kb, "v": vb}
            for bi in range(len(self.blocks)):
                caches[bi]["k"][si, :, :src_len, :] = kh_list[bi][0]
                caches[bi]["v"][si, :, :src_len, :] = vh_list[bi][0]

        # per-row decode mask: hide left-pad slots from decode-step
        # attention (pad slots hold garbage K/V; serial rows have none).
        row_pads = [x_len - l for l in x_lens]
        if any(row_pads):
            pad_mask = mx.concatenate(
                [mx.concatenate(
                    [mx.full((1, 1, 1, p), NEG_INF, mx.float32),
                     mx.zeros((1, 1, 1, t_max - p), mx.float32)], axis=3)
                 for p in row_pads], axis=0)
        else:
            pad_mask = None

        out = mx.concatenate(outs, axis=0)  # (B, 1, D)
        predict_w = self.ar_predict_layer_w.astype(
            mx.result_type(out.dtype, self.ar_predict_layer_w.dtype))
        finished = [False] * B
        tok_rows = [[] for _ in range(B)]
        u_idx = [0] * B
        pos = src_len
        L = prefix_len

        # ---- per-step cost reduction: locals + incremental prev buffer ----
        # prev_rows: one np int32 buffer per row, grown in place; padded view
        # built once per step WITHOUT rebuilding python lists (O(1) append).
        prev_rows = [np.asarray(ys_np[r], dtype=np.int32) for r in range(B)]
        emb_table = self.ar_audio_embedding
        pos_layer = self.ar_audio_position
        blocks_local = self.blocks
        sq_hd = 1.0 / math.sqrt(blocks_local[0].head_dim)
        n_heads = blocks_local[0].num_heads
        head_dim = blocks_local[0].head_dim
        EOS = self.EOS
        for idx in range(1500):
            logits = out[:, -1] @ predict_w.T
            if idx < 11:
                logits = logits[:, :-1]
            logits = logits.astype(mx.float32)

            # batched sampling: one graph, ONE host sync
            L = max(len(prev_rows[r]) for r in range(B))
            prev_np = np.empty((B, L), dtype=np.int32)
            for r in range(B):
                pr = prev_rows[r]
                if len(pr) < L:
                    prev_np[r, :len(pr)] = pr
                    prev_np[r, len(pr):] = pr[-1]
                else:
                    prev_np[r] = pr
            prev = mx.array(prev_np)
            lg = logits
            if repetition_penalty != 1.0:
                score = mx.take_along_axis(lg, prev, axis=-1)
                score = mx.where(score < 0, score * repetition_penalty, score / repetition_penalty)
                lg = mx.put_along_axis(lg, prev, score, axis=-1)
            if top_p is not None and top_p < 1.0:
                raise NotImplementedError("top_p batched path not yet gated")
            lg = lg / max(temperature, 1e-5)
            if top_k is not None and top_k > 0:
                kth = mx.sort(lg, axis=-1)[:, -top_k][:, None]
                lg = mx.where(lg < kth, NEG_INF, lg)
            probs = mx.softmax(lg, axis=-1)
            cdf = mx.cumsum(probs, axis=-1)
            amax_all = mx.argmax(logits, axis=-1)
            cdf_np = np.asarray(cdf)          # single pipeline sync per step
            amax_np = np.asarray(amax_all)
            samples = []
            for r in range(B):
                if finished[r]:
                    tok_rows[r].append(tok_rows[r][-1])
                    continue
                if uniforms is not None:
                    u = uniforms[r][u_idx[r]] if u_idx[r] < len(uniforms[r]) else np.random.random()
                else:
                    u = np.random.random()
                u_idx[r] += 1
                i = int(np.searchsorted(cdf_np[r], u, side='right').item())
                i = min(i, probs.shape[-1] - 1)
                tok_rows[r].append(i)
                prev_rows[r] = np.append(prev_rows[r], i)
                if int(amax_np[r]) == EOS or i == EOS:
                    finished[r] = True
            if all(finished):
                break
            if (len(tok_rows[0]) > early_stop_num):
                break

            emb_tok = mx.array([[t[-1]] for t in tok_rows], dtype=mx.int32)
            emb = emb_table[emb_tok]
            emb_cond = pos_layer.step(emb, prefix_len + idx)
            x = emb_cond
            b, t, d = x.shape
            for bi, block in enumerate(blocks_local):
                qkv = block._linear(x, "qkv_w", "qkv_b")
                q, k, v = qkv[..., :d], qkv[..., d:2*d], qkv[..., 2*d:]
                def split_heads(t4):
                    return t4.reshape(b, t4.shape[1], n_heads, head_dim).transpose(0, 2, 1, 3)
                kh, vh = split_heads(k), split_heads(v)
                cache = caches[bi]
                cache["k"][:, :, pos:pos+t, :] = kh
                cache["v"][:, :, pos:pos+t, :] = vh
                qh = split_heads(q)
                o = mx.fast.scaled_dot_product_attention(
                    qh, cache["k"][:, :, :pos+t, :], cache["v"][:, :, :pos+t, :],
                    scale=sq_hd, mask=pad_mask[:, :, :, :pos+t] if pad_mask is not None else None)
                o = o.transpose(0, 2, 1, 3).reshape(b, t, d)
                attn = block._linear(o, "out_w", "out_b")
                x = block._ln(x + attn, 1, None)
                x = block._ln(x + block._mlp(x), 2, None)
            out = x
            mx.eval(out)  # fast_cache discipline: never eval KV buffers
            pos += 1

        results = []
        for r in range(B):
            toks = tok_rows[r]
            if self.EOS in toks:
                toks = toks[:toks.index(self.EOS)]  # trim at first EOS; frozen padding dropped
            results.append(mx.array([toks], dtype=mx.int32))
        return results


def _sample(logits: mx.array, previous_tokens: mx.array, top_k: int, top_p: float,
            temperature: float, repetition_penalty: float, key: mx.array | None,
            _return_info: bool = False):
    """AR/models/utils.py sample() for batch=1; optional host stop information."""
    original_logits = logits
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
    idx_np = np.empty((probs.shape[0],), dtype=np.int32)
    cdf_np = np.asarray(cdf)
    for b in range(probs.shape[0]):
        idx_np[b] = np.searchsorted(cdf_np[b], u[b], side='right')
        idx_np[b] = min(idx_np[b], probs.shape[-1] - 1)
    sample = mx.array(idx_np)[:, None]
    if _return_info:
        # The CDF evaluation has already materialized the unmodified logits.
        amax = int(np.argmax(np.asarray(original_logits)[0]))
        return sample, int(idx_np[0]), amax
    return sample


def batch_segments(inputs: list, batch_size: int = 8, threshold: float = 0.75) -> list:
    """Length-sorted bucketing for infer_batch (official TTS.to_batch rule).

    Sorts segments by phone count, then accepts a batch window when
    median/mean >= threshold (official batch_threshold=0.75), shrinking
    from the right otherwise. Returns a list of index lists; run
    infer_batch per bucket and scatter results back.
    """
    import numpy as np
    index_and_len = [[i, ph.shape[1] if hasattr(ph, "shape") else len(ph)]
                     for i, (ph, bt, pr) in enumerate(inputs)]
    index_and_len.sort(key=lambda x: x[1])
    arr = np.array(index_and_len, dtype=np.int64)
    buckets = []
    pos = 0
    while pos < arr.shape[0]:
        pos_end = min(pos + batch_size, arr.shape[0])
        while pos < pos_end:
            window = arr[pos:pos_end, 1].astype(np.float32)
            score = window[(pos_end - pos) // 2] / (window.mean() + 1e-8)
            if score >= threshold or (pos_end - pos) == 1:
                buckets.append(arr[pos:pos_end, 0].tolist())
                pos = pos_end
                break
            pos_end -= 1
    return buckets


def infer_batch_bucketed(self_inputs, gpt, batch_size: int = 8,
                         threshold: float = 0.75, uniforms=None, **kwargs) -> list:
    """infer_batch over length-sorted buckets; returns results in input order.

    ``uniforms`` (optional) are per-INPUT-ORDER uniform lists; they are
    reordered per bucket before replay so row r consumes the right stream.
    """
    buckets = batch_segments(self_inputs, batch_size, threshold)
    results = [None] * len(self_inputs)
    for bucket in buckets:
        segs = [self_inputs[i] for i in bucket]
        u_bucket = [uniforms[i] for i in bucket] if uniforms is not None else None
        outs = gpt.infer_batch(segs, uniforms=u_bucket, **kwargs)
        for i, out in zip(bucket, outs):
            results[i] = out
    return results
