"""s1 (AR/GPT) training forward: official forward_old ported to MLX.

Ports AR/models/t2s_model.py Text2SemanticDecoder.forward_old (the
cuda_graph reference, lines ~450-510) as a training-mode forward over the
INFERENCE weight layout (the gpt.safetensors keys: ar_text_embedding,
block.<i>.qkv_w/..., bert_proj.weight, ar_predict_layer.weight, ...) —
same weights as gsovits_mlx/gpt/t2s.py loads, no KV-cache streaming,
differentiable via mx.grad over a flat params dict.

Official semantics preserved exactly:

* x block: ar_text_embedding(x) + bert_proj(bert^T) + ar_text_position.
* y block: codes = y * (1 - y_mask) (pads zeroed); y, targets = pad_y_eos(
  codes, y_mask_int, eos=1024): decoder input y = targets[:, :-1] (starts
  with the row's first real code; EOS at every pad slot), targets = codes
  + eos*mask extended with a trailing EOS column — every target row ends
  with EOS.
* attention mask: concat[x_attn_mask (x rows see all x, never y),
  y_attn_mask (y rows see all x + causal triu(1) within y)] logical_or
  padding (x: make_pad_mask_LEFT, y: make_pad_mask) broadcast over
  (B*heads, 1, src) -> additive -inf float, built in fp16 (official
  autocast materializes the mask in x.dtype).
* logits = ar_predict_layer(xy_dec[:, x_len-1:]) with x_len = x_lens.max():
  a (B, Y+1, D) slice — the LAST x position plus every y position. The
  last-x position (which has attended only x) predicts the first y token;
  y position j predicts target j+1; the final y position predicts EOS.
  This is why the bucket sampler matters: rows share x_len (max) and the
  per-row pad tails land on EOS targets.
* loss = F.cross_entropy(logits.permute(0,2,1), targets, reduction="sum")
  over ALL (B, Y+1) positions; acc = MulticlassAccuracy top_k=3,
  ignore_index=EOS, multidim_average="global".

Numerics: fp16 forward + fp32 CE (logits cast to fp32 — official 16-mixed
autocast keeps CE fp32). Weights stay fp32 (masters); the trainer casts to
fp16 per forward (S1Trainer/train_s1.py).

Dropout: the official TRAIN stack would run dropout 0.1 (positional
embeddings + per-layer), config dropout=0 covers only TokenEmbedding.
This port defaults to p_dropout=0 (task spec; identical math in eval);
GSOVITS_S1_TRAIN_DROPOUT>0 is an experimentation knob.

Left-pad quirk (verified against official make_pad_mask_left): rows are
right-padded by collate, but x_mask marks the row's FIRST (max_len -
x_len) positions — the official mask hides leading slots while the actual
pads sit at the tail. Replicated bit-for-bit (mask test vs torch).
"""

from __future__ import annotations

import math
import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np

__all__ = ["S1TrainModel", "make_pad_mask_left_np", "make_pad_mask_np",
           "build_train_attn_mask", "top3_accuracy", "pad_y_eos_np"]

NEG_INF = -1e9


def make_pad_mask_np(lengths, max_len: int = 0) -> np.ndarray:
    """Official make_pad_mask: True at pad positions (right)."""
    lengths = np.asarray(lengths)
    max_len = max(max_len, int(lengths.max()))
    seq_range = np.arange(max_len)
    return seq_range[None, :] >= lengths[:, None]  # (B, T)


def make_pad_mask_left_np(lengths, max_len: int = 0) -> np.ndarray:
    """Official make_pad_mask_left: True at the row's FIRST (max-len) slots."""
    lengths = np.asarray(lengths)
    max_len = max(max_len, int(lengths.max()))
    n = lengths.shape[0]
    seq_range = np.arange(max_len)[None, :].repeat(n, axis=0)
    seq_range = seq_range - (max_len - lengths)[:, None]
    return seq_range < 0  # (B, T)


def pad_y_eos_np(codes: np.ndarray, y_mask_int: np.ndarray, eos: int):
    """Official pad_y_eos: (y_in, targets) = (tgt[:, :-1], tgt) where
    tgt = pad(codes, (0,1), 0) + eos * pad(y_mask_int, (0,1), 1)."""
    B, Y = codes.shape
    tgt = np.zeros((B, Y + 1), dtype=np.int64)
    tgt[:, :Y] = codes + eos * y_mask_int
    tgt[:, Y] = eos
    return tgt[:, :-1].copy(), tgt


def build_train_attn_mask(x_lens, y_lens, X: int, Y: int,
                          num_head: int) -> np.ndarray:
    """(B, 1, src, src) float32 additive mask, official layout.

    Official construction: xy_attn (src, src) = [x rows see only x;
    y rows see all x + causal y] logical_or padding row (x left-pad
    quirk, y right-pad) — torch broadcasting expands the (B*H,1,src)
    padding to the FULL (B*H, src, src) grid, so y queries DO attend
    causal y columns (verified: collapsing to one row breaks parity at
    every y position). We return the per-sample 2D grid (B,1,src,src);
    callers broadcast over heads (SDPA wants (B,H,L,S)).
    """
    B = len(x_lens)
    x_pad = make_pad_mask_left_np(x_lens, X)  # (B, X)
    y_pad = np.arange(Y)[None, :] >= np.asarray(y_lens)[:, None]  # (B, Y)
    xy_pad = np.concatenate([x_pad, y_pad], axis=1)  # (B, src)

    x_attn = np.concatenate(
        [np.zeros((X, X), dtype=bool), np.ones((X, Y), dtype=bool)], axis=1)
    y_attn = np.concatenate(
        [np.zeros((Y, X), dtype=bool),
         np.triu(np.ones((Y, Y), dtype=bool), 1)], axis=1)
    causal = np.concatenate([x_attn, y_attn], axis=0)  # (src, src)

    mask = causal[None, :, :] | xy_pad[:, None, :]  # (B, src, src)
    out = np.zeros(mask.shape, dtype=np.float32)
    out[mask] = -np.inf
    return out[:, None, :, :]  # (B, 1, src, src)


def top3_accuracy(logits: mx.array, targets: mx.array,
                  eos: int = 1024) -> float:
    """MulticlassAccuracy(top_k=3, ignore_index=EOS,
    multidim_average="global").item(): hits / non-EOS count.

    logits (B, V, T) fp32, targets (B, T) int.
    """
    top3 = mx.argsort(logits, axis=1)[:, -3:, :]  # (B, 3, T)
    hit = (top3 == targets[:, None, :]).any(axis=1)  # (B, T)
    keep = targets != eos
    hits = (hit * keep).sum()
    total = keep.sum()
    return float(hits / mx.maximum(total, 1))


class S1TrainModel:
    """Training wrapper over the inference-layout s1 weight dict.

    params: {name: mx.array} in gpt.safetensors layout (fp32 masters).
    forward() is pure-functional over ``params`` so mx.grad works.
    """

    def __init__(self, config: dict, dropout: float | None = None,
                 dtype=None):
        m = config["model"]
        self.model_dim = m["hidden_dim"]
        self.embedding_dim = m["embedding_dim"]
        self.num_head = m["head"]
        self.num_layers = m["n_layer"]
        self.vocab_size = m["vocab_size"]
        self.phoneme_vocab_size = m["phoneme_vocab_size"]
        self.EOS = m["EOS"]
        self.head_dim = self.model_dim // self.num_head
        if dropout is None:
            dropout = float(os.environ.get("GSOVITS_S1_TRAIN_DROPOUT", "0"))
        self.dropout = dropout
        # forward dtype: fp16 (official 16-mixed) or fp32 (parity runs)
        if dtype is None:
            import mlx.core as _mx
            dtype = _mx.float16
        self.dtype = dtype
        # sin/cos positional table (official SinePositionalEmbedding,
        # scale=False alpha=True; alpha is a LEARNED scalar param in params)
        position = np.arange(0, 4096, dtype=np.float32)[:, None]
        div_term = np.exp(
            np.arange(0, self.embedding_dim, 2, dtype=np.float32)
            * -(math.log(10000.0) / self.embedding_dim))
        pe = np.zeros((4096, self.embedding_dim), dtype=np.float32)
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self._pe = mx.array(pe)[None]  # (1, 4096, D) fp32

    # -- parameters ------------------------------------------------------------
    def parameters(self) -> dict:
        return self.params

    def load(self, params: dict) -> "S1TrainModel":
        self.params = params
        return self

    # -- functional helpers ------------------------------------------------------
    def _pos(self, alpha, x: mx.array) -> mx.array:
        """SinePositionalEmbedding.forward: x + alpha * pe[:T] (scale=False)."""
        T = x.shape[1]
        pe = self._pe[:, :T].astype(x.dtype)
        return x + alpha.astype(x.dtype) * pe

    # -- forward -------------------------------------------------------------------
    def forward(self, params: dict, batch: dict, *, return_logits: bool = False):
        """forward_old over one collated batch.

        batch: collate() output (numpy arrays). Returns (loss_sum fp32,
        acc float[, logits fp32 (B, Y+1, V) if return_logits]).
        """
        phones = mx.array(batch["phoneme_ids"].astype(np.int32))
        x_lens = np.asarray(batch["phoneme_ids_len"], dtype=np.int64)
        y_np = np.asarray(batch["semantic_ids"], dtype=np.int64)
        y_lens = np.asarray(batch["semantic_ids_len"], dtype=np.int64)

        B, X = phones.shape
        Y = y_np.shape[1]
        x_len = int(x_lens.max())
        y_len = int(y_lens.max())
        assert X == x_len and Y == y_len, "collate must pad to the max lens"

        # ---- cast working weights to the forward dtype (16-mixed default) ----
        dtype = self.dtype
        p16 = {k: (v.astype(dtype) if v.dtype != dtype else v)
               for k, v in params.items()}

        # ---- x block ----
        bert = mx.array(np.asarray(batch["bert_feature"], dtype=np.float32))
        x = p16["ar_text_embedding"][phones]  # (B, X, D)
        x = x + (mx.transpose(bert, (0, 2, 1)) @ p16["bert_proj.weight"].T
                 + p16["bert_proj.bias"])
        x = self._pos(p16["ar_text_position.alpha"], x)

        # ---- y block: codes / pad_y_eos ----
        y_mask_int = (np.arange(Y)[None, :] >= y_lens[:, None]).astype(np.int64)
        codes = y_np * (1 - y_mask_int)
        y_in, targets_np = pad_y_eos_np(codes, y_mask_int, self.EOS)

        y_emb = p16["ar_audio_embedding"][mx.array(y_in.astype(np.int32))]
        y_pos = self._pos(p16["ar_audio_position.alpha"], y_emb)

        # ---- attention mask: (B, 1, src, src) additive, broadcast over heads ----
        attn_mask = mx.array(
            build_train_attn_mask(x_lens, y_lens, X, Y, self.num_head),
            dtype=dtype)

        # ---- transformer stack (post-norm) ----
        h = mx.concatenate([x, y_pos], axis=1)
        for i in range(self.num_layers):
            h = self._block_fwd(p16, i, h, attn_mask)

        # official slice: LAST x position + all y positions -> (B, Y+1, D)
        dec = h[:, x_len - 1:]
        logits = dec @ p16["ar_predict_layer.weight"].T  # (B, Y+1, V)
        logits32 = logits.astype(mx.float32)
        targets = mx.array(targets_np.astype(np.int32))  # (B, Y+1)

        # cross-entropy SUM over all (B, Y+1) positions (official reduction:
        # "duration越长, 梯度更新也应该更多, 所以用 sum")
        logp = nn.log_softmax(logits32, axis=-1)
        tgt_logp = mx.take_along_axis(logp, targets[:, :, None], axis=-1)[..., 0]
        loss = -tgt_logp.sum()

        acc = top3_accuracy(mx.transpose(logits32, (0, 2, 1)), targets, self.EOS)
        if return_logits:
            return loss, acc, logits32
        return loss, acc

    def _block_fwd(self, p16: dict, i: int, x: mx.array,
                   attn_mask: mx.array) -> mx.array:
        """One post-norm TransformerEncoderLayer (official _sa/_ff blocks,
        dropout 0)."""
        pre = f"block.{i}."
        d = self.model_dim
        H = self.num_head
        hd = self.head_dim
        b, t, _ = x.shape

        qkv = x @ p16[pre + "qkv_w"].T + p16[pre + "qkv_b"]
        q, k, v = qkv[..., :d], qkv[..., d:2 * d], qkv[..., 2 * d:]

        def split(t4):
            return t4.reshape(b, t4.shape[1], H, hd).transpose(0, 2, 1, 3)

        qh, kh, vh = split(q), split(k), split(v)
        # (B, 1, src, src) broadcasts over heads in SDPA (official semantic)
        o = mx.fast.scaled_dot_product_attention(
            qh, kh, vh, scale=1.0 / math.sqrt(hd), mask=attn_mask)
        o = o.transpose(0, 2, 1, 3).reshape(b, t, d)
        o = o @ p16[pre + "out_w"].T + p16[pre + "out_b"]
        x = mx.fast.layer_norm(x + o, p16[pre + "norm1.g"],
                               p16[pre + "norm1.b"], 1e-5)
        h1 = mx.maximum(x @ p16[pre + "mlp1.w"].T + p16[pre + "mlp1.b"], 0.0)
        h2 = h1 @ p16[pre + "mlp2.w"].T + p16[pre + "mlp2.b"]
        x = mx.fast.layer_norm(x + h2, p16[pre + "norm2.g"],
                               p16[pre + "norm2.b"], 1e-5)
        return x
