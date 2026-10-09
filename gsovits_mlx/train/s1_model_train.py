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

NEG_INF = -1e4
# NOTE: NOT -np.inf — the mask is cast to fp16 for the forward (official
# autocast materializes it in x.dtype) and -inf overflows fp16 to NaN,
# silently poisoning every masked SDPA position (found via probe nan
# losses, 2026-10-09). -1e4 is the standard fp16-safe additive floor:
# exp(-1e4 * 1/sqrt(64)) == 0 in fp16 softmax.


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
    out[mask] = NEG_INF
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

    use_gradient_checkpoint (default from GSOVITS_S1_GRAD_CKPT, '0'): two-pass
    checkpointed forward/backward — the forward retains only the block
    INPUT h per layer (per-layer QKV/attn/MLP/score buffers are dropped)
    and the backward recomputes one block at a time. Implemented MANUALLY
    (train_loss_and_grads): mlx 0.32.2's mx.custom_function segfaults on
    pytree/multi-arg vjps (verified on CPU: list args and 4+ explicit array
    args both crash; the pin is frozen per AGENTS.md so we cannot upgrade).
    The VJP reductions are computed in fp32 ((out*g).sum() with fp32 cast);
    ckpt vs non-ckpt grads are NOT bit-identical (fp16 grad tail + extra
    reduction) — the parity tests use the non-checkpointed default.
    Peak-memory lever for the long bucket (src~617): the non-checkpointed
    path transiently spiked whole-process phys_footprint to 13.96GB (bs=4).
    """

    _BLOCK_KEYS = ("qkv_w", "qkv_b", "out_w", "out_b", "norm1.g", "norm1.b",
                   "mlp1.w", "mlp1.b", "mlp2.w", "mlp2.b",
                   "norm2.g", "norm2.b")
    _HEAD_KEYS = ("ar_text_embedding", "ar_audio_embedding",
                  "bert_proj.weight", "bert_proj.bias",
                  "ar_text_position.alpha", "ar_audio_position.alpha")

    # -- checkpointed two-pass forward/backward -------------------------------
    def _embed_head(self, p16: dict, batch: dict):
        """x/y embeddings + bert_proj + positional — returns (h0, x_len,
        targets_np, y_in). Same math as forward()'s pre-block stack."""
        phones = mx.array(batch["phoneme_ids"].astype(np.int32))
        x_lens = np.asarray(batch["phoneme_ids_len"], dtype=np.int64)
        y_np = np.asarray(batch["semantic_ids"], dtype=np.int64)
        y_lens = np.asarray(batch["semantic_ids_len"], dtype=np.int64)
        B, X = phones.shape
        Y = y_np.shape[1]
        x_len = int(x_lens.max())
        y_len = int(y_lens.max())
        bert = mx.array(np.asarray(batch["bert_feature"], dtype=np.float32))
        x = p16["ar_text_embedding"][phones]  # (B, X, D)
        x = x + (mx.transpose(bert, (0, 2, 1)) @ p16["bert_proj.weight"].T
                 + p16["bert_proj.bias"])
        x = self._pos(p16["ar_text_position.alpha"], x)
        y_mask_int = (np.arange(Y)[None, :] >= y_lens[:, None]).astype(np.int64)
        codes = y_np * (1 - y_mask_int)
        y_in, targets_np = pad_y_eos_np(codes, y_mask_int, self.EOS)
        y_emb = p16["ar_audio_embedding"][mx.array(y_in.astype(np.int32))]
        y_pos = self._pos(p16["ar_audio_position.alpha"], y_emb)
        h = mx.concatenate([x, y_pos], axis=1)
        return h, x_len, targets_np

    def _predict_head(self, h, p16: dict, x_len: int, targets_np: np.ndarray):
        """Official slice + logits + fp32 CE-sum over (B, Y+1)."""
        dec = h[:, x_len - 1:]
        logits = dec @ p16["ar_predict_layer.weight"].T
        logits32 = logits.astype(mx.float32)
        targets = mx.array(targets_np.astype(np.int32))
        logp = nn.log_softmax(logits32, axis=-1)
        tgt_logp = mx.take_along_axis(logp, targets[:, :, None],
                                      axis=-1)[..., 0]
        loss = -tgt_logp.sum()
        return loss, logits32, targets

    def train_loss_and_grads(self, params: dict, batch: dict):
        """Checkpointed forward+backward, no custom_function.

        Pass 1 (eager): h0 = embed head (eval'd, graph severed); blocks run
        one at a time keeping ONLY h_i inputs; loss from the small predict
        head graph.
        Pass 2 (backward): grad of the predict head wrt (h_L, predict
        params); then per block i=L-1..0 grad of (block(h_i, p_i) *
        g_{i+1}).astype(fp32).sum() wrt (h_i, p_i) — one block live at a
        time; finally grad of (embed(h0 params) * g_0).sum() for the head.
        Returns (loss_f, acc, grads {name: fp32}).
        """
        dtype = self.dtype
        p16 = {k: (v.astype(dtype) if v.dtype != dtype else v)
               for k, v in params.items()}

        h0, x_len, targets_np = self._embed_head(p16, batch)
        mx.eval(h0)
        h = h0
        hs = [h0]
        attn_mask = self._mask_for(batch, dtype)
        for i in range(self.num_layers):
            pre = f"block.{i}."
            pt = {k: p16[pre + k] for k in self._BLOCK_KEYS}
            h = self._block_core(pt, h, attn_mask)
            mx.eval(h)
            hs.append(h)

        loss, logits32, targets = self._predict_head(h, p16, x_len, targets_np)

        # -- backward: predict head --
        pred_keys = ["ar_predict_layer.weight"]

        def head_loss(hh, w):
            l, _, _ = self._predict_head(hh, {"ar_predict_layer.weight": w},
                                         x_len, targets_np)
            return l

        hv = mx.value_and_grad(head_loss, argnums=(0, 1))(
            h, p16["ar_predict_layer.weight"])
        g_loss, (g_h, g_w) = hv[0], hv[1]
        mx.eval(g_loss, g_h, g_w)
        loss_f = float(g_loss)
        grads = {"ar_predict_layer.weight": g_w.astype(mx.float32)}
        acc = top3_accuracy(mx.transpose(logits32, (0, 2, 1)), targets,
                            self.EOS)

        # -- backward: blocks, one at a time --
        g = g_h
        for i in range(self.num_layers - 1, -1, -1):
            pre = f"block.{i}."
            pt = [p16[pre + k] for k in self._BLOCK_KEYS]

            def red(hh, *tt, _i=i):
                out = self._block_core(dict(zip(self._BLOCK_KEYS, tt)), hh,
                                       attn_mask)
                return (out.astype(mx.float32) * g.astype(mx.float32)).sum()

            res = mx.grad(red, argnums=tuple(range(len(pt) + 1)))(
                hs[i], *pt)
            g = res[0]
            for k, key in enumerate(self._BLOCK_KEYS):
                grads[pre + key] = res[k + 1].astype(mx.float32)
            mx.eval(g, *(grads[pre + key] for key in self._BLOCK_KEYS))

        # -- backward: embed head --
        head_keys = list(self._HEAD_KEYS)

        def head_red(ph):
            psub = {k: ph[k] for k in head_keys}
            hh, _, _ = self._embed_head(psub, batch)
            return (hh.astype(mx.float32) * g.astype(mx.float32)).sum()

        hg = mx.grad(head_red)({k: p16[k] for k in head_keys})
        for k in head_keys:
            grads[k] = hg[k].astype(mx.float32)
        mx.eval(*(grads[k] for k in head_keys))
        return loss_f, acc, grads

    def _mask_for(self, batch, dtype):
        X = batch["phoneme_ids"].shape[1]
        Y = batch["semantic_ids"].shape[1]
        return mx.array(
            build_train_attn_mask(
                np.asarray(batch["phoneme_ids_len"]),
                np.asarray(batch["semantic_ids_len"]), X, Y, self.num_head),
            dtype=dtype)

    def __init__(self, config: dict, dropout: float | None = None,
                 dtype=None, use_checkpoint: bool | None = None):
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
        if use_checkpoint is None:
            use_checkpoint = os.environ.get("GSOVITS_S1_GRAD_CKPT", "0") == "1"
        self.use_gradient_checkpoint = use_checkpoint
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

    def _block_core(self, pt: dict, x: mx.array,
                    attn_mask: mx.array) -> mx.array:
        """One post-norm TransformerEncoderLayer from a SHORT-KEY param dict
        (official _sa/_ff blocks, dropout 0)."""
        d = self.model_dim
        H = self.num_head
        hd = self.head_dim
        b, t, _ = x.shape

        qkv = x @ pt["qkv_w"].T + pt["qkv_b"]
        q, k, v = qkv[..., :d], qkv[..., d:2 * d], qkv[..., 2 * d:]

        def split(t4):
            return t4.reshape(b, t4.shape[1], H, hd).transpose(0, 2, 1, 3)

        qh, kh, vh = split(q), split(k), split(v)
        # (B, 1, src, src) broadcasts over heads in SDPA (official semantic)
        o = mx.fast.scaled_dot_product_attention(
            qh, kh, vh, scale=1.0 / math.sqrt(hd), mask=attn_mask)
        o = o.transpose(0, 2, 1, 3).reshape(b, t, d)
        o = o @ pt["out_w"].T + pt["out_b"]
        x = mx.fast.layer_norm(x + o, pt["norm1.g"], pt["norm1.b"], 1e-5)
        h1 = mx.maximum(x @ pt["mlp1.w"].T + pt["mlp1.b"], 0.0)
        h2 = h1 @ pt["mlp2.w"].T + pt["mlp2.b"]
        x = mx.fast.layer_norm(x + h2, pt["norm2.g"], pt["norm2.b"], 1e-5)
        return x

    def _block_fwd(self, p16: dict, i: int, x: mx.array,
                   attn_mask: mx.array) -> mx.array:
        pre = f"block.{i}."
        pt = {k: p16[pre + k] for k in self._BLOCK_KEYS}
        return self._block_core(pt, x, attn_mask)

    def _block_ckpt_removed():  # noqa: D401 - see train_loss_and_grads
        """mx.custom_function checkpointing was removed: mlx 0.32.2
        segfaults on pytree/multi-arg vjps (verified 2026-10-09, CPU):
        list args -> bus error; 4 explicit array args -> scrambled grad
        shapes then segfault. The manual two-pass implementation lives in
        train_loss_and_grads."""
