"""s2 CFM training (SynthesizerTrnV3.forward + CFM.forward port) in MLX.

Ports, from the official cuda_graph repo (READ-ONLY reference
/Users/baicai1145/repos/gpt-sovits/GPT-SoVITS-cuda_graph_accel_v5):

* ``module/models.py SynthesizerTrnV3.forward`` (L1305-1333): fp32 island —
  y_mask -> ge = ref_enc(y[:, :704] * y_mask, y_mask); freeze_quantizer=True
  path runs ssl_proj + quantizer + enc_p under no_grad (enc_p IS frozen too:
  __init__ L1300-1303 set_no_grad(ssl_proj)/set_no_grad(quantizer)/
  set_no_grad(enc_p)); quantized nearest x2; fea = bridge(x) (Sequential
  Conv1d + nn.LeakyReLU() -> negative_slope 0.01) -> nearest interp x1.875
  (v3) / x2 (v4/v5) -> wns1 Encoder(512,512,512,k5,d1,L8) with ge at
  mel_lengths; prompt_len = floor(rand(B) * mel_lengths*2/3); mel/fea
  truncated to min length; CFM loss.
* ``module/models.py CFM.forward`` (L1174-1207): per-sample t~U(0,1),
  x0~N(0,1), vt = x1-x0, xt = x0 + t*vt; prompt zeroing per sample; the 30%
  two-step branch (python ``random.random() < 0.3``; base = torch.randint(
  2, 8, (b,)) — NOTE torch.randint's high is EXCLUSIVE so per-sample bases
  are integers in [2, 8); d = 2^-base; d_input = d masked <1e-2 -> 0;
  v1/v2 estimator calls DETACHED (no_grad); vt = (v1+v2)/2 detached;
  dt = 2*d); final estimator call at (xt, t, dt) WITH grad; loss =
  mean over batch of MSE(vt_pred[i,:,prompt_i:mel_len_i],
  vt[i,:,prompt_i:mel_len_i]) (torch MSELoss = mean over ALL elements).
* DiT training forward (f5_tts/model/backbones/dit.py DiT.forward, the
  non-infer path): text_embed(text)+pos -> InputEmbedding(x, cond=prompt,
  text_embed) -> blocks (AdaLN from t + d_embed(dt) when use_step_embedding)
  -> norm_out -> proj_out. Head-0-only RoPE and the trailing mask multiply
  are quirks of this checkpoint family already implemented (and inference-
  verified) in gsovits_mlx/sovits/dit.py; the trainer calls
  ``estimator(x0, cond0, x_lens, time, d, mu, use_grad_ckpt=False)`` with
  infer=False which exercises exactly that layout.

v5 training finding (documented per task): official v5 s2 training is
BROKEN upstream — webui.py L1498 training-version Radio choices are
["v1","v2","v4","v2Pro","v2ProPlus"] (no v5dev/v5turbo), and
SynthesizerTrnV3.__init__ L1295-1299 unconditionally replaces self.cfm with
module/v5_inference.CFMV5 (an @inference_mode inference-only class with no
forward), so s2_train_v3_lora.py's ``net_g(...)`` call would TypeError for
v5. The estimator DiT is identical for v5 (only use_step_embedding=False),
so this trainer trains v5dev/v5turbo with the plain CFM loss path (same as
v3/v4), matching what the official code builds before its training-breaking
CFMV5 swap.

LoRA: gsovits_mlx.train.lora creates rank-32 adapters on every Attention
to_k/to_q/to_v/to_out.0 (official peft targets; alpha=r -> scaling 1.0).
During training each wrapped projection computes
``x @ W.T + b + (x @ A.T) @ B.T``. The BASE DiT weights are frozen (peft
marks them requires_grad=False); full-model training continues on ref_enc /
bridge / wns1 / linear_mel as in the official trainer (peft wraps ONLY
net_g.cfm).
"""

from __future__ import annotations

import math
import random

import mlx.core as mx
import mlx.nn as nn

from ..sovits.models_v1v2 import _nearest_interp
from ..sovits.models_v3 import SynthesizerTrnV3
from .lora import LoRALinear

__all__ = ["CFMTrainingLoss", "S2V3TrainModel", "dit_lora_forward",
           "FROZEN_PARAM_PREFIXES", "FULL_TRAIN_PARAM_PREFIXES"]

# official freeze set (freeze_quantizer=True): ssl_proj + quantizer + enc_p
FROZEN_PARAM_PREFIXES = ("ssl_proj.", "enc_p.")
# non-LoRA full-model trainer trains everything except the frozen set
# (NOTE linear_mel is UNUSED by SynthesizerTrnV3.forward — only the V3b
# variant consumes it; it stays in the trainable set to mirror the official
# optimizer, which simply never sees a grad for it)
FULL_TRAIN_PARAM_PREFIXES = ("ref_enc.", "bridge_0.", "wns1.", "linear_mel.",
                             "dit.")
# LoRA trainer: full-model on the trunk + adapters only in the DiT
LORA_FULL_TRAIN_PARAM_PREFIXES = ("ref_enc.", "bridge_0.", "wns1.",
                                  "linear_mel.")


# ---------------------------------------------------------------------------
# DiT training forward with LoRA deltas on q/k/v/out projections
# ---------------------------------------------------------------------------

def dit_lora_forward(dit, adapters: dict[str, LoRALinear], xt, prompt, x_lens,
                     t, d, mu) -> mx.array:
    """Training-layout DiT forward (f5_tts dit.py non-infer path) with LoRA.

    adapters: {"<block>.attn.<suffix>": LoRALinear}; missing keys = no
    adapter on that projection. Math per projection:
      y = x @ W.T + b + (x @ A.T) @ B.T   (scaling = alpha/r = 1.0)
    Everything else mirrors dit.DiT.__call__(infer=False, use_grad_ckpt=False)
    exactly (same ops, same order, same dtypes).
    """
    x = mx.transpose(xt, (0, 2, 1))       # (B, T, C_mel)
    cond = mx.transpose(prompt, (0, 2, 1))
    text = mx.transpose(mu, (0, 2, 1))
    seq_len = x.shape[1]
    mask = dit_sequence_mask(x_lens, seq_len).astype(x.dtype)

    # t = time_embed(time); dt = d_embed(dt_base); t += dt  (v3/v4 only:
    # use_step_embedding; v5 DiT has no d_embed)
    t0w = dit.time_embed.time_mlp_0_w
    td = t0w.dtype
    t_emb = dit.time_embed(t.astype(td))
    if dit.use_step_embedding:
        d0w = dit.d_embed.time_mlp_0_w
        t_emb = t_emb + dit.d_embed(d.astype(d0w.dtype))

    text_embed = dit.text_embed(text, seq_len, drop_text=False)
    x = dit.input_embed(x, cond, text_embed, drop_audio_cond=False)
    t_emb = t_emb.astype(x.dtype)

    rope = dit._rope(seq_len, x.dtype)

    from ..sovits import dit as dit_mod

    for block in dit.transformer_blocks:
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.attn_norm(x, t_emb)
        attn = block.attn
        # -- attention with LoRA (same math as dit.Attention.__call__) --
        b, tlen, _ = norm.shape
        a_q = adapters.get(f"{bi}.to_q") if False else adapters.get(
            _attn_adapter_name(dit, block, attn, "to_q"))
        a_k = adapters.get(_attn_adapter_name(dit, block, attn, "to_k"))
        a_v = adapters.get(_attn_adapter_name(dit, block, attn, "to_v"))
        a_o = adapters.get(_attn_adapter_name(dit, block, attn, "to_out.0"))
        qw, qb = attn._qkv_w()
        qkv = norm @ qw.T + qb
        inner = qw.shape[0] // 3
        if a_q is not None or a_k is not None or a_v is not None:
            parts = []
            for a in (a_q, a_k, a_v):
                if a is None:
                    parts.append(mx.zeros((b, tlen, inner), dtype=qkv.dtype))
                else:
                    parts.append((norm @ a.lora_A.T) @ a.lora_B.T)
            qkv = qkv + mx.concatenate(parts, axis=-1)
        q, k, v = mx.split(qkv, 3, axis=-1)
        q_h = q.reshape(b, tlen, attn.heads, attn.dim_head).transpose(0, 2, 1, 3)
        k_h = k.reshape(b, tlen, attn.heads, attn.dim_head).transpose(0, 2, 1, 3)
        v_h = v.reshape(b, tlen, attn.heads, attn.dim_head).transpose(0, 2, 1, 3)
        q_h = mx.concatenate([dit_mod.apply_rope(q_h[:, :1], rope), q_h[:, 1:]], axis=1)
        k_h = mx.concatenate([dit_mod.apply_rope(k_h[:, :1], rope), k_h[:, 1:]], axis=1)
        out = mx.fast.scaled_dot_product_attention(
            q_h, k_h, v_h, scale=1.0 / math.sqrt(attn.dim_head),
            mask=mask[:, None, None, :].astype(q_h.dtype)).astype(q_h.dtype)
        out = out.transpose(0, 2, 1, 3).reshape(b, tlen, attn.heads * attn.dim_head)
        attn_input = out  # to_out input
        out = attn_input @ attn.to_out_0_w.T + attn.to_out_0_b
        if a_o is not None:
            out = out + (attn_input @ a_o.lora_A.T) @ a_o.lora_B.T
        out = out * mask[:, :, None]
        x = x + gate_msa[:, None] * out

        norm = dit_mod._modulated_norm(x, scale_mlp, shift_mlp)
        ff_out = block.ff(norm)
        x = x + gate_mlp[:, None] * ff_out

    x = dit.norm_out(x, t_emb)
    return x @ dit.proj_out_w.T + dit.proj_out_b


def dit_sequence_mask(x_lens, max_length: int) -> mx.array:
    ids = mx.arange(max_length)
    return (ids[None, :] < mx.asarray(x_lens)[:, None]).astype(mx.float32)


def _attn_adapter_name(dit, block, attn, suffix: str) -> str:
    """Adapter key for this attention's projection; relies on the
    S2V3TrainModel-annotated ``_lora_prefix`` (falls back to index lookup)."""
    prefix = getattr(attn, "_lora_prefix", None)
    if prefix is None:
        for bi, blk in enumerate(dit.transformer_blocks):
            if blk is block:
                prefix = f"dit.blocks.{bi}.attn"
                break
    return f"{prefix}.{suffix}"


# ---------------------------------------------------------------------------
# CFM training loss (models.py CFM.forward)
# ---------------------------------------------------------------------------

class CFMTrainingLoss:
    """CFM.forward port. x1/mu: (B, C, T); returns fp32 scalar loss."""

    def __init__(self, estimator, rng: random.Random | None = None):
        self.estimator = estimator  # callable(xt, prompt, x_lens, t, d, mu)
        self.rng = rng or random

    def __call__(self, x1: mx.array, x_lens, prompt_lens, mu: mx.array,
                 key: mx.array | None = None) -> tuple[mx.array, dict]:
        b = x1.shape[0]
        dtype = x1.dtype
        if key is None:
            key = mx.random.key(random.getrandbits(63))
        key, sub = mx.random.split(key)
        t = mx.random.uniform(0.0, 1.0, (b,), key=key).astype(dtype)
        x0 = mx.random.normal(x1.shape, key=sub).astype(dtype)
        vt = x1 - x0
        xt = x0 + t[:, None, None] * vt
        dt = mx.zeros((b,), dtype=dtype)
        prompt = mx.zeros_like(x1)

        pl = [int(p) for p in prompt_lens]
        for i in range(b):
            if pl[i] > 0:
                prompt[i, :, : pl[i]] = x1[i, :, : pl[i]]
                xt[i, :, : pl[i]] = 0.0

        info = {"two_step": False}
        if self.rng.random() < 0.3:
            info["two_step"] = True
            # official torch.randint(2, 8, (b,)) — high EXCLUSIVE -> [2, 8)
            base = mx.array([self.rng.randint(2, 7) for _ in range(b)],
                            dtype=dtype)
            d = 1.0 / mx.power(2.0, base)
            d_input = mx.where(d < 1e-2, mx.zeros_like(d), d)
            # detached estimator calls (torch .detach()): stop_gradient
            # severs the graph; eval realizes the arrays so memory does not
            # accumulate across the two extra estimator calls. The estimator
            # consumes (B, C, T) xt and returns (B, T, C); the official code
            # transposes each output back to (B, C, T) before x_mid / vt.
            v1 = mx.stop_gradient(mx.transpose(
                self.estimator(xt, prompt, x_lens, t, d_input, mu), (0, 2, 1)))
            mx.eval(v1)
            x_mid = xt + d[:, None, None] * v1
            v2 = mx.stop_gradient(mx.transpose(
                self.estimator(x_mid, prompt, x_lens, t + d, d_input, mu),
                (0, 2, 1)))
            mx.eval(v2)
            vt = mx.stop_gradient((v1 + v2) / 2)
            mx.eval(vt)
            dt = 2 * d
            # vt back to estimator layout for the loss slice below is not
            # needed: loss slices vt in (B, C, T) like the official code

        vt_pred = mx.transpose(
            self.estimator(xt, prompt, x_lens, t, dt, mu), (0, 2, 1))
        # loss = mean over batch of MSE(vt_pred[i,:,p_i:L_i], vt[i,:,p_i:L_i])
        # torch MSELoss = SUM of squared error over ALL elements / N_elem;
        # the per-sample divisor is (C * (L_i - p_i)). NOTE: when samples have
        # different (L_i - p_i) the torch code divides each sample's MSE by N_i
        # (per-sample mean), then / b — replicated exactly here.
        total = mx.zeros((), dtype=mx.float32)
        for i in range(b):
            lo, hi = pl[i], int(x_lens[i])
            if hi <= lo:
                continue
            pred = vt_pred[i, :, lo:hi].astype(mx.float32)
            tgt = vt[i, :, lo:hi].astype(mx.float32)
            total = total + mx.sum((pred - tgt) ** 2) / pred.size
        return total / b, info


# ---------------------------------------------------------------------------
# Trainable model wrapper
# ---------------------------------------------------------------------------

class S2V3TrainModel:
    """Trainable view over the inference SynthesizerTrnV3.

    Trainable params (flat {name: fp32 array}):
      LoRA trainer (official): ref_enc.*/bridge_0.*/wns1.*/linear_mel.* +
      <dit...>.lora_A/.lora_B adapters; the DiT BASE weights + ssl_proj +
      enc_p + quantizer are frozen.
      Full trainer (--no-lora): everything except ssl_proj.*/enc_p.*.

    forward(batch) = official SynthesizerTrnV3.forward with
    freeze_quantizer=True semantics, returning (loss, info).
    """

    def __init__(self, model: SynthesizerTrnV3, version: str,
                 lora_adapters: list[LoRALinear] | None = None,
                 rng: random.Random | None = None):
        self.model = model
        self.version = version
        self.adapters = list(lora_adapters or [])
        self.use_lora = bool(self.adapters)
        self.adapter_by_name = {a.name: a for a in self.adapters}
        self.rng = rng or random
        # annotate attention modules with their LoRA prefix so
        # dit_lora_forward can look adapters up
        for bi, block in enumerate(model.cfm.estimator.transformer_blocks):
            block.attn._lora_prefix = f"dit.blocks.{bi}.attn"

    # -- parameter plumbing ---------------------------------------------------
    def trainable_params(self) -> dict:
        from mlx.utils import tree_flatten
        tree = dict(tree_flatten(self.model.parameters()))
        params = {}
        if self.use_lora:
            prefixes = LORA_FULL_TRAIN_PARAM_PREFIXES
        else:
            prefixes = FULL_TRAIN_PARAM_PREFIXES
        for name, arr in tree.items():
            if name.startswith(FROZEN_PARAM_PREFIXES):
                continue
            if name.startswith(prefixes) or name.startswith("dit.") \
                    and not self.use_lora:
                params[name] = arr.astype(mx.float32)
        if self.use_lora:
            for a in self.adapters:
                params.update(a.params())
        return params

    def apply_params(self, params: dict) -> None:
        """Write updated params back into the model/adapters.

        Adapter params update the LoRALinear objects; model weights are
        written through nn.Module.update PRESERVING the incoming dtype (the
        Trainer passes an fp16 working copy each forward; masters are fp32).
        Stale fused QKV packs are dropped when the underlying weights change
        (no-lora mode) or change dtype.
        """
        from mlx.utils import tree_unflatten
        model_updates = {}
        dit_changed = False
        for name, arr in params.items():
            if name.endswith(".lora_A") or name.endswith(".lora_B"):
                base = name.rsplit(".", 1)[0]
                a = self.adapter_by_name[base]
                if name.endswith(".lora_A"):
                    a.lora_A = arr
                else:
                    a.lora_B = arr
            else:
                model_updates[name] = arr
                if name.startswith("dit."):
                    dit_changed = True
        if model_updates:
            self.model.update(tree_unflatten(list(model_updates.items())))
        if dit_changed and not self.use_lora:
            # base DiT weights updated -> fused QKV packs are stale
            for blk in self.model.cfm.estimator.transformer_blocks:
                blk.attn._qkv_w_cache = None
        elif dit_changed:
            # LoRA mode: base frozen, but a dtype cast still invalidates packs
            for blk in self.model.cfm.estimator.transformer_blocks:
                cache = blk.attn._qkv_w_cache
                if cache is not None and \
                        cache[0].dtype != blk.attn.to_q_w.dtype:
                    blk.attn._qkv_w_cache = None

    # -- forward ----------------------------------------------------------------
    def _estimator(self, xt, prompt, x_lens, t, d, mu) -> mx.array:
        if self.use_lora:
            return dit_lora_forward(self.model.cfm.estimator,
                                    self.adapter_by_name, xt, prompt,
                                    x_lens, t, d, mu)
        return self.model.cfm.estimator(
            xt, prompt, x_lens, t, d, mu, use_grad_ckpt=False)

    def forward(self, ssl, spec, mel, ssl_lengths, spec_lengths, text,
                text_lengths, mel_lengths, key: mx.array | None = None):
        """All inputs mx arrays. spec: (B, 1025, T) linear spectrogram.
        Returns (loss fp32 scalar, info dict)."""
        model = self.model

        y_mask = (mx.arange(spec.shape[2])[None, :]
                  < mx.asarray(spec_lengths)[:, None]).astype(spec.dtype)
        y_mask = y_mask[:, None, :]  # (B, 1, T)
        ge = model.ref_enc(spec[:, :704] * y_mask, y_mask)  # (B, 512)

        # frozen trunk: ssl_proj -> quantizer -> nearest x2 -> enc_p
        sslp = model.ssl_proj(ssl)
        codes = model.quantizer.encode(sslp)
        quantized = model.quantizer.decode(codes)
        quantized = _nearest_interp(quantized, quantized.shape[-1] * 2)
        x, m_p, logs_p, y_mask2 = model.enc_p(
            quantized, mx.asarray(spec_lengths), text,
            mx.asarray(text_lengths), ge)

        fea = nn.leaky_relu(model.bridge_0(x), 0.01)
        sc = 1.875 if self.version == "v3" else 2.0
        fea = _nearest_interp(fea, int(fea.shape[-1] * sc), scale_factor=sc)
        fea, _ = model.wns1(fea, mx.asarray(mel_lengths), ge)

        B = ssl.shape[0]
        prompt_len_max = mel_lengths * (2.0 / 3.0)
        if key is None:
            key = mx.random.key(random.getrandbits(63))
        key, sub = mx.random.split(key)
        prompt_len = mx.floor(mx.random.uniform(0.0, 1.0, (B,), key=sub)
                              * prompt_len_max).astype(mx.int32)

        minn = min(mel.shape[-1], fea.shape[-1])
        mel = mel[:, :, :minn]
        fea = fea[:, :, :minn]

        cfm = CFMTrainingLoss(self._estimator, rng=self.rng)
        loss, info = cfm(mel, mel_lengths, prompt_len, fea, key=key)
        return loss, info
