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

def upcast_training_model(model: SynthesizerTrnV3) -> None:
    """Cast ALL weights to fp32 for training — including the ones
    nn.Module.parameters() does NOT reach (plain-object attributes):

    * ``model.quantizer.embed`` — plain array on a plain object; leaving it
      fp16 makes the L2-distance GEMM run at fp16 precision (~0.5 absolute
      error on ~545-magnitude distances) and flips ~50% of the code picks
      vs the official fp32 quantizer (measured on batch_002).
    * ``model.cfm.estimator`` (the DiT) — plain-object holder; fp16 weights
      gave up to ~12% CFM-loss drift vs the torch fp32 reference.

    Everything inside model.parameters() (enc_p/ref_enc/ssl_proj/bridge/
    wns1/linear_mel) is upcast through model.update.
    """
    from mlx.utils import tree_map
    model.update(tree_map(lambda v: v.astype(mx.float32) if hasattr(v, "dtype")
                          else v, model.parameters()))
    est = model.cfm.estimator
    est.update(tree_map(lambda v: v.astype(mx.float32) if hasattr(v, "dtype")
                        else v, est.parameters()))
    for blk in est.transformer_blocks:
        blk.attn._qkv_w_cache = None
        blk.attn._cdtype = None
    model.quantizer.embed = model.quantizer.embed.astype(mx.float32)


def dit_lora_forward(dit, adapters: dict[str, LoRALinear], xt, prompt, x_lens,
                     t, d, mu) -> mx.array:
    """Training-layout DiT forward (f5_tts dit.py non-infer path) with LoRA.

    LoRA is applied by MATERIALIZING ``W + B @ A`` into the attention module
    weights for the duration of the call and then running the VERIFIED
    inference forward ``DiT.__call__(..., use_grad_ckpt=False)`` — the base
    weights are frozen in LoRA mode, so the fused weight is an exact,
    differentiable-through expression: mx.grad flows to A/B through the
    matmul while every other op follows the parity-verified DiT code path
    (task-1..12 lineage). This avoids reimplementing the attention block
    (an earlier hand-rolled port drifted ~10% vs torch on some batches).

    adapters: {"dit.blocks.<i>.attn.<suffix>": LoRALinear} keyed by prefix
    (S2V3TrainModel sets ``attn._lora_prefix``); scaling = alpha/r = 1.0.
    Fused QKV packs are invalidated around the swap (task-12 discipline).
    """
    saved = {}  # {(block_idx, attr): base_weight}
    any_touched = False
    for bi, block in enumerate(dit.transformer_blocks):
        attn = block.attn
        touched = False
        for suffix, attr in (("to_q", "to_q_w"), ("to_k", "to_k_w"),
                             ("to_v", "to_v_w"), ("to_out.0", "to_out_0_w")):
            a = adapters.get(f"dit.blocks.{bi}.attn.{suffix}")
            if a is None:
                continue
            base_w = getattr(attn, attr)
            saved[(bi, attr)] = base_w
            # W + B@A (scaling = alpha/r = 1.0); dtype follows the base
            # weight (fp32 in training)
            setattr(attn, attr, base_w + (a.lora_B @ a.lora_A).astype(base_w.dtype))
            touched = True
        if touched:
            attn._qkv_w_cache = None
            any_touched = True
    try:
        return dit(xt, prompt, x_lens, t, d, mu, use_grad_ckpt=False)
    finally:
        for (bi, attr), w in saved.items():
            setattr(dit.transformer_blocks[bi].attn, attr, w)
        if any_touched:
            for block in dit.transformer_blocks:
                block.attn._qkv_w_cache = None


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
    """CFM.forward port. x1/mu: (B, C, T); returns fp32 scalar loss.

    ``det`` mode (deterministic reference): consumes (t, x0, prompt_lens,
    two_step gate/base) arrays baked into the batch npz instead of drawing —
    used by the torch-CPU parity comparison so both sides share randomness.
    """

    def __init__(self, estimator, rng: random.Random | None = None):
        self.estimator = estimator  # callable(xt, prompt, x_lens, t, d, mu)
        self.rng = rng or random

    def __call__(self, x1: mx.array, x_lens, prompt_lens, mu: mx.array,
                 key: mx.array | None = None,
                 det: dict | None = None) -> tuple[mx.array, dict]:
        b = x1.shape[0]
        dtype = x1.dtype
        if det is not None:
            t = det["t"].astype(dtype)
            x0 = det["x0"].astype(dtype)[:, :, : x1.shape[-1]]
            pl = [int(p) for p in det["prompt_lens"]]
            gate = det["gate"]
        else:
            if key is None:
                key = mx.random.key(random.getrandbits(63))
            key, sub = mx.random.split(key)
            t = mx.random.uniform(0.0, 1.0, (b,), key=key).astype(dtype)
            x0 = mx.random.normal(x1.shape, key=sub).astype(dtype)
            pl = [int(p) for p in prompt_lens]
            gate = self.rng.random()
        vt = x1 - x0
        xt = x0 + t[:, None, None] * vt
        dt = mx.zeros((b,), dtype=dtype)
        prompt = mx.zeros_like(x1)

        for i in range(b):
            if pl[i] > 0:
                prompt[i, :, : pl[i]] = x1[i, :, : pl[i]]
                xt[i, :, : pl[i]] = 0.0

        info = {"two_step": False}
        if gate < 0.3:
            info["two_step"] = True
            if det is not None and "base" in det:
                base = det["base"].astype(dtype)
            else:
                # official torch.randint(2, 8, (b,)) — high EXCLUSIVE
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
        self.lora_adapters = self.adapters  # alias used by forward's cut
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
                text_lengths, mel_lengths, key: mx.array | None = None,
                det: dict | None = None):
        """All inputs mx arrays. spec: (B, 1025, T) linear spectrogram.
        Returns (loss fp32 scalar, info dict).

        det: deterministic-reference state (t/x0/prompt_lens/gate/base) —
        see CFMTrainingLoss(det=...)."""
        model = self.model

        # FROZEN TRUNK UNDER stop_gradient: ssl_proj/quantizer/enc_p/
        # ref_enc/bridge are frozen in LoRA fine-tuning (official
        # set_no_grad); keeping their graphs alive only wastes memory (the
        # low-swap jetsam kill at 2026-10-09 18:39 was exactly this). When
        # LoRA adapters are attached, everything below DiT is a constant.
        # (Non-LoRA full fine-tune keeps bridge/wns1 trainable — there the
        # graph must flow, so only apply the cut in the LoRA path.)
        if self.lora_adapters:
            spec = mx.stop_gradient(spec)
            ssl = mx.stop_gradient(ssl)

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
        if self.lora_adapters:
            # DiT (LoRA targets) is the only grad consumer; fea is a constant
            # conditioning input — cut the graph AFTER wns1 as well.
            fea = mx.stop_gradient(fea)
            ge = mx.stop_gradient(ge)

        B = ssl.shape[0]
        if det is not None:
            prompt_len = det["prompt_lens"]
        else:
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
        loss, info = cfm(mel, mel_lengths, prompt_len, fea, key=key, det=det)
        return loss, info
