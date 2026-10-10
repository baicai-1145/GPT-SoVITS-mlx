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

import numpy as np

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

def upcast_training_model(model: SynthesizerTrnV3,
                          dit_fp16: bool = True,
                          dit_dtype: mx.Dtype | None = None) -> None:
    """Prepare training weight layout (official autocast semantics).

    * trunk (enc_p/ref_enc/ssl_proj/bridge/wns1/linear_mel via
      model.parameters()) and the plain-object attrs the walk MISSES —
      ``quantizer.embed`` (fp16 L2-distance GEMM flips ~50% code picks
      vs fp32 torch; measured on batch_002) and the DiT tree — are
      upcast to fp32 MASTERS.
    * with ``dit_fp16=True`` (default; official s2 trains under
      ``autocast(fp16_run=True)``): the DiT weights are cast BACK to fp16
      — fp16 forward is the official memory/precision regime, and the
      frozen base needs no master. LoRA A/B stay fp32 (masters; official
      peft keeps fp32 params under autocast) and are cast at use time in
      dit_lora_forward (fp16 B@A + fp16 add matches autocast: the fp32
      master only matters for the OPTIMIZER update).
    * ``dit_dtype`` (e.g. mx.bfloat16) overrides the DiT cast entirely —
      bf16 keeps fp32's exponent range so the BACKWARD cannot overflow
      (fp16 backward overflows past block ~8; measured, banned). The
      LoRA delta add stays in the DiT weight dtype; LoRA masters remain
      fp32 for the optimizer.
    """
    if dit_dtype is not None:
        dit_fp16 = False
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
    if dit_fp16:
        est.update(tree_map(lambda v: v.astype(mx.float16) if hasattr(v, "dtype")
                            else v, est.parameters()))
    elif dit_dtype is not None:
        est.update(tree_map(lambda v: v.astype(dit_dtype) if hasattr(v, "dtype")
                            else v, est.parameters()))
    for blk in est.transformer_blocks:
        blk.attn._qkv_w_cache = None
        blk.attn._cdtype = None


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
            # W + B@A (scaling = alpha/r = 1.0), computed in the BASE WEIGHT
            # dtype (fp32 here — see ckpt variant below for why fp16 is
            # unusable for the BACKWARD: grads overflow past block ~8)
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


# ---------------------------------------------------------------------------
# Two-pass checkpointed LoRA DiT forward+backward (s1 train_loss_and_grads
# pattern). WHY: the fused-graph LoRA forward above holds ALL 22 blocks'
# activation graphs (jetsam'd the machine at T=952), and the fp16 alternative
# NaNs in BACKWARD (grad magnitudes grow block-on-block and overflow fp16
# past block ~8; forward stays finite — see .tmp probes 2026-10-09). The
# two-pass form keeps ONLY the block inputs (B×T×1024 fp32) and recomputes
# one block at a time during backward — one block's graph alive at any
# moment. fp32 weights everywhere: no overflow, and fwd/bwd numerics equal
# the parity-verified fp32 path (grad check vs the fused path is the test).
# ---------------------------------------------------------------------------

def dit_ckpt_heads(dit, xt, prompt, x_lens, t, d, mu):
    """Eager head: everything before block 0 and after the last block.

    Returns (h0, tail_fn) where tail_fn(h) finishes the estimator. Both
    parts are eval'd/severed from the frozen-trunk graph; LoRA deltas are
    NOT applied here (blocks own them in the ckpt path).
    """
    x = mx.transpose(xt, (0, 2, 1))
    cond = mx.transpose(prompt, (0, 2, 1))
    text = mx.transpose(mu, (0, 2, 1))
    seq_len = x.shape[1]
    mask = (mx.arange(seq_len)[None, :]
            < mx.asarray(x_lens)[:, None].astype(mx.float32))
    mask = mask.astype(mx.bool_)
    text_embed = dit.text_embed(text.astype(mx.float32), seq_len)
    h = dit.input_embed(x, cond, text_embed)
    rope = dit._rope(seq_len, mx.float32)
    t0w = getattr(dit.time_embed, "time_mlp_0_w", None)
    td = t0w.dtype if t0w is not None else mx.float32
    emb = dit.time_embed(t.astype(td))
    if getattr(dit, "use_step_embedding", False):
        d0w = getattr(dit.d_embed, "time_mlp_0_w", None)
        dtd = d0w.dtype if d0w is not None else mx.float32
        emb = emb + dit.d_embed(d.astype(dtd))
    mx.eval(h, rope, emb)

    def tail(h_last):
        y = h_last
        if dit.long_skip_w is not None:
            # NOTE: long_skip needs the PRE-block residual; h0 is that only
            # when no blocks ran. For v3 (long_skip absent) this is exact.
            if getattr(dit, "_ckpt_h0", None) is not None:
                y = mx.concatenate([y, dit._ckpt_h0], axis=-1) @ dit.long_skip_w.T
        y = dit.norm_out(y, emb)
        return y @ dit.proj_out_w.T + dit.proj_out_b

    return h, mask, rope, emb, tail


def dit_ckpt_block_fwd(dit, adapters, i, h, mask, rope, emb):
    """Block i forward with LoRA delta materialized (differentiable)."""
    block = dit.transformer_blocks[i]
    attn = block.attn
    saved = {}
    for suffix, attr in (("to_q", "to_q_w"), ("to_k", "to_k_w"),
                         ("to_v", "to_v_w"), ("to_out.0", "to_out_0_w")):
        a = adapters.get(f"dit.blocks.{i}.attn.{suffix}")
        if a is None:
            continue
        base_w = getattr(attn, attr)
        saved[attr] = base_w
        setattr(attn, attr, base_w + (a.lora_B @ a.lora_A).astype(base_w.dtype))
    if saved:
        attn._qkv_w_cache = None
    try:
        return block(h, emb, mask, rope)
    finally:
        for attr, w in saved.items():
            setattr(attn, attr, w)
        if saved:
            attn._qkv_w_cache = None


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

    # -- two-pass checkpointed training (memory-safe DiT backward) -------------
    def _prep_ckpt(self, ssl, spec, mel, ssl_lengths, spec_lengths, text,
                   text_lengths, mel_lengths, det):
        """Shared trunk prep for the checkpointed path: returns everything
        the CFM loss needs plus the (eval'd, severed) DiT head state."""
        model = self.model
        ssl = mx.stop_gradient(ssl)
        text = mx.stop_gradient(text)

        y_mask = (mx.arange(spec.shape[2])[None, :]
                  < mx.asarray(spec_lengths)[:, None]).astype(spec.dtype)
        y_mask = y_mask[:, None, :]
        # ge side is TRAINABLE — run under the value_and_grad of the outer
        # trunk loss (kept small: ref_enc output is (B, 512)).
        ge = model.ref_enc(spec[:, :704] * y_mask, y_mask)

        sslp = model.ssl_proj(ssl)
        codes = model.quantizer.encode(sslp)
        quantized = model.quantizer.decode(codes)
        quantized = _nearest_interp(quantized, quantized.shape[-1] * 2)
        x, m_p, logs_p, y_mask2 = model.enc_p(
            quantized, mx.asarray(spec_lengths), text,
            mx.asarray(text_lengths), ge)
        x = mx.stop_gradient(x)
        fea = nn.leaky_relu(model.bridge_0(x), 0.01)
        sc = 1.875 if self.version == "v3" else 2.0
        fea = _nearest_interp(fea, int(fea.shape[-1] * sc), scale_factor=sc)
        fea, _ = model.wns1(fea, mx.asarray(mel_lengths), ge)
        return ge, fea

    def train_loss_and_grads(self, masters: dict, batch: dict,
                             det: dict | None = None) \
            -> tuple[float, dict, dict]:
        """Checkpointed forward+backward (s1 pattern; see dit_ckpt_* docs).

        Pass 1 (eager): trunk to fea (severed after enc_p); DiT head
        (input_embed/rope/emb) eval'd; blocks run one at a time keeping
        ONLY h_i inputs; small tail (norm_out/proj_out) + CFM loss build
        the only live graphs besides the current block.
        Pass 2 (backward): grad of (tail+CFM-loss) wrt h_L; then per block
        i=L-1..0 grad of (block(h_i) * g_{i+1}).sum() wrt (h_i, lora A/B);
        then grad of the ge-side trunk loss (ref_enc/bridge/wns1) through
        the small fea-graph.
        Returns (loss_f, info, grads {name: fp32}).
        """
        import random as _random
        model = self.model
        est = model.cfm.estimator
        adapters = self.adapter_by_name if self.use_lora else {}

        b_np = {k: np.asarray(getattr(batch, k)) for k in (
            "ssl", "spec", "mel", "ssl_lengths", "spec_lengths",
            "text", "text_lengths", "mel_lengths")}
        B = b_np["ssl"].shape[0]
        mel_lengths = mx.array(b_np["mel_lengths"].astype(np.float32))
        spec_lengths = mx.array(b_np["spec_lengths"])

        # -- pass 1a: trunk (ge side differentiable, frozen side severed) ---
        def trunk_loss(p):
            self.apply_params({k: v for k, v in p.items()
                               if not k.startswith("dit.")})
            ge, fea = self._prep_ckpt(
                mx.array(b_np["ssl"]), mx.array(b_np["spec"]),
                mx.array(b_np["mel"]), mx.array(b_np["ssl_lengths"]),
                spec_lengths, mx.array(b_np["text"].astype(np.int32)),
                mx.array(b_np["text_lengths"]), mel_lengths, None)
            return ge, fea

        # frozen side runs once (no grads needed): ge/fea values
        self.apply_params({k: v for k, v in masters.items()
                           if not k.startswith("dit.")})
        ge, fea = self._prep_ckpt(
            mx.array(b_np["ssl"]), mx.array(b_np["spec"]),
            mx.array(b_np["mel"]), mx.array(b_np["ssl_lengths"]),
            spec_lengths, mx.array(b_np["text"].astype(np.int32)),
            mx.array(b_np["text_lengths"]), mel_lengths, None)

        # -- CFM state (same draw logic as CFMTrainingLoss) ------------------
        ml_np = b_np["mel_lengths"]
        if det is not None:
            t = det["t"].astype(mx.float32)
            x0_full = det["x0"].astype(mx.float32)
            gate = det["gate"]
            pl = [int(p) for p in det["prompt_lens"]]
        else:
            key = mx.random.key(random.getrandbits(63))
            key, sub = mx.random.split(key)
            t = mx.random.uniform(0.0, 1.0, (B,), key=key).astype(mx.float32)
            x0_full = mx.random.normal((B, 100, b_np["mel"].shape[2]), key=sub)
            gate = self.rng.random()
            pr = mx.random.uniform(0.0, 1.0, (B,), key=sub)
            pr_np = np.asarray(pr)
            pl = []
            for i in range(B):
                pl.append(int(np.floor(pr_np[i] * (ml_np[i] * 2.0 / 3.0))))

        minn = min(b_np["mel"].shape[2], fea.shape[-1])
        mel = mx.array(b_np["mel"])[:, :, :minn]
        fea = fea[:, :, :minn]
        x0 = x0_full[:, :, :minn]
        vt = mel - x0
        xt = x0 + t[:, None, None] * vt
        dt = mx.zeros((B,), dtype=mx.float32)
        prompt = mx.zeros_like(mel)
        for i in range(B):
            if pl[i] > 0:
                prompt[i, :, :pl[i]] = mel[i, :, :pl[i]]
                xt[i, :, :pl[i]] = 0.0
        x_lens = mel_lengths

        info = {"two_step": False}
        if gate < 0.3:
            info["two_step"] = True
            base = mx.array([self.rng.randint(2, 7) for _ in range(B)],
                            dtype=mx.float32)
            d = 1.0 / mx.power(2.0, base)
            d_input = mx.where(d < 1e-2, mx.zeros_like(d), d)
            v1 = mx.stop_gradient(mx.transpose(
                self._estimator(xt, prompt, x_lens, t, d_input, fea), (0, 2, 1)))
            mx.eval(v1)
            x_mid = xt + d[:, None, None] * v1
            v2 = mx.stop_gradient(mx.transpose(
                self._estimator(x_mid, prompt, x_lens, t + d, d_input, fea),
                (0, 2, 1)))
            mx.eval(v2)
            vt = mx.stop_gradient((v1 + v2) / 2)
            mx.eval(vt)
            dt = 2 * d

        # -- pass 1b: DiT head + eager block walk ----------------------------
        xt_t = mx.stop_gradient(xt)
        prompt_t = mx.stop_gradient(prompt)
        h0, mask, rope, emb, tail = dit_ckpt_heads(
            est, xt_t, prompt_t, x_lens, t, dt, fea)
        est._ckpt_h0 = h0  # long_skip residual (v3: unused)
        hs = [h0]
        h = h0
        for i in range(len(est.transformer_blocks)):
            h = dit_ckpt_block_fwd(est, adapters, i, h, mask, rope, emb)
            mx.eval(h)
            hs.append(h)

        # -- tail + CFM loss (live graph; small) -----------------------------
        def tail_loss(h_last):
            y = est.norm_out(h_last, emb)
            out = y @ est.proj_out_w.T + est.proj_out_b
            vt_pred = mx.transpose(out, (0, 2, 1))
            total = mx.zeros((), dtype=mx.float32)
            for i in range(B):
                lo, hi = pl[i], int(ml_np[i])
                if hi <= lo:
                    continue
                pred = vt_pred[i, :, lo:hi].astype(mx.float32)
                tgt = vt[i, :, lo:hi].astype(mx.float32)
                total = total + mx.sum((pred - tgt) ** 2) / pred.size
            return total / B

        loss, g_h = mx.value_and_grad(tail_loss)(hs[-1])
        mx.eval(loss, g_h)
        loss_f = float(loss)

        grads = {}

        # -- pass 2: blocks, one at a time ------------------------------------
        g = g_h
        for i in range(len(est.transformer_blocks) - 1, -1, -1):
            block = est.transformer_blocks[i]
            lora_params = []
            names = []
            for suffix in ("to_q", "to_k", "to_v", "to_out.0"):
                nm = f"dit.blocks.{i}.attn.{suffix}"
                if nm in adapters:
                    lora_params.extend([masters[f"{nm}.lora_A"],
                                        masters[f"{nm}.lora_B"]])
                    names.extend([f"{nm}.lora_A", f"{nm}.lora_B"])

            def red(hh, *lp, _i=i, _np=lora_params, _names=names):
                # recompute block i with CURRENT lora params
                a_map = {}
                for j, nm in enumerate(_names):
                    suffix = nm.rsplit(".", 2)[-2] if nm.endswith("B") \
                        else nm.rsplit(".", 2)[-2]
                    base = nm.rsplit(".", 1)[0]
                    a_map.setdefault(base, {})["A" if nm.endswith("A") else "B"] = lp[j]
                block = est.transformer_blocks[_i]
                attn = block.attn
                saved = {}
                for suffix, attr in (("to_q", "to_q_w"), ("to_k", "to_k_w"),
                                     ("to_v", "to_v_w"),
                                     ("to_out.0", "to_out_0_w")):
                    nm2 = f"dit.blocks.{_i}.attn.{suffix}"
                    if nm2 not in a_map:
                        continue
                    base_w = getattr(attn, attr)
                    saved[attr] = base_w
                    A = a_map[nm2]["A"]
                    Bm = a_map[nm2]["B"]
                    setattr(attn, attr,
                            base_w + (Bm @ A).astype(base_w.dtype))
                if saved:
                    attn._qkv_w_cache = None
                try:
                    out = block(hh, emb, mask, rope)
                    return (out.astype(mx.float32) * g.astype(mx.float32)).sum()
                finally:
                    for attr, w in saved.items():
                        setattr(attn, attr, w)
                    if saved:
                        attn._qkv_w_cache = None

            if lora_params:
                res = mx.grad(red, argnums=tuple(range(len(lora_params) + 1)))(
                    hs[i], *lora_params)
                g = res[0]
                for j, nm in enumerate(names):
                    grads[nm] = res[j + 1].astype(mx.float32)
                mx.eval(g, *(grads[nm] for nm in names))
            else:
                # no adapters on this block (shouldn't happen for v3)
                def red0(hh, _i=i):
                    out = est.transformer_blocks[_i](hh, emb, mask, rope)
                    return (out.astype(mx.float32) * g.astype(mx.float32)).sum()
                g = mx.grad(red0)(hs[i])
                mx.eval(g)

        # -- ge-side trunk grads (small graph) --------------------------------
        def full_trunk_loss(p):
            self.apply_params(p)
            ge2, fea2 = self._prep_ckpt(
                mx.array(b_np["ssl"]), mx.array(b_np["spec"]),
                mx.array(b_np["mel"]), mx.array(b_np["ssl_lengths"]),
                spec_lengths, mx.array(b_np["text"].astype(np.int32)),
                mx.array(b_np["text_lengths"]), mel_lengths, None)
            # reduce like the DiT consumption: fea enters the (severed) head;
            # its gradient is g_fea — but head is severed, so chain manually:
            # dL/dfea = dL/d(mu) of the tail... not available. Use the
            # straight trick: fea only enters via the head; grad wrt fea is
            # obtained from the head-loss recomputation below.
            return ge2, fea2

        # gradient of tail wrt fea (mu input) — recompute tail with fea live:
        # the head path input_embed consumes fea; recompute h-side grad:
        # (kept simple: re-run input_embed+tail as a function of fea)
        def head_fea_loss(fea_p):
            x = mx.stop_gradient(xt_t).transpose(0, 2, 1)
            cond = mx.stop_gradient(prompt_t).transpose(0, 2, 1)
            seq_len = x.shape[1]
            text_embed = est.text_embed(mx.stop_gradient(
                fea_p).transpose(0, 2, 1).astype(mx.float32), seq_len)
            hh = est.input_embed(x, cond, text_embed)
            for i in range(len(est.transformer_blocks)):
                hh = dit_ckpt_block_fwd(est, adapters, i, hh, mask, rope, emb)
            y = est.norm_out(hh, emb)
            out = y @ est.proj_out_w.T + est.proj_out_b
            vt_pred = mx.transpose(out, (0, 2, 1))
            total = mx.zeros((), dtype=mx.float32)
            for i in range(B):
                lo, hi = pl[i], int(ml_np[i])
                if hi <= lo:
                    continue
                pred = vt_pred[i, :, lo:hi].astype(mx.float32)
                tgt = vt[i, :, lo:hi].astype(mx.float32)
                total = total + mx.sum((pred - tgt) ** 2) / pred.size
            return total / B

        # NOTE: full head recompute for fea-grad would hold ALL block graphs
        # again. Instead chain block-wise (backward direction reuse of hs):
        # grad wrt h0 already computed (g after block 0 = grad wrt h0);
        # head-side: dL/dh0 -> input_embed -> dL/d(text_embed + cond).
        def head0_loss(fea_p):
            x = mx.stop_gradient(xt_t).transpose(0, 2, 1)
            cond = mx.stop_gradient(prompt_t).transpose(0, 2, 1)
            seq_len = x.shape[1]
            text_embed = est.text_embed(fea_p.transpose(0, 2, 1)
                                        .astype(mx.float32), seq_len)
            hh = est.input_embed(x, cond, text_embed)
            return (hh.astype(mx.float32) * g.astype(mx.float32)).sum()

        # g is grad wrt h0 at this point (after the block-0 iteration)
        g_fea_raw = mx.grad(head0_loss)(mx.stop_gradient(fea))
        # trunk: fea/grad path through bridge/wns1/ref_enc
        def trunk_fea_loss(p):
            self.apply_params(p)
            ge2, fea2 = self._prep_ckpt(
                mx.array(b_np["ssl"]), mx.array(b_np["spec"]),
                mx.array(b_np["mel"]), mx.array(b_np["ssl_lengths"]),
                spec_lengths, mx.array(b_np["text"].astype(np.int32)),
                mx.array(b_np["text_lengths"]), mel_lengths, None)
            return (fea2.astype(mx.float32)
                    * g_fea_raw.astype(mx.float32)).sum()

        trunk_keys = [k for k in masters if not k.startswith("dit.")]
        tg = mx.grad(trunk_fea_loss)({k: masters[k] for k in trunk_keys})
        for k in trunk_keys:
            grads[k] = tg[k].astype(mx.float32)
        mx.eval(*(grads[k] for k in trunk_keys))
        return loss_f, info, grads

    def forward(self, ssl, spec, mel, ssl_lengths, spec_lengths, text,
                text_lengths, mel_lengths, key: mx.array | None = None,
                det: dict | None = None):
        """All inputs mx arrays. spec: (B, 1025, T) linear spectrogram.
        Returns (loss fp32 scalar, info dict).

        det: deterministic-reference state (t/x0/prompt_lens/gate/base) —
        see CFMTrainingLoss(det=...)."""
        model = self.model

        # FROZEN-TRUNK CUT (memory): official set_no_grad freezes ONLY
        # ssl_proj/quantizer/enc_p in the LoRA trainer; ref_enc (ge),
        # bridge and wns1 REMAIN TRAINABLE. So the cut severs exactly the
        # frozen segment: inputs to ssl_proj/enc_p and their output x.
        # ge/bridge/wns1 keep full graphs (their params receive grads).
        ssl = mx.stop_gradient(ssl)
        text = mx.stop_gradient(text)

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

        x = mx.stop_gradient(x)  # enc_p output: frozen segment boundary
        fea = nn.leaky_relu(model.bridge_0(x), 0.01)
        sc = 1.875 if self.version == "v3" else 2.0
        fea = _nearest_interp(fea, int(fea.shape[-1] * sc), scale_factor=sc)
        fea, _ = model.wns1(fea, mx.asarray(mel_lengths), ge)

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
