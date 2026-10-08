"""Minimal peft-style LoRA for MLX linear layers (official s2 v3 lora semantics).

Official: peft.LoraConfig(target_modules=["to_k","to_q","to_v","to_out.0"],
r=lora_rank(=32), lora_alpha=lora_rank, init_lora_weights=True) applied to
``net_g.cfm`` (GPT_SoVITS/s2_train_v3_lora.py). peft wraps each matched
``nn.Linear`` as ``W x + scaling * B(A x)`` with:

* ``scaling = lora_alpha / r = 1.0`` (alpha == rank in the official config);
* init: ``B = 0`` and ``A ~ kaiming_uniform(a=sqrt(5))`` — i.e. exactly the
  torch default ``nn.Linear`` reset_parameters init for the A matrix
  (fan_in = A.shape[1], bound = sqrt(6 / ((1 + a^2) * fan_in)) with
  a = sqrt(5) -> bound = sqrt(1 / fan_in)), uniform in [-bound, bound].

Mapping to OUR DiT (gsovits_mlx/sovits/dit.py Attention): the torch names
to_q/to_k/to_v/to_out.0 correspond to weights to_q_w/to_k_w/to_v_w/to_out_0_w
(+ biases) with ``y = x @ W.T + b`` — identical math to nn.Linear, so
LoRA applies as ``x @ (W + B @ A).T + b`` (per-call: ``x @ W.T + (x @ A.T) @ B.T + b``).
"""

from __future__ import annotations

import math

import mlx.core as mx

__all__ = ["kaiming_uniform_bound", "LoRALinear", "inject_lora", "merge_lora"]


def kaiming_uniform_bound(fan_in: int, a: float = math.sqrt(5.0)) -> float:
    """torch.nn.init.kaiming_uniform_ bound for a=sqrt(5) (nn.Linear default).

    gain = sqrt(2 / (1 + a^2)) = 1/sqrt(3); bound = gain * sqrt(3 / fan_in)
    = sqrt(1 / fan_in) — matches nn.Linear.reset_parameters.
    """
    gain = math.sqrt(2.0 / (1.0 + a * a))
    return gain * math.sqrt(3.0 / fan_in)


class LoRALinear:
    """Stateful LoRA adapter record for one linear weight.

    The forward math lives in s2_cfm._attention_forward_lora; this class only
    owns parameters + init so the optimizer/trainer can treat it as a plain
    {name: mx.array} provider.
    """

    def __init__(self, name: str, w: mx.array, rank: int = 32,
                 key: mx.array | None = None):
        self.name = name                      # e.g. "dit.blocks.3.attn.to_q"
        self.rank = int(rank)
        out_dim, in_dim = w.shape             # MLX layout: y = x @ W.T + b
        self.lora_A = mx.random.uniform(
            -kaiming_uniform_bound(in_dim), kaiming_uniform_bound(in_dim),
            (rank, in_dim), key=key).astype(mx.float32)
        self.lora_B = mx.zeros((out_dim, rank), dtype=mx.float32)

    def params(self) -> dict:
        return {f"{self.name}.lora_A": self.lora_A,
                f"{self.name}.lora_B": self.lora_B}


def inject_lora(dit, rank: int = 32, seed: int | None = None,
                key: mx.array | None = None) -> list[LoRALinear]:
    """Create rank-r adapters for every Attention q/k/v/out linear in `dit`.

    Mirrors peft.get_peft_model(net_g.cfm, LoraConfig(target_modules=[
    "to_k","to_q","to_v","to_out.0"], r=rank, lora_alpha=rank)) — module order
    is peft's named_parameters order: transformer block by block, within each
    attention to_k, to_q, to_v, to_out.0 (peft matches by module-name suffix
    and wraps in the module tree order). Fresh randomness per adapter, as in
    peft (one nn.Linear init each).
    """
    if seed is not None:
        key = mx.random.key(seed)
    adapters = []
    for bi, block in enumerate(dit.transformer_blocks):
        attn = block.attn
        for suffix, w in (("to_k", attn.to_k_w), ("to_q", attn.to_q_w),
                          ("to_v", attn.to_v_w), ("to_out.0", attn.to_out_0_w)):
            if key is not None:
                key, sub = mx.random.split(key)
            else:
                sub = None
            adapters.append(
                LoRALinear(f"dit.blocks.{bi}.attn.{suffix}", w, rank=rank, key=sub))
    return adapters


def merge_lora(w: mx.array, adapter: LoRALinear, scaling: float = 1.0) -> mx.array:
    """W + scaling * B @ A (fp32 math, cast back to w.dtype)."""
    delta = (adapter.lora_B @ adapter.lora_A).astype(mx.float32) * scaling
    return (w.astype(mx.float32) + delta).astype(w.dtype)


def merged_training_weights(dit, adapters: list[LoRALinear],
                            scaling: float = 1.0) -> dict:
    """{convert-style DiT weight name: merged fp16 array} for export.

    Keys match s2v3_model_to_arrays outputs (dit.blocks.<i>.attn.to_q.w
    etc.) so the export can override those entries directly.
    """
    out = {}
    by_name = {a.name: a for a in adapters}
    for bi, block in enumerate(dit.transformer_blocks):
        attn = block.attn
        for suffix, w, out_name in ((
                "to_k", attn.to_k_w, "to_k.w"),
                ("to_q", attn.to_q_w, "to_q.w"),
                ("to_v", attn.to_v_w, "to_v.w"),
                ("to_out.0", attn.to_out_0_w, "to_out.w")):
            a = by_name.get(f"dit.blocks.{bi}.attn.{suffix}")
            if a is not None:
                out[f"dit.blocks.{bi}.attn.{out_name}"] = merge_lora(w, a, scaling)
    return out
