#!/usr/bin/env python3
"""bf16 DiT experiment gates (task cfm-mem-opt, Item 2).

GATE (i)   GEMM microbench: fp32 vs fp16 vs bf16 on the real DiT-block
           GEMM shapes at batch=3, seq=700 (dim 1024, ff mult 2):
           qkv (2100x1024x3072), ff up (2100x1024x2048), ff down
           (2100x2048x1024). Synced-single-op AND queued-batch timings
           reported separately (BENCH.md discipline; never summed).
GATE (ii)  Block-stack overflow test at T=300 (the 2026-10-09 fp16 repro
           class: fp16 BACKWARD NaNs from block ~8 while the forward stays
           finite ~4.3e3). Measures the upstream gradient dL/dh_i at every
           block boundary exactly like the two-pass trainer's ``red``
           reduction does — fp16 must reproduce the overflow, bf16 must
           keep all 22 finite (bf16 exponent range == fp32).

Both gates run on GPU behind the canonical gpu.lock. Writes a JSON report
to --out (default .tmp/bf16_gates.json) and prints PASS/FAIL verdicts.

Usage:
  python3 tools/bf16_gates.py --gpu            # both gates
  python3 tools/bf16_gates.py --gpu --gemm-only
  python3 tools/bf16_gates.py --gpu --stack-only
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def gemm_gate() -> dict:
    """DiT-block-shaped GEMM timing: fp32 vs fp16 vs bf16."""
    import mlx.core as mx
    shapes = [  # (M, K, N, tag) — batch 3 x seq 700 rows, real DiT dims
        (2100, 1024, 3072, "qkv_fused"),
        (2100, 1024, 2048, "ff_up"),
        (2100, 2048, 1024, "ff_down"),
    ]
    q_depth = 3   # one of each shape per block invocation
    rounds = 10
    report = {"mode": "gemm", "batch": 3, "seq": 700, "rounds": rounds,
              "queue_depth": q_depth, "dtypes": {}}
    for dt_name, dt in (("fp32", mx.float32), ("fp16", mx.float16),
                        ("bf16", mx.bfloat16)):
        per_shape = {}
        for M, K, N, tag in shapes:
            lefts = [mx.random.normal((M, K), key=mx.random.key(i + 1)).astype(dt)
                     for i in range(q_depth + 1)]
            rights = [mx.random.normal((K, N), key=mx.random.key(i + 101)).astype(dt)
                      for i in range(q_depth + 1)]
            mx.eval(lefts, rights)
            mx.eval(lefts[q_depth] @ rights[q_depth])  # warmup
            best = math.inf
            for _ in range(rounds):  # (a) synced single op (dispatch+sync in)
                s = time.perf_counter()
                r = lefts[q_depth] @ rights[q_depth]
                mx.eval(r)
                best = min(best, time.perf_counter() - s)
            best_q = math.inf
            for _ in range(rounds):  # (b) q distinct GEMMs, ONE eval
                s = time.perf_counter()
                outs = [lefts[i] @ rights[i] for i in range(q_depth)]
                mx.eval(*outs)
                best_q = min(best_q, time.perf_counter() - s)
            flops = 2 * M * K * N
            per_shape[tag] = {
                "shape": f"{M}x{K}x{N}",
                "single_op_synced_ms": round(best * 1e3, 3),
                "batch3_synced_ms": round(best_q * 1e3, 3),
                "batch3_tflops_achieved": round(q_depth * flops / best_q / 1e12, 2),
            }
            del lefts, rights, outs
        report["dtypes"][dt_name] = per_shape
        try:
            mx.clear_cache()
        except Exception:
            pass
    def total(dt):
        return sum(v["batch3_synced_ms"] for v in report["dtypes"][dt].values())
    report["total_batch3_ms"] = {dt: round(total(dt), 3)
                                 for dt in report["dtypes"]}
    return report


def stack_gate(models_root: str, T: int = 300) -> dict:
    """22-block DiT upstream-grad magnitudes at T=300 in fp32/fp16/bf16.

    Pass 1 (eager): forward with all block inputs stored (two-pass trainer
    pass-1b pattern). Pass 2: for each block i, grad of the chained loss
    (blocks i..21 + tail) wrt the stored input h_i — the exact quantity the
    two-pass ``red`` reduction differentiates; its magnitude grows as the
    backward chain lengthens and is what overflows fp16 past block ~8.
    """
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_map
    from gsovits_mlx.pipeline import load_sovits_v3

    model, _ = load_sovits_v3(os.path.join(models_root, "v3"), "v3")
    est = model.cfm.estimator
    B, L = 3, len(est.transformer_blocks)
    report = {"mode": "block_stack", "T": T, "B": B, "n_blocks": L,
              "dtypes": {}}

    # keep originals for recasting between dtype runs
    orig = dict(tree_flatten(est.parameters()))

    x0 = mx.random.normal((B, 100, T), key=mx.random.key(0))
    cond = mx.random.normal((B, 100, T), key=mx.random.key(1))
    mu = mx.random.normal((B, 512, T), key=mx.random.key(2))
    t = mx.full((B,), 0.5)
    d = mx.zeros((B,))
    x_lens = mx.full((B,), float(T))

    def forward_state():
        """Shared head: inputs to block 0 + modulation state (weight-dtype
        aware, like dit_ckpt_heads)."""
        x = mx.transpose(x0, (0, 2, 1)).astype(mx.float32)
        c = mx.transpose(cond, (0, 2, 1)).astype(mx.float32)
        text = mx.transpose(mu, (0, 2, 1)).astype(mx.float32)
        h = est.input_embed(x, c, est.text_embed(text, T))
        rope = est._rope(T, mx.float32)
        mask = (mx.arange(T)[None, :]
                < x_lens[:, None].astype(mx.float32)).astype(mx.bool_)
        td = est.time_embed.time_mlp_0_w.dtype
        emb = est.time_embed(t.astype(td))
        if getattr(est, "use_step_embedding", False):
            dd = est.d_embed.time_mlp_0_w.dtype
            emb = emb + est.d_embed(d.astype(dd))
        return h, rope, mask, emb

    def tail_loss(h_last, emb):
        y = est.norm_out(h_last, emb)
        out = y @ est.proj_out_w.T + est.proj_out_b
        vt_pred = mx.transpose(out, (0, 2, 1))
        tgt = mx.zeros_like(vt_pred)
        return mx.mean((vt_pred - tgt) ** 2)

    for dt_name, dt in (("fp32", mx.float32), ("fp16", mx.float16),
                        ("bf16", mx.bfloat16)):
        est.update(tree_map(lambda v: v.astype(dt) if hasattr(v, "dtype")
                            else v, orig))
        for blk in est.transformer_blocks:
            blk.attn._qkv_w_cache = None
            blk.attn._cdtype = None
            blk._cdtype = None
        entry = {"forward_finite": None, "loss": None,
                 "upstream_grad_absmax": {}, "first_overflow_block": None}
        try:
            # pass 1: eager block walk storing inputs (h0 = block-0 input)
            h, rope, mask, emb = forward_state()
            mx.eval(h, rope, emb)
            hs = [h]
            for blk in est.transformer_blocks:
                h = blk(h, emb, mask, rope)
                mx.eval(h)
                hs.append(h)
            loss = tail_loss(hs[-1], emb)
            mx.eval(loss)
            entry["loss"] = float(loss)
            entry["forward_finite"] = math.isfinite(float(loss))
            # forward activation magnitude (the documented ~4.3e3 fp16 max)
            entry["forward_h_absmax"] = float(max(
                float(mx.abs(hh).max()) for hh in hs))

            # pass 2: upstream grad at every block boundary
            for i in range(L - 1, -1, -1):
                h_in = mx.stop_gradient(hs[i])

                def chain_loss(hv, _i=i):
                    hh = hv
                    for blk in est.transformer_blocks[_i:]:
                        hh = blk(hh, emb, mask, rope)
                    return tail_loss(hh, emb)

                g = mx.grad(chain_loss)(h_in)
                mx.eval(g)
                gmax = float(mx.abs(g).max())
                entry["upstream_grad_absmax"][f"h_{i}"] = gmax
                if not math.isfinite(gmax) and \
                        entry["first_overflow_block"] is None:
                    entry["first_overflow_block"] = i
                del g
                try:
                    mx.clear_cache()
                except Exception:
                    pass
        except Exception as e:
            entry["error"] = f"{type(e).__name__}: {e}"[:300]
        report["dtypes"][dt_name] = entry
        try:
            mx.clear_cache()
        except Exception:
            pass
    # verdicts
    for dt_name in report["dtypes"]:
        e = report["dtypes"][dt_name]
        if "error" in e:
            e["verdict"] = "ERROR"
        else:
            finite = all(math.isfinite(v)
                         for v in e["upstream_grad_absmax"].values())
            e["verdict"] = "PASS" if (finite and e["forward_finite"]) else "FAIL"
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--gemm-only", action="store_true")
    p.add_argument("--stack-only", action="store_true")
    p.add_argument("--T", type=int, default=300)
    p.add_argument("--models-root",
                   default=os.environ.get("GSOVITS_MODELS_ROOT",
                                          os.path.join(_REPO, "models_local")))
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if not args.gpu:
        raise SystemExit("GPU-only tool (dtype kernel behavior is "
                         "Metal-specific); pass --gpu")
    from gsovits_mlx.gpu_lock import acquire_lock, resolve_device, release_lock
    import subprocess as _sp
    procs = _sp.run(["ps", "ax", "-o", "command"], capture_output=True,
                    text=True).stdout.splitlines()
    gpu_procs = [ln.strip() for ln in procs
                 if ("train" in ln or "smoke" in ln or "prepare" in ln
                     or "probe" in ln or "bench" in ln)
                 and "python" in ln and "bf16_gates" not in ln
                 and "grep" not in ln]
    if gpu_procs:
        raise SystemExit("[gpu.lock] other GPU python processes:\n  "
                         + "\n  ".join(gpu_procs[:5]))
    acquire_lock("bf16_gates cfm-mem-opt item2")
    try:
        device = resolve_device(True, verbose=True)
        assert device == "gpu", "must run on GPU"
        import mlx.core as mx
        try:
            mx.metal.set_memory_limit(8 * 1024 * 1024 * 1024)
        except Exception:
            pass
        report = {"timestamp": time.strftime("%F %T")}
        if not args.stack_only:
            report["gemm"] = gemm_gate()
            print("[gemm] total batch3 ms:", report["gemm"]["total_batch3_ms"])
        if not args.gemm_only:
            report["stack"] = stack_gate(args.models_root, T=args.T)
            for dt, e in report["stack"]["dtypes"].items():
                print(f"[stack:{dt}] fwd_finite={e.get('forward_finite')} "
                      f"loss={e.get('loss')} first_overflow="
                      f"{e.get('first_overflow_block')} verdict={e.get('verdict')}")
        out = args.out or os.path.join(_REPO, ".tmp", "bf16_gates.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump(report, f, indent=1)
        print(f"report -> {out}")
    finally:
        release_lock()


if __name__ == "__main__":
    main()
