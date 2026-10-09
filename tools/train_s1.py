#!/usr/bin/env python3
"""s1 (AR/GPT) trainer, MLX port of the official GPT_SoVITS/s1_train.py loop.

Semantics ported from GPT-SoVITS-cuda_graph_accel_v5/GPT_SoVITS
(AR/data + AR/models + s1_train.py + t2s_lightning_module.py):

* dataset/sampler: official filters + bucketing (gsovits_mlx/train/s1_data.py)
* forward: forward_old (gsovits_mlx/train/s1_model_train.py) — fp16 forward,
  fp32 CE, sum reduction, top-3 acc (ignore EOS)
* optimizer: ScaledAdam lr=0.01 betas (0.9,0.95) clipping_scale 2.0
  clipping_update_period 1000 size_update_period 4 scalar_lr_scale 0.1
  (official configure_optimizers defaults)
* scheduler: locked-0.002 WarmupCosine (official hard-lock quirk)
* grad accumulation: official manual_backward every batch; opt.step() only
  when batch_idx>0 and batch_idx%4==0 (then zero_grad + scheduler.step) —
  i.e. the first optimizer step accumulates batches 0..4 (5 batches) and
  later steps every 4; loss is SUM-reduced, no averaging (matching official).
* checkpointing: resume ckpt + export gpt.safetensors/gpt.json (task-1
  ckpt.save_s1_inference) loadable by gsovits_mlx.pipeline.load_gpt.

Usage:
    python tools/train_s1.py --exp-dir .tmp/train_data/exp --version v2 \
        [--init-from models_local/s1v2] [--max-steps 200] [--batch-size 8]

Reads: <exp_dir>/2-name2text.txt, <exp_dir>/3-bert/*.npy,
<exp_dir>/6-name2semantic.tsv (official 1-prepare artifacts, task-2 emits).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np

VERSION_S1 = {
    # version -> (s1 pretrained dir under models root, cleaner version)
    "v1": ("s1v1", "v1"),
    "v2": ("s1v2", "v2"),
    "v2Pro": ("s1v2", "v2"),
    "v2ProPlus": ("s1v2", "v2"),
    "v3": ("s1", "v2"),
    "v4": ("s1", "v2"),
    "v5dev": ("s1", "v2"),
    "v5turbo": ("s1", "v2"),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp-dir", required=True,
                   help="exp dir with 2-name2text.txt / 3-bert/ / 6-name2semantic.tsv")
    p.add_argument("--version", required=True, choices=sorted(VERSION_S1),
                   help="Target version (picks s1 config: max_sec 54 v1 / 57 v2-family).")
    p.add_argument("--init-from", default=None,
                   help="s1 weights dir (gpt.safetensors+gpt.json) to start from "
                        "(default: version's pretrained s1).")
    p.add_argument("--models-root", default=os.environ.get(
        "GSOVITS_MODELS_ROOT", os.path.join(REPO_ROOT, "models_local")))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1,
                   help="Passes over the dataset (max-steps caps total).")
    p.add_argument("--max-steps", type=int, default=None,
                   help="Cap on optimizer steps (smoke runs).")
    p.add_argument("--max-batches", type=int, default=None,
                   help="Cap on total batches across epochs (smoke runs).")
    p.add_argument("--grad-accum", type=int, default=4,
                   help="Batches per optimizer step (official 4).")
    p.add_argument("--seed", type=int, default=1234,
                   help="seed_everything(1234) official default.")
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--out-dir", default=None,
                   help="Output dir for loss log / ckpts / export "
                        "(default <exp_dir>/s1_train_out).")
    p.add_argument("--export-every", type=int, default=0,
                   help="Export gpt.safetensors+gpt.json every N optimizer steps "
                        "(0 = only at end).")
    p.add_argument("--save-every", type=int, default=100,
                   help="Resume-checkpoint every N optimizer steps (0 = off).")
    p.add_argument("--precision", choices=["fp16", "fp32"], default="fp16",
                   help="Forward precision (fp16 = official 16-mixed; fp32 for "
                        "numerical comparison runs).")
    p.add_argument("--cpufast-repo", default=None,
                   help="Explicit CPUFast repo for the vendored text front-end.")
    p.add_argument("--keep-gpu-lock", action="store_true",
                   help="Do not release the GPU lock on exit (debug).")
    p.add_argument("--grad-ckpt", action="store_true",
                   help="Per-block gradient checkpointing (mx.custom_function; "
                        "forward retains block inputs only, backward "
                        "recomputes). Peak-memory lever for long buckets.")
    p.add_argument("--cpu", action="store_true",
                   help="Run on CPU (unit-test path only; GPU required for real "
                        "training per AGENTS.md).")
    p.add_argument("--metal-limit-mb", type=int,
                   default=int(os.environ.get("GSOVITS_METAL_LIMIT_MB", "8192")),
                   help="mx.metal.set_memory_limit guard, applied BEFORE the "
                        "model/data is built: Metal allocations fail loudly "
                        "in-process instead of OOM-killing the machine.")
    p.add_argument("--max-src-bucket", type=float, default=None,
                   help="SMOKE-ONLY: skip batches whose padded src exceeds this "
                        "(official filters allow up to max_sec). Deviation to "
                        "exclude extreme length buckets; documented.")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# GPU lock — canonical main-repo lock via gsovits_mlx.gpu_lock
# (post-reboot law: NEVER a worktree-local lock; one GPU process machine-wide)
# ---------------------------------------------------------------------------

def _check_no_other_gpu_process() -> None:
    """ps aux guard: refuse to start if another python GPU trainer runs."""
    import subprocess
    out = subprocess.run(["ps", "aux"], capture_output=True, text=True).stdout
    own = os.getpid()
    for line in out.splitlines():
        if "python" not in line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[1])
        except ValueError:
            continue
        if pid == own:
            continue
        if ("train_s1" in line or "prepare_data" in line
                or "train_s2" in line or "tools/e2e" in line
                or "bench" in line and "mlx" in line):
            raise SystemExit(
                f"[gpu.lock] another GPU-ish process is running:\n{line}\n"
                "One GPU process machine-wide (AGENTS.md).")


def acquire_train_lock(version: str, exp_dir: str) -> None:
    from gsovits_mlx import gpu_lock
    _check_no_other_gpu_process()
    gpu_lock.acquire_lock(
        f"tools/train_s1.py train_s1 {version} exp={exp_dir} agent=s1-ar")


def refresh_train_lock() -> None:
    from gsovits_mlx import gpu_lock
    try:
        gpu_lock.refresh_lock()
    except OSError:
        pass


def release_train_lock() -> None:
    from gsovits_mlx import gpu_lock
    gpu_lock.release_lock()


def seed_everything(seed: int):
    """Official seed_everything(1234): python random + numpy + mx.random."""
    import random
    import mlx.core as mx
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)


def build_loss_fn(model, precision: str):
    """value_and_grad over (params, batch) -> (loss, grads)."""
    import mlx.core as mx

    base = model.forward

    def loss_fn(p: dict, batch: dict):
        loss, acc = base(p, batch)
        return loss

    if precision == "fp32":
        # run the forward in fp32: cast input weights inside forward uses
        # fp16 unconditionally; provide an fp32 variant by monkey-casting
        # — simplest correct approach: wrap params pre-cast and patch the
        # model's dtype. S1TrainModel.forward always builds p16; for fp32
        # we swap the model method with a fp32-cast clone.
        raise NotImplementedError(
            "fp32 comparison path is provided by tests via S1TrainModel "
            "directly (cast params to fp32 and patch dtype); CLI keeps "
            "fp16 (official 16-mixed).")
    return mx.value_and_grad(loss_fn)


def main(argv=None) -> int:
    args = parse_args(argv)

    s1_dir_name, cleaner_version = VERSION_S1[args.version]
    init_dir = args.init_from or os.path.join(args.models_root, s1_dir_name)
    config_path = os.path.join(init_dir, "gpt.json")
    with open(config_path) as f:
        meta = json.load(f)
    config = meta["config"]
    max_sec = config["data"]["max_sec"]

    out_dir = args.out_dir or os.path.join(args.exp_dir, "s1_train_out")
    os.makedirs(out_dir, exist_ok=True)
    loss_log_path = os.path.join(out_dir, "loss.jsonl")

    if args.cpu:
        os.environ.setdefault("MLX_DEFAULT_DEVICE", "cpu")
    import mlx.core as mx
    lock_held = False
    if not args.cpu:
        acquire_train_lock(args.version, args.exp_dir)
        lock_held = True
        # OOM law: cap Metal BEFORE building anything so allocations fail
        # loudly in-process instead of rebooting the machine.
        try:
            mx.metal.set_memory_limit(args.metal_limit_mb * 1024 * 1024)
            print(f"[mem] metal limit set to {args.metal_limit_mb} MB", flush=True)
        except Exception as e:  # noqa: BLE001 - older mlx lacks the API
            print(f"[mem] metal memory limit not set: {e}", flush=True)

    exit_code = 0
    try:
        exit_code = _run(args, config, cleaner_version, init_dir, out_dir,
                         loss_log_path, mx)
    finally:
        if lock_held and not args.keep_gpu_lock:
            release_train_lock()
    return exit_code


def _run(args, config, cleaner_version, init_dir, out_dir, loss_log_path, mx):
    from gsovits_mlx.io import load_mlx_safetensors
    from gsovits_mlx.train import ckpt as ckpt_mod
    from gsovits_mlx.train.optim import ScaledAdam
    from gsovits_mlx.train.schedule import WarmupCosineLRSchedule
    from gsovits_mlx.train.s1_data import (Text2SemanticDataset,
                                           BucketBatchSampler)
    from gsovits_mlx.train.s1_model_train import S1TrainModel

    seed_everything(args.seed)
    max_sec = config["data"]["max_sec"]

    # --- data ---
    phoneme_path = os.path.join(args.exp_dir, "2-name2text.txt")
    semantic_path = os.path.join(args.exp_dir, "6-name2semantic.tsv")
    dataset = Text2SemanticDataset(
        phoneme_path=phoneme_path, semantic_path=semantic_path,
        version=cleaner_version, max_sec=max_sec,
        pad_val=config["data"]["pad_val"])
    sampler = BucketBatchSampler(dataset, batch_size=args.batch_size,
                                 shuffle=True, seed=0)

    # --- model (fp32 masters) ---
    weights = dict(load_mlx_safetensors(
        os.path.join(init_dir, "gpt.safetensors")))
    weights = {k: v.astype(mx.float32) for k, v in weights.items()}
    model = S1TrainModel(config,
                         use_checkpoint=bool(args.grad_ckpt)).load(weights)
    params = weights

    # --- optimizer (official configure_optimizers defaults) ---
    optim = ScaledAdam(
        params, lr=args.lr, betas=(0.9, 0.95),
        scalar_lr_scale=0.1, clipping_scale=2.0,
        clipping_update_period=1000, size_update_period=4)
    scheduler = WarmupCosineLRSchedule(
        optim,
        init_lr=config["optimizer"]["lr_init"],
        peak_lr=config["optimizer"]["lr"],
        end_lr=config["optimizer"]["lr_end"],
        warmup_steps=config["optimizer"]["warmup_steps"],
        total_steps=config["optimizer"]["decay_steps"])

    def loss_only(p, batch):
        return model.forward(p, batch)[0]

    vg = mx.value_and_grad(loss_only)
    use_ckpt = getattr(model, "use_gradient_checkpoint", False)

    from gsovits_mlx.train.loop import FootprintSampler
    fp_sampler = FootprintSampler(interval=0.3)
    fp_sampler.start()

    def mem_row():
        try:
            active = int(mx.metal.get_active_memory())
            cache = int(mx.metal.get_cache_memory())
        except Exception:
            active = cache = -1
        return {"mx_active": active, "mx_cache": cache,
                "peak_footprint": fp_sampler.peak}

    # --- official loop: backward every batch, step at batch_idx%4==0 (>0) ---
    optim_steps = 0
    batch_counter = 0
    t0 = time.perf_counter()
    last_lock_refresh = t0
    log_rows = []
    skipped_long = 0
    stop = False
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        batches = sampler.batch_indices()
        grads_acc: dict | None = None
        for batch_idx, indices in enumerate(batches):
            batch = dataset.collate([dataset[i] for i in indices])
            if (args.max_src_bucket is not None and
                    (batch["phoneme_ids"].shape[1]
                     + batch["semantic_ids"].shape[1]) > args.max_src_bucket):
                skipped_long += 1
                continue
            if use_ckpt:
                loss_f, _acc, grads = model.train_loss_and_grads(params, batch)
                grads = {k: g.astype(mx.float32) for k, g in grads.items()}
            else:
                loss, grads = vg(params, batch)
                # Eager-eval loss AND grads: an unevaluated grad is a lazy graph
                # node retaining the full forward activations of this micro-batch;
                # accumulating 4 such windows retained ~1GB/step (probe evidence:
                # phys_footprint +1.07GB/step with Metal capped). Evaluating each
                # grad + the accumulated sum severs those references.
                grads = {k: g.astype(mx.float32) for k, g in grads.items()}
            if grads_acc is None:
                grads_acc = grads
            else:
                for k in grads_acc:
                    grads_acc[k] = grads_acc[k] + grads[k]
            if use_ckpt:
                mx.eval(*grads_acc.values())
            else:
                mx.eval(loss, *grads_acc.values())
                loss_f = float(loss)
            # official accumulation: grads sum (no averaging; sum loss)
            batch_counter += 1
            row = {"batch": batch_counter, "epoch": epoch,
                   "batch_idx": batch_idx, "loss": loss_f, **mem_row()}
            log_rows.append(row)
            if batch_counter % 10 == 1 or batch_counter == 1:
                el = time.perf_counter() - t0
                print(f"[e{epoch} b{batch_idx}] loss={loss_f:.1f} "
                      f"({el:.1f}s elapsed)", flush=True)
            # torch manual_backward + step gate: batch_idx>0 and %4==0
            if batch_idx > 0 and batch_idx % args.grad_accum == 0:
                optim.apply_group(0, grads_acc)
                # NOTE: no assign-back needed — since main 1c58f1d the
                # optimizers keep the caller's params dict by reference and
                # apply_group mutates it in place (a per-step dict() copy
                # here would reintroduce one-step-stale forwards).
                grads_acc = None
                scheduler.step()
                optim_steps += 1
                row["optim_step"] = optim_steps
                now = time.perf_counter()
                if now - last_lock_refresh > 600:  # law: refresh <=10 min
                    refresh_train_lock()
                    last_lock_refresh = now
                if args.save_every and optim_steps % args.save_every == 0:
                    ckpt_mod.save_resume(
                        os.path.join(out_dir, "resume"), model,
                        [optim], optim_steps, epoch)
                if args.export_every and optim_steps % args.export_every == 0:
                    ckpt_mod.save_s1_inference(
                        os.path.join(out_dir, f"export_step{optim_steps}"),
                        model, config, epoch)
                if args.max_steps is not None and optim_steps >= args.max_steps:
                    stop = True
                    break
            if args.max_batches is not None and batch_counter >= args.max_batches:
                stop = True
                break
        if stop:
            break

    # final export
    export_dir = os.path.join(out_dir, "export")
    ckpt_mod.save_s1_inference(export_dir, model, config, 0)
    fp_sampler.stop()
    mem = mem_row()
    with open(loss_log_path, "w") as f:
        for row in log_rows:
            f.write(json.dumps(row) + "\n")

    losses = [r["loss"] for r in log_rows]
    summary = {
        "batches": batch_counter,
        "optim_steps": optim_steps,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_mean_first20": float(np.mean(losses[:20])) if losses else None,
        "loss_mean_last20": float(np.mean(losses[-20:])) if losses else None,
        "wall_s": time.perf_counter() - t0,
        "export_dir": export_dir,
        "skipped_long_batches": skipped_long,
        "peak_mem": mem,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
