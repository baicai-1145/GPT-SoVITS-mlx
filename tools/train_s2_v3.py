#!/usr/bin/env python3
"""s2 v3/v4/v5dev/v5turbo CFM+LoRA trainer (MLX port of s2_train_v3_lora.py).

Official semantics (GPT-SoVITS-cuda_graph_accel_v5, read-only reference):

* data: TextAudioSpeakerLoaderV3 (v3) / V4 (v4/v5dev/v5turbo) + collate
  V3/V4 (gsovits_mlx/train/s2_v3_data.py; exp_dir from tools/prepare_data.py).
* model: SynthesizerTrnV3.forward with freeze_quantizer=True — CFM MSE loss
  on the flow-matching velocity; ssl_proj/quantizer/enc_p frozen;
  ref_enc/bridge/wns1/linear_mel train full-model. LoRA rank-32 (default)
  adapters on the DiT attention to_k/to_q/to_v/to_out.0 (scaling 1.0,
  peft-default init); base DiT frozen. --no-lora trains the DiT full-model.
* loop: AdamW(lr 1e-4, betas (0.8, 0.99), eps 1e-9) on TRAINABLE params
  only, fp16 GradScaler, ExponentialLR gamma=0.999875 per epoch, no
  discriminator. Checkpoints + merged-weight sovits export
  (pipeline.load_sovits_v3-loadable) under --out.

Usage:
  python3 tools/train_s2_v3.py --version v3 --exp-dir .tmp/train_data/exp \
      --out .tmp/train_s2_v3_smoke --steps 200 --batch-size 3 --gpu

GPU lock discipline: --gpu (or GSOVITS_GPU_LOCK_OK=1) with .tmp/gpu.lock.d
held; CPU runs are allowed for smoke tests (numerically valid, slower).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

DEFAULT_MODELS_ROOT = os.environ.get(
    "GSOVITS_MODELS_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "models_local"))

_REFRESH_LOCK = None  # set when --gpu: long-run freshness keeper


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", required=True,
                   choices=["v3", "v4", "v5dev", "v5turbo"])
    p.add_argument("--exp-dir", required=True,
                   help="preprocessing exp_dir (2-name2text.txt, 4-cnhubert/,"
                        " 5-wav32k/) from tools/prepare_data.py")
    p.add_argument("--models-root", default=DEFAULT_MODELS_ROOT)
    p.add_argument("--sovits-dir", default=None,
                   help="weights subdir under --models-root (default: version)")
    p.add_argument("--out", required=True, help="output dir (ckpts + export)")
    p.add_argument("--steps", type=int, default=200,
                   help="optimizer steps for a smoke run (0 = whole epochs)")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=3,
                   help="official default minmem//8 (24GB->3)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=50,
                   help="linear lr warmup (official warmup_epochs analog; "
                        "lr=0 at step 0 -> args.lr at warmup end)")
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--no-lora", action="store_true",
                   help="train the DiT base weights too (experiment)")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--loss-log", default=None)
    p.add_argument("--resume", default=None, help="resume checkpoint dir")
    p.add_argument("--export-only", default=None,
                   help="skip training; merge LoRA from this resume dir and"
                        " export (pass the checkpoint dir)")
    p.add_argument("--max-batch-mel", type=int, default=None,
                   help="skip batches whose collated mel width exceeds this"
                        " (smoke memory aid; frames)")
    p.add_argument("--length-buckets", action="store_true",
                   help="OPT-IN length-bucketed batch sampler (official "
                        "DistributedBucketSampler semantics on the CFM "
                        "loader): similar-length batches, big Metal-cache "
                        "win. Default off = official flat shuffle order.")
    p.add_argument("--pad-multiple", type=int, default=None,
                   help="OPT-IN pad quantization: round collated ssl/spec/"
                        "mel widths up to this frame multiple (e.g. 64). "
                        "Extra pad frames are masked out of the loss by "
                        "x_lens/mel_lengths slicing (loss-neutral); the "
                        "allocator reuses a small set of buffer sizes. "
                        "Default None = official collate widths.")
    p.add_argument("--dit-fp32", action="store_true",
                   help="force fp32 fused DiT forward (debug only; banned at "
                        "max length on this machine — OOM'd twice)")
    p.add_argument("--dit-ckpt", dest="dit_ckpt", action="store_true",
                   default=True,
                   help="two-pass checkpointed DiT backward (memory-safe "
                        "default; grad-verified vs fused)")
    p.add_argument("--no-dit-ckpt", dest="dit_ckpt", action="store_false")
    p.add_argument("--dit-dtype", default=None, choices=["fp32", "fp16", "bf16"],
                   help="EXPERIMENTAL DiT compute dtype override. Default: "
                        "fp32 masters with ckpt mode (fp16 backward "
                        "overflows — banned). bf16 keeps fp32's exponent "
                        "range (no overflow) at half the GEMM traffic; "
                        "gated experiment, see .tmp/TRAINING.md addendum.")
    p.add_argument("--dump-batches", default=None,
                   help="dump collated batches as npz into this dir and exit"
                        " (for the torch CPU reference driver)")
    p.add_argument("--gpu", action="store_true")
    return p


def main(argv=None) -> None:
    args = build_argparser().parse_args(argv)

    from gsovits_mlx.gpu_lock import resolve_device, acquire_lock, \
        refresh_lock, release_lock
    if args.gpu:
        # NEW LOCK LAW (f9185a6): canonical main-repo lock only; refuse to
        # run alongside any other GPU python process.
        import subprocess as _sp
        procs = _sp.run(["ps", "ax", "-o", "command"], capture_output=True,
                        text=True).stdout.splitlines()
        gpu_procs = [ln.strip() for ln in procs
                     if ("train" in ln or "smoke" in ln or "prepare" in ln)
                     and "python" in ln and "train_s2_v3" not in ln
                     and "grep" not in ln]
        if gpu_procs:
            raise SystemExit("[gpu.lock] other GPU python processes running:\n  "
                             + "\n  ".join(gpu_procs[:5]))
        acquire_lock("train_s2_v3 task-5 s2-cfm")
        global _REFRESH_LOCK
        _REFRESH_LOCK = refresh_lock
    device = resolve_device(args.gpu, verbose=True)
    if device == "gpu":
        import mlx.core as mx
        try:
            mx.metal.set_memory_limit(8 * 1024 * 1024 * 1024)
        except Exception:
            pass
        print("[metal] memory limit set to 8GB", flush=True)

    import numpy as np
    import mlx.core as mx
    from mlx.utils import tree_map, tree_flatten
    from gsovits_mlx.pipeline import load_sovits_v3
    from gsovits_mlx.train.lora import inject_lora
    from gsovits_mlx.train.s2_cfm import S2V3TrainModel, upcast_training_model
    from gsovits_mlx.train.s2_v3_data import (TextAudioSpeakerLoaderV3V4,
                                              bucket_batches, collate_for,
                                              LengthBucketSampler)
    from gsovits_mlx.train.optim import AdamW
    from gsovits_mlx.train.mixed_precision import GradScaler
    from gsovits_mlx.train.loop import Trainer
    from gsovits_mlx.train.ckpt import save_resume

    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed + 1)   # CFM python rng (2-step branch)

    # -- data -----------------------------------------------------------------------
    # The vendored official text package needs the CPUFast repo as cwd for
    # its relative scratch paths (g2pw/japanese); bootstrap chdirs there.
    # Dataset construction happens here (frontend use only); the worktree
    # cwd is restored right after (AGENTS.md bootstrap-chdir rule: long-lived
    # services must not keep the CPUFast cwd or gpu.lock root resolution
    # breaks).
    from gsovits_mlx.text.preproc import bootstrap
    bootstrap(os.environ.get("GSOVITS_CPUFAST",
                             "/Users/baicai1145/repos/gpt-sovits/GPT-SoVITS-CPUFast"),
              models_root=args.models_root)
    dataset = TextAudioSpeakerLoaderV3V4(args.exp_dir, args.version)
    collate = collate_for(args.version, pad_multiple=args.pad_multiple)
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # -- model ------------------------------------------------------------------
    sovits_dir = os.path.join(args.models_root, args.sovits_dir or args.version)
    model, meta = load_sovits_v3(sovits_dir, args.version)
    # fp32 trunk/LoRA masters + fp16 DiT (official autocast fp16_run=true)
    # ckpt mode NEEDS fp32 DiT weights (fp16 backward overflows — measured);
    # fp16 is only valid for the fused-forward path which is banned at max
    # length anyway. Keep --dit-fp32 as an explicit override for fused mode.
    # --dit-dtype bf16 (gated experiment): bf16 has fp32's exponent range so
    # the BACKWARD cannot overflow; LoRA masters stay fp32 (optimizer), the
    # per-block GEMMs run bf16 with the fp32 residual stream.
    dit_dtype = {"fp32": mx.float32, "fp16": mx.float16,
                 "bf16": mx.bfloat16}.get(args.dit_dtype)
    upcast_training_model(
        model,
        dit_fp16=(not args.dit_ckpt and not args.dit_fp32
                  and dit_dtype is None),
        dit_dtype=dit_dtype)
    if dit_dtype is not None and args.dit_dtype == "bf16":
        print("[mode] bf16 DiT experiment: bf16 weights + fp32 stream "
              "(grad-overflow-safe by exponent range)", flush=True)

    adapters = None if args.no_lora else \
        inject_lora(model.cfm.estimator, rank=args.lora_rank, seed=args.seed)
    tm = S2V3TrainModel(model, args.version, lora_adapters=adapters, rng=rng)
    params = tm.trainable_params()
    opt = AdamW([dict(params)], lr=args.lr, betas=(0.8, 0.99), eps=1e-9,
                weight_decay=0.0)
    # the optimizer owns the fp32 MASTERS (its param_groups copy); keep the
    # adapter objects synced so checkpoints/exports read current values
    masters = opt.param_groups[0]["params"]

    def sync_adapters():
        if tm.use_lora:
            for a in tm.adapters:
                a.lora_A = masters[f"{a.name}.lora_A"]
                a.lora_B = masters[f"{a.name}.lora_B"]

    n_par = sum(p.size for p in masters.values())
    print(f"trainable: {len(masters)} tensors, {n_par/1e6:.2f}M params "
          f"(lora={not args.no_lora}, rank={args.lora_rank})", flush=True)
    scaler = GradScaler()

    os.makedirs(args.out, exist_ok=True)
    loss_log = args.loss_log or os.path.join(args.out, "loss.jsonl")
    fp_log_path = os.path.join(args.out, "footprint_steps.jsonl")
    fp_log = open(fp_log_path, "a", buffering=1)

    def step_mem_rec(step: int, loss: float, t_step: float) -> dict:
        """One per-step memory record: phys_footprint + Metal counters."""
        try:
            fp = FootprintSampler.read_phys_footprint(os.getpid())
        except Exception:
            fp = 0
        try:
            act = int(mx.metal.get_active_memory())
            cache = int(mx.metal.get_cache_memory())
        except Exception:
            act = cache = 0
        return {"step": step, "loss": loss, "footprint": fp,
                "metal_active": act, "metal_cache": cache,
                "metal_active_plus_cache": act + cache, "t_step": t_step}

    def log_step_footprint(rec: dict, flush: bool = False):
        """Append to footprint_steps.jsonl. cache+active steady state is the
        allocator-recycling gate for --length-buckets/--pad-multiple
        (task cfm-mem-opt)."""
        fp_log.write(json.dumps(rec) + "\n")
        if flush:
            fp_log.flush()
            os.fsync(fp_log.fileno())

    # -- export-only path ----------------------------------------------------------
    if args.export_only:
        from gsovits_mlx.train.ckpt import load_resume
        man = load_resume(args.export_only, masters, [opt])
        tm.apply_params(masters)
        sync_adapters()
        out = export_merged(args, tm, meta, args.export_only.rstrip("/")
                            + "_export")
        print(f"export-only: merged weights -> {out}")
        return

    # -- data -----------------------------------------------------------------------
    # loaded ABOVE (before the model) because the frontend bootstrap chdirs
    # into the CPUFast checkout; the loop below only reads dataset[i].
    def epoch_batches(epoch: int):
        """Batch index lists for one epoch (official flat order or the
        opt-in length-bucket sampler; both deterministic per seed/epoch)."""
        if args.length_buckets:
            sampler = LengthBucketSampler(dataset.lengths, args.batch_size,
                                          seed=args.seed)
            sampler.set_epoch(epoch)
            return sampler.batch_indices()
        return bucket_batches(dataset.lengths, args.batch_size,
                              seed=args.seed + epoch)

    if args.dump_batches:
        import numpy as _np
        os.makedirs(args.dump_batches, exist_ok=True)
        n = 0
        for bidx in epoch_batches(0):
            batch = collate([dataset[i] for i in bidx])
            batch = collate([dataset[i] for i in bidx])
            if args.max_batch_mel and batch.mel.shape[-1] > args.max_batch_mel:
                continue
            _np.savez(os.path.join(args.dump_batches, f"batch_{n:03d}.npz"),
                      ssl=batch.ssl, spec=batch.spec, mel=batch.mel,
                      ssl_lengths=batch.ssl_lengths,
                      spec_lengths=batch.spec_lengths,
                      text=batch.text.astype(np.int64),
                      text_lengths=batch.text_lengths,
                      mel_lengths=batch.mel_lengths)
            n += 1
            if n >= max(args.steps, 20):
                break
        print(f"dumped {n} batches -> {args.dump_batches}")
        return
    batch_holder = {}

    use_ckpt = args.dit_ckpt and tm.use_lora
    if use_ckpt:
        print("[mode] two-pass checkpointed DiT backward (memory-safe; "
              "grad-verified vs fused: bit-exact loss, grads <=1e-6)",
              flush=True)

    def loss_of(pdict: dict) -> mx.array:
        tm.apply_params(pdict)
        b = batch_holder["b"]
        loss, _ = tm.forward(
            mx.array(b.ssl), mx.array(b.spec), mx.array(b.mel),
            mx.array(b.ssl_lengths), mx.array(b.spec_lengths),
            mx.array(b.text.astype(np.int32)), mx.array(b.text_lengths),
            mx.array(b.mel_lengths, dtype=mx.float32))
        return loss

    def forward_fn(p16, batch):
        batch_holder["b"] = batch
        return loss_of(p16)

    value_and_grad = mx.value_and_grad(loss_of)

    def backward_fn(p16, loss):
        _, grads = value_and_grad(masters)
        return grads

    trainer = Trainer(None, [opt], scaler=(None if use_ckpt else scaler),
                      log_interval=1, loss_log_path=loss_log)

    start_step = 0
    if args.resume:
        from gsovits_mlx.train.ckpt import load_resume as _lr
        man = _lr(args.resume, masters, [opt])
        tm.apply_params(masters)
        sync_adapters()
        start_step = man["step"]
        trainer.step = start_step
        print(f"resumed at step {start_step}")

    # -- loop ------------------------------------------------------------------------
    from gsovits_mlx.train.loop import FootprintSampler
    t0 = time.time()
    step = start_step
    n_bad = 0
    max_steps = args.steps if args.steps > 0 else None
    stop = False
    for epoch in range(args.epochs):
        if stop:
            break
        for bidx in epoch_batches(epoch):
            batch = collate([dataset[i] for i in bidx])
            if args.max_batch_mel and batch.mel.shape[-1] > args.max_batch_mel:
                continue
            if use_ckpt and args.warmup_steps > 0 and step < args.warmup_steps:
                opt.param_groups[0]["lr"] = (
                    args.lr * (step + 1) / args.warmup_steps)
            elif use_ckpt:
                opt.param_groups[0]["lr"] = args.lr
            if use_ckpt:
                loss_f, info, grads = tm.train_loss_and_grads(masters, batch)
                import math as _math
                if not _math.isfinite(loss_f) or any(
                        _math.isnan(float(mx.abs(g).max())) for g in grads.values()):
                    n_bad += 1
                    print(f"[warn] step {step+1}: non-finite loss/grads — "
                          f"skipping batch (mel_lengths={batch.mel_lengths}, "
                          f"melT={batch.mel.shape[-1]})", flush=True)
                    if os.environ.get("GSOVITS_DUMP_NAN"):
                        np.savez(os.path.join(
                            os.environ["GSOVITS_DUMP_NAN"],
                            f"nanbatch_step{step+1}.npz"),
                            ssl=batch.ssl, spec=batch.spec, mel=batch.mel,
                            ssl_lengths=batch.ssl_lengths,
                            spec_lengths=batch.spec_lengths, text=batch.text,
                            text_lengths=batch.text_lengths,
                            mel_lengths=batch.mel_lengths)
                    if n_bad > 10:
                        raise SystemExit("too many non-finite batches")
                    continue
                opt.set_grads(0, grads)
                opt.step()
                sync_adapters()
                loss = loss_f
                trainer.step += 1
                trainer._log(loss)
                try:
                    mx.clear_cache()
                except Exception:
                    pass
                t_step = time.time() - t0
                rec = step_mem_rec(step + 1, float(loss), t_step)
                log_step_footprint(rec, flush=(step + 1) % 10 == 0)
                if rec["footprint"] > 8 * 1024 * 1024 * 1024:
                    release_lock()
                    raise SystemExit(
                        f"[abort] step {step} footprint "
                        f"{rec['footprint']/1e9:.2f} GB > 8GB "
                        "gate — aborting before machine risk")
            else:
                loss = trainer.train_step(batch, forward_fn, backward_fn)
                sync_adapters()
                rec = step_mem_rec(step + 1, float(loss), time.time() - t0)
                log_step_footprint(rec, flush=(step + 1) % 10 == 0)
            step += 1
            if _REFRESH_LOCK is not None and step % 40 == 0:
                _REFRESH_LOCK()  # 10-min freshness for long runs
            if step % 10 == 0:
                el = time.time() - t0
                print(f"step {step} loss {loss:.4f} "
                      f"({el/max(step-start_step,1):.2f}s/step)", flush=True)
            if max_steps and step - start_step >= max_steps:
                stop = True
                break
        # per-epoch ExponentialLR (official scheduler_g.step() per epoch)
        for g in opt.param_groups:
            g["lr"] *= 0.999875

    trainer.stop_footprint_sampler()
    mem = trainer.memory_report()
    fp_log.close()

    save_resume(os.path.join(args.out, "resume_ckpt"), masters, [opt],
                step=trainer.step, epoch=trainer.epoch,
                extra={"version": args.version, "lora": not args.no_lora,
                       "lora_rank": args.lora_rank})
    export_merged(args, tm, meta)

    with open(os.path.join(args.out, "memory_report.json"), "w") as f:
        json.dump({k: int(v) for k, v in mem.items()}, f, indent=1)
    print(json.dumps(mem, indent=1))
    print(f"done: {trainer.step} steps in {time.time()-t0:.1f}s -> {args.out}",
          flush=True)
    if _REFRESH_LOCK is not None:
        release_lock()


def export_merged(args, tm, meta, out_dir: str | None = None) -> str:
    """Merge LoRA into the base DiT weights; write a loadable sovits dir."""
    from gsovits_mlx.train.ckpt import save_s2v3_merged
    from gsovits_mlx.train.lora import merged_training_weights

    out = out_dir or os.path.join(args.out, "export_merged")
    merged = merged_training_weights(tm.model.cfm.estimator, tm.adapters) \
        if tm.use_lora else None
    save_s2v3_merged(out, tm.model, merged, dict(meta["model_hps"]),
                     args.version)
    print(f"exported merged weights -> {out}", flush=True)
    return out


if __name__ == "__main__":
    main()
