"""Gated memory probe for the s2 v3 trainer (lead directive after 2 OOMs).

3 steps on the LONGEST batch in the dataset (max mel length, batch=1),
printing per-step peak_footprint (phys_footprint via /usr/bin/footprint),
mx active memory, and mx cache. GATE: peak < 9GB AND flat across steps.

Runs the REAL trainer loop path (Trainer.train_step with fp16 working
copies + GradScaler + eager grad eval), not a synthetic forward.
"""

import argparse
import os
import subprocess
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def footprint_mb(pid: int) -> float:
    try:
        out = subprocess.run(["/usr/bin/footprint", str(pid)],
                             capture_output=True, text=True, timeout=15).stdout
        for ln in out.splitlines():
            if "Footprint:" in ln:
                val, unit = ln.split("Footprint:")[1].strip().split()[:2]
                mul = {"B": 1/1e6, "KB": 1/1e3, "MB": 1.0, "GB": 1e3,
                       "TB": 1e6}[unit]
                return float(val) * mul
    except Exception:
        pass
    return 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--version", default="v3")
    p.add_argument("--exp-dir", required=True)
    p.add_argument("--models-root", default=os.path.join(_REPO, "models_local"))
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--mode", default="fp32", choices=["fp32", "fp16"],
                   help="DiT forward precision (fp16 = official autocast "
                        "fp16_run semantics)")
    args = p.parse_args()

    from gsovits_mlx.gpu_lock import acquire_lock, resolve_device, release_lock
    import subprocess as _sp
    _procs = _sp.run(["ps", "ax", "-o", "command"], capture_output=True,
                     text=True).stdout.splitlines()
    _other = [ln.strip() for ln in _procs
              if ("train" in ln or "smoke" in ln or "prepare" in ln)
              and "python" in ln and "mem_probe" not in ln and "grep" not in ln]
    if _other:
        raise SystemExit("[gpu.lock] other GPU python processes:\n  "
                         + "\n  ".join(_other[:5]))
    acquire_lock("mem_probe train_s2_v3 task-5 s2-cfm")
    device = resolve_device(True, verbose=True)
    assert device == "gpu", "probe must run on GPU"
    import mlx.core as mx
    try:
        mx.set_memory_limit(8 * 1024 * 1024 * 1024)
    except Exception:
        pass

    import random
    import numpy as np
    from mlx.utils import tree_map
    from gsovits_mlx.pipeline import load_sovits_v3
    from gsovits_mlx.train.lora import inject_lora
    from gsovits_mlx.train.s2_cfm import (S2V3TrainModel,
                                          upcast_training_model)
    from gsovits_mlx.train.s2_v3_data import (TextAudioSpeakerLoaderV3V4,
                                              collate_for)
    from gsovits_mlx.train.optim import AdamW
    from gsovits_mlx.train.mixed_precision import GradScaler
    from gsovits_mlx.train.loop import Trainer

    # bootstrap frontend (chdirs!)
    from gsovits_mlx.text.preproc import bootstrap
    bootstrap(os.environ.get("GSOVITS_CPUFAST",
                             "/Users/baicai1145/repos/gpt-sovits/GPT-SoVITS-CPUFast"),
              models_root=args.models_root)
    dataset = TextAudioSpeakerLoaderV3V4(args.exp_dir, args.version)
    collate = collate_for(args.version)
    os.chdir(_REPO)

    # LONGEST batch: max mel length sample, batch=1
    longest = int(np.argmax(dataset.lengths))
    print(f"longest sample idx {longest} len {dataset.lengths[longest]}")
    batch = collate([dataset[longest]])

    sovits_dir = os.path.join(args.models_root, args.version)
    if not os.path.exists(os.path.join(sovits_dir, "sovits.safetensors")):
        sovits_dir = os.path.join(
            "/Users/baicai1145/repos/gpt-sovits/GPT-SoVITS-mlx/models_local",
            args.version)
    model, meta = load_sovits_v3(sovits_dir, args.version)
    upcast_training_model(model, dit_fp16=(args.mode == "fp16"))

    adapters = inject_lora(model.cfm.estimator, rank=32, seed=1234)
    tm = S2V3TrainModel(model, args.version, lora_adapters=adapters,
                        rng=random.Random(0))

    trainable = tm.trainable_params()
    masters = {k: v.astype(mx.float32) for k, v in trainable.items()}
    opt = AdamW(masters, lr=1e-4, betas=(0.8, 0.99), eps=1e-9, weight_decay=0.0)
    scaler = GradScaler()
    batch_holder = {}

    def loss_of(pdict):
        tm.apply_params(pdict)
        b = batch_holder["b"]
        loss, _ = tm.forward(
            mx.array(b.ssl), mx.array(b.spec), mx.array(b.mel),
            mx.array(b.ssl_lengths), mx.array(b.spec_lengths),
            mx.array(b.text.astype(np.int32)), mx.array(b.text_lengths),
            mx.array(b.mel_lengths, dtype=mx.float32))
        return loss

    def forward_fn(p16, b):
        batch_holder["b"] = b
        return loss_of(p16)

    vg = mx.value_and_grad(loss_of)

    def backward_fn(p16, loss):
        _, g = vg(masters)
        return g

    trainer = Trainer(None, [opt], scaler=scaler, log_interval=1)
    pid = os.getpid()
    peaks = []
    for i in range(args.steps):
        t0 = time.time()
        loss = trainer.train_step(batch, forward_fn, backward_fn)
        # adapter arrays live on LoRALinear objects; re-sync masters
        for a in adapters:
            a.lora_A = masters[f"{a.name}.lora_A"]
            a.lora_B = masters[f"{a.name}.lora_B"]
        fp = footprint_mb(pid)
        peaks.append(fp)
        try:
            act = mx.get_active_memory() / 1e9
            cache = mx.get_cache_memory() / 1e9
        except Exception:
            act = cache = 0.0
        try:
            mx.clear_cache()
        except Exception:
            pass
        print(f"step {i}: loss {float(loss):.6f} | footprint {fp:.0f} MB | "
              f"mx_active {act:.2f} GB | mx_cache {cache:.2f} GB | "
              f"{time.time()-t0:.1f}s", flush=True)

    ok = all(p < 9000 for p in peaks) and (max(peaks) - min(peaks)) < 1500
    print(f"GATE ({args.mode}): peak {max(peaks):.0f} MB, spread "
          f"{max(peaks)-min(peaks):.0f} MB -> "
          + ("PASS" if ok else "FAIL"))
    release_lock()


if __name__ == "__main__":
    main()
