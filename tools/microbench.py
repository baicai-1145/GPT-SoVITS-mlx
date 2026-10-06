"""Per-component micro-benchmarks (task-5, P2-3).

Fixed captured inputs, NO text front-end, NO HuBERT in the timed path.
One component per process; run under the gpu.lock, ps-clean, for wall-times.

Components
----------
GPT (AR decode, batch=1, seeded sampler identical to e2e):
    gpt-s1v1   s1bert25hz-2kh  (v1)
    gpt-s1v2   s1bert25hz-5kh  (v2/v2Pro/v2ProPlus)
    gpt-s1v3   s1v3.ckpt       (v3/v4/v5dev/v5turbo)
    metrics: tok/s over one full seeded decode (>=110 tokens) + pure-forward
    replay of the captured token stream (teacher-forced steps, no sampling).

SoVITS decoders (fixed codes+refer, ~110 generated tokens):
    sovits-v1 v2 v2pro v2proplus   HiFiGAN direct 32 kHz
    sovits-v3                      CFM 32-step + BigVGAN 24 kHz
    sovits-v4                      CFM 32-step + Generator 48 kHz
    sovits-v5dev                   CFMV5 rolling 32-step cfg 1.30
    sovits-v5turbo                 CFMV5 4-step cfg 0.0
    metric: decode latency (s) + golden-output max|diff| for parity.

Usage
-----
  python3 tools/microbench.py capture --component gpt-s1v3   # under lock
  python3 tools/microbench.py bench   --component gpt-s1v3 --iters 3
  python3 tools/microbench.py bench   --all

Captured artifacts live in .tmp/mb/<component>.npz (+ .meta.json). bench mode
asserts parity against the captured golden output before reporting timing.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

DEFAULT_MODELS_ROOT = "/Volumes/2T/gpt-sovits-models/mlx"
DEFAULT_REF_AUDIO = "/Volumes/2T/gpt-sovits-models/bench/ref_zh_3.5s.wav"
DEFAULT_REF_TEXT = "希望你以后能够做得比我还好哟。"
DEFAULT_TEXT = "你好，欢迎来到各自的旅程。今天我们聊聊机器学习。"
MB_DIR = os.path.join(REPO, ".tmp", "mb")

GPT_COMPONENTS = {
    "gpt-s1v1": "s1v1",
    "gpt-s1v2": "s1v2",
    "gpt-s1v3": "s1",
}
SOVITS_COMPONENTS = ["sovits-v1", "sovits-v2", "sovits-v2pro", "sovits-v2proplus",
                     "sovits-v3", "sovits-v4", "sovits-v5dev", "sovits-v5turbo"]
# SoVITS decoder version + loader kwargs per component
SOVITS_SPEC = {
    "sovits-v1": ("v1",), "sovits-v2": ("v2",),
    "sovits-v2pro": ("v2Pro",), "sovits-v2proplus": ("v2ProPlus",),
    "sovits-v3": ("v3",), "sovits-v4": ("v4",),
    "sovits-v5dev": ("v5dev",), "sovits-v5turbo": ("v5turbo",),
}


def _repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        gd = subprocess.run(["git", "rev-parse", "--git-common-dir"], cwd=here,
                            capture_output=True, text=True, timeout=5).stdout.strip()
        if gd:
            return os.path.dirname(os.path.abspath(gd))
    except Exception:
        pass
    return os.path.dirname(here)


def gpu_lock_guard(allow_unlocked: bool) -> None:
    """Same lock discipline as the e2e scripts (owner file must name
    tools/microbench.py)."""
    lock_dir = os.path.join(_repo_root(), ".tmp", "gpu.lock.d")
    if not os.path.isdir(lock_dir) or allow_unlocked:
        return
    owner = ""
    p = os.path.join(lock_dir, "owner")
    if os.path.isfile(p):
        owner = open(p).read()
    if "tools/microbench.py" in owner:
        return
    sys.exit(f"GPU lock held (owner={owner.strip()!r}); claim it with a tag "
             f"naming tools/microbench.py, or pass --allow-unlocked.")


def _frontend_inputs(models_root: str, cpufast_repo: str) -> dict:
    """Canonical phones/bert via the CPU-only TextFrontend (same as e2e)."""
    from gsovits_mlx.text.preproc import TextFrontend, bootstrap

    bootstrap(cpufast_repo, models_root=models_root)
    fe = TextFrontend(models_root=models_root, lazy=True, device="cpu")
    p = fe.segment_prompt(DEFAULT_REF_TEXT, "zh", "v2")
    t = fe.preprocess(DEFAULT_TEXT, "zh", "cut0", "v2")
    return {
        "p_ids": np.asarray(p.phones, dtype=np.int32),
        "t_ids": np.asarray(t.phones, dtype=np.int32),
        "p_bert": np.asarray(p.bert, dtype=np.float32),
        "t_bert": np.asarray(t.bert, dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# GPT capture / bench
# ---------------------------------------------------------------------------

def capture_gpt(component: str, models_root: str, cpufast_repo: str, out_dir: str) -> None:
    import mlx.core as mx

    from gsovits_mlx.gpt.t2s import Text2SemanticDecoder
    from gsovits_mlx.io import eval_tree, load_mlx_safetensors, release
    from gsovits_mlx.text.hubert import HubertModel
    from gsovits_mlx.pipeline import load_audio_16k

    s1dir = os.path.join(models_root, GPT_COMPONENTS[component])
    fe_inputs = _frontend_inputs(models_root, cpufast_repo)

    # canonical prompt codes via HuBERT + the right quantizer (needs GPU once)
    from gsovits_mlx.text.mel_frontend import spectrogram

    wav16k = load_audio_16k(DEFAULT_REF_AUDIO)
    wav16k = np.concatenate([wav16k, np.zeros(9600, np.float32)])
    hubert_dir = os.path.join(models_root, "hubert")
    hb = HubertModel(load_mlx_safetensors(os.path.join(hubert_dir, "hubert.safetensors")),
                     json.load(open(os.path.join(hubert_dir, "config.json"))))
    h = hb(mx.array(wav16k[None]))
    mx.eval(h)

    from gsovits_mlx.pipeline import _load_sovits_v1v2, load_sovits_v3

    if component == "gpt-s1v1":
        sov, _ = _load_sovits_v1v2(os.path.join(models_root, "v1"), "v1")
    elif component == "gpt-s1v2":
        sov, _ = _load_sovits_v1v2(os.path.join(models_root, "v2"), "v2")
    else:
        sov, _ = load_sovits_v3(os.path.join(models_root, "v5dev"), "v5dev")
    codes = sov.extract_latent(mx.transpose(h, (0, 2, 1)))
    mx.eval(codes)
    prompt_sem = mx.array(codes.reshape(codes.shape[0], -1), mx.int32)
    del sov, hb, h
    from gsovits_mlx.io import release as _rel
    _rel()

    arrays = load_mlx_safetensors(os.path.join(s1dir, "gpt.safetensors"))
    meta = json.load(open(os.path.join(s1dir, "gpt.json")))
    gpt = Text2SemanticDecoder(meta["config"])
    gpt.load(dict(arrays))
    eval_tree(gpt)
    release(arrays)

    all_phones = mx.array([list(fe_inputs["p_ids"]) + list(fe_inputs["t_ids"])], mx.int32)
    all_bert = mx.array(np.concatenate([fe_inputs["p_bert"], fe_inputs["t_bert"]], axis=1))[None]

    t0 = time.perf_counter()
    seq = gpt.infer(all_phones, all_bert, prompt_sem, top_k=15, top_p=1.0,
                    temperature=1.0, repetition_penalty=1.35,
                    key=mx.random.key(0))
    mx.eval(seq)
    dt = time.perf_counter() - t0
    seq_np = np.array(seq)
    n = seq_np.shape[1]
    print(f"[capture] {component}: {n} tokens in {dt:.2f}s ({n/dt:.1f} tok/s)")

    np.savez_compressed(
        os.path.join(out_dir, f"{component}.npz"),
        phones=np.array(all_phones), bert=np.array(all_bert),
        prompt=np.array(prompt_sem), golden=seq_np,
    )
    json.dump({"tokens": n, "decode_s": round(dt, 3),
               "tok_s": round(n / dt, 1)},
              open(os.path.join(out_dir, f"{component}.meta.json"), "w"), indent=1)


def bench_gpt(component: str, models_root: str, iters: int, out_dir: str) -> dict:
    import mlx.core as mx

    from gsovits_mlx.gpt.t2s import Text2SemanticDecoder
    from gsovits_mlx.io import eval_tree, load_mlx_safetensors, release

    z = np.load(os.path.join(out_dir, f"{component}.npz"))
    s1dir = os.path.join(models_root, GPT_COMPONENTS[component])
    arrays = load_mlx_safetensors(os.path.join(s1dir, "gpt.safetensors"))
    meta = json.load(open(os.path.join(s1dir, "gpt.json")))
    gpt = Text2SemanticDecoder(meta["config"])
    gpt.load(dict(arrays))
    eval_tree(gpt)
    release(arrays)

    phones = mx.array(z["phones"])
    bert = mx.array(z["bert"])
    prompt = mx.array(z["prompt"])
    golden = z["golden"]

    result = {"component": component, "iters": iters}
    for i in range(iters):
        t0 = time.perf_counter()
        seq = gpt.infer(phones, bert, prompt, top_k=15, top_p=1.0,
                        temperature=1.0, repetition_penalty=1.35,
                        key=mx.random.key(0))
        mx.eval(seq)
        dt = time.perf_counter() - t0
        seq_np = np.array(seq)
        if i == 0:
            if seq_np.shape != golden.shape or not np.array_equal(seq_np, golden):
                diff = int(np.sum(seq_np != golden)) if seq_np.shape == golden.shape else -1
                print(f"[micro] {component}: PARITY FAIL vs golden "
                      f"(shape {seq_np.shape} vs {golden.shape}, mismatches {diff})")
                result["parity"] = "FAIL"
                result["mismatches"] = diff
            else:
                result["parity"] = "OK"
        result[f"decode_s_{i}"] = round(dt, 3)
        result[f"tok_s_{i}"] = round(seq_np.shape[1] / dt, 1)
    n = golden.shape[1]
    print(f"[micro] {component}: parity={result.get('parity')} "
          f"{n} tok, best {min(result[f'tok_s_{i}'] for i in range(iters))} tok/s "
          f"(decode {min(result[f'decode_s_{i}'] for i in range(iters))}s)")
    return result


# ---------------------------------------------------------------------------
# SoVITS capture / bench
# ---------------------------------------------------------------------------

def capture_sovits(component: str, models_root: str, cpufast_repo: str, out_dir: str) -> None:
    raise SystemExit("sovits capture: wire up after GPT cells land (uses gpt-s1v3 golden codes)")


def bench_sovits(component: str, models_root: str, iters: int, out_dir: str) -> dict:
    raise SystemExit("sovits bench: wire up after capture")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["capture", "bench"])
    ap.add_argument("--component", default=None, choices=list(GPT_COMPONENTS) + SOVITS_COMPONENTS)
    ap.add_argument("--all", action="store_true", help="bench every captured component")
    ap.add_argument("--models-root", default=DEFAULT_MODELS_ROOT)
    ap.add_argument("--cpufast-repo",
                    default=os.environ.get("GPT_SOVITS_CPUFAST",
                                           "/Users/baicai1145/repos/gpt-sovits/GPT-SoVITS-CPUFast"))
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--out-dir", default=MB_DIR)
    ap.add_argument("--json-out", default=None, help="append JSON lines here")
    ap.add_argument("--allow-unlocked", action="store_true")
    args = ap.parse_args()

    gpu_lock_guard(args.allow_unlocked)  # both modes run GPU inference
    os.makedirs(args.out_dir, exist_ok=True)
    results = []

    def emit(r):
        results.append(r)
        if args.json_out:
            with open(args.json_out, "a") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    comps = list(GPT_COMPONENTS) + SOVITS_COMPONENTS if args.all else [args.component]
    if not comps[0]:
        sys.exit("pick --component or --all")
    for comp in comps:
        if comp in GPT_COMPONENTS:
            if args.mode == "capture":
                capture_gpt(comp, args.models_root, args.cpufast_repo, args.out_dir)
            else:
                emit(bench_gpt(comp, args.models_root, args.iters, args.out_dir))
        else:
            if args.mode == "capture":
                capture_sovits(comp, args.models_root, args.cpufast_repo, args.out_dir)
            else:
                emit(bench_sovits(comp, args.models_root, args.iters, args.out_dir))
    if results:
        print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
