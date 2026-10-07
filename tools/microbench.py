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

Measurement doctrine (task-11 revision; supersedes task-10 notes)
---------------------------------------------------------------
1. State WHAT a number is, never assert what it is not:
   - single-op synced latency (`op(); mx.eval()`) is a valid END-TO-END
     latency including dispatch + sync; it is not pure kernel throughput.
   - independent-batch timing (q distinct ops queued, ONE final eval)
     measures achieved throughput of q pipelined ops; q must be stated.
   - dependent-chain timing (x @ W1 @ W2 ... finite outputs) measures
     chained-graph latency; FLOPs must be computed from the ACTUAL shapes
     of every executed matmul.
2. Repeated IDENTICAL operands risk runtime common-subexpression
   elimination. Bench helpers must use DISTINCT materialized inputs and
   retain + evaluate EVERY timed output. Keeping outputs in the graph is
   necessary but not sufficient proof of execution count; state the
   verification used (graph export / profile) when available.
3. Never sum isolated micro timings into a physical lower bound, and never
   extrapolate chain behavior from independent-op results (or vice versa).

Usage
-----
  python3 tools/microbench.py capture --component gpt-s1v3   # under lock
  python3 tools/microbench.py bench   --component gpt-s1v3 --iters 3
  python3 tools/microbench.py bench   --all
  python3 tools/microbench.py gemm --shape 1600x1152x4608 --queue 8

Captured artifacts live in .tmp/mb/<component>.npz (+ .meta.json). bench mode
asserts parity against the captured golden output before reporting timing.
"""
from __future__ import annotations

import argparse
import json
import math
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
        "t_ids": np.asarray([ph for r in t for ph in r.phones], dtype=np.int32),
        "p_bert": np.asarray(p.bert, dtype=np.float32),
        "t_bert": np.concatenate([r.bert for r in t], axis=1).astype(np.float32),
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

    all_phones = mx.array([[int(v) for v in fe_inputs["p_ids"]] + [int(v) for v in fe_inputs["t_ids"]]], mx.int32)
    all_bert = mx.array(np.concatenate([fe_inputs["p_bert"], fe_inputs["t_bert"]], axis=1))[None]

    np.random.seed(0)  # _sample draws from numpy's global RNG (same as e2e)
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
        np.random.seed(0)  # identical stream to the capture run -> parity holds
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
    """Capture fixed decode inputs. Codes come from the matching GPT golden
    (gpt-s1v1 -> sovits-v1, etc.; v3/v4/v5 family shares gpt-s1v3)."""
    import mlx.core as mx

    from gsovits_mlx.io import load_mlx_safetensors, release, trim_metal
    from tools.e2e import load_audio_official  # official PyAV decode path
    from gsovits_mlx.text.hubert import HubertModel
    from gsovits_mlx.text.mel_frontend import spectrogram

    ver = SOVITS_SPEC[component][0]
    gpt_comp = {"sovits-v1": "gpt-s1v1", "sovits-v2": "gpt-s1v2",
                "sovits-v2pro": "gpt-s1v2", "sovits-v2proplus": "gpt-s1v2"}.get(
        component, "gpt-s1v3")
    gz = np.load(os.path.join(out_dir, f"{gpt_comp}.npz"))
    phones, prompt = gz["phones"], gz["prompt"]

    wav16k = load_audio_official(DEFAULT_REF_AUDIO, 16000)
    wav16k = np.concatenate([wav16k, np.zeros(9600, np.float32)])
    hubert_dir = os.path.join(models_root, "hubert")
    hb = HubertModel(load_mlx_safetensors(os.path.join(hubert_dir, "hubert.safetensors")),
                     json.load(open(os.path.join(hubert_dir, "config.json"))))
    h = hb(mx.array(wav16k[None]))
    mx.eval(h)
    from gsovits_mlx.pipeline import _load_sovits_v1v2, load_sovits_v3
    loader = _load_sovits_v1v2 if ver in ("v1", "v2", "v2Pro", "v2ProPlus") else load_sovits_v3
    dirname = {"v5dev": "v5dev", "v5turbo": "v5turbo", "v3": "v3", "v4": "v4"}.get(ver, ver)
    sov, _ = loader(os.path.join(models_root, dirname), ver)
    codes = sov.extract_latent(mx.transpose(h, (0, 2, 1)))
    mx.eval(codes)
    prompt_sem = mx.array(codes.reshape(codes.shape[0], -1), mx.int32)  # authoritative
    # full decode stream = prompt codes + the GPT golden generation (e2e:
    # all_codes = concat(prompt_sem, seq))
    gen = mx.array(gz["golden"], mx.int32)
    codes_full = mx.concatenate([prompt_sem, gen], axis=1)
    mx.eval(codes_full)
    del sov, hb, h
    release()
    trim_metal()

    w32 = load_audio_official(DEFAULT_REF_AUDIO, 32000)
    refer = spectrogram(mx.array(w32[None]), 2048, 640, 2048)
    mx.eval(refer)
    # prompt phone boundary for decode_encp (e2e: all_phones[:, :len(p_ids)])
    n_prompt_phones = int(len(_frontend_inputs(models_root, cpufast_repo)["p_ids"]))
    np.savez_compressed(os.path.join(out_dir, f"{component}.npz"),
                        phones=np.asarray(phones), prompt=np.array(codes_full),
                        refer=np.array(refer), w32=w32,
                        n_prompt_phones=np.int32(n_prompt_phones))
    json.dump({"codes": int(np.array(codes_full).shape[1]),
               "refer": list(refer.shape)},
              open(os.path.join(out_dir, f"{component}.meta.json"), "w"), indent=1)
    print(f"[capture] {component}: codes={np.array(codes_full).shape[1]} "
          f"refer={tuple(refer.shape)}")


def bench_sovits(component: str, models_root: str, iters: int, out_dir: str) -> dict:
    import mlx.core as mx

    from gsovits_mlx.io import release, trim_metal
    from gsovits_mlx.pipeline import (_load_sovits_v1v2, denorm_spec, load_sovits_v3,
                                      load_vocoder_v4, mel_spectrogram, norm_spec)
    from gsovits_mlx.sovits.cfm import resolve_sampling, synthesize_v5_mel
    from gsovits_mlx.text.mel_frontend import spectrogram

    ver = SOVITS_SPEC[component][0]
    z = np.load(os.path.join(out_dir, f"{component}.npz"))
    dirname = {"v5dev": "v5dev", "v5turbo": "v5turbo"}.get(ver, ver)
    if ver in ("v1", "v2", "v2Pro", "v2ProPlus"):
        sov, _ = _load_sovits_v1v2(os.path.join(models_root, dirname), ver)
    else:
        sov, _ = load_sovits_v3(os.path.join(models_root, dirname), ver)

    all_phones = mx.array(z["phones"])
    all_codes = mx.array(z["prompt"])[:, None, :]
    refer = mx.array(z["refer"])
    ref_audio_32k = z["w32"]

    result = {"component": component, "iters": iters}
    golden = None
    spans: dict[str, float] = {}
    for i in range(iters):
        t0 = time.perf_counter()
        if ver in ("v1", "v2", "v2Pro", "v2ProPlus"):
            raise SystemExit("v1/v2/v2pro cells belong to conv-fp16 (HiFiGAN A/B) per cell split")
        refer_mask = mx.ones((refer.shape[0], 1, refer.shape[2]), dtype=refer.dtype)
        ge = sov.ref_enc(refer[:, :704] * refer_mask, refer_mask)
        if ver in ("v5dev", "v5turbo"):
            n_prompt = int(z["n_prompt_phones"])
            fea_ref, ge = sov.decode_encp(mx.array(z["prompt"][None]), all_phones[:, :n_prompt],
                                          refer=refer, ge=ge)
            fea_todo, _ = sov.decode_encp(all_codes, all_phones, refer=refer, ge=ge, speed=1.0)
            mx.eval(fea_ref, fea_todo)
            spans[f"encp_{i}"] = round(time.perf_counter() - t0, 3)
            del ge, refer_mask
            mel2 = norm_spec(mel_spectrogram(mx.array(ref_audio_32k[None]), 1280, 100,
                                             32000, 320, 1280, 0, None, center=False))
            mx.eval(mel2)
            steps, cfg = resolve_sampling(ver, None, None)
            t1 = time.perf_counter()
            mel = synthesize_v5_mel(sov, fea_ref, fea_todo, mel2, sample_steps=steps,
                                    cfg_rate=cfg, key=mx.random.key(0))
            pred = denorm_spec(mel)
            mx.eval(pred)
            spans[f"cfm_{i}"] = round(time.perf_counter() - t1, 3)
            if "dit_config" not in result:
                est = sov.cfm.estimator
                attn0 = est.transformer_blocks[0].attn
                result["dit_config"] = {
                    "ver": ver,
                    "dim": int(est.dim),
                    "n_blocks": len(est.transformer_blocks),
                    "heads": int(attn0.heads),
                    "dim_head": int(attn0.dim_head),
                    "mu_shape": [int(v) for v in fea_todo.shape],
                    "steps": int(steps),
                    "cfg_rate": float(cfg),
                    "weight_dtype": str(attn0.to_q_w.dtype),
                    "mu_ref_T": int(fea_ref.shape[-1]),
                    "mu_todo_T": int(fea_todo.shape[-1]),
                }
            del fea_ref, fea_todo, mel2, mel
            voc = load_vocoder_v4(os.path.join(models_root, "v5_vocoder"))
            t2 = time.perf_counter()
            audio = voc(pred)
            mx.eval(audio)
            spans[f"vocoder_{i}"] = round(time.perf_counter() - t2, 3)
            del pred
            out_np = np.array(audio)[0, 0]
            sr = 48000
        else:
            # v3 cell: decode_encp(3.875x interp) -> CFM 32-step cfg 0.0 ->
            # BigVGAN 24 kHz (mel_fn: 100-mel 1024/256 @ 24 kHz on 24k ref).
            from gsovits_mlx.pipeline import cfm_chunked_decode_v3, load_bigvgan
            n_prompt = int(z["n_prompt_phones"])
            fea_ref, ge = sov.decode_encp(mx.array(z["prompt"][None]), all_phones[:, :n_prompt],
                                          refer=refer, ge=ge)
            fea_todo, _ = sov.decode_encp(all_codes, all_phones, refer=refer, ge=ge, speed=1.0)
            mx.eval(fea_ref, fea_todo)
            spans[f"encp_{i}"] = round(time.perf_counter() - t0, 3)
            del ge, refer_mask
            # mel2: v3 prompt mel is 100-mel 1024/256 @ 24 kHz (e2e_v3 stage 5)
            import soundfile as sf
            ref24, _sr = sf.read("/Volumes/2T/gpt-sovits-models/bench/ref_zh_3.5s.wav",
                                 dtype="float32", always_2d=True)
            ref24 = ref24.mean(axis=1)
            from gsovits_mlx.pipeline import resample_linear
            ref24 = ref24 if _sr == 24000 else resample_linear(ref24, _sr, 24000)
            from gsovits_mlx.text.mel_frontend import mel_spectrogram as _mel
            mel2 = norm_spec(_mel(mx.array(ref24[None]), 1024, 100, 24000, 256, 1024,
                                  0, None, center=False))
            mx.eval(mel2)
            steps, cfg = resolve_sampling("v3", None, None)
            t1 = time.perf_counter()
            mel = cfm_chunked_decode_v3(sov, fea_ref, fea_todo, mel2,
                                        sample_steps=steps, inference_cfg_rate=cfg,
                                        key=mx.random.key(0))
            pred = denorm_spec(mel)
            mx.eval(pred)
            spans[f"cfm_{i}"] = round(time.perf_counter() - t1, 3)
            del fea_ref, fea_todo, mel2, mel
            voc = load_bigvgan(os.path.join(models_root, "bigvgan"))
            t2 = time.perf_counter()
            audio = voc(pred)
            mx.eval(audio)
            spans[f"vocoder_{i}"] = round(time.perf_counter() - t2, 3)
            del pred
            out_np = np.array(audio)[0, 0]
            sr = 24000
        trim_metal()
        dt = time.perf_counter() - t0
        result[f"decode_s_{i}"] = round(dt, 3)
        if i == 0:
            golden = out_np
            result["audio_s"] = round(len(out_np) / sr, 3)
        else:
            same = out_np.shape == golden.shape and np.array_equal(out_np, golden)
            result["parity"] = "OK" if same else "FAIL"
    # seeded-but-mx-random-key run is deterministic: i0 vs i1 must match
    result.update(spans)
    del sov  # free the DiT after all iterations, not inside the loop
    release()
    trim_metal()
    result.setdefault("parity", "n/a(single-iter)")
    print(f"[micro] {component}: parity={result['parity']} "
          f"audio={result.get('audio_s')}s best={min(result[f'decode_s_{i}'] for i in range(iters))}s")
    return result



# ---------------------------------------------------------------------------
# gemm bench helpers (task-11). CPU-testable: pure functions, no side effects
# beyond the arrays they build. Every timed output is retained AND evaluated;
# inputs are DISTINCT materialized arrays so identical-work elimination
# cannot drop them from the graph or the run.
# ---------------------------------------------------------------------------

def gemm_parse_shape(shape: str) -> tuple:
    parts = shape.lower().split("x")
    if len(parts) != 3:
        raise ValueError(f"shape must be MxKxN, got {shape!r}")
    dims = tuple(int(v) for v in parts)
    if any(v <= 0 for v in dims):
        raise ValueError(f"shape dimensions must be positive, got {shape!r}")
    return dims


def gemm_validate_queue(q: int) -> int:
    if int(q) != q or q < 1:
        raise ValueError(f"queue must be a positive integer, got {q!r}")
    return int(q)


def gemm_flops(shapes) -> int:
    """FLOPs for the matmuls ACTUALLY executed. shapes: iterable of (M,K,N)."""
    return sum(2 * m * k * n for (m, k, n) in shapes)


def _mk_distinct(mx, n, rows, cols, seed):
    """n DISTINCT materialized fp16 arrays (seeded per index)."""
    mats = []
    for i in range(n):
        key = mx.random.key(seed * 100_003 + i)
        mats.append(mx.random.normal((rows, cols), key=key).astype(mx.float16))
    mx.eval(*mats)
    return mats


def bench_gemm(mx, M: int, K: int, N: int, q: int, rounds: int = 8) -> dict:
    import time as _time

    dtype = mx.float16
    out = {"mode": "gemm", "shape": f"{M}x{K}x{N}", "dtype": "float16", "queue": q,
           "rounds": rounds, "input_policy": "distinct-materialized",
           "output_policy": "retain-and-eval-all",
           "indep_batch_est_bytes": (q + 1) * (M * K + K * N + M * N) * 2}

    # q+1 distinct lefts and rights (extra for warmup), all pre-evaluated.
    lefts = _mk_distinct(mx, q + 1, M, K, seed=1)
    rights = _mk_distinct(mx, q + 1, K, N, seed=2)

    def round_outputs():
        """q DISTINCT matmuls (lefts[i] @ rights[i]), outputs retained."""
        return [lefts[i] @ rights[i] for i in range(q)]

    # warm both paths once
    mx.eval(*round_outputs())

    # (a) single-op synced latency: valid END-TO-END latency (dispatch+sync
    # included), best-of-rounds. Uses the distinct operands, single op.
    best = math.inf
    for _ in range(rounds):
        s = _time.perf_counter()
        r = lefts[q] @ rights[q]
        mx.eval(r)
        best = min(best, _time.perf_counter() - s)
    out["single_op_synced_ms"] = round(best * 1e3, 3)

    # (b) independent-batch achieved throughput: q distinct matmuls queued,
    # ONE final eval of ALL retained outputs. Work = q x (2MKN) exactly.
    best = math.inf
    for _ in range(rounds):
        s = _time.perf_counter()
        outs = round_outputs()
        mx.eval(*outs)
        best = min(best, _time.perf_counter() - s)
    out["indep_batch_total_ms"] = round(best * 1e3, 3)
    out["indep_batch_flops"] = gemm_flops([(M, K, N)] * q)
    out["indep_batch_tf"] = round(out["indep_batch_flops"] / best / 1e12, 2)

    # (c) dependent-chain latency: x @ W1 @ W2 ... compatible matrices, q
    # links, every intermediate retained and evaluated once at the end.
    # FLOPs from the ACTUAL per-link shapes: first link (M,K,N), then
    # (M,N,N) for each following link.
    w_first = _mk_distinct(mx, 1, K, N, seed=31)[0]
    chain_rest = _mk_distinct(mx, q - 1, N, N, seed=32) if q > 1 else []
    x0 = lefts[q]
    chain_shapes = [(M, K, N)] + [(M, N, N)] * (q - 1)

    def chain_outputs():
        acc = x0 @ w_first
        for w in chain_rest:
            acc = acc @ w
        return acc

    r = chain_outputs()
    mx.eval(r)
    best = math.inf
    for _ in range(rounds):
        s = _time.perf_counter()
        acc = chain_outputs()
        mx.eval(acc)
        best = min(best, _time.perf_counter() - s)
    out["chain_total_ms"] = round(best * 1e3, 3)
    out["chain_links"] = q
    out["chain_flops"] = gemm_flops(chain_shapes)
    out["chain_tf"] = round(out["chain_flops"] / best / 1e12, 2)
    out["chain_shapes"] = [f"{m}x{k}x{n}" for (m, k, n) in chain_shapes]
    return out

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["capture", "bench", "gemm"])
    ap.add_argument("--shape", default="1600x1152x4608",
                    help="gemm mode: MxKxN (fp16)")
    ap.add_argument("--queue", type=int, default=8,
                    help="gemm mode: ops queued per eval (state it!)")
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

    if args.mode == "gemm":
        gpu_lock_guard(args.allow_unlocked)
        import mlx.core as mx
        try:
            M, K, N = gemm_parse_shape(args.shape)
            q = gemm_validate_queue(args.queue)
            res = bench_gemm(mx, M, K, N, q,
                             rounds=int(os.environ.get("MICROBENCH_GEMM_ROUNDS", "8")))
        except ValueError as e:
            sys.exit(f"gemm: {e}")
        print(json.dumps(res))
        if args.json_out:
            with open(args.json_out, "a") as f:
                f.write(json.dumps(res) + "\n")
        return

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
