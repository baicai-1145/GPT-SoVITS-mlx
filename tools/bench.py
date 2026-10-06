"""Benchmark harness for gsovits_mlx pipelines (skeleton).

Runs one synthesis per invocation, timing each pipeline stage and reporting
process peak RSS, as one JSON line on stdout. Stage names are stable so the
output can be concatenated across runs and versions:

    {"version": "v2", "stages": {"frontend_s": .., "prompt_codes_s": ..,
     "refer_spec_s": .., "ar_s": .., "sovits_s": .., "ar_tokens": ..,
     "audio_s": .., "peak_rss_kb": ..}, "total_s": ..}

Stage semantics:
    frontend      text -> phones + BERT features (official cleaner + MLX BERT)
    prompt_codes  ref audio -> HuBERT -> SoVITS quantizer (semantic prompt)
    refer_spec    ref audio -> 32 kHz linear spectrogram crop (v2: 704 bins)
    ar            GPT AR sampling of target semantic tokens
    sovits        SoVITS decode (prompt + gen codes) -> 32 kHz audio
    vocoder       reserved: only non-zero for v3/v4/v5 builds where the
                  HiFiGAN/NiFiGAN vocoder runs as a separate stage (the v1/v2
                  decoder includes the vocoder in `sovits`).

Peak RSS is the child process's high-water mark
(`resource.getrusage(RUSAGE_CHILDREN).ru_maxrss`, normalized to KB — macOS
reports bytes), so it covers MLX Metal buffer cache growth; compare across
identical stage sequences, not across versions.

Examples:
    python3 tools/bench.py --version v2 \
        --text "你好，欢迎来到各自的旅程。今天我们聊聊机器学习。" \
        --ref-audio ref.wav --ref-text "希望你以后能够做得比我还好哟。"
    python3 tools/bench.py --version v2 --models-root /path/to/mlx --repeat 3
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time

SUPPORTED_VERSIONS = ["v1", "v2", "v2Pro", "v2ProPlus", "v3", "v4", "v5dev", "v5turbo"]

DEFAULT_MODELS_ROOT = "/Volumes/2T/gpt-sovits-models/mlx"
DEFAULT_CPUFAST_REPO = "/Users/baicai1145/repos/gpt-sovits/GPT-SoVITS-CPUFast"
DEFAULT_REF_AUDIO = "/Volumes/2T/gpt-sovits-models/bench/ref_zh_3.5s.wav"
DEFAULT_REF_TEXT = "希望你以后能够做得比我还好哟。"
DEFAULT_TEXT = "你好，欢迎来到各自的旅程。今天我们聊聊机器学习。"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", default="v2", choices=SUPPORTED_VERSIONS,
                   help="Model version to benchmark (default: v2).")
    p.add_argument("--text", default=DEFAULT_TEXT)
    p.add_argument("--ref-audio", default=DEFAULT_REF_AUDIO)
    p.add_argument("--ref-text", default=DEFAULT_REF_TEXT)
    p.add_argument("--models-root", default=DEFAULT_MODELS_ROOT,
                   help="Root of MLX model exports; per-version weights live at "
                        f"<root>/<version>/ and GPT at <root>/s1(2)/ (default: {DEFAULT_MODELS_ROOT})")
    p.add_argument("--cpufast-repo",
                   default=os.environ.get("GPT_SOVITS_CPUFAST", DEFAULT_CPUFAST_REPO),
                   help="GPT-SoVITS-CPUFast checkout for the text front-end.")
    p.add_argument("--lang", default="zh")
    p.add_argument("--top-k", type=int, default=15)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--repetition-penalty", type=float, default=1.35)
    p.add_argument("--early-stop-num", type=int, default=None,
                   help="AR cap (default: 2850 for v2-family, 1500 for v1).")
    p.add_argument("--noise-scale", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--repeat", type=int, default=1,
                   help="Run the pipeline this many times; emit one JSON line per run.")
    p.add_argument("--e2e-script", default=None,
                   help="Path to the e2e script to drive (default: tools/e2e_v2.py "
                        "alongside this file). Only v2 is wired today; other "
                        "versions fail fast until their e2e scripts land.")
    p.add_argument("--gpu", action="store_true",
                   help="Opt the e2e children into Metal: sets GSOVITS_GPU_LOCK_OK=1;\n                        children still require a fresh gpu.lock (default-deny).")
    p.add_argument("--keep-audio", default=None,
                   help="Optional path for the rendered wav of the last repeat.")
    return p.parse_args()


# version -> (e2e script, extra CLI args, supports --noise-scale/--early-stop-num)
E2E_SCRIPTS = {
    "v1": ("e2e_v1.py", [], True),
    "v2": ("e2e_v2.py", [], True),
    "v2Pro": ("e2e_v2pro.py", ["--model", "pro"], True),
    "v2ProPlus": ("e2e_v2pro.py", ["--model", "proplus"], True),
    "v3": ("e2e_v3.py", [], False),
    "v4": ("e2e_v4.py", [], False),
    "v5dev": ("e2e_v5.py", ["--model", "dev"], False),
    "v5turbo": ("e2e_v5.py", ["--model", "turbo"], False),
}
# stage names emitted per family ("sovits+vocoder" splits to sovits_vocoder;
# v1/v2 decode includes the vocoder in `sovits`)
_STAGE_NAMES = ("frontend", "prompt_codes", "refer_spec", "ar", "sovits", "sovits_vocoder",
                "cfm", "vocoder", "sv_emb")


def run_one(args: argparse.Namespace, out_path: str | None) -> dict:
    """Drive one synthesis and collect stage timings as a dict."""
    script_name, extra, has_v2_args = E2E_SCRIPTS[args.version]
    script = args.e2e_script or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             script_name)
    if not os.path.isfile(script):
        raise SystemExit(f"e2e script not found: {script}")

    if out_path is None:
        # throwaway render; still fully exercises the pipeline
        out_path = os.path.join(os.path.dirname(script), "..", ".tmp",
                                f"bench_{args.version}_{os.getpid()}.wav")
        out_path = os.path.abspath(out_path)
        unlink_after = True
    else:
        unlink_after = False

    cmd = [sys.executable, script,
           "--text", args.text, "--ref-audio", args.ref_audio,
           "--ref-text", args.ref_text, "--models-root", args.models_root,
           "--cpufast-repo", args.cpufast_repo, "--lang", args.lang,
           "--top-k", str(args.top_k), "--top-p", str(args.top_p),
           "--temperature", str(args.temperature),
           "--repetition-penalty", str(args.repetition_penalty),
           "--seed", str(args.seed), "--bench", "--out", out_path] + extra
    if has_v2_args:
        cmd += ["--noise-scale", str(args.noise_scale)]
        if args.early_stop_num is not None:
            cmd += ["--early-stop-num", str(args.early_stop_num)]

    t0 = time.perf_counter()
    # RUSAGE_CHILDREN.ru_maxrss = max peak RSS across reaped children;
    # macOS reports bytes, Linux kilobytes. Normalize to KB.
    proc = subprocess.run(cmd, capture_output=True, text=True)
    total = time.perf_counter() - t0
    peak_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    if sys.platform == "darwin":
        peak_rss //= 1024
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"e2e run failed with code {proc.returncode}")

    stages: dict[str, float] = {}
    for line in proc.stderr.splitlines():
        # e2e_v2.py --bench emits: "[bench] frontend: ... 0.42s" etc.
        if not line.startswith("[bench] "):
            continue
        body = line[len("[bench] "):]
        if "|" in body:  # trailing summary: "sovits: ... | total X.XXs peak_rss=N KB"
            body, tail = body.split("|", 1)
            if "peak_rss" in tail:
                stages["peak_rss_kb"] = int(tail.split("peak_rss=")[1].split()[0])
            stages["total_e2e_s"] = _last_seconds(tail)
        name, _, rest = body.partition(": ")
        name = name.replace("sovits+vocoder", "sovits_vocoder")
        if name in _STAGE_NAMES:
            stages[name + "_s"] = _last_seconds(rest)
        if "tokens" in rest:
            stages["ar_tokens"] = int(rest.split("tokens")[0].split()[-1])
    stages.setdefault("vocoder_s", 0.0)  # v1/v2 decode includes the vocoder
    stages["audio_s"] = _wav_seconds(out_path) if os.path.isfile(out_path) else 0.0
    stages["peak_rss_kb"] = peak_rss
    if unlink_after and os.path.isfile(out_path):
        os.unlink(out_path)
    return {"version": args.version, "stages": stages, "total_s": round(total, 3)}


def _last_seconds(text: str) -> float:
    """Extract the trailing '<x.xx>s' number from a bench log fragment
    ("frontend: phones=... 0.42s" -> 0.42; tolerates "... tok/s)")."""
    token = text.rstrip().rstrip("s").split()[-1]
    try:
        return float(token)
    except ValueError:
        # e.g. "106.4 tok/s)" -- walk back to the pure number before it
        parts = text.rstrip().split()
        for candidate in reversed(parts):
            c = candidate.rstrip("s")
            try:
                return float(c)
            except ValueError:
                continue
        raise


def _wav_seconds(path: str) -> float:
    import wave
    with wave.open(path, "rb") as w:
        return round(w.getnframes() / w.getframerate(), 3)


def main() -> None:
    args = parse_args()
    # bench spawns the e2e scripts as children; they self-gate via
    # GSOVITS_GPU_LOCK_OK/--gpu. bench --gpu sets the opt-in env for the
    # children (the lock check happens in each child).
    if args.gpu:
        os.environ["GSOVITS_GPU_LOCK_OK"] = "1"
    from gsovits_mlx.gpu_lock import lock_status

    held, owner = lock_status()
    device = "gpu" if (args.gpu and held) else "cpu"
    print(f"[bench] device={device} "
          f"(lock={'held: ' + owner if held else 'not held'})", flush=True)
    for i in range(args.repeat):
        row = run_one(args, args.keep_audio if i == args.repeat - 1 else None)
        row["device"] = device
        print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
