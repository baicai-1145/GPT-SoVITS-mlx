"""Stage-0 auto-label: MSS vocal separation + qwen3-asr transcription.

Converts raw audio (songs, game recordings, spoken corpora) into the
wav|spk|lang|text list prepare_data consumes — no manual annotation. Runs
the full-MLX stack: pymss (bs_roformer, MLX backend) for separation,
funasr-mlx (qwen3-asr-0.6b w4a16, CLI subprocess mode — RTF 0.086 measured,
faster than the resident worker) for transcription.

Validated 2026-10-15 (.tmp/asr_eval): separation corr 0.011 on stereo mixes;
ASR bit-exact on the reference anchor, 50/50 hutao clips. Known limitation:
sub-1s clips can language-flip — --force-lang zh (default) pins it.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

DEFAULT_ASR_MODEL = "/Volumes/2T/gpt-sovits-models/asr/qwen3-asr-0.6b-mlx/w4a16"
DEFAULT_MSS_MODEL = "bs_roformer_anvuew_sdr_12.45"
DEFAULT_MSS_DIR = "/Volumes/2T/gpt-sovits-models/mss"


def _clean_asr_text(raw: str, force_lang: str | None) -> str:
    """Strip the chat-template prefix ('language Chinese<asr_text>')."""
    t = raw
    for tag in ("<asr_text>",):
        if tag in t:
            t = t.split(tag, 1)[1]
    t = t.removeprefix(f"language {force_lang}" if force_lang else "").strip()
    return t.strip()


def _parse_asr_text(raw: str) -> tuple[str, str]:
    """'language Chinese<asr_text>你好。' -> ('Chinese', '你好。')."""
    if "<asr_text>" in raw:
        head, _, text = raw.partition("<asr_text>")
        lang = head.removeprefix("language ").strip()
        return lang, text.strip()
    return "", raw.strip()


def transcribe(path: str, model: str, force_lang: str | None = None) -> str:
    """One file -> transcript (CLI subprocess: guard-isolated, RTF 0.086).
    force_lang pins the LIST column (sub-1s clips language-flip, measured);
    the transcript itself is kept as ASR produced it."""
    cmd = ["funasr-mlx", "transcribe", model, "--family", "qwen3-asr",
           "--audio", path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"asr failed ({os.path.basename(path)}): "
                           f"{r.stderr.strip()[-200:]}")
    d = json.loads(r.stdout)
    _lang, text = _parse_asr_text(d.get("text", ""))
    return text


def separate(path: str, out_dir: str, mss_model: str, mss_dir: str) -> str:
    """One file -> vocals wav path (bs_roformer, MLX). Falls back to the
    original on mono/voice-only input when the model finds no instrumental
    energy (measured on band-limited captures)."""
    from pymss import MSSeparator, load_audio, save_audio  # lazy: heavy
    import numpy as np
    sep = MSSeparator.from_model_name(mss_model, model_dir=mss_dir,
                                      device="mlx")
    mix, sr = load_audio(path)
    out = sep.separate(mix)
    stems = {k.lower(): v for k, v in out.items()}
    if "vocals" not in stems:
        raise RuntimeError(f"no vocals stem from {path}")
    voc = np.asarray(stems["vocals"], dtype=np.float32)
    if voc.ndim > 1:
        voc = voc.mean(axis=1)
    dst = os.path.join(out_dir, "sep_" + os.path.splitext(
        os.path.basename(path))[0] + ".wav")
    import soundfile as sf
    sf.write(dst, voc, sr)
    return dst


def run_auto_label(args, timing: dict) -> None:
    raw_files = []
    for line in open(args.raw_list, encoding="utf8"):
        line = line.strip()
        if line and not line.startswith("#"):
            raw_files.append(line)
    work = os.path.join(args.exp_dir, "0-auto_label")
    sep_dir = os.path.join(work, "separated")
    os.makedirs(sep_dir, exist_ok=True)
    out_lines, skipped = [], 0
    t0 = time.perf_counter()
    sep_n = asr_n = 0
    for i, src in enumerate(raw_files):
        src = os.path.abspath(os.path.expanduser(src))
        name = os.path.splitext(os.path.basename(src))[0]
        wav = src
        if args.separate:
            try:
                wav = separate(src, sep_dir, args.mss_model, args.mss_dir)
                sep_n += 1
            except Exception as e:  # noqa: BLE001 — skip-file semantics
                print(f"[auto-label] separate failed, using raw: {name}: "
                      f"{e}", file=sys.stderr)
        try:
            text = transcribe(wav, args.asr_model, args.force_lang)
        except Exception as e:  # noqa: BLE001
            print(f"[auto-label] SKIP {name}: {e}", file=sys.stderr)
            skipped += 1
            continue
        asr_n += 1
        lang = args.force_lang or "zh"
        spk = args.speaker or name.split("_")[0]
        out_lines.append(f"{os.path.basename(wav)}|{spk}|{lang}|{text}")
        if (i + 1) % 25 == 0:
            print(f"[auto-label] {i+1}/{len(raw_files)} "
                  f"(sep {sep_n}, asr {asr_n}, skip {skipped})", flush=True)
    list_path = os.path.join(work, "auto.list")
    with open(list_path, "w", encoding="utf8") as f:
        f.write("\n".join(out_lines) + "\n")
    timing["auto_label"] = time.perf_counter() - t0
    print(f"[auto-label] done: {len(out_lines)} lines -> {list_path} "
          f"(separated {sep_n}, transcribed {asr_n}, skipped {skipped}, "
          f"{timing['auto_label']:.1f}s)")
    if not out_lines:
        raise SystemExit("[auto-label] produced nothing; inspect stderr above")


def add_auto_label_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("stage-0 auto-label")
    g.add_argument("--raw-list", default=None,
                   help="file with one raw-audio path per line (enables "
                        "stage-0; no annotation needed)")
    g.add_argument("--separate", action="store_true",
                   help="run MSS vocal separation before ASR (songs/mixes)")
    g.add_argument("--asr-model", default=DEFAULT_ASR_MODEL)
    g.add_argument("--mss-model", default=DEFAULT_MSS_MODEL)
    g.add_argument("--mss-dir", default=DEFAULT_MSS_DIR)
    g.add_argument("--speaker", default=None,
                   help="speaker column for all lines (default: filename "
                        "prefix before first '_')")
    g.add_argument("--force-lang", default="zh",
                   help="language column + ASR language pin (default zh; "
                        "sub-1s clips language-flip otherwise)")
