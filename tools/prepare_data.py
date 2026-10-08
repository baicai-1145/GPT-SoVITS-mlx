#!/usr/bin/env python
"""MLX preprocessing pipeline: GPT-SoVITS prepare_datasets equivalent.

Reproduces the official dataset preparation (cuda_graph repo
GPT_SoVITS/prepare_datasets/{1-get-text, 2-get-hubert-wav32k, 2-get-sv,
3-get-semantic}.py) producing the EXACT same exp_dir layout downstream
training consumes:

    <exp_dir>/2-name2text.txt          name\tphones\tword2ph\tnorm_text
    <exp_dir>/3-bert/<name>.npy        (1024, T) fp32 zh bert feature
                                       (official: torch .pt fp16; npy keeps
                                       this repo torch-free — loader in the
                                       s1 dataloader handles it)
    <exp_dir>/4-cnhubert/<name>.npy    (1, 768, T) fp16 ssl feature
    <exp_dir>/5-wav32k/<name>.wav      32 kHz int16 (bit-equal to official)
    <exp_dir>/7-sv_cn/<name>.npy       (1, 20480) fp32 sv embedding (Pro only)
    <exp_dir>/6-name2semantic.tsv      header + name\tcodes lines

Semantic notes (parity documented in docs/PARITY_NOTES.md and
tests/test_prepare_data.py):
* 2-name2text word2ph: official writes the python list repr ("[2, 2, 1]");
  non-zh lines have the literal "None". We write the same repr.
* stage-1 bert gate is zh-only. Run with GSOVITS_FRONTEND_F32=1 and
  GSOVITS_G2PW_FP16=0 for TRAINING data (fp32 BERT max-abs 1.1e-4 vs
  transformers fp32 CPU; fp16 G2PW flips polyphones on ~8% of corpus
  lines — measured 2/24 on the gate set; fp16 BERT adds ~5e-2).
  Known front-end gap: our vendored CPUFast chinese2 (torch g2pw.pth)
  vs the cuda_graph training repo's ONNX g2pW disagree on ~1/24 corpus
  lines (e.g. 不得了 le5/liao3, 什么 sh en2/ir2) — an ONNX-vs-torch
  model difference inside the official repos themselves, NOT an MLX
  bug; our fp32 output is byte-identical to CPUFast torch clean_text.
* stage-2 replicates the official load_audio ffmpeg CLI decode (f32le),
  the 0.95/0.5 amplitude quirks, librosa soxr_hq 32k->16k resample
  (librosa 1.0.0 + soxr; bit-equal to the base venv), HuBERT fp16-saved
  ssl (fp32 compute, max-abs 1.4e-4 vs torch fp32), scipy-wavfile-equal
  int16 wav32k (md5-verified 24/24), NaN-check + skip like official.
* stage-2b sv reads 5-wav32k/<name>.wav, torchaudio sinc_interp_hann
  Resample (numpy port gsovits_mlx/audio_resample.py, 1.8e-7 vs
  torchaudio — NOT soxr, which differs at 5e-2), kaldi fbank +
  ERes2NetV2 forward3 fp32 (embedding max 2.2e-4, fbank-dominated:
  fbank itself is the documented 4.1e-4 numpy-vs-torchaudio parity;
  network alone 5.7e-6).
* stage-3 ssl_proj + quantizer argmin computed in fp32 on the fp16 ssl
  (upcast weights). Reason: the expand-form L2 (||x||²+||e||²-2xe) in
  fp16 suffers catastrophic cancellation (~10 distance error at |d|~700)
  flipping argmin on sub-0.5 margins — 24/2915 frames measured vs torch
  fp32. fp32 diff-form matches torch fp32 codes EXACTLY (24/24 files,
  2915/2915 frames, both v1 and v2 weights). Official CUDA is_half runs
  carry the same fp16 cancellation with a different rounding pattern
  (not bit-replicable on MLX); cross-input end-to-end (each pipeline's
  own HuBERT output): 23/24 files exact, 1 frame in 2915.

Usage:
    python tools/prepare_data.py --list file --wav-dir dir --exp-dir out \
        --version {v1,v2,v2Pro,v2ProPlus} --models-root models_local \
        [--parts N --part i] [--stages all|text,hubert,sv,semantic,merge] \
        [--timing-json out.json]

Recommended training run (parity settings):
    GSOVITS_G2PW_FP16=0 GSOVITS_FRONTEND_F32=1 \
    python tools/prepare_data.py --list wavtest.list --wav-dir wavtest \
        --exp-dir .tmp/train_data/exp --version v2 --stages all

GPU: stages text/hubert/sv/semantic use MLX (Metal) under the shared
.tmp/gpu.lock.d discipline (AGENTS.md); the tool resolves the device once
and refuses GPU-less audio/semantic stages (front-end-only CPU smoke is
legal via GSOVITS_FRONTEND_DEVICE=cpu GSOVITS_PREP_CPU=1).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

LANGUAGE_V1_TO_V2 = {
    "ZH": "zh", "zh": "zh",
    "JP": "ja", "jp": "ja", "JA": "ja", "ja": "ja",
    "EN": "en", "en": "en", "En": "en",
    "KO": "ko", "Ko": "ko", "ko": "ko",
    "yue": "yue", "YUE": "yue", "Yue": "yue",
}

MAXX = 0.95
ALPHA = 0.5


# ---------------------------------------------------------------------------
# helpers shared with official scripts
# ---------------------------------------------------------------------------

def clean_path(path_str: str) -> str:
    """tools/my_utils.clean_path (verbatim)."""
    if path_str.endswith(("\\", "/")):
        return clean_path(path_str[0:-1])
    path_str = path_str.replace("/", os.sep).replace("\\", os.sep)
    return path_str.strip(" '\n\"\u202a")


def load_audio_ffmpeg(file: str, sr: int):
    """tools/my_utils.load_audio: ffmpeg CLI subprocess, f32le pcm capture.

    Bit-equal to the official call chain (ffmpeg-python builds the same
    argv; we run the CLI directly to keep torch/ffmpeg-python out of the
    runtime). Monaural downmix + resample handled by ffmpeg itself.
    """
    file = clean_path(file)
    if not os.path.exists(file):
        raise RuntimeError(f"audio path does not exist: {file}")
    cmd = ["ffmpeg", "-nostdin", "-threads", "0", "-i", file,
           "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(sr), "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {file}: {proc.stderr.decode(errors='replace')[-400:]}")
    import numpy as np
    return np.frombuffer(proc.stdout, np.float32).flatten()


def read_list_lines(list_path: str, i_part: int = 0, all_parts: int = 1) -> list[str]:
    with open(list_path, "r", encoding="utf8") as f:
        lines = f.read().strip("\n").split("\n")
    return lines[int(i_part)::int(all_parts)]


def parse_list_line(line: str):
    """wav_name|spk_name|language|text -> (basename, language, text) with the
    official language mapping; returns None for unsupported languages."""
    wav_name, _spk, language, text = line.split("|")
    if language not in LANGUAGE_V1_TO_V2:
        return None
    return os.path.basename(clean_path(wav_name)), LANGUAGE_V1_TO_V2[language], text


# ---------------------------------------------------------------------------
# stage 1: text
# ---------------------------------------------------------------------------

def run_text(args, timing: dict) -> None:
    from gsovits_mlx.text.preproc import TextFrontend, bootstrap

    t0 = time.perf_counter()
    # models_root is absolute (main() normalizes path args) — set bert_path
    # FIRST so the vendored chinese2 import resolves the g2pw tokenizer even
    # if a stale/relative bert_path leaked into the environment.
    os.environ["bert_path"] = os.path.join(args.models_root, "bert")
    bootstrap(args.cpufast_repo, models_root=args.models_root)
    fe = TextFrontend(models_root=args.models_root, lazy=True,
                      device=os.environ.get("GSOVITS_FRONTEND_DEVICE", "gpu"))
    timing["stage1_model_load"] = time.perf_counter() - t0

    bert_dir = os.path.join(args.exp_dir, "3-bert")
    os.makedirs(bert_dir, exist_ok=True)
    txt_path = os.path.join(args.exp_dir, f"2-name2text-{args.part}.txt")
    if os.path.exists(txt_path):
        print(f"[stage1] {txt_path} exists, skipping (official behavior)")
        return

    # Vendored front-end: official clean_text, executed in place.
    from gsovits_mlx.text.vendored_cpufront import load_cpufront
    import sys as _sys

    load_cpufront(args.cpufast_repo)
    cleaner = _sys.modules["text.cleaner"]
    cleaner_version = "v1" if args.version == "v1" else "v2"

    rows = []
    t0 = time.perf_counter()
    n_zh = 0
    for line in read_list_lines(args.list, args.part, args.parts):
        try:
            parsed = parse_list_line(line)
            if parsed is None:
                print(f"[Warning] The language of {line.split('|')[0]} is not supported for training.")
                continue
            name, lan, text = parsed
            phones, word2ph, norm_text = cleaner.clean_text(
                text.replace("%", "-").replace("￥", ","), lan, cleaner_version)
            if lan == "zh":
                bert = fe._bert_feature_zh(norm_text, word2ph)
                assert bert.shape[1] == len(phones), (bert.shape, len(phones))
                import numpy as np
                np.save(os.path.join(bert_dir, name + ".npy"), bert.astype(np.float32))
                n_zh += 1
            rows.append("%s\t%s\t%s\t%s" % (
                name, " ".join(phones), repr(word2ph), norm_text))
        except Exception:
            import traceback
            print(line[:80], traceback.format_exc(), file=sys.stderr)
    timing["stage1_process"] = time.perf_counter() - t0
    with open(txt_path, "w", encoding="utf8") as f:
        f.write("\n".join(rows) + "\n")
    print(f"[stage1] {len(rows)} lines ({n_zh} zh bert) -> {txt_path}")


# ---------------------------------------------------------------------------
# stage 2: hubert + wav32k
# ---------------------------------------------------------------------------

def run_hubert(args, timing: dict) -> None:
    import numpy as np
    import librosa
    import soundfile as sf
    import mlx.core as mx

    from gsovits_mlx.io import load_mlx_safetensors
    from gsovits_mlx.text.hubert import HubertModel

    hubert_dir = os.path.join(args.exp_dir, "4-cnhubert")
    wav32dir = os.path.join(args.exp_dir, "5-wav32k")
    os.makedirs(hubert_dir, exist_ok=True)
    os.makedirs(wav32dir, exist_ok=True)

    t0 = time.perf_counter()
    hubert_root = os.path.join(args.models_root, "hubert")
    model = HubertModel(
        load_mlx_safetensors(os.path.join(hubert_root, "hubert.safetensors")),
        json.load(open(os.path.join(hubert_root, "config.json"))))
    timing["stage2_model_load"] = time.perf_counter() - t0

    nan_fails = []
    t_io = t_gpu = 0.0
    n_ok = 0
    for line in read_list_lines(args.list, args.part, args.parts):
        try:
            parsed = parse_list_line(line)
            if parsed is None:
                continue
            wav_name, _lan, _text = parsed
            hubert_path = os.path.join(hubert_dir, wav_name + ".npy")
            if os.path.exists(hubert_path):
                continue
            wav_path = os.path.join(args.wav_dir, wav_name)
            t1 = time.perf_counter()
            tmp_audio = load_audio_ffmpeg(wav_path, 32000)
            tmp_max = np.abs(tmp_audio).max()
            if tmp_max > 2.2:
                print(f"{wav_name}-filtered,{tmp_max}")
                continue
            tmp_audio32 = (tmp_audio / tmp_max * (MAXX * ALPHA * 32768)) \
                + ((1 - ALPHA) * 32768) * tmp_audio
            tmp_audio32b = (tmp_audio / tmp_max * (MAXX * ALPHA * 1145.14)) \
                + ((1 - ALPHA) * 1145.14) * tmp_audio
            tmp_audio16 = librosa.resample(tmp_audio32b, orig_sr=32000, target_sr=16000)
            t_io += time.perf_counter() - t1

            t1 = time.perf_counter()
            ssl = model(mx.array(tmp_audio16[None].astype(np.float32)))  # (1,T',768)
            ssl = mx.transpose(ssl, (0, 2, 1))  # (1, 768, T)
            ssl_np = np.asarray(ssl, dtype=np.float32)
            mx.eval(ssl)
            t_gpu += time.perf_counter() - t1
            if np.isnan(ssl_np).sum() != 0:
                nan_fails.append((wav_name, wav_path))
                print(f"nan filtered:{wav_name}")
                continue
            sf.write(os.path.join(wav32dir, wav_name), tmp_audio32.astype("int16"), 32000,
                     subtype="PCM_16")
            np.save(hubert_path, ssl_np.astype(np.float16))
            n_ok += 1
        except Exception:
            import traceback
            print(line[:80], traceback.format_exc(), file=sys.stderr)
    if nan_fails:
        print(f"[stage2] NaN-failed files (official would retry in fp32): "
              f"{[w for w, _ in nan_fails]}")
    timing["stage2_process"] = t_io + t_gpu
    timing["stage2_io"] = t_io
    timing["stage2_gpu"] = t_gpu
    print(f"[stage2] {n_ok} files -> {hubert_dir} + {wav32dir} "
          f"(io {t_io:.1f}s / gpu {t_gpu:.1f}s)")


# ---------------------------------------------------------------------------
# stage 2b: sv (v2Pro/ProPlus)
# ---------------------------------------------------------------------------

def run_sv(args, timing: dict) -> None:
    import numpy as np
    import mlx.core as mx

    from gsovits_mlx.audio_resample import resample_torchaudio
    from gsovits_mlx.pipeline import load_sv_encoder
    from gsovits_mlx.sovits.kaldi_fbank import fbank

    sv_cn_dir = os.path.join(args.exp_dir, "7-sv_cn")
    wav32dir = os.path.join(args.exp_dir, "5-wav32k")
    os.makedirs(sv_cn_dir, exist_ok=True)

    t0 = time.perf_counter()
    sv_model, _ = load_sv_encoder(os.path.join(args.models_root, "sv"))
    timing["stage2b_model_load"] = time.perf_counter() - t0

    n_ok = 0
    t_io = t_gpu = 0.0
    for line in read_list_lines(args.list, args.part, args.parts):
        try:
            parsed = parse_list_line(line)
            if parsed is None:
                continue
            wav_name, _lan, _text = parsed
            sv_path = os.path.join(sv_cn_dir, wav_name + ".npy")
            if os.path.exists(sv_path):
                continue
            wav32_path = os.path.join(wav32dir, wav_name)
            if not os.path.exists(wav32_path):
                continue
            t1 = time.perf_counter()
            import soundfile as sf
            wav32k, sr0 = sf.read(wav32_path, dtype="float32", always_2d=True)
            assert sr0 == 32000
            # official: torchaudio.transforms.Resample(32000, 16000) —
            # sinc_interp_hann (kaiser-free), NOT soxr; numpy port in
            # gsovits_mlx.audio_resample (max diff 1.8e-7 vs torchaudio).
            wav16k = resample_torchaudio(wav32k[:, 0], 32000, 16000)
            t_io += time.perf_counter() - t1
            t1 = time.perf_counter()
            feat = fbank(wav16k)
            emb = sv_model.forward3(mx.array(feat[None]))
            mx.eval(emb)
            t_gpu += time.perf_counter() - t1
            np.save(sv_path, np.asarray(emb, dtype=np.float32).reshape(1, -1))
            n_ok += 1
        except Exception:
            import traceback
            print(line[:80], traceback.format_exc(), file=sys.stderr)
    timing["stage2b_process"] = t_io + t_gpu
    timing["stage2b_io"] = t_io
    timing["stage2b_gpu"] = t_gpu
    print(f"[stage2b] {n_ok} sv embeddings -> {sv_cn_dir} "
          f"(io {t_io:.1f}s / gpu {t_gpu:.1f}s)")


# ---------------------------------------------------------------------------
# stage 3: semantic
# ---------------------------------------------------------------------------

def run_semantic(args, timing: dict) -> None:
    import numpy as np
    import mlx.core as mx

    from gsovits_mlx.pipeline import _load_sovits_v1v2

    hubert_dir = os.path.join(args.exp_dir, "4-cnhubert")
    semantic_path = os.path.join(args.exp_dir, f"6-name2semantic-{args.part}.tsv")
    if os.path.exists(semantic_path):
        print(f"[stage3] {semantic_path} exists, skipping")
        return

    t0 = time.perf_counter()
    # official infers version from the s2G file size; we take the explicit
    # --version and map to the s2 checkpoint family (v1/v2/v2Pro/v2ProPlus
    # all extract codes via the same SynthesizerTrn quantizer path; Pro uses
    # its own s2G with identical quantizer semantics -> per e2e.py
    # VERSION_CONFIG sovits_dir).
    sovits_dir_map = {"v1": "v1", "v2": "v2",
                      "v2Pro": "v2pro", "v2ProPlus": "v2proplus"}
    sov_dir = os.path.join(args.models_root, sovits_dir_map[args.version])
    sov, _meta = _load_sovits_v1v2(sov_dir, args.version)
    timing["stage3_model_load"] = time.perf_counter() - t0

    # fp32 semantic quantization (PARITY choice): the converted weights are
    # fp16 (production inference default) but the quantizer's expand-form L2
    # (||x||^2+||e||^2-2xe) in fp16 suffers catastrophic cancellation
    # (~10 distance error at |d|~700, flipping argmin on sub-0.5 margins,
    # measured 24/2915 frames vs torch fp32). Upcast ssl_proj + codebook to
    # fp32 and compute the diff-form distance — matches torch fp32 codes
    # exactly; official CUDA is_half runs carry the same fp16 cancellation
    # noise with a different rounding pattern (not bit-replicable on MLX).
    import mlx.core as mx
    w32 = sov.ssl_proj.weight.astype(mx.float32)
    b32 = sov.ssl_proj.bias.astype(mx.float32)
    emb32 = sov.quantizer.embed.astype(mx.float32)

    def codes_fp32(ssl_fp16: "np.ndarray") -> "np.ndarray":
        x = mx.array(ssl_fp16.astype(np.float32))  # (1,768,T)
        xt = mx.transpose(x, (0, 2, 1))            # (1,T,768)
        # torch Conv1d(k=2, stride=2, no pad) DROPS a trailing odd sample;
        # truncate to even T (zero-padding instead would contaminate the
        # last window with a fake 0-value frame)
        T = xt.shape[1]
        if T % 2:
            xt = xt[:, :T - 1, :]
        proj = mx.conv1d(xt, w32, stride=2) + b32  # (1,T',768)
        d = mx.sum((proj[:, :, None, :] - emb32[None, None, :, :]) ** 2, axis=-1)
        codes = mx.argmin(d, axis=-1)              # (1,T')
        out = codes
        mx.eval(out)
        return np.asarray(out, np.int32)[0]

    lines_out = []
    t_io = t_gpu = 0.0
    for line in read_list_lines(args.list, args.part, args.parts):
        try:
            parsed = parse_list_line(line)
            if parsed is None:
                continue
            wav_name, _lan, _text = parsed
            t1 = time.perf_counter()
            hubert_path = os.path.join(hubert_dir, wav_name + ".npy")
            if not os.path.exists(hubert_path):
                continue
            ssl = np.load(hubert_path)  # fp16 -> fp32 compute (documented)
            t_io += time.perf_counter() - t1
            t1 = time.perf_counter()
            code_vec = codes_fp32(ssl)
            t_gpu += time.perf_counter() - t1
            semantic = " ".join(str(int(i)) for i in code_vec.tolist())
            lines_out.append("%s\t%s" % (wav_name, semantic))
        except Exception:
            import traceback
            print(line[:80], traceback.format_exc(), file=sys.stderr)
    with open(semantic_path, "w", encoding="utf8") as f:
        f.write("\n".join(lines_out))
    timing["stage3_process"] = t_io + t_gpu
    timing["stage3_io"] = t_io
    timing["stage3_gpu"] = t_gpu
    print(f"[stage3] {len(lines_out)} lines -> {semantic_path} "
          f"(io {t_io:.1f}s / gpu {t_gpu:.1f}s)")


# ---------------------------------------------------------------------------
# merge (webui's part-concat)
# ---------------------------------------------------------------------------

def run_merge(args) -> None:
    # 2-name2text: plain concat of part files (webui order: part 0..N-1).
    # Guard: a previous merge consuming part files leaves the final file
    # present with fewer parts — refuse to silently re-merge leftovers.
    txt_out = os.path.join(args.exp_dir, "2-name2text.txt")
    part_files = [i for i in range(args.parts)
                  if os.path.exists(os.path.join(args.exp_dir, f"2-name2text-{i}.txt"))]
    if not part_files:
        print("[merge] no 2-name2text part files present (already merged?)")
    else:
        if os.path.exists(txt_out):
            raise SystemExit(
                f"[merge] {txt_out} already exists but part files {part_files} "
                "remain — delete the merged file first (double merge would "
                "drop the earlier parts).")
        opt = []
        for i in part_files:
            p = os.path.join(args.exp_dir, f"2-name2text-{i}.txt")
            with open(p, "r", encoding="utf8") as f:
                opt += f.read().strip("\n").split("\n")
            os.remove(p)
        with open(txt_out, "w", encoding="utf8") as f:
            f.write("\n".join(opt) + "\n")
        print(f"[merge] {len(opt)} lines -> {txt_out}")
    # 6-name2semantic: header + concat (webui prepends the header line)
    sem_out = os.path.join(args.exp_dir, "6-name2semantic.tsv")
    part_files = [i for i in range(args.parts)
                  if os.path.exists(os.path.join(args.exp_dir, f"6-name2semantic-{i}.tsv"))]
    if not part_files:
        print("[merge] no 6-name2semantic part files present (already merged?)")
        return
    if os.path.exists(sem_out):
        raise SystemExit(
            f"[merge] {sem_out} already exists but part files {part_files} "
            "remain — delete the merged file first.")
    opt = ["item_name\tsemantic_audio"]
    for i in part_files:
        p = os.path.join(args.exp_dir, f"6-name2semantic-{i}.tsv")
        with open(p, "r", encoding="utf8") as f:
            opt += f.read().strip("\n").split("\n")
        os.remove(p)
    with open(sem_out, "w", encoding="utf8") as f:
        f.write("\n".join(opt) + "\n")
    print(f"[merge] {len(opt) - 1} lines -> {sem_out}")


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--list", required=True, help="wav|spk|lang|text list file")
    ap.add_argument("--wav-dir", required=True, help="dir containing the wavs")
    ap.add_argument("--exp-dir", required=True, help="output exp dir")
    ap.add_argument("--version", default="v2",
                    choices=["v1", "v2", "v2Pro", "v2ProPlus"],
                    help="version arg for clean_text symbol tables + semantic model")
    ap.add_argument("--models-root", default=os.path.join(REPO, "models_local"))
    ap.add_argument("--cpufast-repo", default=None,
                    help="CPUFast checkout for the vendored text front-end "
                         "(default: gsovits_mlx.text.vendored_cpufront default)")
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--stages", default="all",
                    help="comma list of: text,hubert,sv,semantic,merge | all")
    ap.add_argument("--timing-json", default=None,
                    help="write per-stage timing breakdown to this file")
    args = ap.parse_args()

    # bootstrap() chdirs into the CPUFast repo before stage-1 runs, so any
    # RELATIVE path argument would resolve against the wrong cwd afterwards
    # (g2pw tokenizer FileNotFoundError, missing wav list, exp dir inside
    # CPUFast). Normalize every path arg to absolute up front.
    for p in ("list", "wav_dir", "exp_dir", "models_root"):
        setattr(args, p, os.path.abspath(os.path.expanduser(getattr(args, p))))
    if args.cpufast_repo:
        args.cpufast_repo = os.path.abspath(os.path.expanduser(args.cpufast_repo))
    if args.timing_json:
        args.timing_json = os.path.abspath(args.timing_json)

    if args.part >= args.parts:
        raise SystemExit(f"--part {args.part} out of range for --parts {args.parts}")
    stages = (["text", "hubert", "sv", "semantic", "merge"]
              if args.stages == "all" else args.stages.split(","))

    # GPU lock discipline (AGENTS.md): MLX stages need the shared lock.
    from gsovits_mlx.gpu_lock import resolve_device
    need_gpu = any(s in stages for s in ("text", "hubert", "sv", "semantic"))
    frontend_cpu = os.environ.get("GSOVITS_FRONTEND_DEVICE") == "cpu" \
        or os.environ.get("GSOVITS_PREP_CPU") == "1"
    device = resolve_device(flag_gpu=not frontend_cpu, verbose=True) if need_gpu else "cpu"
    if need_gpu and device != "gpu":
        if frontend_cpu:
            # explicit CPU smoke: front-end on mx CPU is legal (parity-tested
            # path); audio/semantic stages still require the lock below.
            device = "cpu"
        else:
            raise SystemExit(
                "[prepare] GPU stages refused: no fresh .tmp/gpu.lock.d. "
                "Take the lock first (see AGENTS.md).")
    heavy_stages = [s for s in stages if s in ("hubert", "sv", "semantic")]
    if heavy_stages and device == "cpu" and not frontend_cpu:
        raise SystemExit(
            f"[prepare] stages {heavy_stages} need Metal (official semantics "
            "verified on GPU); refusing CPU run without GSOVITS_PREP_CPU=1.")

    timing: dict = {"device": device, "parts": args.parts, "part": args.part}
    t_all = time.perf_counter()
    if "text" in stages:
        run_text(args, timing)
    if "hubert" in stages:
        run_hubert(args, timing)
    if "sv" in stages and args.version in ("v2Pro", "v2ProPlus"):
        run_sv(args, timing)
    if "semantic" in stages:
        run_semantic(args, timing)
    if "merge" in stages:
        run_merge(args)
    timing["total"] = time.perf_counter() - t_all
    if args.timing_json:
        with open(args.timing_json, "w") as f:
            json.dump(timing, f, indent=2)
    print(json.dumps(timing, indent=2))


if __name__ == "__main__":
    main()
