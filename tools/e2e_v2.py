"""End-to-end v2 TTS on MLX, replicating GPT-SoVITS-CPUFast inference semantics.

Consolidated from .tmp/e2e_v2b.py after official-parity verification
(see docs/PARITY_NOTES.md for the captured quirks this replicates).

Usage:
    python3 tools/e2e_v2.py \
        --text "你好，欢迎来到各自的旅程。今天我们聊聊机器学习。" \
        --ref-audio /Volumes/2T/gpt-sovits-models/bench/ref_zh_3.5s.wav \
        --ref-text "希望你以后能够做得比我还好哟。" \
        --out out.wav

Requires the GPT-SoVITS-CPUFast checkout ONLY for the text front-end
(text.cleaner + TTS_infer_pack.text_segmentation_method); configure via
--cpufast-repo or the GPT_SOVITS_CPUFAST env var. All model math is MLX.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import resource
import sys
import time
import types

import numpy as np

DEFAULT_MODELS_ROOT = "/Volumes/2T/gpt-sovits-models/mlx"
DEFAULT_CPUFAST_REPO = "/Users/baicai1145/repos/gpt-sovits/GPT-SoVITS-CPUFast"

# v2 AR sampling defaults, verified against official CPUFast
# (seed study + logits replay; see docs/PARITY_NOTES.md)
DEFAULT_TOP_K = 15
DEFAULT_TOP_P = 1.0
DEFAULT_TEMPERATURE = 1.0
DEFAULT_REPETITION_PENALTY = 1.35
DEFAULT_EARLY_STOP_NUM = 2850  # official v2 default (1500 for v1)
DEFAULT_NOISE_SCALE = 0.5
DEFAULT_SEED = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--text", required=True, help="Target text to synthesize.")
    p.add_argument("--ref-audio", required=True,
                   help="Reference audio path (any sample rate; resampled internally).")
    p.add_argument("--ref-text", required=True, help="Transcript of the reference audio.")
    p.add_argument("--models-root", default=DEFAULT_MODELS_ROOT,
                   help="Dir containing bert/, hubert/, s1v2/, v2/ MLX exports "
                        f"(default: {DEFAULT_MODELS_ROOT})")
    p.add_argument("--cpufast-repo",
                   default=os.environ.get("GPT_SOVITS_CPUFAST", DEFAULT_CPUFAST_REPO),
                   help="GPT-SoVITS-CPUFast checkout, used read-only for the text front-end.")
    p.add_argument("--lang", default="zh",
                   help="Text language mode: zh/ja/ko/yue/en/all_zh/all_ja/"
                        "all_ko/all_yue/auto/auto_yue (default: zh; mixed text"
                        " under a single-language label takes the label lang).")
    p.add_argument("--prompt-lang", default=None,
                   help="Language mode for the reference text (default: --lang).")
    p.add_argument("--text-split-method", default="cut0",
                   help="Official cut0..cut5 split method for the target text "
                        "(default cut0 = no splitting, stage-1 behavior).")
    p.add_argument("--ref-cache", default=os.environ.get("GSOVITS_REF_CACHE", ""),
                   help="Enable the (ref_audio, ref_text) disk cache at this "
                        "dir (or GSOVITS_REF_CACHE; empty = off).")
    p.add_argument("--cleaner-version", default="v2", choices=["v1", "v2"],
                   help="Phone-cleaner version (default: v2).")
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--repetition-penalty", type=float, default=DEFAULT_REPETITION_PENALTY)
    p.add_argument("--early-stop-num", type=int, default=DEFAULT_EARLY_STOP_NUM)
    p.add_argument("--noise-scale", type=float, default=DEFAULT_NOISE_SCALE)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help="AR sampling key seed (decode noise uses key(seed) as well).")
    p.add_argument("--out", default="out_v2.wav", help="Output wav path (32 kHz).")
    p.add_argument("--gpu", action="store_true",
                   help="Opt in to Metal for ALL stages: requires GSOVITS_GPU_LOCK_OK=1\n                        and a fresh .tmp/gpu.lock.d owner (default-deny; CPU otherwise).")
    p.add_argument("--bench", action="store_true",
                   help="Print per-stage wall times and peak RSS to stderr.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Audio decoding (mirror official tools/audio_utils.py: PyAV + swr resampler)
# ---------------------------------------------------------------------------

try:
    import av  # PyAV, ships with the official repo's requirements
except ImportError:  # pragma: no cover
    av = None


def _decode_audio_native(path: str) -> tuple[np.ndarray, int]:
    """Decode to (channels, samples) float32 at the file's native rate (official
    _decode_audio_array). Returns (mono_1d, sr) when the file is mono."""
    with av.open(path) as container:
        stream = container.streams.audio[0]
        sr = int(stream.codec_context.sample_rate or stream.rate or 16000)
        channels = int(stream.codec_context.channels or 1)
        resampler = av.AudioResampler(format="fltp", layout="mono" if channels == 1 else "stereo",
                                      rate=sr)
        chunks = []
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray())
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray())
    if not chunks:
        raise RuntimeError(f"failed to decode audio: {path}")
    audio = np.concatenate(chunks, axis=1).astype(np.float32, copy=False)
    return np.ascontiguousarray(audio), sr


def _swr_resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """(channels, samples) or (samples,) -> resampled, official _resample_audio_array."""
    squeeze = audio.ndim == 1
    if squeeze:
        audio = audio[np.newaxis, :]
    frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(audio, np.float32),
                                       format="fltp", layout="mono" if audio.shape[0] == 1 else "stereo")
    frame.sample_rate = int(sr_in)
    resampler = av.AudioResampler(format="fltp",
                                  layout="mono" if audio.shape[0] == 1 else "stereo",
                                  rate=int(sr_out))
    chunks = [out.to_ndarray() for f in ([frame] + [None]) for out in resampler.resample(f)]
    out = np.concatenate(chunks, axis=1).astype(np.float32, copy=False)
    return out[0] if squeeze else out


def load_audio_official(path: str, target_sr: int) -> np.ndarray:
    """Mono float32 wav at target_sr, decoded/resampled exactly like the official
    CPUFast runner (PyAV decode + swr). This is required for bit-exact prompt
    semantic codes; soundfile+linear resample diverges in HuBERT frames.

    Without PyAV, falls back to soundfile + linear resample (audio still
    intelligible; codes may differ from official at a few frames)."""
    if av is not None:
        audio, sr = _decode_audio_native(path)
        if audio.shape[0] > 1:
            audio = audio.mean(axis=0)
        else:
            audio = audio[0]
        return _swr_resample(audio, sr, target_sr) if sr != target_sr else audio
    import warnings
    warnings.warn("PyAV not available; falling back to soundfile + linear resample "
                  "(prompt codes may deviate slightly from official)")
    from gsovits_mlx.pipeline import load_audio_16k, resample_linear
    mono = load_audio_16k(path)
    return mono if target_sr == 16000 else resample_linear(mono, 16000, target_sr)


# ---------------------------------------------------------------------------
# Pipeline (pure MLX below this line)
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    # Process-wide device gate (default-deny): pins MLX to CPU unless
    # --gpu/GSOVITS_GPU_LOCK_OK AND a fresh gpu.lock. See gsovits_mlx.gpu_lock.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from gsovits_mlx.gpu_lock import resolve_device
    resolve_device(flag_gpu=args.gpu, verbose=True)
    # AR sampling draws its inverse-CDF uniform from numpy's global RNG
    # (gsovits_mlx/gpt/t2s.py::_sample); seed it so runs are reproducible.
    np.random.seed(args.seed)
    # init_text_frontend() chdirs into the CPUFast repo (its cleaner imports
    # need the CWD); resolve any relative --out before that happens.
    args.out = os.path.abspath(args.out)
    t_all = time.perf_counter()
    times: dict[str, float] = {}

    import mlx.core as mx
    import soundfile as sf
    
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from gsovits_mlx.text.bert_tokenizer import encode_text, find_tokenizer_json, load_bert_tokenizer
    from gsovits_mlx.io import load_mlx_safetensors
    from gsovits_mlx.text.bert import BertModel
    from gsovits_mlx.text.hubert import HubertModel
    from gsovits_mlx.text.mel_frontend import spectrogram
    from gsovits_mlx.pipeline import _load_sovits_v1v2, load_gpt

    # resolve --ref-cache BEFORE bootstrap() chdirs into the CPUFast repo
    if args.ref_cache:
        args.ref_cache = os.path.abspath(args.ref_cache)
    # --- 1. text front-end: phones + BERT features for prompt and target ---
    # Official semantics via the vendored CPUFast modules (multi-language,
    # pre_seg_text rules, non-zh all-zero bert); zh G2PW runs on MLX (no
    # torch). cut0 + zh defaults keep the stage-1 75/75 parity path exact.
    t0 = time.perf_counter()
    from gsovits_mlx.text.preproc import TextFrontend, bootstrap
    from gsovits_mlx.text import ref_cache

    bootstrap(args.cpufast_repo, models_root=args.models_root)
    # MLX device for the front-end (BERT/G2PW). Default cpu: the e2e GPU
    # stages run under the gpu.lock discipline; the front-end only needs
    # Metal for front-end-only speed runs (GSOVITS_FRONTEND_DEVICE=gpu).
    fe = TextFrontend(models_root=args.models_root, lazy=True,
                      device=os.environ.get("GSOVITS_FRONTEND_DEVICE", "cpu"))
    prompt_lang = args.prompt_lang or args.lang

    def frontend_run():
        cache_key = ref_cache.cache_key(
            args.ref_audio, args.ref_text, prompt_lang, args.cleaner_version) \
            if args.ref_cache else None
        if cache_key:
            got = ref_cache.get_cached(ref_cache.cache_dir(args.ref_cache), cache_key)
            if got:
                _, arrays = got
                return arrays["p_ids"], arrays["p_bert"], \
                    arrays["t_ids"], arrays["t_bert"], True
        p = fe.segment_prompt(args.ref_text, prompt_lang, args.cleaner_version)
        t = fe.preprocess(args.text, args.lang, args.text_split_method,
                          args.cleaner_version)
        p_ids = p.phones
        p_bert = p.bert
        # multiple pre_seg_text segments are separate FrontEndResults; the
        # AR/SoVITS consume the concatenated phones+bert (official
        # batched path concatenates identically)
        t_ids = [ph for r in t for ph in r.phones]
        t_bert = np.concatenate([r.bert for r in t], axis=1)
        if cache_key:
            try:
                ref_cache.put_cached(
                    ref_cache.cache_dir(args.ref_cache), cache_key,
                    {"ref_text": args.ref_text, "lang": args.lang,
                     "prompt_lang": prompt_lang,
                     "version": args.cleaner_version},
                    {"p_ids": np.asarray(p_ids, np.int32), "p_bert": p_bert,
                     "t_ids": np.asarray(t_ids, np.int32), "t_bert": t_bert})
            except OSError:
                pass
        return p_ids, p_bert, t_ids, t_bert, False

    p_ids, p_bert, t_ids, t_bert, cache_hit = frontend_run()
    all_phones = mx.array([list(p_ids) + list(t_ids)], mx.int32)
    # all_bert = concat(prompt_bert, target_bert) along time -- official layout
    all_bert = mx.array(np.concatenate([np.asarray(p_bert), np.asarray(t_bert)], axis=1))[None]  # (1, 1024, Tp+Tt)
    mx.eval(all_phones, all_bert)
    times["frontend"] = time.perf_counter() - t0
    if args.bench:
        print(f"[bench] frontend: phones={all_phones.shape} bert={all_bert.shape} "
              f"{'CACHE_HIT ' if cache_hit else ''}{times['frontend']:.2f}s", file=sys.stderr)

    # --- 2. prompt semantic codes: HuBERT -> SoVITS quantizer ---
    t0 = time.perf_counter()
    # Official _set_prompt_semantic: 16 kHz wav + zero_wav = int(32000 * 0.3)
    # = 9600 trailing zeros, decoded via PyAV/swr (bit-exact codes vs official:
    # 101/101 on the bench ref; see docs/PARITY_NOTES.md).
    wav16k = load_audio_official(args.ref_audio, 16000)
    wav16k = np.concatenate([wav16k, np.zeros(9600, np.float32)])
    hubert_dir = os.path.join(args.models_root, "hubert")
    hb = HubertModel(load_mlx_safetensors(os.path.join(hubert_dir, "hubert.safetensors")),
                     json.load(open(os.path.join(hubert_dir, "config.json"))))
    h = hb(mx.array(wav16k[None]))
    mx.eval(h)
    t_load = time.perf_counter()
    sov, _meta = _load_sovits_v1v2(os.path.join(args.models_root, "v2"), "v2")
    times["model_load"] = time.perf_counter() - t_load
    codes = sov.extract_latent(mx.transpose(h, (0, 2, 1)))
    mx.eval(codes)
    # extract_latent -> (B, T, 1); flatten trailing dims to (B, T) = official
    # prompt_semantic (codes[0, 0] on the torch (B, 1, T) layout).
    prompt_sem = mx.array(codes.reshape(codes.shape[0], -1), mx.int32)  # (1, Tp)
    times["prompt_codes"] = time.perf_counter() - t0
    if args.bench:
        print(f"[bench] model_load: sovits {times['model_load']:.2f}s", file=sys.stderr)
        print(f"[bench] prompt_codes: {prompt_sem.shape} "
              f"{times['prompt_codes']:.2f}s", file=sys.stderr)

    # --- 3. reference spectrogram: 2048 n_fft / 640 hop / 2048 win, first 704 bins ---
    t0 = time.perf_counter()
    # Official runner: PyAV decode at native sr -> swr resample to 32 kHz ->
    # spectrogram(center=False). The SynthesizerTrn ref_enc crops :704 itself,
    # so cropping here is idempotent.
    w32 = load_audio_official(args.ref_audio, 32000)
    sp = spectrogram(mx.array(w32[None]), 2048, 640, 2048)
    refer = sp[:, :704]  # v2 refers crop to 704 bins
    mx.eval(refer)
    times["refer_spec"] = time.perf_counter() - t0
    if args.bench:
        print(f"[bench] refer_spec: {refer.shape} {times['refer_spec']:.2f}s",
              file=sys.stderr)

    # --- 4. AR GPT: gen semantic tokens for the target text ---
    t0 = time.perf_counter()
    gpt = load_gpt(os.path.join(args.models_root, "s1v2"))
    seq = gpt.infer(all_phones, all_bert, prompt_sem,
                    top_k=args.top_k, top_p=args.top_p, temperature=args.temperature,
                    repetition_penalty=args.repetition_penalty,
                    early_stop_num=args.early_stop_num, key=mx.random.key(args.seed))
    mx.eval(seq)
    times["ar"] = time.perf_counter() - t0
    n_gen = seq.shape[1]
    if args.bench:
        print(f"[bench] ar: {n_gen} tokens {times['ar']:.2f}s "
              f"({n_gen / times['ar']:.1f} tok/s)", file=sys.stderr)

    # --- 5. SoVITS decode: prompt + gen codes -> 32 kHz audio ---
    t0 = time.perf_counter()
    all_codes = mx.concatenate([prompt_sem, seq], axis=1)[:, None, :]  # (1, 1, Tp+Tg)
    audio, y_mask = sov.decode(all_codes, all_phones, refer,
                               noise_scale=args.noise_scale,
                               key=mx.random.key(args.seed))
    mx.eval(audio)
    times["sovits"] = time.perf_counter() - t0

    n_out = int(y_mask.shape[2] * 960)  # 2*480 samples per semantic frame
    audio_np = np.array(audio)[0, 0][:n_out]
    sf.write(args.out, audio_np, 32000)
    if args.bench:
        rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(f"[bench] sovits: {audio_np.shape} {times['sovits']:.2f}s | "
              f"total {time.perf_counter() - t_all:.2f}s "
              f"peak_rss={rss_bytes // 1024} KB", file=sys.stderr)
    print(f"saved {args.out} ({len(audio_np) / 32000:.2f}s, {n_gen} AR tokens)")


if __name__ == "__main__":
    main()
