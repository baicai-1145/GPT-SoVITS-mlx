"""End-to-end v4 TTS on MLX: CFM(DiT, 32-step Euler, cfg per resolve_sampling)
+ HiFi-GAN Generator vocoder, 48 kHz output.

Replicates GPT-SoVITS-CPUFast TTS.py v4 path (using_vocoder_synthesis with
use_vocoder=True; init_vocoder('v4')). Same CFM/DiT as v3 (interp scale 2
instead of 1.875), prompt mel via mel_fn_v4 (1280/320 @ 32 kHz), vocoder
Generator upsampling 10*6*2*2*2 = 480x -> 48 kHz, T_ref=500/T_chunk=1000.
See docs/PARITY_NOTES.md "v3 addendum" (shared CFM semantics) and the
"v4 addendum" for version-specific bits.

Usage:
    python3 tools/e2e_v4.py \
        --text "你好，欢迎来到各自的旅程。今天我们聊聊机器学习。" \
        --ref-audio /Volumes/2T/gpt-sovits-models/bench/ref_zh_3.5s.wav \
        --ref-text "希望你以后能够做得比我还好哟。" \
        --out out_v4.wav

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

import numpy as np

DEFAULT_MODELS_ROOT = "/Volumes/2T/gpt-sovits-models/mlx"
DEFAULT_CPUFAST_REPO = "/Users/baicai1145/repos/gpt-sovits/GPT-SoVITS-CPUFast"

# v4 AR/CFM defaults (official CPUFast: TTS_Config.hz=50, v4 uses the s1v3
# ckpt so max_sec=57; resolve_sampling('v4') -> (32, 0.0))
DEFAULT_TOP_K = 15
DEFAULT_TOP_P = 1.0
DEFAULT_TEMPERATURE = 1.0
DEFAULT_REPETITION_PENALTY = 1.35
DEFAULT_EARLY_STOP_NUM = 2850  # 50 * 57
DEFAULT_SEED = 0
DEFAULT_SAMPLE_STEPS = 32
DEFAULT_CFG_RATE = 0.0

SPEC_MIN = -12.0
SPEC_MAX = 2.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--text", required=True, help="Target text to synthesize.")
    p.add_argument("--ref-audio", required=True,
                   help="Reference audio path (any sample rate; resampled internally).")
    p.add_argument("--ref-text", required=True, help="Transcript of the reference audio.")
    p.add_argument("--models-root", default=DEFAULT_MODELS_ROOT,
                   help="Dir containing bert/, hubert/, s1/, v4/, v4_vocoder/ MLX exports "
                        f"(default: {DEFAULT_MODELS_ROOT})")
    p.add_argument("--cpufast-repo",
                   default=os.environ.get("GPT_SOVITS_CPUFAST", DEFAULT_CPUFAST_REPO),
                   help="GPT-SoVITS-CPUFast checkout, used read-only for the text front-end.")
    p.add_argument("--lang", default="zh", help="Text language for the cleaner (default: zh).")
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--repetition-penalty", type=float, default=DEFAULT_REPETITION_PENALTY)
    p.add_argument("--early-stop-num", type=int, default=DEFAULT_EARLY_STOP_NUM)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--sample-steps", type=int, default=DEFAULT_SAMPLE_STEPS,
                   help="CFM Euler steps (official v4 default: resolve_sampling -> 32).")
    p.add_argument("--cfg-rate", type=float, default=DEFAULT_CFG_RATE,
                   help="CFM inference_cfg_rate (official v4 default: 0.0).")
    p.add_argument("--out", default="out_v4.wav", help="Output wav path (48 kHz).")
    p.add_argument("--bench", action="store_true",
                   help="Print per-stage wall times and peak RSS to stderr.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Text front-end (the only piece borrowed from the official CPUFast repo) —
# identical to e2e_v2.py; v4 uses the v2 symbol table and cleaner.
# ---------------------------------------------------------------------------

def init_text_frontend(cpufast_repo: str, bert_path: str):
    """Import the official cleaner + segmentation helpers (read-only use)."""
    if not os.path.isdir(cpufast_repo):
        sys.exit(f"CPUFast repo not found at {cpufast_repo} "
                 "(set --cpufast-repo or GPT_SOVITS_CPUFAST)")
    os.chdir(cpufast_repo)
    sys.path.insert(0, "GPT_SoVITS")
    sys.path.insert(0, ".")
    os.environ["bert_path"] = bert_path
    from text.cleaner import clean_text
    from text import cleaned_text_to_sequence
    from TTS_infer_pack.text_segmentation_method import splits
    return clean_text, cleaned_text_to_sequence, splits


def phones_and_bert(clean_text, cleaned_text_to_sequence, splits, text: str,
                    lang: str, version: str, is_prompt: bool = False):
    """Phones + BERT feature tuple for one segment.

    Official pre_seg_text rule: the '。' separator is prepended to the TARGET
    text only (never the prompt) when the text does not already start with a
    split separator and its first segment is shorter than 4 chars.
    """

    def _get_first(t: str) -> str:
        pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
        m = re.match(pattern + "+", t)
        return m.group() if m else ""

    if not is_prompt and text[0] not in splits and len(_get_first(text)) < 4:
        text = "。" + text

    phones, word2ph, norm = clean_text(text, lang, version)
    ids = cleaned_text_to_sequence(phones, version=version)
    return ids, (ids, word2ph, norm)


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
    CPUFast runner (PyAV decode + swr). Required for bit-exact prompt codes."""
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
    from gsovits_mlx.pipeline import resample_linear
    import soundfile as sf
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    mono = resample_linear(mono, sr, 16000) if sr != 16000 else mono
    return mono if target_sr == 16000 else resample_linear(mono, 16000, target_sr)


# ---------------------------------------------------------------------------
# Pipeline (pure MLX below this line)
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
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
    from gsovits_mlx.text.mel_frontend import mel_spectrogram, spectrogram
    from gsovits_mlx.pipeline import (denorm_spec, load_gpt, load_sovits_v3,
                                      load_vocoder_v4, cfm_chunked_decode_v4,
                                      norm_spec)

    # --- 1. text front-end: phones + BERT features for prompt and target ---
    t0 = time.perf_counter()
    clean_text, cleaned_text_to_sequence, splits = init_text_frontend(
        args.cpufast_repo, os.path.join(args.models_root, "..", "pretrained_models",
                                        "chinese-roberta-wwm-ext-large"))
    bert_dir = os.path.join(args.models_root, "bert")
    bert_arrays = load_mlx_safetensors(os.path.join(bert_dir, "bert.safetensors"))
    bert_arrays = {k: mx.array(np.asarray(v, np.float32)) for k, v in bert_arrays.items()
                   if "position_ids" not in k}
    bert_cfg = json.load(open(os.path.join(bert_dir, "config.json")))
    bert = BertModel(bert_arrays, bert_cfg)
    tok = load_bert_tokenizer(find_tokenizer_json(args.models_root))

    def bert_feat(seg_info):
        cols = []
        for ids, w2p, norm in seg_info:
            enc = encode_text(tok, norm)
            f = bert.get_bert_feature(mx.array(enc["input_ids"], mx.int32),
                                      mx.array(enc["attention_mask"], mx.float32), w2p)
            mx.eval(f)
            cols.append(np.array(f))
        return np.concatenate(cols, axis=1)  # (1024, T)

    # v3 symbols == v2 (text/symbols2.py); cleaner version "v2"
    p_ids, p_info = phones_and_bert(clean_text, cleaned_text_to_sequence, splits,
                                    args.ref_text, args.lang, "v2", is_prompt=True)
    t_ids, t_info = phones_and_bert(clean_text, cleaned_text_to_sequence, splits,
                                    args.text, args.lang, "v2", is_prompt=False)
    p_bert = bert_feat([p_info])
    t_bert = bert_feat([t_info])
    all_phones = mx.array([p_ids + t_ids], mx.int32)
    all_bert = mx.array(np.concatenate([p_bert, t_bert], axis=1))[None]  # (1, 1024, Tp+Tt)
    mx.eval(all_phones, all_bert)
    times["frontend"] = time.perf_counter() - t0
    if args.bench:
        print(f"[bench] frontend: phones={all_phones.shape} bert={all_bert.shape} "
              f"{times['frontend']:.2f}s", file=sys.stderr)

    # --- 2. prompt semantic codes: HuBERT -> v4 SoVITS quantizer ---
    t0 = time.perf_counter()
    # Official _set_prompt_semantic: 16 kHz wav + zero_wav = int(32000 * 0.3) = 9600.
    wav16k = load_audio_official(args.ref_audio, 16000)
    wav16k = np.concatenate([wav16k, np.zeros(9600, np.float32)])
    hubert_dir = os.path.join(args.models_root, "hubert")
    hb = HubertModel(load_mlx_safetensors(os.path.join(hubert_dir, "hubert.safetensors")),
                     json.load(open(os.path.join(hubert_dir, "config.json"))))
    h = hb(mx.array(wav16k[None]))
    mx.eval(h)
    sov, _meta = load_sovits_v3(os.path.join(args.models_root, "v4"), "v4")
    prompt_sem = sov.extract_latent(mx.transpose(h, (0, 2, 1)))  # (1, Tp) int
    mx.eval(prompt_sem)
    times["prompt_codes"] = time.perf_counter() - t0
    if args.bench:
        print(f"[bench] prompt_codes: {prompt_sem.shape} "
              f"{times['prompt_codes']:.2f}s", file=sys.stderr)

    # --- 3. reference spectrogram: 2048/640/2048, v3 ref_enc crops :704 ---
    t0 = time.perf_counter()
    w32 = load_audio_official(args.ref_audio, 32000)
    refer = spectrogram(mx.array(w32[None]), 2048, 640, 2048)  # decode_encp crops :704
    mx.eval(refer)
    times["refer_spec"] = time.perf_counter() - t0
    if args.bench:
        print(f"[bench] refer_spec: {refer.shape} {times['refer_spec']:.2f}s",
              file=sys.stderr)

    # --- 4. AR GPT (s1 == s1v3.ckpt conversion): gen semantic tokens ---
    t0 = time.perf_counter()
    gpt = load_gpt(os.path.join(args.models_root, "s1"))
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

    # --- 5. SoVITS v4 decode: codes -> fea -> chunked CFM -> Generator 48 kHz ---
    t0 = time.perf_counter()
    all_codes = mx.concatenate([prompt_sem, seq], axis=1)[:, None, :]  # (1, 1, Tp+Tg)
    ge = None
    refer_mask = mx.ones((refer.shape[0], 1, refer.shape[2]), dtype=refer.dtype)
    ge = sov.ref_enc(refer[:, :704] * refer_mask, refer_mask)
    fea_ref, ge = sov.decode_encp(mx.array([prompt_sem]), all_phones[:, :len(p_ids)],
                                  refer=refer, ge=ge)
    fea_todo, ge = sov.decode_encp(all_codes, all_phones, refer=refer, ge=ge,
                                   speed=1.0)
    # using_vocoder_synthesis slices fea_todo from T_min (prompt frames already in
    # fea_ref); official drops nothing here because decode_encp is re-run on the
    # FULL code stream and the chunk loop starts at the prompt boundary.
    mx.eval(fea_ref, fea_todo)

    # prompt mel2: mel_fn_v4 (100-mel, 1280/320 @ 32 kHz, center=False) on the ref
    # audio resampled to 32 kHz, then norm_spec
    ref_audio_32k = load_audio_official(args.ref_audio, 32000)
    mel2 = mel_spectrogram(mx.array(ref_audio_32k[None]), 1280, 100, 32000,
                           320, 1280, 0, None, center=False)
    mel2 = norm_spec(mel2)
    mx.eval(mel2)

    pred = cfm_chunked_decode_v4(sov, fea_ref, fea_todo, mel2,
                                 sample_steps=args.sample_steps,
                                 inference_cfg_rate=args.cfg_rate,
                                 key=mx.random.key(args.seed))
    pred = denorm_spec(pred)
    mx.eval(pred)

    voc = load_vocoder_v4(os.path.join(args.models_root, "v4_vocoder"))
    audio = voc(pred)
    mx.eval(audio)
    times["sovits_vocoder"] = time.perf_counter() - t0

    # vocoder upsamples 480x per mel frame (48 kHz)
    audio_np = np.array(audio)[0, 0]
    sf.write(args.out, audio_np, 48000)
    if args.bench:
        rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(f"[bench] sovits+vocoder: {audio_np.shape} {times['sovits_vocoder']:.2f}s | "
              f"total {time.perf_counter() - t_all:.2f}s "
              f"peak_rss={rss_bytes // 1024} KB", file=sys.stderr)
    print(f"saved {args.out} ({len(audio_np) / 48000:.2f}s, {n_gen} AR tokens, "
          f"{args.sample_steps} CFM steps)")


if __name__ == "__main__":
    main()
