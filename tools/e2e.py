"""Unified end-to-end GPT-SoVITS TTS on MLX — one CLI for all 8 versions.

    python3 tools/e2e.py --version v2 \
        --text "你好，欢迎来到各自的旅程。今天我们聊聊机器学习。" \
        --ref-audio ref_zh_3.5s.wav --ref-text "希望你以后能够做得比我还好哟。" \
        --out out_v2.wav

Versions: v1 | v2 | v2Pro | v2ProPlus | v3 | v4 | v5dev | v5turbo.
The per-family pipelines are the ones verified stage-by-stage against
GPT-SoVITS-CPUFast (see docs/PARITY_NOTES.md); tools/e2e_v*.py remain as
thin wrappers (they exec this script with the version pinned) so existing
docs, scripts and tools/bench.py keep working. A wrapper run and the
equivalent `tools/e2e.py --version X` run are byte-identical.

Shared infra (device gate, audio sanity gate, ref-prompt cache, front-end
semantics) lives in gsovits_mlx; see the README front-end +
reproducibility sections for the contracts.
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

DEFAULT_MODELS_ROOT = os.environ.get("GSOVITS_MODELS_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models_local"))
DEFAULT_CPUFAST_REPO = "/Users/baicai1145/repos/gpt-sovits/GPT-SoVITS-CPUFast"

DEFAULT_TOP_K = 15
DEFAULT_TOP_P = 1.0
DEFAULT_TEMPERATURE = 1.0
DEFAULT_REPETITION_PENALTY = 1.35
DEFAULT_NOISE_SCALE = 0.5
DEFAULT_SEED = 0

# Per-version configuration. Everything the shared pipeline needs to know.
VERSION_CONFIG = {
    "v1": dict(
        family="v1v2", s1_dir="s1v1", sovits_dir="v1", cleaner_version="v1",
        early_stop_num=2700,      # hz(50) * max_sec(54) from s1bert25hz-2kh
        eos_drop=True,            # AR drops trailing 1024 before decode
        refer_bins=1025,          # v1 MelStyleEncoder uses the full spec
        out_sr=32000,
    ),
    "v2": dict(
        family="v1v2", s1_dir="s1v2", sovits_dir="v2", cleaner_version="v2",
        early_stop_num=2850,      # 50 * 57
        eos_drop=False,
        refer_bins=704,           # decode_encp crops :704 (idempotent)
        out_sr=32000,
    ),
    "v2Pro": dict(
        family="v1v2", s1_dir="s1v2", sovits_dir="v2pro", cleaner_version="v2",
        early_stop_num=2850, eos_drop=False, refer_bins=704, out_sr=32000,
        sv=True,
    ),
    "v2ProPlus": dict(
        family="v1v2", s1_dir="s1v2", sovits_dir="v2proplus", cleaner_version="v2",
        early_stop_num=2850, eos_drop=False, refer_bins=704, out_sr=32000,
        sv=True,
    ),
    "v3": dict(
        family="v3", s1_dir="s1", sovits_dir="v3", cleaner_version="v2",
        early_stop_num=2850, eos_drop=False, refer_bins=704, out_sr=24000,
        steps=32, cfg_rate=0.0,
    ),
    "v4": dict(
        family="v3", s1_dir="s1", sovits_dir="v4", cleaner_version="v2",
        early_stop_num=2850, eos_drop=False, refer_bins=704, out_sr=48000,
        steps=32, cfg_rate=0.0, vocoder_dir="v4_vocoder",
    ),
    "v5dev": dict(
        family="v5", s1_dir="s1", sovits_dir="v5dev", cleaner_version="v2",
        early_stop_num=2850, eos_drop=False, refer_bins=704, out_sr=48000,
        steps=32, cfg_rate=1.30, vocoder_dir="v5_vocoder",
    ),
    "v5turbo": dict(
        family="v5", s1_dir="s1", sovits_dir="v5turbo", cleaner_version="v2",
        early_stop_num=2850, eos_drop=False, refer_bins=704, out_sr=48000,
        steps=4, cfg_rate=0.0, vocoder_dir="v5_vocoder",
    ),
}

V1_LANGUAGES = ["auto", "en", "zh", "ja", "all_zh", "all_ja"]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", required=True, choices=sorted(VERSION_CONFIG),
                   help="Model family to run (v2Pro* = v2 + ERes2NetV2 sv).")
    p.add_argument("--model", default=None, choices=[None, "pro", "proplus", "dev", "turbo"],
                   help="Legacy variant flag from the e2e_v2pro/e2e_v5 wrapper days. "
                        "Accepted for compatibility; must agree with --version.")
    p.add_argument("--text", required=True, help="Target text to synthesize.")
    p.add_argument("--ref-audio", required=True,
                   help="Reference audio path (any sample rate; resampled internally).")
    p.add_argument("--ref-text", required=True,
                   help="Transcript of the reference audio.")
    p.add_argument("--models-root", default=DEFAULT_MODELS_ROOT,
                   help="Dir with bert/, hubert/, s1*/, <version>/, [*vocoder/], sv/ "
                        f"MLX exports (default: {DEFAULT_MODELS_ROOT})")
    p.add_argument("--cpufast-repo",
                   default=os.environ.get("GPT_SOVITS_CPUFAST", DEFAULT_CPUFAST_REPO),
                   help="GPT-SoVITS-CPUFast checkout, used read-only for the text front-end.")
    p.add_argument("--lang", default="zh",
                   help="Text language mode: zh/ja/ko/yue/en/all_zh/all_ja/"
                        "all_ko/all_yue/auto/auto_yue (default zh; mixed text under"
                        " a single-language label takes the label lang).")
    p.add_argument("--prompt-lang", default=None,
                   help="Language mode for the reference text (default: --lang).")
    p.add_argument("--text-split-method", default="cut0",
                   help="Official cut0..cut5 split method for the target text "
                        "(default cut0 = no splitting, anchor behavior).")
    p.add_argument("--ref-cache", default=os.environ.get("GSOVITS_REF_CACHE", ""),
                   help="Enable the (ref_audio, ref_text) disk cache at this dir "
                        "(or GSOVITS_REF_CACHE; empty = off).")
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--repetition-penalty", type=float, default=DEFAULT_REPETITION_PENALTY)
    p.add_argument("--early-stop-num", type=int, default=None,
                   help="AR frame cap (default: official per-version hz*max_sec).")
    p.add_argument("--noise-scale", type=float, default=DEFAULT_NOISE_SCALE,
                   help="SoVITS decode noise (v1/v2/v2Pro* only).")
    p.add_argument("--steps", type=int, default=None,
                   help="CFM Euler steps override (v3/v4/v5*; default per version).")
    p.add_argument("--cfg-rate", type=float, default=None,
                   help="CFM inference_cfg_rate override (v3/v4/v5*; default per version).")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help="AR sampling key seed (decode noise uses key(seed) too).")
    p.add_argument("--out", default=None, help="Output wav path (default out_<version>.wav).")
    p.add_argument("--gpu", action="store_true",
                   help="Opt in to Metal for ALL stages: requires GSOVITS_GPU_LOCK_OK=1 "
                        "and a fresh .tmp/gpu.lock.d owner (default-deny).")
    p.add_argument("--bench", action="store_true",
                   help="Print per-stage wall times and peak RSS to stderr.")
    p.add_argument("--frontend-only", action="store_true",
                   help="Run ONLY the text front-end (phones/bert JSON; audio stages "
                        "skipped). CPU-legal smoke mode.")
    p.add_argument("--cpu-i-know-broken", action="store_true",
                   help="Force a full pipeline run on CPU anyway (unsupported: "
                        "v3-family CFM yields all-zero audio; v1/v2 30-100x slow).")
    args = p.parse_args(argv)
    cfg = VERSION_CONFIG[args.version]
    # legacy wrapper flags: --model pro|proplus selects v2Pro/v2ProPlus,
    # --model dev|turbo selects v5dev/v5turbo (the old e2e_v2pro/e2e_v5
    # wrappers pinned the family default and let --model switch variants;
    # bench.py still passes these). Conflicts outside the family are errors.
    model_map = {"pro": "v2Pro", "proplus": "v2ProPlus",
                 "dev": "v5dev", "turbo": "v5turbo"}
    if args.model is not None:
        want = model_map[args.model]
        families = {VERSION_CONFIG[w]["family"] for w in model_map.values()}
        if VERSION_CONFIG[want]["family"] != cfg["family"] or args.version not in model_map.values():
            p.error(f"--model {args.model} implies --version {want} "
                    f"(got {args.version})")
        args.version = want
        cfg = VERSION_CONFIG[args.version]
    if args.early_stop_num is None:
        args.early_stop_num = cfg["early_stop_num"]
    if args.out is None:
        args.out = f"out_{args.version}.wav"
    # v1 has no ko/yue symbols — reject before the front-end wastes work
    if cfg["cleaner_version"] == "v1" and args.lang in ("ko", "yue", "all_ko", "all_yue"):
        p.error(f"--lang {args.lang} requires a v2 symbol table (v1 has no ko/yue)")
    return args


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
# Shared pipeline stages
# ---------------------------------------------------------------------------

def run_frontend(args, cfg, times, _device="cpu"):
    """Official text front-end (vendored CPUFast modules, MLX G2PW)."""
    from gsovits_mlx.text.preproc import TextFrontend, bootstrap
    from gsovits_mlx.text import ref_cache

    bootstrap(args.cpufast_repo, models_root=args.models_root)
    # G2PW's tokenizer.json comes from the official bert_path (model_source);
    # the vendored bert export carries it. Set BEFORE the lazy loader runs.
    if not os.environ.get("bert_path"):
        os.environ["bert_path"] = os.path.join(args.models_root, "bert")
    fe = TextFrontend(models_root=args.models_root, lazy=True,
                      device=os.environ.get("GSOVITS_FRONTEND_DEVICE", _device))
    prompt_lang = args.prompt_lang or args.lang
    cv = cfg["cleaner_version"]

    def frontend_run():
        cache_key = ref_cache.cache_key(
            args.ref_audio, args.ref_text, prompt_lang, cv) \
            if args.ref_cache else None
        if cache_key:
            got = ref_cache.get_cached(ref_cache.cache_dir(args.ref_cache), cache_key)
            if got:
                _, arrays = got
                return arrays["p_ids"], arrays["p_bert"], \
                    arrays["t_ids"], arrays["t_bert"], True
        p = fe.segment_prompt(args.ref_text, prompt_lang, cv)
        t = fe.preprocess(args.text, args.lang, args.text_split_method, cv)
        p_ids, p_bert = p.phones, p.bert
        t_ids = [ph for seg in t for ph in seg.phones]
        t_bert = np.concatenate([seg.bert for seg in t], axis=1)
        if cache_key:
            try:
                ref_cache.put_cached(
                    ref_cache.cache_dir(args.ref_cache), cache_key,
                    {"ref_text": args.ref_text, "lang": args.lang,
                     "prompt_lang": prompt_lang, "version": cv},
                    {"p_ids": np.asarray(p_ids, np.int32), "p_bert": p_bert,
                     "t_ids": np.asarray(t_ids, np.int32), "t_bert": t_bert})
            except OSError:
                pass
        return p_ids, p_bert, t_ids, t_bert, False

    p_ids, p_bert, t_ids, t_bert, cache_hit = frontend_run()
    if args.frontend_only:
        print(json.dumps({
            "prompt_phones": list(p_ids), "target_phones": list(t_ids),
            "prompt_bert_shape": list(np.asarray(p_bert).shape),
            "target_bert_shape": list(np.asarray(t_bert).shape),
            "cache_hit": bool(cache_hit), "frontend_s": times.get("frontend"),
        }), flush=True)
        print(f"[frontend-only] prompt {len(p_ids)} phones / target {len(t_ids)} "
              f"phones — audio stages skipped (CPU-legal smoke)", file=sys.stderr)
        sys.exit(0)

    all_phones = np.asarray(list(p_ids) + list(t_ids), np.int32)
    all_bert = np.concatenate([np.asarray(p_bert), np.asarray(t_bert)], axis=1)
    times["frontend_cache_hit"] = cache_hit
    return all_phones, all_bert, p_ids


def run_prompt_codes(args, cfg, times):
    """HuBERT -> SoVITS quantizer -> (1, Tp) semantic codes + sovits model."""
    import mlx.core as mx
    from gsovits_mlx.io import load_mlx_safetensors
    from gsovits_mlx.text.hubert import HubertModel

    # Official _set_prompt_semantic: 16 kHz wav + zero_wav = int(32000 * 0.3).
    wav16k = load_audio_official(args.ref_audio, 16000)
    wav16k = np.concatenate([wav16k, np.zeros(9600, np.float32)])
    hubert_dir = os.path.join(args.models_root, "hubert")
    hb = HubertModel(load_mlx_safetensors(os.path.join(hubert_dir, "hubert.safetensors")),
                     json.load(open(os.path.join(hubert_dir, "config.json"))))
    h = hb(mx.array(wav16k[None]))
    mx.eval(h)
    t_load = time.perf_counter()
    if cfg["family"] == "v1v2":
        from gsovits_mlx.pipeline import _load_sovits_v1v2
        sov, _meta = _load_sovits_v1v2(
            os.path.join(args.models_root, cfg["sovits_dir"]), args.version)
    else:
        from gsovits_mlx.pipeline import load_sovits_v3
        sov, _meta = load_sovits_v3(
            os.path.join(args.models_root, cfg["sovits_dir"]), args.version)
    times["model_load"] = time.perf_counter() - t_load
    prompt_sem = sov.extract_latent(mx.transpose(h, (0, 2, 1)))
    if cfg["family"] == "v1v2":
        prompt_sem = mx.array(prompt_sem.reshape(prompt_sem.shape[0], -1), mx.int32)
    mx.eval(prompt_sem)
    return sov, prompt_sem


def run_refer_spec(args, cfg, times, crop):
    """32 kHz reference spectrogram; v1 keeps all 1025 bins, others crop :704
    (decode_encp cropping makes the full spec idempotent, but v1 NEEDS the
    full spec — kept explicit per PARITY_NOTES)."""
    import mlx.core as mx
    from gsovits_mlx.text.mel_frontend import spectrogram

    w32 = load_audio_official(args.ref_audio, 32000)
    refer = spectrogram(mx.array(w32[None]), 2048, 640, 2048)
    if crop:
        refer = refer[:, :cfg["refer_bins"], :]  # freq bins are axis 1
    mx.eval(refer)
    return refer


def run_ar(args, cfg, times, all_phones_mx, all_bert_mx, prompt_sem):
    import mlx.core as mx
    from gsovits_mlx.pipeline import load_gpt

    t0 = time.perf_counter()
    gpt = load_gpt(os.path.join(args.models_root, cfg["s1_dir"]))
    seq = gpt.infer(all_phones_mx, all_bert_mx, prompt_sem,
                    top_k=args.top_k, top_p=args.top_p, temperature=args.temperature,
                    repetition_penalty=args.repetition_penalty,
                    early_stop_num=args.early_stop_num, key=mx.random.key(args.seed))
    mx.eval(seq)
    times["ar"] = time.perf_counter() - t0
    if cfg["eos_drop"]:
        seq_np = np.array(seq)
        if seq_np.shape[1] and int(seq_np[0, -1]) == 1024:
            seq = seq[:, :-1]
    return seq, seq.shape[1]


def decode_v1v2(args, cfg, times, sov, target_phones_mx, refer, prompt_sem, seq, sv_emb=None):
    import mlx.core as mx

    codes = seq[:, None, :]
    kwargs = dict(sv_emb_raw=sv_emb) if sv_emb is not None else {}
    audio, y_mask = sov.decode(codes, target_phones_mx, refer,
                               noise_scale=args.noise_scale,
                               key=mx.random.key(args.seed), **kwargs)
    mx.eval(audio)
    n_out = int(y_mask.shape[2] * 960)  # 2*480 samples per semantic frame
    return np.array(audio)[0, 0][:n_out]


def decode_v1v2_concat(args, cfg, times, sov, seg_codes, seg_phones, refer,
                       prompt_sem, sv_emb=None):
    """Time-concat parallel decode for v1/v2 (official 并行合成 method 2).

    All segments' semantic codes are concatenated along TIME into one
    sequence; phones likewise; ONE sov.decode call; audio sliced back per
    segment by frame counts (official: tokens*2*upsample; here y_mask
    segments x 960). Segment boundaries share conv receptive-field context
    with neighbors, so per-seg output is NOT bit-equal to independent
    decode — the official production path accepts this and ships it.
    Returns list of per-segment wav arrays (np.float32).
    """
    import mlx.core as mx
    import mlx.nn as nn

    T = sum(c.shape[2] for c in seg_codes)
    codes = mx.concatenate(seg_codes, axis=2)  # (1,1,T)
    phones = mx.concatenate(seg_phones, axis=1)
    kwargs = dict(sv_emb_raw=sv_emb) if sv_emb is not None else {}
    audio, y_mask = sov.decode(codes, phones, refer,
                               noise_scale=args.noise_scale,
                               key=mx.random.key(args.seed), **kwargs)
    mx.eval(audio)
    wav = np.array(audio)[0, 0]  # actual output; sample-per-frame varies
    n_total = len(wav)            # by version (v1 960, v2 640)
    # Slice back proportionally by token share (official slices by
    # tokens*2*upsample_rate; the y_mask frame accounting differs by
    # path so token-proportional is the robust equivalent).
    T_total = sum(c.shape[2] for c in seg_codes)
    out = []
    pos = 0
    for i, c in enumerate(seg_codes):
        if i == len(seg_codes) - 1:
            out.append(wav[pos:])
        else:
            n_samples = int(round(n_total * c.shape[2] / T_total))
            out.append(wav[pos : pos + n_samples])
            pos += n_samples
    return out


def decode_v3family(args, cfg, times, sov, all_phones_mx, refer, prompt_sem, seq, p_ids):
    """v3/v4/v5: codes -> decode_encp -> CFM (chunked / CFMV5) -> vocoder."""
    import mlx.core as mx
    from gsovits_mlx.text.mel_frontend import mel_spectrogram
    from gsovits_mlx.pipeline import denorm_spec, norm_spec
    from gsovits_mlx.sovits.cfm import resolve_sampling

    t0 = time.perf_counter()
    steps, cfg_rate = resolve_sampling(args.version, args.steps, args.cfg_rate)
    codes = seq[:, None, :]
    target_phones = all_phones_mx[:, len(p_ids):]
    refer_mask = mx.ones((refer.shape[0], 1, refer.shape[2]), dtype=refer.dtype)
    ge = sov.ref_enc(refer[:, :704] * refer_mask, refer_mask)
    fea_ref, ge = sov.decode_encp(mx.array([prompt_sem]), all_phones_mx[:, :len(p_ids)],
                                  refer=refer, ge=ge)
    fea_todo, _ = sov.decode_encp(codes, target_phones, refer=refer, ge=ge,
                                  speed=1.0)
    mx.eval(fea_ref, fea_todo)

    # prompt mel2: v3 = mel_fn 100-mel 1024/256 @ 24 kHz; v4/v5 = mel_fn_v4
    # 100-mel 1280/320 @ 32 kHz (both center=False), then norm_spec.
    if args.version == "v3":
        ref = load_audio_official(args.ref_audio, 24000)
        mel2 = mel_spectrogram(mx.array(ref[None]), 1024, 100, 24000,
                               256, 1024, 0, None, center=False)
    else:
        ref = load_audio_official(args.ref_audio, 32000)
        mel2 = mel_spectrogram(mx.array(ref[None]), 1280, 100, 32000,
                               320, 1280, 0, None, center=False)
    mel2 = norm_spec(mel2)
    mx.eval(mel2)

    key = mx.random.key(args.seed)
    if cfg["family"] == "v5":
        from gsovits_mlx.pipeline import load_vocoder_v4
        from gsovits_mlx.sovits.cfm import synthesize_v5_mel
        mel = synthesize_v5_mel(sov, fea_ref, fea_todo, mel2,
                                sample_steps=steps, cfg_rate=cfg_rate, key=key)
        pred = denorm_spec(mel)
        mx.eval(pred)
        voc_dtype = mx.float16 if os.environ.get("GSOVITS_V5_VOCODER_FP16", "1") != "0" else None
        voc = load_vocoder_v4(os.path.join(args.models_root, cfg["vocoder_dir"]),
                             dtype=voc_dtype)
    else:
        from gsovits_mlx.pipeline import cfm_chunked_decode_v3
        pred = cfm_chunked_decode_v3(sov, fea_ref, fea_todo, mel2,
                                     sample_steps=steps,
                                     inference_cfg_rate=cfg_rate, key=key)
        pred = denorm_spec(pred)
        mx.eval(pred)
        if args.version == "v3":
            from gsovits_mlx.pipeline import load_bigvgan
            voc = load_bigvgan(os.path.join(args.models_root, "bigvgan"))
        else:
            from gsovits_mlx.pipeline import load_vocoder_v4
            voc = load_vocoder_v4(os.path.join(args.models_root, cfg["vocoder_dir"]))
    audio = voc(pred) if args.version == "v3" else voc.infer(pred)
    mx.eval(audio)
    times["sovits_vocoder"] = time.perf_counter() - t0
    return np.array(audio)[0, 0], steps, cfg_rate


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = VERSION_CONFIG[args.version]

    # Process-wide device gate (default-deny): pins MLX to CPU unless
    # --gpu/GSOVITS_GPU_LOCK_OK AND a fresh gpu.lock. See gsovits_mlx.gpu_lock.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from gsovits_mlx.gpu_lock import resolve_device
    device = resolve_device(flag_gpu=args.gpu, verbose=True)
    if device == "gpu":
        try:
            import mlx.core as mx
            # Whole-process footprint cap (2026-10-08): MLX Metal buffer cache
            # grows unbounded by default and shows up in phys_footprint as the
            # 7-11GB end-of-run spike. Cap the cache; MLX evicts evictable
            # buffers instead of accumulating. Override: GSOVITS_METAL_LIMIT_MB.
            lim = int(os.environ.get("GSOVITS_METAL_LIMIT_MB", "3072"))
            mx.metal.set_memory_limit(lim * 1024 * 1024)
        except Exception as e:  # noqa: BLE001 - older mlx lacks the API
            print(f"[mem] metal memory limit not set: {e}", file=sys.stderr)
    if not args.frontend_only:
        from gsovits_mlx.gpu_lock import require_gpu_for_pipeline
        # gate on the RESOLVED device (env opt-in counts as gpu), not the
        # raw flag — resolve_device already validated lock freshness+tag.
        require_gpu_for_pipeline(device, cpu_ok_flag=args.cpu_i_know_broken)
    # AR sampling draws its inverse-CDF uniform from numpy's global RNG
    # (gsovits_mlx/gpt/t2s.py::_sample); seed it so runs are reproducible.
    np.random.seed(args.seed)
    args.out = os.path.abspath(args.out)
    t_all = time.perf_counter()
    times: dict[str, float] = {}

    import mlx.core as mx
    import soundfile as sf

    # resolve --ref-cache BEFORE bootstrap() chdirs into the CPUFast repo
    if args.ref_cache:
        args.ref_cache = os.path.abspath(args.ref_cache)

    # --- 1. text front-end (official semantics; zh G2PW on MLX) ---
    t0 = time.perf_counter()
    all_phones, all_bert, p_ids = run_frontend(args, cfg, times, _device=device)
    all_phones_mx = mx.array([all_phones.tolist()], mx.int32)
    all_bert_mx = mx.array(all_bert[None])  # (1, 1024, Tp+Tt)
    mx.eval(all_phones_mx, all_bert_mx)
    times["frontend"] = time.perf_counter() - t0
    if args.bench:
        print(f"[bench] frontend: phones={all_phones_mx.shape} "
              f"bert={all_bert_mx.shape} "
              f"{'CACHE_HIT ' if times.get('frontend_cache_hit') else ''}"
              f"{times['frontend']:.2f}s", file=sys.stderr)

    # --- 2. prompt semantic codes: HuBERT -> SoVITS quantizer ---
    t0 = time.perf_counter()
    sov, prompt_sem = run_prompt_codes(args, cfg, times)
    times["prompt_codes"] = time.perf_counter() - t0
    if args.bench:
        print(f"[bench] model_load: sovits {times['model_load']:.2f}s", file=sys.stderr)
        print(f"[bench] prompt_codes: {prompt_sem.shape} "
              f"{times['prompt_codes']:.2f}s", file=sys.stderr)

    # --- 3. reference spectrogram ---
    t0 = time.perf_counter()
    refer = run_refer_spec(args, cfg, times, crop=args.version != "v1")
    times["refer_spec"] = time.perf_counter() - t0
    if args.bench:
        print(f"[bench] refer_spec: {refer.shape} {times['refer_spec']:.2f}s",
              file=sys.stderr)

    # v2Pro/ProPlus: ERes2NetV2 speaker vector on the RAW 16 kHz ref audio
    sv_emb = None
    if cfg.get("sv"):
        t0 = time.perf_counter()
        from gsovits_mlx.pipeline import load_sv_encoder
        from gsovits_mlx.sovits.kaldi_fbank import fbank
        sv_model, _ = load_sv_encoder(os.path.join(args.models_root, "sv"))
        feat = fbank(load_audio_official(args.ref_audio, 16000))
        sv_emb = sv_model.forward3(mx.array(feat[None]))  # (1, 20480)
        mx.eval(sv_emb)
        times["sv_emb"] = time.perf_counter() - t0
        if args.bench:
            print(f"[bench] sv_emb: {sv_emb.shape} {times['sv_emb']:.2f}s",
                  file=sys.stderr)

    # --- 4. AR GPT ---
    t0 = time.perf_counter()
    seq, n_gen = run_ar(args, cfg, times, all_phones_mx, all_bert_mx, prompt_sem)
    if args.bench:
        print(f"[bench] ar: {n_gen} tokens {times['ar']:.2f}s "
              f"({n_gen / times['ar']:.1f} tok/s)", file=sys.stderr)

    # --- 5. SoVITS decode ---
    t0 = time.perf_counter()
    if cfg["family"] == "v1v2":
        audio_np = decode_v1v2(args, cfg, times, sov, all_phones_mx[:, len(p_ids):], refer,
                               prompt_sem, seq, sv_emb)
    else:
        audio_np, steps, cfg_rate = decode_v3family(
            args, cfg, times, sov, all_phones_mx, refer, prompt_sem, seq, p_ids)
    if "sovits" not in times:
        times["sovits"] = time.perf_counter() - t0

    from gsovits_mlx._audio_check import assert_audible
    assert_audible(audio_np, context=f"{args.out} pre-write")

    sf.write(args.out, audio_np, cfg["out_sr"])
    if args.bench:
        rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        stage = "sovits" if "sovits" in times and times["sovits"] else "sovits_vocoder"
        print(f"[bench] {stage}: {audio_np.shape} "
              f"{times.get('sovits') or times.get('sovits_vocoder'):.2f}s | "
              f"total {time.perf_counter() - t_all:.2f}s "
              f"peak_rss={rss_bytes // 1024} KB", file=sys.stderr)
    extra = ""
    if cfg["family"] in ("v3", "v5"):
        extra = f", {steps} CFM steps, cfg {cfg_rate}"
    elif cfg["family"] == "v3":
        extra = f", {args.steps} CFM steps"
    print(f"saved {args.out} ({len(audio_np) / cfg['out_sr']:.2f}s, {n_gen} AR tokens"
          f"{extra})")


if __name__ == "__main__":
    main()
