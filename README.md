# GPT-SoVITS-mlx

GPT-SoVITS text-to-speech running end-to-end on Apple **MLX** (Metal GPU) —
no torch at inference time. Supports all eight official model families:

| family | weights | output | decode path |
|---|---|---|---|
| v1 | s2G488k | 32 kHz | HiFi-GAN in SynthesizerTrn |
| v2 | s2G2333k | 32 kHz | HiFi-GAN in SynthesizerTrn |
| v2Pro / v2ProPlus | s2Gv2Pro(Plus) + ERes2NetV2 sv | 32 kHz | v2 + speaker-vector condition |
| v3 | s2Gv3 + BigVGAN vocoder | 24 kHz | DiT CFM (32-step Euler) rolling chunks |
| v4 | s2Gv4 + Generator vocoder | 48 kHz | DiT CFM + 100-mel Generator |
| v5dev / v5turbo | s2Gv5dev/turbo + Generator | 48 kHz | CFMV5 static-cache DiT (32/4 steps) |

Verified against the official GPT-SoVITS-CPUFast implementation stage-by-stage;
per-version parity numbers and quirks live in
[docs/PARITY_NOTES.md](docs/PARITY_NOTES.md), performance in
[docs/BENCH.md](docs/BENCH.md).

## Setup

Requires Python 3.12+ and an Apple-silicon Mac. With [uv](https://docs.astral.sh/uv/):

```bash
uv sync                      # runtime deps: mlx, numpy, tokenizers, soundfile
uv sync --extra convert      # + torch (ONLY for weight conversion)
uv sync --extra dev          # + pytest
```

## Usage

One CLI for all versions:

```bash
python3 tools/e2e.py --version v2 \
    --text "你好，欢迎来到各自的旅程。今天我们聊聊机器学习。" \
    --ref-audio ref_zh_3.5s.wav --ref-text "希望你以后能够做得比我还好哟。" \
    --out out_v2.wav

# v1 | v2 | v2Pro | v2ProPlus | v3 | v4 | v5dev | v5turbo
python3 tools/e2e.py --version v5turbo ...
python3 tools/e2e.py --version v2ProPlus ...
```

The old per-version scripts (`tools/e2e_v2.py` etc.) remain as thin
wrappers that exec `e2e.py` with the version pinned — identical behavior
and byte-identical outputs; `tools/bench.py` and existing docs keep
working through them. Useful flags: `--ref-cache DIR` (prompt cache,
<50 ms repeat runs), `--frontend-only` (CPU-legal front-end smoke),
`--seed N` (see Reproducibility), `--bench` (per-stage times).
Benchmark any version with
`python3 tools/bench.py --version v4 --keep-audio out.wav`.

`--models-root` points at the converted MLX exports (default
`/Volumes/2T/gpt-sovits-models/mlx`): `bert/ hubert/ s1*/ <version>/
[*vocoder/] sv/`. The CPUFast checkout is used read-only for the text
front-end (`--cpufast-repo`).

### Text front-end and languages

The front-end (language segmentation → G2P → BERT features) is the official
GPT-SoVITS-CPUFast logic, executed from the CPUFast checkout by a vendoring
loader (`gsovits_mlx/text/vendored_cpufront.py`) that substitutes three
torch-dependent pieces with MLX/numpy implementations (LangSegmenter,
G2PW, TTS_infer_pack stub). The loader exists so the runtime needs **no
torch at all**; running the official package directly is not a supported
mode (it requires torch).

Supported languages / modes:

| mode | v1 | v2 | notes |
|---|---|---|---|
| `zh` `ja` `en` | ✓ | ✓ | single-language label; mixed text takes the label language |
| `ko` `yue` | — | ✓ | v2-only (v1 symbol tables lack them; requesting them under v1 raises) |
| `all_zh` `all_ja` `all_ko` `all_yue` | v1: zh/ja | ✓ | force one language, segment with LangSegmenter |
| `auto` `auto_yue` | v1: auto | ✓ | per-sentence language detection (`auto_yue` renders detected zh as yue) |
| `en` | ✓ | ✓ | literal text, no segmentation |

Reference text language: `--prompt-lang`; target splitting:
`--text-split-method cut0..cut5` (default `cut0`). Non-zh segments get
all-zero BERT features (official `get_bert_inf` semantics).

Vendored-stack contract (env vars):

- `GPT_SOVITS_CPUFAST` / `--cpufast-repo`: path of the CPUFast checkout,
  used READ-ONLY. The loader executes `GPT_SoVITS/text/**` and
  `TTS_infer_pack/text_segmentation_method.py` in place and expects the
  checkout's data files to be intact:
  - `pretrained_models/fast_langdetect/lid.176.bin` (fastText LID model;
    download from the official fastText site if missing),
  - ja/korean scratch dirs are created inside the checkout on first use
    (the process chdirs into the repo for the front-end).
- `GSOVITS_G2PW_SAFETENSORS`: override for the MLX G2PW weights (default
  lookup: `<CPUFast>/GPT_SoVITS/text/G2PWModel/g2pw.safetensors`, then
  `<models-root>/g2pw/g2pw.safetensors`). One-off conversion from the
  official `g2pw.pth` via `tools/export_g2pw_mlx.py` (fp32 — argmax must be
  token-exact; see `docs/PARITY_NOTES.md`).
- `GSOVITS_REF_CACHE`: default dir for `--ref-cache` (see below).
- `GSOVITS_FRONTEND_DEVICE`: `cpu` (default) or `gpu` for the front-end's
  BERT/G2PW forwards.

Pass `--ref-cache DIR` (or set `GSOVITS_REF_CACHE`) to cache the prompt
front-end + HuBERT artifacts keyed by (reference audio mtime+size, text,
lang, version, env fingerprint) — repeat synthesis with the same reference
skips the front-end entirely (<50 ms hit). Cached entries are
generation-specific: keys include the (python, mlx, numpy) versions, so a
version pin change never reuses stale artifacts.

The `[frontend]` extra (`uv sync --extra frontend`) installs the official
G2P stack (jieba-fast, pypinyin, jamo, ko-pron, ToJyutping, split-lang,
g2p-en, g2pk2, cn2an, pyopenjtalk, nltk, python-mecab-ko,
fast-langdetect). Core runtime deps stay `mlx numpy tokenizers soundfile`;
without the extra, pure-zh inference still works via the vendored stack
where the deps it needs are present (install the extra for multi-language).

### GPU lock (multi-agent machines)

All entry scripts default to the **CPU** MLX device; Metal requires an
explicit opt-in (`--gpu` or `GSOVITS_GPU_LOCK_OK=1`) **and** a fresh
`.tmp/gpu.lock.d/owner` file (<=45 min old; set `GSOVITS_RUN_TAG` to require
your tag in it). See `gsovits_mlx/gpu_lock.py`. This keeps GPU work serial
when several agents share the machine.

### Reproducibility (anchor contract)

Seeded synthesis is bit-exact **within** the anchor environment:

> python 3.12 · mlx==0.32.2 · numpy==2.5.2 · device

(the quadruple recorded in `docs/BENCH.md`). `--seed 0` seeds numpy's
global RNG, which the AR inverse-CDF sampler draws from; equal seeds give
identical wav bytes when the quadruple matches.

Empirically (phase-2 finding): changing mlx 0.32.2→0.32.3 **or** numpy
2.5.2→2.5.3 flips AR sampling near-ties — token counts and wavs diverge
(e.g. the v2 bench cell renders 119 tokens on the contract env but 112 on
mlx 0.32.3). The version pins in `pyproject.toml`/`uv.lock` are therefore
load-bearing, not cosmetic.

If you must move a pin: re-render the seed-0 anchors for all 8 versions
(`tools/bench.py --version X --keep-audio ... --seed 0`), update
`docs/BENCH.md`'s contract line and the regression-gate paths, and expect
token-count changes (they are expected behavior, not bugs). Front-end
phones are stable across these versions; only AR sampling flips. The
ref-prompt cache keys already include the interpreter fingerprint, so the
cache cannot cross a pin change.

## Converting weights

`tools/convert_sovits.py` (needs `--extra convert`) turns official
checkpoints into MLX safetensors: GPT (s1) via `convert_s1`, SoVITS via
`convert_sovits_v1v2` / `convert_sovits_v3v5`, vocoders via
`convert_generator_vocoder` / `convert_bigvgan`, and the v2Pro speaker
encoder via `convert_sv_encoder`. v5 ckpts ship with corrupted headers
(version bytes over `PK`) and the v2Pro sv ckpt stores fp16 tensors — both
handled by the converter.

## Architecture

```
text -> cleaner/phones -> BERT features ┐
ref audio -> HuBERT -> SoVITS quantizer ┴-> GPT AR (semantic tokens)
                                              │
ref audio -> spectrogram -> ref_enc (ge) ┴-> SoVITS decode -> waveform
                              (v2Pro: + ERes2NetV2 sv_emb)
v3+: SoVITS decode -> DiT CFM (rolling chunks) -> vocoder
```

MLX ports live in `gsovits_mlx/`: `pipeline.py` (loaders), `text/` (BERT,
HuBERT, kaldi fbank, mel front-ends), `sovits/` (v1/v2 SynthesizerTrn,
v3/v4/v5 CFM + DiT with static KV cache, ERes2NetV2 sv encoder, BigVGAN,
Generator). Numeric-sensitivity hot spots (LayerNorm, softmax, rope, BN) run
fp32; the rest follows the checkpoint dtype.

## Performance

See [docs/BENCH.md](docs/BENCH.md). Highlights on M4 (16 GB): v2 synthesizes
8.8 s audio in ~25 s end-to-end (RTF 2.8, dominated by the CPU front-end);
v5turbo renders in 53 s with 4-step CFM; peak RSS stays ≤ 4.7 GB for every
version (budget 12 GB).
