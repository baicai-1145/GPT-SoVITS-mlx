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

Each family has one end-to-end script under `tools/`:

```bash
python3 tools/e2e_v2.py \
    --text "你好，欢迎来到各自的旅程。今天我们聊聊机器学习。" \
    --ref-audio ref_zh_3.5s.wav --ref-text "希望你以后能够做得比我还好哟。" \
    --out out_v2.wav

python3 tools/e2e_v5.py --model turbo ... --out out_v5turbo.wav
python3 tools/e2e_v2pro.py --model proplus ... --out out_v2proplus.wav
```

`--models-root` points at the converted MLX exports (default
`/Volumes/2T/gpt-sovits-models/mlx`): `bert/ hubert/ s1*/ <version>/ [*vocoder/] sv/`.
The CPUFast checkout is used read-only for the text front-end
(`--cpufast-repo`). Benchmark any version with
`python3 tools/bench.py --version v4 --keep-audio out.wav`.

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
