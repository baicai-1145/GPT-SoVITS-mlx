# MLX ↔ GPT-SoVITS-CPUFast Parity Notes (v2)

Findings from the round of verification frozen at commit `a5066c7`
("fix: 5 critical bugs making v2 output garbage"). Everything here was
established by capturing the official CPUFast pipeline's intermediates on this
machine and diffing against the MLX port, stage by stage. Treat these as the
authoritative semantics for `gsovits_mlx` inference code and for
`tools/e2e_v2.py`.

## Verification assets (read-only, under `.tmp/`)

Do not modify or move; they are the regression baseline for future versions.

| Asset | Contents |
| --- | --- |
| `.tmp/official_ar2.npz` | Official CPUFast AR inputs + outputs: `x` (all_phones), `bert` (all_bert), `prompts` (prompt semantic codes), captured from the official `TTS` runner |
| `.tmp/official_logits1234.npz` / `.npy` | Per-step AR logits + sampled token stream for seed 1234 (official `infer_panel_batch_infer`) |
| `.tmp/official_inputs.npz` | `prompt_semantic`, `all_phones`, `all_bert`, `prompt_phones_len` from the official text preprocessor |
| `.tmp/seedstudy/` | 8×2 decodes: torch `infer_panel_naive` seeds 0–7 vs MLX keys 0–7 (same SoVITS decode path) |
| `.tmp/ab_inputs.npz` / `ab_decode.py` / `ab_mlx.wav` / `ab_torch.wav` | A/B decode harness (torch vs MLX SoVITS decode, o-diff 1.4e-3) |
| `.tmp/rng1.log`, `.tmp/smp.log` | Sampler dissection logs (single-uniform inverse-CDF vs torch Gumbel trace) |
| `.tmp/official_seeds.txt` | 6 official runs with random seeds → duration spread 4.5–5.3 s |

## Official CPUFast input construction quirks

These are all load-bearing; missing any one reproduces garbage audio.

1. **`pre_seg_text` "。" prefix applies to the TARGET text only.**
   The official preprocessor prepends `"。"` when the text does not start with
   a split separator and its first segment is shorter than 4 chars — but only
   for the target text, never the prompt text. Implemented in
   `tools/e2e_v2.py::phones_and_bert`.
2. **Prompt codes = PyAV-decoded 16 kHz wav + 9600 zeros.** `zero_wav =
   int(32000 * 0.3) = 9600` zeros are appended to the 16 kHz reference wav
   before HuBERT. Both halves matter, verified on the bench ref
   (`ref_zh_3.5s.wav` -> official `prompt_semantic`, 101 codes):
   - official PyAV decode (`tools/audio_utils.load_audio_mono`: `av.open` +
     `av.AudioResampler` fltp/swr) + 9600 zeros -> **101/101 codes match**;
   - soundfile + numpy linear resample + 9600 zeros -> only 75/101 match;
   - PyAV decode + 4800 zeros -> 94 frames, ~78/94 match (wrong pad length).
   The pad shifts frame alignment so quantizer codes match the official
   `prompt_semantic` bit-for-bit. `tools/e2e_v2.py` replicates the PyAV path
   (soundfile fallback emits a warning and may drift a few frames).
3. **`all_phones = prompt_phones + target_phones`** concatenated along time and
   fed as one (B, T) sequence to the AR model.
4. **`all_bert = concat(prompt_bert, target_bert)` along the time axis**
   (1, 1024, Tp+Tt); it must line up with `all_phones`. The AR text embedding
   adds `bert_proj(bert_feature)` to the phone embedding.
5. **Reference spectrogram**: official v2 resamples the ref audio to 32 kHz and
   computes `spectrogram(n_fft=2048, hop=640, win=2048)`, cropped to the first
   **704 bins**. Bigger crops break the speaker-encoder path.

## AR stop rule

Generation stops when **either** condition hits after a sampling step:

- `argmax(logits) == EOS` (1024), or
- the sampled token equals EOS.

Additionally, EOS logits are **masked for the first 10 generated tokens**
(`logits[:, :-1]` while `idx < 11`), matching torch. If EOS fires while the
generated sequence is still empty, a zero token is appended so the decode sees
a non-empty code stream. The early-stop cap (v2: 2850 frames, v1: 1500) applies
only after the `y.shape[1] == prefix_len` guard.

## Sampler lessons (torch-comparable sampling)

- The official torch sampler adds **one `torch.rand_like(logits)` noise tensor
  per row** (batch=1 → one uniform) and samples via inverse-CDF on the CDF of
  the top-k-restricted, temperature-scaled distribution. **Drawing an
  independent uniform per column is wrong** — that was the source of the
  original garbage output. In MLX: one uniform of shape (batch, vocab) per
  step (inverse-CDF over the row).
- **Split the RNG key every step** (`key, step_key = mx.random.split(key)`),
  never reuse a key across steps.
- With the same seed the MLX sampler does **not** reproduce the torch token
  stream (Gumbel vs inverse-CDF consume randomness differently — see
  `.tmp/rng1.log`); the distributions agree (`p(token)` within float noise;
  `.tmp/smp.log`) and the duration/waveform statistics of the two decoders
  agree across 8 seeds (`.tmp/seedstudy/`).
- The verified-parity path is: **capture official tokens once, then decode via
  MLX SoVITS** (`.tmp/officialT_genonly.wav` etc.). For free sampling, seed
  only controls MLX-side RNG (key = seed).

## CPUFast standalone trap

`Text2SemanticDecoder` (torch, `AR/models/t2s_model.py`) loads weights with
`load_state_dict`, but **`t2s_transformer` is built in `__init__` from random
init and is only rebound to the loaded weights by `rebuild_t2s_transformer()`**.
After `load_state_dict` you MUST call `rebuild_t2s_transformer()`, otherwise
the transformer runs with the random-weight copy while attention/proj weights
look "loaded". This produced a phantom mismatch during the round; the fix is
one line: `m.load_state_dict(sd); m.rebuild_t2s_transformer(); m.eval()`.

## Official decode consumes gen-only codes; prompt+gen full is equivalent

The official SoVITS decode path slices `item[-idx:]` — **generated codes
only**, prompt codes dropped. Feeding `prompt + gen` in full is **equivalent**
(audibly and numerically): the model re-states the prompt (parroting the
reference clip) before the target text, exactly like the official output.
`tools/e2e_v2.py` concatenates `prompt_sem + seq` and decodes the whole
stream. Verified by `.tmp/officialT_genonly.wav` vs `.tmp/out_v2_torchfull.wav`.

## Verified numbers (this machine)

- Prompt semantic codes (PyAV + 9600 zeros) vs official `prompt_semantic`:
  **101/101 exact**.
- Text front-end (`pre_seg_text` + cleaner) vs official `all_phones`:
  **75/75 exact** (29 prompt + 46 target).
- SoVITS decode A/B, same codes/phones/refer: MLX vs torch **o-diff 1.4e-3**
  (float32, noise key identical).
- Spectrogram front-end vs librosa slaney mel / hann STFT: parity 1e-7.
- AR logits replay vs captured official logits (seed 1234, `.tmp/replay_cmp.py`):
  while following the official token path, per-step max diff stays small and
  the greedy prefix is **token-exact vs torch for the first 20 steps**;
  incremental-vs-full attention logits differ by ≤ 0.0027 over the 116-step
  official replay (both recorded in commit `a5066c7`).
- Official 6 random seeds produce 4.5–5.3 s of audio for the same text —
  sampling variance, not a bug (`.tmp/official_seeds.txt`).
