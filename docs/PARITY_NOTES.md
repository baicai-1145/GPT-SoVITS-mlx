# MLX ↔ GPT-SoVITS-CPUFast Parity Notes (v2)

Findings from the round of verification frozen at commit `a5066c7`
("fix: 5 critical bugs making v2 output garbage"). Everything here was
established by capturing the official CPUFast pipeline's intermediates on this
machine and diffing against the MLX port, stage by stage. Treat these as the
authoritative semantics for `gsovits_mlx` inference code and for
`tools/e2e_v2.py`.

## v1 addendum (task-4, e2e_v1.py)

The v1 port (`tools/e2e_v1.py`, weights s1bert25hz-2kh + s2G488k) reuses the
whole verified v2 pipeline. Version-specific findings:

1. **v1 keeps the FULL 1025-bin refer spectrogram.** Only v2 crops `:704`:
   `SynthesizerTrn.__init__` builds `MelStyleEncoder(spec_channels=1025)` for
   v1 vs `MelStyleEncoder(704)` for v2 (module/models.py), and v1's `get_ge`
   passes `y` uncropped. `tools/e2e_v1.py` computes the refer spec without the
   crop; `SynthesizerTrn.get_ge` already branched on `version == "v1"`.
2. **MelStyleEncoder attention temperature is `sqrt(d_model)`, not
   `sqrt(d_k)`.** torch `MultiHeadAttention` constructs
   `ScaledDotProductAttention(temperature=np.power(d_model, 0.5))`
   (module/modules.py) — nonstandard. The MLX `ScaledMultiHeadAttention`
   (gsovits_mlx/sovits/mrte.py) originally used the textbook `1/sqrt(d_k)`, a
   √2 scale error on ge. Fixed in this round: ge max-abs error on the bench ref
   went 0.256 → 9.5e-7 (v1) / 0.287 → 6.0e-7 (v2). The v2 numbers in the
   verified list below were measured **before** this fix; the fix only improves
   them (dec A/B o-diff 1.4e-3 was measured with ge from the torch side, so it
   is unaffected).
3. **v1 AR early stop = `hz * max_sec` from the s1 ckpt config.** Official
   TTS.py passes `self.configs.hz * self.configs.max_sec`; for the v1 pretrained
   s1bert25hz-2kh (`max_sec: 54`) that is 50×54 = **2700** (v2 5kh ckpt:
   max_sec 57 → 2850). The AR decode-loop hard cap is `MAX_AR_DECODE_STEPS =
   1500` regardless of version (AR/models/t2s_model.py) — an internal batched
   implementation detail, not a per-version audio-length cap.
4. **The official AR drops the sampled EOS before SoVITS decode**
   (`y_list[batch_index] = y_buffer[i, : curr_y_len - 1]`). The v1 pretrained
   sampling streams do hit EOS (seed 0 runs: 114–164 generated tokens); a
   trailing 1024 is also out of range for the 1024-entry codebook (torch
   `F.embedding` would throw; MLX gather silently clamps). `tools/e2e_v1.py`
   strips one trailing EOS token after `gpt.infer`.
5. **s2G488k.pth pickles a top-level `utils.HParams`.** Its `config` resolves
   only inside the official repo (`GPT_SoVITS/utils.py`).
   `tools/convert_sovits.py::_load_sovits_ckpt` now registers a same-named
   attribute-container shim when `utils` is not importable (the converter's
   `hps.get(...)` interface is what matters; all v1 hps are present in the
   ckpt). `convert_sovits_v1v2` needed **no** structural changes: s2G488k is
   wn-fused-free (raw `weight` on ups/resblocks), keeps `flow.flows.N.pre/post`
   at flow level (v1-style), and has a bias-free `dec.conv_post` (converted as
   zero bias).
6. **s1 2kh has plain (non-adaptive) LayerNorm** — `convert_s1`'s existing
   `norm1.weight/bias` branch covers it; `n_layer: 24`, `max_sec: 54` land in
   `gpt.json` verbatim.
7. **v1 v2 sampler defaults**: same top_k=15 / top_p=1.0 / temperature=1.0 /
   repetition_penalty=1.35 / noise_scale=0.5 as v2 (CPUFast has no per-version
   override; `inference.top_k: 5` in the ckpt config is a training-legacy
   value the runtime ignores).

Verified v1 numbers (this machine, `.tmp/parity_v1.py`, torch fp16-rounded
weights vs MLX fp16 exports, identical inputs + identical injected decode
noise): ge 2.2e-7 rel / 9.5e-7 abs, enc_p.x 4.3e-7, enc_p.m_p 4.0e-7,
enc_p.logs_p 1.3e-6, flow.z 5.8e-5 rel, dec.conv_pre 7.1e-5 rel, dec.o
0.0132 max-abs (<2e-2 gates), y_mask exact; e2e wav listens clean (prompt
parroting + target text). Prompt codes 101 frames, phones 29+46 = 75.

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

## v3 addendum (task-5, e2e_v3.py — CFM/DiT + BigVGAN 24 kHz)

The v3 port (`tools/e2e_v3.py`, weights s1v3.ckpt + s2Gv3.pth +
nvidia/bigvgan_v2_24khz_100band_256x) follows TTS.py `using_vocoder_synthesis`.
Findings, all stage-verified against official torch on this machine:

### Two DiT bugs fixed in `gsovits_mlx/sovits/dit.py`

1. **Rope cache double-encoding.** The cache stored `cos|sin`-concatenated
   values and `apply_rope` then took `cos()/sin()` of them AGAIN, corrupting
   every attention. Fix: cache RAW angles `(T, dim_head/2)`
   (`inv_freq = 10000**-arange(0,d,2)/d`, `freqs = t[:,None]*inv_freq`),
   and derive interleaved cos/sin via `repeat(freqs, 2, -1)` inside
   `apply_rope`. `rope_rotate_half` swaps interleaved pairs `x[2i] <-> x[2i+1]`
   (x-transformers packs freqs as `stack((f,f),-1).reshape`).
2. **Partial RoPE is head-0-ONLY (f5-tts quirk, easy to misread).** CPUFast's
   `AttnProcessor` calls `apply_rotary_pos_emb(query, freqs, ...)` on the
   FLAT `(B, T, 1024)` projection BEFORE the head split; `rot_dim =
   freqs.shape[-1] = 64`, so only the first 64 channels rotate — i.e. after
   `view(B,T,16,64)` **only head 0 gets RoPE, heads 1–15 are unrotated**.
   The MLX port originally applied RoPE per-head (all heads rotated), which
   quietly passes a naive same-input unit test on `(B,H,T,64)` but diverges
   7.7 abs at the attention output. Fix: rotate the head-0 slice only.
   Verified vs fp64 ground truth: MLX 1.4e-6 rel; torch fp32 itself carries
   1.2e-3 rel error at this op.

### `_nearest_interp` scale_factor semantics (`models_v1v2.py`)

`torch F.interpolate(mode='nearest')` has TWO index maps: with `size`,
`src = floor(dst * t_in / size)`; with `scale_factor`, `src = floor(dst /
scale_factor)`. They differ at 12 positions for the v3 1.875× upsample
(60→112/112→210), producing 3.6e-2 abs drift in `decode_encp` fea. Fix:
`_nearest_interp(x, size, scale_factor=...)` uses the division form; the v3
call site passes `scale_factor=1.875`. Integer scales (the 2× path) are
identical in both forms. After the fix fea diff is 3.4e-6.

### CFM / Euler / CFG parity (harness: `.tmp/torch_cfm_ref.py` +
`.tmp/mlx_euler_traj_check.py`)

- `decode_encp` fea: 3.4e-6 abs; ge: 2.7e-5 (fp16-converted weights).
- Single DiT step v_pos: 2.5e-5, v_neg: 4.0e-5 (post head-0-rope fix).
- Full 32-step Euler with cfg_rate=0.25, per-step trajectory diff:
  **1.16e-5 max** (fp32 noise floor for this stack).
- Sampling: `resolve_sampling('v3')` → `(sample_steps=32, cfg_rate=0.0)`
  (cfg only becomes 1.30 for v5 family; 0.25 appears only if the caller
  overrides). `x[..., :prompt_len] = 0` after every Euler step; chunk loop
  T_ref=468 / T_chunk=934 replicates `using_vocoder_synthesis`.

### BigVGAN v2 vocoder (`gsovits_mlx/vocoder/bigvgan.py`,
`tools/convert_sovits.py:convert_bigvgan`)

- Weight-norm convs pre-fused at conversion: `w = g * v/||v||_per_out_channel`
  — bit-equivalent to the official `remove_weight_norm()` effective weight.
  **Keep the vocoder fp32**: fp16 rounding of the fused convs amplifies
  through the 256× stack (A/B max-abs 1.1e-2 vs 5e-5 fp32 on a random mel).
- kaiser-sinc resample filters (Activation1d up/down) are buffers recomputed
  at runtime; verified 8.9e-8 vs checkpoint buffers. A previous `_sinc`
  sign-hack produced garbage (NaNs) — fixed to `sin(πx)/(πx)`, 1 at x=0.
- Torch-vs-MLX A/B on random 100-mel input: **max 5e-5, mean 8.3e-6** (fp32).
- Output sr is 24000 (not the SoVITS 32000); mel_fn for the prompt is
  100-mel n_fft 1024 / hop 256 @ 24 kHz center=False; spec_min=-12,
  spec_max=2 normalisation as in TTS.py.

### Conversion artifacts

- `/Volumes/2T/gpt-sovits-models/mlx/v3/` — pre-existing export verified
  against s2Gv3.pth on spot keys (0 diff on fp32 rows; it was already fp32).
- `/Volumes/2T/gpt-sovits-models/mlx/bigvgan/` — new
  `bigvgan.safetensors` (449 arrays, fp32) + `bigvgan.json`.
- `mlx/s1/gpt.safetensors` is the s1v3.ckpt conversion (ar_predict_layer
  diff 0; max_sec 57 → early_stop_num = 50*57 = 2850).

### e2e result

`tools/e2e_v3.py` → `/Volumes/2T/gpt-sovits-models/bench/mlx/v3.wav`
(8.80 s, 24 kHz, 119 AR tokens, 32 CFM steps, ~220 s wall, peak RSS 2.5 GB).
Listen check: prompt parroting + target text clearly intelligible, no
artifacts.

## v4 addendum (task-6, e2e_v4.py — CFM + Generator vocoder 48 kHz)

The v4 port (`tools/e2e_v4.py`, weights gsv-v4-pretrained/s2Gv4.pth +
vocoder.pth) reuses the whole v3 CFM machinery. Version-specific findings:

1. **ckpt `config["model"]` has NO `version` key.** The torch reference
   harness must inject `version="v4"` explicitly — `SynthesizerTrnV3.__init__`
   defaults to `"v3"`, which silently switches decode_encp to the 3.875×
   interp (116 frames for 30 codes) instead of the v4 integer 4× (120
   frames). Official CPUFast sets `hps["model"]["version"] = model_version`
   (TTS.py:579-581). The MLX loader takes the version explicitly.
2. **Interp scale 2 (integer).** `_nearest_interp` without scale_factor is
   exact here (both index maps agree for integer scales).
3. **Chunk constants**: `vocoder_configs` v4 = sr 48000, T_ref=500,
   T_chunk=1000, upsample_rate=480, overlapped_len=12
   (`cfm_chunked_decode_v4`; v3 helper keeps 468/934).
4. **Prompt mel is mel_fn_v4**: 100-mel, n_fft 1280 / hop 320 @ 32 kHz
   center=False (v3 was 1024/256 @ 24 kHz). The ref audio for mel2 is
   resampled to 32 kHz (not 24 kHz) per TTS.py tgt_sr.
5. **Vocoder is the HiFi-GAN `Generator`** (module/models.py:464,
   initial_channel=100, upsample [10,6,2,2,2], kernels [20,12,4,4,4], 512ch,
   LeakyReLU 0.1, final tanh) — same class already verified for v1/v2
   decoders. The vocoder.pth stores PLAIN conv weights (official calls
   `remove_weight_norm()` before capture), so `convert_generator_vocoder`
   does no weight-norm fusion. Torch-vs-MLX A/B on a random 100-mel input:
   **max 6.6e-5, mean 3.3e-6**.
6. **DiT keeps `use_step_embedding=True` for v4** (only v5 family drops it);
   all task-5 DiT fixes (raw-angle rope cache, head-0-only partial RoPE)
   apply unchanged.

Parity summary (official torch CPUFast, this machine): ge 4.6e-5,
decode_encp fea 8.3e-6, single DiT step v_pos 4.1e-5 / v_neg 7.4e-5,
32-step Euler+CFG(0.25) per-step trajectory **2.0e-5 max**.
`tools/e2e_v4.py` → `/Volumes/2T/gpt-sovits-models/bench/mlx/v4.wav`
(9.52 s, 48 kHz, 137 AR tokens, 32 CFM steps, ~64 s wall, peak RSS 2.7 GB);
listen check: prompt parroting + target text intelligible, no artifacts.

## v5 addendum (task-7, e2e_v5.py — v5dev/v5turbo, CFMV5 rolling chunks)

The v5 ports (`tools/e2e_v5.py --model dev|turbo`, weights
gsv-v5-pretrained/{s2Gv5dev,s2Gv5turbo,vocoder}.pth) reuse the v4 DiT/interp
machinery with the v5 CFM. Findings:

1. **Corrupt-header ckpts.** Both s2Gv5 ckpts are torch zip saves whose first
   two bytes were overwritten with the version tag (`07`/`08` instead of
   `PK` — official `my_save2`). Repair = stream-copy with a `PK` header
   (`.tmp/fix_v5_header.py`). Note: `shutil.copyfile` on this host hung in
   kernel IO for these files; use explicit read/write loops.
2. **No d_embed in the v5 family.** `use_step_embedding=False` (models.py
   builds DiT with `use_step_embedding = version not in V5_VERSIONS`); the
   ckpts confirm (no `d_embed.*` keys) despite the task brief saying
   otherwise. `t` is time_embed output alone; dt_cache never used.
3. **CFMV5 static conditioner cache** (precomputed text_embed + input
   projection + rope): the MLX DiT `prepare_static_cache` consumes
   `condition` as (B, C, T) — `CFMV5.inference` receives `mu` as (B, T, C)
   and passes `mu.transpose(2,1)`. Getting this layout wrong silently
   broadcasts the sinus pos-embed wrong and crashes at concat.
4. **Rolling chunk decode** (`synthesize_v5_mel`): reference 500 frames,
   chunk = min(1000-ref, 640), 32-frame rolling tail for prompt
   reconstruction after the first chunk, CFG 1.30 (v5dev, 32 steps) /
   0.0 (v5turbo, 4 steps) via `resolve_sampling`; noise temperature 0.875;
   steps validated to {4,8,16,32}.
5. **fp16 source weights** (info: "fp16 converted from …; aligned to
   s2Gv4.pth keys") are upcast to fp32 at conversion like the other ports.

Parity (official torch CPUFast, this machine, v5dev):
- ge 5.3e-5; decode_encp fea 5.4e-6; single DiT step (static cache)
  v_pos 4.2e-5 / v_neg 3.8e-5.
- Chunk-0 full 32-step Euler+CFG(1.30): **3.0e-5 max, 3.3e-6 mean**
  (torch ref seeds the CFM noise; injected into the MLX loop for
  trajectory comparison).
- vocoder is the same Generator as v4 (identical A/B 6.6e-5).

e2e: `bench/mlx/v5dev.wav` (8.84 s, 120 AR tokens, cfg 1.3) and
`bench/mlx/v5turbo.wav` (8.40 s, 109 AR tokens, 4 steps, cfg 0.0), both
48 kHz, listen-verified intelligible with no chunk-boundary artifacts.
Wall: ~88 s (dev) / ~59 s (turbo), peak RSS 2.3–4.0 GB.

## v2Pro/v2ProPlus addendum (task-8, e2e_v2pro.py — v2 + sv speaker vector)

The v2Pro ports (`tools/e2e_v2pro.py --model pro|proplus`, weights
pretrained_models/v2Pro/{s2Gv2Pro,s2Gv2ProPlus}.pth + sv/pretrained_eres2netv2w24s4ep4.ckpt)
reuse the v2 direct-waveform decode with a speaker-vector condition. Findings:

1. **sv branch.** The 16 kHz reference wav (RAW, no zero_wav padding) goes through a
   numpy port of torchaudio's kaldi fbank (80 bins, povey window^0.85, snip_edges,
   preemphasis 0.97, power spectrum, log with float-eps floor; `kaldi_fbank.py`)
   -> ERes2NetV2 forward3 -> (1, 20480). `SynthesizerTrn.get_ge` adds
   `sv_emb(Linear 20480->gin)` to the ref_enc ge, applies PReLU, then `ge_to512`
   (gin->512) feeds enc_p/MRTE while the 1024-dim ge drives flow + dec.
2. **sv ckpt convs carry biases** (official fusion.py AFF uses bias=True); the
   first converter/loader draft silently dropped them -> systematic ~1.8e-2
   per-stage offsets and a 0.3 embedding error. With biases loaded, MLX
   forward3 matches torch fp32 to **5.7e-6** (full path incl. fbank: 2.7e-5).
   Debugging gotcha: the ckpt stores fp16 tensors (like the v5 ckpts).
3. **PReLU broadcast bug.** `nn.prelu(x, w)` with x (B,C,1) and w (C,) broadcasts
   to (B,C,C) in MLX (aligns trailing dims). prelu_weight must be stored
   (1, C, 1).
4. **v2Pro gin_channels is 1024** (sv_emb out, ge, flow/dec condition); enc_p
   still consumes the 512-dim ge512 — mirrors models.py `ge512 if is_v2pro else ge`.

Parity (torch CPUFast fp32 reference on this machine, ref_zh_3.5s):
- kaldi fbank vs torchaudio: 4.1e-4 max.
- ERes2NetV2 forward3 (torch-fbank input): 5.7e-6 max, 4.1e-7 mean.
- full MLX sv path (numpy fbank + net): 2.7e-5 max.
- SoVITS decode path unchanged from the verified v2 port (ge shape 1024).

e2e: `bench/mlx/v2pro.wav` (8.84 s, 120 AR tokens) and
`bench/mlx/v2proplus.wav` (8.76 s, 118 AR tokens), both 32 kHz, listen-verified
intelligible with clean speaker timbre and no artifacts. Wall ~101 s / ~112 s,
peak RSS 2.2–3.8 GB.

## Phase-2 addendum (front-end unification, env contract, device gate)

Covers main @c12db19: the official multi-language front-end (task-2 +
follow-up), the unified `tools/e2e.py` CLI (task-6), the device gate
hardening, and the env-fingerprint ref cache. The per-version parity
methodology below is the one that made the byte-identity gate pass on all
8 versions in one window.

### Front-end parity methodology (phase-2)

The official front-end is NOT re-implemented — it is executed. `gsovits_mlx/
text/vendored_cpufront.py` execs the CPUFast checkout's `GPT_SoVITS/text/**`
in place (read-only) and substitutes three torch-dependent pieces:

- `text.g2pw.torch_api` ← `g2pw_mlx` (MLX fp32 G2PW; argmax must be
  token-exact, so the export stays fp32 — fp16 logits can flip near-ties);
- `text.LangSegmenter` ← a faithful re-port (the official one pulls
  fast_langdetect; our port keeps identical segmentation on the parity
  corpus);
- `TTS_infer_pack` ← a stub (only the segmentation-method table is needed).

Parity evidence: phones 7/7 + 5/5 exact on the zh/en bench sets, 23/23
mixed-language segments; G2PW polyphone decisions token-exact (fp32);
BERT features to fp32 ulp. The loader means a CPUFast checkout + data
files (fast_langdetect `lid.176.bin`, G2PW export) is a runtime
requirement, documented in the README front-end section.

Byte-exactness traps found by the task-6 8/8 md5 gate:

1. **PyAV is load-bearing.** Without `av`, `load_audio_official` falls
   back to soundfile + linear resample. That changes HuBERT inputs a few
   frames → different prompt semantic codes → different AR stream
   (v2: 112 vs 119 tokens with identical weights/seed). A missing
   optional dependency silently changes the anchor. Rule: runner envs
   must have `av==18.1.0`; the e2e scripts warn loudly on fallback.
2. **Front-end device must equal the process device.** `TextFrontend`
   computes BERT/G2PW; running it on CPU while the audio stages run on
   GPU (or vice versa) yields fp16-vs-fp32-flavored checksum differences
   in `all_bert` and divergent AR sampling. The unified CLI resolves the
   device once and threads it everywhere (same rule as the wrappers).
3. **`_load_sovits_v1v2` needs the real version string** (`v2Pro`,
   `v2ProPlus`) — the `is_v2pro`/sv-embedding branch must activate; a
   "v2" fallback broadcasts 1024-dim ge against 512-dim buffers.

### Env-contract lessons

The anchor contract quadruple (python 3.12, mlx==0.32.2, numpy==2.5.2,
device) is now verified from BOTH directions:

- **mlx/numpy pins are semantic, not cosmetic.** 0.32.2→0.32.3 and
  2.5.2→2.5.3 each flip inverse-CDF near-ties in the AR sampler
  (bench cell 119 → 112 tokens, measured A/B, task-2). Equal pins +
  equal inputs + equal device → bit-identical wavs; anything else moves
  the anchors. Re-render procedure: re-run `bench.py --seed 0` for all 8
  versions, update BENCH.md's contract line.
- **The ref cache cannot cross a pin change.** Cache keys embed an
  environment fingerprint (python/mlx/numpy versions), so a version bump
  orphans stale entries instead of silently reusing them.
- **NAS cold reads masquerade as regressions.** /Volumes/2T cold reads
  drop to 14–68 MB/s; I/O-shaped stages (frontend, model_load) inflate
  up to 100x (model_load 49.8 s cold vs 0.18 s warm) with zero competing
  processes. Measurement doctrine: label cache state per I/O stage,
  re-run once before believing a regression, do ps-clean timing runs on
  warm caches.

### Device gate (post-hardening)

CPU is not a supported compute path for AR/CFM/decode in any version —
v3-family CFM silently produces all-zero audio on CPU, and v1/v2 "work"
at 30–100x slowdown, which is a trap. `require_gpu_for_pipeline` now
hard-stops pipeline runs that resolve to CPU unless `--frontend-only`
(front-end smoke, CPU-legal) or `--cpu-i-know-broken` (explicit force).
Every e2e writer path runs `assert_audible` pre-write: an empty,
non-finite, near-silent (peak ≤ 0.01) or degenerate wav refuses to be
written — a silent zero wav can never be saved again (the 309 s all-zero
v3 incident that motivated this).

### Byte-identity gate (task-6, the closing methodology)

`tools/e2e.py --version X` vs `tools/e2e_v*.py` wrapper runs, seed 0,
GPU, one process at a time under the lock: all 8 versions md5-identical
(v1 a8cf1493…, v2 b218b8e1…, v2Pro 28abb525…, v2ProPlus ddf19ef5…,
v3 2ea7506f…, v4 85951739…, v5dev 3acfa532…, v5turbo eb03e506…), token
counts 156/119×3/113×4 matching the re-cast anchors. The gate caught
four real defects (above) before merge — it is now the standard
acceptance for any refactor that touches entry points.

## s2 training addendum (task-5, trainer vs torch — findings that change semantics)

Source: 20-step deterministic CFM-loss comparison vs the OFFICIAL
`SynthesizerTrnV3` (cuda_graph repo, CPU fp32, eval), identical batches +
baked t/x0/prompt/2-step state (`worktrees/s2-cfm/.tmp/ref_batches/`).
Result: worst per-step rel diff **0.01%**, mean loss identical (0.0653).
Four findings, all merged:

1. **SDPA mask dtype semantics** (`gsovits_mlx/sovits/dit.py`):
   `mx.fast.scaled_dot_product_attention` treats FLOAT masks as ADDITIVE
   score offsets but BOOL masks as keep-masks. The DiT passed the 0/1
   float padding mask → uniform +1 score shift (softmax-invariant on
   all-ones masks — why inference parity never caught it) and padding-key
   leakage on padded batches (~4% mean velocity error, sample 90/521).
   Fix: `mask.astype(mx.bool_)`. Audit rule for future ports: any
   `mx.fast` SDPA call must pass bool (keep) or explicitly-negative
   additive floats — never raw 0/1.

2. **`nn.Module.parameters()` misses plain-object attrs**: the V3 model
   holds `quantizer.embed` and the whole DiT tree on plain objects. fp16
   there flipped ~50% of quantizer code picks (fp16 L2-distance GEMM on
   ~545-magnitude distances) and drifted CFM loss up to 12%. Fix:
   `upcast_training_model()` in `train/s2_cfm.py` walks both trees
   explicitly.

3. **fp16 DiT BACKWARD overflows (fp16 training is banned)**: per-block
   gradient magnitudes grow monotonically through the 22 blocks and
   exceed fp16 (65504) from block ~8 while the FORWARD stays finite
   (~4.3e3 max) — the official `autocast(fp16_run=True)` regime is NOT
   reproducible on MLX for training (no grad-safe fp16 GEMM backward).
   Remedy: fp32 weights + two-pass checkpointed block-wise backward
   (`S2V3TrainModel.train_loss_and_grads`, s1 pattern; retained = block
   inputs only, one block graph alive at a time; ckpt-vs-fused gate:
   loss bit-exact, grads ≤1e-6 abs). The ckpt head initially missed
   `d_embed(dt)` (5.4% loss error) — caught by that gate; keep the gate.

4. **LR × batch-size divergence masquerading as stochastic NaN**: at
   lr=1e-4 with batch=3 the loss climbed 0.07→4.1 then cascaded to NaN
   (11/55 batches); official trains batch=32 with warmup machinery.
   lr=2e-5 + `--warmup-steps 50` → 200/200 steps, stable loss band.
   Known issue: ~5% of batches still produce non-finite loss/grads and
   are skipped (guard is GradScaler-faithful); 100/100 clean in isolated
   hunts across seeds incl. CLI-seed parity — suspected rare draw ×
   post-optimizer weight-state interaction, unproven. Two failing
   batches archived as npz under the task worktree for bisection.

## prepare_data pipeline parity (moved from tools/prepare_data.py docstring)

* 2-name2text word2ph: official list repr ("[2, 2, 1]"); non-zh lines
  literal "None". Parity env for TRAINING data: GSOVITS_FRONTEND_F32=1,
  GSOVITS_G2PW_FP16=0 (fp32 BERT 1.1e-4 vs transformers CPU; fp16 G2PW
  flips polyphones ~8% of corpus lines — 2/24 measured; fp16 BERT +5e-2).
  Known front-end gap: vendored CPUFast chinese2 (torch g2pw.pth) vs
  cuda_graph ONNX g2pW disagree on ~1/24 lines (ONNX-vs-torch model
  difference inside the official repos, not an MLX bug).
* stage-2: ffmpeg f32le decode, 0.95/0.5 amplitude quirks, librosa
  soxr_hq resample (bit-equal), HuBERT fp16-saved ssl (fp32 compute,
  1.4e-4), scipy-equal int16 wav32k (md5 24/24), NaN skip like official.
* stage-2b sv: torchaudio sinc_interp_hann numpy port
  (gsovits_mlx/audio_resample.py, 1.8e-7 — NOT soxr, 5e-2). fbank +
  ERes2NetV2 details: §v2Pro/v2ProPlus addendum above.
* stage-3 quantizer argmin in fp32 on the fp16 ssl: expand-form L2 in
  fp16 has catastrophic cancellation (~10 distance error at |d|~700)
  flipping argmin on sub-0.5 margins (24/2915 frames). fp32 diff-form
  matches torch exactly (24/24 files, 2915/2915 frames, v1+v2 weights).

## s2 GAN trainer contract (moved from tools/train_s2.py docstring)

* AdamW G: 4 param groups — base lr 1e-4; text_embedding/encoder_text/
  mrte at lr*0.4 (text_low_lr_rate); betas (0.8, 0.99), eps 1e-9, wd 0.01.
  AdamW D: single group, same hyperparams. SHARED GradScaler: D step
  scales+steps on the pre-update scale; scaler.update() ONLY after G step.
  ExponentialLR gamma=0.999875 per epoch. clip_grad_value_(None) is a
  NO-OP (official commons clamps only when clip_value non-None).
* freeze_quantizer=True: ssl_proj+quantizer excluded from optimizers;
  quantizer fp32 outside the low-precision graph. fp16 autocast for G
  forward; fp32 losses (mel/kl/fm/gen).
* --g-forward single (default): ONE G forward per step in ONE traced
  value_and_grad — official torch structure (net_g() once, y_hat reused:
  D sees detach(), G sees D as constants). ONE rand_slice_segments +
  ONE posterior-noise draw per step, shared by both branches.
  double (legacy): separate D/G-step traces, each its own G forward,
  SAME RNG key (official single-draw); --double-split-draw restores the
  legacy split-draw deviation for A/B.
* Deviations: G-step D forward reads PRE-update D weights (MLX traced
  grads need frozen inputs; official torch sequential-overlap reads the
  updated .data — measured immaterial on 200-step smokes); DDP grad
  averaging absent; bucket sampler randperm numpy PCG64 not torch
  Philox; v1/v2 have no pretrained s2D on this machine (D inits fresh;
  v2Pro/ProPlus load s2Dv2Pro).
* Precision: fp32 default (official fp16_run=false). fp16 unstable —
  KL exp overflow collapses the scaler (INCIDENTS#fp16-scaler-collapse).
  bf16 opt-in: bf16 G/D graph, fp32 loss heads, fp32 masters, no scaler.

## s2 data pipeline semantics (moved from gsovits_mlx/train/s2_data.py)

* Construction: intersect 2-name2text ∩ 4-cnhubert ∩ 5-wav32k (+7-sv_cn
  Pro), <100-file duplication loop, random.seed(1234) ORDER shuffle
  (stdlib — bit-exact), duration filter 0.6<d<54 via size/(32000*2),
  lengths = size // (2*hop).
* __getitem__: int16 wav → fp32 /32768; MLX stft front-end (hann
  periodic, reflect pad (n_fft-hop)/2, center=False — parity-tested);
  ssl .npy with F.pad(...,(0,1),"replicate") when frame counts differ
  (official quirk). Collate: sort by spec_len DESC (stable), max lens
  2*((max//2)+1), zero-pad.
* Bucket sampler: boundaries [32,300,...,1900], per-bucket repeat to
  batch multiple, in-bucket sequential, batch-level shuffle. RNG
  DEVIATION: torch Philox(seed=epoch) not bit-reproducible outside torch;
  numpy default_rng(epoch) — deterministic per epoch here, but batch
  order/tail composition differ from torch. Dataset-level shuffle IS
  bit-exact. Real official runs reshuffle per-process anyway
  (PYTHONHASHSEED-dependent list(set()) ordering) — batch composition is
  not a parity target.
* Pad-neutrality of pad_multiple/fixed widths (opt-in memory feature):
  extra pad is zero columns beyond *_lengths; all loss/mask paths
  length-bounded; the posterior randn draw follows the padded shape
  exactly like official torch padding — not a semantic deviation
  (proof: docs/TRAINING.md §fixed-shape; measured 83 distinct spec
  widths in 120 b3 batches motivated the feature).

## s2 CFM trainer port details (moved from gsovits_mlx/train/s2_cfm.py)

Reference (READ-ONLY): cuda_graph repo. Line anchors at port time:
* SynthesizerTrnV3.forward L1305-1333: fp32 island — y_mask; ge =
  ref_enc(y[:,:704]*y_mask, y_mask); freeze_quantizer=True runs
  ssl_proj+quantizer+enc_p under no_grad (enc_p frozen too, __init__
  L1300-1303); quantized nearest x2; bridge (Conv1d+LeakyReLU 0.01);
  nearest interp x1.875 (v3) / x2 (v4/v5); wns1 Encoder(512,512,512,
  k5,d1,L8) with ge at mel_lengths; prompt_len = floor(rand(B) *
  mel_lengths*2/3); mel/fea truncated to min; CFM loss.
* CFM.forward L1174-1207: per-sample t~U(0,1), x0~N(0,1), vt=x1-x0,
  xt=x0+t*vt; prompt zeroing; 30% two-step branch (python random.random()
  < 0.3; base = torch.randint(2, 8, (b,)) — HIGH IS EXCLUSIVE, bases in
  [2,8); d=2^-base; d_input masked <1e-2→0; v1/v2 DETACHED; vt=(v1+v2)/2
  detached; dt=2*d); final estimator call WITH grad; loss = mean over ALL
  elements of MSE over the prompt_i:mel_len_i slice (torch MSELoss
  semantics).
* DiT training forward (f5_tts dit.py non-infer path): text_embed+pos →
  InputEmbedding(x, cond=prompt) → blocks (AdaLN from t + d_embed(dt)
  when use_step_embedding) → norm_out → proj_out. Head-0-only RoPE and
  trailing mask multiply are checkpoint-family quirks already in
  gsovits_mlx/sovits/dit.py (inference-verified).
* v5 upstream-broken finding: official v5 s2 training TypeErrors —
  webui.py L1498 Radio has no v5 choices and SynthesizerTrnV3.__init__
  L1295-1299 swaps self.cfm for the inference-only CFMV5. v5's estimator
  DiT is identical (use_step_embedding=False), so this trainer runs
  v5dev/v5turbo on the plain CFM path.
* LoRA: rank-32 on every Attention to_k/to_q/to_v/to_out.0 (official peft
  targets, alpha=r → scaling 1.0); x@W.T + b + (x@A.T)@B.T; base DiT
  frozen; trunk (ref_enc/bridge/wns1/linear_mel) fully trainable; peft
  wraps ONLY net_g.cfm.

## s2 v3/v4/v5 data pipeline semantics (moved from s2_v3_data.py)

* LoaderV3/V4 (official data_utils.py L279/L517): global ``version`` env
  drives cleaned_text_to_sequence (webui exports it before spawning
  s2_train_v3_lora); <100-file repeat max(2,100//n); seed-1234 order
  shuffle; duration filter getsize/32000/2.
* V3 audio: spec32k = spectrogram_torch(2048/640/2048, center=False);
  mel = 100-mel(24k ffmpeg audio, 1024/256/1024, center=False) normalized
  (x+12)/14*2-1; ssl +1 replicate frame on frame-count mismatch.
  V4: mel = 100-mel(32k, 1280/320/1280, center=False). v5 family uses
  the V4 loader (`V3 if version=="v3" else V4`).
* CollateV3: sort spec-len DESC; max_ssl_len1 = 8*((max+8)//8);
  max_ssl_len = max_spec_len = 2*((max//2)+1); max_mel_len =
  int(max_ssl_len1*1.25*1.5). CollateV4: mel row width = max_spec_len*2.
* Memory additions (opt-in, OFF by default): quantize_pad_multiple pads
  to 64-frame multiples (zeros beyond lengths; masked losses unchanged;
  recycles Metal buffer sizes — 5.8GB cache → small set, measured
  2026-10-09); LengthBucketSampler = official bucket semantics
  (boundary buckets, pad-to-batch-multiple, per-epoch shuffles), opt-in
  via --length-buckets; default loader order stays official.

## s1 trainer optimizers (moved from gsovits_mlx/train/optim.py)

ScaledAdam (official AR/modules/optim.py, Xiaomi Apache-2.0) verified
quirks, replicated faithfully (deviating would break golden parity —
tests/test_train_core.py::test_scaled_adam_golden):
* clipping_scale multiplies the grad ONLY where it enters scale_grads
  (per-tensor size/scale update); _step() re-reads p.grad UNCLIPPED for
  exp_avg_sq and the momentum delta — the main Adam-style update is
  never actually clipped (upstream bug preserved).
* clipping ring buffer + threshold live per PARAM GROUP (torch calls
  _get_clipping_scale once per group); ring stored in the first tensor's
  state.
* clipping quartile "median": sorted_norms[min(period-1, (period//4)*2)].
* _step() divides exp_avg_sq by bias_correction2 only when bc2 < 0.99
  (approximation switch, not a stability guard).
* scalar tensors (numel==1): regular-Adam path, lr*scalar_lr_scale=0.1,
  param clamped to ±scalar_max BEFORE adding delta.
* param_rms recomputed only when step % size_update_period == period-1;
  main update alpha = param_rms.clamp(min=param_min_rms).

## s1 training forward contract (moved from gsovits_mlx/train/s1_model_train.py)

Ports AR/models/t2s_model.py Text2SemanticDecoder.forward_old (cuda_graph
~L450-510) over the inference weight layout. Preserved exactly:
* x block: ar_text_embedding(x) + bert_proj(bert^T) + ar_text_position.
* y block: codes = y*(1-y_mask); y, targets = pad_y_eos(codes, y_mask_int,
  eos=1024): decoder input y = targets[:,:-1] (starts with the row's first
  real code; EOS at every pad slot), targets = codes + eos*mask with a
  trailing EOS column — every target row ends with EOS.
* attention mask: concat[x_attn_mask (x rows see all x, never y),
  y_attn_mask (y rows see all x + causal triu(1) within y)] logical_or
  padding (x: make_pad_mask_LEFT, y: make_pad_mask) broadcast
  (B*heads,1,src) → additive -inf float, built in fp16 (official autocast
  materializes the mask in x.dtype).
* logits = ar_predict_layer(xy_dec[:, x_len-1:]) with x_len = x_lens.max():
  (B, Y+1, D) — last x position (attended only x) predicts the first y
  token; y j predicts j+1; final y position predicts EOS. This is why the
  bucket sampler matters: rows share x_len (max) and pad tails land on EOS
  targets.
* loss = cross_entropy(logits.permute(0,2,1), targets, reduction="sum")
  over ALL (B, Y+1); acc = MulticlassAccuracy top_k=3, ignore_index=EOS,
  multidim_average="global".
* Numerics: fp16 forward + fp32 CE (official 16-mixed keeps CE fp32);
  weights fp32 masters, cast per forward.
* Dropout: official TRAIN stack runs 0.1 (positional + per-layer); config
  dropout=0 covers only TokenEmbedding. Port default p_dropout=0 (task
  spec, identical math in eval); GSOVITS_S1_TRAIN_DROPOUT>0 = knob.

## s2 GAN step semantics detail (moved from gsovits_mlx/train/s2_gan.py)

1. fp16 autocast forward of net_g (full SynthesizerTrn.forward incl. the
   autocast-disabled fp32 ssl_proj+quantizer block; freeze_quantizer=True
   eval()s ssl_proj+quantizer every step — VQ self.training branches
   (commit loss, EMA) never run: kl_ssl is EXACTLY 0.0, logged verbatim).
2. mel = spec_to_mel(spec); y_mel = slice_segments(mel, ids_slice, 32);
   y_hat_mel = mel_spectrogram(y_hat) — FRESH stft on the generated
   20480-sample slice (magnitude → mel fb → ln clamp 1e-5).
3. D step first: net_d(y, y_hat.detach()) fp32 LSGAN losses;
   scaler.scale().backward(); unscale_(optim_d); clip_grad_value_(None)
   NO-OP (official commons clamps only when clip_value non-None);
   scaler.step(optim_d).
4. G step: net_d(y, y_hat) feature maps; fp32 losses mel*45 + kl*1.0 +
   feature*2 + generator LS + kl_ssl*1; single shared GradScaler.update()
   ONLY here (D step ran on the pre-update scale, like deferred torch).
5. ExponentialLR gamma=0.999875 per epoch (+ official resume fast-forward).

## text front-end loader (moved from gsovits_mlx/text/vendored_cpufront.py)

Three guarded substitutions (torch-free runtime):
* text.LangSegmenter imports fast_langdetect UNGUARDED at module top (LID
  model never used on the getTexts() path) → gsovits_mlx.text.lang_segmenter.
* text.g2pw.__init__ → g2pw.py → torch_api.py (torch) → stub package with
  gsovits_mlx.text.g2pw_mlx injected as text.g2pw.torch_api (chinese2.py
  imports only G2PWTorchConverter from there).
* TTS_infer_pack/__init__ imports TTS.py (torch) → stub; stdlib-only
  text_segmentation_method.py executed directly.

Data-file contract (resolved via __file__ inside the CPUFast checkout,
read-only): opencpop-strict.txt (zh pinyin→phone), cmudict*.rep +
*_cache.pickle + namedict_cache.pickle (english G2P/names),
jieba_posseg_assets_v1.pkl (zh posseg), opencc_s2tw_assets/ (s2t),
ja_userdic/ (openjtalk), G2PWModel/ (zh polyphonic; g2pw.pth NOT read —
MLX export g2pw.safetensors via g2pw_mlx.py), pretrained_models/
fast_langdetect/ (optional, only for other lang_segmenter consumers).

## text preprocessing (moved from gsovits_mlx/text/preproc.py)

Official TextPreprocessor::get_phones_and_bert_with_phone_units replicated:
* text = re.sub(r' {2,}', ' ', text) at the entry (NOT in pre_seg_text).
* pre_seg_text: strip('\n') → '。'/'.' prefix when the text does not start
  with a split separator and first segment < 4 chars → seg_method (cut0 =
  no-op) → collapse '\n\n' → split/filter/merge_short_text_in_array(5) →
  pad trailing separator → split_big_text(>510).
* per-segment: clean_text_with_phone_units → cleaned_text_to_sequence →
  bert: zh = MLX BERT on normalized text (word2ph repeat, hidden -3),
  else all-zero (1024, len(phones)) (official get_bert_inf).
* not final and len(phones) < 6 → redo WHOLE text as "."+text with
  final=True (short-text recursion).
* replace_consecutive_punctuation("!?…,.-" → first) on raw input inside
  preprocess() before pre_seg_text.
* batch zh (official _preprocess_batch_zh) batches ONLY the BERT forwards;
  same condition (lang in {zh, all_zh}, no [A-Za-z] in any segment) and
  output order.
Deliberate differences: phone_units not exposed; bert fp32 numpy (1024,T);
add_reference_audio/ref-text handled by callers — prompt text must NOT go
through pre_seg_text '。'-prefixing (official TTS.py uses a separate
simpler path).

## s1 data pipeline quirks (moved from gsovits_mlx/train/s1_data.py)

* phoneme-file line filter: keep only lines with exactly 4 tab-separated
  fields (4th field "text" parsed but unused downstream).
* min_num=100 duplication: replicate the WHOLE kept list max(2, 100//len)
  times when fewer than 100 samples survive.
* semantic length filter: len(semantic_ids) > max_sec*hz (hz=25; max_sec
  from the version's s1 config: 54 v1, 57 v2-family).
* missing BERT features → zeros (never skip); shape mismatch → ValueError
  (official hard assert kept).
* sampler pad-to-total via index repetition (drop_last=False); batch
  order shuffled after grouping.
