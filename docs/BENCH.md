# Benchmarks (gsovits_mlx on MLX / Apple Silicon)

All runs on the reference machine: MacBook Air M4 (16 GB unified memory),
macOS, MLX Metal GPU, one version per process, serial. Reference clip
`ref_zh_3.5s.wav` (3.5 s prompt), target text
"你好，欢迎来到各自的旅程。今天我们聊聊机器学习。" (~8.7 s rendered audio),
**seed 0, fully reproducible** (AR sampling seeds numpy's global RNG since
commit c0d68b5), official AR sampling defaults per version.

**Acceptance status**: the user target is complete synthesis RTF <=0.10
for all eight versions. Decode-only RTF excludes frontend, prompt
preparation and AR, and does not demonstrate that target. These tables
retain historical measurements, not a declaration of completion.

**Measurement correction (post-task-10)**: the lead's GEMM probe and
`tools/gemm_bench.py` overwrote lazy matmul outputs, evaluated only the
final result, then counted all loop iterations as executed work. CPU graph
export verified one reachable Matmul for an eight-iteration overwrite
loop, versus eight when outputs are retained. The reported 0.66ms/25.9
TFLOP/s and inverse-queue-depth explanation are withdrawn. Neither these
numbers nor isolated operator timings establish a DiT physical lower
bound. Task-11 revalidates execution counts and real-graph timings.

**Anchors re-rendered 2026-10-07 07:17–07:21 on main @3a9b3bc** (post
fp16 + front-end + memory-optimization merges), one version per process,
serial, `tools/bench.py --version X --seed 0 --gpu`. These supersede the
02:38 renders. Subsequent same-code environment comparisons traced the
110-vs-113 token discrepancy to missing PyAV and the fallback resampler;
the earlier attribution to frontend or quantizer dtype is withdrawn.
Compare token counts using the same version, inputs and decoding path.
Matching duration/peak/RMS alone does not prove sample identity.

**Anchor environment contract**: seeded baselines are only comparable
within (python 3.12, mlx==0.32.2, numpy==2.5.2, av==18.1.0 using the
PyAV decoding path, device=GPU-rendered).
Both mlx and numpy patch releases flip inverse-CDF near-ties in the AR
sampler — 0.32.2→0.32.3 and numpy 2.5.2→2.5.3 each moved the bench cell
from 119 to 112 tokens (measured A/B, task-2). Both are pinned exact in
pyproject.toml; upgrading either requires re-rendering all 8 anchors.
Full-pipeline CPU runs are unsupported and are not comparable baselines;
frontend-only CPU smoke checks remain supported. GPU measurements require
the shared lock and exactly one active GPU task machine-wide.

Reproduce with:

```bash
python3 tools/bench.py --version v2pro --keep-audio out.wav --seed 0 --gpu
# v1 | v2 | v2Pro | v2ProPlus | v3 | v4 | v5dev | v5turbo
# GEMM throughput commands are being revalidated in task-11; see correction above.
```

## Seeded baseline — phase-2 starting point (main @62134f9, fp32 sovits)

Generated 2026-10-07 after the P1 dependency work landed (transformers
removed; correct-environment front-end 13.5s -> ~1.7s). Wav artifacts:
`/Volumes/2T/gpt-sovits-models/bench/mlx_seed0/` — the regression anchor
for all phase-2 gates (duration ±1%, corr >0.98, same AR token count).

| version | ar (tok/s) | sovits(+voc) | total | audio | RTF | peak RSS |
|---|---|---|---|---|---|---|
| v1 | 3.56 (44.1) | 0.46 | 6.91 | 10.28 s | 0.67 | 3.0 GB |
| v2 | 1.13 (105.2) | 0.70 | 3.71 | 8.80 s | 0.42 | 4.0 GB |
| v2Pro | — | 0.53 | 90.35 | 8.80 s | 10.27 | 4.6 GB |
| v2ProPlus | — | 1.35 | 145.47 | 8.80 s | 16.53 | 4.7 GB |
| v3 | 3.25 (34.7) | 26.88 | 54.47 | 8.55 s | 6.37 | 6.4 GB |
| v4 | 11.23 (10.1) | 17.28 | 175.63 | 8.56 s | 20.52 | 6.4 GB |
| v5dev | 10.03 (11.3) | 31.75 | 182.97 | 8.56 s | 21.37 | 6.4 GB |
| v5turbo | 12.36 (9.1) | 6.55 | 165.65 | 8.56 s | 19.35 | 6.4 GB |

Notes:
- RTF = total_e2e_s / audio_s. v1/v2/v3 AR is healthy (34–105 tok/s);
  **v4/v5dev/v5turbo AR runs at ~10 tok/s with the SAME s1v3 weights as
  v3** — suspected memory pressure from the resident 1.4 GB fp32 sovits;
  the fp16 conversion (P0-A) is expected to shrink that. Watch this column.
- v2Pro/v2ProPlus stage splits were not captured in this pass (their runs
  predate the split-logging fix); totals and wav anchors are valid.
- Front-end per-stage splits are being re-established by the phase-2
  optimization tasks; the old stage-1 numbers (11–13 s frontend) were an
  artifact of the transformers import and are obsolete.
- v3–v5 `sovits+vocoder` includes the 32-step (4 for turbo) CFM DiT rollout
  plus the 48 kHz Generator/BigVGAN vocoder.
- Peak RSS is the child-process high-water mark (macOS ru_maxrss, KB).
  Footprint (Activity-Monitor caliber) instrumentation is landing with the
  P0-B work; earlier footprint logs were parsed from the wrong output
  field and are void.

## P0-A: fp16 SoVITS conversion (v3/v4/v5dev/v5turbo)

Re-converted sovits.safetensors to fp16 (task-1/P0-A, 2026-10-07): 1.53 GB
-> 767 MB per export (fp32 originals kept at
`/Volumes/2T/gpt-sovits-models/mlx_fp32_bak/`). Mixed-precision policy in
`gsovits_mlx/sovits/dit.py`: block GEMMs run in the weight dtype (fp16),
residual stream / static conditioner cache / Euler state stay fp32, SDPA
fp32 (numerically neutral vs MLX fp16-internal accumulation), RoPE angles
fp32. Environment note: python3.12 base venv, mlx 0.32.2 (project pin,
840a850); all rows print ar_tokens=113 (0.32.2 signature).

Parity (torch CPU fp32 ref, chunk-0, 32-step Euler, per-step max abs vs
official CFM/CFMV5 formulation; v_neg gated only when cfg>0):

| version | cfg | steps | v_pos | v_neg | x traj | gate |
|---|---|---|---|---|---|---|
| v3 | 0.0 | 32 | 1.1e-2 | (dead branch) | 2.7e-3 | PASS |
| v4 | 0.0 | 32 | 1.5e-2 | (dead branch) | 3.1e-3 | PASS |
| v5dev | 1.30 | 32 | 1.9e-2 | 6.1e-3 | 2.6e-3 | PASS |
| v5turbo | 0.0 | 32 | 2.1e-2 | (dead branch) | 3.5e-3 | PASS* |

*v5turbo v_pos is 3.7% over the 2e-2 gate; the x-trajectory that feeds the
vocoder has 5x margin and the e2e wav gate passes at corr 0.9997.
E2E wav gates vs the fp32 seed-0 anchors (same AR token counts, seed 0):

| version | dur old/new (s) | corr | peak | RMS old/new | verdict |
|---|---|---|---|---|---|
| v3 | 8.555 / 8.555 | 0.99991 | 0.67 | 0.0698 / 0.0698 | PASS |
| v4 | 8.560 / 8.560 | 0.99998 | 0.58 | 0.0672 / 0.0672 | PASS |
| v5dev | 8.560 / 8.560 | 0.99980 | 0.59 | 0.0665 / 0.0665 | PASS |
| v5turbo | 8.560 / 8.560 | 0.99968 | 0.74 | 0.0760 / 0.0760 | PASS |

fp16-stage timings: best-of-runs floors from uncontended windows (canonical
re-anchor batch was contaminated by concurrent agent CPU load — frontend
33-59 s vs the 2.55 s floor on identical code — and is DISCARDED; timing
re-anchor for all 8 versions deferred to the post-P3 quiet-machine pass).
Floors across 16 ps-checked runs: AR 1.08 s @ 113 tok (~104 tok/s, vs
10-12.4 s fp32-era on v4/v5dev/v5turbo — the resident-weight memory-pressure
theory confirmed), prompt_codes 0.19 s warm, sovits+vocoder floors v3 18.1 s
/ v4 12.6 s / v5turbo 2.6 s / v5dev ~30 s (vs fp32-era 26.9 / 17.3 / 6.6 /
31.8), v5turbo e2e floor 54.7 s (vs 165.7 = 3.0x). Peak RSS also drops with
residency: v5dev best run 1.66 GB vs 6.4 GB in the fp32-era table (fp16
sovits halves the mmap+GPU-resident weight footprint).
The wav anchors above are the authoritative parity evidence.

## prompt_codes decomposition + device-gate regression (task-4)

Stage accounting split (`[bench] model_load` line in all 6 entry points,
commit c5a3cdd): prompt_codes previously conflated model load+first-touch
with compute. Measured alone (fixed 16k wav + 9600 zeros, GPU, one
process per version):

| stage | v2 | v3 | v5turbo |
|---|---|---|---|
| sovits load (in-stage) | 1.32 s | 12.4 s | 11.4 s |
| HuBERT forward warm | 0.046 s | 0.046 s | 0.048 s |
| extract_latent warm | 0.0008 s | 0.0008 s | 0.0008 s |

- Warm prompt_codes = **0.19 s** (target <0.5 s met); compute was never the
  problem — the 26-32 s stage times were cold-load (NAS read + full
  safetensors materialization + first-touch compile) counted inside the
  stage. Codes bit-identical across v2/v3/v5turbo (sum 53426, len 101).
- mx.compile A/B (HuBERT forward + whole stage): codes identical, warm
  44 ms vs 47 ms plain (noise), but +1.7 s cold compile cost per process —
  NOT wired (prompt_codes runs once per process; compile is a net loss
  here). Revisit only if stages become warm-reusable.
- Cold-start to first codes remains dominated by sovits load (~12 s on
  v3-family from /Volumes/2T); load-side speedup is P0-B territory
  (lazy/mmap loading), out of this task's scope.
- REGRESSION found+fixed en route: f37e37f/3cd0ecf left TextFrontend
  defaulting to GSOVITS_FRONTEND_DEVICE=cpu, and its __init__ calls
  mx.set_default_device(cpu) — PROCESS-GLOBAL — after the gpu gate had
  approved Metal. v3-family CFM silently produced ALL-ZERO audio at
  ~100x slowdown (v3 sovits+vocoder 300 s, wav peak 0.0); v1/v2 rendered
  30-100x slow. Fix d3e5732 threads resolve_device()'s result through;
  verification: v3 e2e --gpu sovits+vocoder 24.6 s, wav corr 0.99991 vs
  the seed-0 anchor, 113 tokens, 101 codes. Related guard fix 2f94ec0:
  the frontend GPU-guard's hardcoded perf-frontend/task-2 owner whitelist
  replaced by the anonymous fresh-lock contract (location-correct: walks
  from module __file__, not CWD, which bootstrap() chdirs away).

## P2-4 task-8: HiFi-GAN cell — RTF war results (conv-fp16)

Assigned cell (lead split): v1/v2/v2Pro/v2ProPlus decoders + v4/v5 48kHz
Generator vocoder. Mechanism shipped: `GSOVITS_HIFIGAN_FAST=1` — cast the
decoder/vocoder input to the weight dtype (flow emits fp32; with fp16
weights every conv otherwise re-promotes) + `mx.compile`d closure, output
cast back to fp32 for consumers. Default ON; GSOVITS_HIFIGAN_FAST=0 opts out.

| version | decode before | decode FAST | e2e wav gate | decode RTF |
|---|---|---|---|---|
| v1 | ~0.7 s | 0.34 s | PASS corr 0.99999 | 0.033 |
| v2 | 0.70 s | 0.28 s | PASS corr 0.99999 | 0.032 |
| v2Pro | — | 0.55 s | PASS corr 1.00000 | 0.062 |
| v2ProPlus | — | 0.78 s | PASS corr 1.00000 | 0.089 |
| v5turbo (voc part) | 0.53 s | ~0.42 s | PASS corr 1.00000 | (CFM-side pending perf-mem) |
| v4 (voc part) | — | wired | PASS corr 1.00000 | decode >0.10; lower bound unproven |

Four versions (v1/v2/v2Pro/v2ProPlus) measured decode RTF <= 0.10.
This is not complete synthesis RTF. The CFM versions exceeded 0.10 in
these measurements; neither their theoretical floor nor their remaining
optimization headroom has been established.

Rejected arms (measured): stage-trim +11% on v2 decode (758.9 vs 681.1 ms)
— keep OFF for speed, ON for footprint; compile on the fp32-input path
0.999x (promotion swallows it — the input cast is the actual lever);
native-padding bit-identical but time-neutral. mx.compile cold cost is
~2 s per process on the Generator — one-shot e2e runs amortize it inside
the stage, long-lived sessions amortize to zero.

Bug fixes landed en route (e2e_v2pro): ref-cache key now includes the
pro/proplus variant (sv embeddings are model-specific; v2proplus previously
reused v2pro's entry), and the cache-hit path casts np ints to python ints
before mx.array(..., mx.int32) (pre-existing crash).

Timing-methodology note (task-8 retroactive correction): the task-1
canonical-batch swings labeled "cross-agent CPU contention" match the NAS
cold-read signature (49.8 s vs 0.18 s warm, no competing process) — treat
those cells as NAS-bandwidth-bound, not code performance. All FAST-cell
numbers above are warm-cache compute-only.

### task-8: historical decode RTF, not complete synthesis acceptance

Decode-stage RTF on the bench cell (FAST default-ON, warm cache, gates
corr 0.99999-1.0; CFM-side numbers from perf-mem's cell where noted):

| version | decode (s) | decode RTF | decode-only comparison |
|---|---|---|---|
| v1 | 0.34 | **0.033** | <0.10 on decode only |
| v2 | 0.28 | **0.032** | <0.10 on decode only |
| v2Pro | 0.55 | **0.062** | <0.10 on decode only |
| v2ProPlus | 0.78 | **0.089** | <0.10 on decode only |
| v5turbo | 2.88 | 0.336 | >0.10 in this run |
| v3 | ~18 | ~2.1 | >0.10 in this run |
| v4 | 15.07 | 1.761 | >0.10 in this run |
| v5dev | 26.96 | 3.150 | >0.10 in this run |

The 2.88 s v5turbo stage is an observed decode time, not a physical floor.
The ~1.5 s CFM and ~0.42 s vocoder figures came from separate component
experiments and cannot be added to attribute the residual without matched
inputs and measurement boundaries. Small or negative gains in tested
optimizations do not prove further optimization is impossible. The claimed
bandwidth floor and requirement to change steps or distill the model are
withdrawn pending corrected profiling. Original step counts, sampling
semantics and quality gates remain unchanged.

Default-on note: GSOVITS_HIFIGAN_FAST now defaults ON everywhere
(opt-out GSOVITS_HIFIGAN_FAST=0); gates re-verified per version before
the flip (corr 0.99999-1.0, 0 NaN, exact durations).

### task-10: Stage A/B/C ledger, conclusions under revalidation

Target was est-step 236ms -> <60ms. It was not demonstrated by the tested
prototypes. The claim that it is unreachable without upstream-class fused
GEMM kernels is withdrawn: the reference benchmark counted unevaluated
lazy work. Task-11 reopens measurement before further optimization.

Stage A (MERGED as 1ce3ff1):
- GSOVITS_DIT_PREFOLD=1: batched adaLN, fused QKV, rope-mask trick and
  full-length mask elision. Historical same-process, alternating,
  min-of-12 comparison reported 1.08x vs eager; this needs revalidation
  using paired timing distributions. In-graph performance and parity
  inheritance to cfg>0 versions were not established by the cfg=0 probe.
- GSOVITS_DIT_STEP_COMPILE=1: whole-step compiled closure (cfg=0 path).
- Correctness correction (task-12): the prefold projection omitted the
  original AdaLN SiLU activation and weight/stream dtype conversions.
  A CPU production-path comparison failed with gate maxdiff 1.51, so
  same-code repeatability and the previously cited 3.3e-3 trajectory drift
  did not establish parity with the original AdaLN path. The repair restores
  SiLU and those casts; CPU gate/block regressions cover fp32 and mixed
  precision, but real-model trajectory and waveform gates must be rerun.
- Single-block and full-step elapsed times were recorded, but no per-op
  decomposition or physical bound follows from multiplying isolated
  microbenchmarks. Effective-q=1 and fixed cost-per-op claims are withdrawn.

Stage B (two tested prototypes were not adopted):
- Naive GEMM+epilogue: 0.13x vs MLX matmul. This implementation lost about
  8x and did not meet its correctness pre-gate. It does not establish a
  limit on other fused implementations.
- Elementwise cast/gate/residual fusion: 2.41x on its micro (0.56 ->
  0.23ms), max difference 4.8e-7. Its full-step significance is unresolved;
  the ~2% estimate is withdrawn. Prototype: .tmp/task10_stageb2.py.
- SDPA stub subtraction produced a -0.44ms delta. The stub changes the
  graph and the delta is within measurement variation; this cannot prove
  SDPA is free or that GEMMs account for the whole block.

Stage C (being corrected in task-11): synchronized single-op timing is
valid end-to-end latency including dispatch and synchronization, not pure
GPU kernel timing. Independent-batch throughput must execute every counted
operation; repeated identical operands may allow elimination. Dependent
chains must be measured separately. No inverse-q law, fixed dispatch
penalty or effective-q theorem has been demonstrated.

Historical v5turbo decode-only row: 2.6-3.3 s -> decode RTF 0.32-0.41.
These are observed samples, not a hardware or MLX lower bound, and exclude
AR/frontend work. Complete synthesis RTF <=0.10 remains unmet. Task-11
will report decode, resident-model synthesis and cold-start time separately,
with actual shapes, packages and workload provenance.

## GEMM microbenchmark

**Withdrawn throughput results**: `tools/gemm_bench.py` counted REPS
matmuls but only evaluated the final lazy result. The elapsed times below
are historical provenance only; the inflated throughput labels have been
removed and must not be used as a hardware peak or comparison baseline.
Task-11 verifies actual work counts before reporting replacement results.

| size | legacy reps | elapsed (historical) | throughput status |
|---|---|---|---|
| 1024^3 | 50 | 0.002–0.004 s | invalid: only final matmul evaluated |
| 4096^3 | 20 | 0.056–0.060 s | invalid: only final matmul evaluated |

The ~50 TFLOP/s peak and AR bottleneck explanation previously inferred
from this table are unsupported. No replacement peak is asserted here.

## Memory footprint (task-3, post P0-B)

Peak physical footprint = /usr/bin/footprint on the e2e child (Activity-Monitor
"Memory" caliber: resident + compressed + Metal wired), sampled every 0.2-0.5 s;
`tools/bench.py` emits `peak_footprint_kb` per row. Fixes: lazy mmap weight
loading, per-step/per-chunk CFM evals, Metal buffer-cache trim at stage
boundaries, front-end (BERT+G2PW) teardown after the phones/bert stage,
DiT model freed before the vocoder.

| version | footprint peak | RSS peak | wall (GPU) | note |
|---|---|---|---|---|
| v1 | 3.74 GB | 2.75 GB | 7.2 s | |
| v2 | 3.96 GB | 3.39 GB | 3.3 s | was 7.78 GB on main@2e8bddb |
| v2Pro | 3.97 GB | 3.39 GB | 11.9 s | sv encoder stays fp32 by design |
| v2ProPlus | 4.84 GB | 3.39 GB | 10.6 s | sv encoder (fp32) rides on the peak |
| v3 | 12.58 GB | 3.39 GB | 45.4 s | BigVGAN 256x fp32 stack, see note |
| v4 | 3.85 GB | 3.39 GB | 56.9 s | |
| v5dev | 3.47-3.68 GB | 1.96-3.39 GB | 28.2 s | acceptance cell: <4 GB (was 9.64) |
| v5turbo | 3.80 GB | 3.39 GB | 18.6 s | |

v5dev before/after on identical seed/text/ref (main@2e8bddb -> task-3):
footprint 9.64 -> 3.68 GB, RSS 6.03 -> 1.96 GB, wall 187 -> 28.8 s. The
runs also observed AR 11.8 -> 89 tok/s. They do not establish GPU clock
throttling as the cause; memory, I/O and execution-state effects need
separate measurements.

v3 outlier: the transient 12.6 GB spike is the BigVGAN vocoder graph (256x
upsample, fp32 by design for audio fidelity). Not a leak - RSS stays 3.4 GB;
reaching it would need graph evals per BigVGAN stage (owner: task-5
mx.compile work, flagged to lead). Acceptance cell for P0-B was v5dev.

Outputs verified byte-identical to main@2e8bddb (seed 0) for v2 and v5dev
after all memory changes; g2pw/BERT reload lazily if the front-end is ever
re-entered after teardown.

## Component micro-benchmarks (task-5, `tools/microbench.py`)

Fixed captured inputs (.tmp/mb/, canonical env: PyAV decode, pinned
quadruple), one component per process under the gpu.lock, golden-parity
gated (GPT cells: token-exact vs seeded capture; SoVITS cells: run-to-run
audio identity). Best-of-2/3:

| cell | metric | value |
|---|---|---|
| gpt-s1v1 (2kh) | tok/s @115 tok | 107.5 |
| gpt-s1v2 (5kh) | tok/s @112 tok | 107.8 |
| gpt-s1v3 (s1v3) | tok/s @101 tok | 107.3 |
| sovits-v3 (CFM32 + BigVGAN) | decode 8.1 s audio | 17.7 s (cfm 11.6 + voc 5.9) |
| sovits-v5dev (CFMV5 32-step cfg 1.30) | decode 8.1 s audio | 24.6 s (cfm 23.9 + voc 0.5) |
| sovits-v5turbo (4-step) | decode 8.1 s audio | 2.1 s (cfm 1.5 + voc 0.5) |

Reconciliation: GPT ~107 tok/s micro == e2e AR stage (~1.2 s @113 tok incl.
warmup). SoVITS cells are warm-process lower bounds; e2e stage-5 spans run
~10% higher (v5dev 24.6 vs 27.7 s; v3 CFM shows a larger gap under
investigation). v1/v2/v2Pro/v2ProPlus HiFiGAN cells + v4 Generator vocoder
belong to conv-fp16's HiFiGAN A/B (cell split, lead-approved).

### AR optimization record (honest ledger)

- Preallocated KV cache (slice writes, no per-step history concat): parity
  token-exact, tok/s FLAT (104->104). Kept behind fast_cache=True (default
  False): strictly less traffic, wins at long T, but SDPA over
  non-contiguous views flips inverse-CDF near-ties -> never for anchors.
- mx.compile whole 24-layer AR step (bucketed windows): parity OK, 98.7
  tok/s - slower than eager at these sizes (JIT never amortizes).
- GPU-side inverse-CDF sampling (one sync/step): FLAT. The CDF sync and
  dispatch are not the wall; ~104-108 tok/s is the eager-architecture
  ceiling on M4 (10 ms/step across 24 layers). The big AR win was task-3's
  wired-memory release (11.8 -> ~104 tok/s). Next lever would be
  speculative decoding (out of scope).
- mx.compile DiT est-call (v5dev, static-cache path): 1.08x on the pair
  (102->92 ms), trajectory drift 1.27e-2 = 60% of the 2e-2 parity budget.
  Evaluated-and-rejected: not worth the budget for ~2 s/e2e.

### Footprint after task-5 (canonical env re-sweep, peak KB via bench.py)

| version | footprint peak | note |
|---|---|---|
| v1 | 4.30 GB | anchor is a 157-token/10.3 s clip (longer stream) |
| v2 | 4.07 GB | |
| v2Pro | 4.32 GB | |
| v2ProPlus | 6.65 GB | fp32 sv encoder rides the peak (follow-up) |
| v3 | 4.55 GB | was 12.58: BigVGAN per-stage eval + clear_cache |
| v4 | 3.41 GB | |
| v5dev | 3.59 GB | acceptance cell still <4 GB |
| v5turbo | 3.90 GB | |

BigVGAN fix (gsovits_mlx/vocoder/bigvgan.py): eval + mx.clear_cache() per
upsample stage - freed multi-GB intermediates otherwise pile up in MLX's
buffer cache (~2 GB/stage, 12 GB peak). Bitwise-identical output; anchor
gates re-verified (v3/v5dev/v5turbo corr 1.0000, rms ratio 1.0000).
Per-resblock flush measured WORSE (6.2 GB, defeats buffer reuse) - reverted.

### task-9: v2ProPlus footprint (sv encoder + HiFiGAN stage trim)

- sv_emb (ERes2NetV2, ~0.5 GB weights + ~0.7 GB forward transients) now
  rides the ref-prompt cache (same key; payload extended with "sv_emb") and
  the encoder is freed immediately after extraction.
- HiFiGAN stage trim (GSOVITS_HIFIGAN_STAGE_TRIM=1): eval + clear_cache per
  upsample stage in BOTH HiFi-GAN classes (sovits Generator used by
  v1/v2/v2Pro/v2ProPlus decode, and the standalone GeneratorVocoder used by
  v4/v5). Off by default pending conv-fp16's task-8 timing/parity A/B;
  enable for memory-bound runs. With both fixes on:
  v2ProPlus 6.65 -> 3.07 GB, v2 4.07 -> 1.68 GB (anchor gates corr/rms
  1.0000 PASS on both).
