# Training Campaign Knowledge (MLX trainers: s1 AR / s2 GAN / s2 CFM)

Curated from the 2026-10-09 → 2026-10-11 three-day training-integration and
optimization campaign (task-3..task-7 + blitz + metal-kernels + batch-probe
+ cfm-eager). Raw per-run evidence stays in `.tmp/` (RUN_PLAN.md files,
jsonl logs) — this file keeps the distilled acceptance matrices, config
guidance, and methodology lessons that future training runs must know.
Incidents with standing rules live in `docs/INCIDENTS.md`; official-semantics
findings live in `docs/PARITY_NOTES.md`.

Provenance: task-6 acceptance matrix written 2026-10-10 (main @ 856dcfe);
addenda for cfm-mem-opt (task-7), gan-speed-opt, gan-bucket-compile,
metal-kernels, batch-probe, blitz, cfm-eager (main @ 2fb78c7 → 6258951).

## 1. Reference configs (proven smokes; do not re-tune blindly)

- **s1 AR** (`tools/train_s1.py`): `--batch-size 4 --grad-ckpt
  --max-steps 200 --epochs 4` (v1 init `models_local/s1v1`, v1 symbol
  table; epochs>1 required to reach 200 optim steps at 235 batches/epoch).
  `--epochs 1` under-runs (58 optim steps on the 939-item dataset).
  Blitz result: **eager (drop `--grad-ckpt`) is the default-fast path on
  this dataset at b4 + metal cap 6144MB** — +24-25% (order-alternating
  pairs ×3 + 150-batch full-mix confirm incl. the widest src=614 bucket;
  measured peak 8.96GB; losses bit-identical to ckpt). Keep `--grad-ckpt`
  for bigger buckets/datasets.
- **s2 GAN** (`tools/train_s2.py`): `--steps 50 --precision fp32
  --warmup-steps 20 --export-inference`; v1/v2 b3 + lr-scale 0.1875 +
  metal 6144MB; v2Pro/Plus b2 + lr-scale 0.125 + metal 5120MB.
  **Recommended post-blitz config: `--g-forward double --fixed-shape
  --metal-kernels --save-every 0`** (validation below).
- **s2 CFM** (`tools/train_s2_v3.py`): task-5 config `--steps 50
  --batch-size 3 --lr 2e-5 --warmup-steps 50`; task-7 winning combo adds
  `--dit-dtype bf16 --length-buckets --pad-multiple 64 --memory-limit-mb
  6144`. lr=1e-4 at batch=3 diverges (loss 0.07→4.1 → NaN cascade);
  official trains batch=32 with warmup machinery.
- Environment for all cross-run comparisons: `PYTHONHASHSEED=0` (the s2
  loaders' `list(set(&))` file-list order is hash-randomized per process —
  different batch composition at the same step otherwise; see
  INCIDENTS.md#pythonhashseed-nondeterminism). pmset logged start+end per
  run. `--save-every 0` (disk law). One GPU process under the canonical
  lock.

## 2. Task-6 acceptance matrix (8/8 versions, 2026-10-10)

Data `.tmp/train_data/exp` (950 text / 952 wav32k / 953 semantic / 952 sv);
artifacts `/Volumes/2T/gpt-sovits-models/train_artifacts/task6/`.

| Ver | s1 run | s2 run | 回灌 synth (dur, tokens, peak, rms, voiced) |
|-----|--------|--------|---------------------------------------------|
| v1 | NEW 200: CE-sum 2885→1339, mean20 2104→1391; footprint 9.21GB | NEW 50: disc 6.07→3.61, mel 20.5→14.4; Metal 7.16GB | PASS 4.44s, 111 tok, peak .821, rms .0896, 45.8% |
| v2 | task-3 200: 2773→1438; 9.20GB | NEW 50: disc 6.06→4.51, mel 19.8→19.4; 7.15GB | PASS 4.72s, 118 tok, peak .735, rms .0752, 39.4% |
| v2Pro | task-3 s1 (same s1v2 arch) | NEW 50 (b2): disc 4.01→3.01, kl 28.1→5.8; 7.97GB | PASS 4.72s, 118 tok, peak .856, rms .0949, 46.3% |
| v2ProPlus | task-3 s1 | NEW 50 (b2): disc 4.10→3.39, kl 29.8→5.6; 9.17GB | PASS 4.72s, 118 tok, peak .613, rms .0625, 45.7% |
| v3 | pretrained s1 (v3-family uses one AR) | task-5 200: CFM ~0.09 (noisy 0.0003-0.97); probe 6.6GB | PASS 4.51s, 113 tok, 32 steps cfg 0, peak 1.000, rms .2475, 91.7% |
| v4 | pretrained s1 | NEW 50: CFM 0.078→0.13 (mean20 0.063); 6 skips | PASS 4.52s, 113 tok, 32 steps cfg 0, peak .551, rms .0591, 49.4% |
| v5dev | pretrained s1 | NEW 50: CFM mean20 0.216→0.166; 5 skips | PASS 4.52s, 113 tok, 32 steps cfg 1.3, peak .444, rms .0588, 53.9% |
| v5turbo | pretrained s1 | NEW 50: CFM mean20 0.080→0.074; 1 skip | PASS 4.52s, 113 tok, **4 steps** cfg 0, peak .589, rms .0525, 53.3% |

All 8/8 PASS `assert_audible` at e2e write time. Export→loader→e2e 回灌
method: shadow-root `.tmp/huiguan_root` symlink farm (task-5 pattern),
`tools/e2e.py --models-root .tmp/huiguan_root --gpu --seed 0`; identical
token counts across s1-shared versions confirm the trained slots were
actually loaded (weights moved ⇒ audio differs from base render).

Wall-time reference (2026-10-10, LPM-on-battery regime — absolute times
power-contaminated, see INCIDENTS.md#lpm-battery-timing-corruption): s1 v1
200 steps 675s; GAN 50 steps 186-229s; CFM 50 steps 307-321s; e2e 回灌
~40-60s/version.

## 3. Batch-size scaling (batch-probe, 2026-10-11)

50-step fresh same-window arms + interleaved pairs, AC charge-tail,
PYTHONHASHSEED=0, user-authorized 10GB footprint gate:

| trainer | arm | t_step | samples/s | footprint peak | verdict |
|---|---|---|---|---|---|
| s1 AR | b4 (baseline) | 0.920 s/batch | **4.348** | 7.05GB | **b4 optimal** |
| s1 AR | b6 | 1.480 s/batch | 4.054 (−6.8%) | 7.05GB | not worth it |
| CFM | b3 (baseline) | 4.91 s | **0.611** | 6.98GB | **b3 optimal** |
| CFM | b6 | 9.90 s (p90 17.3) | 0.606 (−0.8%) | 7.33GB | not worth it |

Marginal cost per added batch item: s1 +0.28s (+30% vs b4 per-sample);
CFM +1.66s (+68%). Neither shows the sublinear scaling that would justify
b6; CFM b6's step-time tail explodes (intra-bucket pad waste). s1 loss is
SUM over batch tokens — b6/b4 mean20 ratio 1.59 ≈ expected ~1.5 token
ratio (semantics consistent).

## 4. CFM memory/speed opts (task-7, cfm-mem-opt)

`--length-buckets` (LengthBucketSampler = official
DistributedBucketSampler math, (32,300,…,1000] buckets) + `--pad-multiple
64` (zero columns beyond every *_lengths entry; loss-neutral, tested) +
`--memory-limit-mb` (default 8192). All opt-in; default OFF = official
order. v3, task-5 config, seed 1234, Metal cap 6GB:

| metric | base (official order) | bucketed | bucketed @3GB cap |
|---|---|---|---|
| steady cache+active mean/max (GB) | 5.43 / 6.44 | 5.72 / 6.44 | **3.22 / 3.22** |
| metal cache alone mean/max | 3.17 / 4.17 | 3.43 / 4.17 | 0.94 / 0.95 |
| footprint mean/max | 6.18 / 7.22 | 6.40 / 7.17 | 3.91 / 3.94 |
| step time mean/median (s) | 9.25 / 8.02 | 8.42 / 7.02 (−9%) | 9.12 / 7.98 (+8% eviction cost) |
| NaN skips | 4/50 | 1/50 | 1/50 |
| mean20 band drift vs base | — | max 0.0164 | max 0.0133 (scatter) |

Reading: at the 6GB eviction cap cache growth is bounded in every arm; the
historical 5.8GB-cache-on-1GB-live failure mode is gone — the Metal cap is
a **tuned knob, not a runaway**. bf16 DiT (`--dit-dtype bf16`, gated):

- GEMM micro-bench: NO bf16 win at op level (fp32 38.77 / fp16 39.03 /
  bf16 45.68 ms batch3 queued) but e2e **1.20-1.31x faster** (7.04 vs
  8.42-9.25 s/step; halved activation/weight traffic in SDPA+convs too).
- Overflow gate: fp16 NaN from block ~9 backward (reproduced); bf16 all 22
  boundaries finite (fp32 exponent range); loss parity 4.1% rel at 20
  steps — within the noisy-CFM band, NOT bit-parity (8-bit mantissa).
- Verdict: numerically safe, e2e faster; KEPT OPT-IN, fp32 default. bf16 +
  buckets + two-pass = 7.04 s/step = CFM's measured optimum on this box.

## 5. GAN speed/memory opts (gan-speed-opt + gan-bucket-compile)

### Item A — single fused G forward (default `--g-forward single`)

ONE net_g forward per step inside one traced value_and_grad — official
torch structure; D branch sees y_hat via stop_gradient, one RNG draw per
step. 50-step same-seed: step-0 all 5 losses bit-identical vs
double-forward; steps 10/19 drift ≤3.1e-3 rel (fp32 reduction order × GAN
chaos). Speed: **−21%/step (1.27x)** same-window pair (relative claim
survives power contamination — both arms equally derated). The fused trace
must be evaluated in TWO phases (D branch → step D → eval G branch) or it
materializes both backwards at once and busts the 8GB gate (8.81GB
measured).

### Item B — bf16 compute (`--precision bf16`): OPT-IN, b4-only value

KL overflow replay: fp16 exp entry → inf (scaler-collapse mode
reproduced); bf16 all finite. 50-step loss parity: disc/gen 0.35%/0.39%,
kl 1.12% — but **loss_mel drifts monotonically 1.5%→16% = systematic
decoder degradation, FAIL on mel** (power-independent; reproduced in the
LPM-off rerun). Speed at b3: NEUTRAL (the earlier "+14.8% slower" was a
low-power-mode artifact). b4 fits under the 8GB gate ONLY via bf16 (peak
7.33GB; fp32-b4 ~9.5GB estimate); b4's value is memory headroom, not speed
(+1% samples/s, within noise).

### gan-bucket-compile: shape governance findings

- **The GAN trainer ALREADY trains bucketed** — `BucketSampler` is built
  unconditionally (official DistributedBucketSampler port); the official
  s2_train.py itself only ever uses the bucket sampler. The varying
  dimension is the collated trunk width (83 distinct spec widths in 120
  batches — a fresh Metal buffer-size family almost every step).
- `--pad-multiple N` (opt-in): collate widths rounded to N-multiples;
  lengths arrays and data prefixes bit-identical to official collate.
  Pad-neutrality: everything in the step is length-masked except exactly
  two width-coupled draws — posterior noise and ref_enc ge — and official
  torch couples BOTH the same way (measured on real weights). Under pinned
  draws loss_mel is bit-identical across pad widths; kl ≤1e-6 rel.
- `--fixed-shape 960` + **double** trace: the winning memory combo —
  cache mean==max==6.44GB (one working set), footprint −32% (6.71 vs
  9.85GB), t_step −12% (2.35 vs 2.67s). With the fused single trace at
  width 960 it ABORTS on the 8GB gate (15.03GB in-step — 3.4× trunk
  activations). Lazy per-step collate required under width knobs
  (pre-collating a fixed-width epoch holds ~8.9GB CPU RAM).
- `mx.compile` (`--compile-mode fwd|step`): both arms run end-to-end,
  step-0 bit-identical to eager (compile is numerically transparent) but
  **25-28% SLOWER with upward drift** on MLX 0.32.2 — not viable for
  formal runs; flags kept for experiment reproduction only.

### Recommended GAN config (validated stacked, blitz)

`--g-forward double --fixed-shape --metal-kernels --save-every 0` —
kernels' +3.7% ADDS on top of fixed+double (order-alt pairs; step-0
mel/gen/kl bit-identical, disc 8.0e-8 / fm 7.7e-7 documented WN row-sum
order).

## 6. Metal kernels (metal-kernels task, `--metal-kernels`, default OFF)

Three fused kernels (mx.custom_function primals + analytic VJPs; a raw
metal_kernel is opaque to mx.grad — gradients silently severed):
`fused_bias_lrelu` (bit-exact, 4.9x isolated), `fused_wn_scale` (≤2.1e-6
rel — GPU serial row-sum vs MLX tree sum), `fused_gate` (≤9.2e-7 rel).
Kernel law: grid=total threads, threadgroup 256, guard elem<N; MLX 0.32.2
quirks documented in `gsovits_mlx/train/metal_kernels.py` docstring
(function-body source, constant-vs-device address space by size,
custom_function vjp signature).

Acceptance: **−3.7% t_step, t=2.31, p<0.05** (6/6 interleaved 10-step
pairs favor ON). The 8% target assumed a dispatch-bound step; the step is
GPU-compute-bound, so elementwise fusion's honest ceiling is ~5%. Loss
gates: step-0 mel/gen/kl bit-identical, disc/fm within documented
tolerance; 50-step bands inside run-to-run scatter; footprint identical.
Next candidate if more is needed: conv+bias+lrelu epilogue via im2col+GEMM
— closed dead on 0.32.2, see §7.

## 7. Blitz verdicts (2026-10-11, honest negatives included)

| item | verdict | evidence |
|------|---------|----------|
| s1 ckpt_groups (GSOVITS_S1_CKPT_GROUPS) | NO WIN (+0.7% noise) | order-alt pairs ×3; loss bit-identical; kept as memory-shape lever |
| CFM ckpt_groups (GSOVITS_S2_CKPT_GROUPS) | NO WIN (+3.0%) + NaN-skip hazard | grouped recompute NaN-skips batch 0 deterministically (3/3 runs; bf16 boundary rounding on a known near-NaN batch); default stays 1 |
| GAN conv→im2col+GEMM / k-slice / Winograd | CLOSED: premise dead on MLX 0.32.2 | mx.conv1d runs 0.73-1.97 TF at real WN shapes (dedicated conv_1d kernels in metallib); im2col 0.42-0.78×, k-slice 0.41-0.67×, Winograd 0.18-0.53× |
| s1 QKV fusion + mask reuse | NO-OP (already implemented; mask build 1.3ms = 0.1%) | code read + timing |
| WN effective-weight caching | DEAD by arithmetic (0.02%/step; mx.grad VJP reuses fwd nodes) | CPU-side graph-build counting |
| s1 eager (no --grad-ckpt) | **+24-25% WIN** (default-fast on this dataset, b4, cap 6144MB) | §1 |
| GAN stacked config | +3.7% kernels validated on fixed+double | §5 |

Grand structural finding: all three trainers are GPU-COMPUTE-bound on
MLX 0.32.2 at real shapes; graph-shape/dispatch levers are exhausted.
Next-tier levers are stack-level (MLX fused GEMM epilogues / bf16 GEMM),
not graph-shape.

## 8. CFM eager backward: DEAD on 16GB (definitive negative)

Single-graph eager (all 22 DiT blocks in one lazy graph + one mx.grad)
froze the machine at step 9 of its first A/B arm: the eval-instant
full-graph cotangent materialization peaks at **18.79GB** (OS-side 0.2s
sampling) while the trainer's per-step sampler saw 5.08GB at the same
step — sub-second transients are invisible to step samplers AND 2s
watchers. Full incident + sampler lesson:
INCIDENTS.md#cfm-eager-1879gb-transient-freeze. `--eager-backward` stays
in the tree as a documented dead-end flag for >24GB machines; anyone
retrying must instrument with `mx.metal.get_peak_memory()` or ≤0.2s OS RSS
sampling FIRST. A 3-step probe passing does NOT clear a single-graph path
(transient scales with bucket width; probe batches were narrow).

Numerics assets from the attempt (kept, real value):
1. **gelu-compile parity finding** (mlx 0.32.2): nn.gelu/gelu_approx are
   @mx.compile-wrapped; traced vs untraced evaluation rounds differently
   (~1e-6 fwd, ~0.3 median grad rel under amplification). Bit-parity
   between traced/untraced paths REQUIRES mx.disable_compile for BOTH.
   Recorded in PARITY_NOTES (s2-training addendum, finding 4 notes the
   regime; switch: `S2V3TrainModel` parity compile-disable).
2. det['base'] fix: both two-pass and eager paths ignored the baked
   two-step base (only CFMTrainingLoss honored it) — 24% loss divergence
   caught by parity tests.
3. CFM draw key pinned to tm.rng (was global random.getrandbits).
4. `tests/test_train_s2v3_eager.py`: comparison harness for ANY future
   CFM backward variant.

Two-pass remains CFM's optimum (bf16 + buckets + two-pass = 7.04s/step).
s1-eager won because its 24-block AR graph fits; CFM's 22-block DiT at
T~300-1000 does not — inverting s1's finding.

## 9. Known issues & documented gaps (standing)

1. **s2 GAN fp16 banned** (task-4): KL exp overflow + scaler collapse ⇒
   fp32 is the official `fp16_run=false` semantics for these checkpoints
   (INCIDENTS.md#fp16-scaler-collapse).
2. **CFM NaN-skip batches** (GradScaler-faithful skip): v4 6/50, v5dev
   5/50, v5turbo 1/50, v3 ~5%. Long-mel batches; skipped, not crashed.
   ~5% of batches still produce non-finite loss/grads at proper lr;
   100/100 clean in isolated hunts; suspected rare draw × post-optimizer
   weight-state interaction, unproven. 2 failing batches archived as npz.
3. **s1 trainer predates the footprint gate** — no per-step assert
   (batch-probe verified the premise false). Its b6 probe showed
   unbounded Metal cache growth (0.55→4.13GB/13 batches; loop never calls
   clear_cache). Mitigation used: `--metal-limit-mb 6144` + external
   watcher. Hardening candidate.
4. **footprint_steps.jsonl t_step field**: CFM's is CUMULATIVE elapsed
   (use deltas); GAN's is per-step; s1 has none (stdout 10-batch windows).
5. **v1/v2 s2D absent on machine** ⇒ D fresh init (official expects s2D);
   v2Pro/Plus load pretrained s2D.
6. **RNG stream differences vs official**: bucket-sampler randperm numpy
   PCG64 (not Philox); grad-accum RNG draws differ from DDP. Documented in
   trainer headers.
7. **memory_report.json zeros** (v3-family runs): FootprintSampler thread
   only started on the non-ckpt trainer path; cosmetic.
8. **chdir trap**: s1 trainer chdirs to CPUFast during dataset bootstrap —
   pass ABSOLUTE `--exp-dir/--init-from`.
9. **Lock order bug fixed 856dcfe** (train_s2): lock must be acquired
   BEFORE `resolve_device` (first launch pinned CPU silently). Canonical
   order: ps guard → acquire → resolve → 8GB gate → refresh/40 steps →
   release.
10. **Crash-path lock leak** (CFM trainer): a pre-loop crash (e.g. the
    bootstrap chdir trap on a relative --exp-dir) exits without releasing
    the lock; stale-lock takeover per protocol. try/finally wrap is a
    follow-up candidate (INCIDENTS.md#lock-leaks).

## 10. Methodology lessons (measurement discipline)

- **Order-alternating interleave is MANDATORY on this box.**
  Candidate-first pairs measured "+25%" for a change that was actually ±0
  (s1 ckpt-groups window-1). Alternate (A,B),(B,A) or use rapid 10-step
  pairs ×6 (ON first); 50-step fresh arms back-to-back where power is
  stable.
- **Power regime dominates absolute times.** AC "finishing charge" tail
  throttles erratically (monotone drift 5.2→6.7s across 13 min);
  low-power-mode ON silently derates 2.4×. Log `pmset -g batt` start+end
  per run; only same-window relative comparisons are trustworthy; never
  overwrite historicals — add the power column
  (INCIDENTS.md#lpm-battery-timing-corruption).
- **Per-step footprint sampling cannot see sub-second eval transients.**
  Single-graph mx.grad materializes the whole cotangent at eval instant;
  step-sampled jsonl understated the peak by >13GB. Gate whole-graph
  regimes with device peak counters or ≤0.2s OS RSS sampling.
- **CPU-side graph-build counting settles "does backward re-derive X"
  for free** (mx.grad VJP node reuse) — no GPU time needed.
- **Isolated op micro-benchmarks under-predict or mis-predict e2e**
  (bf16 GEMM slower isolated, faster e2e; fused kernels 4.9× isolated,
  1.04× e2e). Queued-depth micro ratios fluctuate 0.5-1.7× with launch
  latency; deep dependent queues approximate the in-graph situation.
  Never sum isolated micro-times into a physical floor (BENCH.md
  measurement doctrine).
- **Hash-seed the loaders** for any cross-run loss comparison
  (PYTHONHASHSEED=0; §1).

### Step-1 dispatch census + fusion rationale (from metal_kernels.py)

Per-step CPU-side census (bc_fix_d, v2 b3, 5 steps): transpose 1682 |
sum 1367 | sqrt 1359 | pad 592 | conv1d 540 | conv2d 120 | zeros_like
370 | leaky_relu 316 | maximum 346 | tanh 70 + sigmoid 68 (WN gating) |
mean 56 | astype ~large (uncountable; folded analytically). Ranked
fusable patterns: #1 bias+LeakyReLU tails (D convs + dec resblocks:
transpose+astype+add+leaky = 4 dispatches, 4 full-tensor passes, up to
(3,1024,20480) fp32 = 251MB → ONE kernel, 1 read + 1 write);
#2 WN effective weight g*v/||v|| (~120 convs/fwd across enc_q 17,
flow 4x7, dec 34, D 43; astype+mul+mul+sum+sqrt+div = 6 dispatches on
small weight tensors, recomputed fwd AND bwd → ONE kernel; dispatch-count
kill, not bandwidth); #3 WN gating tanh*sigmoid (340 tanh + 328 sigmoid +
~700 binary ops at trunk width → ONE kernel, 3 reads + 1 write).
