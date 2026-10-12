# Incident Register (machine-safety & measurement incidents)

Chronological register of incidents from the GPT-SoVITS-mlx port + training
campaign (2026-10-07 → 2026-10-11). Each entry: date, symptom, root cause,
standing rule. Raw evidence stays in `.tmp/` RUN_PLAN.md files and herdr
reports; this file is the durable distillation. Machine context: MacBook
Air M4 16GB, 12GB memory red line, single GPU process law.

The companion documents: `docs/PARITY_NOTES.md` (official-vs-MLX semantics),
`docs/TRAINING.md` (configs/acceptance), `AGENTS.md` (standing rules).

---

## 2026-10-09 ~07:40 — 3-way concurrent-trainer OOM + watchdog reboot (freeze #1)

- **Symptom**: machine watchdog-rebooted from OOM; a v3 smoke run died at
  step 4/200 (loss.jsonl 4 lines).
- **Root cause**: three teammates each held a *worktree-local*
  `.tmp/gpu.lock.d`; the old lock resolution walked up from CWD and
  accepted the first lock found, so mutual exclusion silently broke —
  three concurrent MLX trainers on a 16GB machine.
- **Fix**: `gsovits_mlx/gpu_lock.py` rewritten (main f9185a6): single
  canonical lock `<main-repo-root>/.tmp/gpu.lock.d`, resolved from the
  package location (topmost .git-bearing ancestor), cwd-independent;
  worktree-local locks are never honored nor created.
- **Standing rule**: AGENTS.md GPU-lock law — acquire via
  `acquire_lock(owner)` before any Metal work, ps-aux precheck, refresh
  every 15 min, ONE GPU process machine-wide.

## 2026-10-09 (afternoon) — s2-CFM GPU smokes OOM the machine repeatedly (freezes #2-#4)

<a id="s2-cfm-oom-series"></a>

- **Symptom**: multiple machine freezes/kills during early s2-CFM (task-5)
  GPU attempts; trainer jetsam-killed at step ~3 "due to low-swap" in one
  instance (18:39); first 200-step smoke attempt died at step 4/200.
- **Root causes (stacked)**:
  1. The fused-graph LoRA forward held ALL 22 DiT blocks' activation
     graphs (T=952 worst bucket) plus the frozen trunk inside the
     autograd graph — dead weight the official code freezes via
     set_no_grad.
  2. fp16 DiT *backward* overflows (per-block grads grow monotonically,
     exceed 65504 from block ~8 while forward stays finite) — the official
     `autocast(fp16_run=True)` regime is not reproducible on MLX 0.32.2.
  3. Runs launched before any memory probe existed (process failure by the
     worker, acknowledged in its GPU re-authorization plan).
- **Fixes**: two-pass checkpointed DiT backward (retain only block inputs;
     one block graph alive at a time, 0683093); frozen-trunk
     stop_gradient cuts (71a0e65); fp32 DiT masters; per-step 8GB
     footprint abort gate in-trainer; mem_probe tool + written
     plan-before-GPU protocol.
- **Standing rule**: fp16 DiT training is banned (PARITY_NOTES s2-training
  addendum, finding 3); every new GPU regime gets a memory probe before a
  long run; the in-trainer footprint gate is the hard law.

## 2026-10-09 (evening) — lock-order bug: train_s2 first launch silently CPU-pinned

- **Symptom**: first `tools/train_s2.py` launch ran on CPU silently
  (caught only via the `[device] cpu` log line + resolve semantics).
- **Root cause**: `resolve_device` was called BEFORE lock acquisition.
- **Fix**: 856dcfe — canonical order: ps guard → acquire → resolve → 8GB
  gate → refresh/40 steps → release.
- **Standing rule**: acquire the lock before any device/env resolution
  that can pin the process.

## 2026-10-09/10 — fp16 GradScaler collapse (s2 GAN v2 smoke)

<a id="fp16-scaler-collapse"></a>

- **Symptom**: KL loss exp(-2·logs_p) overflows fp16 range → inf → the
  shared GradScaler's scale collapses (1e-56) → training dead.
- **Root cause**: official `fp16_run=True` autocast keeps fp32 loss heads
  and torch's grad-safe kernels; MLX fp16 GEMM backward has no such
  guarantee and the KL head overflowed in fp16.
- **Fix/decision**: fp32 is the default for the s2 GAN trainer on these
  checkpoints (= official `fp16_run=false` config); bf16 arm keeps fp32
  loss heads (fp32 exponent range cannot hit this mode) —
  `tools/train_s2.py` Precision section.
- **Standing rule**: any low-precision arm must gate the KL head
  finiteness explicitly (replay gate, logs_p∈[−30,−5]).

## 2026-10-10 20:12 — disk-full from resume checkpoints (ENOSPC #1)

- **Symptom**: internal disk hit 100% (118MB free); a completed 50-step
  run died writing its final summary; another killed pre-start. An
  interleaved arm elsewhere hit ENOSPC t_step spikes (10.75/8.49s).
- **Root cause**: 11 × 931MB `resume/` checkpoint dirs accumulated;
  `--save-every 0` did not skip the FINAL resume save. (An earlier
  near-miss on 10-09: `/` at 363MB free, pytest tmp writes failing —
  reclaimed ~1.3-2.0GB of .tmp duplicates.)
- **Fix**: da4a400 — `--save-every 0` now skips the final resume save
  too; `df` check before every run; purge resume dirs after acceptance
  runs.
- **Standing rule**: acceptance runs pass `--save-every 0`; watch disk;
  exports harvested then deleted. ENOSPC once killed a report mid-write —
  append run logs after every run, not at the end.

## 2026-10-10 19:26-19:55 — LPM-on-battery timing contamination

<a id="lpm-battery-timing-corruption"></a>

- **Symptom**: every absolute t_step/wall number from an A/B session was
  power-contaminated (fp32-single 4.94 s/step "measured"); bf16 looked
  +14.8% slower; b4 throughput looked +19%.
- **Root cause**: all timing runs executed on battery with
  lowpowermode=1 (charge 69-71%); LPM silently derates ~2.4× (LPM-off
  battery rerun: 2.09 s/step).
- **Corrections**: bf16 speed-neutral at b3 (0.0%); b4 value is memory
  headroom (+1% samples/s); the relative single-vs-double 1.27× claim
  survives (same-window, both arms equally derated). Historical numbers
  kept with power-state annotation, never overwritten.
- **Standing rule**: log `pmset -g batt` start+end for every GPU timing
  run; only same-window relative comparisons are trustworthy; AC
  "finishing charge" tail is a separate erratic regime (monotone drift
  5.2→6.7s across 13 min) — interleave.

## 2026-10-11 ~07:23-07:27 — CFM eager single-graph backward: 18.79GB eval transient machine freeze (freeze #5)

<a id="cfm-eager-1879gb-transient-freeze"></a>

- **Symptom**: gate-4 A/B arm p1_eager froze the machine at step 9
  (SIGKILL exit 137, load avg ~50, user-observed full-memory freeze);
  subsequent arms all refused on lock-held (the killed trainer leaked the
  lock).
- **Root cause**: mx.eval of one big value_and_grad realizes the whole
  22-block activation + cotangent graph transiently. TRUE PEAK 18.79GB
  (16.70-17.24GB sustained, OS-side 0.2s RSS sampling) vs 5.08GB seen by
  the trainer's per-step sampler at the same step — the transient is
  invisible to step samplers AND 2s external watchers. The 3-step gate-3
  probe passed only because its batches missed the widest buckets
  (transient scales with bucket width). 18.79GB exceeds the 12GB machine
  red line by 6.8GB; no batch-size regime closes the gap (b1 still holds
  ~⅓ transient on top of ~7GB steady state).
- **Disposition**: eager CFM backward is DEAD on 16GB (definitive
  negative, lead-confirmed); `--eager-backward` kept as documented
  dead-end flag for >24GB machines. Gate 1+2 (CPU) had passed bit-exact —
  numerics were never the issue; memory instrumentation was.
- **Standing rules**: (1) for any whole-graph mx.grad regime, gate with
  `mx.metal.get_peak_memory()` (device counter) or ≤0.2s OS-side RSS
  sampling — per-step jsonl + 2s watchers both miss sub-second eval
  transients; (2) a passing short probe does NOT clear a single-graph
  memory path; (3) crash paths must release the gpu.lock (see lock
  leaks).

## Lock leaks (recurring, 2026-10-10/11)

<a id="lock-leaks"></a>

- **Symptom**: crashed/killed trainers leave `.tmp/gpu.lock.d` held;
  subsequent runs refuse ("lock held") until stale-lock takeover (>15 min
  mtime, owner dead). Hit by batch-probe (CFM probe #1 crashed pre-loop on
  the bootstrap-chdir trap), cfm-eager (freeze + 3 probe attempts that
  crashed after acquiring).
- **Root cause**: trainer bodies lack try/finally release around the
  crash path; some crash-after-acquire paths (bootstrap/g2pw issues).
- **Standing rule**: takeover only when mtime >15 min and owner dead
  (protocol held in practice); follow-up candidate: wrap CFM trainer body
  in try/finally release. Related trap: pass ABSOLUTE `--exp-dir` (the
  bootstrap chdir makes relative paths crash pre-loop).

## Measurement-corruption near-misses (standing rules, no damage)

- **PYTHONHASHSEED nondeterminism** (proven, 2026-10-10):
  <a id="pythonhashseed-nondeterminism"></a> s2 loaders build
  the file list via `list(set(...) & ...)` — set iteration order is
  hash-randomized per process → different batch composition at the same
  step across runs (3 construction probes → 2 distinct batch0 md5;
  PYTHONHASHSEED=0 ×3 → identical). Official s2_train.py has the same
  line (hidden at DDP scale). Rule: PYTHONHASHSEED=0 for all cross-run
  loss comparisons; `sorted()` fix is a documented follow-up candidate.
- **Fresh-D init drew from the unseeded global stream** → D weights
  non-reproducible run-to-run; fixed by deff619 (mx.random.seed before
  model build = official torch.manual_seed parity). MLX keyed RNG
  (key/split) is pure across processes and immune to global-stream draws
  (2-proc probes bit-equal) — the fix matters only for init-time draws.
- **One-GPU-process violation (self-reported, metal-kernels task)**: a
  pytest light-suite launch overlapped a live trainer (torch GPU test
  concurrent with trainer); both killed, acceptance rerun clean. Rule: NO
  pytest while any trainer run is live. A separate 20ms GPU micro-probe
  without the lock was also self-reported (cfm-eager, pre-work).
- **3-step-probe false clearance**: covered by the 18.79GB entry above —
  short probes with narrow batches cannot clear width-scaled transients.
- **Jetsam/low-swap kill at step ~3** (2026-10-09 18:39): covered by the
  s2-CFM freezes entry (frozen-trunk graph weight was the driver).

---

### Freeze-event count note

Five machine-freeze/reboot events total, 2026-10-09 → 2026-10-11:
#1 the 3-way lock OOM reboot; #2-#4 the early s2-CFM OOM/jetsam kills
(fused-graph + frozen-trunk + pre-probe launches); #5 the CFM eager
18.79GB transient. The blitz GPU-law header's "violations killed the box
5x" refers to this series. All five produced standing rules now encoded
in AGENTS.md / gpu_lock.py / trainer gates.
