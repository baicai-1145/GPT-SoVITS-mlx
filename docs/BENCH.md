# Benchmarks (gsovits_mlx on MLX / Apple Silicon)

All runs on the reference machine: MacBook Air M4 (16 GB unified memory),
macOS, MLX Metal GPU, one version per process, serial. Reference clip
`ref_zh_3.5s.wav` (3.5 s prompt), target text
"你好，欢迎来到各自的旅程。今天我们聊聊机器学习。" (~8.7 s rendered audio),
**seed 0, fully reproducible** (AR sampling seeds numpy's global RNG since
commit c0d68b5), official AR sampling defaults per version.

Reproduce with:

```bash
python3 tools/bench.py --version v2pro --keep-audio out.wav   # any of:
# v1 | v2 | v2Pro | v2ProPlus | v3 | v4 | v5dev | v5turbo
python3 tools/gemm_bench.py 4096 20                            # GEMM microbench
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

## GEMM microbenchmark

`tools/gemm_bench.py` (square fp16 matmul, MLX Metal GPU, includes launch
overhead; treat as an upper bound):

| size | reps | time | throughput |
|---|---|---|---|
| 1024^3 | 50 | 0.002–0.004 s | 26.1–54.7 TFLOP/s |
| 4096^3 | 20 | 0.056–0.060 s | 45.9–49.5 TFLOP/s |

M4 GPU fp16 is bandwidth/FMA bound around ~50 TFLOP/s peak for these shapes;
real pipelines (attention, dynamic shapes, per-stage launches) land well
below this, which is why the AR decoder (per-token sequential) runs at
~26–37 tok/s in the stage-1 codebase (the seeded table above supersedes
those stage splits).
