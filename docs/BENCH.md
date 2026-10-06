# Benchmarks (gsovits_mlx on MLX / Apple Silicon)

All runs on the reference machine: MacBook Air M4 (16 GB unified memory),
macOS, MLX Metal GPU, one version per process, serial. Reference clip
`ref_zh_3.5s.wav` (3.5 s prompt), target text
"你好，欢迎来到各自的旅程。今天我们聊聊机器学习。" (~8.7 s rendered audio),
seed 0, official AR sampling defaults per version.

Reproduce with:

```bash
python3 tools/bench.py --version v2pro --keep-audio out.wav   # any of:
# v1 | v2 | v2Pro | v2ProPlus | v3 | v4 | v5dev | v5turbo
python3 tools/gemm_bench.py 4096 20                            # GEMM microbench
```

## End-to-end (stage times in seconds)

| version | frontend | prompt_codes | sv_emb | refer | ar (tok/s) | sovits(+voc) | total | audio | RTF | peak RSS |
|---|---|---|---|---|---|---|---|---|---|---|
| v1 | 11.4 | 4.9 | — | 0.0 | 3.8 (36.6) | 0.9 | 21.9 | 9.56 s | 2.29 | 2.7 GB |
| v2 | 13.5 | 4.6 | — | 0.0 | 4.6 (26.3) | 1.0 | 25.1 | 8.84 s | 2.84 | 3.3 GB |
| v2Pro | 11.8 | 7.6 | 11.2 | 0.0 | 3.9 (29.7) | 0.8 | 29.2 | 8.68 s | 3.36 | 4.6 GB |
| v2ProPlus | 11.5 | 9.2 | 13.0 | 0.0 | 3.8 (31.8) | 1.6 | 31.4 | 8.88 s | 3.53 | 4.7 GB |
| v3 | 11.6 | 26.7 | — | 0.1 | 3.9 (29.9) | 33.1 | 76.5 | 8.67 s | 8.83 | 4.0 GB |
| v4 | 11.8 | 28.3 | — | 0.1 | 4.7 (27.4) | 21.0 | 67.0 | 9.16 s | 7.32 | 4.0 GB |
| v5dev | 11.6 | 29.2 | — | 0.0 | 3.5 (32.8) | 34.6 | 80.3 | 8.68 s | 9.25 | 4.2 GB |
| v5turbo | 12.6 | 31.6 | — | 0.1 | 3.5 (35.7) | 4.2 | 53.3 | 9.04 s | 5.90 | 3.8 GB |

Notes:
- RTF = total_e2e_s / audio_s (>1 = slower than realtime). The front-end
  (text cleaner + BERT on CPU/GPU) and HuBERT prompt coding dominate short
  utterances; AR decode and the CFM+vocoder stack dominate long ones.
- v3–v5 `sovits+vocoder` includes the 32-step (4 for turbo) CFM DiT rollout
  plus the 48 kHz Generator/BigVGAN vocoder.
- Peak RSS is the child-process high-water mark (macOS ru_maxrss, KB) —
  all versions stay far under the 12 GB budget.
- v5turbo renders the same audio with 4 CFM steps: 34.6 -> 4.2 s in the
  SoVITS stage vs v5dev.

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
~26–37 tok/s.
