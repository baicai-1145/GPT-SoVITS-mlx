"""GEMM microbenchmark: MLX fp16 1024^3 matmul throughput (Metal GPU).

Timing includes only GPU work after an explicit mx.eval sync; run with a
larger N or more reps if the numbers look compute-bound-implausible (startup
effects). 1024^3 is small enough that M-series AMX/GPU fixed-function paths
report >>100 TFLOP/s — treat as an upper bound, not sustainable throughput.

Usage: python3 tools/gemm_bench.py [N] [reps]
"""
import sys
import time

import mlx.core as mx

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 50
WARMUP = 10

a = mx.random.normal((N, N)).astype(mx.float16)
b = mx.random.normal((N, N)).astype(mx.float16)
for _ in range(WARMUP):
    c = a @ b
mx.eval(c)

t0 = time.perf_counter()
for _ in range(REPS):
    c = a @ b
mx.eval(c)
dt = time.perf_counter() - t0

flops = 2.0 * N ** 3 * REPS
print(f"mlx fp16 matmul {N}^3 x {REPS}: {dt:.3f}s -> {flops / dt / 1e9:.1f} GFLOP/s")
