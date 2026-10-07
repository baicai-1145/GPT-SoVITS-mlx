"""GEMM microbenchmark: MLX fp16 matmul latency/throughput (Metal GPU).

MIGRATED (task-11): the legacy loop overwrote its output (`c = a @ b` in a
REPS loop with a single trailing mx.eval), so only ONE matmul stayed
reachable in the lazy graph while FLOPs were multiplied by REPS — an
inflated-throughput generator. This entrypoint now delegates to the
corrected production helper in tools/microbench.py (distinct materialized
inputs, every timed output retained and evaluated, work counted exactly).

Usage (unchanged invocation):
  python3 tools/gemm_bench.py [N] [reps]

For richer modes (independent batches vs dependent chains, stated queue
depth, arbitrary MxKxN shapes) use:
  python3 tools/microbench.py gemm --shape MxKxN --queue q
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from microbench import bench_gemm, gemm_parse_shape, gemm_validate_queue


def main() -> None:
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    # The old CLI conflated "reps" with "work". Treat reps as the queue
    # depth q (counted work) and fail fast on values that would silently
    # regress to the overwrite bug semantics.
    if N <= 0:
        sys.exit("gemm_bench: N must be positive")
    q = gemm_validate_queue(reps)
    shape = gemm_parse_shape(f"{N}x{N}x{N}")

    try:
        import mlx.core as mx
    except ImportError:
        sys.exit("gemm_bench: mlx is required")

    from microbench import gpu_lock_guard
    gpu_lock_guard(False)

    res = bench_gemm(mx, *shape, q=q, rounds=8)
    f = res["single_op_synced_ms"] * 1e-3
    tot = res["indep_batch_total_ms"] * 1e-3
    fl = res["indep_batch_flops"]
    print(f"mlx fp16 matmul {N}^3:")
    print(f"  single-op synced latency : {res['single_op_synced_ms']:.3f} ms "
          f"(end-to-end incl. dispatch+sync)")
    print(f"  independent batch q={q}   : {tot:.3f} s total, "
          f"{fl / tot / 1e12:.1f} TFLOP/s achieved (work={fl / 1e9:.1f} GFLOP, "
          f"outputs retained+evaluated)")
    print(f"  dependent chain q={q}     : {res['chain_total_ms']:.3f} s total, "
          f"{res['chain_flops'] / (res['chain_total_ms'] * 1e-3) / 1e12:.1f} TFLOP/s "
          f"(per-link FLOPs counted)")
    print("legacy note: the old REP-multiplied GFLOP/s figure was invalid "
          "(one reachable matmul); see tools/microbench.py gemm for full modes")


if __name__ == "__main__":
    main()
