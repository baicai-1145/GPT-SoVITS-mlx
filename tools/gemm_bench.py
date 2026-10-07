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

    # Lead review fix: reps used to map onto the batch queue depth q, which
    # made the legacy default (50) allocate unbounded multi-GB batches at
    # large N. reps now means TIMING ROUNDS; q is bounded and preflighted.
    if N <= 0:
        sys.exit("gemm_bench: N must be positive")
    rounds = reps if 1 <= reps <= 512 else sys.exit(
        "gemm_bench: reps must be a positive number of timing rounds in 1..512")
    q = 4  # bounded queue depth; use tools/microbench.py gemm --queue for more
    shape = gemm_parse_shape(f"{N}x{N}x{N}")

    # shared memory preflight (same policy as microbench gemm mode)
    from microbench import gemm_preflight_check
    try:
        gemm_preflight_check(N, N, N, q)
    except ValueError as e:
        sys.exit(f"gemm_bench: {e}")

    try:
        import mlx.core as mx
    except ImportError:
        sys.exit("gemm_bench: mlx is required")

    from microbench import gpu_lock_guard
    gpu_lock_guard(False)

    # Keep RAW unrounded seconds: TF divides by the unrounded measurement,
    # never by a display-rounded ms (perf-mem finding).
    res = bench_gemm(mx, *shape, q=q, rounds=rounds)
    raw = res.pop("_raw_seconds", {}) or {}
    tot_s = raw.get("indep_batch_total_s")
    chain_s = raw.get("chain_total_s")
    fl = res["indep_batch_flops"]
    print(f"mlx fp16 matmul {N}^3:")
    print(f"  single-op synced latency : {res['single_op_synced_ms']:.3f} ms "
          f"(end-to-end incl. dispatch+sync)")
    indep_tf = fl / tot_s / 1e12 if tot_s else 0.0
    print(f"  independent batch q={q}   : {res['indep_batch_total_ms']:.3f} ms total, "
          f"{indep_tf:.3f} TFLOP/s achieved (work={fl / 1e9:.2f} GFLOP, "
          f"outputs retained+evaluated)")
    chain_tf = res["chain_flops"] / chain_s / 1e12 if chain_s else 0.0
    assert res["chain_total_ms"] > 0 and chain_tf > 0, "chain metric not finite+positive"
    print(f"  dependent chain q={q}     : {res['chain_total_ms']:.3f} ms total, "
          f"{chain_tf:.3f} TFLOP/s (per-link FLOPs counted)")
    print("legacy note: the old REP-multiplied GFLOP/s figure was invalid "
          "(one reachable matmul); see tools/microbench.py gemm for full modes")


if __name__ == "__main__":
    main()
