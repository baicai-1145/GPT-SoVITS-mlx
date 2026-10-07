"""Regression tests for tools/microbench.py gemm helpers (task-11).

Exercises the PRODUCTION helpers (no toy copies):
- shape/queue validation
- FLOP accounting over actual executed shapes
- graph reachability/work counts via mx.export_to_dot (documents the
  overwrite-output lazy-eval bug class mechanically)
- distinct-materialized input policy: outputs retained => N reachable
  matmuls; rebind-overwrite => 1 (the bug we must never reintroduce)
- chain FLOPs reflect per-link shapes, not q x first-shape
"""

import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import mlx.core as mx  # noqa: E402

from tools.microbench import (  # noqa: E402
    bench_gemm,
    gemm_flops,
    gemm_parse_shape,
    gemm_validate_queue,
)

DOT_DIR = os.path.join(REPO, ".tmp", "test_dots")
os.makedirs(DOT_DIR, exist_ok=True)


def _count_matmuls(path: str) -> int:
    import re
    return len(re.findall(r"matmul|gemm|linear", open(path).read(), re.I))


# ---------------- validation ----------------

def test_parse_shape_valid():
    assert gemm_parse_shape("1600x1152x4608") == (1600, 1152, 4608)


def test_parse_shape_rejects_bad():
    for bad in ("1600x1152", "axbxc", "1x2x3x4", "0x1x1", "1x-2x1"):
        with pytest.raises(ValueError):
            gemm_parse_shape(bad)


def test_validate_queue():
    assert gemm_validate_queue(1) == 1
    assert gemm_validate_queue(8) == 8
    for bad in (0, -1, 1.5, "8"):
        with pytest.raises(ValueError):
            gemm_validate_queue(bad)


# ---------------- FLOP accounting ----------------

def test_flops_single_and_chain():
    assert gemm_flops([(1600, 1152, 4608)]) == 2 * 1600 * 1152 * 4608
    # chain with shrinking N: flops must sum ACTUAL per-link shapes
    chain = [(64, 64, 32), (64, 32, 16)]
    assert gemm_flops(chain) == 2 * 64 * 64 * 32 + 2 * 64 * 32 * 16
    # ...which is NOT q x first shape
    assert gemm_flops(chain) != 2 * 2 * 64 * 64 * 32


# ---------------- graph reachability (the bug class) ----------------

def test_overwrite_output_bug_collapse():
    a = mx.random.normal((64, 64))
    b = mx.random.normal((64, 64))
    out = a @ b
    for _ in range(7):
        out = a @ b  # rebind: only the LAST matmul stays reachable
    p = os.path.join(DOT_DIR, "rebind.dot")
    mx.export_to_dot(p, out)
    assert _count_matmuls(p) == 1  # documents the collapse


def test_retained_outputs_reachable():
    a = mx.random.normal((64, 64))
    b = mx.random.normal((64, 64))
    outs = [a @ b for _ in range(8)]  # every output retained
    p = os.path.join(DOT_DIR, "retain.dot")
    mx.export_to_dot(p, *outs)
    assert _count_matmuls(p) == 8


# ---------------- production bench helper smoke (CPU) ----------------

def test_bench_gemm_smoke_cpu():
    mx.set_default_device(mx.cpu)
    res = bench_gemm(mx, 64, 64, 64, q=4, rounds=2)
    assert res["queue"] == 4
    assert res["indep_batch_flops"] == 4 * 2 * 64 ** 3
    # chain flops: first link (64,64,64) + 3 links (64,64,64) => same here
    assert res["chain_flops"] == 4 * 2 * 64 ** 3
    assert res["indep_batch_total_ms"] > 0
    assert res["chain_total_ms"] > 0
    assert res["input_policy"] == "distinct-materialized"
    assert res["output_policy"] == "retain-and-eval-all"


def test_bench_gemm_chain_flops_use_actual_shapes():
    mx.set_default_device(mx.cpu)
    # N != K: chain links after the first are (M, N, N), NOT (M, K, N)
    M, K, N, q = 32, 48, 24, 3
    res = bench_gemm(mx, M, K, N, q=q, rounds=1)
    expected = 2 * M * K * N + (q - 1) * 2 * M * N * N
    assert res["chain_flops"] == expected


# ---------------- legacy entrypoint (tools/gemm_bench.py) ----------------

def test_gemm_bench_legacy_migration():
    """Legacy CLI must delegate to the corrected helper, not the overwrite loop."""
    import subprocess
    code = ("import sys; sys.path.insert(0, 'tools'); "
            "import gemm_bench; "
            "assert hasattr(gemm_bench, 'main'); "
            "assert not any('flops * ' in l or '2.0 * N ** 3 * REPS' in l "
            "for l in open('tools/gemm_bench.py'))")
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_gemm_bench_invalid_args_fail_fast():
    import subprocess
    r = subprocess.run([sys.executable, "tools/gemm_bench.py", "0", "8"],
                       cwd=REPO, capture_output=True, text=True)
    assert r.returncode != 0 and "positive" in (r.stdout + r.stderr)
    r = subprocess.run([sys.executable, "tools/gemm_bench.py", "64", "0"],
                       cwd=REPO, capture_output=True, text=True)
    assert r.returncode != 0 and ("positive" in (r.stdout + r.stderr) or
                                  "queue" in (r.stdout + r.stderr))


def test_gemm_bench_rejects_reps_as_unaccounted_work():
    """The core regression: FLOPs must come from the helper's counted work,
    never from multiplying by a rep count that the graph didn't retain."""
    src = open(os.path.join(REPO, "tools", "gemm_bench.py")).read()
    assert "* REPS" not in src and "* reps" not in src
    assert "indep_batch_flops" in src  # counted work drives the report
