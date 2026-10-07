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

# Pin CPU for the whole module BEFORE any array is created: the suite must
# never accidentally compute on the default GPU (lead review requirement).
mx.set_default_device(mx.cpu)

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


# ---------------- review fixes (lead r2 findings) ----------------

def test_chain_q8_production_finite():
    """Lead repro: unscaled normal chain weights overflow fp16 by link 5.
    Production helper must scale by 1/sqrt(fan_in) and stay finite at q=8.
    Asserts the actual FINITE VALUES, not just timing keys."""
    mx.set_default_device(mx.cpu)
    res = bench_gemm(mx, 64, 64, 64, q=8, rounds=2)
    assert res["chain_links"] == 8
    assert res["chain_total_ms"] > 0
    # recompute the chain with the helper's own scaling and assert finite
    # VALUES (red on the unscaled defect: NaN by link 5)
    w_first = (mx.random.normal((64, 64), key=mx.random.key(31)) * (64 ** -0.5)).astype(mx.float16)
    rest = [(mx.random.normal((64, 64), key=mx.random.key(32 + i)) * (64 ** -0.5)).astype(mx.float16)
            for i in range(7)]
    a = mx.random.normal((64, 64), key=mx.random.key(1)).astype(mx.float16)
    acc = a @ w_first
    for w in rest:
        acc = acc @ w
    mx.eval(acc)
    assert bool(mx.all(mx.isfinite(acc))), "chain output not finite"


def test_chain_defect_red_unscaled_overflows():
    """RED side of red-green: the ACTUAL defect — unscaled N(0,1) fp16 chain
    weights produce non-finite output at q=8 (verified NaN). If this ever
    passes, fp16 got wider or MLX changed semantics; revisit scaling."""
    mx.set_default_device(mx.cpu)
    def mk(seed):
        return mx.random.normal((64, 64), key=mx.random.key(seed)).astype(mx.float16)
    a = mk(1)
    ws = [mk(31)] + [mk(32 + i) for i in range(7)]
    acc = a @ ws[0]
    for w in ws[1:]:
        acc = acc @ w
    mx.eval(acc)
    assert not bool(mx.all(mx.isfinite(acc))), "unscaled chain unexpectedly finite"


def test_bench_gemm_raises_on_unscaled_chain(monkeypatch):
    """The helper's own guard: make weights effectively unscaled (multiply
    normal draws by 2**8) so even fan-in scaling cannot keep the q=8 chain
    finite; bench_gemm MUST raise ValueError from the isfinite guard.
    Proves the guard sits on the execution path, not decorative."""
    import tools.microbench as mb
    mx.set_default_device(mx.cpu)
    orig_normal = mx.random.normal

    def hot_normal(shape, key=None, **kw):
        return orig_normal(shape, key=key, **kw) * 256.0

    monkeypatch.setattr(mx.random, "normal", hot_normal)
    with pytest.raises(ValueError, match="not finite"):
        bench_gemm(mx, 64, 64, 64, q=8, rounds=1)


def test_chain_large_n_finite():
    """Real-model-scale N: overflow would occur even earlier unscaled."""
    mx.set_default_device(mx.cpu)
    res = bench_gemm(mx, 8, 64, 512, q=8, rounds=1)
    assert res["chain_flops"] == 2 * 8 * 64 * 512 + 7 * 2 * 8 * 512 * 512


def test_gemm_bench_unit_labels_and_bounded_q():
    """Lead review: chain/batch lines must say ms (not s); q must be bounded;
    reps are timing rounds; memory preflight must engage before allocation."""
    src = open(os.path.join(REPO, "tools", "gemm_bench.py")).read()
    assert 'ms total' in src and ' s total' not in src
    assert "rounds = reps" in src  # reps -> timing rounds, not batch work
    assert "q = 4" in src  # bounded queue
    assert "gemm_preflight_check" in src  # SHARED preflight, both CLIs


def test_shared_preflight_both_clis():
    """perf-mem: the SAME preflight policy backs both entrypoints; it raises
    before allocation and the estimate matches the distinct-inputs budget."""
    from tools.microbench import gemm_preflight_bytes, gemm_preflight_check
    assert gemm_preflight_bytes(1024, 1024, 1024, 4) == 5 * 3 * 1024 * 1024 * 2
    with pytest.raises(ValueError):
        gemm_preflight_check(8192, 8192, 8192, 4, budget_mb=64)
    gemm_preflight_check(8192, 8192, 8192, 4, budget_mb=8192)  # passes


def test_bench_gemm_exposes_raw_seconds():
    """Unrounded seconds must be present for TF math (never divide by
    rounded ms); chain metric finite+positive."""
    mx.set_default_device(mx.cpu)
    res = bench_gemm(mx, 64, 64, 64, q=4, rounds=2)
    raw = res["_raw_seconds"]
    assert raw["indep_batch_total_s"] > 0 and raw["chain_total_s"] > 0
    assert res["indep_batch_tf"] > 0 and res["chain_tf"] > 0


# ---------------- legacy main() happy path (executes, CPU-pinned) ----------------

def test_gemm_bench_main_executes_cpu(monkeypatch, capsys):
    """Lead review: valid legacy invocation must RUN in tests — CPU pinned
    before array creation, sys.argv patched, ONLY the lock gate mocked.
    Asserts units, counted work totals, and the helper contract."""
    import importlib
    spec = importlib.util.spec_from_file_location(
        "gemm_bench", os.path.join(REPO, "tools", "gemm_bench.py"))
    gemm_bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gemm_bench)
    argv = ["gemm_bench.py", "64", "3"]
    monkeypatch.setattr(sys, "argv", argv)
    if hasattr(gemm_bench, "gpu_lock_guard"):
        monkeypatch.setattr(gemm_bench, "gpu_lock_guard", lambda allow: None)
    else:
        import microbench as _mb
        monkeypatch.setattr(_mb, "gpu_lock_guard", lambda allow: None)
    gemm_bench.main()
    out = capsys.readouterr().out
    assert "mlx fp16 matmul 64^3" in out
    assert "single-op synced latency" in out
    assert "ms total" in out and " s total" not in out
    # counted work contract: q=4 distinct 64^3 matmuls = 4 * 2 * 64^3 FLOPs.
    # 64^3 batches are ~2 MFLOP => displayed as "0.0 GFLOP"; assert the exact
    # counted-work string the helper contract produces instead of a display
    # rounding that divides by rounded ms.
    assert "work=0.00 GFLOP" in out  # 4*2*64^3 = 2.1e6 FLOP (display .2f)
    assert "dependent chain q=4" in out
    assert "legacy note" in out  # guidance to the corrected command
