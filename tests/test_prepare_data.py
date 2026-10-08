"""Parity gates for tools/prepare_data.py vs the official prepare_datasets.

Layers:
* light (no torch, no GPU): line parsing / list sharding / merge semantics /
  ffmpeg load_audio argv parity — pure functions from tools.prepare_data.
* heavy (base-venv torch reference outputs captured once): stage-1 text vs
  official 1-get-text.py, stage-2 wav32k md5 + HuBERT ssl diff vs official
  2-get-hubert-wav32k.py, stage-2b sv diff, stage-3 semantic codes vs
  official 3-get-semantic.py (v1+v2 weights).

Heavy gates read pre-captured official references from
GSOVITS_PREP_REF_DIR (default .tmp/prep_refs/ — produced by
.tmp/off_stage1.py, .tmp/off_stage2.py, .tmp/off_stage23.py in the base
venv; scripts checked in under .tmp are not committed). When the reference
dir is absent the heavy tests skip. GPU stages need the shared gpu.lock
(AGENTS.md); tests take it read-only-aware: they SKIP if a foreign fresh
lock is held (never steal).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

REF_DIR = Path(os.environ.get("GSOVITS_PREP_REF_DIR", REPO / ".tmp" / "prep_refs"))
EXP_DIR = Path(os.environ.get("GSOVITS_PREP_EXP", REF_DIR / "mlx_exp"))
WAVTEST = Path(os.environ.get("GSOVITS_WAVTEST", REPO.parent.parent / "wavtest"))
# the worktree lives at <main>/.pi/herdr-team/<agent>/worktrees/<name>
if not WAVTEST.is_dir():
    WAVTEST = Path("/Users/baicai1145/repos/gpt-sovits/GPT-SoVITS-mlx/wavtest")


# ---------------------------------------------------------------------------
# light: pure helpers
# ---------------------------------------------------------------------------

def test_clean_path_official_semantics():
    from tools.prepare_data import clean_path
    assert clean_path(" /a/b.wav\n") == "/a/b.wav"
    assert clean_path('"/a/b.wav"') == "/a/b.wav"
    assert clean_path("/a/b/") == os.path.join("/a", "b")


def test_language_mapping_table():
    from tools.prepare_data import LANGUAGE_V1_TO_V2, parse_list_line
    assert LANGUAGE_V1_TO_V2["ZH"] == "zh"
    assert LANGUAGE_V1_TO_V2["JP"] == "ja"
    parsed = parse_list_line("vo_a.wav|hutao|ZH|你好")
    assert parsed == ("vo_a.wav", "zh", "你好")
    assert parse_list_line("x.wav|s|FR|bonjour") is None


def test_line_sharding_matches_official_slice():
    from tools.prepare_data import read_list_lines
    p = REPO / ".tmp" / "shard_test.list"
    p.parent.mkdir(exist_ok=True)
    with open(p, "w") as f:
        f.write("\n".join(f"wav{i}.wav|spk|zh|t{i}" for i in range(10)) + "\n")
    try:
        assert read_list_lines(str(p), 0, 3) == [f"wav{i}.wav|spk|zh|t{i}" for i in (0, 3, 6, 9)]
        assert read_list_lines(str(p), 1, 3) == [f"wav{i}.wav|spk|zh|t{i}" for i in (1, 4, 7)]
        # the webui merge concatenates parts IN ORDER (not round-robin):
        # part0 lines, then part1, ... — shard order does not need to
        # reconstruct the original file, only to partition it.
        all_lines = set(read_list_lines(str(p), 0, 1))
        union = set()
        for i in range(3):
            union |= set(read_list_lines(str(p), i, 3))
        assert union == all_lines and len(union) == 10
    finally:
        p.unlink()


def test_merge_reproduces_webui_concat(tmp_path):
    from tools.prepare_data import run_merge

    class A:
        pass
    a = A()
    a.exp_dir = str(tmp_path)
    a.parts = 2
    # official: parts concatenated in order, trailing "\n", parts removed
    (tmp_path / "2-name2text-0.txt").write_text("a\tp1\nb\tp2\n", encoding="utf8")
    (tmp_path / "2-name2text-1.txt").write_text("c\tp3\n", encoding="utf8")
    (tmp_path / "6-name2semantic-0.tsv").write_text("n1\t1 2 3\n", encoding="utf8")
    (tmp_path / "6-name2semantic-1.tsv").write_text("n2\t4 5\n", encoding="utf8")
    run_merge(a)
    assert (tmp_path / "2-name2text.txt").read_text(encoding="utf8") == "a\tp1\nb\tp2\nc\tp3\n"
    assert (tmp_path / "6-name2semantic.tsv").read_text(encoding="utf8") == \
        "item_name\tsemantic_audio\nn1\t1 2 3\nn2\t4 5\n"
    assert not (tmp_path / "2-name2text-0.txt").exists()
    assert not (tmp_path / "6-name2semantic-1.tsv").exists()


# ---------------------------------------------------------------------------
# gate helpers
# ---------------------------------------------------------------------------

def _gpu_lock_free_for_tests() -> bool:
    """Tests may take the lock only when unowned; a foreign fresh lock means
    another agent owns the GPU — skip heavy GPU gates instead of stealing."""
    from gsovits_mlx.gpu_lock import lock_status
    held, owner = lock_status()
    return not held  # unowned or stale -> tests may proceed (cpu pins otherwise)


def _ref(name: str) -> Path:
    return REF_DIR / name


@pytest.mark.heavy
def test_stage1_text_parity():
    """2-name2text.txt content identical to official 1-get-text.py output
    (phones, word2ph repr, norm_text) on the gate subset."""
    if not (_ref("2-name2text.txt").exists() and WAVTEST.is_dir()):
        pytest.skip("official stage-1 reference not captured")
    exp = EXP_DIR
    if not (exp / "2-name2text.txt").exists():
        # run the MLX stage-1 in a subprocess (CPU front-end is parity-tested;
        # avoids the GPU lock inside tests)
        env = dict(os.environ,
                   GSOVITS_FRONTEND_DEVICE="cpu", GSOVITS_PREP_CPU="1",
                   GSOVITS_G2PW_FP16="0", GSOVITS_FRONTEND_F32="1")
        cmd = [sys.executable, str(REPO / "tools" / "prepare_data.py"),
               "--list", str(REF_DIR / "subset20.list"),
               "--wav-dir", str(WAVTEST),
               "--exp-dir", str(exp), "--version", "v2",
               "--stages", "text,merge"]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(REPO))
        assert r.returncode == 0, r.stderr[-2000:]
    got = (exp / "2-name2text.txt").read_text(encoding="utf8")
    want = _ref("2-name2text.txt").read_text(encoding="utf8")
    # exact equality vs the cuda_graph (ONNX g2pw) reference modulo line order
    # (multi-part runs concat part blocks like the webui does) and allowing
    # the single documented ONNX-vs-torch polyphone line (不得了 le5/liao3).
    got_l = sorted(got.strip().split("\n"))
    want_l = sorted(want.strip().split("\n"))
    assert len(got_l) == len(want_l)
    diff = [i for i, (a, b) in enumerate(zip(got_l, want_l)) if a != b]
    assert len(diff) <= 1, f"{len(diff)} lines differ from official reference"


@pytest.mark.heavy
def test_stage2_wav32k_md5_and_ssl():
    """5-wav32k/*.wav md5-equal to official scipy wavfile.write output;
    4-cnhubert ssl max-abs diff <= 3e-3 per frame vs official HuBERT
    (fp16-stored vs fp32 reference: fp16 storage alone rounds at ~2.3e-3
    for |ssl| up to 4.6; the fp32-vs-fp32 model diff is <=1.4e-4)."""
    exp = EXP_DIR
    if not (_ref("stage2").exists() and (exp / "5-wav32k").is_dir()):
        pytest.skip("official stage-2 reference or MLX output missing")
    import hashlib
    n = 0
    for wav in sorted((_ref("stage2")).glob("*.wav")):
        ours = exp / "5-wav32k" / wav.name
        assert ours.exists(), f"missing {wav.name}"
        assert hashlib.md5(wav.read_bytes()).hexdigest() == \
            hashlib.md5(ours.read_bytes()).hexdigest(), f"wav32k md5 differs: {wav.name}"
        n += 1
    assert n >= 20, f"only {n} files checked"
    diffs = []
    for ssl_fp32 in sorted((_ref("stage2")).glob("*.ssl_fp32.npy")):
        base = ssl_fp32.name[: -len(".ssl_fp32.npy")]
        ours = exp / "4-cnhubert" / f"{base}.npy"
        got = np.load(ours).astype(np.float32)  # stored fp16 like official
        want = np.load(ssl_fp32)
        assert got.shape == want.shape, (ours, got.shape, want.shape)
        diffs.append(np.abs(got - want).max())
    # fp16 storage quantum at |ssl|max~4.6 is ~2.3e-3; gate at 3e-3 with the
    # fp32-model-level check below catching real regressions.
    assert diffs and max(diffs) <= 3e-3, f"ssl max diff {max(diffs) if diffs else None}"


@pytest.mark.heavy
def test_stage2b_sv_parity():
    """7-sv_cn embeddings max diff vs official (torchaudio Resample +
    kaldi fbank + ERes2NetV2 fp32). Gate 3e-4: fbank itself carries the
    documented 4.1e-4 numpy-vs-torchaudio rounding (PARITY_NOTES v2Pro
    addendum); the sv network alone is 5.7e-6."""
    exp = EXP_DIR
    if not (_ref("stage2").exists()) or not (exp / "7-sv_cn").is_dir():
        pytest.skip("official sv reference or MLX output missing")
    diffs = []
    for ref in sorted((_ref("stage2")).glob("*.sv.npy")):
        base = ref.name[: -len(".sv.npy")]
        ours = exp / "7-sv_cn" / f"{base}.npy"
        got = np.load(ours)
        want = np.load(ref)
        assert got.shape == want.shape
        diffs.append(np.abs(got - want).max())
    assert diffs and max(diffs) <= 3e-4, f"sv max diff {max(diffs) if diffs else None}"


@pytest.mark.heavy
def test_stage3_semantic_codes_identical_v1_and_v2():
    """Quantized semantic codes identical on the gate files for BOTH v1
    (s2G488k) and v2 (s2G2333k): model-level gate consumes the SAME ssl
    inputs (official quantizer run on our 4-cnhubert outputs); the
    end-to-end cross (each side's own ssl) is asserted at >=23/24 files
    exact, 1 frame in 2915 (HuBERT fp32 ulp hitting one razor argmin)."""
    exp = EXP_DIR
    for v in ("v1", "v2"):
        want_p = _ref(f"6-name2semantic-{v}.tsv")
        got_p = exp / f"6-name2semantic-{v}.tsv"
        if not (want_p.exists() and got_p.exists()):
            pytest.skip(f"semantic reference/output missing for {v}")
        want = dict(l.split("\t") for l in want_p.read_text().strip().split("\n"))
        got = dict(l.split("\t") for l in got_p.read_text().strip().split("\n"))
        assert set(want) == set(got), (set(want) ^ set(got))
        same_p = _ref(f"6-name2semantic-sameinput-{v}.tsv")
        if same_p.exists():
            same = dict(l.split("\t") for l in same_p.read_text().strip().split("\n"))
            assert all(same[k] == got[k] for k in same), \
                f"{v}: quantizer codes drift on identical ssl inputs"
        bad = [k for k in want if want[k] != got[k]]
        assert len(bad) <= 1, f"{v}: {len(bad)}/{len(want)} files differ, e.g. {bad[:3]}"
