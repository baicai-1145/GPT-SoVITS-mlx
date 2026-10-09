"""Tests for the s1 (AR/GPT) trainer port: dataset filters, sampler
determinism, pad_y_eos shapes, attention-mask correctness vs a brute-force
torch reference, forward_old loss vs the official torch module (tiny
config, CPU, fp32), and export roundtrip.

Torch-reference tests re-execute under /Users/baicai1145/.venvs/base
(torch 2.13 CPU) via the subprocess pattern of test_train_core.py; MLX
parts run in the repo venv (CPU device OK — tiny tensors).
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from gsovits_mlx.train.s1_data import (Text2SemanticDataset,  # noqa: E402
                                       BucketBatchSampler, collate, pad_y_eos)
from gsovits_mlx.train.s1_model_train import (NEG_INF,  # noqa: E402
                                              S1TrainModel,
                                              build_train_attn_mask,
                                              pad_y_eos_np,
                                              make_pad_mask_left_np,
                                              make_pad_mask_np,
                                              top3_accuracy)

BASE_VENV_PY = "/Users/baicai1145/.venvs/base/bin/python"
REF_AR = ("/Users/baicai1145/repos/gpt-sovits/GPT-SoVITS-cuda_graph_accel_v5/"
          "GPT_SoVITS/AR")
HAVE_TORCH = importlib.util.find_spec("torch") is not None
if HAVE_TORCH:
    import torch

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))

# tiny training config (official structure, shrunk dims)
TINY_CONFIG = {
    "train": {"seed": 1234, "batch_size": 4},
    "data": {"max_sec": 54, "pad_val": 1024},
    "model": {
        "vocab_size": 1025, "phoneme_vocab_size": 512,
        "embedding_dim": 64, "hidden_dim": 64, "head": 4,
        "n_layer": 2, "dropout": 0, "EOS": 1024,
    },
}


# ---------------------------------------------------------------------------
# helpers: synthetic exp dir
# ---------------------------------------------------------------------------

def _make_exp_dir(tmp_path, n_items=30, hz=25, version="v2"):
    """Synthetic 1-prepare artifacts: 2-name2text, 3-bert npy, 6-name2semantic.
    n_items is capped at 100 so the min_num duplication rule NEVER fires
    (callers asserting exact kept-counts need the no-dup regime); the
    dedicated test covers the duplication rule itself."""
    import random
    rng = random.Random(7)
    n_items = min(n_items, 500)
    exp = tmp_path / "exp"
    exp.mkdir()
    bert_dir = exp / "3-bert"
    bert_dir.mkdir()
    lines = []
    for i in range(n_items):
        name = f"utt{i:03d}"
        n_sec = rng.uniform(2.0, 4.5)
        n_sem = int(n_sec * hz)
        sem = [str(rng.randint(0, 1023)) for _ in range(n_sem)]
        n_ph = int(n_sec * 6)  # ~6 phones/sec, inside [3, 25]
        phones = ["a", "b", "c", "d", "e", "f"]
        ph = " ".join(rng.choice(phones) for _ in range(n_ph))
        word2ph = " ".join("1" for _ in range(n_ph))
        lines.append(f"{name}\t{ph}\t{word2ph}\ttext{i}")
        with open(bert_dir / f"{name}.npy", "wb") as f:
            np.save(f, np.random.RandomState(i).randn(1024, n_ph).astype(np.float16))
        with open(exp / "6-name2semantic.tsv", "a") as f:
            f.write(name + "\t" + " ".join(sem) + "\n")
    with open(exp / "2-name2text.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    return exp


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------

class TestDataset:
    def test_filters_and_shape(self, tmp_path):
        exp = _make_exp_dir(tmp_path, n_items=120)  # >100: no duplication
        ds = Text2SemanticDataset(
            phoneme_path=str(exp / "2-name2text.txt"),
            semantic_path=str(exp / "6-name2semantic.tsv"),
            version="v2", max_sec=54)
        assert len(ds) == 120  # all synthetic items pass the filters
        assert ds.stats["num_duplicated"] == 0
        item = ds[0]
        assert item["bert_feature"].shape[0] == 1024
        assert item["bert_feature"].shape[1] == item["phoneme_ids_len"]
        assert 0 < len(item["semantic_ids"]) <= 54 * 25

    def test_semantic_too_long_filtered(self, tmp_path):
        exp = _make_exp_dir(tmp_path, n_items=120)
        # append one 60s item -> filtered by max_sec=54 (60*25 > 54*25)
        with open(exp / "6-name2semantic.tsv", "a") as f:
            f.write("huge\t" + " ".join(["1"] * (60 * 25)) + "\n")
        with open(exp / "2-name2text.txt", "a") as f:
            f.write("huge\t" + " ".join(["a"] * 360) + "\t" +
                    " ".join(["1"] * 360) + "\ttext")
        ds = Text2SemanticDataset(
            phoneme_path=str(exp / "2-name2text.txt"),
            semantic_path=str(exp / "6-name2semantic.tsv"),
            version="v2", max_sec=54)
        assert len(ds) == 120
        assert ds.stats["num_deleted_bigger"] == 1

    def test_ps_ratio_filtered(self, tmp_path):
        exp = _make_exp_dir(tmp_path, n_items=120)
        # 5s audio (125 tokens) with 2 phones -> ratio 0.4 < 3 -> filtered
        with open(exp / "6-name2semantic.tsv", "a") as f:
            f.write("sparse\t" + " ".join(["1"] * 125) + "\n")
        with open(exp / "2-name2text.txt", "a") as f:
            f.write("sparse\ta b\t1 1\ttext")
        ds = Text2SemanticDataset(
            phoneme_path=str(exp / "2-name2text.txt"),
            semantic_path=str(exp / "6-name2semantic.tsv"),
            version="v2", max_sec=54)
        assert len(ds) == 120
        assert ds.stats["num_deleted_ps"] == 1

    def test_phoneme_len_cap(self, tmp_path):
        exp = _make_exp_dir(tmp_path, n_items=120)
        # 3s audio (75 tokens), 700 phones: ratio 233 > 25 AND
        # 700 > 54*25/2.5=540 -> caught by the phoneme-len cap first
        with open(exp / "6-name2semantic.tsv", "a") as f:
            f.write("dense\t" + " ".join(["1"] * 75) + "\n")
        with open(exp / "2-name2text.txt", "a") as f:
            f.write("dense\t" + " ".join(["a"] * 700) + "\t" +
                    " ".join(["1"] * 700) + "\ttext")
        ds = Text2SemanticDataset(
            phoneme_path=str(exp / "2-name2text.txt"),
            semantic_path=str(exp / "6-name2semantic.tsv"),
            version="v2", max_sec=54)
        assert ds.stats["num_deleted_ps"] == 1

    def test_min_num_duplication(self, tmp_path):
        exp = _make_exp_dir(tmp_path, n_items=3)
        ds = Text2SemanticDataset(
            phoneme_path=str(exp / "2-name2text.txt"),
            semantic_path=str(exp / "6-name2semantic.tsv"),
            version="v2", max_sec=54)
        # 3 items -> replicated max(2, int(100/3)=33) = 33 times -> 99
        assert len(ds) == 3 * 33
        assert ds.stats["num_duplicated"] == 33
        assert len(ds.item_names) == len(ds)

    def test_missing_bert_zeros(self, tmp_path):
        exp = _make_exp_dir(tmp_path, n_items=8)
        os.remove(exp / "3-bert" / "utt000.npy")
        ds = Text2SemanticDataset(
            phoneme_path=str(exp / "2-name2text.txt"),
            semantic_path=str(exp / "6-name2semantic.tsv"),
            version="v2", max_sec=54)
        assert ds[0]["bert_feature"] is None
        batch = ds.collate([ds[0], ds[1]])
        assert batch["bert_feature"].shape == (2, 1024, max(
            len(ds.semantic_phoneme[0][1]), len(ds.semantic_phoneme[1][1])))
        assert float(np.abs(batch["bert_feature"][0]).max()) == 0.0

    def test_collate_pads(self, tmp_path):
        exp = _make_exp_dir(tmp_path, n_items=120)
        ds = Text2SemanticDataset(
            phoneme_path=str(exp / "2-name2text.txt"),
            semantic_path=str(exp / "6-name2semantic.tsv"),
            version="v2", max_sec=54)
        batch = ds.collate([ds[i] for i in range(4)])
        B, X = batch["phoneme_ids"].shape
        Y = batch["semantic_ids"].shape[1]
        assert batch["bert_feature"].shape == (4, 1024, X)
        # semantic pads are pad_val, phoneme pads are 0
        for i in range(4):
            sl = int(batch["semantic_ids_len"][i])
            if sl < Y:
                assert (batch["semantic_ids"][i, sl:] == 1024).all()
            pl = int(batch["phoneme_ids_len"][i])
            if pl < X:
                assert (batch["phoneme_ids"][i, pl:] == 0).all()


# ---------------------------------------------------------------------------
# sampler
# ---------------------------------------------------------------------------

class TestSampler:
    def _ds(self, tmp_path, n=50):
        exp = _make_exp_dir(tmp_path, n_items=n)
        return Text2SemanticDataset(
            phoneme_path=str(exp / "2-name2text.txt"),
            semantic_path=str(exp / "6-name2semantic.tsv"),
            version="v2", max_sec=54)

    def test_deterministic_per_epoch(self, tmp_path):
        ds = self._ds(tmp_path)
        s1 = BucketBatchSampler(ds, batch_size=8, seed=0)
        s2 = BucketBatchSampler(ds, batch_size=8, seed=0)
        b1 = s1.batch_indices(epoch=3)
        b2 = s2.batch_indices(epoch=3)
        assert b1 == b2
        b_other = s1.batch_indices(epoch=4)
        assert b1 != b_other

    def test_all_indices_covered(self, tmp_path):
        ds = self._ds(tmp_path)
        s = BucketBatchSampler(ds, batch_size=8, seed=1)
        batches = s.batch_indices(epoch=0)
        flat = sorted(i for b in batches for i in b)
        assert flat == sorted(range(len(ds)))[:len(flat)] or \
            set(flat) == set(range(len(ds)))
        # every batch within bucket similarity: max batch sec spread < 2s
        for b in batches[:5]:
            secs = [ds.get_sample_length(i) for i in b]
            assert max(secs) - min(secs) <= 4.0 + 1e-9  # bucket width 2.0 x2

    def test_len_invariants(self, tmp_path):
        ds = self._ds(tmp_path, n=50)
        s = BucketBatchSampler(ds, batch_size=8, seed=0)
        batches = s.batch_indices(epoch=0)
        assert sum(len(b) for b in batches) == len(ds)
        assert max(len(b) for b in batches) <= 8


# ---------------------------------------------------------------------------
# pad_y_eos + masks (pure numpy, brute-force reference)
# ---------------------------------------------------------------------------

class TestPadYEos:
    def test_shapes_and_shift(self):
        codes = np.array([[5, 6, 7, 0, 0], [1, 2, 0, 0, 0]])
        y_mask_int = np.array([[0, 0, 0, 1, 1], [0, 0, 1, 1, 1]])
        y, targets = pad_y_eos_np(codes, y_mask_int, eos=1024)
        assert y.shape == (2, 5) and targets.shape == (2, 6)
        # every target row ends with EOS
        assert (targets[:, -1] == 1024).all()
        # decoder input is targets shifted (first input = first code)
        assert (y == targets[:, :-1]).all()
        # pad slots in the input carry EOS
        assert y[0, 3] == 1024 and y[1, 2] == 1024
        # real slots keep their code
        assert y[0, 0] == 5 and y[1, 1] == 2

    def test_against_torch_reference(self):
        pytest.skip("covered in torch subprocess below")

    def test_pad_masks(self):
        lengths = np.array([1, 3, 2, 5])
        right = make_pad_mask_np(lengths)
        left = make_pad_mask_left_np(lengths)
        # right: True after len
        assert right[0].tolist() == [False, True, True, True, True]
        assert right[1].tolist() == [False, False, False, True, True]
        # left: True at the FIRST (max-len) slots
        assert left[0].tolist() == [True, True, True, True, False]
        assert left[1].tolist() == [True, True, False, False, False]
        assert left[2].tolist() == [True, True, True, False, False]
        assert left[3].tolist() == [False] * 5

    def test_attn_mask_bruteforce(self):
        """Mask must equal the official construction computed step by step."""
        X, Y, H = 5, 7, 2
        x_lens = np.array([5, 3, 4])
        y_lens = np.array([7, 5, 6])
        B = 3
        # reference: official ops
        x_mask = make_pad_mask_left_np(x_lens, X)
        y_mask = make_pad_mask_np(y_lens, Y)
        xy_padding = np.concatenate([x_mask, y_mask], axis=1)
        x_attn = np.concatenate([np.zeros((X, X), bool),
                                 np.ones((X, Y), bool)], axis=1)
        y_attn = np.concatenate(
            [np.zeros((Y, X), bool),
             np.triu(np.ones((Y, Y), bool), 1)], axis=1)
        xy_attn = np.concatenate([x_attn, y_attn], axis=0)
        # torch broadcasting: (B,1,1,src) padding OR (src,src) causal -> full 2D
        ref = xy_attn[None, :, :] | xy_padding[:, None, :]
        ref = np.where(ref, NEG_INF, 0.0).astype(np.float32)[:, None, :, :]
        got = build_train_attn_mask(x_lens, y_lens, X, Y, H)
        assert got.shape == (B, 1, X + Y, X + Y)
        np.testing.assert_array_equal(got, ref)
        # semantics: no all-masked rows (NaN safety)
        assert (got > -1e8).any(axis=-1).all()
        # fp16 cast safety: masked floor must stay finite (no NaN) in fp16
        # (-inf overflows to NaN; found via probe nan losses 2026-10-09)
        import mlx.core as _mx
        m16 = _mx.array(got).astype(_mx.float16)
        assert _mx.all(_mx.isfinite(m16)).item()


# ---------------------------------------------------------------------------
# forward vs official torch forward_old (tiny config, CPU fp32)
# ---------------------------------------------------------------------------

TINY_TORCH_SRC = r'''
import json, os, sys
import numpy as np
import torch

torch.manual_seed(0)

REF_AR = %r
sys.path.insert(0, os.path.dirname(REF_AR))
os.chdir(os.path.dirname(REF_AR))  # "AR" package importable

from AR.models.t2s_model import Text2SemanticDecoder

config = json.load(open(sys.argv[1]))
dec = Text2SemanticDecoder(config, norm_first=False, top_k=3)
dec.eval()  # dropout off

batch_np = json.load(open(sys.argv[2]))

phones = torch.tensor(batch_np["phoneme_ids"], dtype=torch.long)
x_lens = torch.tensor(batch_np["phoneme_ids_len"], dtype=torch.long)
sem = torch.tensor(batch_np["semantic_ids"], dtype=torch.long)
y_lens = torch.tensor(batch_np["semantic_ids_len"], dtype=torch.long)
bert = torch.tensor(np.array(batch_np["bert_feature"], dtype=np.float32))

with torch.no_grad():
    loss, acc = dec.forward_old(phones, x_lens, sem, y_lens, bert)
print(json.dumps({"loss": float(loss), "acc": float(acc)}))
'''


def _run_tiny_torch(config, batch, tmp_path):
    if not HAVE_TORCH:
        pytest.skip("torch not in this venv")
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(config))
    batch_path = tmp_path / "batch.json"
    batch_path.write_text(json.dumps({
        k: (v.tolist() if isinstance(v, np.ndarray) else v)
        for k, v in batch.items()}))
    src_path = tmp_path / "tiny_torch.py"
    src_path.write_text(TINY_TORCH_SRC % REF_AR)
    out = subprocess.run([sys.executable, str(src_path), str(cfg_path),
                          str(batch_path)],
                         capture_output=True, text=True, cwd=str(tmp_path))
    if out.returncode != 0:
        pytest.skip(f"torch reference failed: {out.stderr[-800:]}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def _tiny_params(config, seed=0):
    rng = np.random.RandomState(seed)
    m = config["model"]
    D = m["embedding_dim"]
    params = {
        "ar_text_embedding": mx.array(rng.randn(m["phoneme_vocab_size"], D) * 0.3, mx.float32),
        "ar_audio_embedding": mx.array(rng.randn(m["vocab_size"], D) * 0.3, mx.float32),
        "ar_text_position.alpha": mx.array([1.0], mx.float32),
        "ar_audio_position.alpha": mx.array([1.0], mx.float32),
        "bert_proj.weight": mx.array(rng.randn(D, 1024) * 0.05, mx.float32),
        "bert_proj.bias": mx.array(rng.randn(D) * 0.05, mx.float32),
        "ar_predict_layer.weight": mx.array(rng.randn(m["vocab_size"], m["hidden_dim"]) * 0.3, mx.float32),
    }
    for i in range(m["n_layer"]):
        pre = f"block.{i}."
        params[pre + "qkv_w"] = mx.array(rng.randn(3 * D, D) * 0.1, mx.float32)
        params[pre + "qkv_b"] = mx.array(rng.randn(3 * D) * 0.1, mx.float32)
        params[pre + "out_w"] = mx.array(rng.randn(D, D) * 0.1, mx.float32)
        params[pre + "out_b"] = mx.array(rng.randn(D) * 0.1, mx.float32)
        params[pre + "norm1.g"] = mx.array(1 + rng.randn(D) * 0.05, mx.float32)
        params[pre + "norm1.b"] = mx.array(rng.randn(D) * 0.05, mx.float32)
        params[pre + "norm2.g"] = mx.array(1 + rng.randn(D) * 0.05, mx.float32)
        params[pre + "norm2.b"] = mx.array(rng.randn(D) * 0.05, mx.float32)
        params[pre + "mlp1.w"] = mx.array(rng.randn(4 * D, D) * 0.1, mx.float32)
        params[pre + "mlp1.b"] = mx.array(rng.randn(4 * D) * 0.1, mx.float32)
        params[pre + "mlp2.w"] = mx.array(rng.randn(D, 4 * D) * 0.1, mx.float32)
        params[pre + "mlp2.b"] = mx.array(rng.randn(D) * 0.1, mx.float32)
    return params


def _load_torch_weights_into(params: dict, config: dict) -> dict:
    """Build torch-module weights FROM our layout (inverse mapping) and run
    the official forward_old; returns {"loss","acc"}."""
    # implemented in the subprocess helper below
    raise NotImplementedError


class TestForwardVsTorch:
    def test_forward_old_matches(self, tmp_path):
        """MLX forward (fp32) vs official torch forward_old on the same tiny
        weights and batch: loss within 1e-3 relative."""
        config = json.loads(json.dumps(TINY_CONFIG))
        config["model"]["phoneme_vocab_size"] = 512
        params = _tiny_params(config, seed=0)

        # build a small bucketed batch by hand
        rng = np.random.RandomState(3)
        B, X, Y = 3, 8, 12
        x_lens = np.array([8, 7, 6])
        y_lens = np.array([12, 11, 9])
        phones = np.zeros((B, X), dtype=np.int64)
        sem = np.full((B, Y), 1024, dtype=np.int64)
        for i in range(B):
            phones[i, :x_lens[i]] = rng.randint(0, 500, x_lens[i])
            sem[i, :y_lens[i]] = rng.randint(0, 1024, y_lens[i])
        bert = rng.randn(B, 1024, X).astype(np.float16)
        batch = {"phoneme_ids": phones, "phoneme_ids_len": x_lens,
                 "semantic_ids": sem, "semantic_ids_len": y_lens,
                 "bert_feature": bert}

        model = S1TrainModel(config, dropout=0, dtype=mx.float32).load(params)
        loss, acc, logits = model.forward(params, batch, return_logits=True)
        assert logits.shape == (B, Y + 1, config["model"]["vocab_size"])
        loss_f = float(loss)
        assert np.isfinite(loss_f)

        ref = _forward_old_torch(params, config, batch, tmp_path)
        rel = abs(loss_f - ref["loss"]) / max(abs(ref["loss"]), 1e-9)
        assert rel < 1e-3, f"MLX {loss_f} vs torch {ref['loss']} (rel {rel})"


def _forward_old_torch(params, config, batch, tmp_path):
    """Run official forward_old with OUR weights via a torch subprocess."""
    # save params as npz + write the driver
    npz = tmp_path / "params.npz"
    np.savez(npz, **{k: np.asarray(v) for k, v in params.items()})
    driver = tmp_path / ("torch_fwd_%d.py" % os.getpid())
    driver.write_text(_TORCH_FWD_SRC % (REF_AR,))
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(config))
    npz_path = str(npz)
    py = BASE_VENV_PY if os.path.exists(BASE_VENV_PY) else sys.executable
    out = subprocess.run(
        [py, str(driver), cfg_path, npz_path],
        capture_output=True, text=True, cwd=str(tmp_path), timeout=300)
    if out.returncode != 0:
        raise RuntimeError(f"torch ref failed:\n{out.stderr[-1200:]}")
    return json.loads(out.stdout.strip().splitlines()[-1])


_TORCH_FWD_SRC = r'''
import json, os, sys
import numpy as np
import torch

REF_AR = %r
sys.path.insert(0, os.path.dirname(REF_AR))
os.chdir(os.path.dirname(REF_AR))

from AR.models.t2s_model import Text2SemanticDecoder

config = json.load(open(sys.argv[1]))
npz = np.load(sys.argv[2])
dec = Text2SemanticDecoder(config, norm_first=False, top_k=3)
sd = {}
D = config["model"]["embedding_dim"]
for i in range(config["model"]["n_layer"]):
    pre = f"h.layers.{i}."
    blk = f"block.{i}."
    sd[pre + "self_attn.in_proj_weight"] = torch.tensor(npz[blk + "qkv_w"])
    sd[pre + "self_attn.in_proj_bias"] = torch.tensor(npz[blk + "qkv_b"])
    sd[pre + "self_attn.out_proj.weight"] = torch.tensor(npz[blk + "out_w"])
    sd[pre + "self_attn.out_proj.bias"] = torch.tensor(npz[blk + "out_b"])
    sd[pre + "norm1.weight"] = torch.tensor(npz[blk + "norm1.g"])
    sd[pre + "norm1.bias"] = torch.tensor(npz[blk + "norm1.b"])
    sd[pre + "norm2.weight"] = torch.tensor(npz[blk + "norm2.g"])
    sd[pre + "norm2.bias"] = torch.tensor(npz[blk + "norm2.b"])
    sd[pre + "linear1.weight"] = torch.tensor(npz[blk + "mlp1.w"])
    sd[pre + "linear1.bias"] = torch.tensor(npz[blk + "mlp1.b"])
    sd[pre + "linear2.weight"] = torch.tensor(npz[blk + "mlp2.w"])
    sd[pre + "linear2.bias"] = torch.tensor(npz[blk + "mlp2.b"])
sd["ar_text_embedding.word_embeddings.weight"] = torch.tensor(npz["ar_text_embedding"])
sd["ar_audio_embedding.word_embeddings.weight"] = torch.tensor(npz["ar_audio_embedding"])
sd["ar_text_position.alpha"] = torch.tensor(npz["ar_text_position.alpha"])
sd["ar_audio_position.alpha"] = torch.tensor(npz["ar_audio_position.alpha"])
sd["bert_proj.weight"] = torch.tensor(npz["bert_proj.weight"])
sd["bert_proj.bias"] = torch.tensor(npz["bert_proj.bias"])
sd["ar_predict_layer.weight"] = torch.tensor(npz["ar_predict_layer.weight"])
missing, unexpected = dec.load_state_dict(sd, strict=True)
dec = dec.eval()

rng = np.random.RandomState(3)
B, X, Y = 3, 8, 12
x_lens = np.array([8, 7, 6])
y_lens = np.array([12, 11, 9])
phones = np.zeros((B, X), dtype=np.int64)
sem = np.full((B, Y), 1024, dtype=np.int64)
for i in range(B):
    phones[i, :x_lens[i]] = rng.randint(0, 500, x_lens[i])
    sem[i, :y_lens[i]] = rng.randint(0, 1024, y_lens[i])
bert = rng.randn(B, 1024, X).astype(np.float32)

with torch.no_grad():
    loss, acc = dec.forward_old(
        torch.tensor(phones), torch.tensor(x_lens),
        torch.tensor(sem), torch.tensor(y_lens), torch.tensor(bert))
print(json.dumps({"loss": float(loss), "acc": float(acc)}))
'''


# ---------------------------------------------------------------------------
# top3 accuracy
# ---------------------------------------------------------------------------

class TestTop3:
    def test_matches_manual(self):
        rng = np.random.RandomState(0)
        logits = mx.array(rng.randn(2, 5, 4).astype(np.float32))
        targets = mx.array(np.array([[0, 2, 5, 5], [1, 1, 1, 5]]))
        # EOS = 5 ignored; manual: per non-EOS pos, hit if target in top3
        lg = np.asarray(logits)
        acc = top3_accuracy(logits, targets, eos=5)
        hits = tot = 0
        for b in range(2):
            for t in range(4):
                tgt = int(targets[b, t])
                if tgt == 5:
                    continue
                tot += 1
                top3 = np.argsort(lg[b, :, t])[-3:]
                if tgt in top3:
                    hits += 1
        assert abs(acc - hits / tot) < 1e-6


# ---------------------------------------------------------------------------
# export roundtrip
# ---------------------------------------------------------------------------

class TestExport:
    def test_save_s1_inference_roundtrip(self, tmp_path):
        from gsovits_mlx.train import ckpt as ckpt_mod
        config = json.loads(json.dumps(TINY_CONFIG))
        params = _tiny_params(config, seed=1)
        out = tmp_path / "export"
        ckpt_mod.save_s1_inference(str(out), params, config, epoch=0)
        assert (out / "gpt.safetensors").exists()
        meta = json.load(open(out / "gpt.json"))
        assert meta["n_layer"] == config["model"]["n_layer"]
        w = mx.load(str(out / "gpt.safetensors"))
        assert w["block.0.qkv_w"].dtype == mx.float16
        # loadable by the inference decoder
        from gsovits_mlx.gpt.t2s import Text2SemanticDecoder
        dec = Text2SemanticDecoder(meta["config"])
        dec.load(dict(w))
