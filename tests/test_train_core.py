"""Tests for gsovits_mlx.train core: GradScaler, AdamW, ScaledAdam (torch
golden), locked-LR schedule, mixed-precision Trainer, checkpoint roundtrip.

Golden-vs-torch tests need a torch-capable interpreter. The repo venv has no
torch (inference engine is torch-free), so those tests self-skip there and a
subprocess runner (test_golden_via_base_venv) re-executes them under
/Users/baicai1145/.venvs/base (torch 2.13 CPU + mlx 0.32.2 + numpy 2.5.2 —
the same pins as the repo venv). MLX-only tests run everywhere MLX runs
(CPU device is fine: no GPU lock needed for these tiny tensors).
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from gsovits_mlx.train.mixed_precision import GradScaler  # noqa: E402
from gsovits_mlx.train.optim import AdamW, ScaledAdam  # noqa: E402
from gsovits_mlx.train.schedule import WarmupCosineLRSchedule, LOCKED_LR  # noqa: E402

BASE_VENV_PY = "/Users/baicai1145/.venvs/base/bin/python"
REF_OPTIM = ("/Users/baicai1145/repos/gpt-sovits/GPT-SoVITS-cuda_graph_accel_v5/"
             "GPT_SoVITS/AR/modules/optim.py")

HAVE_TORCH = importlib.util.find_spec("torch") is not None
if HAVE_TORCH:
    import torch  # noqa: F401


def _load_ref_scaled_adam():
    spec = importlib.util.spec_from_file_location("ref_optim", REF_OPTIM)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ScaledAdam


# ---------------------------------------------------------------------------
# GradScaler
# ---------------------------------------------------------------------------

class _DummyOpt:
    def __init__(self, params):
        self.param_groups = [{"params": dict(params), "grads": dict(params)}]
        self.applied = 0

    def apply_group(self, gi, grads):
        self.applied += 1


def test_grad_scaler_growth_and_backoff():
    p = mx.array([1.0, 2.0])
    opt = _DummyOpt({"w": p})
    s = GradScaler(init_scale=1024.0, growth_interval=3)
    # 3 finite steps -> scale grows x2
    for _ in range(3):
        opt.param_groups[0]["grads"] = {"w": mx.array([0.5, -0.5]) * s.get_scale()}
        s.step(opt)
        s.update()
    assert s.get_scale() == 2048.0
    # inf step -> backoff x0.5, tracker reset
    opt.param_groups[0]["grads"] = {"w": mx.array([float("inf"), 1.0]) * s.get_scale()}
    s.step(opt)
    assert s.found_inf[0] is True
    s.update()
    assert s.get_scale() == 1024.0
    # tracker reset: needs another full growth_interval of finite steps
    for _ in range(2):
        opt.param_groups[0]["grads"] = {"w": mx.array([0.5, -0.5]) * s.get_scale()}
        s.step(opt)
        s.update()
    assert s.get_scale() == 1024.0
    opt.param_groups[0]["grads"] = {"w": mx.array([0.5, -0.5]) * s.get_scale()}
    s.step(opt)
    s.update()
    assert s.get_scale() == 2048.0


def test_grad_scaler_inf_skips_step_and_keeps_state():
    p = mx.array([1.0, 2.0])
    opt = _DummyOpt({"w": p})
    s = GradScaler(init_scale=8.0)
    # finite step
    opt.param_groups[0]["grads"] = {"w": mx.array([2.0, 4.0])}  # scale=8 -> unscaled 0.25,0.5
    stepped = s.step(opt, apply_fn=lambda o, gs: o.apply_group(0, gs))
    assert stepped is True
    assert opt.applied == 1
    got = np.array(opt.param_groups[0]["grads"]["w"])
    assert np.allclose(got, [0.25, 0.5])
    s.update()
    # nan step: skipped entirely
    opt.param_groups[0]["grads"] = {"w": mx.array([float("nan"), 1.0]) * 8.0}
    stepped = s.step(opt, apply_fn=lambda o, gs: o.apply_group(0, gs))
    assert stepped is False
    assert opt.applied == 1  # no additional apply
    s.update()
    assert s.get_scale() == 4.0


# ---------------------------------------------------------------------------
# AdamW golden vs torch (CPU)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAVE_TORCH, reason="torch not installed in this interpreter")
def test_adamw_golden_vs_torch():
    rng = np.random.RandomState(0)
    w0 = rng.randn(6, 4).astype(np.float32)
    b0 = rng.randn(4).astype(np.float32)
    steps = 25
    grads = [(rng.randn(6, 4).astype(np.float32), rng.randn(4).astype(np.float32))
             for _ in range(steps)]

    tp = [torch.nn.Parameter(torch.tensor(w0)), torch.nn.Parameter(torch.tensor(b0))]
    topt = torch.optim.AdamW([tp[0], tp[1]], lr=0.01, betas=(0.8, 0.99),
                             eps=1e-9, weight_decay=0.1)
    mp_opt = AdamW({"w": mx.array(w0), "b": mx.array(b0)}, lr=0.01,
                   betas=(0.8, 0.99), eps=1e-9, weight_decay=0.1)
    for i in range(steps):
        gw, gb = grads[i]
        tp[0].grad = torch.tensor(gw)
        tp[1].grad = torch.tensor(gb)
        topt.step()
        mp_opt.apply_group(0, {"w": mx.array(gw)})
        mp_opt.apply_group(0, {"b": mx.array(gb)})  # same group here (single dict)
    # NOTE: single-param-dict group -> both in group 0
    mw = np.array(mp_opt.param_groups[0]["params"]["w"])
    mb = np.array(mp_opt.param_groups[0]["params"]["b"])
    assert np.abs(np.asarray(tp[0].data) - mw).max() < 1e-6
    assert np.abs(np.asarray(tp[1].data) - mb).max() < 1e-6


@pytest.mark.skipif(not HAVE_TORCH, reason="torch not installed in this interpreter")
def test_adamw_param_groups_golden():
    rng = np.random.RandomState(1)
    w0 = rng.randn(5, 3).astype(np.float32)
    v0 = rng.randn(3).astype(np.float32)
    tp = [torch.nn.Parameter(torch.tensor(w0)), torch.nn.Parameter(torch.tensor(v0))]
    topt = torch.optim.AdamW([
        {"params": [tp[0]], "lr": 0.02, "weight_decay": 0.0},
        {"params": [tp[1]], "lr": 0.005, "weight_decay": 0.2},
    ], lr=0.01, betas=(0.8, 0.99), eps=1e-9, weight_decay=0.1)
    mp_opt = AdamW([
        {"params": {"w": mx.array(w0)}, "lr": 0.02, "weight_decay": 0.0},
        {"params": {"v": mx.array(v0)}, "lr": 0.005, "weight_decay": 0.2},
    ], lr=0.01, betas=(0.8, 0.99), eps=1e-9, weight_decay=0.1)
    for i in range(15):
        gw = rng.randn(5, 3).astype(np.float32)
        gv = rng.randn(3).astype(np.float32)
        tp[0].grad = torch.tensor(gw)
        tp[1].grad = torch.tensor(gv)
        topt.step()
        mp_opt.apply_group(0, {"w": mx.array(gw)})
        mp_opt.apply_group(1, {"v": mx.array(gv)})
    assert np.abs(np.asarray(tp[0].data) -
                  np.array(mp_opt.param_groups[0]["params"]["w"])).max() < 1e-6
    assert np.abs(np.asarray(tp[1].data) -
                  np.array(mp_opt.param_groups[1]["params"]["v"])).max() < 1e-6


# ---------------------------------------------------------------------------
# ScaledAdam golden vs torch reference (CPU), 30 steps, clipped grads
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAVE_TORCH, reason="torch not installed in this interpreter")
def test_scaled_adam_golden():
    RefScaledAdam = _load_ref_scaled_adam()
    rng = np.random.RandomState(3)
    shapes = [(4, 6), (3, 5, 2), ()]
    init = [rng.randn(*s).astype(np.float32) * 0.5 if s
            else np.float32(rng.randn() * 0.5) for s in shapes]
    steps = 30
    g = np.random.RandomState(11)
    grads = []
    for i in range(steps):
        gg = []
        for s in shapes:
            a = g.randn(*s).astype(np.float32) if s else np.float32(g.randn())
            gg.append((a * 0.3).astype(np.float32))
        if i % 3 == 0:  # spikes engage the clipper
            gg[0] = (gg[0] * 50).astype(np.float32)
        grads.append(gg)

    tp = [torch.nn.Parameter(torch.tensor(x)) for x in init]
    topt = RefScaledAdam(
        [{"params": [tp[0], tp[2]], "lr": 0.02}, {"params": [tp[1]], "lr": 0.01}],
        lr=0.02, clipping_scale=2.0, size_update_period=4,
        clipping_update_period=8, parameters_names=[["a", "s"], ["b"]])
    mp_opt = ScaledAdam(
        [{"params": {"t0": mx.array(init[0]), "t2": mx.array(init[2])},
          "lr": 0.02},
         {"params": {"t1": mx.array(init[1])}, "lr": 0.01}],
        lr=0.02, clipping_scale=2.0, size_update_period=4,
        clipping_update_period=8)

    clip_active = False
    for i in range(steps):
        # numpy side
        mp_opt.apply_group(0, {"t0": mx.array(grads[i][0]),
                               "t2": mx.array(grads[i][2])})
        mp_opt.apply_group(1, {"t1": mx.array(grads[i][1])})
        # torch side
        for p_, gr in zip(tp, grads[i]):
            p_.grad = torch.tensor(gr)
        topt.step()
        clip_active = clip_active or i >= 9
    assert clip_active, "clipper never engaged; test degenerated"
    d0 = np.abs(np.asarray(tp[0].data, np.float64) -
                np.array(mp_opt.param_groups[0]["params"]["t0"])).max()
    d1 = np.abs(np.asarray(tp[1].data, np.float64) -
                np.array(mp_opt.param_groups[1]["params"]["t1"])).max()
    d2 = np.abs(float(tp[2].data) -
                float(mp_opt.param_groups[0]["params"]["t2"])).max()
    assert d0 < 1e-5, d0
    assert d1 < 1e-5, d1
    assert d2 < 1e-5, d2


# ---------------------------------------------------------------------------
# Locked-LR schedule quirk
# ---------------------------------------------------------------------------

def test_schedule_locked_lr_default():
    opt = _DummyOpt({"w": mx.array([1.0])})
    s = WarmupCosineLRSchedule(opt, init_lr=1e-6, peak_lr=1e-3, end_lr=1e-5,
                               warmup_steps=100, total_steps=1000)
    for i in range(50):
        lr = s.step()
        assert lr == LOCKED_LR == 0.002
        assert opt.param_groups[0]["lr"] == 0.002
        assert s.end_lr == 0.002  # official quirk: step() overwrote end_lr


def test_schedule_unlocked_math():
    opt = _DummyOpt({"w": mx.array([1.0])})
    s = WarmupCosineLRSchedule(opt, init_lr=1e-6, peak_lr=1e-3, end_lr=1e-5,
                               warmup_steps=10, total_steps=110,
                               current_step=0, lr_locked=False)
    # warmup: step0 -> init_lr
    assert s.step() == pytest.approx(1e-6)
    # mid warmup linear
    s2 = WarmupCosineLRSchedule(opt, init_lr=0.0, peak_lr=1.0, end_lr=0.0,
                                warmup_steps=10, total_steps=110,
                                lr_locked=False)
    lrs = [s2.step() for _ in range(5)]
    assert lrs[0] == pytest.approx(0.0)
    assert lrs[4] == pytest.approx(0.4)  # 4/10 of warmup
    # cosine: decay_ratio=0 -> peak
    s3 = WarmupCosineLRSchedule(opt, init_lr=0.0, peak_lr=1.0, end_lr=0.0,
                                warmup_steps=10, total_steps=110,
                                current_step=10, lr_locked=False)
    assert s3.step() == pytest.approx(1.0)
    # half window -> cos(pi/2)=0 -> coeff 0.5
    s4 = WarmupCosineLRSchedule(opt, init_lr=0.0, peak_lr=1.0, end_lr=0.0,
                                warmup_steps=0, total_steps=100,
                                current_step=50, lr_locked=False)
    assert s4.step() == pytest.approx(0.5)
    # past total -> end_lr
    s5 = WarmupCosineLRSchedule(opt, init_lr=0.0, peak_lr=1.0, end_lr=0.1,
                                warmup_steps=10, total_steps=100,
                                current_step=200, lr_locked=False)
    assert s5.step() == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Trainer: fp32-master mixed precision + fp16 forward
# ---------------------------------------------------------------------------

def _tiny_linear_loss(params16, batch):
    w = params16["w"].astype(mx.float32)
    x = mx.array(batch["x"])
    return mx.mean((x * w) ** 2)


def _tiny_linear_grads(params16, loss):
    # analytic grad of mean((x*w)^2) wrt w: 2*mean(x^2*w*x)... use mx.grad-free
    # manual: d/dw mean(x^2 w^2) = 2 w mean(x^2)
    w = params16["w"].astype(mx.float32)
    x = mx.array([1.0, 2.0])
    return {"w": 2.0 * w * mx.mean(x * x)}


def test_trainer_fp16_forward_fp32_master(tmp_path):
    from gsovits_mlx.train.loop import Trainer

    w0 = np.array([0.5, -0.3], np.float32)
    opt = AdamW({"w": mx.array(w0)}, lr=0.05, betas=(0.8, 0.99), eps=1e-9,
                weight_decay=0.0)
    scaler = GradScaler(init_scale=4.0, growth_interval=1000)
    trainer = Trainer(None, [opt], scaler=scaler, log_interval=1,
                      sample_footprint=False,
                      loss_log_path=str(tmp_path / "loss.jsonl"))
    batch = {"x": np.array([1.0, 2.0], np.float32)}
    for i in range(10):
        loss = trainer.train_step(
            batch,
            forward_fn=lambda p16, b: _tiny_linear_loss(p16, b),
            backward_fn=_tiny_linear_grads)
        assert np.isfinite(loss)
    # params stayed fp32 and moved
    w = np.array(opt.param_groups[0]["params"]["w"])
    assert opt.param_groups[0]["params"]["w"].dtype == mx.float32
    assert not np.allclose(w, w0)
    # fp16 copy policy: cast per forward, master untouched by casting
    master = opt.param_groups[0]["params"]
    f16 = Trainer.cast_master_to_fp16(master)
    assert f16["w"].dtype == mx.float16
    assert master["w"].dtype == mx.float32
    # jsonl log exists with 10 records
    lines = [json.loads(l) for l in open(tmp_path / "loss.jsonl")]
    assert len(lines) == 10 and lines[-1]["step"] == 10


def test_trainer_memory_report(tmp_path):
    from gsovits_mlx.train.loop import Trainer, analytic_bytes

    params = {"w": mx.zeros((4, 4))}
    opt = AdamW(params, lr=0.1)
    t = Trainer(None, [opt], sample_footprint=False)
    rep = t.memory_report()
    assert rep["weights"] == 4 * 4 * 4
    # AdamW states: m+v fp32 -> 2 * weights
    assert rep["optimizer_states"] == 2 * rep["weights"]
    assert rep["activations_peak"] == 0  # CPU: metal unavailable


# ---------------------------------------------------------------------------
# Checkpoint roundtrip
# ---------------------------------------------------------------------------

def test_resume_roundtrip(tmp_path):
    from gsovits_mlx.train.ckpt import save_resume, load_resume

    rng = np.random.RandomState(0)
    params = {"w": mx.array(rng.randn(3, 2).astype(np.float32)),
              "b": mx.array(rng.randn(2).astype(np.float32))}
    opt = AdamW(params, lr=0.01, betas=(0.8, 0.99), weight_decay=0.1)
    # a couple of steps so optimizer state is non-trivial
    for i in range(3):
        opt.apply_group(0, {"w": mx.array(rng.randn(3, 2).astype(np.float32) * 0.1),
                            "b": mx.array(rng.randn(2).astype(np.float32) * 0.1)})
    params = opt.param_groups[0]["params"]  # save the CURRENT (post-step) params
    path = str(tmp_path / "ckpt")
    save_resume(path, params, [opt], step=123, epoch=4,
                extra={"version": "v2"})

    opt2 = AdamW({"w": mx.zeros((3, 2)), "b": mx.zeros((2,))},
                 lr=0.05, weight_decay=0.0)
    loaded = load_resume(path, None, [opt2])
    assert loaded["step"] == 123 and loaded["epoch"] == 4
    assert loaded["extra"]["version"] == "v2"
    assert np.allclose(np.array(loaded["params"]["w"]),
                       np.array(params["w"]))
    # optimizer state (m,v,step per param) restored
    s1 = opt._states[0]["w"]
    s2 = opt2._states[0]["w"]
    assert s2["step"] == s1["step"] == 3
    assert np.allclose(np.array(s2["m"]), np.array(s1["m"]))
    assert np.allclose(np.array(s2["v"]), np.array(s1["v"]))
    # restore params into opt2 (the trainer-side contract: masters come from
    # the checkpoint; optimizer state was already loaded above)
    opt2.param_groups[0]["params"].update(loaded["params"])
    # continuing training from the restored state reproduces the trajectory
    g = {"w": mx.array(rng.randn(3, 2).astype(np.float32)),
         "b": mx.array(rng.randn(2).astype(np.float32))}
    opt.apply_group(0, g)
    opt2.apply_group(0, g)
    assert np.allclose(np.array(opt.param_groups[0]["params"]["w"]),
                       np.array(opt2.param_groups[0]["params"]["w"]))
    assert np.allclose(np.array(opt.param_groups[0]["params"]["b"]),
                       np.array(opt2.param_groups[0]["params"]["b"]))


def test_scaled_adam_resume_roundtrip(tmp_path):
    from gsovits_mlx.train.ckpt import save_resume, load_resume

    rng = np.random.RandomState(5)
    params = {"t0": mx.array(rng.randn(4, 3).astype(np.float32)),
              "t1": mx.array(np.float32(0.3))}
    opt = ScaledAdam([{"params": {"t0": params["t0"]}},
                      {"params": {"t1": params["t1"]}}],
                     lr=0.02, size_update_period=4, clipping_update_period=8)
    for i in range(9):  # step past init so states/rings are populated
        opt.apply_group(0, {"t0": mx.array(rng.randn(4, 3).astype(np.float32) * 0.1)})
        opt.apply_group(1, {"t1": mx.array(np.float32(0.1))})
    params = {"t0": opt.param_groups[0]["params"]["t0"],
              "t1": opt.param_groups[1]["params"]["t1"]}
    path = str(tmp_path / "sa")
    save_resume(path, params, [opt], step=9, epoch=1)
    opt2 = ScaledAdam([{"params": {"t0": mx.zeros((4, 3))}},
                       {"params": {"t1": mx.array(0.0)}}],
                      lr=0.02, size_update_period=4, clipping_update_period=8)
    loaded = load_resume(path, None, [opt2])
    opt2.param_groups[0]["params"]["t0"] = mx.array(loaded["params"]["t0"])
    opt2.param_groups[1]["params"]["t1"] = mx.array(loaded["params"]["t1"])
    g0 = mx.array(rng.randn(4, 3).astype(np.float32))
    g1 = mx.array(np.float32(0.05))
    opt.apply_group(0, {"t0": g0})
    opt.apply_group(1, {"t1": g1})
    opt2.apply_group(0, {"t0": g0})
    opt2.apply_group(1, {"t1": g1})
    assert np.allclose(np.array(opt.param_groups[0]["params"]["t0"]),
                       np.array(opt2.param_groups[0]["params"]["t0"]), atol=1e-6)
    assert np.isclose(float(opt.param_groups[1]["params"]["t1"]),
                      float(opt2.param_groups[1]["params"]["t1"]), atol=1e-7)


# ---------------------------------------------------------------------------
# Inference-weight exporters
# ---------------------------------------------------------------------------

def test_s1_inference_export_schema(tmp_path):
    from gsovits_mlx.pipeline import load_gpt
    from gsovits_mlx.train.ckpt import save_s1_inference

    cfg = json.load(open("/Users/baicai1145/repos/gpt-sovits/GPT-SoVITS-mlx/"
                         "models_local/s1/gpt.json"))
    params = mx.load("/Users/baicai1145/repos/gpt-sovits/GPT-SoVITS-mlx/"
                     "models_local/s1/gpt.safetensors")

    class M:
        def parameters(self):
            return params

    out = str(tmp_path / "s1")
    save_s1_inference(out, M(), cfg["config"], epoch=7)
    meta = json.load(open(os.path.join(out, "gpt.json")))
    assert meta["n_layer"] == 24
    assert meta["config"]["model"]["n_layer"] == 24
    arrs = mx.load(os.path.join(out, "gpt.safetensors"))
    assert all(v.dtype == mx.float16 for v in arrs.values())
    assert set(arrs) == set(params)
    # loadable by the inference engine (CPU is fine for load)
    gpt = load_gpt(out)
    assert gpt.num_layers == 24


def test_s2_inference_export_v2(tmp_path):
    from gsovits_mlx.pipeline import _load_sovits_v1v2
    from gsovits_mlx.train.ckpt import save_s2_inference

    root = "/Users/baicai1145/repos/gpt-sovits/GPT-SoVITS-mlx/models_local/v2"
    params = mx.load(os.path.join(root, "sovits.safetensors"))
    meta = json.load(open(os.path.join(root, "sovits.json")))
    out = str(tmp_path / "v2")
    save_s2_inference(out, params, {"model": meta["model_hps"],
                                    "data": meta["data"]}, "v2")
    m, mmeta = _load_sovits_v1v2(out, "v2")
    assert mmeta["version"] == "v2"
    # key parity with the reference export
    ref = mx.load(os.path.join(root, "sovits.safetensors"))
    now = mx.load(os.path.join(out, "sovits.safetensors"))
    assert set(ref) == set(now)
    for k in ref:
        assert ref[k].dtype == now[k].dtype, k
        assert np.allclose(np.array(ref[k]).astype(np.float32),
                           np.array(now[k]).astype(np.float32), atol=1e-3), k


def test_s2_inference_export_v2pro_fp32_sv(tmp_path):
    root = "/Users/baicai1145/repos/gpt-sovits/GPT-SoVITS-mlx/models_local/v2pro"
    from gsovits_mlx.train.ckpt import save_s2_inference

    params = mx.load(os.path.join(root, "sovits.safetensors"))
    meta = json.load(open(os.path.join(root, "sovits.json")))
    out = str(tmp_path / "v2pro")
    save_s2_inference(out, params, {"model": meta["model_hps"],
                                    "data": meta["data"]}, "v2Pro")
    arrs = mx.load(os.path.join(out, "sovits.safetensors"))
    for k, v in arrs.items():
        if k.startswith(("sv_emb.", "ge_to512.", "prelu")):
            assert v.dtype == mx.float32, k
        else:
            assert v.dtype == mx.float16, k
    j = json.load(open(os.path.join(out, "sovits.json")))
    assert j["version"] == "v2Pro"
    assert j["model_hps"]["gin_channels"] == 1024


def test_s2_inference_export_v3(tmp_path):
    from gsovits_mlx.pipeline import load_sovits_v3
    from gsovits_mlx.train.ckpt import save_s2_inference

    root = "/Users/baicai1145/repos/gpt-sovits/GPT-SoVITS-mlx/models_local/v3"
    params = mx.load(os.path.join(root, "sovits.safetensors"))
    meta = json.load(open(os.path.join(root, "sovits.json")))
    out = str(tmp_path / "v3")
    save_s2_inference(out, params, {"model": meta["model_hps"],
                                    "data": meta["data"],
                                    "dit": meta["dit"]}, "v3")
    m, mmeta = load_sovits_v3(out, "v3")
    assert mmeta["version"] == "v3"
    assert mmeta["dit"]["depth"] == meta["dit"]["depth"]
    ref = mx.load(os.path.join(root, "sovits.safetensors"))
    now = mx.load(os.path.join(out, "sovits.safetensors"))
    assert set(ref) == set(now)


# ---------------------------------------------------------------------------
# Re-run the torch golden tests under the torch-capable base venv when the
# repo venv lacks torch.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(HAVE_TORCH, reason="torch available in-process")
def test_golden_via_base_venv():
    if not os.path.isfile(BASE_VENV_PY):
        pytest.skip("base venv python not found")
    r = subprocess.run(
        [BASE_VENV_PY, "-m", "pytest", "-q",
         os.path.basename(__file__) + "::test_adamw_golden_vs_torch",
         os.path.basename(__file__) + "::test_adamw_param_groups_golden",
         os.path.basename(__file__) + "::test_scaled_adam_golden",
         "--no-header"],
        cwd=os.path.dirname(__file__), capture_output=True, text=True,
        timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "3 passed" in r.stdout, r.stdout  # goldens really executed
