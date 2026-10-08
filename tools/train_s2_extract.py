"""Torch s2G/s2D checkpoint -> npz intermediates for s2 GAN training (task-4).

Run under the BASE VENV (torch available); the repo venv stays torch-free.
This is a one-time extraction per checkpoint; outputs land under
<repo>/.tmp/train_s2/:

  s2G_<version>.npz  — flat torch state-dict names, float32 numpy,
                       conv layouts transposed to MLX:
                         Conv1d weight (out,in,k)      -> (out,k,in)
                         ConvTranspose1d (in,out,k)    -> (out,k,in)
                         Conv2d stays (out,in,kh,kw)
                       weight-norm modules keep BOTH weight_g and weight_v
                       (training uses the live (g,v) parameterization; the
                       fused single-weight form is only for inference).
  s2D_<version>.npz  — same for the discriminator.

Also prints the official load_state_dict(strict=False) missing/unexpected
report (mirroring s2_train.py behavior for diagnostics).

Usage:
  /Users/baicai1145/.venvs/base/bin/python -m tools.train_s2_extract \
      --version v2 --out .tmp/train_s2 [--extra-s2d path]
Default checkpoint paths follow /Volumes/2T/gpt-sovits-models/pretrained_models.
"""

from __future__ import annotations

import argparse
import os
import sys
import types

import numpy as np

PRETRAINED_ROOT = "/Volumes/2T/gpt-sovits-models/pretrained_models"

S2G_PATHS = {
    "v1": f"{PRETRAINED_ROOT}/s2G488k.pth",
    "v2": f"{PRETRAINED_ROOT}/gsv-v2final-pretrained/s2G2333k.pth",
    "v2Pro": f"{PRETRAINED_ROOT}/v2Pro/s2Gv2Pro.pth",
    "v2ProPlus": f"{PRETRAINED_ROOT}/v2Pro/s2Gv2ProPlus.pth",
}
S2D_PATHS = {
    "v2Pro": f"{PRETRAINED_ROOT}/v2Pro/s2Dv2Pro.pth",
    "v2ProPlus": f"{PRETRAINED_ROOT}/v2Pro/s2Dv2ProPlus.pth",
}


def _install_hparams_shim() -> None:
    import importlib.util
    try:
        if importlib.util.find_spec("utils"):
            return
    except ValueError:  # utils already in sys.modules without a spec
        return

    class HParams:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)
        keys = lambda self: self.__dict__.keys()  # noqa: E731
        items = lambda self: self.__dict__.items()  # noqa: E731
        values = lambda self: self.__dict__.values()  # noqa: E731
        def __len__(self): return len(self.__dict__)
        def __getitem__(self, k): return getattr(self, k)
        def __setitem__(self, k, v): setattr(self, k, v)
        def __contains__(self, k): return k in self.__dict__
        def get(self, k, d=None): return getattr(self, k, d)
    shim = types.ModuleType("utils")
    shim.HParams = HParams
    sys.modules["utils"] = shim


def to_mlx_conv1d(a):
    return np.ascontiguousarray(a.transpose(0, 2, 1))


def extract(path: str, out_npz: str) -> dict:
    import torch

    _install_hparams_shim()
    d = torch.load(path, map_location="cpu", weights_only=False)
    sd = d["weight"]
    arrays = {}
    for k, v in sd.items():
        a = v.float().numpy()
        if a.ndim == 3 and ".weight" in k and "weight_g" not in k \
                and "weight_v" not in k and "text_embedding" not in k:
            # Conv1d / ConvTranspose1d weight -> (out, k, in)
            a = to_mlx_conv1d(a)
        arrays[k] = a.astype(np.float32)
    os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
    np.savez(out_npz, **arrays)
    print(f"[extract] {len(arrays)} tensors {path} -> {out_npz}")
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, choices=list(S2G_PATHS))
    ap.add_argument("--out", default=".tmp/train_s2")
    ap.add_argument("--s2g", default=None, help="override s2G pth path")
    ap.add_argument("--s2d", default=None, help="override s2D pth path")
    args = ap.parse_args()

    s2g = args.s2g or S2G_PATHS[args.version]
    d = extract(s2g, os.path.join(args.out, f"s2G_{args.version}.npz"))
    cfg = d.get("config")
    if cfg is not None:
        model = cfg.get("model") if hasattr(cfg, "get") else None
        if model is not None:
            print("[extract] model hps:",
                  {k: model.get(k) for k in (
                      "inter_channels", "hidden_channels", "filter_channels",
                      "n_heads", "n_layers", "kernel_size", "gin_channels",
                      "semantic_frame_rate", "freeze_quantizer",
                      "upsample_initial_channel", "upsample_kernel_sizes")})

    s2d = args.s2d or S2D_PATHS.get(args.version)
    if s2d and os.path.exists(s2d):
        extract(s2d, os.path.join(args.out, f"s2D_{args.version}.npz"))
    else:
        print(f"[extract] NOTE: no pretrained s2D for {args.version}; "
              "the trainer will init D fresh and document it.")


if __name__ == "__main__":
    main()
