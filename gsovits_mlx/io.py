"""MLX weight loading: safetensors -> model param dicts with name mapping."""

from __future__ import annotations

import json
import os

import mlx.core as mx
from safetensors import safe_open


def load_mlx_safetensors(path: str) -> dict[str, mx.array]:
    tensors = {}
    with safe_open(path, framework="mlx") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def model_dir_paths(model_dir: str, component: str) -> tuple[str, str]:
    """Returns (weights, config) paths in converted model dir."""
    w = os.path.join(model_dir, f"{component}.safetensors")
    c = os.path.join(model_dir, f"{component}.json")
    return w, c
