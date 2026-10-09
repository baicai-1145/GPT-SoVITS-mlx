"""Flat-param loading for the s2 GAN trainer.

The repo venv is torch-free; training params come from npz intermediates
produced by tools/train_s2_extract.py (run once under the base venv).
Layout conventions are documented there (torch names, MLX conv layouts,
weight-norm kept as separate weight_g/weight_v pairs for live
reparameterization during training).
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np


def load_train_params_npz(path: str) -> dict:
    """npz -> {torch_name: mx.array fp32} (MLX conv layouts, (g,v) pairs)."""
    z = np.load(path)
    return {k: mx.array(z[k].astype(np.float32)) for k in z.files}
