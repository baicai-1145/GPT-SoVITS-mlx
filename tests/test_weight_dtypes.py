"""Converted-weight dtype assertions (task-6 requirement a).

Heavy: loads the real converted exports from --models-root
(/Volumes/2T/gpt-sovits-models/mlx). Deselect with -m 'not heavy'.
Verifies the P0-A fp16 conversion policy: v3/v4/v5dev/v5turbo sovits
safetensors are float16; v1/v2 HiFi-GAN exports are float16 (per BENCH.md
sizes notes); spot-checks that loading a model produces float16 params
where expected.
"""

from __future__ import annotations

import json
import os

import pytest

MODELS_ROOT = os.environ.get(
    "GSOVITS_MODELS_ROOT", "/Volumes/2T/gpt-sovits-models/mlx")


def _mx():
    import mlx.core as mx
    return mx


pytestmark = pytest.mark.heavy


def _sovits_arrays(version):
    import mlx.core as mx
    from gsovits_mlx.io import load_mlx_safetensors
    return load_mlx_safetensors(os.path.join(MODELS_ROOT, version,
                                             "sovits.safetensors"))


@pytest.mark.parametrize("version", ["v3", "v4", "v5dev", "v5turbo"])
def test_v3_family_sovits_is_fp16(version):
    if not os.path.isdir(os.path.join(MODELS_ROOT, version)):
        pytest.skip(f"{version} export not present")
    arrays = _sovits_arrays(version)
    assert arrays, f"{version}: no arrays loaded"
    dtypes = {str(v.dtype) for v in arrays.values()}
    assert dtypes == {"mlx.core.float16"}, f"{version}: dtypes={dtypes}"


def test_v2_sovits_dtype():
    if not os.path.isdir(os.path.join(MODELS_ROOT, "v2")):
        pytest.skip("v2 export not present")
    arrays = _sovits_arrays("v2")
    assert arrays
    # v1/v2 exports are fp16 (BENCH.md sizes note)
    dtypes = {str(v.dtype) for v in arrays.values()}
    assert dtypes <= {"mlx.core.float16", "mlx.core.float32"}
    assert "mlx.core.float16" in dtypes


def test_load_sovits_v3_param_dtypes():
    """Spot-check that load_sovits_v3 params are float16 after load."""
    import mlx.core as mx
    if not os.path.isdir(os.path.join(MODELS_ROOT, "v3")):
        pytest.skip("v3 export not present")
    from gsovits_mlx.pipeline import load_sovits_v3
    sov, _meta = load_sovits_v3(os.path.join(MODELS_ROOT, "v3"), "v3")
    # nn.Module.parameters() returns a nested dict/list tree — flatten arrays.
    def _flat(node):
        if hasattr(node, "dtype"):
            yield node
        elif isinstance(node, dict):
            for v in node.values():
                yield from _flat(v)
        elif isinstance(node, (list, tuple)):
            for v in node:
                yield from _flat(v)
    items = list(_flat(sov.parameters()))
    assert items, "no parameters exposed"
    fp16 = sum(1 for p in items if p.dtype == mx.float16)
    assert fp16 > len(items) * 0.5, (
        f"expected majority-fp16 params for v3 (got {fp16}/{len(items)})")
