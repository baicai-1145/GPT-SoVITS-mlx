"""MLX weight loading: safetensors -> model param dicts with name mapping.

Memory notes (task-3, footprint work): the legacy loader materialized the
WHOLE file into a dict before model loaders copied arrays onto nn modules,
so peak residency was source + copy simultaneously. The loader now returns
a lazy mmap-backed mapping; tensors materialize on access and the backing
dict holds only metadata until then. Model loaders call ``release`` (or
just ``del`` the mapping) once ``eval_tree`` has realized the parameters.
"""

from __future__ import annotations

import gc
import json
import os
from collections.abc import Mapping

import mlx.core as mx
from safetensors import safe_open


class MappedSafetensors(Mapping[str, mx.array]):
    """Lazy, mmap-backed read-only safetensors mapping.

    ``get``/``__getitem__`` materialize one tensor at a time from disk, so
    loading never holds a second full copy of the weights in memory. Note
    the tensor returned by one ``get`` call is NOT cached: model loaders
    assign it straight onto module attributes (single owner), which keeps
    residency equal to the live model. A cached dict would only add a
    second reference and defeat the purpose.
    """

    def __init__(self, path: str):
        self._path = path
        self._file = safe_open(path, framework="mlx")
        self._keys = tuple(self._file.keys())

    def __getitem__(self, key: str) -> mx.array:
        return self._file.get_tensor(key)

    def __contains__(self, key: object) -> bool:
        return key in self._keys

    def __iter__(self):
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def get(self, key: str, default=None):
        if key in self._keys:
            return self._file.get_tensor(key)
        return default

    def keys(self):
        return self._keys


def load_mlx_safetensors(path: str) -> MappedSafetensors:
    """Lazily mmap the file; callers iterate/``get`` per key (see class doc)."""
    return MappedSafetensors(path)


def release(*_discarded) -> None:
    """Reclaim freed temporaries now.

    Callers must first drop their own references (``del``) — the objects
    passed here are not (cannot be) cleared. Frees the safetensors mmap
    handles and any loader-created reference cycles promptly instead of at
    some later allocation point.
    """
    gc.collect()


def trim_metal() -> None:
    """Return MLX's cached Metal buffers to the OS (no-op on CPU builds).

    Freed mx arrays go to MLX's internal buffer cache, NOT back to the OS,
    so the physical footprint only drops when the cache is flushed. Call at
    STAGE BOUNDARIES (after the stage's outputs are realized and its
    temporaries dropped) — clearing per-step would defeat buffer reuse and
    slow the loop down. Pure allocator hygiene: no math effect.
    """
    try:
        import mlx.core as _mx

        metal = getattr(_mx, "metal", None)
        if metal is not None and hasattr(metal, "clear_cache"):
            metal.clear_cache()
    except Exception:
        pass


def eval_tree(*roots) -> None:
    """mx.eval a whole object graph of arrays / containers / modules.

    Accepts mx.array, (nested) list/tuple/dict, and objects exposing
    ``parameters()`` (mlx.nn.Module) or arbitrary instance attributes that
    hold arrays (the ad-hoc parameter-holder style used across gsovits_mlx,
    e.g. Text2SemanticDecoder). Realizes everything reachable so later
    ``del`` can actually free it.
    """
    seen: set[int] = set()
    arrays: list[mx.array] = []

    def walk(obj) -> None:
        if obj is None or isinstance(obj, (str, bytes, int, float, bool)):
            return
        if isinstance(obj, mx.array):
            arrays.append(obj)
            return
        if hasattr(obj, "parameters") and callable(getattr(obj, "parameters")):
            for p in obj.parameters():
                walk(p)
            return
        oid = id(obj)
        if oid in seen:
            return
        seen.add(oid)
        if isinstance(obj, dict):
            for v in obj.values():
                walk(v)
        elif isinstance(obj, (list, tuple, set, frozenset)):
            for v in obj:
                walk(v)
        elif hasattr(obj, "__dict__"):
            for v in vars(obj).values():
                walk(v)

    for r in roots:
        walk(r)
    if arrays:
        mx.eval(arrays)


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def model_dir_paths(model_dir: str, component: str) -> tuple[str, str]:
    """Returns (weights, config) paths in converted model dir."""
    w = os.path.join(model_dir, f"{component}.safetensors")
    c = os.path.join(model_dir, f"{component}.json")
    return w, c
