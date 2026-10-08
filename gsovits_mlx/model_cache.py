"""Bounded residency for immutable inference network resources.

One entry per category; SoVITS versions share a slot. No input/output caching.
Existing callers keep their model; replacing an export affects the next load.
Cache bypass clears all categories. Callers must release old model references
on switches; the cache cannot bound resources retained by those callers.
"""
from functools import wraps
import os
from pathlib import Path
from threading import RLock

import mlx.core as mx

_MODELS = {}
_LOCK = RLock()


def clear_model_cache():
    """Drop resident network references; does not clear frontend resources."""
    with _LOCK:
        _MODELS.clear()


def resident_model(category, *, patterns=("*.safetensors", "*.json")):
    """Cache immutable models; reload changed exports, never mutate in place.

    patterns must cover all files read by the loader, including nested exports.
    The defaults cover the six converted inference loaders in pipeline.py.
    """
    def decorate(load):
        @wraps(load)
        def cached(path, *args, **kwargs):
            if os.environ.get("GSOVITS_MODEL_CACHE", "1") == "0":
                clear_model_cache()
                return load(path, *args, **kwargs)
            root = Path(path).resolve()
            files = tuple((str(f), s.st_dev, s.st_ino, s.st_mtime_ns, s.st_size)
                          for f in sorted({f for pattern in patterns for f in root.glob(pattern)})
                          for s in (f.stat(),))
            # AR's mixed kernel choice is captured when blocks are constructed.
            ar_mode = (os.environ.get("GSOVITS_AR_MIXED_GEMV", "1"),
                       os.environ.get("GSOVITS_AR_PROMOTE_CACHE", "1")) if category == "gpt" else ()
            key = (load.__module__, load.__qualname__, str(root), files, args,
                   tuple(sorted(kwargs.items())), str(mx.default_device()), ar_mode)
            with _LOCK:
                entry = _MODELS.get(category)
                if entry is not None and entry[0] == key:
                    return entry[1]
                # Evict before loading the replacement to limit peak residency.
                _MODELS.pop(category, None)
                model = load(str(root), *args, **kwargs)
                _MODELS[category] = (key, model)
                return model
        return cached
    return decorate
