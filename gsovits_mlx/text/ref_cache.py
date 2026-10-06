"""Disk cache for (ref_audio, ref_text) prompt front-end artifacts.

The reference-audio prompt (HuBERT semantic codes + reference spec) and its
text front-end (phones + bert) depend only on (ref_audio file, ref_text,
prompt_lang, cleaner version) — recomputing them on every synthesis is the
single biggest fixed cost of repeated inference with the same reference
(the dominant real-world pattern), so the results are cached on disk keyed
by an mtime+size content proxy of the audio file plus the full input
parameters.

Cache layout (one JSON + one npz per key, atomic tmp-rename writes):
    <cache_dir>/prompt/<key>.json   {"meta": ...}   — human-readable, hit check
    <cache_dir>/prompt/<key>.npz    arrays

The key is sha256 over: path basename, file size, int(mtime * 1000),
ref_text, prompt_lang, version, format tag. (mtime+size is the standard
cheap proxy; the file content itself is NOT hashed — use clear() or a new
cache_dir when a reference file changes in place within the same millisecond
and identical size.)

The frontend cache stores phones/bert; the audio-side artifacts
(prompt_semantic codes, refer spec) are cached by the caller with the same
helper (`cached_call`), because their computation needs the models.

All functions are stdlib-only and safe to import without mlx/official deps
(loading happens only when the caller evaluates the returned payloads).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from typing import Any, Callable

FORMAT_TAG = "gsovits-mlx/refcache-v2"

_ENV_FINGERPRINT: str | None = None


def env_fingerprint() -> str:
    """Interpreter/env fingerprint for cache invalidation.

    MLX kernel ulps AND numpy float paths differ across patch versions and
    flip AR near-ties (see BENCH.md anchor contract: the seeded baselines
    are only valid within one (python, mlx, numpy, device) quadruple), so
    cached prompt artifacts (phones/bert/codes) must never be reused across
    a version change. Cached after first call (importlib.metadata lookups
    are not free and this sits on the cache-hit path).
    """
    global _ENV_FINGERPRINT
    if _ENV_FINGERPRINT is None:
        import importlib.metadata as im
        import sys

        parts = [sys.version.split()[0]]
        for dist in ("mlx", "mlx-metal", "numpy"):
            try:
                parts.append(im.version(dist))
            except im.PackageNotFoundError:
                parts.append("-")
        _ENV_FINGERPRINT = "/".join(parts)
    return _ENV_FINGERPRINT


def cache_key(ref_audio: str, ref_text: str, prompt_lang: str,
              version: str, extra: str = "") -> str:
    st = os.stat(ref_audio)
    h = hashlib.sha256()
    h.update(os.path.basename(ref_audio).encode())
    h.update(str(st.st_size).encode())
    h.update(str(int(st.st_mtime * 1000)).encode())
    h.update(ref_text.encode())
    h.update(prompt_lang.encode())
    h.update(version.encode())
    h.update(FORMAT_TAG.encode())
    h.update(env_fingerprint().encode())
    if extra:
        h.update(extra.encode())
    return h.hexdigest()[:32]


def cache_dir(root: str | None = None) -> str:
    base = root or os.environ.get("GSOVITS_REF_CACHE",
                                  os.path.join(os.path.expanduser("~"),
                                               ".cache", "gsovits-mlx", "refcache"))
    os.makedirs(os.path.join(base, "prompt"), exist_ok=True)
    return base


def get_cached(cache_dir_: str, key: str) -> tuple[dict, dict] | None:
    """Returns (meta, arrays_dict) or None. arrays values are np arrays or
    JSON-safe scalars depending on what was stored."""
    meta_path = os.path.join(cache_dir_, "prompt", key + ".json")
    npz_path = os.path.join(cache_dir_, "prompt", key + ".npz")
    try:
        with open(meta_path) as fh:
            meta = json.load(fh)
    except (OSError, ValueError):
        return None
    if meta.get("format") != FORMAT_TAG:
        return None
    if not os.path.exists(npz_path):
        return None
    import numpy as np

    with np.load(npz_path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    return meta, arrays


def put_cached(cache_dir_: str, key: str, meta: dict,
               arrays: dict[str, Any]) -> None:
    import numpy as np

    d = os.path.join(cache_dir_, "prompt")
    os.makedirs(d, exist_ok=True)
    meta = dict(meta, format=FORMAT_TAG, env=env_fingerprint())
    fd, tmp_json = tempfile.mkstemp(dir=d, suffix=".json")
    with os.fdopen(fd, "w") as fh:
        json.dump(meta, fh)
    fd, tmp_npz = tempfile.mkstemp(dir=d, suffix=".npz")
    os.close(fd)
    np.savez(tmp_npz, **{k: v for k, v in arrays.items()})
    os.replace(tmp_json, os.path.join(d, key + ".json"))
    os.replace(tmp_npz, os.path.join(d, key + ".npz"))


def cached_call(cache_dir_: str | None, key: str,
                meta_fn: Callable[[], dict],
                compute_fn: Callable[[], dict[str, Any]]) -> tuple[dict, dict, bool]:
    """Get-or-compute: returns (meta, arrays, hit). compute_fn returns the
    arrays dict; meta_fn builds the metadata (only called on miss)."""
    root = cache_dir(cache_dir_)
    hit = get_cached(root, key)
    if hit is not None:
        meta, arrays = hit
        return meta, arrays, True
    meta = meta_fn()
    arrays = compute_fn()
    try:
        put_cached(root, key, meta, arrays)
    except OSError:
        pass  # cache write failures never break inference
    return meta, arrays, False


def clear(cache_dir_: str | None = None) -> int:
    """Delete all cached entries; returns the number removed."""
    d = os.path.join(cache_dir(cache_dir_), "prompt")
    n = 0
    for name in os.listdir(d):
        try:
            os.unlink(os.path.join(d, name))
            n += 1
        except OSError:
            pass
    return n // 2  # json+npz pairs
