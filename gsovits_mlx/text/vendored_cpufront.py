"""Official CPUFast text front-end, executed IN PLACE from that checkout.

The GPT-SoVITS-CPUFast checkout (GPT_SOVITS_CPUFAST env or explicit path)
is the single source of truth for text-front-end semantics — its modules
run read-only from that checkout so official changes are picked up
verbatim; this package adds no code copies.

The loader only exists because three official packages cannot import in
a torch-free runtime (guarded substitutions, details + the data-file
contract in docs/PARITY_NOTES.md §text front-end loader):
text.LangSegmenter (unguarded fast_langdetect import → our
lang_segmenter), text.g2pw (torch_api → injected g2pw_mlx), and
TTS_infer_pack (stub; stdlib-only text_segmentation_method.py executed
directly). Everything else — text/symbols/symbols2/phone_units/
tone_sandhi/zh_normalization/opencc_s2tw/jieba_posseg_fast/chinese/
chinese2/japanese/english/korean/cantonese/cleaner — is official code
executed as-is.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types

DEFAULT_CPUFAST_REPO = "/Users/baicai1145/repos/gpt-sovits/GPT-SoVITS-CPUFast"

_TEXT_PKG = "text"
_TTS_PKG = "TTS_infer_pack"

_LANGSEGMENTER_SUBS = ("text.LangSegmenter", "text.LangSegmenter.langsegmenter")
_G2PW_SUBS_STUB = ("text.g2pw.compact_pypinyin", "text.g2pw.pronunciation",
                   "text.g2pw.utils", "text.g2pw.dataset")

_loaded_repo: str | None = None
_MISSING = object()
_saved_modules: dict = {}


def _preserve_shadowed(name: str) -> None:
    """Remember a sys.modules entry the vendored load is about to
    shadow so it can be restored afterwards (see load_cpufront)."""
    if name not in _saved_modules:
        _saved_modules[name] = sys.modules.get(name)


def _exec_module(fullname: str, path: str) -> types.ModuleType:
    mod = types.ModuleType(fullname)
    mod.__file__ = path
    mod.__package__ = fullname.rpartition(".")[0]
    sys.modules[fullname] = mod
    src = open(path, encoding="utf-8").read()
    code = compile(src, path, "exec")
    exec(code, mod.__dict__)  # noqa: S102 - trusted local checkout
    return mod


def _stub_pkg(fullname: str, pkg_path: str) -> types.ModuleType:
    mod = types.ModuleType(fullname)
    mod.__path__ = [pkg_path]
    mod.__file__ = os.path.join(pkg_path, "__init__.py")
    mod.__package__ = fullname
    sys.modules[fullname] = mod
    return mod


def _real_pkg(fullname: str, pkg_path: str) -> types.ModuleType:
    """Execute the package's real __init__.py (safe for these packages)."""
    mod = _stub_pkg(fullname, pkg_path)
    init = os.path.join(pkg_path, "__init__.py")
    if os.path.exists(init):
        src = open(init, encoding="utf-8").read()
        exec(compile(src, init, "exec"), mod.__dict__)  # noqa: S102
    return mod


def _load_submodule(pkg: str, name: str) -> types.ModuleType:
    fullname = f"{pkg}.{name}"
    path = os.path.join(sys.modules[pkg].__path__[0], name + ".py")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{fullname}: expected module file at {path}")
    return _exec_module(fullname, path)


def cpufast_repo(explicit: str | None = None) -> str:
    repo = explicit or os.environ.get("GPT_SOVITS_CPUFAST", DEFAULT_CPUFAST_REPO)
    repo = os.path.abspath(repo)
    if not os.path.isdir(os.path.join(repo, "GPT_SoVITS", "text")):
        raise FileNotFoundError(
            f"GPT-SoVITS-CPUFast checkout not found at {repo} "
            "(set GPT_SOVITS_CPUFAST or pass --cpufast-repo)")
    return repo


def load_cpufront(explicit: str | None = None) -> str:
    """Register all official text front-end modules in sys.modules.

    Idempotent: returns the repo path on subsequent calls without redoing
    work. Raises RuntimeError if a foreign 'text' package is already loaded.
    """
    global _loaded_repo
    if _loaded_repo is not None:
        return _loaded_repo

    repo = cpufast_repo(explicit)
    # The official modules exec with CWD + "." in sys.path pointing at
    # the CPUFast repo; any same-named top-level package found there
    # (notably CPUFast tools/) must not shadow the caller's afterwards.
    _preserve_shadowed("tools")
    text_dir = os.path.join(repo, "GPT_SoVITS", "text")
    tts_dir = os.path.join(repo, "GPT_SoVITS", _TTS_PKG)

    existing = sys.modules.get(_TEXT_PKG)
    if existing is not None and not getattr(existing, "__gsovits_vendored__", False):
        raise RuntimeError(
            "a foreign 'text' package is already imported; the vendored "
            "CPUFast front-end must be loaded before anything imports 'text'")

    # --- top-level packages -------------------------------------------------
    pkg = _stub_pkg(_TEXT_PKG, text_dir)
    pkg.__gsovits_vendored__ = True
    _stub_pkg(_TTS_PKG, tts_dir)

    # --- stdlib-only helpers, dependency order ------------------------------
    _load_submodule(_TEXT_PKG, "symbols")
    _load_submodule(_TEXT_PKG, "symbols2")
    _real_pkg(_TEXT_PKG, text_dir)  # real __init__: cleaned_text_to_sequence
    _load_submodule(_TEXT_PKG, "phone_units")
    _load_submodule(_TEXT_PKG, "tone_sandhi")

    zn = _stub_pkg("text.zh_normalization", os.path.join(text_dir, "zh_normalization"))
    zn.__gsovits_vendored__ = True
    for name in ("constants", "char_convert", "chronology", "num",
                 "text_normlization"):
        _load_submodule("text.zh_normalization", name)

    _load_submodule(_TEXT_PKG, "opencc_s2tw")
    _load_submodule(_TEXT_PKG, "jieba_posseg_fast")

    # --- language modules (each resolves its own data via __file__) ---------
    for name in ("japanese", "english", "korean", "cantonese", "chinese"):
        _load_submodule(_TEXT_PKG, name)

    # --- chinese2 with the MLX g2pw in place of torch_api -------------------
    g2pw_dir = os.path.join(text_dir, "g2pw")
    g2pw_pkg = _stub_pkg("text.g2pw", g2pw_dir)
    g2pw_pkg.__gsovits_vendored__ = True
    for name in ("compact_pypinyin", "pronunciation", "utils", "dataset"):
        _load_submodule("text.g2pw", name)
    from gsovits_mlx.text import g2pw_mlx  # noqa: E402 - registers itself
    g2pw_mlx.register_as_torch_api()
    g2pw_mlx.ensure_converter_class()

    _load_submodule(_TEXT_PKG, "chinese2")
    _load_submodule(_TEXT_PKG, "cleaner")

    # --- LangSegmenter: our faithful re-port under the official names ------
    langseg_dir = os.path.join(text_dir, "LangSegmenter")
    ls_pkg = _stub_pkg("text.LangSegmenter", langseg_dir)
    ls_pkg.__gsovits_vendored__ = True
    from gsovits_mlx.text import lang_segmenter  # noqa: E402
    for sub in _LANGSEGMENTER_SUBS:
        sys.modules[sub] = lang_segmenter
    fast_dir = os.path.join(repo, "pretrained_models", "fast_langdetect")
    if os.path.isdir(fast_dir):
        lang_segmenter.configure(fast_dir)

    # --- text segmentation methods (cut0..cut5) -----------------------------
    _exec_module(f"{_TTS_PKG}.text_segmentation_method",
                 os.path.join(tts_dir, "text_segmentation_method.py"))

    _loaded_repo = repo
    prev = _saved_modules.get("tools", _MISSING)
    if prev is None or prev is _MISSING:
        sys.modules.pop("tools", None)
    else:
        sys.modules["tools"] = prev
    return repo


def front_end(explicit: str | None = None):
    """Convenience accessor returning the official callables used by preproc.

    Returns (clean_text, clean_text_with_phone_units, cleaned_text_to_sequence,
    splits, get_seg_method, split_big_text, LangSegmenter_texts).
    """
    load_cpufront(explicit)
    text = sys.modules[_TEXT_PKG]
    cleaner = importlib.import_module("text.cleaner")
    seg = importlib.import_module(f"{_TTS_PKG}.text_segmentation_method")
    return (
        cleaner.clean_text,
        cleaner.clean_text_with_phone_units,
        text.cleaned_text_to_sequence,
        seg.splits,
        seg.get_method,
        seg.split_big_text,
        sys.modules["text.LangSegmenter"].get_texts,
    )
