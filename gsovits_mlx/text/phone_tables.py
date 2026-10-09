"""Symbol tables for s2 training data (text/cleaned_text_to_sequence).

The id mapping is a pure function of the symbol LIST order in the official
text/symbols.py (v1) and text/symbols2.py (v2): ids = index in
[pad] + c + v + ja_symbols + pu_symbols + arpa (+ sorted ko/yue for v2).
Rather than vendoring the full lists here, this module loads the symbol
tables from an official checkout via the vendored front-end loader (which is
stdlib-only for symbols/symbols2) or falls back to exec'ing the two files
directly from a given repo path.

The v2 table has 732 symbols and v1 322 (== the embedding rows in the
checkpoints); this is asserted on load.
"""

from __future__ import annotations

import os
import sys

_EXPECTED_LEN = {"v1": 322, "v2": 732}


def _symbols_from_repo(repo: str, version: str) -> list:
    fname = "symbols.py" if version == "v1" else "symbols2.py"
    path = os.path.join(repo, "GPT_SoVITS", "text", fname)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    mod = {}
    with open(path, encoding="utf8") as f:
        exec(compile(f.read(), path, "exec"), mod)  # noqa: S102
    return mod["symbols"]


def symbol_to_id(version: str = "v2", repo: str | None = None) -> dict:
    """{symbol: id} matching the official _symbol_to_id table."""
    version = "v1" if version == "v1" else "v2"
    if repo is not None:
        symbols = _symbols_from_repo(repo, version)
    else:
        try:
            from gsovits_mlx.text import vendored_cpufront
            symbols = _symbols_from_repo(vendored_cpufront.DEFAULT_CPUFAST_REPO,
                                         version)
        except Exception:
            env = os.environ.get("GPT_SOVITS_CPUFAST")
            if not env:
                raise
            symbols = _symbols_from_repo(env, version)
    table = {s: i for i, s in enumerate(symbols)}
    assert len(table) == _EXPECTED_LEN[version], (version, len(table))
    return table
