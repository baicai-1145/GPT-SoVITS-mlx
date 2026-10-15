#!/usr/bin/env python
"""Forwarder to gsovits_mlx.cli.prepare_data."""
from __future__ import annotations

import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from gsovits_mlx.cli.prepare_data import *  # noqa: F403
from gsovits_mlx.cli.prepare_data import main

if __name__ == "__main__":
    main()
