#!/usr/bin/env python3
"""Forwarder to gsovits_mlx.cli.e2e."""
from __future__ import annotations

import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from gsovits_mlx.cli.e2e import *  # noqa: F403
from gsovits_mlx.cli.e2e import main

if __name__ == "__main__":
    main()
