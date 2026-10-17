#!/usr/bin/env python
"""Thin wrapper: see :mod:`dagnese_fno.cli.train` (or run ``dif-fno-train``)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dagnese_fno.cli.train import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
