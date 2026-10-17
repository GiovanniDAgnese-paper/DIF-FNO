#!/usr/bin/env python
"""Thin wrapper: see :mod:`dagnese_fno.cli.generate_data` (or run ``dif-fno-generate-data``)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dagnese_fno.cli.generate_data import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
