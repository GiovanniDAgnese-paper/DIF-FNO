#!/usr/bin/env python
"""Deprecated entry point -- superseded by :mod:`scripts.generate_data.py`.

Historically this script wrote ``real_data/darcy_<domain>.pt``; that behaviour is
preserved (``--out`` defaults to ``real_data``) so old commands keep working::

    python scripts/generate_real_data.py          # == python scripts/generate_data.py --out real_data
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dagnese_fno.cli.generate_data import main  # noqa: E402


def main_legacy(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--out" not in argv and "-o" not in argv:
        argv = ["--out", "real_data"] + argv
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(main_legacy())
