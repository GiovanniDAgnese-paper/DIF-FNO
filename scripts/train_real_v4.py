#!/usr/bin/env python
"""Deprecated entry point -- superseded by :mod:`scripts.train.py`.

It reproduces the DIF-FNO v4 invocation (60 epochs, softplus barrier,
``lambda_barrier = 10``, ``tag = v4``) and the historical result filename
``results/real_<model>_<domain>_seed<seed>_<tag>.json``.

Two conventions changed in 1.1.0 and are *not* restored by this shim, because
they were bugs rather than choices (see ``docs/THEORY.md``):

* ``det J`` is now normalised by the grid spacing, so ``tau`` is expressed in
  physical units (identity :math:`\\Rightarrow \\det J = 1`);
* the MSE is averaged over the masked cells instead of the bounding box.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dagnese_fno.cli.train import config_from_args, parse_args  # noqa: E402
from dagnese_fno.training import save_result, train_one  # noqa: E402


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    cfg = config_from_args(args)
    result = train_one(cfg)
    path = save_result(
        result, cfg.out_dir,
        filename=f"real_{cfg.model}_{cfg.domain}_seed{cfg.seed}_{cfg.tag}.json",
    )
    print("Saved:", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
