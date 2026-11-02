#!/usr/bin/env python
"""Minimal end-to-end example: the D'Agnese Chart Operator on a Darcy problem.

Run it with::

    python examples/quickstart.py

It generates 40 samples on the star domain at 32 x 32, trains the FNO baseline
and three chart-operator variants for a handful of epochs, and prints the
guarantees next to the accuracy, so that the reader can see which number is a
theorem and which is not.

Nothing is written outside a temporary directory.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dagnese_fno.data import generate_dataset, save_dataset  # noqa: E402
from dagnese_fno.training import TrainConfig, train_one  # noqa: E402


def main() -> int:
    torch.manual_seed(0)
    tmp = Path(tempfile.mkdtemp(prefix="dagnese-dao-"))
    data = generate_dataset("star", n=32, n_samples=40, seed=0, verbose=False)
    save_dataset(data, tmp / "darcy_star.pt")
    print(f"dataset: {tuple(data['coeffs'].shape)} -> {tmp / 'darcy_star.pt'}")

    base = {"domain": "star", "epochs": 15, "batch_size": 8, "data_dir": str(tmp),
            "width": 16, "modes1": 8, "modes2": 8, "log_every": 100}

    variants = {
        "fno (baseline)": TrainConfig(model="fno", **base),
        "dao (chart)": TrainConfig(model="chart", **{**base, "lambda_eq": 0.0}),
        "dao + eq loss": TrainConfig(model="chart", **{**base, "lambda_eq": 10.0}),
        "dao per-sample": TrainConfig(model="chart", **{**base, "lambda_eq": 10.0,
                                                       "chart_conditioning": "global",
                                                       "latent_dim": 8}),
    }

    print(f"\n{'variant':16s} {'rel L2':>8s} {'rel H1':>8s} {'fold %':>7s} "
          f"{'min detJ':>9s} {'cert. lb':>9s} {'inv err':>8s} {'eq res %':>9s}")
    for name, cfg in variants.items():
        result = train_one(cfg, verbose=False)
        f = result["final"]
        print(f"{name:16s} {f['rel_l2']:8.4f} {f['rel_h1']:8.4f} "
              f"{(f.get('fold_pct') or 0.0):7.2f} {(f.get('min_detJ') or float('nan')):9.4f} "
              f"{(f.get('certified_det_lower_bound') or 0.0):9.2e} "
              f"{(f.get('chart_inverse_error') or 0.0):8.1e} "
              f"{(f.get('eq_residual_pct') or 0.0):9.1f}")

    print("""
Read the columns like this:
  rel L2 / H1 : measured accuracy (15 epochs on 40 samples: a sanity check, not a claim)
  fold %      : measured fraction of folded cells -- zero by construction for the DAO
  min detJ    : measured minimum of det J over the grid
  cert. lb    : exact lower bound on the *trained* knots; the parameter-free
                floor eps^2 = 1e-6 holds for every parameter value
  inv err     : max |Phi^-1(Phi(xi)) - xi| of the closed-form inverse
  eq res %    : RMS deviation of the cell mass from the mean (the equidistribution loss)
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
