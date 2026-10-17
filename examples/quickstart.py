#!/usr/bin/env python
"""Minimal end-to-end example: train DIF-FNO on a synthetic Darcy problem.

Run it with::

    python examples/quickstart.py

It generates 40 samples on the star domain at 32 x 32, trains the FNO baseline
and two DIF-FNO variants for a handful of epochs, and prints the topological
diagnostics that the benchmark tables are built from.
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
    tmp = Path(tempfile.mkdtemp(prefix="dif-fno-"))
    data = generate_dataset("star", n=32, n_samples=40, seed=0, verbose=False)
    save_dataset(data, tmp / "darcy_star.pt")
    print(f"dataset: {tuple(data['coeffs'].shape)} -> {tmp / 'darcy_star.pt'}")

    base = {"domain": "star", "epochs": 15, "batch_size": 8, "data_dir": str(tmp),
            "width": 16, "modes1": 8, "modes2": 8, "log_every": 5}

    variants = {
        "fno (baseline)": TrainConfig(model="fno", **dict(base)),
        "dif + barrier": TrainConfig(model="dif", **{**base, "lambda_barrier": 10.0, "tau": 0.05}),
        "dif certified": TrainConfig(model="dif", **{**base,
                                                    "lambda_barrier": 1.0,
                                                    "conditioning": "global",
                                                    "lipschitz_mode": "spectral",
                                                    "lipschitz_const": 0.9,
                                                    "deformation_scale": 0.5}),
    }

    print(f"\n{'variant':16s} {'rel L2':>8s} {'rel H1':>8s} {'fold %':>7s} "
          f"{'min detJ':>9s} {'|phi-Id|':>9s} {'certified':>10s}")
    for name, cfg in variants.items():
        res = train_one(cfg, verbose=False)
        f = res["final"]
        det = f"{f['min_detJ']:9.4f}" if f["min_detJ"] is not None else "        -"
        fold = f"{f['fold_pct']:7.2f}" if f["fold_pct"] is not None else "      -"
        disp = f"{f['displacement_mean']:9.4f}" if f["displacement_mean"] is not None else "        -"
        cert = (f"{f['certified_det_lower_bound']:10.4f}"
                if f.get("certified_det_lower_bound") is not None else "         -")
        print(f"{name:16s} {f['rel_l2']:8.4f} {f['rel_h1']:8.4f} {fold} {det} {disp} {cert}")

    print("\n`certified` is the provable lower bound det(J) >= (1 - kappa*L)^2;")
    print("it is positive only for the Lipschitz-constrained variant (see docs/THEORY.md).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
