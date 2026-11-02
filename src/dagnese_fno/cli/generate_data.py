"""Generate the Darcy-flow datasets used by the DIF-FNO benchmark.

Examples
--------
Generate the three original domains at the reference resolution::

    python scripts/generate_data.py --domains star lshape annulus --n 64 --samples 100

Add the airfoil domain and write to a custom folder::

    python scripts/generate_data.py --domains naca0012 --out data --n 64 --samples 100
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..data import DOMAINS, generate_dataset, mask_summary, save_dataset


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--domains", nargs="+", default=sorted(DOMAINS), choices=sorted(DOMAINS),
                   help="domains to generate (default: all)")
    p.add_argument("--n", type=int, default=64, help="grid resolution n x n (default: 64)")
    p.add_argument("--samples", type=int, default=100, help="number of (a, u) pairs (default: 100)")
    p.add_argument("--out", type=str, default="data", help="output directory (default: data)")
    p.add_argument("--seed", type=int, default=None,
                   help="override the per-domain default seed")
    p.add_argument("--n-modes", type=int, default=6, help="Fourier modes of the coefficient field")
    p.add_argument("--contrast", type=float, default=2.0,
                   help="log-contrast gamma of a = exp(gamma * a_hat)")
    p.add_argument("--forcing", type=float, default=1.0, help="constant right-hand side f")
    p.add_argument("--overwrite", action="store_true", help="regenerate even if the file exists")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Merge into an existing manifest, so that generating a subset of the
    # domains (``--domains slit``) does not erase the record of the others.
    manifest_path = out_dir / "manifest.json"
    manifest = {"generated": {}}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            manifest.setdefault("generated", {})
        except json.JSONDecodeError:  # pragma: no cover - corrupt manifest
            manifest = {"generated": {}}
    manifest["args"] = vars(args)

    for domain in args.domains:
        seed = args.seed if args.seed is not None else DOMAINS[domain].default_seed
        target = out_dir / f"darcy_{domain}.pt"
        if target.exists() and not args.overwrite:
            print(f"[skip] {target} already exists (use --overwrite to regenerate)")
            continue
        print(f"\n=== {domain.upper()}  n={args.n}  samples={args.samples}  seed={seed} ===", flush=True)
        t0 = time.time()
        data = generate_dataset(
            domain, n=args.n, n_samples=args.samples, seed=seed,
            n_modes=args.n_modes, contrast=args.contrast, forcing=args.forcing,
        )
        save_dataset(data, target)
        info = mask_summary(np.asarray(data["mask"]))
        info.update({
            "seed": seed,
            "seconds": round(time.time() - t0, 2),
            "coeffs_shape": list(data["coeffs"].shape),
            "coeff_min": float(data["coeffs"].min()),
            "coeff_max": float(data["coeffs"].max()),
            "target_max": float(data["targets"].max()),
            "spacing": data["spacing"],
        })
        manifest["generated"][domain] = info
        print(f"  -> {target}  ({info['seconds']}s, fill={info['fill_fraction']:.3f})")

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"\nDONE. Manifest: {out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
