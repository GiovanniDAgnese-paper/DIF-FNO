"""Dataset generation, storage and loading.

A dataset is a dictionary of tensors plus a JSON-serialisable ``spec``:

``coeffs``   ``(N, n, n)`` float32   coefficient fields ``a``
``targets``  ``(N, n, n)`` float32   Darcy solutions ``u``
``mask``     ``(n, n)``    bool      active nodes of the domain
``coords``   ``(n, n, 2)`` float32   node coordinates (``'ij'`` order)
``spacing``  float                  physical grid spacing ``2 / (n - 1)``
``spec``     dict                   domain, resolution, seeds, generator settings

Everything is seeded: the same ``(domain, n, n_samples, seed)`` gives
bit-identical tensors, which the test suite checks.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from .darcy import sample_coeff_field, solve_darcy
from .masks import DOMAINS, get_mask, mask_summary

__all__ = [
    "generate_dataset",
    "save_dataset",
    "load_dataset",
    "reference_coords",
    "dataset_path",
]


def reference_coords(n: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """``(n, n, 2)`` coordinates of the reference grid on :math:`[-1,1]^2`, ``'ij'`` order."""
    t = torch.linspace(-1.0, 1.0, n, dtype=dtype)
    return torch.stack(torch.meshgrid(t, t, indexing="ij"), dim=-1)


def dataset_path(data_dir: str | Path, domain: str) -> Path:
    """Canonical location of the dataset of one domain."""
    return Path(data_dir) / f"darcy_{domain}.pt"


def generate_dataset(
    domain: str,
    n: int = 64,
    n_samples: int = 100,
    seed: int | None = None,
    n_modes: int = 6,
    contrast: float = 2.0,
    forcing: float = 1.0,
    verbose: bool = False,
) -> dict:
    """Generate ``n_samples`` Darcy pairs on ``domain``.

    Parameters
    ----------
    domain:
        A key of :data:`dagnese_fno.data.masks.DOMAINS`.
    n:
        Grid resolution (``n x n`` nodes on the box :math:`[-1,1]^2`).
    n_samples:
        Number of ``(a, u)`` pairs.
    seed:
        Seed of the coefficient-field generator; defaults to the per-domain
        seed of :data:`~dagnese_fno.data.masks.DOMAINS`.
    contrast:
        Log-contrast :math:`\\gamma` of ``a = exp(gamma * z)``.
    forcing:
        Constant right-hand side ``f``.
    """
    if domain not in DOMAINS:
        raise KeyError(f"unknown domain '{domain}'; available: {sorted(DOMAINS)}")
    if n < 5:
        raise ValueError("n must be >= 5")
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")

    spec = DOMAINS[domain]
    seed = spec.default_seed if seed is None else int(seed)
    mask = get_mask(domain, n)
    h = 2.0 / (n - 1)
    rng = np.random.default_rng(seed)
    f = np.full((n, n), float(forcing))

    coeffs = np.empty((n_samples, n, n), dtype=np.float32)
    targets = np.empty((n_samples, n, n), dtype=np.float32)
    t0 = time.time()
    for k in range(n_samples):
        a = sample_coeff_field(n, mask, rng, n_modes=n_modes, contrast=contrast)
        u = solve_darcy(mask, a, h, f)
        coeffs[k] = a.astype(np.float32)
        targets[k] = u.astype(np.float32)
        if verbose and (k + 1) % max(1, n_samples // 5) == 0:
            print(f"  [{domain}] {k + 1}/{n_samples} pairs "
                  f"({time.time() - t0:.1f}s)", flush=True)

    return {
        "coeffs": torch.from_numpy(coeffs),
        "targets": torch.from_numpy(targets),
        "mask": torch.from_numpy(mask),
        "coords": reference_coords(n),
        "spacing": float(h),
        "spec": {
            "domain": domain,
            "description": spec.description,
            "n": int(n),
            "n_samples": int(n_samples),
            "seed": int(seed),
            "n_modes": int(n_modes),
            "contrast": float(contrast),
            "forcing": float(forcing),
            "spacing": float(h),
            "mask": mask_summary(mask),
            "generator": "finite-volume TPFA with harmonic face conductivity",
        },
    }


def save_dataset(data: dict, path: str | Path) -> Path:
    """Store a dataset (``.pt``) together with its JSON spec (sidecar ``.json``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "coeffs": data["coeffs"].contiguous(),
        "targets": data["targets"].contiguous(),
        "mask": data["mask"].to(torch.bool).contiguous(),
        "coords": data["coords"].contiguous(),
        "spacing": float(data["spacing"]),
        "spec": data.get("spec", {}),
    }
    torch.save(payload, path)
    path.with_suffix(".json").write_text(json.dumps(payload["spec"], indent=2, default=str))
    return path


def load_dataset(path: str | Path) -> dict:
    """Load a dataset written by :func:`save_dataset` (reconstructing missing fields)."""
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "coords" not in payload or payload["coords"] is None:
        n = int(payload["coeffs"].shape[-1])
        payload["coords"] = reference_coords(n)
    payload["mask"] = payload["mask"].to(torch.bool)
    payload.setdefault("spec", {})
    return payload
