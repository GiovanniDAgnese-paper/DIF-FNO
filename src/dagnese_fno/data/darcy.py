"""Darcy flow on a masked domain: coefficient sampling and finite-volume solver.

The benchmark solves, on each domain :math:`\\Omega \\subset [-1,1]^2`,

.. math:: -\\nabla\\cdot(a\\,\\nabla u) = f \\quad \\text{in } \\Omega,
          \\qquad u = 0 \\quad \\text{on } \\partial\\Omega,

with :math:`f \\equiv \\mathrm{const}` and a *log-normal* permeability

.. math:: a(x) = \\exp\\bigl(\\gamma\\,\\hat a(x)\\bigr), \\qquad
          \\hat a = \\sum_{k,l=1}^{M} c_{kl}\\,\\sin(\\pi k u)\\sin(\\pi l v),
          \\quad c_{kl} \\sim \\mathcal N(0,\\,(k^2+l^2)^{-2}),

normalised so that :math:`\\max_\\Omega \\lvert \\hat a \\rvert = 1`, i.e. the
log-contrast of :math:`a` is exactly :math:`2\\gamma`.  The spectral decay of
the coefficients makes :math:`\\hat a` smooth (a truncated Karhunen--Loeve-type
expansion), so :math:`\\nabla a` is finite and the conditioning constant
:math:`\\kappa = \\sqrt{1 + \\lVert \\nabla a \\rVert_\\infty^2}` of the
certificate is well defined.

The PDE is discretised with the **5-point finite-volume / flux** stencil

.. math:: \\frac{1}{h^2}\\sum_{j \\sim i} a_{ij}\\,(u_i - u_j) = f_i,
          \\qquad a_{ij} = \\tfrac12\\,(a_i + a_j),

on the unknown nodes (those whose four neighbours are active); the first
inactive layer carries the homogeneous Dirichlet condition.  The matrix is
symmetric positive definite and the scheme is second-order accurate for smooth
:math:`a` (verified by a manufactured-solution test in ``tests/test_data.py``).
Everything is driven by a ``numpy`` ``Generator``, so a dataset is a pure
function of ``(domain, n, n_samples, seed, n_modes, contrast, forcing)``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from ..utils.provenance import file_sha256
from .masks import DOMAINS, grid_2d, mask_summary

__all__ = [
    "sample_coeff_field",
    "assemble_darcy",
    "solve_darcy",
    "generate_dataset",
    "save_dataset",
    "load_dataset",
]

#: Neighbour offsets of the 5-point stencil (down, up, left, right).
_STENCIL = ((1, 0), (-1, 0), (0, 1), (0, -1))


# --------------------------------------------------------------------------- #
# Permeability
# --------------------------------------------------------------------------- #
def sample_coeff_field(n: int, mask: np.ndarray, rng: np.random.Generator,
                       n_modes: int = 6, contrast: float = 2.0) -> np.ndarray:
    """Draw one log-normal permeability field :math:`a = \\exp(\\gamma\\,\\hat a)`.

    Outside the mask the field is set to ``1`` (it is never used by the solver,
    but keeping it finite avoids NaN in the finite differences).

    Returns a ``(n, n)`` float array with ``min a > 0`` and log-contrast
    :math:`\\le 2\\gamma` on the active cells.
    """
    if n_modes < 1:
        raise ValueError(f"n_modes must be >= 1, got {n_modes}")
    if contrast < 0:
        raise ValueError(f"contrast must be non-negative, got {contrast}")
    m = np.asarray(mask, dtype=bool)
    if m.shape != (n, n):
        raise ValueError(f"mask has shape {m.shape}, expected ({n}, {n})")

    x, _, _ = grid_2d(n)
    knots = 0.5 * (x + 1.0)                              # [-1, 1] -> [0, 1]
    modes = np.arange(1, n_modes + 1, dtype=float)
    basis = np.sin(np.pi * np.outer(knots, modes))       # (n, M): sin(pi k u)

    sigma = 1.0 / (modes[:, None] ** 2 + modes[None, :] ** 2)
    coeffs = rng.normal(0.0, 1.0, size=(n_modes, n_modes)) * sigma
    a_hat = basis @ coeffs @ basis.T                     # (n, n)

    if m.any():
        a_hat = a_hat / max(np.abs(a_hat[m]).max(), 1e-300)
    a = np.ones((n, n), dtype=float)
    a[m] = np.exp(contrast * a_hat[m])
    return a


# --------------------------------------------------------------------------- #
# Discretisation
# --------------------------------------------------------------------------- #
def assemble_darcy(mask: np.ndarray, a, h: float, f=None):
    """Assemble the 5-point flux discretisation of :math:`-\\nabla\\cdot(a\\nabla u) = f`.

    Parameters
    ----------
    mask:
        ``(n, n)`` boolean array of active cells.
    a:
        ``(n, n)`` permeability, or a scalar (constant field).
    h:
        Grid spacing (the same in both directions).
    f:
        ``(n, n)`` right-hand side, a scalar, or ``None`` (``f = 1``).

    Returns
    -------
    ``(A, b, interior, idx)`` where ``A`` is a symmetric positive definite
    ``scipy.sparse`` matrix of size ``interior.sum()``, ``b`` the right-hand
    side, ``interior`` the ``(n, n)`` boolean mask of the unknowns (active cells
    whose four neighbours are active) and ``idx`` an ``(n, n)`` integer array
    mapping a grid node to its unknown number (``-1`` when it is not unknown).
    Unknowns adjacent to an inactive cell see the Dirichlet value ``u = 0``
    there, which contributes to the diagonal only.
    """
    m = np.asarray(mask, dtype=bool)
    n = m.shape[0]
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(f"mask must be square, got shape {m.shape}")
    if h <= 0:
        raise ValueError(f"spacing must be positive, got {h}")

    a = np.broadcast_to(np.asarray(a, dtype=float), (n, n)).astype(float, copy=True)
    if f is None:
        rhs = np.ones((n, n), dtype=float)
    else:
        rhs = np.broadcast_to(np.asarray(f, dtype=float), (n, n)).astype(float, copy=True)
    if (a <= 0).any():
        raise ValueError("the permeability must be strictly positive")

    # Unknowns: active cells whose whole 5-point stencil is active.
    interior = np.zeros((n, n), dtype=bool)
    interior[1:-1, 1:-1] = (m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1]
                            & m[1:-1, :-2] & m[1:-1, 2:])
    ii, jj = np.nonzero(interior)
    n_unknowns = int(ii.size)
    idx = -np.ones((n, n), dtype=np.int64)
    idx[ii, jj] = np.arange(n_unknowns)

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    unknown_ids = np.arange(n_unknowns)
    inv_h2 = 1.0 / (h * h)

    for di, dj in _STENCIL:
        ni, nj = ii + di, jj + dj
        a_face = 0.5 * (a[ii, jj] + a[ni, nj]) * inv_h2
        rows.append(unknown_ids)                 # diagonal: sum of the face weights
        cols.append(unknown_ids)
        vals.append(a_face)
        is_unknown = interior[ni, nj]
        if is_unknown.any():
            rows.append(unknown_ids[is_unknown])
            cols.append(idx[ni[is_unknown], nj[is_unknown]])
            vals.append(-a_face[is_unknown])

    if n_unknowns:
        rows_arr = np.concatenate(rows)
        cols_arr = np.concatenate(cols)
        vals_arr = np.concatenate(vals)
    else:
        rows_arr = np.zeros(0, dtype=np.int64)
        cols_arr = np.zeros(0, dtype=np.int64)
        vals_arr = np.zeros(0, dtype=float)
    A = coo_matrix((vals_arr, (rows_arr, cols_arr)), shape=(n_unknowns, n_unknowns)).tocsr()
    A.sum_duplicates()
    return A, rhs[ii, jj].astype(float), interior, idx


def solve_darcy(mask: np.ndarray, a, h: float, f=None) -> np.ndarray:
    """Solve :math:`-\\nabla\\cdot(a\\nabla u) = f` and return the ``(n, n)`` field.

    The solution is ``0`` on every inactive cell (and on the Dirichlet ring).
    """
    A, b, interior, _ = assemble_darcy(mask, a, h, f)
    u = np.zeros(np.asarray(mask).shape, dtype=float)
    if b.size:
        u[interior] = spsolve(A.tocsc(), b)
    return u


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
def generate_dataset(domain: str, n: int = 64, n_samples: int = 100,
                     seed: int | None = None, n_modes: int = 6,
                     contrast: float = 2.0, forcing: float = 1.0,
                     dtype: torch.dtype = torch.float32,
                     verbose: bool = False) -> dict:
    """Build the ``(a, u)`` pairs of one domain.

    The dataset is a deterministic function of its arguments: one
    ``numpy`` generator seeded with ``seed`` drives all the samples, so the same
    call returns bitwise-identical tensors (and, once saved, bitwise-identical
    files).  This is what makes the benchmark reproducible without shipping the
    binary data.

    Returns a dict with ``coeffs`` ``(S, n, n)``, ``targets`` ``(S, n, n)``,
    ``coords`` ``(n, n, 2)``, ``mask`` ``(n, n)`` bool, ``spacing`` and ``spec``.
    """
    if domain not in DOMAINS:
        raise KeyError(f"unknown domain {domain!r}; available: {sorted(DOMAINS)}")
    spec = DOMAINS[domain]
    seed = spec.default_seed if seed is None else int(seed)
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")

    mask = np.asarray(spec.mask_fn(n), dtype=bool)
    h = 2.0 / (n - 1)
    rng = np.random.default_rng(seed)

    coeffs = np.empty((n_samples, n, n), dtype=np.float64)
    targets = np.empty((n_samples, n, n), dtype=np.float64)
    for s in range(n_samples):
        a = sample_coeff_field(n, mask, rng, n_modes=n_modes, contrast=contrast)
        coeffs[s] = a
        targets[s] = solve_darcy(mask, a, h, f=forcing)
        if verbose and (s + 1) % 10 == 0:
            print(f"  [{domain}] {s + 1}/{n_samples} samples", flush=True)

    _, X, Y = grid_2d(n)
    coords = np.stack([X, Y], axis=-1)

    spec_dict = {
        "domain": domain,
        "description": spec.description,
        "simply_connected": spec.simply_connected,
        "n": int(n),
        "spacing": float(h),
        "n_samples": int(n_samples),
        "seed": int(seed),
        "n_modes": int(n_modes),
        "contrast": float(contrast),
        "forcing": float(forcing),
        "pde": "-div(a grad u) = f on the active cells, u = 0 on the inactive layer",
        "discretisation": "5-point finite volume, arithmetic face averaging, h = 2/(n-1)",
        "coefficient": "a = exp(contrast * a_hat), a_hat = truncated sine series, "
                       "Gaussian coefficients with 1/(k^2+l^2)^2 variance, max|a_hat| = 1",
        "fill_fraction": float(mask.mean()),
        "generator": "dagnese_fno.data.generate_dataset",
    }
    return {
        "coeffs": torch.tensor(coeffs, dtype=dtype),
        "targets": torch.tensor(targets, dtype=dtype),
        "coords": torch.tensor(coords, dtype=dtype),
        "mask": torch.tensor(mask, dtype=torch.bool),
        "spacing": float(h),
        "spec": spec_dict,
    }


def save_dataset(data: dict, path: str | Path) -> Path:
    """Write ``data`` to ``path`` (``.pt``) plus a human-readable ``.json`` sidecar.

    The sidecar carries the ``spec`` and the SHA-256 of the binary, so a dataset
    regenerated elsewhere can be checked byte-for-byte.  No timestamp is stored:
    the file must be reproducible.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)
    meta = dict(data.get("spec", {}))
    meta["file"] = path.name
    meta["sha256"] = file_sha256(path)
    meta["shapes"] = {k: list(v.shape) for k, v in data.items() if hasattr(v, "shape")}
    meta["mask_summary"] = mask_summary(np.asarray(data["mask"].cpu().numpy()
                                                   if hasattr(data["mask"], "cpu")
                                                   else data["mask"], dtype=bool))
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    return path


def load_dataset(path: str | Path) -> dict:
    """Load a dataset written by :func:`save_dataset` (tensors on the CPU)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}")
    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict) or "coeffs" not in data:
        raise ValueError(f"{path} does not look like a DIF-FNO dataset (keys: "
                         f"{sorted(data) if isinstance(data, dict) else type(data)})")
    return data
