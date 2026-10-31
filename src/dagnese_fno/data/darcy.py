"""The Darcy benchmark: coefficient fields and a finite-volume reference solver.

We solve

.. math::
    -\nabla\cdot\big(a(x)\nabla u(x)\big) = f(x), \qquad x \in \Omega_h,\qquad u = 0 \text{ on } \partial\Omega_h,

on a *masked* Cartesian grid with a cell-centred (TPFA) finite-volume scheme:

.. math::
    \sum_{\text{faces } \sigma} a_\sigma \,(u_P - u_N) = f_P \, h^2 ,
    \qquad a_\sigma = \frac{2 a_P a_N}{a_P + a_N} \;(\text{harmonic mean}).

The harmonic mean is the correct face conductivity for a piecewise-smooth
coefficient (it is exact for layered media and gives the physically meaningful
answer at a jump), and it keeps the matrix symmetric.

The system is assembled with vectorised COO construction and factorised with
SuperLU, which is ~100x faster than the ``lil_matrix`` loop of the DIF-FNO v4
data generator.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

__all__ = [
    "assemble_darcy",
    "interior_mask",
    "solve_darcy",
    "sample_coeff_field",
]


# --------------------------------------------------------------------------- #
# Geometry of the discrete problem
# --------------------------------------------------------------------------- #
def interior_mask(mask: np.ndarray) -> np.ndarray:
    """Nodes of ``mask`` whose four neighbours are also in ``mask``.

    These are the unknowns; every other active node (the ring adjacent to the
    complement) and every inactive node carries the Dirichlet value ``u = 0``.
    """
    m = np.asarray(mask, dtype=bool)
    nb = (m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:])
    out = np.zeros_like(m)
    out[1:-1, 1:-1] = nb
    return out


def assemble_darcy(
    mask: np.ndarray,
    a: np.ndarray,
    h: float,
    f: np.ndarray | None = None,
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    """Assemble the finite-volume system ``A u = b``.

    Returns ``(A, b, interior, index)`` where ``A`` is ``(n_unknowns, n_unknowns)``
    CSR, ``b`` the right-hand side, ``interior`` the boolean mask of the unknowns
    and ``index`` the integer array mapping node ``(i, j)`` to its unknown number
    (``-1`` outside).
    """
    mask = np.asarray(mask, dtype=bool)
    a = np.asarray(a, dtype=float)
    n = mask.shape[0]
    if mask.shape != (n, n) or a.shape != (n, n):
        raise ValueError(f"mask and a must be square and equal, got {mask.shape} and {a.shape}")
    if h <= 0:
        raise ValueError("h must be positive")
    if np.any(a <= 0):
        raise ValueError("the coefficient field must be strictly positive")
    if f is None:
        f = np.ones((n, n))
    f = np.asarray(f, dtype=float)

    interior = interior_mask(mask)
    index = np.full((n, n), -1, dtype=np.int64)
    n_unknown = int(interior.sum())
    index[interior] = np.arange(n_unknown)

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    diag = np.zeros(n_unknown)

    for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        # Unknown node P and its neighbour N across the face in direction (di, dj).
        pi = slice(max(0, -di), n - max(0, di))
        pj = slice(max(0, -dj), n - max(0, dj))
        ni = slice(max(0, di), n - max(0, -di))
        nj = slice(max(0, dj), n - max(0, -dj))

        p_int = interior[pi, pj]
        neighbour = mask[ni, nj]
        neighbour_int = interior[ni, nj]

        # Harmonic-mean face conductivity between P and N (exact for layered media).
        ap, an = a[pi, pj], a[ni, nj]
        cond = 2.0 * ap * an / (ap + an)

        # Every face of an unknown contributes to its diagonal (the Dirichlet
        # faces too, because there u = 0 and the flux is cond * (u_P - 0)).
        face = p_int & neighbour
        np.add.at(diag, index[pi, pj][face], cond[face])

        off = p_int & neighbour_int
        if off.any():
            rows.append(index[pi, pj][off])
            cols.append(index[ni, nj][off])
            vals.append(-cond[off])

    # Diagonal entries: sum of the conductances of all faces (Dirichlet faces
    # contribute to the diagonal only, because u = 0 there).
    rows.append(np.arange(n_unknown))
    cols.append(np.arange(n_unknown))
    vals.append(diag)

    A = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_unknown, n_unknown),
    ).tocsr()
    b = (f[interior] * h * h).astype(float)
    return A, b, interior, index


def solve_darcy(
    mask: np.ndarray,
    a: np.ndarray,
    h: float,
    f: np.ndarray | None = None,
) -> np.ndarray:
    """Solve the masked Darcy problem; returns ``u`` with ``u[~mask] = 0``."""
    A, b, interior, _ = assemble_darcy(mask, a, h, f)
    u = np.zeros(mask.shape, dtype=float)
    if A.shape[0] == 0:
        return u
    lu = splu(A.tocsc())
    u[interior] = lu.solve(b)
    return u


# --------------------------------------------------------------------------- #
# Random coefficient fields
# --------------------------------------------------------------------------- #
def sample_coeff_field(
    n: int,
    mask: np.ndarray,
    rng: np.random.Generator | None = None,
    n_modes: int = 6,
    contrast: float = 2.0,
) -> np.ndarray:
    """Log-normal coefficient field with bounded contrast.

    ``log a`` is a random trigonometric polynomial whose amplitude decays like
    :math:`|k|^{-1.5}` (a rough, Matérn-like spectrum that produces the
    high-contrast channels characteristic of subsurface permeability), clipped to
    :math:`[-1, 1]` and scaled by ``contrast``:

    .. math:: a = \\exp(\\gamma\\,\\hat z), \\qquad \\hat z \\in [-0.98, 0.98],

    hence :math:`\\max a / \\min a \\le e^{1.96\\gamma}` and :math:`a > 0`
    everywhere.  Outside the mask the field is set to :math:`1`.
    """
    if contrast <= 0:
        raise ValueError("contrast must be positive")
    if n_modes < 1:
        raise ValueError("n_modes must be >= 1")
    rng = np.random.default_rng(0) if rng is None else rng
    mask = np.asarray(mask, dtype=bool)

    x, y = np.meshgrid(np.linspace(-1.0, 1.0, n), np.linspace(-1.0, 1.0, n), indexing="ij")
    z = np.zeros((n, n))
    for i in range(n_modes):
        for j in range(n_modes):
            if i == 0 and j == 0:
                continue
            amp = rng.normal() / (1.0 + i * i + j * j) ** 0.75
            phase = rng.uniform(0.0, 2.0 * np.pi)
            z += amp * np.cos(0.5 * np.pi * (i * x + j * y) + phase)

    active = z[mask]
    std = active.std()
    z = (z - active.mean()) / (std if std > 1e-12 else 1.0)
    z = np.clip(z, -0.98, 0.98)

    a = np.exp(contrast * z)
    a[~mask] = 1.0
    return a
