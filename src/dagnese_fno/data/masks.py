"""Non-convex computational domains used by the benchmark.

Every domain is represented as a boolean mask on the *same* tensor-product grid
that covers the reference square :math:`[-1,1]^2`,

.. math:: x_i = -1 + i\\,h, \\qquad h = \\frac{2}{n-1}, \\qquad i = 0,\\dots,n-1,

with ``'ij'`` ordering, so that ``mask[i, j]`` is the cell centred at
:math:`(x_i, y_j)`.  The Darcy problem is solved on the *active* cells
(``mask == True``) with homogeneous Dirichlet conditions on the first inactive
layer, which is what the finite-volume solver in :mod:`dagnese_fno.data.darcy`
assembles.

Four domains are registered in :data:`DOMAINS`:

==============  =============================================================
``star``        5-lobed star, smooth but strongly non-convex
``lshape``      square minus a quadrant: the textbook re-entrant corner
``annulus``     doubly connected (a hole), the case a convex chart cannot cover
``naca0012``    square minus a symmetric NACA 0012 profile (this work)
==============  =============================================================

Adding a domain means writing a mask function and registering a
:class:`DomainSpec` here; the generator, the solver, the CLI and the benchmark
sweep pick it up automatically.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import generate_binary_structure, label

from .meshes import naca0012_thickness

__all__ = [
    "DomainSpec",
    "DOMAINS",
    "grid_2d",
    "clear_frame",
    "star_mask",
    "lshape_mask",
    "annulus_mask",
    "naca0012_mask",
    "get_mask",
    "mask_summary",
]


# --------------------------------------------------------------------------- #
# Grid helpers
# --------------------------------------------------------------------------- #
def grid_2d(n: int, extent: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(x, X, Y)``: the 1-D nodes and the two ``'ij'`` coordinate fields.

    ``X`` varies along ``axis 0`` and carries the first spatial coordinate, so
    both fields are indexed as ``X[i, j] = x_i`` and ``Y[i, j] = y_j``.
    """
    x = np.linspace(-extent, extent, n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    return x, X, Y


def clear_frame(mask: np.ndarray) -> np.ndarray:
    """Zero the outermost row/column of ``mask`` (copy).

    The Dirichlet problem is well posed only if the boundary of the bounding box
    is inactive, otherwise the "ring" on which :math:`u = 0` is imposed would cut
    through the domain.  Every mask except the star is framed this way.
    """
    m = np.array(mask, dtype=bool, copy=True)
    m[0, :] = False
    m[-1, :] = False
    m[:, 0] = False
    m[:, -1] = False
    return m


# --------------------------------------------------------------------------- #
# The masks
# --------------------------------------------------------------------------- #
def star_mask(n: int, r_tip: float = 0.75, lobes: int = 5, sharpness: float = 0.5) -> np.ndarray:
    """Five-lobed star: :math:`r \\le r_{\\rm tip}\\,(s + (1-s)\\cos(m\\theta))`.

    With ``s = 0.5`` the boundary passes through the origin between two lobes,
    which makes the domain *strongly* non-convex: the segment joining two
    adjacent lobe tips leaves it.
    """
    _, X, Y = grid_2d(n)
    r = np.hypot(X, Y)
    theta = np.arctan2(Y, X)
    r_max = r_tip * ((1.0 - sharpness) + sharpness * np.cos(lobes * theta))
    return r <= r_max


def lshape_mask(n: int) -> np.ndarray:
    """Unit square minus the open quadrant :math:`(0,1]^2` (re-entrant corner at 0)."""
    _, X, Y = grid_2d(n)
    m = ~((X > 0.0) & (Y > 0.0))
    return clear_frame(m)


def annulus_mask(n: int, r_inner: float = 0.45, r_outer: float = 0.9) -> np.ndarray:
    """Doubly connected domain :math:`r_{\\rm in} \\le \\lVert x \\rVert \\le r_{\\rm out}`.

    The hole is the reason a *single* convex chart cannot be used: any
    diffeomorphism of the bounding box that covers an annulus has to move mass
    around the inner disc.
    """
    _, X, Y = grid_2d(n)
    r = np.hypot(X, Y)
    return clear_frame((r >= r_inner) & (r <= r_outer))


def naca0012_mask(n: int, x_le: float = 0.1, x_te: float = 0.9) -> np.ndarray:
    """Bounding box minus a symmetric NACA 0012 profile of chord ``x_te - x_le``.

    The profile occupies :math:`\\lvert y \\rvert \\le y_t\\!\\left(\\frac{x-x_{\\rm le}}
    {x_{\\rm te}-x_{\\rm le}}\\right)`, with :math:`y_t` the standard thickness
    distribution (see :func:`dagnese_fno.data.meshes.naca0012_thickness`).  The
    result is doubly connected, like the annulus, but with a *thin* obstacle and
    a cusp at the trailing edge -- the hardest of the four for the chart.
    """
    _, X, Y = grid_2d(n)
    chord = (x_le <= X) & (x_te >= X)
    xi = np.clip((X - x_le) / (x_te - x_le), 0.0, 1.0)
    half_thickness = naca0012_thickness(xi) * chord
    inside_airfoil = chord & (np.abs(Y) <= half_thickness)
    return clear_frame(~inside_airfoil)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DomainSpec:
    """A registered domain: how to build its mask and how to seed its dataset."""

    name: str
    mask_fn: Callable[[int], np.ndarray]
    default_seed: int
    simply_connected: bool
    description: str


DOMAINS: dict[str, DomainSpec] = {
    "star": DomainSpec(
        name="star",
        mask_fn=star_mask,
        default_seed=0,
        simply_connected=True,
        description="5-lobed star: smooth boundary, strongly non-convex",
    ),
    "lshape": DomainSpec(
        name="lshape",
        mask_fn=lshape_mask,
        default_seed=1,
        simply_connected=True,
        description="square minus a quadrant: re-entrant corner (singular solution)",
    ),
    "annulus": DomainSpec(
        name="annulus",
        mask_fn=annulus_mask,
        default_seed=2,
        simply_connected=False,
        description="doubly connected: an inner hole the chart has to flow around",
    ),
    "naca0012": DomainSpec(
        name="naca0012",
        mask_fn=naca0012_mask,
        default_seed=3,
        simply_connected=False,
        description="square minus a symmetric NACA 0012 airfoil (thin obstacle + cusp)",
    ),
}


def get_mask(name: str, n: int) -> np.ndarray:
    """Boolean ``(n, n)`` mask of the registered domain ``name``.

    Raises ``KeyError`` for an unknown domain, listing the available ones.
    """
    if name not in DOMAINS:
        raise KeyError(f"unknown domain {name!r}; available: {sorted(DOMAINS)}")
    mask = np.asarray(DOMAINS[name].mask_fn(n))
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)
    return mask


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def mask_summary(mask: np.ndarray) -> dict:
    """Geometric diagnostics of a mask: fill fraction, components, holes, box."""
    m = np.asarray(mask, dtype=bool)
    # Digital topology: 4-connectivity for the domain, 8-connectivity for its
    # complement.  Using the same connectivity for both would report spurious
    # holes where two complement cells only touch diagonally.
    cross = generate_binary_structure(2, 1)
    full = generate_binary_structure(2, 2)
    _, n_components = label(m, structure=cross)
    comp_labels, n_complement = label(~m, structure=full)
    # The complement always has one component touching the frame: the "outside".
    # Every other component is a hole of the domain.
    border = set(np.unique(np.concatenate([comp_labels[0, :], comp_labels[-1, :],
                                           comp_labels[:, 0], comp_labels[:, -1]])))
    holes = [c for c in range(1, int(n_complement) + 1) if c not in border]
    ii, jj = np.nonzero(m)
    summary = {
        "shape": list(m.shape),
        "n_cells": int(m.size),
        "n_active": int(m.sum()),
        "fill_fraction": float(m.mean()),
        "n_components": int(n_components),
        "n_holes": len(holes),
    }
    if m.any():
        summary["bbox_cells"] = [int(ii.min()), int(ii.max()), int(jj.min()), int(jj.max())]
    return summary
