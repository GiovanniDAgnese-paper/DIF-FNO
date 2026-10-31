"""Computational domains as boolean masks on the reference box :math:`[-1,1]^2`.

Every domain is the intersection of an analytic shape with the open box: the
outer ring of nodes is always inactive (``clear_frame``), which is what makes
the Dirichlet problem of :mod:`dagnese_fno.data.darcy` well posed -- the ring
plays the role of :math:`\\partial\\Omega_h`.

Coordinates follow the library convention: node ``(i, j)`` is
``(x_i, y_j)`` with ``np.meshgrid(x, x, indexing="ij")``, i.e. **axis 0 carries
the first coordinate**.  Confusing the two flips the sign of every Jacobian
determinant and turns every topological guarantee into its opposite, which is
why the convention is stated once, here, and tested
(``tests/test_data.py::test_masks_are_well_formed``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

__all__ = [
    "DomainSpec",
    "DOMAINS",
    "get_mask",
    "clear_frame",
    "mask_summary",
    "star_mask",
    "lshape_mask",
    "annulus_mask",
    "naca0012_mask",
    "naca0012_thickness",
    "slit_mask",
    "double_hole_mask",
    "unit_coordinates",
]


# --------------------------------------------------------------------------- #
# Geometry primitives
# --------------------------------------------------------------------------- #
def unit_coordinates(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Node coordinates of an ``n x n`` grid on :math:`[-1,1]^2` (``'ij'`` order)."""
    if n < 3:
        raise ValueError("n must be >= 3")
    x = np.linspace(-1.0, 1.0, n)
    return np.meshgrid(x, x, indexing="ij")


def clear_frame(mask: np.ndarray) -> np.ndarray:
    """Return a copy of ``mask`` with the outer ring of nodes deactivated."""
    m = np.array(mask, dtype=bool, copy=True)
    m[0, :] = False
    m[-1, :] = False
    m[:, 0] = False
    m[:, -1] = False
    return m


def _shape_mask(n: int, inside: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> np.ndarray:
    x, y = unit_coordinates(n)
    return clear_frame(np.asarray(inside(x, y), dtype=bool))


# --------------------------------------------------------------------------- #
# The four benchmark domains
# --------------------------------------------------------------------------- #
def star_mask(n: int) -> np.ndarray:
    """Five-lobed star :math:`r \\le a + b\\cos(5\\theta)` (non-convex, simply connected).

    With :math:`a = 0.62`, :math:`b = 0.18` the radius oscillates between
    :math:`0.44` and :math:`0.80`: the segment joining two adjacent lobe tips
    leaves the domain, so the star is genuinely non-convex.
    """
    def inside(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        r = np.hypot(x, y)
        theta = np.arctan2(y, x)
        return r <= 0.62 + 0.18 * np.cos(5.0 * theta)

    return _shape_mask(n, inside)


def lshape_mask(n: int) -> np.ndarray:
    """L-shaped domain: the box minus the open first quadrant (re-entrant corner)."""
    def inside(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (x <= 0.0) | (y <= 0.0)

    return _shape_mask(n, inside)


def annulus_mask(n: int) -> np.ndarray:
    """Annulus :math:`0.35 \\le r \\le 0.85` (non-convex *and* doubly connected)."""
    def inside(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        r = np.hypot(x, y)
        return (r >= 0.35) & (r <= 0.85)

    return _shape_mask(n, inside)


def naca0012_thickness(xc, t: float = 0.12):
    """Half-thickness of a NACA 0012 profile at chordwise position ``xc`` in ``[0, 1]``.

    Uses the standard closed-trailing-edge polynomial; ``xc`` may be a scalar or
    an array (NumPy or a float).
    """
    xc = np.clip(np.asarray(xc, dtype=float), 0.0, 1.0)
    return 5.0 * t * (0.2969 * np.sqrt(xc) - 0.126 * xc - 0.3516 * xc**2
                      + 0.2843 * xc**3 - 0.1036 * xc**4)


def naca0012_mask(n: int, x_le: float = -0.8, x_te: float = 0.8) -> np.ndarray:
    """Box minus a NACA 0012 airfoil of chord ``x_te - x_le`` (non-convex, with a hole-free wake)."""
    def inside(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        xc = (x - x_le) / (x_te - x_le)
        chord = x_te - x_le
        airfoil = (xc >= 0.0) & (xc <= 1.0) & (np.abs(y) <= chord * naca0012_thickness(xc))
        return ~airfoil

    return _shape_mask(n, inside)


def slit_mask(n: int, crack_half_width: float = 0.035) -> np.ndarray:
    """Box with a horizontal crack entering from the left wall.

    The crack is a band ``|y| <= crack_half_width`` for ``x <= 0``.  It ends in a
    re-entrant tip at the origin, where the gradient of the solution is singular
    in the continuum limit; the mesh sees a thin inactive strip of three cells at
    ``N = 64`` and the Dirichlet frame opens the crack at the wall.
    """
    def inside(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        crack = (x <= 0.0) & (np.abs(y) <= crack_half_width)
        return ~crack

    return _shape_mask(n, inside)


def double_hole_mask(n: int, radius: float = 0.24, centre: float = 0.36) -> np.ndarray:
    """Box with two circular holes centred at ``(+-centre, 0)`` (multiply connected)."""
    def inside(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        left = np.hypot(x + centre, y) <= radius
        right = np.hypot(x - centre, y) <= radius
        return ~(left | right)

    return _shape_mask(n, inside)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DomainSpec:
    """Everything the rest of the library needs to know about a domain."""

    name: str
    default_seed: int
    description: str
    mask_fn: Callable[[int], np.ndarray]

    def mask(self, n: int) -> np.ndarray:
        return self.mask_fn(n)


DOMAINS: dict[str, DomainSpec] = {
    "star": DomainSpec(
        "star", 0,
        "five-lobed star: non-convex, simply connected, analytic boundary",
        star_mask,
    ),
    "lshape": DomainSpec(
        "lshape", 1,
        "L-shaped domain: re-entrant 270-degree corner (singular gradients)",
        lshape_mask,
    ),
    "annulus": DomainSpec(
        "annulus", 2,
        "annulus 0.35 <= r <= 0.85: doubly connected, non-convex",
        annulus_mask,
    ),
    "naca0012": DomainSpec(
        "naca0012", 3,
        "box with a NACA 0012 airfoil removed: sharp trailing edge, thin gap",
        naca0012_mask,
    ),
    "slit": DomainSpec(
        "slit", 4,
        "box with a horizontal crack from the left wall: singular crack tip",
        slit_mask,
    ),
    "double_hole": DomainSpec(
        "double_hole", 5,
        "box with two circular holes: multiply connected, two obstacles",
        double_hole_mask,
    ),
}


def get_mask(name: str, n: int) -> np.ndarray:
    """Boolean ``n x n`` mask of the domain ``name`` (raises ``KeyError`` if unknown)."""
    spec = DOMAINS[name]
    mask = np.asarray(spec.mask(n), dtype=bool)
    if mask.shape != (n, n):
        raise RuntimeError(f"mask for '{name}' has shape {mask.shape}, expected {(n, n)}")
    return mask


def mask_summary(mask: np.ndarray) -> dict:
    """Area, boundary size and Euler characteristic of a mask (used in the manifest)."""
    m = np.asarray(mask, dtype=bool)
    active = int(m.sum())
    # 4-connectivity boundary: active nodes with at least one inactive neighbour
    padded = np.pad(m, 1, constant_values=False)
    nb = (padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:])
    boundary = int((m & ~nb).sum())
    return {
        "n_nodes": int(m.size),
        "n_active": active,
        "fill_fraction": float(active / m.size),
        "n_boundary": boundary,
        "perimeter_over_area": float(boundary / active) if active else float("inf"),
    }
