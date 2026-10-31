"""Body-fitted NACA 0012 mesh: a *generative* grid, not a mask.

The benchmark's other domains are masks on a Cartesian box, so the FNO and the
finite-volume solver share the same grid.  An airfoil is different: the
boundary is curved, and a mask is a staircase approximation of it.  This module
builds the body-fitted **H-grid** used by the mesh study (``scripts/`` and
``docs/THEORY.md`` §P8):

* ``axis 1`` is the chordwise index :math:`\\xi` (coordinate ``x``),
* ``axis 0`` is the wall-normal index :math:`\\eta` (coordinate ``y``),
* the two rows adjacent to :math:`\\eta = 0` lie *on* the airfoil surface,
* the first and last rows lie on the far field :math:`y = \\pm 1`.

The grid is a transfinite (TFI) interpolation between the airfoil surface and
the far field,

.. math::
    y(\\eta, \\xi) = \\eta + (1 - |\\eta|)\\,\\operatorname{sign}(\\eta)\\,y_s(\\xi),

where :math:`y_s` is the NACA 0012 half-thickness; because
:math:`\\mathrm{d}y/\\mathrm{d}\\eta = 1 - y_s > 0` and :math:`x` depends on
:math:`\\xi` only, the map :math:`J = \\partial(y, x)/\\partial(\\eta, \\xi)` has
:math:`\\det J = (1 - y_s)\\,x'(\\xi) > 0`: the mesh is folding-free by
construction, which ``tests/test_meshes.py`` verifies with the same discrete
Jacobian that the model is trained with.
"""

from __future__ import annotations

import numpy as np
import torch

from .masks import naca0012_thickness

__all__ = [
    "naca0012_thickness",
    "naca0012_surface",
    "generate_dagnese_naca0012_mesh",
    "naca0012_grid_spacing",
]


def naca0012_surface(n: int, x_le: float = 0.1, x_te: float = 0.9,
                     t: float = 0.12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chordwise stations (cosine clustered) and upper/lower surface ordinates.

    Returns ``(x, upper, lower)`` with ``upper = -lower``.
    """
    if n < 3:
        raise ValueError("n must be >= 3")
    s = np.linspace(0.0, 1.0, n)
    xc = 0.5 * (1.0 - np.cos(np.pi * s))              # cosine clustering at both ends
    x = x_le + (x_te - x_le) * xc
    yt = naca0012_thickness(xc, t=t)
    return x, yt, -yt


def generate_dagnese_naca0012_mesh(
    nx: int,
    ny: int,
    x_le: float = 0.1,
    x_te: float = 0.9,
    t: float = 0.12,
) -> torch.Tensor:
    """Body-fitted H-grid around a NACA 0012 section.

    The output has shape ``(2 * ny, nx, 2)``: ``axis 0`` is the wall-normal
    index (it carries the coordinate ``y``), ``axis 1`` the chordwise index (it
    carries ``x``).  Rows ``0 .. ny - 1`` are below the airfoil, rows
    ``ny .. 2 * ny - 1`` above it, both ordered so that ``y`` increases with the
    row index.
    """
    if nx < 5 or ny < 4:
        raise ValueError("need nx >= 5 and ny >= 4")

    x, _, _ = naca0012_surface(nx, x_le=x_le, x_te=x_te, t=t)
    chord = x_te - x_le
    xc = np.clip((x - x_le) / chord, 0.0, 1.0)
    # Half-thickness in *normalised* chord units: this is the height of the
    # airfoil surface above the chord line in the coordinates of the grid.
    surface = np.where((x >= x_le) & (x <= x_te), naca0012_thickness(xc, t=t), 0.0)

    eta = np.linspace(-1.0, 1.0, 2 * ny)
    y = eta[:, None] + (1.0 - np.abs(eta))[:, None] * np.sign(eta)[:, None] * surface[None, :]

    # Stored as (x, y) components: axis 0 is the wall-normal index (y), axis 1
    # the chordwise one (x), so ``mesh.flip(-1)`` gives the (y, x) ordering in
    # which det J = dy/deta * dx/dxi > 0.
    out = np.stack([np.broadcast_to(x[None, :], y.shape), y], axis=-1)
    return torch.from_numpy(out.astype(np.float32))


def naca0012_grid_spacing(nx: int, ny: int, x_le: float = 0.1, x_te: float = 0.9) -> float:
    """Smallest cell size of the H-grid (used as ``spacing`` by the Jacobian tools)."""
    x, _, _ = naca0012_surface(nx, x_le=x_le, x_te=x_te)
    dx = float(np.diff(x).min()) if nx > 1 else 0.0
    dy = 2.0 / max(2 * ny - 1, 1)
    return float(min(dx, dy))
