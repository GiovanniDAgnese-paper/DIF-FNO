"""Body-fitted H-grid around a symmetric NACA 0012 profile.

The four benchmark domains live on a tensor-product grid, but the *motivation*
of the paper is a computational mesh that follows a body: this module builds the
genuine two-block (H-topology) grid around the NACA 0012 that the v4 code only
claimed to build (the original ``dagnese_mesh_generator`` returned a plain
tensor grid and ignored the thickness distribution).

Geometry
--------
The profile is the standard symmetric 4-digit section

.. math:: y_t(\\xi) = 5t\\,\\bigl(0.2969\\sqrt{\\xi} - 0.1260\\,\\xi
          - 0.3516\\,\\xi^2 + 0.2843\\,\\xi^3 - 0.1036\\,\\xi^4\\bigr),
          \\qquad t = 0.12,

(the last coefficient closes the trailing edge, :math:`y_t(1) = 0`), placed on
:math:`\\xi = (x - x_{\\rm le})/(x_{\\rm te} - x_{\\rm le}) \\in [0,1]`.

The grid has ``2 * ny`` rows: ``ny`` below the wake cut and ``ny`` above it.
Each block is a transfinite interpolation between the wall :math:`\\pm y_t` and
the far field :math:`\\pm 1`,

.. math:: y(\\eta) = y_t + \\eta\\,(y_{\\rm far} - y_t), \\qquad \\eta \\in [0, 1],

with :math:`\\eta` sampled on ``ny`` points that exclude :math:`\\eta = 0`, so
that the wall sits *between* the two blocks (the cut) instead of being a grid
line.  The chordwise coordinate is cosine-clustered towards the leading and
trailing edges.

The result is a strictly monotone, folding-free ``(2*ny, nx, 2)`` tensor of
:math:`(x, y)` coordinates, i.e. a valid computational grid on which the
Jacobian diagnostics of :mod:`dagnese_fno.geometry` can be evaluated directly.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = [
    "NACA_THICKNESS_RATIO",
    "naca0012_thickness",
    "naca0012_surface",
    "naca0012_grid_spacing",
    "generate_dagnese_naca0012_mesh",
]

#: Thickness-to-chord ratio of a NACA 0012.
NACA_THICKNESS_RATIO: float = 0.12

# Coefficients of the symmetric 4-digit thickness distribution (Abbott & von Doenhoff).
_A = (0.2969, -0.1260, -0.3516, 0.2843, -0.1036)


def naca0012_thickness(xi, thickness_ratio: float = NACA_THICKNESS_RATIO):
    """Half-thickness :math:`y_t(\\xi)` of a symmetric NACA 4-digit airfoil.

    ``xi`` is the chord fraction in :math:`[0,1]`; values outside are clipped, so
    the function is ``0`` ahead of the leading edge and behind the trailing edge.
    """
    x = np.clip(np.asarray(xi, dtype=float), 0.0, 1.0)
    yt = (_A[0] * np.sqrt(x) + _A[1] * x + _A[2] * x**2
          + _A[3] * x**3 + _A[4] * x**4)
    return 5.0 * thickness_ratio * yt


def naca0012_surface(nx: int, x_le: float = 0.1, x_te: float = 0.9,
                     thickness_ratio: float = NACA_THICKNESS_RATIO):
    """Return ``(x, y_upper, y_lower)`` sampled with cosine clustering.

    Cosine spacing puts half of the points in the first and last quarter of the
    chord, which is where the leading-edge suction peak and the trailing-edge
    cusp live.
    """
    if nx < 2:
        raise ValueError(f"nx must be at least 2, got {nx}")
    theta = np.linspace(0.0, np.pi, nx)
    x = x_le + (x_te - x_le) * 0.5 * (1.0 - np.cos(theta))
    y_upper = naca0012_thickness((x - x_le) / (x_te - x_le), thickness_ratio)
    return x, y_upper, -y_upper


def naca0012_grid_spacing(nx: int, ny: int, x_le: float = 0.1, x_te: float = 0.9,
                          far_field: float = 1.0) -> float:
    """Nominal cell size :math:`\\sqrt{h_\\xi\\,h_\\eta}` of the H-grid.

    A single number is not a faithful description of a graded grid (the cells at
    the wall are an order of magnitude smaller than the nominal one); this is the
    geometric mean of the two *average* spacings and is meant for reporting only.
    """
    h_xi = (x_te - x_le) / max(nx - 1, 1)
    h_eta = (2.0 * far_field) / max(2 * ny - 1, 1)
    return float(np.sqrt(h_xi * h_eta))


def generate_dagnese_naca0012_mesh(nx: int = 64, ny: int = 64, x_le: float = 0.1,
                                   x_te: float = 0.9, far_field: float = 1.0,
                                   dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Two-block body-fitted H-grid around the profile.

    Parameters
    ----------
    nx, ny:
        Chordwise points and points *per block*; the mesh has ``2 * ny`` rows.
    x_le, x_te:
        Leading- and trailing-edge abscissae (chord ``x_te - x_le``).
    far_field:
        Half-height of the outer boundary.

    Returns
    -------
    ``(2*ny, nx, 2)`` tensor, ``mesh[..., 0] = x`` and ``mesh[..., 1] = y``.
    The rows are ordered from the lower far field (``y = -far_field``) to the
    upper one (``y = +far_field``), so ``mesh[..., 1]`` is strictly increasing
    with the row index and the grid is folding-free.
    """
    if nx < 2 or ny < 2:
        raise ValueError(f"nx and ny must be at least 2, got {nx}, {ny}")
    x, y_wall, _ = naca0012_surface(nx, x_le, x_te)

    # eta of the two blocks: the wall (eta = 0) is the cut *between* rows
    # ny - 1 and ny, so the first off-wall row sits at eta = 1 / (2ny - 1).
    s = 1.0 / (2 * ny - 1)
    ramp = np.arange(ny, dtype=float) / (ny - 1)
    eta_upper = s + (1.0 - s) * ramp                  # rows ny .. 2ny-1
    eta_lower = 1.0 - (1.0 - s) * ramp                # rows 0 .. ny-1

    gap = far_field - y_wall                          # (nx,)
    y_upper = y_wall[None, :] + eta_upper[:, None] * gap[None, :]
    y_lower = -(y_wall[None, :] + eta_lower[:, None] * gap[None, :])
    y = np.concatenate([y_lower, y_upper], axis=0)     # (2ny, nx)
    x_grid = np.broadcast_to(x[None, :], y.shape)

    mesh = np.stack([x_grid, y], axis=-1)
    return torch.tensor(mesh, dtype=dtype)
