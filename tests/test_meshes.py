"""Body-fitted NACA 0012 mesh: geometry of the airfoil and validity of the grid."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dagnese_fno.data.meshes import (
    generate_dagnese_naca0012_mesh,
    naca0012_grid_spacing,
    naca0012_surface,
    naca0012_thickness,
)
from dagnese_fno.geometry import jacobian_det


def test_thickness_distribution():
    assert naca0012_thickness(0.0) == pytest.approx(0.0, abs=1e-12)
    assert naca0012_thickness(1.0) == pytest.approx(5 * 0.12 * (0.2969 - 0.126 - 0.3516 + 0.2843 - 0.1036), rel=1e-9)
    # Maximum thickness of a NACA 0012 is ~12% of the chord, near x/c = 0.3
    xs = np.linspace(0, 1, 1001)
    yt = naca0012_thickness(xs)
    assert yt.max() == pytest.approx(0.06, rel=1e-2)          # half-thickness = t/2
    assert xs[np.argmax(yt)] == pytest.approx(0.3, abs=0.02)


def test_surface_is_symmetric():
    x, up, lo = naca0012_surface(64)
    assert x.shape == up.shape == lo.shape
    assert np.allclose(up, -lo)
    assert np.allclose(x[0], 0.1) and np.allclose(x[-1], 0.9)


def test_mesh_shape_and_finiteness():
    nx = ny = 64
    mesh = generate_dagnese_naca0012_mesh(nx, ny)
    assert mesh.shape == (2 * ny, nx, 2)
    assert torch.isfinite(mesh).all()


def test_mesh_monotone_in_the_wall_normal_direction():
    """y must increase monotonically with the row index (a valid grid)."""
    mesh = generate_dagnese_naca0012_mesh(64, 48)
    y = mesh[..., 1].numpy()
    assert np.all(np.diff(y, axis=0) > 0)


def test_mesh_is_folding_free():
    """The grid is not tangled: det J > 0 in the (eta, xi) -> (y, x) ordering.

    ``axis 0`` is the wall-normal index (it carries the second coordinate y) and
    ``axis 1`` the chordwise index (coordinate x), so the mesh Jacobian is the
    determinant of :math:`\\partial(y, x)/\\partial(\\eta, \\xi)`, which the
    library convention requires to be positive.
    """
    nx = ny = 64
    mesh = generate_dagnese_naca0012_mesh(nx, ny).flip(-1)     # -> (y, x) components
    hx = 2.0 / (nx - 1)
    hy = 2.0 / (2 * ny - 1)              # y spans [-1, 1] over 2*ny rows
    det = jacobian_det(mesh, spacing=(hy, hx))
    assert float(det.min()) > 0, float(det.min())


def test_mesh_follows_the_airfoil_surface():
    """The two rows adjacent to the cut eta = 0 sit on the airfoil surface."""
    nx = ny = 128
    mesh = generate_dagnese_naca0012_mesh(nx, ny, x_le=0.1, x_te=0.9)
    x = mesh[0, :, 0].numpy()
    y = mesh[..., 1].numpy()
    mid = ny                                     # first row with eta > 0
    surface_y = y[mid]
    expected = naca0012_thickness((x - 0.1) / 0.8) * ((x >= 0.1) & (x <= 0.9))
    expected += (1.0 / (2 * ny - 1)) * (1 - expected)
    assert np.allclose(surface_y, expected, atol=1e-6)
    # Far field rows are at |y| = 1
    assert np.allclose(np.abs(y[0]), 1.0, atol=1e-6)
    assert np.allclose(np.abs(y[-1]), 1.0, atol=1e-6)


def test_grid_spacing_helper():
    assert naca0012_grid_spacing(101, 101) > 0
