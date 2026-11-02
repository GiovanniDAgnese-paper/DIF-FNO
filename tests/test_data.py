"""Data generation: geometry of the masks and verification of the Darcy solver."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dagnese_fno.data import (
    DOMAINS,
    assemble_darcy,
    clear_frame,
    generate_dataset,
    get_mask,
    load_dataset,
    naca0012_mask,
    sample_coeff_field,
    save_dataset,
    solve_darcy,
)


# --------------------------------------------------------------------------- #
# Masks
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", sorted(DOMAINS))
def test_masks_are_well_formed(name):
    m = get_mask(name, 64)
    assert m.shape == (64, 64)
    assert m.dtype == np.bool_
    assert 0.05 < m.mean() < 0.98
    # Dirichlet problem is well posed only if the outer frame is inactive
    if name != "star":  # the star does not touch the frame of the box
        assert not (m[0, :].any() or m[-1, :].any() or m[:, 0].any() or m[:, -1].any())


def test_annulus_is_doubly_connected():
    from scipy.ndimage import label

    m = get_mask("annulus", 64)
    complement = ~m
    labels, n = label(complement)
    # inner disc + outer frame (which touches the border)
    assert n >= 1
    inner = labels[32, 32]
    assert inner != labels[0, 0], "the annulus must have a hole"


def test_star_is_non_convex():
    """A straight segment between two points of the star leaves the domain."""
    m = get_mask("star", 128)
    x = np.linspace(-1, 1, 128)
    X, Y = np.meshgrid(x, x, indexing="ij")
    theta1, theta2 = 0.0, 2 * np.pi / 5           # two adjacent lobes (5theta = 0, 2pi)
    p1 = np.array([0.74 * np.cos(theta1), 0.74 * np.sin(theta1)])
    p2 = np.array([0.74 * np.cos(theta2), 0.74 * np.sin(theta2)])
    i1 = np.unravel_index(np.argmin((X - p1[0]) ** 2 + (Y - p1[1]) ** 2), X.shape)
    i2 = np.unravel_index(np.argmin((X - p2[0]) ** 2 + (Y - p2[1]) ** 2), X.shape)
    assert m[i1] and m[i2]
    mid = ((i1[0] + i2[0]) // 2, (i1[1] + i2[1]) // 2)
    assert not m[mid], "the straight segment exits the star -> non-convex"


def test_lshape_has_a_reentrant_corner():
    m = np.asarray(get_mask("lshape", 64), dtype=np.int8)
    # Corner node: exactly three of the four quadrants around it are active.
    corner = 31
    around = [m[corner - 1, corner - 1], m[corner - 1, corner + 1],
              m[corner + 1, corner - 1], m[corner + 1, corner + 1]]
    assert sum(around) == 3


def test_naca0012_mask_removes_the_airfoil():
    m = naca0012_mask(128)
    x = np.linspace(-1, 1, 128)
    X, Y = np.meshgrid(x, x, indexing="ij")
    # Inside the airfoil (mid-chord, on the centreline) the mask must be False.
    inside = (np.abs(X - 0.5) < 0.05) & (np.abs(Y) < 0.01)
    assert not m[inside].any()
    # Far above the airfoil the mask must be True.
    above = (np.abs(X - 0.5) < 0.05) & (np.abs(Y - 0.8) < 0.01)
    assert m[above].all()


def test_slit_is_a_crack_from_the_wall():
    """The slit opens on the left wall, ends inside the box, and is one cell-set."""
    m = get_mask("slit", 64)
    x = np.linspace(-1, 1, 64)
    X, Y = np.meshgrid(x, x, indexing="ij")
    crack = (X < -0.2) & (np.abs(Y) < 0.02)
    assert crack.any() and not m[crack].any(), "the crack must be inactive"
    # The crack must reach the Dirichlet frame: every node of the first active
    # column inside the band is inactive.
    tip = (X > -0.05) & (X <= 0.0) & (np.abs(Y) < 0.02)   # last crack column
    assert tip.any() and not m[tip].any(), "the crack tip must be inactive"
    # ...and the cells just past the tip are active: the crack stops inside.
    past = (X > 0.05) & (X < 0.15) & (np.abs(Y) < 0.02)
    assert m[past].all()
    # Two rows apart from the crack the domain is active (it is a slit, not a cut).
    above = (np.abs(X + 0.5) < 0.05) & (np.abs(Y - 0.15) < 0.01)
    assert m[above].all()
    # And the mask is a single connected piece.
    from scipy.ndimage import label
    _, n = label(m)
    assert n == 1


def test_double_hole_has_two_holes():
    from scipy.ndimage import label

    m = get_mask("double_hole", 64)
    x = np.linspace(-1, 1, 64)
    X, Y = np.meshgrid(x, x, indexing="ij")
    for centre in (-0.36, 0.36):
        hole = np.hypot(X - centre, Y) < 0.12
        assert hole.any() and not m[hole].any(), f"hole at x={centre} must be inactive"
    assert m[(np.abs(X) < 0.05) & (np.abs(Y) < 0.15)].all(), "the bridge must be active"
    labels, n = label(m)
    assert n == 1


def test_clear_frame():
    m = np.ones((16, 16), dtype=bool)
    c = clear_frame(m)
    assert not (c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any())
    assert c[1:-1, 1:-1].all()


def test_unknown_domain_raises():
    with pytest.raises(KeyError):
        get_mask("does-not-exist", 32)


# --------------------------------------------------------------------------- #
# Coefficient field
# --------------------------------------------------------------------------- #
def test_coeff_field_is_positive_and_bounded():
    mask = get_mask("star", 32)
    rng = np.random.default_rng(0)
    a = sample_coeff_field(32, mask, rng)
    assert a[mask].min() > 0
    assert a[mask].max() / a[mask].min() < np.exp(2 * 2.0) + 1e-6   # contrast exp(2*gamma)
    assert np.allclose(a[~mask], 1.0)


def test_coeff_field_is_deterministic_given_the_seed():
    mask = get_mask("star", 32)
    a1 = sample_coeff_field(32, mask, np.random.default_rng(7))
    a2 = sample_coeff_field(32, mask, np.random.default_rng(7))
    assert np.allclose(a1, a2)


# --------------------------------------------------------------------------- #
# Solver: manufactured solution
# --------------------------------------------------------------------------- #
def _manufactured(n, a_const=True):
    """Homogeneous-Dirichlet manufactured problem on the *staircased* domain.

    The unknowns are the mask nodes whose 4-point stencil is inside the mask;
    the Dirichlet ring is the mask layer adjacent to the complement, i.e.
    x = h and x = 1 - h.  Choosing u = sin(k(x-h)) sin(k(y-h)) with
    k = pi / (1 - 2h) makes u vanish exactly on that ring.
    """
    mask = clear_frame(np.ones((n, n), dtype=bool))
    h = 1.0 / (n - 1)
    x = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    k = np.pi / (1 - 2 * h)
    u = np.sin(k * (X - h)) * np.sin(k * (Y - h))
    if a_const:
        a = np.ones_like(u)
        f = 2 * k**2 * u                          # -div(1 * grad u)
    else:
        a = 1.0 + 0.5 * np.sin(np.pi * X) * np.sin(np.pi * Y)
        # -div(a grad u) computed analytically
        ux = k * np.cos(k * (X - h)) * np.sin(k * (Y - h))
        uy = k * np.sin(k * (X - h)) * np.cos(k * (Y - h))
        uxx = -(k**2) * np.sin(k * (X - h)) * np.sin(k * (Y - h))
        uyy = -(k**2) * np.sin(k * (X - h)) * np.sin(k * (Y - h))
        ax = 0.5 * np.pi * np.cos(np.pi * X) * np.sin(np.pi * Y)
        ay = 0.5 * np.pi * np.sin(np.pi * X) * np.cos(np.pi * Y)
        f = -(a * (uxx + uyy) + ax * ux + ay * uy)
    return mask, a, h, f, u


@pytest.mark.parametrize("a_const", [True, False])
def test_solver_is_second_order_accurate(a_const):
    errors = []
    for n in (33, 65):
        mask, a, h, f, u_exact = _manufactured(n, a_const)
        u = solve_darcy(mask, a, h, f)
        interior = np.zeros_like(mask)
        nb = (mask[1:-1, 1:-1] & mask[:-2, 1:-1] & mask[2:, 1:-1]
              & mask[1:-1, :-2] & mask[1:-1, 2:])
        interior[1:-1, 1:-1] = nb
        errors.append(np.abs(u[interior] - u_exact[interior]).max())
    rate = np.log2(errors[0] / errors[1])
    assert errors[1] < errors[0]
    assert rate > 1.8, f"observed order {rate:.2f} (errors {errors})"


def test_solver_is_exact_on_linear_fields():
    """Patch test: the 5-point flux stencil reproduces -div(a grad u) = 0 for u linear."""
    n = 33
    mask = clear_frame(np.ones((n, n), dtype=bool))
    h = 1.0 / (n - 1)
    x = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    for u in (X, Y, 3 * X - 2 * Y + 1.0):
        A, b, interior, idx = assemble_darcy(mask, np.ones((n, n)), h, np.zeros((n, n)))
        # Strict interior: nodes whose four neighbours are also unknowns, otherwise the
        # Dirichlet contribution (u taken as 0 at the ring) is part of the equation.
        strict = interior & np.roll(interior, 1, 0) & np.roll(interior, -1, 0) \
            & np.roll(interior, 1, 1) & np.roll(interior, -1, 1)
        res = A @ u[interior] - b
        assert np.abs(res[idx[strict]]).max() < 1e-10


def test_solver_spd_and_symmetric():
    mask = get_mask("star", 32)
    h = 2.0 / 31
    a = sample_coeff_field(32, mask, np.random.default_rng(0))
    A, b, interior, idx = assemble_darcy(mask, a, h)
    A = A.tocsr()
    assert np.allclose((A - A.T).data, 0.0)
    assert (A.diagonal() > 0).all()
    assert b.shape[0] == A.shape[0]


def test_solution_respects_the_mask():
    mask = get_mask("annulus", 48)
    h = 2.0 / 47
    a = sample_coeff_field(48, mask, np.random.default_rng(1))
    u = solve_darcy(mask, a, h)
    assert np.isfinite(u).all()
    assert np.allclose(u[~mask], 0.0)
    assert u[mask].max() > 0           # non-trivial solution with f = 1
    assert (u[mask] >= -1e-12).all()   # maximum principle: f >= 0 => u >= 0


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def test_generate_dataset_is_reproducible(tmp_path):
    d1 = generate_dataset("star", n=32, n_samples=4, seed=42, verbose=False)
    d2 = generate_dataset("star", n=32, n_samples=4, seed=42, verbose=False)
    assert torch.equal(d1["coeffs"], d2["coeffs"])
    assert torch.equal(d1["targets"], d2["targets"])
    assert d1["coeffs"].shape == (4, 32, 32)
    assert d1["spacing"] == pytest.approx(2.0 / 31)


def test_dataset_roundtrip(tmp_path):
    d = generate_dataset("lshape", n=32, n_samples=3, seed=0, verbose=False)
    path = save_dataset(d, tmp_path / "darcy_test.pt")
    loaded = load_dataset(path)
    assert torch.equal(loaded["coeffs"], d["coeffs"])
    assert torch.equal(loaded["mask"], d["mask"])
    assert loaded["spec"]["domain"] == "lshape"
    assert (tmp_path / "darcy_test.json").exists()


def test_unknown_domain_raises_in_generator():
    with pytest.raises(KeyError):
        generate_dataset("nope", n=16, n_samples=1, verbose=False)
