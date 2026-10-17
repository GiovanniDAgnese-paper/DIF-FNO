"""Tests for the discrete Jacobian calculus.

These tests are the regression barrier for the two defects that invalidate a
naive implementation:

1. forgetting to divide by the grid spacing (which makes ``det J`` of the
   identity equal to ``h^2`` and the barrier loss resolution dependent);
2. mixing up the index convention (which flips the sign of ``det J`` and turns
   every guarantee into its opposite).
"""

from __future__ import annotations

import pytest
import torch

from dagnese_fno.geometry import (
    certified_det_lower_bound,
    det_jacobian_stats,
    displacement_stats,
    folding_mask,
    folding_percentage,
    jacobian_components,
    jacobian_det,
    jacobian_det_autograd,
    jacobian_singular_values,
)


def make_grid(n, low=-1.0, high=1.0, batch=None):
    t = torch.linspace(low, high, n)
    coords = torch.stack(torch.meshgrid(t, t, indexing="ij"), dim=-1)
    if batch is not None:
        coords = coords.unsqueeze(0).expand(batch, -1, -1, -1).contiguous()
    return coords, (high - low) / (n - 1)


# --------------------------------------------------------------------------- #
# Identity, scalings, rotations: exact determinants
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [16, 33, 64, 129])
def test_identity_has_unit_determinant_at_every_resolution(n):
    """det J = 1 for the identity -- the property a naive stencil violates."""
    coords, h = make_grid(n)
    det = jacobian_det(coords, spacing=h)
    assert torch.allclose(det, torch.ones_like(det), atol=1e-8)


@pytest.mark.parametrize("n", [17, 32, 64])
def test_uniform_scaling(n, factor=2.0):
    coords, h = make_grid(n)
    det = jacobian_det(factor * coords, spacing=h)
    # In 2D a uniform scaling by `factor` scales areas (and hence det J) by factor**2.
    assert torch.allclose(det, torch.full_like(det, factor**2), atol=1e-6)


def test_anisotropic_scaling():
    coords, h = make_grid(33)
    scale = torch.tensor([2.0, 3.0])
    det = jacobian_det(coords * scale, spacing=h)
    assert torch.allclose(det, torch.full_like(det, 6.0), atol=1e-5)


@pytest.mark.parametrize("theta", [0.0, 0.3, 1.1, 2.7])
def test_rotation_preserves_area(theta):
    coords, h = make_grid(48)
    c, s = torch.cos(torch.tensor(theta)), torch.sin(torch.tensor(theta))
    R = torch.tensor([[c, -s], [s, c]])
    phi = coords @ R.T
    det = jacobian_det(phi, spacing=h)
    assert torch.allclose(det, torch.ones_like(det), atol=1e-5)


def test_shear_preserves_area():
    coords, h = make_grid(48)
    x, y = coords[..., 0], coords[..., 1]
    phi = torch.stack([x + 0.4 * y, y], dim=-1)
    det = jacobian_det(phi, spacing=h)
    assert torch.allclose(det, torch.ones_like(det), atol=1e-5)


def test_reflection_is_detected_as_folding():
    """An orientation-reversing map has det J < 0 and 100% folded cells."""
    coords, h = make_grid(32)
    phi = torch.stack([-coords[..., 0], coords[..., 1]], dim=-1)
    det = jacobian_det(phi, spacing=h)
    assert float(det.max()) < 0
    assert folding_percentage(det) == pytest.approx(100.0)
    assert bool(folding_mask(det).all())


# --------------------------------------------------------------------------- #
# Consistency: finite differences vs autodiff, resolution independence
# --------------------------------------------------------------------------- #
def test_matches_autograd_on_a_smooth_map():
    n = 64
    base, h = make_grid(n)
    coords = base.clone().requires_grad_(True)
    phi = torch.stack([
        coords[..., 0] + 0.05 * torch.sin(2 * torch.pi * coords[..., 1]),
        coords[..., 1] + 0.05 * torch.cos(2 * torch.pi * coords[..., 0]),
    ], dim=-1)
    det_fd = jacobian_det(phi.detach(), spacing=h)
    det_ad = jacobian_det_autograd(phi, coords)
    # Central differences: O(h^2) error; exclude the one-sided boundary ring.
    err = (det_fd - det_ad.detach()).abs()[2:-2, 2:-2]
    # Second-order central differences on h = 2/63: the truncation error is O(h^2).
    assert float(err.max()) < 1e-2, float(err.max())


def test_result_is_independent_of_the_resolution():
    """The same continuous map sampled on 32^2, 64^2 and 128^2 gives the same det J.

    This is the property a spacing-normalised stencil guarantees -- and the one
    that the raw forward-difference stencil of DIF-FNO v4 violates (there
    det J_Id = h^2, so the barrier silently changes strength with the grid).
    """
    def stats(n):
        coords, h = make_grid(n)
        phi = torch.stack([
            coords[..., 0] + 0.1 * torch.sin(torch.pi * coords[..., 1]),
            coords[..., 1] - 0.1 * torch.cos(torch.pi * coords[..., 0]),
        ], dim=-1)
        det = jacobian_det(phi, spacing=h)
        return torch.tensor([det.min(), det.mean(), det.max()])

    ref = stats(128)
    for n in (32, 64):
        got = stats(n)
        assert torch.allclose(got, ref, atol=5e-3), (n, got.tolist(), ref.tolist())


def test_non_uniform_grid_uses_local_spacing():
    """On a stretched grid the derivative must divide by the *local* increment.

    A linear map has an exact derivative, so the test isolates the spacing
    handling from the truncation error.
    """
    t = torch.tensor([0.0, 0.1, 0.5, 1.0, 2.0])          # strongly non-uniform
    coords = torch.stack(torch.meshgrid(t, t, indexing="ij"), dim=-1)
    x, y = coords[..., 0], coords[..., 1]
    phi = torch.stack([3 * x + 2 * y, -x + 5 * y], dim=-1)
    det = jacobian_det(phi, coords=coords)
    # J = [[3, 2], [-1, 5]] -> det = 15 + 2 = 17 everywhere, including the boundary.
    assert torch.allclose(det, torch.full_like(det, 17.0), atol=1e-4)


def test_coords_and_spacing_agree_on_uniform_grids():
    coords, h = make_grid(32)
    a = jacobian_det(coords, spacing=h)
    b = jacobian_det(coords, coords=coords)
    assert torch.allclose(a, b, atol=1e-8)


def test_singular_values_of_identity():
    coords, h = make_grid(32)
    sv = jacobian_singular_values(coords, spacing=h)
    assert sv.shape == coords.shape
    assert torch.allclose(sv, torch.ones_like(sv), atol=1e-5)
    assert bool((sv[..., 0] >= sv[..., 1]).all())          # descending order


# --------------------------------------------------------------------------- #
# Statistics and masks
# --------------------------------------------------------------------------- #
def test_masked_statistics_ignore_the_outside():
    coords, h = make_grid(32)
    phi = torch.stack([-coords[..., 0], coords[..., 1]], dim=-1)  # det = -1 everywhere
    det = jacobian_det(phi, spacing=h)
    mask = torch.zeros(32, 32, dtype=torch.bool)
    mask[:16, :16] = True
    stats = det_jacobian_stats(det, mask)
    assert stats["min"] == pytest.approx(-1.0, abs=1e-6)
    assert stats["n_cells"] == 256
    assert stats["fold_pct"] == pytest.approx(100.0)
    assert folding_percentage(det) == pytest.approx(100.0)


def test_displacement_stats_of_identity_are_zero():
    coords, h = make_grid(32)
    s = displacement_stats(coords, coords)
    assert s["mean"] == pytest.approx(0.0, abs=1e-12)
    assert s["max"] == pytest.approx(0.0, abs=1e-12)


def test_jacobian_components_shapes():
    coords, h = make_grid(16, batch=3)
    j00, j01, j10, j11 = jacobian_components(coords, spacing=h)
    assert j00.shape == (3, 16, 16)
    # float32 finite differences leave a ~1e-6 round-off on the off-diagonal terms
    assert torch.allclose(j00, torch.ones_like(j00), atol=1e-5)
    assert torch.allclose(j11, torch.ones_like(j11), atol=1e-5)
    assert torch.allclose(j01, torch.zeros_like(j01), atol=1e-5)
    assert torch.allclose(j10, torch.zeros_like(j10), atol=1e-5)


def test_invalid_inputs_raise():
    coords, h = make_grid(16)
    with pytest.raises(ValueError):
        jacobian_det(torch.rand(16, 16, 3))          # wrong last dimension
    with pytest.raises(ValueError):
        jacobian_det(coords, spacing=0.0)            # non-positive spacing
    with pytest.raises(ValueError):
        jacobian_det(coords[..., :1], spacing=h)     # wrong last dimension
    with pytest.raises(ValueError):
        jacobian_det_autograd(coords, coords)        # coords must require grad


# --------------------------------------------------------------------------- #
# Certificate
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "lip,kappa,expected",
    [(0.0, 1.0, 1.0), (0.5, 1.0, 0.25), (0.9, 1.0, 0.01), (1.0, 1.0, 0.0),
     (1.5, 1.0, 0.0), (0.5, 2.0, 0.0)],
)
def test_certified_lower_bound(lip, kappa, expected):
    assert certified_det_lower_bound(lip, kappa) == pytest.approx(expected, abs=1e-12)


def test_certificate_is_sharp_for_a_linear_map():
    """phi = (1 - L) * Id has det J = (1-L)^2, i.e. the bound is attained."""
    coords, h = make_grid(32)
    lip = 0.4
    phi = (1 - lip) * coords
    det = jacobian_det(phi, spacing=h)
    assert float(det.min()) >= certified_det_lower_bound(lip) - 1e-6
    assert float(det.min()) == pytest.approx((1 - lip) ** 2, rel=1e-4)
