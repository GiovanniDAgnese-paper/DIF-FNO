"""Tests for the D'Agnese Triangular Monotone Chart (proposition P5).

Every claim of the theorem is exercised here under the same name:

* P5.i   global injectivity (not only ``det J > 0``),
* P5.ii  orientation, with a determinant that is a product of positive numbers,
* P5.iii exact inverse in closed form,
* P5.iv  no restriction on the size of the deformation (adversarial parameters),
* P5.v   identity at initialisation,
* P5.vi  closure under composition.
"""

from __future__ import annotations

import pytest
import torch

from dagnese_fno.geometry import (
    DagneseTriangularChart,
    chart_det_consistency,
    jacobian_det,
)


def reference_grid(n: int = 32) -> torch.Tensor:
    t = torch.linspace(-1.0, 1.0, n)
    return torch.stack(torch.meshgrid(t, t, indexing="ij"), dim=-1)


def neighbour_gap(phi: torch.Tensor) -> float:
    """Smallest distance between a node and its four grid neighbours.

    ``torch.cdist`` cannot be used here: it computes distances through
    ``|x|^2 + |y|^2 - 2 x.y``, which suffers catastrophic cancellation for
    points that are 1e-5 apart at a distance of 1 from the origin and can
    return exactly ``0``.  The grid-neighbour differences are the quantity that
    injectivity of a monotone grid actually requires.
    """
    gaps = [
        (phi[1:, :, :] - phi[:-1, :, :]).norm(dim=-1).min(),
        (phi[:, 1:, :] - phi[:, :-1, :]).norm(dim=-1).min(),
    ]
    return float(torch.stack(gaps).min())


def randomise(chart: DagneseTriangularChart, scale: float = 1.0, seed: int = 0) -> None:
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in chart.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * scale)


# --------------------------------------------------------------------------- #
# P5.v -- identity at initialisation
# --------------------------------------------------------------------------- #
def test_identity_at_initialisation():
    coords = reference_grid()
    chart = DagneseTriangularChart(hidden=16, n_fourier=2)
    phi, det = chart(coords)
    assert torch.allclose(phi, coords, atol=1e-6)
    assert torch.allclose(det, torch.ones_like(det), atol=1e-5)


def test_identity_is_exact_at_any_resolution():
    chart = DagneseTriangularChart(hidden=8, n_fourier=1)
    for n in (16, 33, 64):
        phi, det = chart(reference_grid(n))
        assert torch.allclose(phi, reference_grid(n), atol=1e-6)
        assert float(det.min()) == pytest.approx(1.0, abs=1e-5)


# --------------------------------------------------------------------------- #
# P5.i -- global injectivity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_chart_is_injective(seed):
    coords = reference_grid(24)
    chart = DagneseTriangularChart(hidden=16, n_fourier=2)
    randomise(chart, 1.5, seed)
    phi, _ = chart(coords)

    # P5.i in practice: neighbouring nodes never collapse onto each other.
    assert neighbour_gap(phi) > 1e-6

    # ... and the structural form is preserved: first coordinate constant along
    # rows, strictly increasing with the row index; second strictly increasing.
    g, h = chart._knots(coords, None, 0)
    assert bool((torch.diff(g, dim=-1) > 0).all())
    assert bool((torch.diff(h, dim=-1) > 0).all())


def test_injectivity_survives_extreme_parameters():
    """P5.iv: the certificate is not a statement about small deformations."""
    coords = reference_grid(24)
    chart = DagneseTriangularChart(hidden=16, n_fourier=2)
    randomise(chart, 50.0, 3)
    phi, det = chart(coords)
    assert float((det <= 0).float().mean()) == 0.0
    assert neighbour_gap(phi) > 0.0


# --------------------------------------------------------------------------- #
# P5.ii -- orientation
# --------------------------------------------------------------------------- #
def test_det_positive_for_adversarial_parameters():
    coords = reference_grid()
    for scale in (0.0, 0.5, 2.0, 20.0, 500.0):
        chart = DagneseTriangularChart(hidden=16, n_fourier=2)
        randomise(chart, scale, seed=int(scale * 3) + 1)
        _, det = chart(coords)
        assert float(det.min()) > 0.0, f"folded cell at scale {scale}"
        assert float((det <= 0).float().mean()) == 0.0


def test_det_matches_the_finite_difference_stencil_at_identity():
    coords = reference_grid()
    chart = DagneseTriangularChart(hidden=8, n_fourier=1)
    phi, det = chart(coords)
    fd = jacobian_det(phi, spacing=2.0 / 31)
    assert torch.allclose(det, fd, atol=1e-5)


def test_certificate_is_a_valid_lower_bound():
    coords = reference_grid()
    chart = DagneseTriangularChart(hidden=16, n_fourier=2)
    randomise(chart, 1.0, 7)
    _, det = chart(coords)
    lb = chart.certified_det_lower_bound(coords)
    assert float(lb) <= float(det.min()) + 1e-9
    assert float(lb) > 0.0


def test_parameter_free_floor_is_a_lower_bound():
    """The guarantee does not depend on the parameters at all.

    The floor ``(2 eps / (K dxi))^2`` is exact in real arithmetic.  In float32 the
    cumulative sum that produces the knots perturbs each increment by O(eps_32)
    *absolute*, i.e. ~0.3 % *relative* for an increment of ~3e-5, so the measured
    minimum can undershoot the ideal floor by a fraction of a percent.  The
    library therefore reports two numbers -- this floor (a theorem about the
    parameters) and ``certified_det_lower_bound`` (exact for the knots actually
    computed) -- and the test allows for the round-off of the first one.
    """
    coords = reference_grid()
    chart = DagneseTriangularChart(hidden=16, n_fourier=2)
    randomise(chart, 100.0, 11)
    _, det = chart(coords)
    floor = chart.parameter_free_det_floor(coords.shape[-2])
    assert floor > 0.0
    assert float(det.min()) >= floor * (1 - 1e-2)
    assert float(chart.certified_det_lower_bound(coords)) <= float(det.min()) + 1e-12


def test_min_increment_keeps_increments_strictly_positive():
    coords = reference_grid()
    chart = DagneseTriangularChart(hidden=8, n_fourier=1, min_increment=1e-3)
    randomise(chart, 1e3, 5)
    g, h = chart._knots(coords, None, 0)
    # float32 round-off in the cumulative sum perturbs the increments by O(eps32)
    # absolute, i.e. a fraction of a percent at 3e-5: allow for it explicitly.
    assert float(torch.diff(g, dim=-1).min()) >= 1e-3 / (coords.shape[-2] - 1) * 0.99
    assert float(torch.diff(h, dim=-1).min()) >= 1e-3 / (coords.shape[-2] - 1) * 0.99


# --------------------------------------------------------------------------- #
# P5.iii -- exact inverse
# --------------------------------------------------------------------------- #
def test_inverse_is_exact_on_the_nodes():
    coords = reference_grid()
    chart = DagneseTriangularChart(hidden=16, n_fourier=2)
    randomise(chart, 1.0, 2)
    assert chart.inverse_consistency_error(coords) < 1e-5


def test_inverse_inverts_the_piecewise_linear_map_off_grid():
    coords = reference_grid()
    chart = DagneseTriangularChart(hidden=16, n_fourier=2)
    # A moderately deformed chart: the inverse of the piecewise-bilinear map is
    # exact, but a chart whose smallest cell is 5e-4 loses digits in float32
    # (see ``inverse_consistency_error``).
    randomise(chart, 0.5, 4)
    g, h = chart._knots(coords, None, 0)
    # Points strictly inside the square (the map is defined on the closed square,
    # and ``_apply_layer`` clamps at the boundary).
    points = (coords + 0.08 * torch.randn_like(coords)).clamp(-0.98, 0.98)
    forward = chart._apply_layer((g, h), points)
    back = chart._invert_layer((g.expand(1, -1), h.expand(1, -1, -1)), forward.reshape(1, 32, 32, 2))
    assert float((back.reshape_as(points) - points).abs().max()) < 1e-4


def test_det_integrates_to_the_area_of_the_image():
    """``sum_cells det J * dxi^2 == area(image) == 4`` exactly (a telescoping sum)."""
    coords = reference_grid()
    chart = DagneseTriangularChart(hidden=16, n_fourier=2)
    randomise(chart, 1.0, 6)
    _, det = chart(coords)
    dxi = 2.0 / (coords.shape[-2] - 1)
    # det at a node is the determinant of the cell to its lower-right, so the
    # cells are the (n-1) x (n-1) block.
    total = float(det[:-1, :-1].sum()) * dxi * dxi
    assert total == pytest.approx(4.0, rel=1e-4)


# --------------------------------------------------------------------------- #
# P5.vi -- composition
# --------------------------------------------------------------------------- #
def test_composition_stays_admissible():
    coords = reference_grid()
    chart = DagneseTriangularChart(hidden=12, n_fourier=2, n_layers=3)
    randomise(chart, 1.0, 9)
    phi, det = chart(coords)
    assert float(det.min()) > 0.0
    # Three compositions accumulate float32 round-off; the inverse is exact in
    # real arithmetic, and the tolerance below reflects the condition number.
    assert chart.inverse_consistency_error(coords) < 1e-2
    assert neighbour_gap(phi) > 1e-6


def test_composition_with_one_layer_is_the_same_object():
    coords = reference_grid()
    a = DagneseTriangularChart(hidden=8, n_fourier=1, n_layers=1)
    b = DagneseTriangularChart(hidden=8, n_fourier=1, n_layers=1)
    b.load_state_dict(a.state_dict())
    assert torch.allclose(a(coords)[0], b(coords)[0])


# --------------------------------------------------------------------------- #
# Conditioning, gradients, resolution
# --------------------------------------------------------------------------- #
def test_latent_conditioning_produces_distinct_charts():
    coords = reference_grid()
    chart = DagneseTriangularChart(hidden=16, n_fourier=2, latent_dim=4)
    randomise(chart, 0.5, 12)
    codes = torch.randn(3, 4)
    phi = chart(coords, codes)[0]
    assert phi.shape == (3, 32, 32, 2)
    assert not torch.allclose(phi[0], phi[1], atol=1e-3)
    _, det = chart(coords, codes)
    assert float(det.min()) > 0.0


def test_gradients_flow_through_forward_and_inverse():
    coords = reference_grid(16)
    chart = DagneseTriangularChart(hidden=8, n_fourier=1)
    randomise(chart, 0.5, 13)
    phi, det = chart(coords)
    inv = chart.inverse(phi)
    (det.sum() + phi.sum() + inv.sum()).backward()
    grads = [p.grad.abs().sum().item() for p in chart.parameters() if p.grad is not None]
    assert sum(1 for g in grads if g > 0) == len(grads)


def test_chart_is_resolution_consistent():
    """The continuous map is a property of the parameters, not of the grid."""
    chart = DagneseTriangularChart(hidden=16, n_fourier=2)
    randomise(chart, 0.5, 14)
    fine = reference_grid(128)
    coarse = reference_grid(64)
    phi_fine, _ = chart(fine)
    phi_coarse, _ = chart(coarse)
    # Compare at the nodes shared by both grids (every second node).
    sub = phi_fine[::2, ::2]
    assert float((sub - phi_coarse).abs().max()) < 0.05
    assert float((phi_fine - fine).norm(dim=-1).mean()) == pytest.approx(
        float((phi_coarse - coarse).norm(dim=-1).mean()), rel=0.25)


def test_det_consistency_helper_runs():
    coords = reference_grid()
    chart = DagneseTriangularChart(hidden=8, n_fourier=1)
    assert chart_det_consistency(chart, coords) < 1e-4


def test_bad_inputs_are_rejected():
    chart = DagneseTriangularChart(hidden=8)
    with pytest.raises(ValueError):
        chart(torch.zeros(8, 8, 3))
    with pytest.raises(ValueError):
        DagneseTriangularChart(n_layers=0)
    with pytest.raises(ValueError):
        DagneseTriangularChart(activation="not-an-activation")
    with pytest.raises(ValueError):
        DagneseTriangularChart(min_increment=1.5)
    with pytest.raises(ValueError):
        DagneseTriangularChart(latent_dim=4)(reference_grid(8))


# --------------------------------------------------------------------------- #
# inverse_at: the exact inverse at arbitrary points
# --------------------------------------------------------------------------- #
def test_inverse_at_agrees_with_inverse_on_the_grid():
    n = 24
    coords = grid(n) if "grid" in globals() else None
    t = torch.linspace(-1.0, 1.0, n)
    coords = torch.stack(torch.meshgrid(t, t, indexing="ij"), dim=-1)
    chart = DagneseTriangularChart(hidden=16, n_fourier=2)
    randomise(chart, 0.5, seed=1)
    phi, _ = chart(coords.unsqueeze(0))
    a = chart.inverse(phi, reference=coords)[0]
    b = chart.inverse_at(phi.reshape(-1, 2).unsqueeze(0), reference=coords)[0]
    assert torch.allclose(a.reshape(-1, 2), b, atol=1e-6)


def test_inverse_at_handles_arbitrary_points_and_shapes():
    n = 24
    t = torch.linspace(-1.0, 1.0, n)
    coords = torch.stack(torch.meshgrid(t, t, indexing="ij"), dim=-1)
    chart = DagneseTriangularChart(hidden=16, n_fourier=2)
    randomise(chart, 0.5, seed=2)
    # points inside the image of the chart, monotone along both axes
    u = torch.linspace(-0.9, 0.9, 37)
    v = torch.linspace(-0.9, 0.9, 11)
    pts = torch.stack(torch.meshgrid(u, v, indexing="ij"), dim=-1).reshape(-1, 2)
    out = chart.inverse_at(pts.unsqueeze(0), reference=coords)
    assert out.shape == (1, pts.shape[0], 2)
    assert float(out.abs().max()) <= 1.0 + 1e-6
    # the recovered coordinates increase with the physical coordinates:
    # the flat index runs over u fastest, so stride 11 steps in u and 1 in v
    assert float(torch.diff(out[0, ::11, 0]).min()) > 0
    assert float(torch.diff(out[0, :11, 1]).min()) > 0


def test_inverse_at_rejects_malformed_input():
    chart = DagneseTriangularChart(hidden=8, n_fourier=1)
    with pytest.raises(ValueError):
        chart.inverse_at(torch.zeros(4, 3))
