"""Tests for the pullback / pushforward transport (proposition P6).

The point of these tests is that the chart acts on the *field*: a change of
variables must be exact (up to interpolation), it must preserve integrals with
the right Jacobian weight, and the round trip must converge at second order.
"""

from __future__ import annotations

import pytest
import torch

from dagnese_fno.geometry import (
    DagneseTriangularChart,
    convergence_rate,
    jacobian_weight,
    pullback,
    pushforward,
    sample_field,
)


def grid(n: int) -> torch.Tensor:
    t = torch.linspace(-1.0, 1.0, n)
    return torch.stack(torch.meshgrid(t, t, indexing="ij"), dim=-1)


def bump(coords: torch.Tensor) -> torch.Tensor:
    return torch.exp(-6.0 * (coords[..., 0] ** 2 + coords[..., 1] ** 2))


def randomise(chart: DagneseTriangularChart, scale: float = 0.6, seed: int = 0) -> None:
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in chart.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * scale)


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def test_sample_field_matches_manual_bilinear():
    n = 17
    f = torch.rand(1, n, n, dtype=torch.float64)
    coords = grid(n).to(torch.float64)
    # A point in the middle of a cell, compared with an explicit bilinear formula.
    x = torch.tensor([[[0.13, -0.41]]], dtype=torch.float64)
    out = sample_field(f, x)
    u = (float(x[0, 0, 0]) + 1) / 2 * (n - 1)
    v = (float(x[0, 0, 1]) + 1) / 2 * (n - 1)
    i, j = int(u), int(v)
    t, s = u - i, v - j
    manual = ((1 - t) * (1 - s) * f[0, i, j] + t * (1 - s) * f[0, i + 1, j]
              + (1 - t) * s * f[0, i, j + 1] + t * s * f[0, i + 1, j + 1])
    assert float(out[0, 0, 0]) == pytest.approx(float(manual), rel=1e-10)


def test_sample_field_is_the_identity_on_the_grid():
    n = 32
    f = torch.rand(2, n, n)
    coords = grid(n).expand(2, -1, -1, -1)
    assert torch.allclose(sample_field(f, coords), f, atol=1e-5)


def test_orientation_of_the_coordinate_convention():
    """A field that varies along axis 0 must vary along the first coordinate."""
    n = 17
    f = grid(n)[..., 0].unsqueeze(0) * 1.0
    point = torch.tensor([[[0.0, 0.5]]])
    assert float(sample_field(f, point)[0, 0, 0]) == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# Pullback and pushforward
# --------------------------------------------------------------------------- #
def test_pullback_of_the_identity_is_the_identity():
    n = 32
    coords = grid(n)
    chart = DagneseTriangularChart(hidden=8, n_fourier=1)
    phi, det = chart(coords)
    f = bump(coords).unsqueeze(0)
    assert torch.allclose(pullback(f, phi, det, alpha=0.0), f, atol=1e-5)


def test_round_trip_is_the_identity():
    n = 48
    coords = grid(n)
    chart = DagneseTriangularChart(hidden=16, n_fourier=2)
    randomise(chart, 0.4, 1)
    phi, det = chart(coords.unsqueeze(0))
    f = bump(coords).unsqueeze(0)
    f_chart = pullback(f, phi, det, alpha=0.0)
    phi_inv = chart.inverse(coords.unsqueeze(0), reference=coords)
    back = pushforward(f_chart, phi_inv)
    # Relative to the amplitude of the field: this is a resampling error, not a
    # loss of information.
    assert float((back - f).detach().abs().max()) < 0.03 * float(f.abs().max())


def test_round_trip_converges_at_second_order():
    """P6: the round-trip error is the interpolation error, O(h^2)."""
    errors, resolutions = [], []
    for n in (32, 64, 128):
        coords = grid(n)
        chart = DagneseTriangularChart(hidden=16, n_fourier=2)
        randomise(chart, 0.4, 2)
        phi, det = chart(coords.unsqueeze(0))
        f = bump(coords).unsqueeze(0)
        back = pushforward(pullback(f, phi, det), chart.inverse(coords.unsqueeze(0), reference=coords))
        errors.append(float((back - f).abs().max()))
        resolutions.append(n)
    rate = convergence_rate(errors, resolutions)
    assert rate > 1.5, f"observed order {rate:.2f} (errors {errors})"


def test_pullback_preserves_the_integral_with_the_jacobian_weight():
    """``int f(x) dx = int f(phi(xi)) |det J| dxi``: the Piola factor is correct."""
    n = 64
    coords = grid(n)
    chart = DagneseTriangularChart(hidden=16, n_fourier=2)
    randomise(chart, 0.5, 3)
    phi, det = chart(coords.unsqueeze(0))
    f = bump(coords).unsqueeze(0)
    density = pullback(f, phi, det, alpha=1.0)
    dxi = 2.0 / (n - 1)
    assert float(density.sum()) * dxi * dxi == pytest.approx(float(f.sum()) * dxi * dxi, rel=0.02)


def test_jacobian_weight_switches_between_scalars_and_densities():
    det = torch.tensor([[0.5, 2.0]])
    assert torch.allclose(jacobian_weight(det, 0.0), torch.ones_like(det))
    assert torch.allclose(jacobian_weight(det, 1.0), det)
    assert torch.allclose(jacobian_weight(det, 2.0), det**2)


def test_chart_operator_actually_uses_the_chart():
    """A deformation must change the transported field (v1's defect was that it did not)."""
    n = 32
    coords = grid(n)
    chart = DagneseTriangularChart(hidden=16, n_fourier=2)
    f = bump(coords).unsqueeze(0)
    phi_id, det_id = chart(coords.unsqueeze(0))
    moved = pullback(f, phi_id, det_id)
    assert torch.allclose(moved, f, atol=1e-5)          # identity chart -> identity
    randomise(chart, 1.0, 4)
    phi, det = chart(coords.unsqueeze(0))
    moved = pullback(f, phi, det)
    assert float((moved - f).abs().max()) > 1e-2        # a real chart moves the field


def test_density_round_trip_recovers_the_field():
    """alpha=1: pullback then pushforward is the identity (|det J|^-1 undoes |det J|)."""
    n = 48
    coords = grid(n)
    chart = DagneseTriangularChart(hidden=16, n_fourier=2)
    randomise(chart, 0.3, seed=4)
    phi, det = chart(coords.unsqueeze(0))
    phi_inv = chart.inverse(coords.unsqueeze(0), reference=coords)
    f = bump(coords).unsqueeze(0).expand(1, n, n).clone()
    back = pushforward(pullback(f, phi, det, alpha=1.0), phi_inv, det, alpha=1.0)
    assert float((back - f).detach().abs().max()) < 5e-2 * float(f.abs().max())


def test_jacobian_weight_is_a_power_of_the_determinant():
    det = torch.rand(4, 5, 5) + 0.5
    assert torch.allclose(jacobian_weight(det, 0.0), torch.ones_like(det))
    assert torch.allclose(jacobian_weight(det, 1.0), det, atol=1e-6)
    assert torch.allclose(jacobian_weight(det, -1.0) * det, torch.ones_like(det), atol=1e-5)
