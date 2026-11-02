"""Tests for the D'Agnese Equidistribution Loss (proposition P7).

The claims are:

1. the loss is non-negative and vanishes **iff** the monitor mass is equally
   distributed over the cells;
2. it is scale invariant (in the monitor and in the domain);
3. equidistribution really is the minimiser of the worst-cell mass (the
   classical de Boor statement), verified in 1-D by brute force;
4. the gradient reaches the chart parameters.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dagnese_fno.losses import DagneseEquidistributionLoss, gradient_monitor


def grid(n: int) -> torch.Tensor:
    t = torch.linspace(-1.0, 1.0, n)
    return torch.stack(torch.meshgrid(t, t, indexing="ij"), dim=-1)


# --------------------------------------------------------------------------- #
# The loss itself
# --------------------------------------------------------------------------- #
def test_zero_for_equidistributed_cells():
    det = torch.ones(1, 16, 16)
    monitor = torch.ones(1, 16, 16)
    loss = DagneseEquidistributionLoss()(monitor, det)
    assert float(loss) == pytest.approx(0.0, abs=1e-12)


def test_positive_when_cells_carry_different_mass():
    det = torch.ones(1, 16, 16)
    monitor = torch.ones(1, 16, 16)
    monitor[:, :8] = 3.0
    loss = DagneseEquidistributionLoss()(monitor, det)
    assert float(loss) > 0.1


def test_zero_iff_equidistributed():
    """A monitor that is exactly ``1/det`` is equidistributed by construction."""
    torch.manual_seed(0)
    det = torch.rand(1, 12, 12) + 0.1
    monitor = 1.0 / det
    assert float(DagneseEquidistributionLoss()(monitor, det)) == pytest.approx(0.0, abs=1e-10)


def test_scale_invariance():
    torch.manual_seed(1)
    det = torch.rand(1, 12, 12) + 0.1
    monitor = torch.rand(1, 12, 12) + 0.1
    loss = DagneseEquidistributionLoss()
    a = float(loss(monitor, det))
    b = float(loss(10.0 * monitor, det))
    assert a == pytest.approx(b, rel=1e-6)


def test_shapes_are_checked():
    with pytest.raises(ValueError):
        DagneseEquidistributionLoss()(torch.ones(1, 8, 8), torch.ones(1, 9, 9))
    with pytest.raises(ValueError):
        gradient_monitor(torch.ones(1, 8, 8), 0.1, beta=0)


# --------------------------------------------------------------------------- #
# The monitor
# --------------------------------------------------------------------------- #
def test_monitor_is_positive_and_grows_with_the_gradient():
    coords = grid(32)
    flat = torch.ones(1, 32, 32)
    steep = (1.0 + 3.0 * coords[..., 0]).unsqueeze(0)
    m_flat = gradient_monitor(flat, 2 / 31)
    m_steep = gradient_monitor(steep, 2 / 31)
    assert float(m_flat.min()) > 0 and float(m_steep.min()) > 0
    assert float(m_steep.mean()) > float(m_flat.mean())


def test_monitor_uses_the_physical_spacing():
    """``|grad a|`` must be a physical derivative, not a finite difference."""
    t = torch.linspace(-1, 1, 33)
    field = t.reshape(1, 33, 1).expand(1, 33, 33)
    # d a / d x = 1, so the monitor is (floor + 1)^beta at both resolutions.
    coarse = gradient_monitor(field, 2 / 32)
    fine = gradient_monitor(field[:, ::2], 2 / 16)
    assert float(coarse[0, 1:-1, 1:-1].mean()) == pytest.approx(
        float(fine[0, 1:-1, 1:-1].mean()), rel=0.2)


# --------------------------------------------------------------------------- #
# Optimality of equidistribution (1-D brute force)
# --------------------------------------------------------------------------- #
def _worst_cell_mass(nodes: np.ndarray, m: np.ndarray, x: np.ndarray) -> float:
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (m[1:] + m[:-1]) * np.diff(x))])
    mass = np.interp(nodes, x, cum)
    return float(np.max(np.diff(mass)))


def test_equidistribution_minimises_the_worst_cell():
    """de Boor: equal monitor mass per cell minimises ``max_C int_C m``."""
    n = 201
    x = np.linspace(0.0, 1.0, n)
    m = 1.0 + 8.0 * np.exp(-((x - 0.3) ** 2) / 0.01)     # a sharp peak
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (m[1:] + m[:-1]) * np.diff(x))])
    cum /= cum[-1]
    cells = 21
    targets = np.linspace(0.0, 1.0, cells + 1)
    equidistributed = np.interp(targets, cum, x)

    rng = np.random.default_rng(0)
    best = _worst_cell_mass(equidistributed, m, x)
    for _ in range(200):
        perturbation = rng.normal(scale=0.02, size=cells)
        nodes = np.concatenate([[0.0], np.cumsum(np.diff(equidistributed) * np.exp(perturbation))])
        nodes = nodes / nodes[-1]
        best = min(best, _worst_cell_mass(nodes, m, x))
    assert _worst_cell_mass(equidistributed, m, x) <= best + 1e-3


# --------------------------------------------------------------------------- #
# Gradient
# --------------------------------------------------------------------------- #
def test_gradient_reaches_the_chart_parameters():
    from dagnese_fno.geometry import DagneseTriangularChart

    n = 16
    coords = grid(n)
    chart = DagneseTriangularChart(hidden=8, n_fourier=1)
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in chart.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * 0.5)
    phi, det = chart(coords.unsqueeze(0))
    monitor = torch.rand(1, n, n) + 0.5
    loss = DagneseEquidistributionLoss()(monitor, det)
    loss.backward()
    total = sum(p.grad.abs().sum().item() for p in chart.parameters() if p.grad is not None)
    assert total > 0


def test_residual_reports_a_percentage():
    det = torch.ones(1, 8, 8)
    monitor = torch.ones(1, 8, 8)
    monitor[:, :4] = 2.0
    residual = DagneseEquidistributionLoss().residual(monitor, det)
    assert residual.shape == (1,)
    assert float(residual) > 0


# --------------------------------------------------------------------------- #
# The metric-corrected ("chart") data term
# --------------------------------------------------------------------------- #
def test_chart_loss_space_matches_the_physical_one():
    """Jacobian-weighted quadrature is the same integral, up to O(h^2)."""
    from dagnese_fno.losses import DiffeomorphicOperatorLoss

    torch.manual_seed(0)
    n = 48
    coords = grid(n)
    x = coords[..., 0]
    y = coords[..., 1]
    det = 1.0 + 0.3 * torch.cos(2.0 * x) * torch.cos(2.0 * y)      # a plausible chart
    pred = torch.sin(3 * x) * torch.cos(2 * y)
    target = pred + 0.05 * torch.cos(4 * x)
    pred, target = pred[None], target[None]

    physical = DiffeomorphicOperatorLoss(loss_space="physical")
    chart = DiffeomorphicOperatorLoss(loss_space="chart")
    a = float(physical(pred, target)[0])
    b = float(chart(pred, target, det_j=det[None])[0])
    assert b == pytest.approx(a, rel=0.1)


def test_chart_loss_space_requires_the_determinant():
    from dagnese_fno.losses import DiffeomorphicOperatorLoss

    loss = DiffeomorphicOperatorLoss(loss_space="chart")
    with pytest.raises(ValueError):
        loss(torch.zeros(1, 4, 4), torch.ones(1, 4, 4))


def test_unknown_loss_space_is_rejected():
    from dagnese_fno.losses import DiffeomorphicOperatorLoss

    with pytest.raises(ValueError):
        DiffeomorphicOperatorLoss(loss_space="fourier")
