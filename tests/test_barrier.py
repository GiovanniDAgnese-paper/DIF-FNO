"""Properties of the D'Agnese barrier and of the companion folding penalties.

Each test below is one of the claims made about :math:`\\mathcal{L}_{\\text{barrier}}`
in the paper: smoothness, monotonicity, linear growth (no saturation) and the
vanishing limit inside the admissible region.
"""

from __future__ import annotations

import pytest
import torch

from dagnese_fno.losses import (
    DAgneseBarrierLoss,
    HingeFoldingPenalty,
    LogBarrierLoss,
    dagnese_barrier,
    get_folding_penalty,
    hinge_penalty,
    log_barrier,
)

DET = torch.linspace(-2.0, 2.0, 401, requires_grad=True)


# --------------------------------------------------------------------------- #
def test_non_negative():
    for f in (dagnese_barrier, hinge_penalty):
        assert float(f(DET.detach()).min()) >= 0.0


def test_monotonically_decreasing_in_det():
    for f in (dagnese_barrier, log_barrier, hinge_penalty):
        values = f(DET.detach())
        diffs = values[1:] - values[:-1]
        assert float(diffs.max()) <= 1e-12, "penalty must be non-increasing in det J"


def test_linear_growth_with_unit_slope():
    """As det J -> -inf the softplus barrier is the linear function -det J + eps."""
    tau = 0.05
    det = torch.tensor([-50.0, -100.0, -1000.0])
    value = dagnese_barrier(det, tau=tau)
    assert torch.allclose(value, -det, rtol=1e-3)


def test_vanishes_inside_the_admissible_region():
    tau = 0.05
    det = torch.tensor([1.0, 5.0, 100.0])
    assert float(dagnese_barrier(det, tau=tau).max()) < 1e-8


def test_gradient_never_saturates_and_is_bounded_by_one():
    """|dL/d(det J)| <= 1 everywhere and -> 1 inside the fold: no dead zone, no blow-up.

    Unlike a log barrier the gradient is bounded; unlike a hinge it does not
    switch off abruptly: it decays like exp(-det J / tau) in the admissible
    region (numerically it underflows to 0 only for det J >> tau, where there is
    nothing left to correct).
    """
    tau = 0.05
    det = torch.linspace(-100.0, 5.0, 2001, requires_grad=True)
    (grad,) = torch.autograd.grad(dagnese_barrier(det, tau=tau).sum(), det)
    g = grad.detach()
    assert float(g.max()) <= 0.0 + 1e-6                 # never increases the loss
    assert float(g.min()) >= -1.0 - 1e-6                # bounded by one
    assert float(g[det < -1].min()) < -0.999            # active, slope ~1 in the fold
    # Transition region: strictly negative and O(1)
    mid = (det < 0) & (det > -5 * tau)
    assert float(g[mid].max()) < -1e-3
    # Far inside the admissible region: exponentially small but never a hard kink
    assert float(g[det > 2.0].abs().max()) < 1e-8


def test_log_barrier_gradient_blows_up():
    """The classical log barrier is unbounded -- this is what the softplus fixes."""
    tau = 0.05
    det_far = torch.tensor([-1.0], requires_grad=True)
    det_near = torch.tensor([1e-6], requires_grad=True)
    g_near = torch.autograd.grad(log_barrier(det_near, tau=tau).sum(), det_near)[0]
    g_soft = torch.autograd.grad(dagnese_barrier(det_near, tau=tau).sum(), det_near)[0]
    assert abs(float(g_near)) > 1e3           # log barrier: -tau/det -> explodes
    assert abs(float(g_soft)) <= 1.0          # softplus: bounded
    assert float(dagnese_barrier(det_far.detach(), tau=tau)) < float("inf")


def test_hinge_has_a_dead_zone():
    det = torch.tensor([0.5, 1.0, 5.0], requires_grad=True)
    (g,) = torch.autograd.grad(hinge_penalty(det).sum(), det)
    assert torch.allclose(g, torch.zeros_like(g))       # zero gradient where det > 0
    assert float(hinge_penalty(torch.tensor([-1.0]))) == 1.0


def test_margin_shifts_the_barrier():
    tau, margin = 0.05, 0.2
    det = torch.tensor([0.1])
    assert float(dagnese_barrier(det, tau, margin)) > float(dagnese_barrier(det, tau, 0.0))
    # With a margin, det J = margin/2 is still penalised, det J >> margin is not.
    assert float(dagnese_barrier(torch.tensor([margin / 2]), tau, margin)) > 1e-3
    assert float(dagnese_barrier(torch.tensor([margin + 20 * tau]), tau, margin)) < 1e-6


def test_temperature_controls_the_transition_width():
    det = torch.tensor([0.0])
    assert float(dagnese_barrier(det, tau=0.5)) > float(dagnese_barrier(det, tau=0.01))


def test_closed_form_value():
    tau = 0.05
    det = torch.tensor([0.3, -0.2])
    expected = tau * torch.log1p(torch.exp(-det / tau))
    assert torch.allclose(dagnese_barrier(det, tau), expected, atol=1e-6)


# --------------------------------------------------------------------------- #
def test_module_interface_and_reductions():
    det = torch.randn(4, 8, 8) * 0.1 + 0.5
    mean_loss = DAgneseBarrierLoss(tau=0.05, reduction="mean")(det)
    sum_loss = DAgneseBarrierLoss(tau=0.05, reduction="sum")(det)
    none_loss = DAgneseBarrierLoss(tau=0.05, reduction="none")(det)
    assert none_loss.shape == det.shape
    assert sum_loss.item() == pytest.approx(none_loss.sum().item(), rel=1e-6)
    assert mean_loss.item() == pytest.approx(none_loss.mean().item(), rel=1e-6)


def test_gradient_flows_to_the_parameters():
    model = torch.nn.Linear(4, 4)
    det = model(torch.randn(3, 4)).mean(dim=-1)         # something that depends on params
    loss = DAgneseBarrierLoss(tau=0.05)(det)
    loss.backward()
    assert model.weight.grad is not None
    assert torch.isfinite(model.weight.grad).all()


@pytest.mark.parametrize("name,cls", [("softplus", DAgneseBarrierLoss), ("log", LogBarrierLoss),
                                      ("hinge", HingeFoldingPenalty)])
def test_registry(name, cls):
    assert isinstance(get_folding_penalty(name, tau=0.05), cls)
    assert get_folding_penalty("none") is None
    with pytest.raises(ValueError):
        get_folding_penalty("does-not-exist")


@pytest.mark.parametrize("tau", [0.0, -1.0])
def test_invalid_temperature(tau):
    with pytest.raises(ValueError):
        DAgneseBarrierLoss(tau=tau)
    with pytest.raises(ValueError):
        LogBarrierLoss(tau=tau)
    with pytest.raises(ValueError):
        dagnese_barrier(torch.zeros(1), tau=tau)


def test_barrier_is_smooth_everywhere():
    """Second derivative exists and is bounded -> the objective stays C^2."""
    det = torch.linspace(-5, 5, 501, requires_grad=True)
    y = dagnese_barrier(det, tau=0.2)
    (g,) = torch.autograd.grad(y.sum(), det, create_graph=True)
    (h,) = torch.autograd.grad(g.sum(), det)
    assert torch.isfinite(h).all()
    assert float(h.abs().max()) < 1e3
