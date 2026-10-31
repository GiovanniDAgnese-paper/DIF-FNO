"""Tests for the D'Agnese Chart Operator.

The three structural promises of the model are tested here:

(a) at initialisation it is *exactly* the FNO baseline (same parameters, same
    prediction), so the chart can only change what training decides to change;
(b) every intermediate state is admissible -- the certificate is positive for
    random parameter values, including the ones reached by an optimizer step;
(c) the chart is actually used: perturbing it changes the prediction.
"""

from __future__ import annotations

import pytest
import torch

from dagnese_fno.models import FNO2d, DagneseChartOperator, build_model


def grid(n: int = 24) -> torch.Tensor:
    t = torch.linspace(-1.0, 1.0, n)
    return torch.stack(torch.meshgrid(t, t, indexing="ij"), dim=-1)


def make_pair(**kwargs):
    kw = dict(width=8, modes1=4, modes2=4, fc_channels=16)
    kw.update(kwargs)
    chart_model = DagneseChartOperator(**kw)
    fno = FNO2d(in_channels=3, out_channels=1, **{k: v for k, v in kw.items()
                                                  if k in ("width", "modes1", "modes2",
                                                           "fc_channels")})
    chart_model.fno.load_state_dict(fno.state_dict())
    return chart_model, fno


# --------------------------------------------------------------------------- #
# (a) it contains the baseline
# --------------------------------------------------------------------------- #
def test_model_is_the_baseline_at_initialisation():
    n = 24
    coords = grid(n)
    a = torch.rand(3, n, n)
    chart_model, fno = make_pair()
    u_chart, aux = chart_model(coords, a)
    x = torch.cat([coords.unsqueeze(0).expand(3, -1, -1, -1), a.unsqueeze(-1)], dim=-1)
    u_fno = fno(x)
    assert torch.allclose(u_chart, u_fno, atol=1e-6)
    assert torch.allclose(aux["phi"], coords.unsqueeze(0).expand(3, -1, -1, -1), atol=1e-6)


def test_parameter_count_matches_the_baseline_up_to_the_chart():
    chart_model, fno = make_pair()
    extra = chart_model.count_parameters() - sum(p.numel() for p in fno.parameters())
    assert extra == sum(p.numel() for p in chart_model.chart.parameters())
    assert 0 < extra < sum(p.numel() for p in fno.parameters())
    # At the benchmark width the chart is a small fraction of the model.
    big = DagneseChartOperator()
    assert sum(p.numel() for p in big.chart.parameters()) < 0.06 * big.count_parameters()


# --------------------------------------------------------------------------- #
# (b) every state is admissible
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scale", [0.0, 0.5, 3.0, 30.0])
def test_certificate_is_positive_for_random_parameters(scale):
    n = 24
    coords = grid(n)
    a = torch.rand(2, n, n)
    model, _ = make_pair()
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in model.chart.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * scale)
    u, aux = model(coords, a)
    assert float(aux["det"].detach().min()) > 0.0
    assert float(aux["det_lb"].detach().min()) > 0.0
    assert torch.isfinite(u).all()


def test_optimizer_steps_never_fold_the_grid():
    n = 20
    coords = grid(n)
    model, _ = make_pair()
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    a = torch.rand(4, n, n)
    target = torch.rand(4, n, n)
    for _ in range(5):
        u, aux = model(coords, a)
        loss = ((u - target) ** 2).mean() + 0.001 * (aux["phi"] ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        _, aux = model(coords, a)
        assert float((aux["det"].detach() <= 0).float().mean()) == 0.0
        assert float(aux["det_lb"].detach().min()) > 0.0


# --------------------------------------------------------------------------- #
# (c) the chart is used
# --------------------------------------------------------------------------- #
def test_perturbing_the_chart_changes_the_prediction():
    n = 24
    coords = grid(n)
    a = torch.rand(1, n, n)
    model, _ = make_pair()
    u0, _ = model(coords, a)
    gen = torch.Generator().manual_seed(1)
    with torch.no_grad():
        last = model.chart.nets[0][-1]
        last.weight.copy_(torch.randn(last.weight.shape, generator=gen))
    u1, aux = model(coords, a)
    assert float((u1 - u0).detach().abs().max()) > 1e-3
    assert float((aux["a_chart"] - a).detach().abs().max()) > 1e-3


def test_global_conditioning_uses_the_field():
    n = 20
    coords = grid(n)
    model = DagneseChartOperator(width=8, modes1=4, modes2=4, fc_channels=16,
                                 chart_conditioning="global", latent_dim=4)
    gen = torch.Generator().manual_seed(2)
    with torch.no_grad():
        for p in model.chart.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * 0.5)
        for p in model.encoder.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * 0.5)
    a = torch.rand(2, n, n)
    a[1] = torch.rand(n, n)
    _, aux = model(coords, a)
    assert not torch.allclose(aux["phi"][0], aux["phi"][1], atol=1e-4)
    assert float(aux["det"].detach().min()) > 0


def test_resolution_change_is_supported():
    model, _ = make_pair()
    for n in (20, 32, 48):
        coords = grid(n)
        a = torch.rand(1, n, n)
        u, aux = model(coords, a)
        assert u.shape == (1, n, n)
        assert aux["phi"].shape == (1, n, n, 2)


def test_aux_is_complete():
    model, _ = make_pair()
    n = 16
    _, aux = model(grid(n), torch.rand(1, n, n))
    for key in ("phi", "det", "a_chart", "u_chart", "phi_inverse", "det_lb"):
        assert key in aux


def test_bad_input_is_rejected():
    model, _ = make_pair()
    with pytest.raises(ValueError):
        model(grid(16), torch.rand(16, 16))


def test_build_model_knows_the_chart():
    for kind in ("chart", "dao", "chart-fno", "cpo"):
        model = build_model(kind, width=8, modes1=4, modes2=4)
        assert isinstance(model, DagneseChartOperator)


def test_equidistribution_masses_helper():
    model, _ = make_pair()
    n = 16
    coords = grid(n)
    a = torch.rand(2, n, n)
    monitor, det = model.equidistribution_masses(coords, a, 2 / (n - 1))
    assert monitor.shape == det.shape == (2, n, n)
    assert float(monitor.min()) > 0
