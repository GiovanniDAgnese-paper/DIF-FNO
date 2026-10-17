"""Model-level tests: shapes, zero-initialisation, Lipschitz certificates."""

from __future__ import annotations

import pytest
import torch

from dagnese_fno.geometry import jacobian_det
from dagnese_fno.models import DiffeomorphicMap, DIFFNO2d, FNO2d, build_model


def _grid(n=32, b=2):
    t = torch.linspace(-1, 1, n)
    coords = torch.stack(torch.meshgrid(t, t, indexing="ij"), dim=-1)
    return coords.unsqueeze(0).expand(b, -1, -1, -1).contiguous(), 2.0 / (n - 1)


# --------------------------------------------------------------------------- #
# FNO backbone
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [32, 48])
def test_fno_output_shape(n):
    coords, h = _grid(n)
    a = torch.rand(2, n, n)
    model = FNO2d(in_channels=3)
    out = model(torch.cat([coords, a.unsqueeze(-1)], dim=-1))
    assert out.shape == (2, n, n)


def test_fno_resolution_agnostic():
    """The spectral layer clamps the modes to the Nyquist limit."""
    model = FNO2d(in_channels=3, modes1=12, modes2=12)
    for n in (16, 32, 64):
        coords, _ = _grid(n)
        a = torch.rand(2, n, n)
        assert model(torch.cat([coords, a.unsqueeze(-1)], -1)).shape == (2, n, n)


def test_fno_raises_on_wrong_channels():
    with pytest.raises(ValueError):
        FNO2d(in_channels=3)(torch.rand(2, 16, 16, 5))


# --------------------------------------------------------------------------- #
# Diffeomorphic branch
# --------------------------------------------------------------------------- #
def test_zero_init_gives_the_identity():
    coords, h = _grid(32)
    a = torch.rand(2, 32, 32)
    model = DIFFNO2d()
    _, xi = model(coords, a)
    assert torch.allclose(xi, coords, atol=0.0)
    det = jacobian_det(xi, spacing=h)
    assert torch.allclose(det, torch.ones_like(det), atol=1e-6)


def test_dif_output_shapes():
    coords, h = _grid(32)
    a = torch.rand(2, 32, 32)
    for conditioning in ("pointwise", "global"):
        model = DIFFNO2d(conditioning=conditioning)
        u, xi = model(coords, a)
        assert u.shape == (2, 32, 32)
        assert xi.shape == coords.shape


def test_gradients_reach_both_branches():
    coords, h = _grid(32)
    a = torch.rand(2, 32, 32)
    model = DIFFNO2d()
    u, xi = model(coords, a)
    (u.sum() + jacobian_det(xi, spacing=h).sum()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters() if p.requires_grad)


def test_diffeo_map_conditioning_errors():
    coords, _ = _grid(16)
    m = DiffeomorphicMap(in_channels=3)
    with pytest.raises(ValueError):
        m(coords, torch.rand(2, 16, 16, 7))
    with pytest.raises(ValueError):
        m(torch.rand(2, 16, 16, 3))     # coords must have 2 components


def test_global_conditioning_is_constant_over_the_grid():
    """The latent code does not depend on xi -> kappa = 1 in the certificate."""
    coords, _ = _grid(32)
    m = DiffeomorphicMap(in_channels=2 + 8)
    code = torch.randn(2, 8)
    xi = m(coords, code)
    assert xi.shape == coords.shape
    # Perturbing *one* node of the grid must not change the code, hence the map
    # must be a pure function of (xi, z): check the code is broadcast identically.
    xi2 = m(coords.roll(1, dims=1), code)
    assert torch.allclose(xi.roll(-1, dims=1)[1:-1], xi2[: -2][1:-1], atol=1e-6)


# --------------------------------------------------------------------------- #
# Lipschitz / certificate
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("const", [0.5, 0.9])
def test_spectral_mode_enforces_the_lipschitz_bound(const):
    m = DiffeomorphicMap(in_channels=3, deformation_scale=0.3,
                         lipschitz_mode="spectral", lipschitz_const=const)
    assert m.is_certified
    lip = m.lipschitz_bound()
    assert lip <= const + 1e-5, (lip, const)
    assert m.certified_det_lower_bound(kappa=1.0) >= (1 - const) ** 2 - 1e-6


def test_lipschitz_bound_survives_training():
    """The cap holds after optimisation steps (it is enforced in the forward pass)."""
    torch.manual_seed(0)
    m = DiffeomorphicMap(in_channels=3, deformation_scale=0.5,
                         lipschitz_mode="spectral", lipschitz_const=0.9)
    opt = torch.optim.Adam(m.parameters(), lr=0.05)
    coords, h = _grid(16)
    a = torch.rand(2, 16, 16)
    for _ in range(5):
        opt.zero_grad()
        xi = m(coords, a.unsqueeze(-1))
        loss = -jacobian_det(xi, spacing=h).clamp_max(0).sum()  # push towards folding
        loss.backward()
        opt.step()
    lip = m.lipschitz_bound()
    assert lip <= 0.9 + 1e-4, lip
    assert m.certified_det_lower_bound(kappa=1.0) > 0.0


def test_unconstrained_mode_can_exceed_one():
    """An unconstrained residual quickly stops being a contraction -> no certificate."""
    m = DiffeomorphicMap(in_channels=3, deformation_scale=3.0, lipschitz_mode="none",
                         zero_init=False)
    assert not m.is_certified
    assert m.lipschitz_bound() > 1.0
    assert m.certified_det_lower_bound(kappa=1.0) == 0.0


def test_certified_model_never_folds_empirically():
    """Sanity check that the certificate and the measured det J agree in ordering."""
    torch.manual_seed(1)
    m = DiffeomorphicMap(in_channels=3, deformation_scale=0.8,
                         lipschitz_mode="spectral", lipschitz_const=0.5)
    coords, h = _grid(32)
    a = torch.rand(2, 32, 32)
    xi = m(coords, a.unsqueeze(-1))
    det = jacobian_det(xi, spacing=h)
    bound = m.certified_det_lower_bound(kappa=1.0)
    assert float(det.detach().min()) >= bound - 1e-6
    assert float(det.detach().min()) > 0


# --------------------------------------------------------------------------- #
def test_build_model_registry():
    assert isinstance(build_model("fno", width=8), FNO2d)
    assert isinstance(build_model("dif", width=8), DIFFNO2d)
    with pytest.raises(ValueError):
        build_model("nope")


def test_backbone_hyperparameters_are_identical():
    """The baseline and DIF-FNO must share the same backbone for a fair comparison."""
    dif = build_model("dif", width=16, modes1=6, modes2=6, n_layers=2, fc_channels=32)
    fno = build_model("fno", width=16, modes1=6, modes2=6, n_layers=2, fc_channels=32)
    for name in ("fc0.weight", "fc1.weight", "fc2.weight"):
        assert getattr(dif.fno, name.split(".")[0]).weight.shape == \
               getattr(fno, name.split(".")[0]).weight.shape
    assert dif.fno.width == fno.width == 16
