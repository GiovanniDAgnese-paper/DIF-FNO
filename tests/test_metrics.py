"""Relative error metrics: invariances, masking and reductions."""

from __future__ import annotations

import pytest
import torch

from dagnese_fno.metrics import metric_bundle, relative_H1, relative_h1, relative_l2, relative_linf


def _fields(n=32, b=3):
    torch.manual_seed(3)
    t = torch.linspace(0, 1, n)
    x, y = torch.meshgrid(t, t, indexing="ij")
    target = torch.sin(2 * torch.pi * x) * torch.cos(2 * torch.pi * y)
    target = target.unsqueeze(0).expand(b, -1, -1).contiguous()
    return target, 2.0 / (n - 1)


def test_zero_error_for_identical_fields():
    u, h = _fields()
    assert float(relative_l2(u, u)) == pytest.approx(0.0, abs=1e-12)
    assert float(relative_h1(u, u, spacing=h)) == pytest.approx(0.0, abs=1e-12)


def test_scale_invariance():
    """Doubling the prediction gives a relative error of exactly 1."""
    u, h = _fields()
    assert float(relative_l2(2 * u, u)) == pytest.approx(1.0, abs=1e-6)
    assert float(relative_h1(2 * u, u, spacing=h)) == pytest.approx(1.0, abs=1e-6)
    assert float(relative_H1(2 * u, u, spacing=h)) == pytest.approx(1.0, abs=1e-6)


def test_constant_offset_is_invisible_to_the_h1_seminorm():
    u, h = _fields()
    assert float(relative_h1(u + 5.0, u, spacing=h)) == pytest.approx(0.0, abs=1e-6)
    # ... but not to the full H1 norm.
    assert float(relative_H1(u + 5.0, u, spacing=h)) > 0.1


def test_mask_excludes_the_outside():
    u, h = _fields()
    pred = u.clone()
    pred[:, :16, :] += 100.0                      # garbage on half of the domain
    unmasked = float(relative_l2(pred, u))
    mask = torch.zeros(32, 32, dtype=torch.bool)
    mask[16:, :] = True
    masked = float(relative_l2(pred, u, mask))
    assert unmasked > 0.5
    assert masked == pytest.approx(0.0, abs=1e-12)


def test_h1_is_resolution_independent():
    """The seminorm error of a fixed continuous field does not depend on the grid."""
    def err(n):
        torch.manual_seed(0)
        t = torch.linspace(0, 1, n)
        x, y = torch.meshgrid(t, t, indexing="ij")
        u = torch.sin(2 * torch.pi * x) * torch.sin(2 * torch.pi * y)
        p = u + 0.05 * torch.sin(6 * torch.pi * x) * torch.sin(6 * torch.pi * y)
        return float(relative_h1(p[None], u[None], spacing=1.0 / (n - 1)))
    assert err(64) == pytest.approx(err(128), rel=5e-2)


@pytest.mark.parametrize("reduction", ["batch", "sample"])
def test_reductions(reduction):
    u, h = _fields(b=4)
    pred = u * 1.1
    out = relative_l2(pred, u, reduction=reduction)
    if reduction == "batch":
        assert out.ndim == 0
        assert float(out) == pytest.approx(0.1, abs=1e-6)
    else:
        assert out.shape == (4,)
        assert torch.allclose(out, torch.full_like(out, 0.1), atol=1e-6)


def test_invalid_reduction_raises():
    u, _ = _fields(b=2)
    with pytest.raises(ValueError):
        relative_l2(u, u, reduction="nope")


def test_shape_mismatch_raises():
    u, _ = _fields()
    with pytest.raises(ValueError):
        relative_l2(u, u[..., :16])


def test_relative_linf():
    u, _ = _fields()
    pred = u + 0.5 * u.abs().max()
    assert float(relative_linf(pred, u)) == pytest.approx(0.5, abs=1e-6)


def test_metric_bundle_keys_and_types():
    u, h = _fields(b=2)
    out = metric_bundle(u * 1.05, u, spacing=h)
    assert set(out) == {"rel_l2", "rel_h1", "rel_H1", "rel_linf"}
    assert all(isinstance(v, float) for v in out.values())
    assert out["rel_l2"] == pytest.approx(0.05, abs=1e-6)
