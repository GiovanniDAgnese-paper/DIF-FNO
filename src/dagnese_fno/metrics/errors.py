"""Relative error metrics on (possibly masked) structured grids.

All metrics are *relative* and dimensionless:

.. math::
    E_{L^2} = \\frac{\\|u_\\theta - u\\|_{L^2(\\Omega)}}{\\|u\\|_{L^2(\\Omega)}},
    \\qquad
    E_{H^1} = \\frac{\\|\\nabla u_\\theta - \\nabla u\\|_{L^2(\\Omega)}}{\\|\\nabla u\\|_{L^2(\\Omega)}},
    \\qquad
    E_{H^1_\\mathrm{full}} = \\frac{\\|u_\\theta - u\\|_{H^1(\\Omega)}}{\\|u\\|_{H^1(\\Omega)}}.

:math:`E_{H^1}` is the classical *seminorm* relative error and is the quantity
reported in the DIF-FNO v4 tables; :math:`E_{H^1_\\mathrm{full}}` additionally
contains the :math:`L^2` part.  Derivatives are evaluated with the same
spacing-aware stencil used for :math:`\\det J_\\phi`
(:mod:`dagnese_fno.geometry.jacobian`), so the metrics are resolution
independent.
"""

from __future__ import annotations

import torch

from ..geometry.jacobian import Spacing, _as_spacing_pair, derivative

__all__ = ["relative_l2", "relative_h1", "relative_H1", "relative_linf", "metric_bundle"]

_EPS = 1e-12


def _as_mask(mask: torch.Tensor | None, ref: torch.Tensor) -> torch.Tensor:
    """Return a boolean mask broadcastable to ``ref``'s shape (all-True if None)."""
    if mask is None:
        return torch.ones_like(ref, dtype=torch.bool)
    m = mask.to(torch.bool).to(ref.device)
    if m.dim() < ref.dim():
        m = m.reshape((1,) * (ref.dim() - m.dim()) + tuple(m.shape))
    return m.expand_as(ref)


def _check_reduction(reduction: str) -> None:
    if reduction not in ("batch", "sample"):
        raise ValueError(f"unknown reduction '{reduction}' (use 'batch' or 'sample')")


def relative_l2(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    reduction: str = "batch",
) -> torch.Tensor:
    """Relative :math:`L^2` error.

    ``reduction="batch"`` returns a single scalar for the whole batch (the
    convention used in DIF-FNO v4); ``"sample"`` returns one value per sample.
    """
    _check_reduction(reduction)
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    m = _as_mask(mask, target)
    diff2 = (pred - target) ** 2
    ref2 = target ** 2
    if reduction == "sample":
        dims = list(range(1, diff2.dim()))
        num = (diff2 * m).sum(dim=dims)
        den = (ref2 * m).sum(dim=dims)
        return (num / den.clamp_min(_EPS)).sqrt()
    num = (diff2 * m).sum()
    den = (ref2 * m).sum()
    return (num / den.clamp_min(_EPS)).sqrt()


def relative_h1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    spacing: Spacing = 1.0,
    coords: torch.Tensor | None = None,
    reduction: str = "batch",
) -> torch.Tensor:
    """Relative :math:`H^1` **seminorm** error (gradients only)."""
    _check_reduction(reduction)
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    m = _as_mask(mask, target)
    h0, h1 = _as_spacing_pair(spacing, coords, pred.device, pred.dtype)
    dims = list(range(1, pred.dim()))

    def grad2(f: torch.Tensor) -> torch.Tensor:
        g0 = derivative(f.unsqueeze(-1), -3, h0).squeeze(-1)
        g1 = derivative(f.unsqueeze(-1), -2, h1).squeeze(-1)
        return g0 ** 2 + g1 ** 2

    num = (pred - target)
    gnum = grad2(num)
    gden = grad2(target)
    if reduction == "sample":
        return (((gnum * m).sum(dim=dims)) / (gden * m).sum(dim=dims).clamp_min(_EPS)).sqrt()
    return ((gnum * m).sum() / (gden * m).sum().clamp_min(_EPS)).sqrt()


def relative_H1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    spacing: Spacing = 1.0,
    coords: torch.Tensor | None = None,
    reduction: str = "batch",
) -> torch.Tensor:
    """Relative **full** :math:`H^1` error (:math:`\\|e\\|_{L^2}^2 + \\|\\nabla e\\|_{L^2}^2`)."""
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    m = _as_mask(mask, target)
    h0, h1 = _as_spacing_pair(spacing, coords, pred.device, pred.dtype)
    dims = list(range(1, pred.dim()))
    err = pred - target
    g0 = derivative(err.unsqueeze(-1), -3, h0).squeeze(-1)
    g1 = derivative(err.unsqueeze(-1), -2, h1).squeeze(-1)
    t0 = derivative(target.unsqueeze(-1), -3, h0).squeeze(-1)
    t1 = derivative(target.unsqueeze(-1), -2, h1).squeeze(-1)
    num = (err ** 2 + g0 ** 2 + g1 ** 2) * m
    den = (target ** 2 + t0 ** 2 + t1 ** 2) * m
    if reduction == "sample":
        return (num.sum(dim=dims) / den.sum(dim=dims).clamp_min(_EPS)).sqrt()
    return (num.sum() / den.sum().clamp_min(_EPS)).sqrt()


def relative_linf(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Relative :math:`L^\\infty` error (max abs error / max abs target)."""
    m = _as_mask(mask, target)
    err = (pred - target).abs()
    num = err.masked_fill(~m, 0.0).amax()
    den = target.abs().masked_fill(~m, 0.0).amax()
    return num / den.clamp_min(_EPS)


def metric_bundle(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    spacing: Spacing = 1.0,
    coords: torch.Tensor | None = None,
    reduction: str = "batch",
) -> dict:
    """All scalar metrics at once (detached, as plain floats)."""
    out = {
        "rel_l2": relative_l2(pred, target, mask, reduction),
        "rel_h1": relative_h1(pred, target, mask, spacing, coords, reduction),
        "rel_H1": relative_H1(pred, target, mask, spacing, coords, reduction),
        "rel_linf": relative_linf(pred, target, mask),
    }
    return {k: (float(v) if v.ndim == 0 else v.detach()) for k, v in out.items()}
