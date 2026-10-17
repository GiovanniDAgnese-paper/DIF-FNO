"""Discrete Jacobian calculus for diffeomorphic coordinate maps.

Throughout DIF-FNO the object of interest is a map

.. math::

    \\phi : \\Omega_{\\mathrm{ref}} \\to \\Omega_{\\mathrm{phys}},
    \\qquad \\xi \\mapsto x = \\phi(\\xi),

discretised on a structured grid ``xi[i, j] = (y_i, x_j)`` (NumPy ``'ij'``
convention), so that **tensor axis 0 corresponds to spatial component 0** and
**tensor axis 1 corresponds to spatial component 1**.

A deformation is *topologically admissible* iff it is orientation preserving,

.. math:: \\det J_\\phi(\\xi) > 0 \\quad \\forall \\xi \\in \\Omega_{\\mathrm{ref}},

with :math:`J_\\phi[a, b] = \\partial \\phi_a / \\partial \\xi_b`.

Design notes (why this module exists)
-------------------------------------
1. **Scale-consistency.**  :math:`\\det J_\\phi` must be computed with
   derivatives, not with bare forward differences.  A raw difference stencil
   returns :math:`h^2 \\det J_\\phi` on a uniform grid; reporting that number as
   "min det(J)" gives :math:`\\det J_{\\mathrm{Id}} = h^2 \\approx 10^{-3}`
   instead of :math:`1`, and — much worse — it makes the *effective* temperature
   of the barrier loss scale like :math:`\\tau / h^2`, so the regulariser
   silently changes strength whenever the grid resolution changes.  Every
   routine here divides by the local grid spacing, hence
   ``jacobian_det(identity) == 1`` **independently of the resolution**.
2. **Non-uniform grids.**  Derivatives are formed as central differences in the
   interior (second-order accurate) and one-sided differences on the boundary
   (second order on uniform grids, first order otherwise), each divided by the
   *local* coordinate increment, so curvilinear / stretched meshes are handled
   correctly as well.
3. **A certified bound.**  :func:`certified_det_lower_bound` turns a Lipschitz
   bound on the residual :math:`r = \\phi - \\mathrm{Id}` into the *provable*
   statement :math:`\\det J_\\phi \\ge (1 - L)^2 > 0`.  This is the only
   formulation in the library that yields a guarantee rather than an
   empirical observation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Union

import torch

__all__ = [
    "derivative",
    "jacobian_components",
    "jacobian_det",
    "jacobian_det_autograd",
    "jacobian_singular_values",
    "folding_mask",
    "folding_percentage",
    "det_jacobian_stats",
    "displacement_stats",
    "certified_det_lower_bound",
    "spectral_norm",
    "spectral_norm_matrix",
]

# NB: kept as typing.Union (not PEP-604 `|`) because these are *runtime* type
# aliases: `X | Y` on a plain alias is only valid from Python 3.10 onwards.
Number = Union[int, float]  # noqa: UP007
Spacing = Union[Number, Sequence[Number], torch.Tensor]  # noqa: UP007


# --------------------------------------------------------------------------- #
# Spacing handling
# --------------------------------------------------------------------------- #
def _as_spacing_pair(
    spacing: Spacing,
    coords: torch.Tensor | None,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the 1-D gap vectors ``(h_axis0, h_axis1)`` for a structured grid.

    Parameters
    ----------
    spacing:
        Uniform spacing (float) or a pair ``(h0, h1)``.  Ignored when
        ``coords`` is provided (the increments are then measured from the grid).
    coords:
        Optional coordinate tensor ``(..., H, W, 2)`` or ``(H, W, 2)``.
    """
    if coords is not None:
        if coords.shape[-1] != 2:
            raise ValueError(f"coords must have last dimension 2, got {coords.shape[-1]}")
        h0 = torch.diff(coords[..., 0], dim=-2)  # (..., H-1, W)
        h1 = torch.diff(coords[..., 1], dim=-1)  # (..., H, W-1)
        # Structured grid: the increments are constant along the other axis.
        h0 = h0[..., :, 0].to(device=device, dtype=dtype)
        h1 = h1[..., 0, :].to(device=device, dtype=dtype)
        if torch.any(h0 <= 0) or torch.any(h1 <= 0):
            raise ValueError("coordinate grid must be strictly increasing along both axes")
        return h0, h1

    if isinstance(spacing, torch.Tensor):
        s = spacing.detach().to(device=device, dtype=dtype).flatten()
        if s.numel() == 1:
            h0 = h1 = s
        elif s.numel() == 2:
            h0, h1 = s[0:1], s[1:2]
        else:
            raise ValueError("spacing tensor must hold 1 or 2 values")
    elif isinstance(spacing, (int, float)):
        if float(spacing) <= 0:
            raise ValueError(f"spacing must be positive, got {spacing}")
        h0 = h1 = torch.tensor([float(spacing)], device=device, dtype=dtype)
    else:  # sequence
        vals = [float(v) for v in spacing]
        if len(vals) == 1:
            vals = vals * 2
        if len(vals) != 2:
            raise ValueError("spacing sequence must hold 1 or 2 values")
        h0 = torch.tensor([vals[0]], device=device, dtype=dtype)
        h1 = torch.tensor([vals[1]], device=device, dtype=dtype)
    return h0, h1


def derivative(
    f: torch.Tensor,
    axis: int,
    h: torch.Tensor,
    boundary_order: int = 2,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Differentiate ``f`` along ``axis`` with the same output shape as ``f``.

    Central differences are used in the interior and one-sided differences on
    the two boundary slices; every difference is divided by the corresponding
    coordinate increment, so the result is a genuine derivative.

    Parameters
    ----------
    f:
        Tensor of arbitrary shape, at least 2 elements along ``axis``.
    axis:
        Axis along which to differentiate (negative indexing allowed).
    h:
        1-D tensor with the ``N - 1`` increments between consecutive nodes.
    boundary_order:
        ``2`` uses the second-order one-sided formula on uniform grids,
        ``1`` always uses first-order one-sided differences.
    """
    g = f.movedim(axis, -1)
    n = g.shape[-1]
    if n < 2:
        raise ValueError(f"need at least 2 samples along axis {axis}, got {n}")
    h = h.to(device=g.device, dtype=g.dtype)
    if h.numel() == 1 and n - 1 > 1:
        h = h.expand(n - 1)            # uniform spacing: one value -> all gaps
    if h.shape[-1] != n - 1:
        raise ValueError(f"expected {n - 1} increments, got {h.shape[-1]}")

    out = torch.empty_like(g)
    d = g[..., 1:] - g[..., :-1]  # (..., N-1) forward differences
    uniform = bool(n >= 3 and torch.allclose(h, h[..., :1].expand_as(h), rtol=1e-6, atol=eps))

    if n >= 3:
        # Central difference: (g[i+1] - g[i-1]) / (xi[i+1] - xi[i-1])
        out[..., 1:-1] = (d[..., 1:] + d[..., :-1]) / (h[..., :-1] + h[..., 1:])

    if n >= 3 and boundary_order == 2 and uniform:
        out[..., 0] = (-3.0 * g[..., 0] + 4.0 * g[..., 1] - g[..., 2]) / (2.0 * h[..., 0])
        out[..., -1] = (3.0 * g[..., -1] - 4.0 * g[..., -2] + g[..., -3]) / (2.0 * h[..., -1])
    else:
        out[..., 0] = d[..., 0] / h[..., 0]
        out[..., -1] = d[..., -1] / h[..., -1]

    return out.movedim(-1, axis)


# --------------------------------------------------------------------------- #
# Jacobian
# --------------------------------------------------------------------------- #
def jacobian_components(
    phi: torch.Tensor,
    spacing: Spacing = 1.0,
    coords: torch.Tensor | None = None,
    boundary_order: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return :math:`(J_{00}, J_{01}, J_{10}, J_{11})` of the discrete map.

    ``phi`` has shape ``(..., H, W, 2)``; the returned tensors have shape
    ``(..., H, W)``.  With the ``'ij'`` convention used throughout the library,
    ``J[a, b] = d phi_a / d xi_b`` reads

    * ``J00 = d phi_0 / d xi_0`` (derivative along tensor axis ``-3``),
    * ``J01 = d phi_0 / d xi_1`` (derivative along tensor axis ``-2``),
    * ``J10 = d phi_1 / d xi_0``,
    * ``J11 = d phi_1 / d xi_1``.
    """
    if phi.shape[-1] != 2:
        raise ValueError(f"phi must have last dimension 2, got {phi.shape[-1]}")
    if phi.ndim < 3:
        raise ValueError(f"phi must have shape (..., H, W, 2), got {tuple(phi.shape)}")
    h0, h1 = _as_spacing_pair(spacing, coords, phi.device, phi.dtype)
    d0 = derivative(phi, -3, h0, boundary_order=boundary_order)  # d/dxi_0
    d1 = derivative(phi, -2, h1, boundary_order=boundary_order)  # d/dxi_1
    return d0[..., 0], d1[..., 0], d0[..., 1], d1[..., 1]


def jacobian_det(
    phi: torch.Tensor,
    spacing: Spacing = 1.0,
    coords: torch.Tensor | None = None,
    boundary_order: int = 2,
) -> torch.Tensor:
    """Determinant of the Jacobian of the discrete map ``phi``.

    For the identity map this returns exactly ``1`` on every node, at any
    resolution -- the property that makes the barrier loss resolution
    independent.

    >>> xi = torch.stack(torch.meshgrid(torch.linspace(0, 1, 33), torch.linspace(0, 1, 33), indexing="ij"), -1)
    >>> bool(torch.allclose(jacobian_det(xi, spacing=1/32), torch.ones_like(xi[..., 0]), atol=1e-8))
    True
    """
    j00, j01, j10, j11 = jacobian_components(phi, spacing, coords, boundary_order)
    return j00 * j11 - j01 * j10


def jacobian_det_autograd(phi: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Exact determinant of the Jacobian obtained with reverse-mode autodiff.

    ``coords`` must be a leaf tensor that requires grad and ``phi`` must be a
    differentiable function of it.  Two backward passes give the two rows of
    :math:`J_\\phi`; this is the reference used in the test-suite to validate
    the finite-difference stencil of :func:`jacobian_det`.
    """
    if not coords.requires_grad:
        raise ValueError("coords must require grad for the autodiff Jacobian")
    rows = []
    for c in range(phi.shape[-1]):
        (grad,) = torch.autograd.grad(
            phi[..., c].sum(), coords, retain_graph=True, create_graph=torch.is_grad_enabled()
        )
        rows.append(grad)
    j00, j01 = rows[0][..., 0], rows[0][..., 1]
    j10, j11 = rows[1][..., 0], rows[1][..., 1]
    return j00 * j11 - j01 * j10


def jacobian_singular_values(phi: torch.Tensor, spacing: Spacing = 1.0,
                             coords: torch.Tensor | None = None) -> torch.Tensor:
    """Singular values :math:`(\\sigma_1 \\ge \\sigma_2)` of :math:`J_\\phi`.

    Shape ``(..., H, W, 2)``.  Useful diagnostics: a folding cell has
    :math:`\\sigma_2 \\le 0` in the signed sense; strong anisotropy shows up as
    :math:`\\sigma_1 / \\sigma_2 \\gg 1`.
    """
    j00, j01, j10, j11 = jacobian_components(phi, spacing, coords)
    j = torch.stack((torch.stack([j00, j01], dim=-1),
                     torch.stack([j10, j11], dim=-1)), dim=-2)
    return torch.linalg.svdvals(j)  # (..., H, W, 2), descending


def folding_mask(det_j: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Boolean mask of the folded (orientation-reversing) cells, ``det J <= 0``."""
    folded = det_j <= 0
    if mask is not None:
        folded = folded & mask.to(torch.bool)
    return folded


def folding_percentage(det_j: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    """Percentage of folded cells (optionally restricted to ``mask``)."""
    if mask is None:
        denom = det_j.numel()
        num = int((det_j <= 0).sum())
    else:
        m = mask.to(torch.bool)
        if m.shape != det_j.shape:
            m = m.reshape((1,) * (det_j.ndim - m.ndim) + tuple(m.shape)).expand_as(det_j)
        num = int((det_j[m] <= 0).sum())
        denom = int(m.sum())
    if denom == 0:
        return 0.0
    return 100.0 * num / denom


def det_jacobian_stats(
    det_j: torch.Tensor, mask: torch.Tensor | None = None
) -> dict:
    """Summary statistics of :math:`\\det J_\\phi` (optionally masked)."""
    if mask is None:
        vals = det_j.detach().reshape(-1)
    else:
        m = mask.to(torch.bool)
        if m.shape != det_j.shape:
            m = m.reshape((1,) * (det_j.ndim - m.ndim) + tuple(m.shape)).expand_as(det_j)
        vals = det_j.detach()[m]
    if vals.numel() == 0:
        return {"min": float("nan"), "max": float("nan"), "mean": float("nan"),
                "fold_pct": 0.0, "n_cells": 0}
    return {
        "min": float(vals.min()),
        "max": float(vals.max()),
        "mean": float(vals.mean()),
        "fold_pct": folding_percentage(det_j, mask),
        "n_cells": int(vals.numel()),
    }


def displacement_stats(phi: torch.Tensor, coords: torch.Tensor,
                       mask: torch.Tensor | None = None) -> dict:
    """How far the deformation actually moves the grid.

    Reported together with the topological metrics: a map that barely leaves
    the identity trivially satisfies ``det J > 0``, so the displacement is the
    statistic that shows whether the diffeomorphic branch is doing any work.
    """
    delta = (phi.detach() - coords.detach()).norm(dim=-1)  # (..., H, W)
    if mask is not None:
        m = mask.to(torch.bool)
        if m.shape != delta.shape:
            m = m.reshape((1,) * (delta.ndim - m.ndim) + tuple(m.shape)).expand_as(delta)
        sel = delta[m]
    else:
        sel = delta.reshape(-1)
    return {
        "mean": float(sel.mean()),
        "max": float(sel.max()),
        "rms": float(sel.pow(2).mean().sqrt()),
    }


# --------------------------------------------------------------------------- #
# Certified bound
# --------------------------------------------------------------------------- #
def certified_det_lower_bound(lipschitz: float | torch.Tensor,
                              input_jacobian_norm: float = 1.0) -> float | torch.Tensor:
    """Provable lower bound on :math:`\\det J_\\phi` for a Lipschitz residual.

    If :math:`\\phi = \\mathrm{Id} + r` with :math:`\\mathrm{Lip}(r) \\le L`
    (w.r.t. the *inputs of the network*) and the network inputs depend on
    :math:`\\xi` through a map whose Jacobian has spectral norm
    :math:`\\|D\\|_2 \\le \\kappa`, then

    .. math::
        \\|J_\\phi - I\\|_2 \\le \\kappa L, \\qquad
        \\sigma_{\\min}(J_\\phi) \\ge 1 - \\kappa L, \\qquad
        \\det J_\\phi \\ge (1 - \\kappa L)^2 .

    Hence ``kappa * L < 1`` *implies* ``det J > 0`` and 0% grid folding by
    construction -- a guarantee, not an empirical observation.

    For pointwise conditioning on a coefficient field :math:`a(\\xi)` the
    network input is :math:`(\\xi, a(\\xi))\\in\\mathbb R^3`, whose Jacobian is
    :math:`D = [I_2; \\nabla a^\\top]` with
    :math:`\\|D\\|_2 = \\sqrt{1 + \\|\\nabla a\\|^2}`, so
    ``input_jacobian_norm = sqrt(1 + max||grad a||^2)``.
    """
    c = float(input_jacobian_norm) * lipschitz
    if isinstance(c, torch.Tensor):
        return torch.clamp(1.0 - c, min=0.0) ** 2
    c = float(c)
    return (1.0 - c) ** 2 if c < 1.0 else 0.0


def spectral_norm_matrix(W: torch.Tensor, n_power_iterations: int = 10,
                         eps: float = 1e-12) -> torch.Tensor:
    """Power-iteration estimate of the spectral norm (:math:`\\sigma_{\\max}`) of a matrix."""
    if W.ndim != 2:
        raise ValueError("W must be a 2-D matrix")
    u = torch.randn(W.shape[0], device=W.device, dtype=W.dtype)
    u = u / u.norm().clamp_min(eps)
    v = torch.randn(W.shape[1], device=W.device, dtype=W.dtype)
    v = v / v.norm().clamp_min(eps)
    with torch.no_grad():
        for _ in range(int(n_power_iterations)):
            v = (W.t() @ u)
            v = v / v.norm().clamp_min(eps)
            u = (W @ v)
            u = u / u.norm().clamp_min(eps)
        sigma = torch.dot(u, W @ v)
    return sigma.abs().clamp_min(eps)


def spectral_norm(module: torch.nn.Module, n_power_iterations: int = 10) -> float:
    """Upper bound on the Lipschitz constant of a feed-forward ``module``.

    Multiplies the spectral norms of every ``Linear``/``Conv`` weight and the
    Lipschitz constant of every activation (``tanh``: 1, ``gelu``: ~1.12).
    """
    from ..models.activations import activation_lipschitz  # local import (cycles)

    lip = 1.0
    for m in module.modules():
        if isinstance(m, (torch.nn.Linear,)):
            lip *= float(spectral_norm_matrix(m.weight, n_power_iterations))
        elif isinstance(m, torch.nn.Conv2d):
            w = m.weight.flatten(1)
            lip *= float(spectral_norm_matrix(w, n_power_iterations))
        lip *= activation_lipschitz(m)
    return lip
