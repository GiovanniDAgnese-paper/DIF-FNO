"""The complete DIF-FNO training objective.

.. math::
    \\mathcal{L}_{\\text{total}} =
    \\mathcal{L}_{\\mathrm{MSE}}
    + \\alpha\\,\\mathcal{L}_{H^1}
    + \\lambda_{\\text{barrier}}\\,\\mathcal{L}_{\\text{barrier}}(\\det J_\\phi)

Note on :math:`\\mathcal{L}_{\\mathrm{MSE}}`.  The loss is averaged **over the
masked cells only**,

.. math::
    \\mathcal{L}_{\\mathrm{MSE}} = \\frac{1}{|\\Omega_h|}
    \\sum_{\\xi \\in \\Omega_h} (u_\\theta(\\xi) - u(\\xi))^2 ,

instead of over the whole bounding box.  Averaging over the full bounding box
(as done in DIF-FNO v4) makes the effective learning rate depend on the area of
the domain: on the annulus only :math:`\\approx 20\\%` of the cells are active,
so the gradient is scaled down by a factor five relative to the L-shape
(:math:`\\approx 75\\%` active).  ``normalize_by_mask=True`` removes that
domain-dependent bias from the comparison.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..geometry.jacobian import Spacing, _as_spacing_pair, derivative
from ..metrics.errors import _as_mask
from .barrier import get_folding_penalty

__all__ = ["masked_mse", "DiffeomorphicOperatorLoss"]

_EPS = 1e-12


def masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    normalize_by_mask: bool = True,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Masked mean-squared error (optionally normalised by the number of active cells).

    ``weight`` (optional, same shape as ``target``) turns the mean into a weighted
    mean: with ``weight = |det J|`` this is the Jacobian-weighted (chart) MSE,
    i.e. the physical :math:`L^2` error computed with chart-space quadrature.
    """
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    m = _as_mask(mask, target).to(pred.dtype)
    w = m if weight is None else weight.to(pred.dtype) * m
    sq = ((pred - target) ** 2) * w
    denom = w.sum().clamp_min(_EPS) if normalize_by_mask else torch.as_tensor(
        float(target.numel()), device=pred.device, dtype=pred.dtype)
    return sq.sum() / denom


class DiffeomorphicOperatorLoss(nn.Module):
    """MSE + (optional) :math:`H^1` + (optional) topological barrier.

    Parameters
    ----------
    alpha_h1:
        Weight :math:`\\alpha` of the :math:`H^1` seminorm term (0 in v4).
    lambda_barrier:
        Weight :math:`\\lambda_{\\text{barrier}}` of the folding penalty.
    barrier:
        ``"softplus"`` (D'Agnese), ``"log"``, ``"hinge"`` or ``"none"``.
    tau, margin:
        Barrier temperature and safety margin :math:`\\varepsilon` (physical units).
    normalize_by_mask:
        See :func:`masked_mse`.
    loss_space:
        ``"physical"`` (default) evaluates the data term on the physical grid;
        ``"chart"`` evaluates the *same* integral with chart-space quadrature,
        weighting every node by :math:`|\\det J_\\phi|`.  The two differ only by
        the quadrature error of the transport (:math:`O(h^2)`, P6), which is the
        point: the chart loss is not a different objective, it is the physical
        loss computed in the coordinates in which the network works.
    """

    def __init__(
        self,
        alpha_h1: float = 0.0,
        lambda_barrier: float = 0.0,
        barrier: str = "softplus",
        tau: float = 0.05,
        margin: float = 0.0,
        normalize_by_mask: bool = True,
        h1_full: bool = False,
        loss_space: str = "physical",
    ) -> None:
        super().__init__()
        self.alpha_h1 = float(alpha_h1)
        self.lambda_barrier = float(lambda_barrier)
        self.barrier_name = str(barrier)
        self.tau = float(tau)
        self.margin = float(margin)
        self.normalize_by_mask = bool(normalize_by_mask)
        self.h1_full = bool(h1_full)
        if loss_space not in ("physical", "chart"):
            raise ValueError(f"unknown loss_space '{loss_space}'")
        self.loss_space = str(loss_space)
        self.penalty: nn.Module | None = get_folding_penalty(
            barrier, tau=tau, margin=margin, reduction="mean"
        )

    # ------------------------------------------------------------------ #
    def data_terms(self, pred, target, mask, spacing=1.0, coords=None, det_j=None):
        """Return ``(mse, h1)`` components (differentiable)."""
        weight = None
        if self.loss_space == "chart":
            if det_j is None:
                raise ValueError("loss_space='chart' requires det_j")
            weight = det_j.abs()
        mse = masked_mse(pred, target, mask, self.normalize_by_mask, weight=weight)
        h1 = torch.zeros((), device=pred.device, dtype=pred.dtype)
        if self.alpha_h1 > 0:
            m = _as_mask(mask, target).to(pred.dtype)
            h0, h1s = _as_spacing_pair(spacing, coords, pred.device, pred.dtype)
            err = pred - target
            g0 = derivative(err.unsqueeze(-1), -3, h0).squeeze(-1)
            g1 = derivative(err.unsqueeze(-1), -2, h1s).squeeze(-1)
            num = ((g0 ** 2 + g1 ** 2) * m).sum()
            den = m.sum().clamp_min(_EPS) if self.normalize_by_mask else torch.as_tensor(
                float(target.numel()), device=pred.device, dtype=pred.dtype)
            h1 = num / den
            if self.h1_full:
                h1 = h1 + mse
        return mse, h1

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
        det_j: torch.Tensor | None = None,
        spacing: Spacing = 1.0,
        coords: torch.Tensor | None = None,
    ):
        """Return ``(loss, components)`` where ``components`` is a dict of scalars."""
        mse, h1 = self.data_terms(pred, target, mask, spacing, coords, det_j)
        loss = mse + self.alpha_h1 * h1
        components = {"mse": mse, "h1": h1, "barrier": torch.zeros_like(mse)}

        if self.lambda_barrier > 0 and self.penalty is not None:
            if det_j is None:
                raise ValueError("lambda_barrier > 0 requires det_j")
            bar = self.penalty(det_j)
            components["barrier"] = bar
            loss = loss + self.lambda_barrier * bar
        components["total"] = loss
        return loss, components

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (f"alpha_h1={self.alpha_h1}, lambda_barrier={self.lambda_barrier}, "
                f"barrier={self.barrier_name}(tau={self.tau}, margin={self.margin}), "
                f"loss_space={self.loss_space}")
