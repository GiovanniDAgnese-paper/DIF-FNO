"""The D'Agnese topological barrier and companion penalties on :math:`\\det J_\\phi`.

All penalties act on :math:`d = \\det J_\\phi` **in physical units**
(:math:`d = 1` for the identity -- see
:mod:`dagnese_fno.geometry.jacobian`).  Expressing them in physical units is
what makes the temperature :math:`\\tau` a meaningful length and the
regularisation independent of the grid resolution.

D'Agnese barrier (v4, default):

.. math::
    \\mathcal L_{\\text{barrier}}(d) = \\tau\\; \\mathrm{softplus}\\!\\left(\\frac{\\varepsilon - d}{\\tau}\\right)

with margin :math:`\\varepsilon \\ge 0`.  Properties (see ``docs/THEORY.md``):

* :math:`C^\\infty` in :math:`d` -- no kink, no dead zone;
* monotone decreasing, :math:`\\mathcal L \\to 0` for :math:`d \\gg \\tau`;
* asymptotically **linear** with slope 1 as :math:`d \\to -\\infty`, so the
  gradient never vanishes (no saturation) -- unlike :math:`\\max(0, -d)^2`-type
  penalties far inside the folded region;
* the gradient is bounded, :math:`|\\partial_d \\mathcal L| \\le 1`, so it cannot
  blow up the update the way a log-barrier does.

None of this constitutes a *guarantee*: it is a soft penalty.  The only
guarantee available in this library is the Lipschitz certificate
(:func:`dagnese_fno.geometry.jacobian.certified_det_lower_bound`).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "dagnese_barrier",
    "log_barrier",
    "hinge_penalty",
    "DAgneseBarrierLoss",
    "LogBarrierLoss",
    "HingeFoldingPenalty",
    "get_folding_penalty",
]


def dagnese_barrier(det_j: torch.Tensor, tau: float = 0.05, margin: float = 0.0) -> torch.Tensor:
    """Element-wise softplus barrier :math:`\\tau \\cdot \\mathrm{softplus}((\\varepsilon - d)/\\tau)`."""
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    return tau * F.softplus((float(margin) - det_j) / tau)


def log_barrier(det_j: torch.Tensor, tau: float = 0.05, margin: float = 0.0) -> torch.Tensor:
    """Classical log barrier :math:`-\\tau \\log\\big((d - \\varepsilon)/\\tau\\big)`.

    Defined only for :math:`d > \\varepsilon`; it blows up at the boundary (which
    is why it needs a safeguard in practice) and vanishes for large :math:`d`.
    Kept here for the ablation reported in the README.
    """
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    z = (det_j - float(margin)) / tau
    return -tau * torch.log(torch.clamp(z, min=1e-6))


def hinge_penalty(det_j: torch.Tensor, tau: float = 0.05, margin: float = 0.0) -> torch.Tensor:
    """ReLU hinge :math:`\\max(0, \\varepsilon - d)` (common mesh-quality penalty).

    Continuous, but with a kink at :math:`d = \\varepsilon` and zero gradient
    wherever :math:`d > \\varepsilon`.
    """
    return F.relu(float(margin) - det_j)


class _BaseFoldingPenalty(nn.Module):
    reduction: str

    def _reduce(self, x: torch.Tensor) -> torch.Tensor:
        if self.reduction == "mean":
            return x.mean()
        if self.reduction == "sum":
            return x.sum()
        return x


class DAgneseBarrierLoss(_BaseFoldingPenalty):
    """D'Agnese topological barrier loss (softplus formulation, v4).

    .. math::
        \\mathcal L_{\\text{barrier}}(\\det J_\\phi)
        = \\Big\\langle \\tau\\, \\mathrm{softplus}\\big((\\varepsilon - \\det J_\\phi)/\\tau\\big) \\Big\\rangle

    Parameters
    ----------
    tau:
        Temperature.  It sets the width of the transition layer around
        :math:`\\det J = \\varepsilon` *in physical units*: for
        :math:`\\det J \\gtrsim \\varepsilon + 5\\tau` the penalty is numerically
        zero, for :math:`\\det J \\lesssim \\varepsilon - 5\\tau` it is the
        linear function :math:`\\varepsilon - \\det J`.
    margin:
        Enforce a strict margin :math:`\\det J \\ge \\varepsilon > 0` instead of
        mere positivity; useful to keep the map well conditioned, not just
        orientation preserving.
    reduction:
        ``"mean"`` | ``"sum"`` | ``"none"``.
    """

    def __init__(self, tau: float = 0.05, margin: float = 0.0, reduction: str = "mean") -> None:
        super().__init__()
        if tau <= 0:
            raise ValueError("tau must be positive")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"unknown reduction '{reduction}'")
        self.tau = float(tau)
        self.margin = float(margin)
        self.reduction = reduction

    def forward(self, det_J: torch.Tensor) -> torch.Tensor:
        return self._reduce(dagnese_barrier(det_J, self.tau, self.margin))

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"tau={self.tau}, margin={self.margin}, reduction={self.reduction}"


class LogBarrierLoss(_BaseFoldingPenalty):
    """Classical logarithmic barrier on :math:`\\det J_\\phi` (ablation baseline)."""

    def __init__(self, tau: float = 0.05, margin: float = 0.0, reduction: str = "mean") -> None:
        super().__init__()
        if tau <= 0:
            raise ValueError("tau must be positive")
        self.tau = float(tau)
        self.margin = float(margin)
        self.reduction = reduction

    def forward(self, det_J: torch.Tensor) -> torch.Tensor:
        return self._reduce(log_barrier(det_J, self.tau, self.margin))


class HingeFoldingPenalty(_BaseFoldingPenalty):
    """ReLU hinge penalty on :math:`\\det J_\\phi` (ablation baseline)."""

    def __init__(self, tau: float = 0.05, margin: float = 0.0, reduction: str = "mean") -> None:
        super().__init__()
        self.tau = float(tau)
        self.margin = float(margin)
        self.reduction = reduction

    def forward(self, det_J: torch.Tensor) -> torch.Tensor:
        return self._reduce(hinge_penalty(det_J, self.tau, self.margin))


_REGISTRY = {
    "softplus": DAgneseBarrierLoss,
    "dagnese": DAgneseBarrierLoss,
    "log": LogBarrierLoss,
    "hinge": HingeFoldingPenalty,
    "relu": HingeFoldingPenalty,
    "none": None,
}


def get_folding_penalty(name: str = "softplus", **kwargs) -> nn.Module | None:
    """Instantiate a folding penalty by name (``softplus`` | ``log`` | ``hinge`` | ``none``)."""
    key = str(name).lower()
    if key not in _REGISTRY:
        raise ValueError(f"unknown folding penalty '{name}'; available: {sorted(_REGISTRY)}")
    cls = _REGISTRY[key]
    return None if cls is None else cls(**kwargs)
