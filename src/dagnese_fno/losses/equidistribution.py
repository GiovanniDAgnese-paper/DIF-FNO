"""The D'Agnese Equidistribution Loss: the objective that gives the chart a job.

A diffeomorphic chart with ``det J > 0`` is *valid*; it is not necessarily
*useful*.  The classical theory of moving meshes (de Boor 1973; Babuška &
Rheinboldt 1978) supplies the criterion: given a monitor function
:math:`m > 0` and :math:`N_c` cells, the mesh that minimises

.. math::
    \\max_C \\int_C m \\; dx

equidistributes the monitor,

.. math::
    \\int_{C_i} m \\, dx = \\frac{1}{N_c}\\int_\\Omega m \\, dx \\quad \\forall i ,

and the minimum value is :math:`\\frac{1}{N_c}\\int m`.

Proposition P7 (D'Agnese, optimality of the equidistributed chart)
-------------------------------------------------------------------
Let :math:`I_m = \\int_\\Omega m` and let :math:`m \\ge m_0 > 0`.  For any chart
:math:`\\Phi` and any :math:`f \\in C^2` with :math:`|f''| \\le M`,

.. math::
    \\|f - I_h f\\|_\\infty \\;\\le\\; \\frac{M}{8}\\max_C h_C^2
    \\;\\le\\; \\frac{M}{8 m_0^2} \\max_C \\Big(\\int_C m\\Big)^2 ,

and the last factor is *minimised*, with value :math:`(I_m/N_c)^2`, exactly when
the monitor is equidistributed.  Hence the equidistributed chart minimises a
certified upper bound on the representation error of the field at fixed
resolution -- which is precise sense in which an adaptive chart can help, and
the only sense in which the present work claims it does.

The loss implemented here is the normalised variance of the cell masses,

.. math::
    \\mathcal L_{\\text{eq}}(\\Phi) = \\frac{1}{N_c}\\sum_C
    \\Big(\\frac{M_C}{\\bar M} - 1\\Big)^2 ,
    \\qquad M_C = \\int_C m(\\Phi)\\,|\\det J_\\Phi|\\,d\\xi ,

which is :math:`\\ge 0`, vanishes **if and only if** the chart equidistributes
the monitor, and is scale free (it does not change if the monitor is multiplied
by a constant, or if the domain is rescaled).

Worst-cell control
------------------
For any vector :math:`x` with mean one, :math:`|x_C-1| \\le
\\sqrt{N_c\\,V(x)}` by Cauchy--Schwarz, so

.. math::
    \\max_C \\frac{M_C}{\\bar M} \\le 1 + \\sqrt{N_c\\,\\mathcal L_{\\mathrm{eq}}},
    \\qquad
    \\min_C \\frac{M_C}{\\bar M} \\ge 1 - \\sqrt{N_c\\,\\mathcal L_{\\mathrm{eq}}} .

The implementation evaluates the functional on the full :math:`N \\times N`
node array (the one-sided boundary stencil duplicates the first and last
row/column), so :math:`N_c = N^2` in this bound -- conservative, and identical
to the cell count up to the boundary.

Why the monitor mass and not just the monitor
---------------------------------------------
Because :math:`M_C` is the *physical* quantity a cell represents: multiplying by
:math:`|\\det J|` converts the chart density into a physical mass.  Without that
factor the loss could be driven to zero by shrinking every cell, i.e. it would
be gameable (the same defect that the DIF-FNO v1 barrier has in another guise).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..geometry.jacobian import Spacing, _as_spacing_pair, derivative

__all__ = ["DagneseEquidistributionLoss", "gradient_monitor"]


def gradient_monitor(field: torch.Tensor, spacing: float, beta: float = 0.5,
                     floor: float = 1e-3, smooth: int = 0) -> torch.Tensor:
    """Monitor :math:`m = (\\varepsilon + |\\nabla a|^2)^\\beta` (spacing aware).

    It is the practical surrogate of the interpolation-theoretic monitor
    :math:`|f''|^{1/2}`: for a piecewise-linear interpolant the error scales like
    the *first* derivative times the cell size when the function is only
    piecewise smooth (which is the case for a log-normal Darcy coefficient), and
    the square root of a gradient measure is what equidistributes the product
    :math:`h\\,|\\nabla a|`.  ``floor`` keeps :math:`m` bounded below, which is
    what the theorem needs (:math:`m \\ge m_0 > 0`).
    """
    if beta <= 0:
        raise ValueError("beta must be positive")
    h0, h1 = _as_spacing_pair(spacing, None, field.device, field.dtype)
    g0 = derivative(field.unsqueeze(-1), -3, h0).squeeze(-1)
    g1 = derivative(field.unsqueeze(-1), -2, h1).squeeze(-1)
    m = (floor + g0**2 + g1**2).pow(beta)
    if smooth and smooth > 1:
        k = int(smooth) | 1
        pad = k // 2
        x = m.unsqueeze(1)
        x = nn.functional.avg_pool2d(nn.functional.pad(x, (pad, pad, pad, pad), mode="replicate"),
                                     k, stride=1)
        m = x.squeeze(1)
    return m


class DagneseEquidistributionLoss(nn.Module):
    """Normalised variance of the monitor mass per cell.

    Parameters
    ----------
    beta, floor, smooth:
        Passed to :func:`gradient_monitor`.
    use_mask:
        Restrict the variance to the cells inside the domain.  Off by default:
        the chart covers the whole square and the outside of the domain carries a
        constant coefficient, so the monitor is well defined everywhere.

    The module is *not* a network: it is a functional of ``(a, det J)`` and is
    differentiable with respect to both.
    """

    def __init__(self, beta: float = 0.5, floor: float = 1e-3, smooth: int = 0,
                 reduction: str = "mean", monitor_weight: bool = True) -> None:
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"unknown reduction '{reduction}'")
        self.beta = float(beta)
        self.floor = float(floor)
        self.smooth = int(smooth)
        self.reduction = reduction
        self.monitor_weight = bool(monitor_weight)

    def monitor(self, a: torch.Tensor, spacing: float) -> torch.Tensor:
        return gradient_monitor(a, spacing, self.beta, self.floor, self.smooth)

    def forward(self, monitor: torch.Tensor, det_j: torch.Tensor) -> torch.Tensor:
        """``monitor`` and ``det_j`` are ``(B, N, N)``; returns a scalar (or ``(B,)``)."""
        if monitor.shape != det_j.shape:
            raise ValueError(f"shape mismatch: {tuple(monitor.shape)} vs {tuple(det_j.shape)}")
        mass = monitor * det_j.abs() if self.monitor_weight else monitor
        flat = mass.reshape(mass.shape[0], -1)
        mean = flat.mean(dim=1, keepdim=True).clamp_min(1e-12)
        residual = (flat / mean - 1.0) ** 2
        per_sample = residual.mean(dim=1)
        if self.reduction == "none":
            return per_sample
        if self.reduction == "sum":
            return per_sample.sum()
        return per_sample.mean()

    def residual(self, monitor: torch.Tensor, det_j: torch.Tensor) -> torch.Tensor:
        """Root-mean-square relative deviation of the cell masses (a diagnostic, in %)."""
        mass = monitor * det_j.abs()
        flat = mass.reshape(mass.shape[0], -1)
        mean = flat.mean(dim=1, keepdim=True).clamp_min(1e-12)
        return 100.0 * (flat / mean - 1.0).pow(2).mean(dim=1).sqrt()

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"beta={self.beta}, floor={self.floor}, smooth={self.smooth}"
