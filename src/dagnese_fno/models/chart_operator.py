"""The D'Agnese Chart Operator: a neural operator that *lives* in a certified chart.

.. math::
    u = \\mathcal P_{\\phi^{-1}} \\; \\mathcal F_\\theta\\big(\\mathcal P_\\phi a\\big),

with :math:`\\mathcal P_\\phi` the exact pullback of
:mod:`dagnese_fno.geometry.transport` and :math:`\\phi` a
:class:`~dagnese_fno.geometry.monotone.DagneseTriangularChart`.

Why this and not DIF-FNO v1
---------------------------
v1 concatenated :math:`\\phi(\\xi)` to the input channels and never transported
the field: the chart was a reparameterisation of *coordinates*, so the operator
was never evaluated in the chart and there was nothing to gain from it.  Here the
field is transported, the operator is evaluated on the reference square, and the
prediction is transported back; the chart therefore changes the discretisation
of the quantity of interest, which is the only mechanism through which a chart
can matter.

Three exact properties make the construction safe
-------------------------------------------------
**(a) It contains the baseline.**  With a zero-initialised chart
:math:`\\phi = \\mathrm{Id}` and :math:`\\mathcal P_\\phi = \\mathrm{Id}`, so the
model *is* the FNO with the same parameters at ``t = 0`` -- the chart can only
change what training decides to change.

**(b) Every intermediate state is admissible.**  The chart is globally injective
with :math:`\\det J > 0` for *all* parameter values (P5), so no training step can
produce an invalid mesh; there is no barrier to tune and no projection step.

**(c) The loss is the physical one.**  The data term is evaluated on the physical
grid after the pushforward, so the reported :math:`L^2` and :math:`H^1` errors are
directly comparable with the baseline.  (Training in the chart with an
*unweighted* MSE would be gameable by shrinking cells; the metric-correct chart
loss is the Jacobian-weighted one, :math:`\\int (\\tilde u-\\tilde u_\\theta)^2
|\\det J|\\,d\\xi`, and is available through ``loss_space="chart"``.)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..geometry.monotone import DagneseTriangularChart
from ..geometry.transport import pullback, pushforward, sample_field
from .dif_fno import ConditionEncoder
from .fno import FNO2d

__all__ = ["DagneseChartOperator"]


class DagneseChartOperator(nn.Module):
    """Certified-chart neural operator.

    Parameters
    ----------
    chart_hidden, chart_layers, chart_fourier, chart_activation:
        Geometry of the :class:`~dagnese_fno.geometry.monotone.DagneseTriangularChart`.
        ``chart_layers = 1`` is a single triangular chart (globally injective by
        P5); more layers compose charts and increase expressivity.
    chart_conditioning:
        ``"none"`` -- one chart for the whole dataset (a *learned adaptive mesh*);
        ``"global"`` -- a chart per sample, conditioned on a latent code of the
        coefficient field (a *learned mesh model*, still certified because the
        certificate does not depend on the conditioning).
    alpha:
        Jacobian exponent of the pullback: ``0`` for a scalar coefficient (the
        Darcy ``a``), ``1`` for a density (Piola transform).
    """

    def __init__(
        self,
        width: int = 24,
        modes1: int = 12,
        modes2: int = 12,
        n_layers: int = 2,
        fc_channels: int = 64,
        out_channels: int = 1,
        chart_hidden: int = 32,
        chart_layers: int = 1,
        chart_fourier: int = 3,
        chart_activation: str = "tanh",
        chart_conditioning: str = "none",
        latent_dim: int = 8,
        alpha: float = 0.0,
        chart_min_increment: float = 1e-3,
        activation: str = "gelu",
        init: str = "v4",
    ) -> None:
        super().__init__()
        if chart_conditioning not in ("none", "global"):
            raise ValueError(f"unknown chart_conditioning '{chart_conditioning}'")
        self.chart_conditioning = chart_conditioning
        self.alpha = float(alpha)

        self.chart = DagneseTriangularChart(
            hidden=chart_hidden,
            n_layers=chart_layers,
            latent_dim=latent_dim if chart_conditioning == "global" else 0,
            n_fourier=chart_fourier,
            activation=chart_activation,
            min_increment=chart_min_increment,
        )
        self.encoder = ConditionEncoder(latent_dim) if chart_conditioning == "global" else None
        self.fno = FNO2d(
            in_channels=3,
            out_channels=out_channels,
            width=width,
            modes1=modes1,
            modes2=modes2,
            n_layers=n_layers,
            fc_channels=fc_channels,
            activation=activation,
            init=init,
        )

    # ------------------------------------------------------------------ #
    def code(self, a: torch.Tensor, mask: torch.Tensor | None = None):
        return None if self.encoder is None else self.encoder(a, mask)

    def forward(self, coords: torch.Tensor, a: torch.Tensor,
                mask: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
        """Return ``(u, aux)`` with ``u`` on the physical grid ``coords``.

        ``aux`` carries ``phi``, ``det`` (analytic Jacobian determinant),
        ``a_chart`` (the transported coefficient) and ``code``.
        """
        if a.dim() != 3:
            raise ValueError(f"a must be (B, H, W), got {tuple(a.shape)}")
        if coords.dim() == 3:
            coords = coords.unsqueeze(0).expand(a.shape[0], *coords.shape)
        code = self.code(a, mask)
        phi, det = self.chart(coords, code)
        a_chart = pullback(a, phi, det, alpha=self.alpha)
        u_chart = self.fno(torch.cat([coords, a_chart.unsqueeze(-1)], dim=-1))
        phi_inverse = self.chart.inverse(coords, code, reference=coords)
        u = pushforward(u_chart, phi_inverse, det, alpha=0.0)
        det_lb = self.chart.certified_det_lower_bound(coords, code)
        return u, {"phi": phi, "det": det, "a_chart": a_chart, "code": code,
                   "u_chart": u_chart, "phi_inverse": phi_inverse, "det_lb": det_lb}

    def forward_aux(self, coords: torch.Tensor, a: torch.Tensor,
                    mask: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
        """Uniform adapter used by the training loop."""
        return self.forward(coords, a, mask)

    # ------------------------------------------------------------------ #
    def certified_det_lower_bound(self, coords: torch.Tensor,
                                  code: torch.Tensor | None = None) -> torch.Tensor:
        """Provable lower bound on ``det J`` (positive for every parameter value).

        For a chart conditioned on a latent code, the bound depends on the code;
        without one, the *parameter- and code-free* floor of
        :meth:`DagneseTriangularChart.parameter_free_det_floor` is returned, which
        is a theorem for every input.
        """
        if code is None and self.chart.latent_dim > 0:
            floor = self.chart.parameter_free_det_floor(coords.shape[-2])
            return coords.new_full((1,), float(floor))
        return self.chart.certified_det_lower_bound(coords, code)

    @property
    def is_certified(self) -> bool:
        """The chart is certified by construction -- always ``True``."""
        return True

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def equidistribution_masses(self, coords: torch.Tensor, a: torch.Tensor, spacing: float,
                                beta: float = 0.5, floor: float = 1e-3,
                                mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Chart-space monitor and analytic ``det J`` (inputs of the D'Agnese loss)."""
        from ..losses.equidistribution import gradient_monitor

        if coords.dim() == 3:
            coords = coords.unsqueeze(0).expand(a.shape[0], *coords.shape)
        code = self.code(a, mask)
        phi, det = self.chart(coords, code)
        monitor = gradient_monitor(a, spacing, beta=beta, floor=floor)
        return sample_field(monitor, phi), det

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"alpha={self.alpha}, chart_conditioning={self.chart_conditioning}"
