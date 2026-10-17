"""DIF-FNO: diffeomorphic branch + Fourier Neural Operator.

The model is the composition

.. math::
    u = \\underbrace{\\mathcal F_\\theta}_{\\text{FNO}}
        \\big(\\underbrace{\\phi_\\psi(\\xi, a)}_{\\text{diffeo}},\\, a\\big)

where the FNO backbone is *identical* to the one used for the baseline, so the
benchmark isolates the effect of the learned coordinate map.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .diffeo import DiffeomorphicMap
from .fno import FNO2d

__all__ = ["ConditionEncoder", "DIFFNO2d"]


class ConditionEncoder(nn.Module):
    """Encode :math:`a` into a *global* latent code :math:`z \\in \\mathbb R^{d}`.

    Crucially :math:`z` does **not** depend on :math:`\\xi`, so the Jacobian of
    the network inputs w.r.t. :math:`\\xi` is the identity and the injectivity
    certificate reads :math:`\\det J_\\phi \\ge (1 - L)^2` with :math:`\\kappa = 1`
    (pointwise conditioning pays the extra factor
    :math:`\\sqrt{1 + \\|\\nabla a\\|^2}`).
    """

    def __init__(self, latent_dim: int = 8, hidden: int = 64, coarse: int = 8) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.coarse = coarse
        self.mlp = nn.Sequential(
            nn.Linear(coarse * coarse, hidden),
            nn.Tanh(),
            nn.Linear(hidden, latent_dim),
            nn.Tanh(),
        )

    def forward(self, a: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """:math:`a`: ``(B, H, W)`` -> ``(B, d)``."""
        x = a.unsqueeze(1)  # (B, 1, H, W)
        if mask is not None:
            x = x * mask.to(x.dtype).reshape(1, *a.shape[-2:])
        x = nn.functional.adaptive_avg_pool2d(x, (self.coarse, self.coarse))
        return self.mlp(x.flatten(1))


class DIFFNO2d(nn.Module):
    """Diffeomorphic Fourier Neural Operator.

    Parameters
    ----------
    conditioning:
        ``"pointwise"`` feeds :math:`(\\xi, a(\\xi))` to the deformation
        network (maximal local adaptivity, weaker certificate);
        ``"global"`` feeds :math:`(\\xi, z(a))` with a per-sample latent code
        (certificate holds with :math:`\\kappa = 1`).
    See :class:`~dagnese_fno.models.fno.FNO2d` and
    :class:`~dagnese_fno.models.diffeo.DiffeomorphicMap` for the remaining
    arguments.
    """

    def __init__(
        self,
        width: int = 24,
        modes1: int = 12,
        modes2: int = 12,
        n_layers: int = 2,
        fc_channels: int = 64,
        out_channels: int = 1,
        diffeo_hidden: int = 32,
        diffeo_layers: int = 2,
        deformation_scale: float = 0.1,
        lipschitz_mode: str = "none",
        lipschitz_const: float = 0.9,
        conditioning: str = "pointwise",
        latent_dim: int = 8,
        activation: str = "gelu",
        diffeo_activation: str = "tanh",
        init: str = "v4",
    ) -> None:
        super().__init__()
        if conditioning not in ("pointwise", "global"):
            raise ValueError(f"unknown conditioning '{conditioning}'")
        self.conditioning = conditioning
        cond_channels = 1 if conditioning == "pointwise" else int(latent_dim)

        self.diffeo = DiffeomorphicMap(
            in_channels=2 + cond_channels,
            hidden=diffeo_hidden,
            n_layers=diffeo_layers,
            deformation_scale=deformation_scale,
            lipschitz_mode=lipschitz_mode,
            lipschitz_const=lipschitz_const,
            activation=diffeo_activation,
        )
        self.encoder = ConditionEncoder(latent_dim) if conditioning == "global" else None
        # The backbone sees the deformed coordinates instead of the physical ones.
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
    def forward(self, coords: torch.Tensor, a: torch.Tensor,
                mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(u, phi(xi))`` with ``coords`` ``(B, H, W, 2)`` and ``a`` ``(B, H, W)``."""
        if a.dim() != 3:
            raise ValueError(f"a must be (B, H, W), got {tuple(a.shape)}")
        if self.conditioning == "pointwise":
            condition = a.unsqueeze(-1)
        else:
            condition = self.encoder(a, mask)
        xi = self.diffeo(coords, condition)
        x = torch.cat([xi, a.unsqueeze(-1)], dim=-1)
        u = self.fno(x)
        return u, xi

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def lipschitz_bound(self, n_power_iterations: int = 10) -> float:
        return self.diffeo.lipschitz_bound(n_power_iterations)

    def certified_det_lower_bound(self, kappa: float = 1.0) -> float:
        return self.diffeo.certified_det_lower_bound(kappa)
