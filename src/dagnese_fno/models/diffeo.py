"""The diffeomorphic branch: :math:`\\phi(\\xi) = \\xi + s\\, r(\\xi, a)`.

Two properties make this parameterisation the right object to study:

**P1 (identity at initialisation).**  The last layer is zero-initialised, hence
:math:`r \\equiv 0`, :math:`\\phi = \\mathrm{Id}` and :math:`\\det J_\\phi = 1`
at :math:`t = 0`: training *starts* inside the admissible set.

**P2 (certifiable invertibility).**  Writing :math:`\\phi = \\mathrm{Id} + r`,
:math:`J_\\phi = I + \\partial r`.  If the residual is Lipschitz with constant
:math:`\\mathrm{Lip}(r) \\le L` (w.r.t. the network inputs) and the network
inputs depend on :math:`\\xi` with Jacobian spectral norm :math:`\\kappa`, then

.. math::
    \\|J_\\phi - I\\|_2 \\le \\kappa L \\;\\Longrightarrow\\;
    \\sigma_{\\min}(J_\\phi) \\ge 1 - \\kappa L \\;\\Longrightarrow\\;
    \\det J_\\phi \\ge (1 - \\kappa L)^2 .

So ``kappa * L < 1`` gives 0% grid folding **by construction**.  The library
can enforce :math:`L` with (one-sided) spectral normalisation of every layer
(:class:`SpectralLinear`), which is what ``lipschitz_mode="spectral"`` does.
Without it, ``det J > 0`` is only encouraged by the barrier loss and remains an
*a posteriori* observation, not a guarantee.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..geometry.jacobian import certified_det_lower_bound, spectral_norm_matrix
from .activations import activation_lipschitz, get_activation

__all__ = ["SpectralLinear", "DiffeomorphicMap"]


class SpectralLinear(nn.Module):
    """Linear layer with a one-sided spectral-norm cap.

    The effective weight is :math:`W_{\\mathrm{eff}} = W \\cdot \\min(1, c/\\sigma(W))`
    with :math:`\\sigma(W)` estimated by power iteration.  Because the cap is
    one-sided (no amplification when :math:`\\sigma(W) < c`), the layer stays
    well-behaved when the weights are zero -- which is exactly what happens at
    initialisation, since the diffeomorphic branch starts at the identity.

    Consequence: :math:`\\sigma(W_{\\mathrm{eff}}) \\le c` at all times, so the
    product of the caps over the layers bounds the Lipschitz constant of the
    whole network.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        cap: float | None = 1.0,
        n_power_iterations: int = 1,
        eps: float = 1e-8,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.cap = cap
        self.n_power_iterations = int(n_power_iterations)
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.register_buffer("u", nn.init.normal_(torch.empty(out_features), mean=0.0, std=1.0))
        self.register_buffer("v", nn.init.normal_(torch.empty(in_features), mean=0.0, std=1.0))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / fan_in**0.5 if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    @torch.no_grad()
    def _update_iterates(self) -> None:
        w = self.weight
        u, v = self.u, self.v
        for _ in range(self.n_power_iterations):
            v = w.t().mv(u)
            nv = v.norm()
            if nv < self.eps:
                return  # zero (or numerically zero) weight: keep previous iterates
            v = v / nv
            u = w.mv(v)
            nu = u.norm()
            if nu < self.eps:
                return
            u = u / nu
        self.u.copy_(u)
        self.v.copy_(v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        if self.cap is not None and self.training:
            self._update_iterates()
        if self.cap is None:
            w_eff = w
        else:
            if not self.training:
                self._update_iterates()
            sigma = torch.dot(self.u, w.mv(self.v)).abs()
            # min(1, cap/sigma) evaluated without division by zero
            factor = torch.clamp(self.cap / torch.clamp(sigma, min=self.eps), max=1.0)
            w_eff = w * factor
        return nn.functional.linear(x, w_eff, self.bias)

    def sigma(self, n_power_iterations: int = 10) -> torch.Tensor:
        """Current spectral norm of the *effective* weight (post-cap)."""
        w = self.weight
        if self.cap is None:
            return spectral_norm_matrix(w, n_power_iterations)
        sigma = spectral_norm_matrix(w, n_power_iterations)
        return torch.minimum(sigma, torch.as_tensor(self.cap, device=sigma.device, dtype=sigma.dtype))

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"in={self.in_features}, out={self.out_features}, cap={self.cap}"


class DiffeomorphicMap(nn.Module):
    """Input-conditioned, zero-initialised residual deformation.

    Parameters
    ----------
    in_channels:
        Dimension of the conditioning input: ``2 + k`` where ``k`` is the number
        of extra channels concatenated to the coordinates (``k = 1`` for
        pointwise conditioning on :math:`a(\\xi)`, ``k = latent_dim`` for global
        conditioning on a latent code).
    hidden, n_layers:
        MLP trunk.
    deformation_scale:
        Output scale :math:`s` of the residual.  In Lipschitz mode this is
        overridden so that :math:`s \\prod_i \\sigma(W_i) \\le` ``lipschitz_const``.
    lipschitz_mode:
        ``"none"`` (plain ``nn.Linear``) or ``"spectral"`` (capped spectral
        norms -- enables the certificate of :func:`certified_det_lower_bound`).
    lipschitz_const:
        Target Lipschitz constant :math:`L < 1` of the residual.
    activation:
        Activation of the trunk (``tanh`` is 1-Lipschitz and is the natural
        choice when a certificate is wanted).
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden: int = 32,
        n_layers: int = 2,
        deformation_scale: float = 0.1,
        lipschitz_mode: str = "none",
        lipschitz_const: float = 0.9,
        activation: str = "tanh",
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        if lipschitz_mode not in ("none", "spectral"):
            raise ValueError(f"unknown lipschitz_mode '{lipschitz_mode}'")
        if lipschitz_const <= 0:
            raise ValueError("lipschitz_const must be positive")
        self.in_channels = in_channels
        self.hidden = hidden
        self.n_layers = n_layers
        self.lipschitz_mode = lipschitz_mode
        self.lipschitz_const = float(lipschitz_const)
        self.activation = activation
        self.zero_init = zero_init

        act = get_activation(activation)
        layers: list[nn.Module] = []
        dims = [in_channels] + [hidden] * n_layers
        n_linear = n_layers + 1
        # Per-layer spectral cap gamma such that  s * gamma**n_linear = lipschitz_const.
        gamma = (float(lipschitz_const) / max(float(deformation_scale), 1e-12)) ** (1.0 / n_linear)
        for a, b in zip(dims[:-1], dims[1:]):
            if lipschitz_mode == "spectral":
                layers.append(SpectralLinear(a, b, cap=gamma))
            else:
                layers.append(nn.Linear(a, b))
            layers.append(_Activation(activation, act))
        if lipschitz_mode == "spectral":
            layers.append(SpectralLinear(hidden, 2, cap=gamma))
        else:
            layers.append(nn.Linear(hidden, 2))
        self.net = nn.Sequential(*layers)
        self.layer_cap = gamma if lipschitz_mode == "spectral" else None

        # P1: phi = Id at t = 0 (and det J = 1).
        if zero_init:
            with torch.no_grad():
                nn.init.zeros_(self.net[-1].weight)
                if self.net[-1].bias is not None:
                    nn.init.zeros_(self.net[-1].bias)

        # Residual scale s.  In spectral mode the per-layer caps guarantee
        # Lip(r) <= s * gamma**n_linear = lipschitz_const.
        self.register_buffer("_scale", torch.tensor(float(deformation_scale)))
        self.deformation_scale = float(deformation_scale)

    # ------------------------------------------------------------------ #
    @property
    def scale(self) -> float:
        return float(self._scale.item())

    def lipschitz_bound(self, n_power_iterations: int = 10) -> float:
        """Upper bound on :math:`\\mathrm{Lip}(r)` (rigorous in spectral mode).

        :math:`\\mathrm{Lip}(r) \\le s \\prod_i \\sigma(W_i) \\prod_j L_{\\sigma_j}`.
        In ``"none"`` mode the same quantity is returned, but it is only an
        estimate of the local behaviour (it is still the correct global bound
        for the *linear* part of the network; it just is not enforced).
        """
        lip = 1.0
        for m in self.net.modules():
            if isinstance(m, SpectralLinear):
                lip *= float(m.sigma(n_power_iterations).item())
            elif isinstance(m, nn.Linear):
                lip *= float(spectral_norm_matrix(m.weight, n_power_iterations).item())
            else:
                lip *= activation_lipschitz(m)
        return self.scale * lip

    def certified_det_lower_bound(self, kappa: float = 1.0,
                                  n_power_iterations: int = 10) -> float:
        """Provable lower bound on :math:`\\det J_\\phi`, :math:`(1 - \\kappa L)^2_+`.

        ``kappa`` (:math:`\\kappa`) is the spectral norm of the Jacobian of the
        *inputs* of the network with respect to :math:`\\xi`:

        * :math:`\\kappa = 1` for global conditioning (the latent code does not
          depend on :math:`\\xi`);
        * :math:`\\kappa = \\sqrt{1 + \\max_\\xi \\|\\nabla a(\\xi)\\|^2}` for
          pointwise conditioning on the coefficient field.

        The bound is rigorous for both Lipschitz modes -- spectral normalisation
        is what makes it *useful* (i.e. positive), not what makes it *true*.
        """
        return float(certified_det_lower_bound(self.lipschitz_bound(n_power_iterations), kappa))

    @property
    def is_certified(self) -> bool:
        """``True`` when the folding-free property is enforced by construction.

        In ``"none"`` mode :math:`L` is free to grow past :math:`1/\\kappa`
        during training, so ``det J > 0`` is only encouraged by the barrier and
        must be verified *a posteriori*.
        """
        return self.lipschitz_mode == "spectral" and self.lipschitz_const < 1.0

    # ------------------------------------------------------------------ #
    def forward(self, coords: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        """Return :math:`\\phi(\\xi)`; ``coords`` is ``(..., H, W, 2)``.

        ``condition`` may be a pointwise field ``(..., H, W, k)`` or a global
        code ``(..., k)`` that is broadcast over the grid.
        """
        if coords.shape[-1] != 2:
            raise ValueError(f"coords must have last dim 2, got {coords.shape[-1]}")
        if condition is None:
            inp = coords
        elif condition.shape[-1] == self.in_channels - 2 and condition.dim() == coords.dim():
            inp = torch.cat([coords, condition], dim=-1)
        elif condition.dim() == 2 and condition.shape[-1] == self.in_channels - 2:
            # (B, k) global code -> (B, 1, 1, k) then broadcast over the grid.
            k = condition.shape[-1]
            code = condition.reshape(condition.shape[0], *([1] * (coords.dim() - 2)), k)
            code = code.expand(*coords.shape[:-1], k)
            inp = torch.cat([coords, code], dim=-1)
        else:
            raise ValueError(f"cannot reconcile condition {tuple(condition.shape)} with coords {tuple(coords.shape)}")
        if inp.shape[-1] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} input channels, got {inp.shape[-1]}")
        return coords + self._scale * self.net(inp)


class _Activation(nn.Module):
    """Thin wrapper so that the trunk exposes the Lipschitz constant of its activation."""

    def __init__(self, name: str, fn) -> None:
        super().__init__()
        self.name = name
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x)
