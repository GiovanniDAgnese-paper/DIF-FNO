"""Fourier Neural Operator backbone used by both the baseline and DIF-FNO.

The baseline and DIF-FNO share the *same* backbone (width, modes, depth,
initialisation), so the only difference between the two columns of the
benchmark table is the diffeomorphic branch that produces the computational
coordinates.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .activations import get_activation
from .spectral import SpectralConv2d

__all__ = ["FNO2d"]


class FNO2d(nn.Module):
    """Standard 2-D FNO acting on fields sampled on a structured grid.

    Parameters
    ----------
    in_channels:
        Number of input channels, e.g. ``3`` for :math:`(\\xi_1, \\xi_2, a)`.
    out_channels:
        Number of output fields (``1`` for Darcy).
    width:
        Lifting width.
    modes1, modes2:
        Fourier modes retained per spectral layer.
    n_layers:
        Number of Fourier layers (``2`` reproduces DIF-FNO v4).
    fc_channels:
        Width of the projection MLP head.
    activation:
        Name of the activation used after every Fourier layer.
    init:
        Spectral-weight initialisation (see :class:`~dagnese_fno.models.spectral.SpectralConv2d`).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        width: int = 24,
        modes1: int = 12,
        modes2: int = 12,
        n_layers: int = 2,
        fc_channels: int = 64,
        activation: str = "gelu",
        init: str = "v4",
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.n_layers = n_layers

        act = get_activation(activation)
        self.activation = activation

        self.fc0 = nn.Linear(in_channels, width)
        self.spectral = nn.ModuleList(
            SpectralConv2d(width, width, modes1, modes2, init=init) for _ in range(n_layers)
        )
        self.local = nn.ModuleList(nn.Conv2d(width, width, 1) for _ in range(n_layers))
        self.fc1 = nn.Linear(width, fc_channels)
        self.fc2 = nn.Linear(fc_channels, out_channels)
        self._act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """:math:`x`: ``(B, H, W, C_in)`` -> ``(B, H, W, C_out)``."""
        if x.shape[-1] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} channels, got {x.shape[-1]}")
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        for spec, loc in zip(self.spectral, self.local):
            x = self._act(spec(x) + loc(x))
        x = x.permute(0, 2, 3, 1)
        x = self._act(self.fc1(x))
        return self.fc2(x).squeeze(-1) if self.out_channels == 1 else self.fc2(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
