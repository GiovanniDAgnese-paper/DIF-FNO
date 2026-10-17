"""Fourier (spectral) convolution layer for the FNO backbone."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

__all__ = ["SpectralConv2d"]


class SpectralConv2d(nn.Module):
    """2-D spectral convolution: learn a multiplier on the lowest Fourier modes.

    The layer applies

    .. math::
        (\\mathcal K v)(x) = \\mathcal F^{-1}\\big(R_\\theta \\cdot \\mathcal F v\\big)

    where :math:`R_\\theta` acts only on the ``modes1 x modes2`` lowest modes.
    Real inputs are handled with ``rfft2``/``irfft2``; the number of retained
    modes is clamped to the Nyquist limit at runtime, so the same module can be
    evaluated on grids of different resolutions.

    Parameters
    ----------
    in_channels, out_channels:
        Channel dimensions.
    modes1, modes2:
        Number of Fourier modes kept along each axis (clamped at runtime).
    init:
        ``"v4"`` reproduces the initialisation used in the published DIF-FNO v4
        runs (real part :math:`\\sim U(0, 1) / (c_{in} c_{out})`, zero imaginary
        part); ``"normal"`` uses a complex normal initialisation with matched
        variance, which is the recommended default for new work.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        init: str = "v4",
    ) -> None:
        super().__init__()
        if modes1 < 1 or modes2 < 1:
            raise ValueError("number of modes must be >= 1")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)
        if init not in ("v4", "normal"):
            raise ValueError(f"unknown init '{init}'")

        scale = 1.0 / (in_channels * out_channels)
        shape = (in_channels, out_channels, self.modes1, self.modes2)
        if init == "v4":
            w1 = scale * torch.rand(*shape, dtype=torch.cfloat)
            w2 = scale * torch.rand(*shape, dtype=torch.cfloat)
        else:
            # Complex normal with variance `scale` per real/imag component.
            std = math.sqrt(0.5 * scale)
            w1 = torch.randn(*shape, dtype=torch.cfloat) * std
            w2 = torch.randn(*shape, dtype=torch.cfloat) * std
        self.weight1 = nn.Parameter(w1)
        self.weight2 = nn.Parameter(w2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """:math:`x`: ``(B, C_in, H, W)`` -> ``(B, C_out, H, W)``."""
        if x.dim() != 4:
            raise ValueError(f"expected (B, C, H, W), got {tuple(x.shape)}")
        dtype = torch.promote_types(x.dtype, torch.float32)
        if x.dtype.is_complex:
            xf = torch.fft.fft2(x)
            mode = "full"
        else:
            xf = torch.fft.rfft2(x.to(dtype))
            mode = "half"
        n_modes_h = xf.size(-2)
        n_modes_w = xf.size(-1)
        m1 = min(self.modes1, n_modes_h)
        m2 = min(self.modes2, n_modes_w)

        out = torch.zeros(
            xf.size(0), self.out_channels, n_modes_h, n_modes_w,
            dtype=xf.dtype, device=x.device,
        )
        w1 = self.weight1.to(xf.dtype)[..., :m1, :m2]
        w2 = self.weight2.to(xf.dtype)[..., :m1, :m2]
        out[:, :, :m1, :m2] = torch.einsum("bixy,ioxy->boxy", xf[:, :, :m1, :m2], w1)
        out[:, :, -m1:, :m2] = torch.einsum("bixy,ioxy->boxy", xf[:, :, -m1:, :m2], w2)

        if mode == "half":
            return torch.fft.irfft2(out, s=(x.size(-2), x.size(-1))).to(x.dtype)
        return torch.fft.ifft2(out, s=(x.size(-2), x.size(-1))).to(x.dtype)

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (f"in={self.in_channels}, out={self.out_channels}, "
                f"modes=({self.modes1}, {self.modes2})")
