"""Pullback and pushforward along a chart: the operations that make a chart act
on a *field* rather than on the coordinate channels.

DIF-FNO v1 fed :math:`\\phi(\\xi)` to the network as two extra input channels and
never transported the field, so the chart could not change the discretisation
of anything: it was a reparameterisation of the input, not a change of
variables.  The operators below implement the change of variables itself,

.. math::
    (\\mathcal P_\\phi a)(\\xi) = a\\big(\\phi(\\xi)\\big)\\,
    \\big|\\det J_\\phi(\\xi)\\big|^\\alpha ,
    \\qquad
    (\\mathcal P_{\\phi^{-1}} \\tilde u)(x) = \\tilde u\\big(\\phi^{-1}(x)\\big),

where :math:`\\alpha = 0` for a scalar field (a Darcy coefficient) and
:math:`\\alpha = 1` for a density (a conserved quantity; the Jacobian factor is
then the Piola transform).

Proposition P6 (D'Agnese, exactness)
------------------------------------
``pullback`` then ``pushforward`` is the identity on the *nodally sampled*
field up to the interpolation error of the bilinear interpolant,
:math:`\\|\\mathcal P_{\\phi^{-1}}\\mathcal P_\\phi f - f\\|_\\infty =
O(h^2\\,\\|D^2 f\\|_\\infty)`: a change of variables does not lose information,
it only re-discretises it.  ``tests/test_transport.py`` measures the rate.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["sample_field", "pullback", "pushforward", "jacobian_weight", "convergence_rate"]


def sample_field(field: torch.Tensor, points: torch.Tensor, mode: str = "bilinear",
                 padding_mode: str = "border") -> torch.Tensor:
    """Evaluate a regular field at arbitrary points (bilinear by default).

    ``field``  : ``(..., N, M)``
    ``points`` : ``(..., P, Q, 2)`` in :math:`[-1,1]^2`, ``'ij'`` convention
                 (component 0 <-> tensor axis -2)

    Returns ``(..., P, Q)``.  Leading batch dimensions are broadcast.

    ``grid_sample`` expects ``(x, y)`` ordering, i.e. the reverse of the library
    convention, hence the ``flip``: getting it wrong transposes the geometry,
    so it is tested explicitly.
    """
    if points.shape[-1] != 2:
        raise ValueError(f"points must have last dimension 2, got {points.shape[-1]}")
    f_lead, p_lead = field.shape[:-2], points.shape[:-3]
    lead = torch.broadcast_shapes(f_lead, p_lead)
    n, m = field.shape[-2:]
    ph, pw = points.shape[-3], points.shape[-2]

    f = field.reshape(*f_lead, 1, n, m).expand(*lead, 1, n, m)
    p = points.reshape(*p_lead, ph, pw, 2).expand(*lead, ph, pw, 2)
    b = 1
    for d in lead:
        b *= int(d)
    out = F.grid_sample(
        f.reshape(b, 1, n, m).contiguous(),
        p.reshape(b, ph, pw, 2).flip(-1).contiguous(),
        mode=mode, padding_mode=padding_mode, align_corners=True,
    )
    return out.reshape(*lead, ph, pw)


def jacobian_weight(det_j: torch.Tensor, alpha: float, eps: float = 1e-12) -> torch.Tensor:
    """``|det J|^alpha``: 0 for scalars, 1 for densities (Piola), any power for interpolation."""
    if alpha == 0.0:
        return torch.ones_like(det_j)
    return det_j.abs().clamp_min(eps).pow(float(alpha))


def pullback(field: torch.Tensor, phi: torch.Tensor, det_j: torch.Tensor | None = None,
             alpha: float = 0.0) -> torch.Tensor:
    """Transport ``field`` to the chart: ``(P_phi f)(xi) = f(phi(xi)) |det J|^alpha``."""
    out = sample_field(field, phi)
    if det_j is not None:
        out = out * jacobian_weight(det_j, alpha)
    return out


def pushforward(field_chart: torch.Tensor, phi_inverse: torch.Tensor,
                det_j: torch.Tensor | None = None, alpha: float = 0.0) -> torch.Tensor:
    r"""Transport a chart field back to the physical grid.

    .. math::
        (\mathcal P_{\phi^{-1}} \tilde f)(x)
        = \tilde f\big(\phi^{-1}(x)\big)\,\big|\det J_{\phi}(\phi^{-1}(x))\big|^{-\alpha},

    where ``det_j`` is :math:`|\det J_\phi|` *on the reference grid*; it is
    sampled at :math:`\phi^{-1}(x)` before the power is taken, because the change
    of variables for the inverse map uses the determinant of the forward map
    **at the point that maps to** :math:`x`, not at the reference node.  (Using
    the reference-grid determinant directly is a common and silent error: it is
    wrong by the variation of the determinant along the chart.)  With
    ``alpha = 1`` the composition ``pushforward(pullback(f))`` therefore returns
    ``f``, i.e. the pair is an inverse pair for densities as well as for scalars.
    """
    out = sample_field(field_chart, phi_inverse)
    if det_j is not None and alpha != 0.0:
        det_at = sample_field(det_j, phi_inverse)
        out = out * jacobian_weight(det_at, -float(alpha))
    return out


def convergence_rate(errors: list[float], resolutions: list[int]) -> float:
    """Observed order of convergence from a sequence of errors (log-log least squares)."""
    if len(errors) != len(resolutions) or len(errors) < 2:
        raise ValueError("need at least two (error, resolution) pairs of equal length")
    x = torch.log(torch.tensor([1.0 / r for r in resolutions], dtype=torch.float64))
    y = torch.log(torch.tensor(errors, dtype=torch.float64))
    x = x - x.mean()
    y = y - y.mean()
    return float((x * y).sum() / (x * x).sum())
