"""Backwards-compatible shim -- use :mod:`dagnese_fno.losses.barrier` instead.

The canonical implementation now lives in
:mod:`dagnese_fno.losses.barrier` and is re-exported here so that
``from dagnese_fno.barrier_loss import DAgneseBarrierLoss`` (as used in
DIF-FNO v4 and in the paper's companion code) keeps working.
"""

from __future__ import annotations

import warnings

from .losses.barrier import DAgneseBarrierLoss

warnings.warn(
    "dagnese_fno.barrier_loss is deprecated; import "
    "dagnese_fno.losses.barrier (or dagnese_fno.DAgneseBarrierLoss) instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Historical alias kept for the v4 scripts.
DAgneseBarrierLossV4 = DAgneseBarrierLoss

__all__ = ["DAgneseBarrierLoss", "DAgneseBarrierLossV4"]
