"""
D'Agnese DIF-FNO -- Diffeomorphic Implicit Fourier Neural Operators.

Official implementation of the **D'Agnese Topological Barrier Loss**
:math:`\\mathcal{L}_{\\text{barrier}}` and of the DIF-FNO architecture, which
learns an input-conditioned diffeomorphism

.. math:: \\phi_\\psi : \\Omega_{\\mathrm{ref}} \\to \\Omega_{\\mathrm{comp}},
          \\qquad \\det J_{\\phi} > 0,

before the Fourier Neural Operator backbone, so that the operator is evaluated
on a folded-free computational grid.

Quick start
-----------
>>> import torch
>>> from dagnese_fno import DIFFNO2d, DAgneseBarrierLoss, jacobian_det
>>> coords = torch.stack(torch.meshgrid(torch.linspace(0, 1, 32),
...                                     torch.linspace(0, 1, 32), indexing="ij"), -1)[None]
>>> a = torch.rand(1, 32, 32)
>>> model = DIFFNO2d()
>>> u, xi = model(coords, a)                      # u: (1, 32, 32), xi: (1, 32, 32, 2)
>>> det = jacobian_det(xi, spacing=1 / 31)        # == 1 at initialisation
>>> loss = DAgneseBarrierLoss(tau=0.05)(det)
"""

from __future__ import annotations

from . import data, evaluation, geometry, losses, metrics, models, training, utils
from .geometry import (
    DagneseTriangularChart,
    certified_det_lower_bound,
    det_jacobian_stats,
    displacement_stats,
    folding_percentage,
    jacobian_components,
    jacobian_det,
    jacobian_det_autograd,
    pullback,
    pushforward,
    sample_field,
)
from .losses import (
    DAgneseBarrierLoss,
    DagneseEquidistributionLoss,
    DiffeomorphicOperatorLoss,
    HingeFoldingPenalty,
    LogBarrierLoss,
    dagnese_barrier,
    get_folding_penalty,
)
from .metrics import relative_H1, relative_h1, relative_l2
from .models import DagneseChartOperator, DiffeomorphicMap, DIFFNO2d, FNO2d, build_model
from .training import TrainConfig, train_one
from .utils import describe_environment, provenance_block, set_seed

__version__ = "1.1.0"
__author__ = "Giovanni D'Agnese"
__email__ = "jovannidagnese2@gmail.com"

__all__ = [
    # sub-packages
    "data", "evaluation", "geometry", "losses", "metrics", "models",
    "training", "utils",
    # models
    "FNO2d", "DIFFNO2d", "DagneseChartOperator", "DiffeomorphicMap", "build_model",
    # losses
    "DAgneseBarrierLoss", "DagneseEquidistributionLoss",
    "LogBarrierLoss", "HingeFoldingPenalty",
    "DiffeomorphicOperatorLoss", "dagnese_barrier", "get_folding_penalty",
    # geometry
    "jacobian_det", "jacobian_components", "jacobian_det_autograd",
    "folding_percentage", "det_jacobian_stats", "displacement_stats",
    "certified_det_lower_bound",
    "DagneseTriangularChart", "sample_field", "pullback", "pushforward",
    # metrics
    "relative_l2", "relative_h1", "relative_H1",
    # training
    "TrainConfig", "train_one",
    # utils
    "set_seed", "describe_environment", "provenance_block",
]
