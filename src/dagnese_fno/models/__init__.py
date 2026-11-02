"""Neural-operator models: FNO baseline, diffeomorphic branch, DIF-FNO."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from .chart_operator import DagneseChartOperator
from .dif_fno import ConditionEncoder, DIFFNO2d
from .diffeo import DiffeomorphicMap, SpectralLinear
from .fno import FNO2d
from .spectral import SpectralConv2d

__all__ = [
    "FNO2d",
    "DagneseChartOperator",
    "DIFFNO2d",
    "DiffeomorphicMap",
    "SpectralLinear",
    "SpectralConv2d",
    "ConditionEncoder",
    "build_model",
]

MODEL_REGISTRY = {
    "fno": FNO2d,
    "dif": DIFFNO2d,
    "dif-fno": DIFFNO2d,
    "dif_fno": DIFFNO2d,
    "chart": DagneseChartOperator,
    "dao": DagneseChartOperator,
    "chart-fno": DagneseChartOperator,
    "chart_fno": DagneseChartOperator,
    "cpo": DagneseChartOperator,
}


def build_model(kind: str, **kwargs: Any) -> nn.Module:
    """Factory used by the training scripts.

    ``kind`` is one of ``{"fno", "dif"}``; unknown keyword arguments that do not
    apply to the selected model are ignored, so the same config dictionary can
    be used for the baseline and for DIF-FNO (this is what makes the comparison
    fair: the backbone hyperparameters are literally the same object).
    """
    key = str(kind).lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"unknown model '{kind}'; available: {sorted(MODEL_REGISTRY)}")
    cls = MODEL_REGISTRY[key]

    if cls is DagneseChartOperator:
        allowed = {"width", "modes1", "modes2", "n_layers", "fc_channels", "out_channels",
                   "chart_hidden", "chart_layers", "chart_fourier", "chart_activation",
                   "chart_conditioning", "latent_dim", "alpha", "chart_min_increment",
                   "activation", "init"}
        kwargs = {k: v for k, v in kwargs.items() if k in allowed}
    elif cls is FNO2d:
        allowed = {"in_channels", "out_channels", "width", "modes1", "modes2",
                   "n_layers", "fc_channels", "activation", "init"}
        kwargs = {k: v for k, v in kwargs.items() if k in allowed}
        kwargs.setdefault("in_channels", 3)
    else:
        allowed = {"width", "modes1", "modes2", "n_layers", "fc_channels", "out_channels",
                   "diffeo_hidden", "diffeo_layers", "deformation_scale", "lipschitz_mode",
                   "lipschitz_const", "conditioning", "latent_dim", "activation",
                   "diffeo_activation", "init"}
        kwargs = {k: v for k, v in kwargs.items() if k in allowed}
    return cls(**kwargs)
