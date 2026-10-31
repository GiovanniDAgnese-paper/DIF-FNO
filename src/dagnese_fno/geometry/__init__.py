"""Discrete differential geometry of the learned coordinate maps."""

from .jacobian import (
    certified_det_lower_bound,
    det_jacobian_stats,
    displacement_stats,
    folding_mask,
    folding_percentage,
    jacobian_components,
    jacobian_det,
    jacobian_det_autograd,
    jacobian_singular_values,
    spectral_norm,
    spectral_norm_matrix,
)
from .monotone import DagneseTriangularChart, chart_det_consistency, invert_monotone_1d
from .transport import (
    convergence_rate,
    jacobian_weight,
    pullback,
    pushforward,
    sample_field,
)

__all__ = [
    "certified_det_lower_bound",
    "det_jacobian_stats",
    "displacement_stats",
    "folding_mask",
    "folding_percentage",
    "jacobian_components",
    "jacobian_det",
    "jacobian_det_autograd",
    "jacobian_singular_values",
    "spectral_norm",
    "spectral_norm_matrix",
    "DagneseTriangularChart",
    "invert_monotone_1d",
    "chart_det_consistency",
    "sample_field",
    "pullback",
    "pushforward",
    "jacobian_weight",
    "convergence_rate",
]
