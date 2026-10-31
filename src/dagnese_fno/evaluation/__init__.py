"""Benchmark analysis: paired statistics and significance testing.

Neural-operator sweeps are noisy and the interesting effects are small, so a
table of bare means is not evidence.  This sub-package provides the paired
machinery used by ``scripts/run_benchmark.py``:

* :func:`~dagnese_fno.evaluation.statistics.summarise` -- mean, s.d., s.e.m. and
  two confidence intervals for one cell of the table;
* :func:`~dagnese_fno.evaluation.statistics.paired_test` -- the paired
  difference against a reference variant, with the paired *t*-test, the
  Wilcoxon signed-rank test and both intervals;
* :func:`~dagnese_fno.evaluation.statistics.t_interval` /
  :func:`~dagnese_fno.evaluation.statistics.mean_ci` -- the two intervals
  themselves (Student-t, the one we quote, and percentile bootstrap);
* :func:`~dagnese_fno.evaluation.statistics.sign_test` -- the exact one-sided
  sign test over *groups* (geometries), for the case where the runs are nested;
* :func:`~dagnese_fno.evaluation.statistics.significance_stars` and
  :func:`~dagnese_fno.evaluation.statistics.format_pm` -- rendering.
"""

from .statistics import (
    format_pm,
    mean_ci,
    paired_test,
    sign_test,
    significance_stars,
    summarise,
    t_interval,
)

__all__ = [
    "paired_test",
    "mean_ci",
    "t_interval",
    "summarise",
    "sign_test",
    "format_pm",
    "significance_stars",
]
