"""Inferential statistics for benchmark comparisons.

Neural-operator benchmarks are noisy: the dominant variance comes from the
train/validation split and from the initialisation, both driven by the seed.
Reporting ``mean_A - mean_B`` without a dispersion is therefore close to
meaningless.  Every comparison here is **paired** (same seed, same domain, same
data), which removes the domain difficulty from the comparison and keeps only
the model effect.

Two intervals are produced for every paired difference:

* ``ci_low`` / ``ci_high`` -- the **Student-t** interval of the mean difference,
  which is the interval consistent with the paired *t*-test we report;
* ``boot_ci_low`` / ``boot_ci_high`` -- a percentile bootstrap interval,
  distribution-free but *known to undercover* at small ``n`` (measured coverage
  of the nominal 95% interval: ~0.86 at ``n = 12``, ~0.92 at ``n = 25``).

With the 6 seeds of the benchmark the *t* interval is the one to quote; the
bootstrap is reported alongside it as a robustness check, and the Wilcoxon
signed-rank test -- exact and free of the normality assumption -- is the
confirmatory test at ``n >= 6``.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import comb as binom

import numpy as np

__all__ = ["mean_ci", "t_interval", "paired_test", "summarise",
           "sign_test", "format_pm", "significance_stars"]


def _clean(values: Sequence[float]) -> np.ndarray:
    """Drop ``None``/NaN/inf and return a float array."""
    return np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)


def mean_ci(values: Sequence[float], n_bootstrap: int = 10_000, alpha: float = 0.05,
            seed: int = 0) -> tuple[float, float, float]:
    """Percentile bootstrap interval: ``(mean, ci_low, ci_high)``."""
    x = _clean(values)
    if x.size == 0:
        return (float("nan"),) * 3
    if x.size == 1:
        return float(x[0]), float(x[0]), float(x[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_bootstrap, x.size))
    boots = x[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(x.mean()), float(lo), float(hi)


def t_interval(values: Sequence[float], alpha: float = 0.05) -> tuple[float, float, float]:
    """Student-t interval: ``(mean, ci_low, ci_high)``.

    Exact for Gaussian differences (the usual assumption for a mean of errors)
    and far better behaved than the percentile bootstrap at the sample sizes of
    a benchmark sweep (``n`` = number of seeds).
    """
    x = _clean(values)
    if x.size == 0:
        return (float("nan"),) * 3
    if x.size == 1:
        return float(x[0]), float(x[0]), float(x[0])
    sem = x.std(ddof=1) / np.sqrt(x.size)
    if sem == 0.0:
        return float(x.mean()), float(x.mean()), float(x.mean())
    try:
        from scipy import stats

        half = float(stats.t.ppf(1 - alpha / 2, df=x.size - 1)) * sem
    except Exception:                                   # pragma: no cover - scipy is a dependency
        half = 1.96 * sem
    return float(x.mean()), float(x.mean() - half), float(x.mean() + half)


def paired_test(a: Sequence[float], b: Sequence[float], alpha: float = 0.05) -> dict:
    """Paired comparison of two measurement vectors (``a`` vs ``b``).

    Returns the mean difference ``a - b`` (negative = ``a`` better when the
    metric is an error), its Student-t and bootstrap intervals, the paired
    *t*-test p-value and the Wilcoxon signed-rank p-value (the latter only at
    ``n >= 6``, where it is the confirmatory test).
    """
    # Pairing is the whole point: a pair with a missing member is dropped, not
    # shifted (dropping the holes first and then truncating would compare
    # seed i of ``a`` against seed j of ``b``).
    pairs = [(float(u), float(v)) for u, v in zip(a, b)
             if u is not None and v is not None and np.isfinite(u) and np.isfinite(v)]
    x = np.array([u for u, _ in pairs], dtype=float)
    y = np.array([v for _, v in pairs], dtype=float)
    n = int(x.size)

    out = {"n": n, "mean_diff": float("nan"), "std_diff": float("nan"),
           "ci_low": float("nan"), "ci_high": float("nan"),
           "boot_ci_low": float("nan"), "boot_ci_high": float("nan"),
           "t_pvalue": float("nan"), "wilcoxon_pvalue": float("nan")}
    if n == 0:
        return out

    d = x - y
    mean, lo, hi = t_interval(d, alpha=alpha)
    _, blo, bhi = mean_ci(d)
    out.update({"mean_diff": float(d.mean()),
                "std_diff": float(d.std(ddof=1)) if n > 1 else 0.0,
                "ci_low": lo, "ci_high": hi,
                "boot_ci_low": blo, "boot_ci_high": bhi})

    degenerate = n > 1 and np.allclose(d, d[0])
    if degenerate:
        # Zero-variance differences: the t statistic diverges.  The signed-rank
        # test still has an exact answer, computed below.
        out["t_pvalue"] = 0.0 if d[0] != 0 else 1.0
    try:
        from scipy import stats

        if n > 1 and not degenerate:
            out["t_pvalue"] = float(stats.ttest_rel(x, y).pvalue)
        if n >= 6:
            out["wilcoxon_pvalue"] = float(stats.wilcoxon(x, y).pvalue)
    except Exception:                                   # pragma: no cover
        pass
    return out


def summarise(values: Sequence[float], n_bootstrap: int = 5_000) -> dict:
    """Mean, s.d., standard error, Student-t and bootstrap CIs, extremes."""
    x = _clean(values)
    if x.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "sem": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan"),
                "boot_ci_low": float("nan"), "boot_ci_high": float("nan"),
                "min": float("nan"), "max": float("nan")}
    mean, lo, hi = t_interval(x)
    _, blo, bhi = mean_ci(x, n_bootstrap=n_bootstrap)
    return {"n": int(x.size), "mean": float(mean),
            "std": float(x.std(ddof=1)) if x.size > 1 else 0.0,
            "sem": float(x.std(ddof=1) / np.sqrt(x.size)) if x.size > 1 else 0.0,
            "ci_low": float(lo), "ci_high": float(hi),
            "boot_ci_low": float(blo), "boot_ci_high": float(bhi),
            "min": float(x.min()), "max": float(x.max())}


def sign_test(k: int, n: int) -> float:
    """Exact one-sided sign test: ``P(X >= k)`` for ``X ~ Binomial(n, 1/2)``.

    This is the test for the *nested* case the paired tests cannot see: when the
    runs are grouped by geometry (or by PDE), the independent units are the
    groups, not the runs, and ``n`` above counts groups.  With six geometries all
    improving, the exact p-value is ``2**-6 = 0.0156`` -- which is the honest
    resolution of a comparison whose runs share their geometry.

    Returns ``0.0`` when ``k > n`` and ``1.0`` when ``k <= 0``.
    """
    k, n = int(k), int(n)
    if n < 1:
        return float("nan")
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return sum(binom(n, i) for i in range(k, n + 1)) / 2.0 ** n


def format_pm(mean: float, half_width: float | None = None, digits: int = 4,
              plus: bool = False) -> str:
    """``0.1234 ± 0.0056`` (or ``+0.0012 ± 0.0034``)."""
    if mean is None or not np.isfinite(mean):
        return "--"
    if half_width is None or not np.isfinite(half_width):
        return f"{mean:{'+' if plus else ''}.{digits}f}"
    return f"{mean:{'+' if plus else ''}.{digits}f} ± {half_width:.{digits}f}"


def significance_stars(p: float) -> str:
    """``***`` p<0.001, ``**`` p<0.01, ``*`` p<0.05, ``n.s.`` otherwise."""
    if p is None or not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."
