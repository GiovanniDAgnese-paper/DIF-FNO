"""Inferential statistics: bootstrap CIs, paired tests, formatting.

These are the numbers that decide whether a difference in the benchmark tables
means anything, so they are tested as carefully as the model code: against
closed-form values where possible, and against their own qualitative behaviour
(a real effect must be detected, noise must not) otherwise.
"""

from __future__ import annotations

import numpy as np
import pytest

from dagnese_fno.evaluation import (
    format_pm,
    mean_ci,
    paired_test,
    significance_stars,
    summarise,
    t_interval,
)


# --------------------------------------------------------------------------- #
# mean_ci
# --------------------------------------------------------------------------- #
def test_mean_ci_recovers_the_mean():
    x = [1.0, 2.0, 3.0, 4.0]
    mean, lo, hi = mean_ci(x, n_bootstrap=2000)
    assert mean == pytest.approx(2.5)
    assert lo <= mean <= hi


def test_mean_ci_shrinks_with_the_sample_size():
    rng = np.random.default_rng(0)
    small = rng.normal(0, 1, 10)
    large = rng.normal(0, 1, 400)
    _, lo_s, hi_s = mean_ci(small, n_bootstrap=2000)
    _, lo_l, hi_l = mean_ci(large, n_bootstrap=2000)
    assert (hi_l - lo_l) < (hi_s - lo_s), "more data must give a tighter interval"


def test_mean_ci_degenerate_inputs():
    assert all(np.isnan(v) for v in mean_ci([]))
    assert all(np.isnan(v) for v in mean_ci([None, None]))
    assert all(np.isnan(v) for v in mean_ci([float("nan"), float("inf")]))
    assert mean_ci([0.5]) == (0.5, 0.5, 0.5)


def test_mean_ci_is_deterministic():
    x = list(np.linspace(0, 1, 17))
    assert mean_ci(x, n_bootstrap=500) == mean_ci(x, n_bootstrap=500)


def _coverage(interval, n, trials=300):
    rng = np.random.default_rng(7)
    covered = 0
    for _k in range(trials):
        x = rng.normal(0.0, 1.0, n)
        _, lo, hi = interval(x)
        covered += lo <= 0.0 <= hi
    return covered / trials


def test_t_interval_is_well_calibrated():
    """The interval we quote must actually be a 95% interval."""
    for n in (6, 12, 25):
        cov = _coverage(lambda x: t_interval(x), n)
        assert cov == pytest.approx(0.95, abs=0.04), f"n={n}: coverage {cov:.3f}"


def test_percentile_bootstrap_undercovers_at_small_n():
    """Documented limitation: the bootstrap interval is too narrow at n ~ 6-25.

    This is *why* the tables quote the Student-t interval and not this one.
    """
    cov_small = _coverage(lambda x: mean_ci(x, n_bootstrap=400, seed=1), 12)
    cov_large = _coverage(lambda x: mean_ci(x, n_bootstrap=400, seed=1), 25)
    assert 0.80 < cov_small < 0.93
    assert cov_small < cov_large < 0.99          # it improves with n


# --------------------------------------------------------------------------- #
# summarise
# --------------------------------------------------------------------------- #
def test_summarise_reports_both_intervals():
    rng = np.random.default_rng(2)
    s = summarise(rng.normal(0.1, 0.02, 8), n_bootstrap=500)
    assert s["sem"] == pytest.approx(s["std"] / np.sqrt(8), rel=1e-9)
    assert s["ci_low"] < s["mean"] < s["ci_high"]
    assert s["boot_ci_low"] < s["mean"] < s["boot_ci_high"]
    # the bootstrap is the narrower of the two at these sample sizes
    assert (s["boot_ci_high"] - s["boot_ci_low"]) < (s["ci_high"] - s["ci_low"])


def test_summarise_drops_missing_values_and_reports_extremes():
    s = summarise([1.0, None, 3.0, float("nan"), 5.0], n_bootstrap=500)
    assert s["n"] == 3
    assert s["mean"] == pytest.approx(3.0)
    assert s["std"] == pytest.approx(2.0)          # sample s.d. of 1, 3, 5
    assert s["min"] == pytest.approx(1.0)
    assert s["max"] == pytest.approx(5.0)
    assert s["ci_low"] <= s["mean"] <= s["ci_high"]


def test_summarise_empty():
    s = summarise([])
    assert s["n"] == 0 and np.isnan(s["mean"])


def test_summarise_single_value_has_zero_dispersion():
    s = summarise([0.25])
    assert s["n"] == 1 and s["mean"] == pytest.approx(0.25) and s["std"] == 0.0


# --------------------------------------------------------------------------- #
# paired_test
# --------------------------------------------------------------------------- #
def test_paired_test_detects_a_constant_shift():
    b = [0.10, 0.12, 0.11, 0.13, 0.09, 0.12]
    a = [x - 0.01 for x in b]                       # a is uniformly better
    out = paired_test(a, b)
    assert out["n"] == 6
    assert out["mean_diff"] == pytest.approx(-0.01)
    assert out["std_diff"] == pytest.approx(0.0, abs=1e-12)
    assert out["ci_high"] < 0, "the whole interval must be below zero"
    assert out["t_pvalue"] == 0.0
    assert out["wilcoxon_pvalue"] < 0.05


def test_paired_test_finds_nothing_in_pure_noise():
    rng = np.random.default_rng(3)
    b = rng.normal(0.1, 0.01, 30)
    a = b + rng.normal(0.0, 0.01, 30)               # no systematic effect
    out = paired_test(a, b)
    assert out["ci_low"] < 0 < out["ci_high"]
    assert out["t_pvalue"] > 0.05
    assert significance_stars(out["t_pvalue"]) == "n.s."


def test_paired_test_detects_a_small_effect_with_enough_pairs():
    rng = np.random.default_rng(11)
    b = rng.normal(0.12, 0.01, 24)
    a = b - 0.008 + rng.normal(0.0, 0.004, 24)      # systematic, small
    out = paired_test(a, b)
    assert out["mean_diff"] < 0
    assert out["ci_high"] < 0
    assert out["t_pvalue"] < 0.01
    assert out["wilcoxon_pvalue"] < 0.01


def test_paired_test_is_directional():
    a, b = [1.0, 2.0, 3.0], [1.5, 2.5, 3.5]
    assert paired_test(a, b)["mean_diff"] == pytest.approx(-0.5)
    assert paired_test(b, a)["mean_diff"] == pytest.approx(+0.5)


def test_paired_test_wilcoxon_needs_six_pairs():
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [1.1, 2.1, 3.1, 4.1, 5.1]
    out = paired_test(a, b)
    assert out["n"] == 5
    assert np.isnan(out["wilcoxon_pvalue"]), "too few pairs for the signed-rank test"
    assert out["t_pvalue"] < 0.05                   # the t-test still runs
    # ... and at n = 6 the exact signed-rank p-value appears (2/2**6 = 0.03125)
    out6 = paired_test(a + [6.0], b + [6.1])
    assert out6["wilcoxon_pvalue"] == pytest.approx(2 / 2 ** 6, rel=1e-3)
    assert out6["t_pvalue"] == 0.0                  # constant difference

    out_noisy = paired_test([1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                            [1.2, 1.9, 3.4, 3.7, 5.2, 6.4])
    assert 0.0 < out_noisy["t_pvalue"] < 1.0
    assert 0.0 < out_noisy["wilcoxon_pvalue"] < 1.0


def test_paired_test_keeps_the_pairs_aligned():
    """A pair with a missing member is dropped, never shifted onto another seed."""
    out = paired_test([1.0, 2.0, None, 4.0], [0.5, 1.5, 2.5])
    assert out["n"] == 2
    assert out["mean_diff"] == pytest.approx(0.5)
    assert paired_test([1.0, 2.0, 3.0], [0.5, 1.5])["n"] == 2
    assert paired_test([], [])["n"] == 0
    assert np.isnan(paired_test([], [])["mean_diff"])


# --------------------------------------------------------------------------- #
# formatting
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("p,expected", [
    (1e-6, "***"), (0.0009, "***"), (0.005, "**"), (0.04, "*"),
    (0.4, "n.s."), (float("nan"), ""), (None, ""),
])
def test_significance_stars(p, expected):
    assert significance_stars(p) == expected


def test_format_pm():
    assert format_pm(0.12345) == "0.1235"
    assert format_pm(0.12345, 0.006) == "0.1235 ± 0.0060"
    assert format_pm(-0.0012, 0.0003, plus=True) == "-0.0012 ± 0.0003"
    assert format_pm(float("nan")) == "--"
    assert format_pm(1.0, float("nan")) == "1.0000"
