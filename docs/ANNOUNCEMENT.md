# Announcement pack

Use these summaries for the canonical release. The numerical sources are
[`results/v2/STATS.md`](../results/v2/STATS.md),
[`results/study/STUDY.md`](../results/study/STUDY.md) and the separate historical
[v1 benchmark](../results/repro/table.md).

## One line

> Giovanni D'Agnese's DIF-FNO 1.2.0 implements a Fourier neural operator with a
> chart that cannot fold by construction; the paired Darcy benchmark did not
> detect a statistically significant accuracy difference against FNO.

## Short abstract

The **D'Agnese Chart Operator** evaluates a Fourier neural operator on an
adaptively learned triangular chart of the reference square. The chart is
piecewise bilinear, globally injective, and has a positive Jacobian determinant
on cell interiors and a closed-form inverse. At initialisation the operator
recovers its FNO backbone up to floating-point round-off. The optional
**D'Agnese Equidistribution Loss** penalises uneven monitor mass across cells;
it minimises an upper bound on representation error when equidistribution is
attained, not the PDE prediction error.

On six Darcy geometries and three seeds per configuration, `chart-eq` versus
FNO had a mean paired difference in relative L2 of **−0.0004**, with a 95 %
confidence interval **[−0.0179, +0.0172]**. This is not evidence of superiority
or equivalence. All **36 chart runs** (54 runs including the FNO baselines)
reported zero folded cells. In a separate deterministic representation study,
the adaptive mesh helped on a boundary layer and a ridge, but worsened the
error on the tested smooth field.

**Software, manuscript sources and reproduction materials:**
[10.5281/zenodo.24406366](https://doi.org/10.5281/zenodo.24406366).

## When discussing the old barrier

In the historical benchmark, the variant **without** the barrier (`dif-nobar`)
folded about 12–22 % of cells across three geometries. The softplus-barrier
variant had no observed folding in that test but was **not** structurally
certified. Do not attribute the current chart's proof to the old loss, or use
those historical runs as if they shared the current data pipeline.

## Reproduce or extend

```bash
make stats paper-tables   # reconstruct statistics, tables and figure from archived runs
make study                # deterministic verification, no PDE training
make timing               # measured ≈2.8× training-step wall time at 64² on CPU
python -m pytest          # test suite
```

See [`paper/README.md`](../paper/README.md) for manuscript build instructions.
The software DOI is not a DOI for a peer-reviewed article.
