# Statistics, 6 domains, reference `fno`

Runs: 57  |  domains: annulus, double_hole, lshape, naca0012, slit, star

## `chart-eq` vs `fno`, rel_l2

- pooled: n = 18, mean = -0.0182, sd = 0.0177, t = -4.36, p(two-sided) = 0.0004, p(Wilcoxon, less) = 0.0003
- improving runs: 16/18  |  gain: +8.58 % (mean of ratios), +9.26 % (ratio of means)
- domains negative: 6/6 (sign-test p = 0.0156)

| domain | n | paired mean |
|---|---|---|
| annulus | 3 | -0.0121 |
| double_hole | 3 | -0.0337 |
| lshape | 3 | -0.0147 |
| naca0012 | 3 | -0.0132 |
| slit | 3 | -0.0314 |
| star | 3 | -0.0039 |

## `chart-eq` vs `fno`, rel_h1

- pooled: n = 18, mean = -0.0413, sd = 0.0283, t = -6.18, p(two-sided) = 0.0000, p(Wilcoxon, less) = 0.0001
- improving runs: 16/18  |  gain: +9.70 % (mean of ratios), +10.22 % (ratio of means)
- domains negative: 6/6 (sign-test p = 0.0156)

| domain | n | paired mean |
|---|---|---|
| annulus | 3 | -0.0153 |
| double_hole | 3 | -0.0680 |
| lshape | 3 | -0.0419 |
| naca0012 | 3 | -0.0466 |
| slit | 3 | -0.0626 |
| star | 3 | -0.0133 |

## `chart-fixed` vs `fno`, rel_l2

- pooled: n = 18, mean = -0.0108, sd = 0.0159, t = -2.89, p(two-sided) = 0.0102, p(Wilcoxon, less) = 0.0080
- improving runs: 11/18  |  gain: +4.95 % (mean of ratios), +5.52 % (ratio of means)
- domains negative: 6/6 (sign-test p = 0.0156)

| domain | n | paired mean |
|---|---|---|
| annulus | 3 | -0.0106 |
| double_hole | 3 | -0.0185 |
| lshape | 3 | -0.0050 |
| naca0012 | 3 | -0.0103 |
| slit | 3 | -0.0194 |
| star | 3 | -0.0012 |

## `chart-fixed` vs `fno`, rel_h1

- pooled: n = 18, mean = -0.0202, sd = 0.0205, t = -4.18, p(two-sided) = 0.0006, p(Wilcoxon, less) = 0.0005
- improving runs: 16/18  |  gain: +4.75 % (mean of ratios), +5.00 % (ratio of means)
- domains negative: 6/6 (sign-test p = 0.0156)

| domain | n | paired mean |
|---|---|---|
| annulus | 3 | -0.0120 |
| double_hole | 3 | -0.0315 |
| lshape | 3 | -0.0143 |
| naca0012 | 3 | -0.0263 |
| slit | 3 | -0.0278 |
| star | 3 | -0.0093 |

## `dif-barrier` vs `fno`, rel_l2

- pooled: n = 3, mean = -0.0034, sd = 0.0064, t = -0.91, p(two-sided) = 0.4573, p(Wilcoxon, less) = 0.3750
- improving runs: 2/3  |  gain: +1.79 % (mean of ratios), +2.07 % (ratio of means)
- domains negative: 1/1 (sign-test p = 0.5000)

| domain | n | paired mean |
|---|---|---|
| star | 3 | -0.0034 |

## `dif-barrier` vs `fno`, rel_h1

- pooled: n = 3, mean = +0.0019, sd = 0.0080, t = +0.40, p(two-sided) = 0.7273, p(Wilcoxon, less) = 0.6250
- improving runs: 2/3  |  gain: -0.60 % (mean of ratios), -0.54 % (ratio of means)
- domains negative: 0/1 (sign-test p = 1.0000)

| domain | n | paired mean |
|---|---|---|
| star | 3 | +0.0019 |

## Admissibility (worst case over all runs)

| variant | fold [%] | min det J | cert. lower bound | inverse error |
|---|---|---|---|---|
| `chart-eq` | 0.00 | 0.0326 | 0.0204 | 1.31e-06 |
| `chart-fixed` | 0.00 | 0.2735 | 0.2353 | 3.58e-07 |
| `dif-barrier` | 0.00 | 0.4624 | 0.0000 | -- |

