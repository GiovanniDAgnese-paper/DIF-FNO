# Statistics, 6 domains, reference `fno`

Runs: 54  |  domains: annulus, double_hole, lshape, naca0012, slit, star

## `chart-eq` vs `fno`, rel_l2

- pooled: n = 18, mean = -0.0004, sd = 0.0352, t = -0.04, p(two-sided) = 0.9662, p(Wilcoxon, less) = 0.3198 n.s.
- 95 % CI of the mean difference: [-0.0179, +0.0172]
- improving runs: 10/18  |  gain: -0.61 % (mean of ratios), +0.26 % (ratio of means)
- domains negative: 3/6 (sign-test p = 0.6562)

| domain | n | paired mean |
|---|---|---|
| annulus | 3 | -0.0056 |
| double_hole | 3 | +0.0044 |
| lshape | 3 | +0.0033 |
| naca0012 | 3 | -0.0045 |
| slit | 3 | -0.0089 |
| star | 3 | +0.0092 |

## `chart-eq` vs `fno`, rel_h1

- pooled: n = 18, mean = +0.0199, sd = 0.0551, t = +1.53, p(two-sided) = 0.1435, p(Wilcoxon, less) = 0.9093 n.s.
- 95 % CI of the mean difference: [-0.0075, +0.0473]
- improving runs: 8/18  |  gain: -9.52 % (mean of ratios), -7.60 % (ratio of means)
- domains negative: 1/6 (sign-test p = 0.9844)

| domain | n | paired mean |
|---|---|---|
| annulus | 3 | +0.0047 |
| double_hole | 3 | +0.0263 |
| lshape | 3 | +0.0785 |
| naca0012 | 3 | +0.0053 |
| slit | 3 | -0.0040 |
| star | 3 | +0.0086 |

## `chart-fixed` vs `fno`, rel_l2

- pooled: n = 18, mean = -0.0047, sd = 0.0351, t = -0.57, p(two-sided) = 0.5730, p(Wilcoxon, less) = 0.0982 n.s.
- 95 % CI of the mean difference: [-0.0222, +0.0127]
- improving runs: 12/18  |  gain: +3.60 % (mean of ratios), +3.48 % (ratio of means)
- domains negative: 5/6 (sign-test p = 0.1094)

| domain | n | paired mean |
|---|---|---|
| annulus | 3 | -0.0057 |
| double_hole | 3 | -0.0079 |
| lshape | 3 | -0.0054 |
| naca0012 | 3 | -0.0100 |
| slit | 3 | -0.0083 |
| star | 3 | +0.0088 |

## `chart-fixed` vs `fno`, rel_h1

- pooled: n = 18, mean = -0.0054, sd = 0.0463, t = -0.50, p(two-sided) = 0.6269, p(Wilcoxon, less) = 0.1061 n.s.
- 95 % CI of the mean difference: [-0.0284, +0.0176]
- improving runs: 12/18  |  gain: +1.87 % (mean of ratios), +2.06 % (ratio of means)
- domains negative: 4/6 (sign-test p = 0.3438)

| domain | n | paired mean |
|---|---|---|
| annulus | 3 | +0.0022 |
| double_hole | 3 | -0.0156 |
| lshape | 3 | -0.0049 |
| naca0012 | 3 | -0.0132 |
| slit | 3 | -0.0082 |
| star | 3 | +0.0073 |

## Admissibility (worst case over all runs)

| variant | fold [%] | min det J | cert. lower bound | inverse error |
|---|---|---|---|---|
| `chart-eq` | 0.00 | 0.3210 | 0.2042 | 2.91e-07 |
| `chart-fixed` | 0.00 | 0.4467 | 0.4019 | 3.58e-07 |

