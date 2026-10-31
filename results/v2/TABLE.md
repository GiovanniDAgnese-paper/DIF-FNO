# Benchmark, v2 (paired seeds)

Protocol: 100 samples/geometry, N=64, 80/20 split, 40 epochs, Adam 1e-3, batch 10, seeds 0-2.  Paired difference vs `fno` at the same seed; negative = better.

| Domain | fno L2 | chart-fixed L2 | chart-eq L2 | Δchart-fixed L2 | Δchart-eq L2 |
|---|---|---|---|---|---|
| star | 0.1636 | 0.1624 | 0.1597 | -0.0012 ± 0.0119 | -0.0039 ± 0.0124 |
| lshape | 0.2045 | 0.1995 | 0.1898 | -0.0050 ± 0.0110 | -0.0147 ± 0.0080 |
| annulus | 0.1835 | 0.1729 | 0.1713 | -0.0106 ± 0.0179 | -0.0121 ± 0.0178 |
| naca0012 | 0.1975 | 0.1872 | 0.1843 | -0.0103 ± 0.0145 | -0.0132 ± 0.0146 |
| slit | 0.2289 | 0.2095 | 0.1975 | -0.0194 ± 0.0191 | -0.0314 ± 0.0175 |
| double_hole | 0.1986 | 0.1801 | 0.1649 | -0.0185 ± 0.0038 | -0.0337 ± 0.0052 |

*Ablation, identical architecture and budget:* `chart-eq` minus `chart-fixed` = **-0.0073 ± 0.0072** over n = 18 (t = -4.34, p = 0.0004, 17/18 runs negative).

## Admissibility

| Domain | variant | fold % | min detJ | certified lb | kappa | inv err | eq res % |
|---|---|---|---|---|---|---|---|
| star | chart-fixed | 0.00 | 0.930 | 8.54e-01 | 1.00 | 2.2e-07 | -- |
| star | chart-eq | 0.00 | 0.942 | 9.13e-01 | 1.00 | 1.2e-07 | 258.7 |
| lshape | chart-fixed | 0.00 | 0.408 | 3.66e-01 | 1.00 | 2.5e-07 | -- |
| lshape | chart-eq | 0.00 | 0.045 | 2.67e-02 | 1.00 | 4.8e-07 | 167.3 |
| annulus | chart-fixed | 0.00 | 0.867 | 7.65e-01 | 1.00 | 1.8e-07 | -- |
| annulus | chart-eq | 0.00 | 0.907 | 8.62e-01 | 1.00 | 1.8e-07 | 216.2 |
| naca0012 | chart-fixed | 0.00 | 0.343 | 3.10e-01 | 1.00 | 2.8e-07 | -- |
| naca0012 | chart-eq | 0.00 | 0.042 | 2.57e-02 | 1.00 | 6.0e-07 | 139.8 |
| slit | chart-fixed | 0.00 | 0.393 | 3.60e-01 | 1.00 | 2.2e-07 | -- |
| slit | chart-eq | 0.00 | 0.046 | 2.17e-02 | 1.00 | 5.6e-07 | 141.2 |
| double_hole | chart-fixed | 0.00 | 0.436 | 4.03e-01 | 1.00 | 1.8e-07 | -- |
| double_hole | chart-eq | 0.00 | 0.043 | 2.19e-02 | 1.00 | 3.2e-07 | 146.5 |

Not tabulated (incomplete coverage): `dif-barrier` (star only).
