# Benchmark, v2 (paired seeds)

Protocol: 100 samples/geometry, N=64, 80/20 split, 40 epochs, Adam 1e-3, batch 10, seeds 0-2.  Paired difference vs `fno` at the same seed; negative = better.

| Domain | fno L2 | chart-fixed L2 | chart-eq L2 | Δchart-fixed L2 | Δchart-eq L2 |
|---|---|---|---|---|---|
| star | 0.2723 | 0.2812 | 0.2815 | +0.0088 ± 0.0755 | +0.0092 ± 0.0763 |
| lshape | 0.0962 | 0.0907 | 0.0995 | -0.0054 ± 0.0148 | +0.0033 ± 0.0133 |
| annulus | 0.1447 | 0.1390 | 0.1391 | -0.0057 ± 0.0211 | -0.0056 ± 0.0219 |
| naca0012 | 0.0906 | 0.0805 | 0.0860 | -0.0100 ± 0.0125 | -0.0045 ± 0.0078 |
| slit | 0.1041 | 0.0958 | 0.0952 | -0.0083 ± 0.0081 | -0.0089 ± 0.0051 |
| double_hole | 0.1118 | 0.1039 | 0.1162 | -0.0079 ± 0.0115 | +0.0044 ± 0.0147 |

*Ablation, identical architecture and budget:* `chart-eq` minus `chart-fixed` = **+0.0044 ± 0.0073** over n = 18 (t = +2.55, p = 0.0206, 6/18 runs negative).

## Admissibility

| Domain | variant | fold % | min detJ | certified lb | kappa | inv err | eq res % |
|---|---|---|---|---|---|---|---|
| star | chart-fixed | 0.00 | 0.957 | 9.40e-01 | 1.00 | 1.2e-07 | -- |
| star | chart-eq | 0.00 | 0.953 | 9.37e-01 | 1.00 | 1.8e-07 | 264.3 |
| lshape | chart-fixed | 0.00 | 0.541 | 5.16e-01 | 1.00 | 2.3e-07 | -- |
| lshape | chart-eq | 0.00 | 0.337 | 2.29e-01 | 1.00 | 2.1e-07 | 114.2 |
| annulus | chart-fixed | 0.00 | 0.843 | 7.70e-01 | 1.00 | 1.3e-07 | -- |
| annulus | chart-eq | 0.00 | 0.934 | 9.13e-01 | 1.00 | 2.0e-07 | 181.3 |
| naca0012 | chart-fixed | 0.00 | 0.526 | 5.08e-01 | 1.00 | 2.0e-07 | -- |
| naca0012 | chart-eq | 0.00 | 0.383 | 3.75e-01 | 1.00 | 1.5e-07 | 89.7 |
| slit | chart-fixed | 0.00 | 0.574 | 5.53e-01 | 1.00 | 7.9e-08 | -- |
| slit | chart-eq | 0.00 | 0.372 | 3.63e-01 | 1.00 | 2.9e-07 | 88.2 |
| double_hole | chart-fixed | 0.00 | 0.751 | 7.38e-01 | 1.00 | 1.3e-07 | -- |
| double_hole | chart-eq | 0.00 | 0.695 | 6.77e-01 | 1.00 | 1.7e-07 | 120.2 |
