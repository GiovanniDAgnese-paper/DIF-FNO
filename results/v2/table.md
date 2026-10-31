# DIF-FNO benchmark

*54 runs, seeds [0, 1, 2]; mean $\pm$ s.d. over the seeds.*

| Domain | fno $L^2$ | chart-fixed $L^2$ | chart-eq $L^2$ | fno $H^1$ | chart-fixed $H^1$ | chart-eq $H^1$ |
|---|---|---|---|---|---|---|
| star | 0.2723 $\pm$ 0.0472 | 0.2812 $\pm$ 0.0465 | 0.2815 $\pm$ 0.0473 | 0.3876 $\pm$ 0.0554 | 0.3948 $\pm$ 0.0666 | 0.3961 $\pm$ 0.0684 |
| lshape | 0.0962 $\pm$ 0.0131 | 0.0907 $\pm$ 0.0069 | 0.0995 $\pm$ 0.0061 | 0.2088 $\pm$ 0.0167 | 0.2039 $\pm$ 0.0098 | 0.2873 $\pm$ 0.0415 |
| annulus | 0.1447 $\pm$ 0.0325 | 0.1390 $\pm$ 0.0141 | 0.1391 $\pm$ 0.0151 | 0.3100 $\pm$ 0.0350 | 0.3122 $\pm$ 0.0291 | 0.3147 $\pm$ 0.0323 |
| double_hole | 0.1118 $\pm$ 0.0141 | 0.1039 $\pm$ 0.0035 | 0.1162 $\pm$ 0.0108 | 0.2387 $\pm$ 0.0115 | 0.2231 $\pm$ 0.0005 | 0.2651 $\pm$ 0.0201 |
| naca0012 | 0.0906 $\pm$ 0.0163 | 0.0805 $\pm$ 0.0065 | 0.0860 $\pm$ 0.0084 | 0.2103 $\pm$ 0.0239 | 0.1970 $\pm$ 0.0136 | 0.2156 $\pm$ 0.0216 |
| slit | 0.1041 $\pm$ 0.0127 | 0.0958 $\pm$ 0.0148 | 0.0952 $\pm$ 0.0136 | 0.2162 $\pm$ 0.0135 | 0.2081 $\pm$ 0.0234 | 0.2123 $\pm$ 0.0214 |
| **Average (common domains)** | 0.1366 $\pm$ 0.0631 | 0.1318 $\pm$ 0.0692 | 0.1362 $\pm$ 0.0672 | 0.2619 $\pm$ 0.0661 | 0.2565 $\pm$ 0.0730 | 0.2818 $\pm$ 0.0628 |
| **Average (all domains)** | 0.1366 $\pm$ 0.0631 | 0.1318 $\pm$ 0.0692 | 0.1362 $\pm$ 0.0672 | 0.2619 $\pm$ 0.0661 | 0.2565 $\pm$ 0.0730 | 0.2818 $\pm$ 0.0628 |

## Topology

| Domain | fno fold % | chart-fixed fold % | chart-eq fold % | fno min det J | chart-fixed min det J | chart-eq min det J | fno certified | chart-fixed certified | chart-eq certified |
|---|---|---|---|---|---|---|---|---|---|
| star | nan | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | +nan | +0.957 $\pm$ 0.005 | +0.953 $\pm$ 0.007 | nan | 0.940 $\pm$ 0.003 | 0.937 $\pm$ 0.006 |
| lshape | nan | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | +nan | +0.541 $\pm$ 0.110 | +0.337 $\pm$ 0.015 | nan | 0.516 $\pm$ 0.131 | 0.229 $\pm$ 0.034 |
| annulus | nan | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | +nan | +0.843 $\pm$ 0.050 | +0.934 $\pm$ 0.006 | nan | 0.770 $\pm$ 0.046 | 0.913 $\pm$ 0.004 |
| double_hole | nan | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | +nan | +0.751 $\pm$ 0.079 | +0.695 $\pm$ 0.100 | nan | 0.738 $\pm$ 0.084 | 0.677 $\pm$ 0.113 |
| naca0012 | nan | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | +nan | +0.526 $\pm$ 0.071 | +0.383 $\pm$ 0.017 | nan | 0.508 $\pm$ 0.087 | 0.375 $\pm$ 0.017 |
| slit | nan | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | +nan | +0.574 $\pm$ 0.024 | +0.372 $\pm$ 0.007 | nan | 0.553 $\pm$ 0.028 | 0.363 $\pm$ 0.005 |
| **Average (common domains)** | -- | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | -- | +0.447 | +0.321 | -- | 0.671 $\pm$ 0.159 | 0.582 $\pm$ 0.277 |
| **Average (all domains)** | -- | 0.00 $\pm$ 0.00 | 0.00 $\pm$ 0.00 | -- | +0.447 | +0.321 | -- | 0.671 $\pm$ 0.159 | 0.582 $\pm$ 0.277 |

## Displacement

| Domain | fno $|\phi-\mathrm{Id}|$ | chart-fixed $|\phi-\mathrm{Id}|$ | chart-eq $|\phi-\mathrm{Id}|$ |
|---|---|---|---|
| star | nan | 0.023 $\pm$ 0.001 | 0.023 $\pm$ 0.001 |
| lshape | nan | 0.081 $\pm$ 0.021 | 0.104 $\pm$ 0.007 |
| annulus | nan | 0.029 $\pm$ 0.013 | 0.021 $\pm$ 0.002 |
| double_hole | nan | 0.034 $\pm$ 0.008 | 0.078 $\pm$ 0.039 |
| naca0012 | nan | 0.084 $\pm$ 0.028 | 0.099 $\pm$ 0.006 |
| slit | nan | 0.067 $\pm$ 0.019 | 0.110 $\pm$ 0.005 |
| **Average (common domains)** | -- | 0.053 $\pm$ 0.025 | 0.072 $\pm$ 0.037 |
| **Average (all domains)** | -- | 0.053 $\pm$ 0.025 | 0.072 $\pm$ 0.037 |

## Paired differences against `fno` (same seed, same domain)

Negative = better than the baseline. Brackets: 95% Student-t interval of the paired difference (the bootstrap interval is narrower but undercovers at n = 6 and is only reported in `CONSOLIDATED.json`). Stars: paired *t*-test, or the exact Wilcoxon signed-rank test when n >= 6; neither is corrected for the number of comparisons.

| Domain | chart-eq $\Delta L^2$ | chart-fixed $\Delta L^2$ | chart-eq $\Delta H^1$ | chart-fixed $\Delta H^1$ |
|---|---|---|---|---|
| star | +0.0092 [-0.2229, +0.2412] n.s. (n=3) | +0.0088 [-0.2210, +0.2387] n.s. (n=3) | +0.0086 [-0.2941, +0.3112] n.s. (n=3) | +0.0073 [-0.2906, +0.3051] n.s. (n=3) |
| lshape | +0.0033 [-0.0373, +0.0439] n.s. (n=3) | -0.0054 [-0.0504, +0.0395] n.s. (n=3) | +0.0785 [+0.0153, +0.1417] * (n=3) | -0.0049 [-0.0687, +0.0590] n.s. (n=3) |
| annulus | -0.0056 [-0.0724, +0.0612] n.s. (n=3) | -0.0057 [-0.0699, +0.0586] n.s. (n=3) | +0.0047 [-0.1089, +0.1183] n.s. (n=3) | +0.0022 [-0.1052, +0.1096] n.s. (n=3) |
| double_hole | +0.0044 [-0.0403, +0.0492] n.s. (n=3) | -0.0079 [-0.0430, +0.0272] n.s. (n=3) | +0.0263 [-0.0473, +0.1000] n.s. (n=3) | -0.0156 [-0.0430, +0.0118] n.s. (n=3) |
| naca0012 | -0.0045 [-0.0283, +0.0193] n.s. (n=3) | -0.0100 [-0.0480, +0.0280] n.s. (n=3) | +0.0053 [-0.0393, +0.0499] n.s. (n=3) | -0.0132 [-0.0696, +0.0432] n.s. (n=3) |
| slit | -0.0089 [-0.0245, +0.0067] n.s. (n=3) | -0.0083 [-0.0330, +0.0164] n.s. (n=3) | -0.0040 [-0.0239, +0.0159] n.s. (n=3) | -0.0082 [-0.0347, +0.0184] n.s. (n=3) |
| **pooled** | -0.0004 [-0.0179, +0.0172] n.s. (n=18) | -0.0047 [-0.0222, +0.0127] n.s. (n=18) | +0.0199 [-0.0075, +0.0473] n.s. (n=18) | -0.0054 [-0.0284, +0.0176] n.s. (n=18) |
