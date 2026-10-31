# Deterministic verification of P5--P8

## P5 -- the certificate cannot be lost (adversarial parameters)

| parameter scale | min det J | folded % | certified lb | parameter-free floor |
|---|---|---|---|---|
| 0 | 1.000e+00 | 0.00 | 1.000e+00 | 1.000e-06 |
| 0.1 | 8.863e-01 | 0.00 | 8.797e-01 | 1.000e-06 |
| 1 | 1.409e-06 | 0.00 | 1.224e-06 | 1.000e-06 |
| 3 | 9.978e-07 | 0.00 | 9.978e-07 | 1.000e-06 |
| 10 | 9.974e-07 | 0.00 | 9.974e-07 | 1.000e-06 |
| 30 | 9.974e-07 | 0.00 | 9.974e-07 | 1.000e-06 |
| 100 | 9.993e-07 | 0.00 | 9.993e-07 | 1.000e-06 |
| 1000 | 9.993e-07 | 0.00 | 9.993e-07 | 1.000e-06 |

## P6 -- round trip is the interpolation error

### smooth field

| n | max error | rms error | inverse error |
|---|---|---|---|
| 32 | 5.603e-02 | 1.229e-02 | 4.17e-07 |
| 64 | 1.430e-02 | 3.158e-03 | 6.56e-07 |
| 128 | 3.462e-03 | 8.146e-04 | 6.56e-07 |
| 256 | 9.037e-04 | 2.043e-04 | 1.16e-06 |

Observed order (rms / max): **1.97 / 1.99**

### multiscale field

| n | max error | rms error | inverse error |
|---|---|---|---|
| 32 | 2.022e-01 | 3.610e-02 | 4.17e-07 |
| 64 | 7.938e-02 | 1.112e-02 | 6.56e-07 |
| 128 | 2.700e-02 | 3.019e-03 | 6.56e-07 |
| 256 | 7.448e-03 | 7.748e-04 | 1.16e-06 |

Observed order (rms / max): **1.85 / 1.58**

## P7 -- equidistribution reduces the representation error

### ridge

| n | uniform max | adaptive max | gain | uniform rms | adaptive rms | gain |
|---|---|---|---|---|---|---|
| 9 | 5.273e-01 | 4.230e-01 | 1.25x | 1.004e-01 | 8.153e-02 | 1.23x |
| 17 | 1.581e-01 | 1.176e-01 | 1.34x | 2.416e-02 | 1.860e-02 | 1.30x |
| 33 | 5.159e-02 | 4.154e-02 | 1.24x | 6.654e-03 | 5.286e-03 | 1.26x |
| 65 | 1.316e-02 | 1.132e-02 | 1.16x | 1.793e-03 | 1.423e-03 | 1.26x |

### boundary_layer

| n | uniform max | adaptive max | gain | uniform rms | adaptive rms | gain |
|---|---|---|---|---|---|---|
| 9 | 5.590e-01 | 5.127e-01 | 1.09x | 1.124e-01 | 1.216e-01 | 0.92x |
| 17 | 3.865e-01 | 1.729e-01 | 2.23x | 7.709e-02 | 3.527e-02 | 2.19x |
| 33 | 1.099e-01 | 4.823e-02 | 2.28x | 1.863e-02 | 8.710e-03 | 2.14x |
| 65 | 3.646e-02 | 1.407e-02 | 2.59x | 5.933e-03 | 2.456e-03 | 2.42x |

### multiscale

| n | uniform max | adaptive max | gain | uniform rms | adaptive rms | gain |
|---|---|---|---|---|---|---|
| 9 | 5.587e-01 | 5.844e-01 | 0.96x | 7.441e-02 | 7.465e-02 | 1.00x |
| 17 | 2.600e-01 | 2.532e-01 | 1.03x | 2.147e-02 | 2.118e-02 | 1.01x |
| 33 | 8.217e-02 | 7.813e-02 | 1.05x | 5.720e-03 | 5.661e-03 | 1.01x |
| 65 | 2.244e-02 | 2.204e-02 | 1.02x | 1.528e-03 | 1.510e-03 | 1.01x |

## P7b -- the loss learns the mesh (chart trained by gradient descent only)

| field | loss (init -> final) | mass ratio | cell ratio | cert. lb | repr. error uniform | chart | gain |
|---|---|---|---|---|---|---|---|
| boundary_layer | 0.4315 -> 0.00090 | 1.30 | 5.44 | 0.235 | 4.0567e-02 | 9.9197e-03 | 4.09x |
| ridge | 0.1763 -> 0.00117 | 1.49 | 3.35 | 0.379 | 1.3397e-02 | 6.6857e-03 | 2.00x |
| smooth | 0.0294 -> 0.00216 | 1.81 | 2.28 | 0.679 | 1.7143e-03 | 3.1516e-03 | 0.54x |

## P8 -- chart equivariance of the operator

| n | relative difference between two charts | displacement |
|---|---|---|
| 32 | 4.422e-02 | 0.461 |
| 64 | 1.142e-02 | 0.479 |
| 128 | 2.895e-03 | 0.488 |
