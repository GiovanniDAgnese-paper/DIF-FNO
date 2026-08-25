# DIF-FNO: Diffeomorphic Fourier Neural Operators

Official implementation of **DIF-FNO (Diffeomorphic Fourier Neural Operator)**, an architecture designed for learning solution operators of PDEs on complex, deformed geometries with topological bijectivity guarantees (\det J > 0) and exact Sobolev H^1 accuracy.

## Benchmark Results (Deformed Darcy Flow)

| Model | L2 Error (64x64) | H1 Error (64x64) | Latency (ms) | Topological Guarantee |
| :--- | :---: | :---: | :---: | :---: |
| **DIF-FNO (Ours)** | **0.0182** | **0.0648** | 4.12 | **Yes** |
| Geo-FNO | 0.0425 | 0.2104 | 3.85 | No |
| Masked-FNO | 0.0881 | 0.2401 | 2.91 | N/A |

## Directory Structure

```text
├── src/                # Core architecture & diffeomorphism modules
├── benchmarks/         # Training scripts & Stress-test ablation
├── figures/            # Generated convergence & ablation plots
├── tables/             # LaTeX formatted result tables
├── docs/               # Main paper LaTeX source (main.tex)
└── README.md
