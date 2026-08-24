# DIF-FNO: Diffeomorphic Fourier Neural Operator

Official PyTorch implementation of **DIF-FNO** (*Diffeomorphic Fourier Neural Operator for Complex Geometries*).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch->=2.0-ee4c2c.svg)](https://pytorch.org/)

## 🌟 Key Features
* **Grid Folding Guarantee:** Explicit regularizer enforcing $\det(J) > 0$.
* **Physical Derivative Accuracy:** Geometric Chain Rule for exact $H^1$ error computation.
* **Resolution Invariance:** Stable transfer from $64\times 64$ up to $256\times 256$.
* **Zero-Shot Generalization:** Parametric polar geometry generator included.

## 🚀 Quick Start

```bash
# Clone the repository
git clone [https://github.com/GiovanniDagnese-paper/DIF-FNO.git](https://github.com/GiovanniDagnese-paper/DIF-FNO.git)
cd DIF-FNO

# Run Evaluation Benchmark
python3 run_full_benchmark.py
