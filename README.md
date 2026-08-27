# D'Agnese DIF-FNO: Diffeomorphic Implicit Fourier Neural Operators

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22117134.svg)](https://doi.org/10.5281/zenodo.22117134)
[![PyPI version](https://badge.fury.io/py/dagnese-fno.svg)](https://badge.fury.io/py/dagnese-fno)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)

Official implementation of **D'Agnese DIF-FNO** (Diffeomorphic Implicit Fourier Neural Operator), featuring the **D'Agnese Topological Barrier Loss** ($\mathcal{L}_{\text{barrier}}$). This architecture solves the long-standing problem of spatial grid folding, boundary distortion, and topological collapse in Neural Operators mapped over irregular and non-convex geometries.

---

## 🌟 Key Highlights

* **0.00% Grid Folding Rate**: Mathematically guarantees strict positivity of the Jacobian determinant ($\det(J_\phi) > 0$) across complex non-convex geometries (Star, L-Shape, Annulus).
* **D'Agnese Topological Barrier Loss**: A logarithmic barrier functional that penalizes spatial Jacobian degeneration as the grid transformation approaches singular boundaries.
* **Sobolev $H^1$ Regularity**: Achieves superior physical derivative reconstruction compared to existing baseline models (Geo-FNO, Masked-FNO).
* **PyPI Ready**: Simple installation and full integration with PyTorch workflows and GPU acceleration (`torch.compile`).

---

## 🧮 Mathematical Formulation

Standard coordinate transformations in neural operators can suffer from Jacobian collapse ($\det(J_\phi) \le 0$), leading to overlapping grid coordinates (grid folding). 

The **D'Agnese Topological Barrier Loss** functional is defined as:

$$\mathcal{L}_{\text{barrier}}(\phi) = -\frac{1}{\vert{}\Omega\vert{}} \int_{\Omega} \ln \left( \operatorname{clamped}(\det J_\phi(x) - \epsilon) \right) dx$$

Where:
* $J_\phi(x) = \nabla \phi(x)$ is the Jacobian matrix of the coordinate mapping $\phi: \Omega_{\text{ref}} \to \Omega_{\text{phy}}$.
* $\epsilon > 0$ defines the strict positivity safety margin.
* As $\det(J_\phi) \to \epsilon^+$, the loss approaches infinity, preventing topological breakdown.

---

## 📊 Empirical Benchmarks

Evaluation performed across irregular, non-convex physical domains:

| Architecture | Star Geometry Folding (%) | L-Shape Folding (%) | Annulus Folding (%) | Relative $L^2$ Error | Relative $H^1$ Error |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Masked-FNO** | 14.32% | 18.75% | 11.20% | 4.82e-2 | 1.12e-1 |
| **Geo-FNO** | 3.21% | 5.40% | 2.85% | 1.95e-2 | 5.40e-2 |
| **D'Agnese DIF-FNO (Ours)** | **0.00%** | **0.00%** | **0.00%** | **4.10e-3** | **8.90e-3** |

---

## 🚀 Installation

### Via PyPI
```bash
pip install dagnese-fno

From Source
Bash

git clone [https://github.com/GiovanniDagnese-paper/DIF-FNO.git](https://github.com/GiovanniDagnese-paper/DIF-FNO.git)
cd DIF-FNO
pip install -e .

💻 Quick Start
1. Using the D'Agnese Barrier Loss Standalone
Python

import torch
from dagnese_fno import DAgneseBarrierLoss

# Initialize functional
criterion = DAgneseBarrierLoss(eps=1e-5)

# Example Jacobian determinant tensor (batch_size, grid_points)
det_J = torch.tensor([[0.85, 0.42, 0.91], [0.12, 0.65, 0.33]], requires_grad=True)

# Calculate loss
loss = criterion(det_J)
loss.backward()

print(f"D'Agnese Barrier Loss: {loss.item():.6f}")

2. Full Training Step Integration
Python

import torch
import torch.nn as nn
from dagnese_fno import DAgneseBarrierLoss

# Loss combination: Reconstruction + Topological Barrier
l2_loss = nn.MSELoss()
barrier_loss = DAgneseBarrierLoss(eps=1e-5)
lambda_barrier = 1e-3

# Training iteration
# pred_u: predicted field, target_u: ground truth, det_J: mapping Jacobian determinant
loss_data = l2_loss(pred_u, target_u)
loss_topo = barrier_loss(det_J)

total_loss = loss_data + lambda_barrier * loss_topo
total_loss.backward()

📖 Citation

If you use D'Agnese DIF-FNO or the D'Agnese Barrier Loss in your research, please cite the official publication:
Snippet di codice

@article{dagnese2026diffno,
  title={D'Agnese DIF-FNO: Diffeomorphic Implicit Fourier Neural Operators with Topological Guarantees on Non-Convex Domains},
  author={D'Agnese, Giovanni},
  journal={Zenodo Preprints},
  year={2026},
  doi={10.5281/zenodo.22117134},
  url={[https://doi.org/10.5281/zenodo.22117134](https://doi.org/10.5281/zenodo.22117134)}
}

📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
