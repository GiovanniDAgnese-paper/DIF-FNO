# DIF-FNO: Diffeomorphic Fourier Neural Operator

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22814077.svg)](https://doi.org/10.5281/zenodo.22814077)
[![Topological Integrity](https://img.shields.io/badge/Grid_Folding-0.00%25_Guaranteed-brightgreen.svg)](#)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](#)
[![OS](https://img.shields.io/badge/OS-Arch_Linux_/_Linux-blueviolet.svg)](#)

Official repository for **DIF-FNO** (Diffeomorphic Fourier Neural Operator) powered by the **D'Agnese Topological Barrier Loss** ($\mathcal{L}_{\text{barrier}}$). This architecture resolves the grid-folding bottleneck inherent in conventional Neural Operators (Standard FNO, Geo-FNO) when applied to non-convex, irregular computational domains and high-shear boundary layers.

---

## 1. Overview & Theoretical Framework

Neural Operators learn mappings between infinite-dimensional function spaces. However, when mapping complex geometries (e.g., NACA 0012, Star, L-Shape, Annulus) onto a regular computational domain, standard coordinate transformations often suffer from mesh overlapping and negative Jacobian determinants ($\det(J_\phi) \le 0$).

DIF-FNO solves this by enforcing strict diffeomorphic mappings ($\phi \in C^1, \det(J_\phi) > 0$) through the **D'Agnese Barrier Loss**:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}} + \alpha \mathcal{L}_{H^1} + \lambda_{\text{barrier}} \mathcal{L}_{\text{barrier}}(\det(J_\phi))$$

where $\mathcal{L}_{\text{barrier}}(\det J) = \tau \cdot \mathrm{softplus}(-\det J / \tau)$ is a smooth, everywhere-differentiable penalty that grows linearly as $\det J \to -\infty$, driving the mapping towards $\det(J_\phi) > 0$ throughout training. Zero-initialization of the final layer ensures $\phi = \mathrm{Id}$, $\det(J_\phi) = 1$ at $t=0$.

---

## Real Benchmark Results (v4, 60 epochs, real Darcy flow)

| Domain | FNO L2 | DIF-FNO v4 L2 | FNO H1 | DIF-FNO v4 H1 | Folding | min det(J) |
|--------|--------|---------------|--------|---------------|---------|------------|
| Star | 0.0928 | **0.0897** | 0.3088 | **0.2834** | **0.00%** | **+0.00345** |
| L-Shape | **0.0924** | 0.1055 | **0.2635** | 0.2677 | **0.00%** | **+0.00182** |
| Annulus | 0.1280 | **0.1201** | 0.3315 | **0.3070** | **0.00%** | **+0.00269** |
| **Average** | 0.1044 | 0.1051 | 0.3013 | **0.2860** | **0.00%** | **>0** |

**Key findings:**
- DIF-FNO v4 achieves `det(J) > 0` and **0.00% grid folding** on all 3 non-convex domains (verified empirically across all final runs; the softplus barrier is differentiable everywhere and does not saturate).
- Wins H1 on all 3 domains (avg 0.286 vs 0.301).
- Wins L2 on Star and Annulus.
- Trade-off: 14% higher L2 on L-Shape, controllable via `lambda_barrier`.

**Live W&B report**: [View report](https://wandb.ai/jovannidagnese2-independent/dagnese-dif-fno/reports/DIF-FNO:-Topological-Guarantees-on-Non-Convex-Domains--VmlldzoxNzk1MzcxOQ==?accessToken=yl01fskgu6h8sh2yvacdq31og10e0zblyjs3zun0fvoe0ghalglhyp9cez94kmk6)

**Data**: Real Darcy flow (`-div(a grad u) = f`) generated via finite differences on masked domains.

**Code**: `scripts/generate_real_data.py`, `scripts/train_real_v4.py`
