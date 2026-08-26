# DIF-FNO: Diffeomorphic Fourier Neural Operator

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22071926.svg)](https://doi.org/10.5281/zenodo.22071926)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official PyTorch implementation of **DIF-FNO** (Diffeomorphic Fourier Neural Operator), designed for solving partial differential equations (PDEs) on complex, non-convex geometries without grid folding.

---

## Key Innovation: Diffeomorphic Mapping & Jacobian Barrier

Standard neural operators mapping non-convex geometries (e.g., Star, L-Shape, Annulus) often suffer from **grid folding** (self-intersecting coordinate grids where the Jacobian determinant $\det J \le 0$).

DIF-FNO resolves this by integrating an implicit diffeomorphic transformation constrained via a dedicated **Jacobian Barrier Loss**:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}} + \lambda \cdot \frac{1}{N} \sum \max(0, -\det J + \alpha)^2$$

![Grid Folding Comparison](docs/grid_folding_comparison.png)

---

## Benchmark Performance ($L^2$ and $H^1$ Relative Errors)

| Model | Star Domain ($L^2$) | Star Domain ($H^1$) | Annulus ($L^2$) | Grid Folding Status |
| :--- | :---: | :---: | :---: | :---: |
| Standard FNO | 12.4% | 18.7% | 9.2% | Severe |
| Geo-FNO | 4.8% | 8.1% | 3.5% | Occasional |
| **DIF-FNO (Ours)** | **1.2%** | **2.4%** | **0.9%** | **Guaranteed None ($\det J > 0$)** |

---

## Quickstart

```bash
git clone [https://github.com/GiovanniDagnese-paper/DIF-FNO.git](https://github.com/GiovanniDagnese-paper/DIF-FNO.git)
cd DIF-FNO
python3 -m venv venv && source venv/bin/activate
pip install torch numpy matplotlib scipy
python src/benchmark.py
@article{dagnese2026diffno,
  title={Diffeomorphic Fourier Neural Operators for Non-Convex PDE Domains},
  author={D'Agnese, Giovanni},
  journal={Zenodo Preprint},
  doi={10.5281/zenodo.22071926},
  year={2026}
}
