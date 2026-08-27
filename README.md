# D'Agnese DIF-FNO: Diffeomorphic Implicit Fourier Neural Operators

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22117134.svg)](https://doi.org/10.5281/zenodo.22117134)
[![PyPI version](https://badge.fury.io/py/dagnese-fno.svg)](https://badge.fury.io/py/dagnese-fno)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of **D'Agnese DIF-FNO**, introducing the **D'Agnese Topological Barrier Loss** ($\mathcal{L}_{\text{barrier}}$). This architecture guarantees global $C^1$-diffeomorphic invertibility and eliminates grid folding (**0.00% folding rate**) in Fourier Neural Operators on irregular, non-convex geometries (Star, L-Shape, Annulus).

## Installation

```bash
pip install dagnese-fno
