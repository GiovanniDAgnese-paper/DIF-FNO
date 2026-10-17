# DIF-FNO — Diffeomorphic Implicit Fourier Neural Operator

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22814077.svg)](https://doi.org/10.5281/zenodo.22814077)
[![CI](https://github.com/GiovanniDAgnese-paper/DIF-FNO/actions/workflows/ci.yml/badge.svg)](https://github.com/GiovanniDAgnese-paper/DIF-FNO/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-112%20passing-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Certified](https://img.shields.io/badge/certified%20det%20J%20%3E%200-(1-%CE%BA%C2%B7L)%C2%B2%20%3E%200-brightgreen.svg)](docs/THEORY.md)

Official implementation of **DIF-FNO** and of the **D'Agnese Topological Barrier
Loss** $\mathcal{L}_{\text{barrier}}$: a Fourier Neural Operator that first learns
an input-conditioned diffeomorphism $\phi_\psi$ and then evaluates the operator
on the deformed (computational) coordinates, so that
$\det(J_\phi) > 0$ — no grid folding — on non-convex domains.

> **Version 1.1.0 — what changed.** The 1.0 line reported "0.00 % grid folding"
> as an empirical observation obtained with a barrier whose *effective*
> temperature scaled like $\tau/h^2$ (the Jacobian determinant was not
> normalised by the grid spacing). This release fixes that, re-runs the whole
> benchmark (4 domains × 3 seeds × 5 configurations = 54 runs, all reproducible
> from this repository), and adds the configuration for which $\det J_\phi > 0$
> is a **theorem** rather than an observation:
> $\det(J_\phi) \ge (1 - \kappa L)^2 > 0$. See
> [`docs/THEORY.md`](docs/THEORY.md) and [§6](#6-what-is-proved-and-what-is-not).

---

## 1. The idea in one figure

```
        a(x)  ─────────────────────────────►  FNO backbone  ──►  u(x)
                                                   ▲
   ξ ──► φ_ψ(ξ, a) ──► ξ'                          │
        (diffeomorphism, det J > 0)   (ξ'_x, ξ'_y, a) ─┘
```

* **Baseline** (`fno`): the backbone sees the physical coordinates $(\xi_1, \xi_2, a)$.
* **DIF-FNO** (`dif`): the backbone sees the *learned* coordinates
  $(\phi_1, \phi_2, a)$, with $\phi = \mathrm{Id} + s\,r_\psi$ zero-initialised.

The two columns of every table below share the **same** backbone
(`width`, `modes`, `depth`, initialisation); the only difference is the
diffeomorphic branch ($+1.2$ k / $+6.2$ k parameters out of $\sim 335$ k).

---

## 2. Install

```bash
git clone https://github.com/GiovanniDAgnese-paper/DIF-FNO.git && cd DIF-FNO
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # runtime + pytest + ruff
```

CPU-only PyTorch (much smaller download):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

## 3. Thirty-second tour

```python
import torch
from dagnese_fno import DIFFNO2d, DAgneseBarrierLoss, jacobian_det

coords = torch.stack(torch.meshgrid(torch.linspace(-1, 1, 64),
                                    torch.linspace(-1, 1, 64), indexing="ij"), -1)[None]
a = torch.rand(1, 64, 64)

model = DIFFNO2d()                       # zero-init: phi = Id, det J = 1
u, xi = model(coords, a)                 # u: (1, 64, 64), xi: (1, 64, 64, 2)

det = jacobian_det(xi, spacing=2 / 63)   # == 1 at initialisation, any resolution
loss = DAgneseBarrierLoss(tau=0.05)(det) # ~0 while det J > 0
```

Certified (provably folding-free) variant:

```python
model = DIFFNO2d(deformation_scale=0.5, conditioning="global",
                 lipschitz_mode="spectral", lipschitz_const=0.9)
u, xi = model(coords, a, mask=None)
# det J >= (1 - kappa*L)^2 with kappa = 1  ->  guaranteed > 0 for all inputs
```

Runnable end-to-end example: [`examples/quickstart.py`](examples/quickstart.py).

---

## 4. Reproducing the benchmark

```bash
# 1) data: Darcy flow -div(a grad u) = f, 100 samples per domain, 64 x 64    (~10 s)
python scripts/generate_data.py --domains star lshape annulus naca0012 --n 64 --samples 100

# 2) one run
python scripts/train.py --domain star --model dif --lambda-barrier 10 --tau 0.05 --epochs 60

# 3) everything in the tables below (54 runs, ~1 h on a 2-core CPU: ~1 min/run)
python scripts/run_benchmark.py --domains star lshape annulus naca0012 \
       --variants fno dif-barrier dif-barrier-large dif-nobar dif-certified \
       --seeds 0 1 2 --epochs 60 --out results/repro

# 4) figures
python scripts/make_figures.py --results results/repro --out figures/repro
```

Everything is seeded (`--seed` drives the model *and* the train/validation
split), and every run dumps its configuration, per-epoch history, environment
and metrics to `results/repro/<variant>_<model>_<domain>_seed<seed>.json`.
`make benchmark` runs the same sweep through the `Makefile`.

---

## 5. Results (this repository, 54 runs)

**Setup.** Darcy flow on 4 masked domains, $64 \times 64$, 100 samples,
80/20 split, 60 epochs, Adam(1e-3), batch 10, backbone `width=24`,
`modes=(12, 12)`, 2 Fourier layers (334 737 parameters), 3 seeds per cell.
Seed-to-seed variability is the dominant source of noise: the standard deviation
over the 3 seeds is 0.007–0.04 on $L^2$ and 0.01–0.05 on $H^1$, i.e. of the same
order as the differences between configurations — which is why the tables below
should be read together with the paired comparison in
`results/repro/table.md`.

| configuration | deformation scale | $\lambda_{\text{barrier}}$ | Star | L-Shape | Annulus | NACA 0012 |
|---|---|---|---|---|---|---|
| `fno` (baseline)                | –   | –    | 0.1061 | 0.1070 | 0.1461 | 0.1128 |
| `dif-barrier` (v4 config)       | 0.1 | 10   | 0.1132 | 0.1012 | 0.1460 | 0.1084 |
| `dif-barrier-large`             | 1.0 | 10   | 0.1135 | 0.1007 | 0.1459 | – |
| `dif-nobar` (ablation)          | 1.0 | 0    | 0.1019 | 0.0973 | 0.1336 | – |
| `dif-certified` **(guaranteed)**| 0.5 | 1    | 0.1018 | 0.1038 | 0.1531 | 0.1058 |

Relative $H^1$ seminorm error (same layout):

| configuration | Star | L-Shape | Annulus | NACA 0012 |
|---|---|---|---|---|
| `fno`                  | 0.2620 | 0.2601 | 0.2730 | 0.2664 |
| `dif-barrier`          | 0.2806 | 0.2603 | 0.2966 | 0.2572 |
| `dif-barrier-large`    | 0.2797 | 0.2590 | 0.2957 | – |
| `dif-nobar`            | 0.2522 | 0.2500 | 0.2751 | – |
| `dif-certified`        | 0.2603 | **0.2499** | 0.3125 | **0.2476** |

Topological diagnostics (mean over seeds; `|φ−Id|` is the mean displacement of
the learned map, `certified` is the *proved* lower bound on $\det J_\phi$):

| configuration | folding | $\min \det J_\phi$ | $\lvert\phi-\mathrm{Id}\rvert$ | certified bound |
|---|---|---|---|---|
| `fno`               | n/a          | n/a     | n/a   | n/a |
| `dif-nobar`         | **12 – 22 %** | −19 … −34 | 0.62 – 0.88 | 0 (none) |
| `dif-barrier`       | 0.00 %       | +0.50 … +0.65 | 0.023 – 0.046 | 0 (none) |
| `dif-barrier-large` | 0.00 %       | +0.43 … +0.59 | 0.05 – 0.13 | 0 (none) |
| `dif-certified`     | **0.00 %**   | **+1.06** | **0.28 – 0.49** | **0.42 – 0.52** |

Figures: [`figures/repro/`](figures/repro) · raw data:
[`results/repro/CONSOLIDATED.json`](results/repro/CONSOLIDATED.json) ·
tables: [`results/repro/table.md`](results/repro/table.md),
[`table.tex`](results/repro/table.tex).

### 5.1 What the numbers say

1. **The barrier does what it claims.** At a deformation scale where the
   unconstrained map folds 12–22 % of the cells ($\min \det J \approx -30$),
   the same architecture trained with $\lambda_{\text{barrier}} = 10$ folds
   **0.00 %** of them, on every domain and every seed. This is the ablation the
   1.0 line was missing.
2. **The v4 configuration barely deforms the grid.** With
   `deformation_scale = 0.1` the learned displacement is
   $\lvert\phi-\mathrm{Id}\rvert \approx 0.03$ — 1.5 % of the domain — so
   "0.00 % folding" was close to a statement about the identity map. Version 1.1
   reports the displacement alongside the folding percentage for exactly this
   reason.
3. **Accuracy is essentially flat across configurations.** Per-seed differences
   are $\pm 0.01$–$0.02$ on $L^2$, i.e. of the same order as the effect being
   measured: the paired differences against the baseline
   (`results/repro/table.md`) are not significant at 3 seeds. In particular the
   *folded* variant (`dif-nobar`) is nominally the most accurate one — folding
   is cheap in pure regression terms here, because the FNO simply learns to
   compensate. **The case for a valid chart is not a lower $L^2$ error at this
   resolution**: it is that the deformed grid is a legitimate coordinate system
   (for mesh-based downstream operators, error estimation, adaptivity,
   interpretation, or when the operator is trained at one resolution and
   evaluated at another).
4. **The certified variant is the interesting operating point.** It deforms the
   grid by $0.28$–$0.49$ (an order of magnitude more than the v4 configuration),
   keeps $\min \det J_\phi \approx 1.06$, and carries the *proof*
   $\det J_\phi \ge (1 - L)^2 \in [0.42, 0.52]$ — while matching the baseline in
   accuracy (best $L^2$ on Star and NACA 0012, best $H^1$ on L-Shape and
   NACA 0012, worst on Annulus).

### 5.2 Relation to the published v4 table

| Domain | v4 reported (FNO → DIF) | this reproduction (FNO → DIF-barrier) |
|---|---|---|
| Star    | $L^2$ 0.0928 → 0.0897, $H^1$ 0.3088 → 0.2834 | 0.1061 → 0.1132, 0.2620 → 0.2806 |
| L-Shape | 0.0924 → 0.1055, 0.2635 → 0.2677 | 0.1070 → 0.1012, 0.2601 → 0.2603 |
| Annulus | 0.1280 → 0.1201, 0.3315 → 0.3070 | 0.1461 → 0.1460, 0.2730 → 0.2966 |

The published v4 runs are preserved under `results/` (`real_*_v3/v4*.json`,
`FINAL_CONSOLIDATED.json`). They are **not** bitwise reproducible: the datasets
were never shipped, and two numerical conventions changed on purpose (§7). Under
the corrected conventions the sign of the effect is not preserved — the "$H^1$
wins on all three domains" claim of v4 is not reproduced, and the differences we
do measure sit inside the seed noise. The qualitative claims (no folding, no
accuracy loss) survive; the quantitative ones should be taken from §5.

---

## 6. What is proved and what is not

| statement | status | where |
|---|---|---|
| $\phi = \mathrm{Id}$, $\det J_\phi = 1$ at initialisation | **proved** (zero-init last layer) | [`docs/THEORY.md`](docs/THEORY.md) P1, `tests/test_models.py` |
| barrier is $C^\infty$, monotone, bounded gradient, linear growth | **proved** | P2, `tests/test_barrier.py` |
| $\mathrm{Lip}(r) \le L$ and $\kappa L < 1 \Rightarrow \det J_\phi \ge (1-\kappa L)^2 > 0$ | **proved** (spectral-norm-capped residual) | P3, `tests/test_models.py` |
| $\det J_\phi$ is resolution independent | **proved** (spacing-normalised stencil) | P4, `tests/test_jacobian.py` |
| softplus barrier $\Rightarrow \det J_\phi > 0$ | **not** true — a soft penalty is not a constraint | P2 scope note |
| $\det J_\phi > 0$ on a grid $\Rightarrow \phi$ globally injective | **not** true (local vs global) | P3 caveat |

The badge at the top refers to the **certified** configuration only. Please do
not describe the softplus barrier as a guarantee.

---

## 7. Numerical conventions changed in 1.1.0

| # | v4 (1.0.0) | 1.1.0 | why it matters |
|---|---|---|---|
| 1 | $\det J$ from raw forward differences | divided by the local grid spacing | v4 reported $h^2\det J$: the identity gives $1.008\cdot10^{-3}$ instead of $1$, so the published "min det(J)" is off by $h^2$, and the barrier's effective temperature was $\tau/h^2 \approx 4.96$ — it silently changed strength with the resolution and was numerically inert over $d\in[-1,1]$ |
| 2 | MSE averaged over the bounding box | averaged over the masked cells | the effective learning rate depended on the area of the domain (46 % active cells on the annulus vs 70 % on the L-shape) |
| 3 | $\tau = 0.005$ (unnormalised units) | $\tau = 0.05$ (physical units) | same *effective* value; now resolution independent |
| 4 | `lil_matrix` assembly | vectorised COO + SuperLU | $\sim 100\times$ faster data generation (100 samples: 1.5 s vs minutes) |

`scripts/train_real_v4.py` and `scripts/generate_real_data.py` are kept as
deprecated wrappers for citation stability; they document the differences.

---

## 8. Repository map

```
src/dagnese_fno/
├── geometry/jacobian.py    spacing-aware Jacobian calculus + injectivity certificate
├── losses/barrier.py       D'Agnese softplus barrier (+ log / hinge ablations)
├── losses/combined.py      L = MSE + alpha*H1 + lambda*barrier
├── models/spectral.py      SpectralConv2d (Fourier layer)
├── models/fno.py           FNO2d backbone (shared by baseline and DIF-FNO)
├── models/diffeo.py        DiffeomorphicMap, SpectralLinear (Lipschitz cap)
├── models/dif_fno.py       DIFFNO2d + ConditionEncoder (global conditioning)
├── metrics/errors.py       relative L2 / H1 / full H1 / Linf (masked)
├── data/masks.py           star, lshape, annulus, naca0012
├── data/darcy.py           Darcy generator + finite-volume solver
├── data/meshes.py          body-fitted NACA 0012 H-grid
├── training/trainer.py     the training loop and every reported metric
├── cli/                    dif-fno-train, dif-fno-generate-data
├── barrier_loss.py         deprecated shim (v4 import path)
└── dagnese_mesh_generator.py  deprecated shim (v4 import path)

scripts/     generate_data.py · train.py · run_benchmark.py · make_figures.py
             (+ generate_real_data.py, train_real_v4.py: deprecated wrappers)
tests/       112 tests: geometry, losses, models, metrics, solver, end-to-end
docs/        THEORY.md — the four propositions, their proofs and their scope
examples/    quickstart.py
results/     published v4 runs + results/repro (this reproduction)
figures/     final/ (paper) + repro/ (regenerated)
```

## 9. Extending

* **New domain**: add a mask to `data/masks.py` + register it in `DOMAINS`;
  add a geometry test; generate the data.
* **New barrier**: subclass the interface of `losses/barrier.py` and add it to
  `get_folding_penalty`; the training loop picks it up from `--barrier`.
* **Larger guarantees**: `--lipschitz-const` trades deformation magnitude against
  the certified bound $(1-L)^2$; `--conditioning global` is what makes
  $\kappa = 1$ (pointwise conditioning pays $\kappa = \sqrt{1+\lVert\nabla a\rVert^2}\approx 3\cdot10^2$).

See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## 10. Citation

```bibtex
@software{dagnese_dif_fno_2026,
  author    = {D'Agnese, Giovanni},
  title     = {D'Agnese DIF-FNO: Diffeomorphic Implicit Fourier Neural Operators
               with Topological Guarantees on Non-Convex Domains},
  version   = {1.1.0},
  year      = {2026},
  doi       = {10.5281/zenodo.22814077},
  url       = {https://github.com/GiovanniDAgnese-paper/DIF-FNO}
}
```

[`CITATION.cff`](CITATION.cff) carries the same metadata.

## 11. License

MIT — see [`LICENSE`](LICENSE).
