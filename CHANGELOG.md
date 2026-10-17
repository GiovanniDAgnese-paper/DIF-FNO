# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.org/en/1.1.0/) and the project uses
semantic versioning.

## [1.1.0] -- 2026-10-17

A full re-engineering of the repository around three goals: make the
topological claim *provable* rather than empirical, make the numbers
reproducible, and make the code testable.

### Added

- **Certified injectivity.** `dagnese_fno.models.diffeo.SpectralLinear` caps the
  spectral norm of every layer of the deformation network, which enforces
  `Lip(r) <= L` and therefore the *theorem* `det J_phi >= (1 - kappa*L)^2 > 0`
  (`dagnese_fno.geometry.certified_det_lower_bound`, `docs/THEORY.md` P3).
- **Global conditioning** (`conditioning="global"`): a per-sample latent code
  `z(a)` fed to the deformation network instead of the pointwise field, which is
  what makes the certificate usable (`kappa = 1` instead of
  `sqrt(1 + max||grad a||^2) ~ 3e2`).
- **`dagnese_fno.geometry`**: a spacing-aware, resolution-independent Jacobian
  calculus (central differences + second-order one-sided boundaries, support for
  non-uniform/curvilinear grids), `jacobian_det_autograd`, singular values,
  folding statistics and displacement statistics.
- **`dagnese_fno.metrics`**: relative `L2`, `H1` seminorm, full `H1` and
  `L-inf` errors, all masked and spacing-consistent.
- **`dagnese_fno.losses`**: the D'Agnese softplus barrier plus the classical log
  barrier and the ReLU hinge (ablation baselines), and
  `DiffeomorphicOperatorLoss` assembling `MSE + alpha*H1 + lambda*barrier`.
- **`dagnese_fno.data`**: vectorised (COO/SuperLU) Darcy assembly ~100x faster
  than the previous `lil_matrix` loop, a `naca0012` domain, and a real
  body-fitted H-grid generator (`data/meshes.py`) replacing the previously
  inert `dagnese_mesh_generator`.
- **`dagnese_fno.training`**: a single, configurable training loop recording the
  full history, the `det J` statistics, the displacement, the Lipschitz bound
  and the certified bound; deterministic seeding, checkpointing, early stopping,
  optional W&B.
- **CLI entry points** `dif-fno-train` / `dif-fno-generate-data` and
  `scripts/run_benchmark.py` (multi-domain, multi-variant, multi-seed sweep with
  markdown/LaTeX consolidation) and `scripts/make_figures.py`.
- **Test suite**: 112 tests (`tests/`) covering the Jacobian calculus, the
  barrier properties, the models, the metrics, the solver (manufactured-solution
  convergence and patch test) and end-to-end training.
- **Infrastructure**: `LICENSE` (MIT), `pyproject.toml` (PEP 621 metadata,
  ruff/pytest/coverage config), `Makefile`, GitHub Actions CI, `docs/THEORY.md`,
  `CHANGELOG.md`, `CONTRIBUTING.md`.

### Changed

- `det J` is now normalised by the grid spacing: `det J = 1` for the identity at
  every resolution. The previous raw forward-difference stencil returned
  `h^2 det J`, which (i) made the reported "min det(J)" smaller by a factor
  `h^2 = 1.008e-3` at `n = 64` and (ii) changed the *effective* temperature of
  the barrier to `tau/h^2`, i.e. made the regulariser resolution dependent and
  numerically inert (`docs/THEORY.md` P4).
- The MSE term is averaged over the masked cells (`normalize_by_mask=True`),
  removing the domain-area-dependent effective learning rate of the v4 loss
  (the annulus activates ~46% of the cells, the L-shape ~70%).
- `tau` is expressed in physical units; the default is `0.05` (v4 used `0.005`
  in *unnormalised* units, i.e. `~4.96` in physical units).

### Fixed

- `tests/test_jacobian.py` was vacuous: it differentiated only the diagonal
  terms of a map whose off-diagonal terms are zero, so `det J` was identically
  `1` and the test could never fail. It is now a full determinant check with
  exact (rotation/shear/scaling/reflection) and autodiff-verified cases.
- `dagnese_mesh_generator.generate_dagnese_naca0012_mesh` computed the NACA
  thickness distribution and then discarded it; the new implementation builds a
  body-fitted, folded-free H-grid.
- `setup.py` duplicated the metadata now declared in `pyproject.toml`, and the
  repository URL pointed at a non-existent capitalisation of the owner.
- `CITATION.cff`/`__init__.py`/`setup.py` disagreed on the version (8.0.0 / 1.0.0 / 1.0.0).

### Deprecated

- `dagnese_fno.barrier_loss` and `dagnese_fno.dagnese_mesh_generator` (shims
  that emit a `DeprecationWarning`), `scripts/generate_real_data.py` and
  `scripts/train_real_v4.py` (thin wrappers preserving the historical CLI).

## [1.0.0] -- 2026-09-17

- Initial release accompanying the DIF-FNO v4 results (Star, L-Shape, Annulus;
  Darcy flow; 60 epochs).
