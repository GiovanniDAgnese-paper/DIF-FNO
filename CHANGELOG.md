# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.org/en/1.1.0/) and the project uses
semantic versioning.

## [1.2.0] -- 2026-10-31

The **D'Agnese Chart Operator**: the chart stops being something to penalise and
becomes a class in which folding cannot happen.

### Added

- **`dagnese_fno.geometry.DagneseTriangularChart`** — a constructive
  diffeomorphism of the reference square,
  `Phi(xi) = (g(xi1), h(xi1, xi2))`, generated from strictly positive
  increments.  Global injectivity, `det J > 0` as a product of positive factors,
  and a closed-form inverse hold *by construction* for every parameter value
  (`docs/THEORY.md` P5.i–vi).
  * `certified_det_lower_bound` — the exact minimum over the computed knots.
  * `parameter_free_det_floor(n)` — the theorem's bound `eps^2`, independent of
    the parameters, the resolution and the deformation size.
  * `inverse_at(points)` — the exact inverse at **arbitrary** points (not only on
    a square grid), which is what an adaptive solver needs to locate a point.
  * `inverse_consistency_error`, `chart_det_consistency` — diagnostics.
- **`dagnese_fno.geometry.transport`** — `pullback`, `pushforward`,
  `sample_field`, `jacobian_weight`, `convergence_rate` (bilinear, batched,
  differentiable).
- **`dagnese_fno.losses.DagneseEquidistributionLoss`** — the normalised variance
  of the monitor mass per cell (`docs/THEORY.md` P7): non-negative, zero exactly
  on equidistributing meshes, scale free, and a bound on the worst cell through
  `sqrt(N_c * L)`.  Includes `gradient_monitor` and a `residual()` diagnostic.
- **`dagnese_fno.models.DagneseChartOperator`** (`"chart"`, `"dao"`,
  `"chart-fno"`, `"cpo"`) — an FNO evaluated on the chart, **exactly equal to the
  FNO at initialisation** (`3e-8`), optionally conditioned on the input field so
  that the chart becomes a per-sample mesh model.  `forward_aux` returns the
  chart, the exact `det`, the transported field, the inverse and the certified
  bound.
- **`loss_space="chart"`** — metric-corrected MSE
  `int (u - u_theta)^2 |det J| dxi`, the natural loss on a deformed mesh.
- **Training**: `--model chart`, `--chart-hidden/-layers/-fourier/-activation/
  -conditioning/-alpha/-min-increment`, `--lambda-eq`, `--eq-beta/-floor/-smooth`;
  `evaluate` reports the certified bound, the condition number, the inverse error
  and the equidistribution residual.
- **`scripts/run_chart_study.py`** — the deterministic verification of P5–P8
  (no training, no sampling): adversarial parameter sweep, transport
  convergence, a chart learned by gradient descent on the equidistribution loss
  alone, and a chart-gauge consistency check.  Writes `results/study/STUDY.{json,md}`
  and four figures.
- **`scripts/make_paper_table.py`** — turns benchmark results into the paper's
  LaTeX table, the repository markdown table and a paired-difference figure, so
  the three cannot drift apart.
- **`paper/`** — the manuscript (`dagnese-chart-operator.tex`, `references.bib`)
  with the full proofs and the verification tables.
- **Tests**: `tests/test_monotone.py` (P5.i–vi, conditioning, gradients,
  resolution consistency), `tests/test_transport.py` (bilinear exactness,
  convergence order, Jacobian-weighted integral), `tests/test_equidistribution.py`
  (characterisation, scale invariance, de Boor optimality in 1-D),
  `tests/test_chart_operator.py` (identity at init, admissibility under random
  and optimised parameters, resolution change).

### Changed

- `run_benchmark.py` gains `--eval-every` and `--threads`; new variants
  `chart-fixed`, `chart-eq`, `chart-cond`, `chart-deep`.
- `TrainConfig.forward_batch` now returns a 6-tuple ending in `aux`; the
  chart-aware `evaluate` uses it (a chart with a latent code cannot be called
  without the code).
- Documentation: `docs/THEORY.md` gains Part II (P5–P8) with the measured
  verification tables and a longer "what is not claimed" section;
  `docs/DESIGN_NOTES.md` records the design decisions, their evidence and the
  open questions.

### Results (v2 benchmark, 6 geometries x 3 seeds, paired)

- The chart operator with the D'Agnese Equidistribution Loss improves the
  relative L2 error on every geometry. Pooled over 18 paired runs:
  `dL2 = -0.0182` (-8.6 % in the mean of per-run ratios; t = -4.36, p = 4e-4,
  Wilcoxon 3e-4, 16/18 runs, 6/6 geometries, domain-level sign test p = 0.016)
  and `dH1 = -0.0413` (-9.7 %, p < 1e-4).
- The ablation isolates the mechanism: the chart *alone* gives `-0.0108`
  (p = 0.010); adding the equidistribution loss at the identical budget gives a
  further `-0.0073` in L2 on 17/18 runs (p = 4e-4) and `-0.0211` in H1 on 18/18.
- Zero folded cells, `min det J >= 0.033`, certified bound `>= 0.020`, exact
  inverse error `<= 1.3e-6` on every run; the chart compresses cells by up to
  30x where the monitor asks for it.
- Cost: x2.8 wall clock per training step at 64^2 on CPU (`scripts/bench_timing.py`).
- Two new benchmark geometries: `slit` (a crack from the wall, singular tip) and
  `double_hole` (a plate with two circular holes). Both are covered by
  `tests/test_data.py`.

### Fixed

- `det J` is never formed as `ad - bc` for a chart: on a strongly stretched cell
  the two products differ by up to `1e6` and the difference can be negative for a
  provably orientation-preserving map.
- A pure softmax on the increments underflows in float32 above a logit span of
  ~90 and silently folds the grid (measured: 24–99 % of cells at parameter scale
  `1e2`); the mixture `(1-eps) softmax + eps/K` removes the failure.
- `evaluate` no longer calls the chart without its latent code (crashed for
  `chart_conditioning="global"`).

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
