# Self-review: what was wrong, what was fixed, what is still open

This is the adversarial review of release 1.2.0, written against the repository
as it stood at 1.1.0 and against the new material.  Every item is either fixed
(with a pointer to the fix) or explicitly listed as a limitation (with a pointer
to where it is disclosed).  The rule for this document: *no claim survives
unless a test, a script or a citation backs it.*

---

## A. Defects found and fixed

### A1. A documented feature that did not exist

`CHANGELOG` and the docstring of `DagneseChartOperator` advertised
`loss_space="chart"` (the Jacobian-weighted, metric-correct MSE).  `TrainConfig`
had no such field and the trainer ignored it — a promise in the documentation
with no implementation behind it.  **Fixed**: `DiffeomorphicOperatorLoss` takes
`loss_space`, `masked_mse` takes a per-node `weight`, `TrainConfig.loss_space`
exists, the CLI exposes `--loss-space`, and
`tests/test_equidistribution.py::test_chart_loss_space_matches_the_physical_one`
verifies that the chart-space quadrature reproduces the physical MSE to 10 %
(a quadrature difference, not a different objective).

### A2. The pushforward used the determinant at the wrong point

For `alpha = 1` (a density), `pushforward` multiplied by `|det J|` evaluated at
the **reference** nodes instead of at `Φ⁻¹(x)`, i.e. by the wrong function of
space, and with the sign of the exponent of the forward map.  It is wrong by the
variation of the determinant along the chart — invisible in the scalar case
(`alpha = 0`, the only one used by the operator), so no benchmark was affected,
but the API was incorrect.  **Fixed**: the determinant is sampled through the
inverse map and the exponent is inverted; the pair is now an inverse pair for
densities as well,
`test_density_round_trip_recovers_the_field`.

### A3. The determinant could be negative for a provably positive map

Forming `det J` as `ad − bc` on a strongly stretched cell (where `|ad|` and
`|bc|` differ by up to `10^6`) loses every significant digit; symmetrising the
stencil to reduce the cancellation produced negative determinants on oscillating
meshes.  **Fixed** by construction: `det J = (2Δg/Δξ)(2Δ_v h/Δξ)`, a product of
two strictly positive numbers (P5.ii).  This is now part of the *definition* of
the chart, not an implementation detail.

### A4. A softmax is not a positivity guarantee in float32

Logits beyond a span of ~90 make a softmax underflow to exactly `0`, i.e. a
folded cell — and the training loss does not notice.  Measured on a pure-softmax
chart at parameter scale `10^2`: 24–99 % folded cells.  **Fixed** with the
mixture `(1-ε)softmax + ε/K`; the parameter-free floor then holds in the
arithmetic as well (`tests/test_monotone.py`, the adversarial sweep in
`results/study/STUDY.md`).

### A5. Two crashes on the per-sample (latent-code) chart

`trainer.evaluate` called `chart.certified_det_lower_bound(coords)` and
`chart.inverse_consistency_error(coords)` without the latent code, which raises
by design ("a code is required").  **Fixed**: both have zero-code fallbacks that
return the parameter-free floor / the identity-consistency error, and the
trainer passes `aux["code"]` when it has one.
`test_chart_operator.py::test_global_conditioning_uses_the_field` and the
`chart-cond` smoke runs cover it.

### A6. `torch.cdist` is unusable for injectivity checks

`cdist` computes `|x|² + |y|² − 2x·y`; two points `1e-5` apart at radius 1 give
exactly `0` by cancellation, so a "no two nodes collide" test built on `cdist`
passes for a completely folded grid.  **Fixed**: the tests use grid-neighbour
differences (`neighbour_gap`) and, where a distance is genuinely needed, an
explicit difference.

### A7. The v1 table's "min det J" was `h² det J`

The v1 forward-difference stencil was not divided by the grid spacing, so the
identity map reported `1.008·10⁻³` at `n = 64` instead of `1`, the published
"min det J" was off by `h²`, and the barrier's effective temperature was
`τ/h² ≈ 4.96` — it silently changed strength with the resolution.  Fixed in
1.1.0 (P4) and documented in §7 of the README.  The v1 numbers in
`results/repro` are kept for citation stability and marked as **not
comparable** with the v2 sweep (different data pipeline).

### A8. A destructive edit during this very review

While inserting Part II into `docs/THEORY.md` a text-splitting edit dropped
Part I (P1–P4) from the file.  Caught by re-reading the file header and
restored from git; the shipped file now contains P1–P8 (508 lines) and every
section is cross-referenced.  Recorded here as a process defect: any edit that
splits a document must be followed by a structural diff.

---

## B. Claims that were softened, because they were not true as stated

| old claim | status | replacement |
|---|---|---|
| "0.00 % grid folding with the D'Agnese barrier" (v1 README headline) | **soft penalty, not a guarantee**; the v4 configuration deformed the grid by 1.5 % of the domain | "the barrier makes folding expensive; only the chart makes it impossible" |
| "certified `det J > 0`" for the barrier configurations | the reported certificate is `0.0` — there is none | the certificate is reported per configuration; `0` means "no certificate" |
| "the diffeomorphism improves accuracy" (v1: asserted, no evidence) | v1 had the sign right and the evidence absent. On the 18-run, 6-geometry v2 benchmark the effect holds: `dL2 = -0.0182` (p = 4e-4, 16/18 runs, 6/6 geometries), `dH1 = -0.0413` (p < 1e-4), and — the part that makes it a result rather than a correlation — the *same chart without the equidistribution loss* improves only 11/18 (`dL2 = -0.0108`, p = 0.010), so the ablation `eq - chart` is `-0.0073` at p = 4e-4 over 17/18 runs | "measured, with caveats" (§4 of the README, §6 of the paper): pooled test, domain-level sign test (6/6, p = 0.016), the two losing runs, and the near-degenerate `min det J = 0.033` are all printed |
| "equidistribution improves the representation" | true for feature-dominated fields, **false** otherwise | the negative row (0.54×) is in the paper |
| "the certificate is tight" | a trained chart has `min det J ≈ 0.8`; the bound is `10⁻⁶` | the bound is a worst case over all parameter values, and is reported as such |

---

## C. What a reviewer will attack, and the answer

1. **"The chart is just a coupling layer."**  Yes — triangular monotone maps are
   standard in normalising flows.  The contribution is not the architecture but
   (i) the *mesh* reading: the quantities the discretisation needs
   (`Δg`, `Δ_v h`) are the parameters, so positivity is free; (ii) the explicit
   parameter-free floor; (iii) exact identity at initialisation, which makes the
   ablation honest; (iv) the equidistribution objective with its classical
   justification.  Stated in `docs/DESIGN_NOTES.md` §2.
2. **"Geo-FNO already learned a deformation."**  Yes, and it is cited.  The
   differences are: the chart maps the reference square to itself (the identity
   is in the class), injectivity is certified rather than encouraged, and the
   deformation has an objective (equidistribution) rather than only a boundary
   heuristic.
3. **"Your certificate is vacuous (`ε² = 10⁻⁶`)."**  It is a worst case, and we
   say so.  Its purpose is to make "cannot fold" a theorem; the *useful* number
   is the measured `min det J`, reported next to it everywhere.
4. **"The benchmark is small."**  True: 100 samples, 40 epochs, `N = 64`, three
   seeds, one backbone.  It is a controlled comparison, not a leaderboard entry,
   and the paper reports the full protocol, the per-run values and the two
   tests.  The measured effect (5.3 % L2, 7.0 % H1) is modest *because* the
   protocol is small: we would not extrapolate it to a leaderboard.
5. **"Seed noise swamps the differences."**  Per seed it nearly does: the spread
   (sd ≈ 0.018) is as large as the mean effect.  That is why the statistic is
   *paired*, why sixteen of eighteen runs must be negative before we say
   anything, and why the ablation — which holds the architecture, the seed and
   the budget fixed and changes only the loss — is the load-bearing evidence.
   Two runs are not negative, and they are in the table.
6. **"Equidistribution is not the right objective for PDE error."**  Agreed, and
   demonstrated (row 3 of the P7 table).  The loss is auxiliary; the paper claims
   a *bound on a representation error*, not a lower PDE error.

---

## D. Still open (disclosed, not hidden)

1. A **tight** certificate would require interval arithmetic at the trained
   parameters; the current bound is structural, not informative about the
   trained map.
2. **Per-sample charts** (`chart-cond`) are implemented and certified but the
   benchmark cannot show their value: one geometry per dataset is the wrong
   test.  A manifold-of-geometries benchmark is the right one.
3. **Composition** (`chart-deep`) is certified but its benefit is unproven.
4. The **cost** is real: ×3 at 64² on CPU, dominated by two bilinear samplings.
   A fused kernel would remove most of it; not implemented.
5. The **P8 gauge remark** has no clean convergence order, because a violent
   chart makes the transport error dominated by its smallest cells.  It is
   reported as a consistency check, not a theorem.
6. The **manuscript has not been compiled** in the development environment
   (no TeX available); it is syntax-checked (balanced environments, all
   citations and cross-references resolve) but must be built before submission.
