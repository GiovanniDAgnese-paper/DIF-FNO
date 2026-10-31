# Design notes: from DIF-FNO to the D'Agnese Chart Operator

This note records *why* the v2 object exists, what it buys, what it costs, and
what it does not buy.  It is deliberately blunt: the v1 line was published with
an over-claimed table, and the v2 line is built so that the same mistake cannot
be repeated silently.

---

## 1. The defect of v1, stated precisely

DIF-FNO v1 learns $\phi_\psi(\xi)=\xi+s\,r_\psi(\xi,z)$ and penalises
$\det J_\phi\le0$ with a softplus barrier.

Three separate problems, in increasing order of severity:

1. **A penalty is not a constraint.** For any finite $\lambda$, a large enough
   data gradient folds the grid.  Measured on v1: `dif-nobar` folds
   $12$–$22\%$ of the cells; `dif-barrier` folds $0.00\%$ — the barrier works,
   but by making folding unattractive, not impossible.
2. **The certified variant shrinks the class.** The only v1 *theorem*
   ($\det J\ge(1-\kappa L)^2$) requires $\mathrm{Lip}(r)\le L$ with
   $\kappa L<1$. With pointwise conditioning $\kappa\approx 3\cdot10^{2}$, so
   $L<3\cdot10^{-3}$: the map is confined to an infinitesimal neighbourhood of
   the identity.  The v1 certified runs deform the grid by $0.28$–$0.49$ only
   because they use *global* conditioning, and even then the certificate
   $(1-L)^2\in[0.42,0.52]$ is obtained by capping the residual's Lipschitz
   constant, which also caps the deformation.
3. **The chart had no job.** Nothing in v1 said *where* the grid should deform.
   The learned deformation was whatever the data term happened to prefer, and
   the v4 configuration settled on a displacement of $0.03$ — 1.5 % of the
   domain.  A vacuous deformation with a valid certificate is still vacuous.

v2 addresses (1) by construction, (2) by a parametrisation whose certificate
does not involve a Lipschitz constant, and (3) by a loss with a classical
meaning.

---

## 2. What is genuinely new in v2

| # | object | novel? | why |
|---|---|---|---|
| 1 | triangular chart from positive increments | **partly** | triangular/coupling maps are standard in flows (Real NVP, autoregressive flows).  New here: the *mesh* interpretation, the explicit floor $\varepsilon^2$, exact identity at initialisation, and the fact that the two diagonal entries are the quantities the discretisation needs. |
| 2 | operator evaluated on the chart | **no** | that is Geo-FNO's idea (a learned deformation to a computational space).  The difference is certification and the objective. |
| 3 | D'Agnese Equidistribution Loss | **partly** | the *criterion* is classical (de Boor; Huang–Russell).  New: the exact functional (normalised variance of monitor mass), its characterisation and its use as a differentiable objective for a *neural* mesh, with a demonstration that gradient descent recovers the classical mesh. |
| 4 | the combination | **yes** | a certified chart + an equidistribution objective + an operator that is exactly the baseline at $t=0$. |

We should not claim 1 or 3 as new mathematics.  The contribution is the
*construction* (a class in which the guarantee is free) and the *empirical
programme* (what it costs, what it buys, when it does nothing).

---

## 3. Design decisions and their evidence

### 3.1 Positive increments, not a softplus or a tanh

* Softplus: $\mathrm{softplus}(-100)=0$ in float32 → folded cell.
* Pure softmax: underflows at logit span $\gtrsim90$ → measured $24$–$99\%$ folded
  cells at parameter scale $10^2$.
* Mixture $(1-\varepsilon)\mathrm{softmax}+\varepsilon/K$ with $\varepsilon=10^{-3}$:
  zero folded cells in every adversarial row, and exact identity at
  initialisation preserved.
* $\varepsilon=10^{-6}$ is *mathematically* fine and *numerically* dead: the
  cumulative sum absorbs the increment ($O(\varepsilon_{32})$ absolute error per
  term) and the knots come back with a repeated value — a folded cell created by
  arithmetic.

### 3.2 $\det J$ as a product, never $ad-bc$

A strongly stretched cell has $|ad|$ and $|bc|$ differing by up to $10^6$; the
difference loses every digit and can come out negative for a provably
orientation-preserving map.  Symmetrising the determinant (the "obvious fix")
is worse: it returned negative values on oscillating meshes in our first
implementation.  The product form is part of the definition, not an
implementation detail.

### 3.3 The chart is the identity at initialisation

This is what makes the ablation honest: at $t=0$ the operator *is* the FNO
($3\cdot10^{-8}$ from interpolation round-off).  Any difference after training is
attributable to the chart, not to extra capacity or a different init.

### 3.4 Equidistribution as an auxiliary loss, weighted

P7 gives the chart a target, but the target is a *bound* on a representation
error, not the PDE error, and §P7's table shows it can be counterproductive when
the monitor is already uniform ($0.54\times$).  Therefore:

* $\lambda_{\mathrm{eq}}$ is a hyperparameter and its default in the benchmark
  variants is explicit (`0` for `chart-fixed`, `10` for `chart-eq`);
* the identity is always in the class, so a chart that does not help can be
  switched off by the data term;
* the reported `eq_residual_pct` diagnoses whether the term is active at all.

### 3.5 What we chose *not* to do

* No per-sample chart in the main table.  It costs an encoder, doubles the
  inference graph and its benefit is unproven; it exists (`chart-cond`) and is
  reported separately.
* No multi-layer chart in the main table, for the same reason (`chart-deep`).
* No claim of a lower PDE error.  The v1 line already learned that a valid chart
  is not automatically a more accurate one; the v2 benchmark measures it again
  rather than assuming it.

---

## 4. Cost

At $N=64$, batch $10$, two CPU threads (median of a steady-state training step,
`scripts/bench_timing.py`): FNO $47$ ms/step, DAO $131$ ms/step
($\times2.8$), DAO with a per-sample chart $143$ ms/step ($\times3.0$).
Parameters:
$334\,737$ (FNO), $336\,339$ ($+1\,602$, one-layer chart), $341\,275$ (global
conditioning).  The overhead is two bilinear samplings and the knot
construction; it is the price of a chart that cannot fold.

Training time per run at $N=64$, 40 epochs, 100 samples: FNO $\approx39$ s,
DAO $\approx68$ s on two threads.

---

## 5. Open questions (honest list)

1. **When does adaptivity pay?**  P7(iv) is a bound; the empirical answer in
   §P7 is "for fields with a localised layer, by 2–4×, in representation error —
   not necessarily in PDE error".
2. **Per-sample charts.**  A mesh model should help when the family of solutions
   has heterogeneous features.  Our star/L-shape/annulus benchmark is
   homogeneous by construction (one geometry, one coefficient distribution), so
   it is the wrong test; a manifold-of-geometries benchmark is the right one.
3. **The certificate is loose by design.**  A trained chart has
   $\min\det J\approx0.8$ but the certificate reports $10^{-6}$.  A *tight*
   certificate would need to be evaluated at the trained parameters, which is a
   different (and more useful) object: $\min\det J$ over the training
   distribution, certified by interval arithmetic.  Not implemented.
4. **Composition.**  Three composed charts should be more expressive than one;
   we have not demonstrated a case where that matters.
5. **Non-tensor-product charts.**  The separable structure is what gives the
   theorem; a general triangular map (with $h$ depending on $u$ arbitrarily) is
   also covered, but a genuinely two-dimensional coupling (checkerboard) is not
   implemented.

---

## 6. Why six geometries, and why the ablation is the load-bearing run

The first version of the benchmark had four geometries and twelve paired runs.
The pooled paired test gave `p = 0.027` for `chart-eq`, but the runs are *nested*
in geometries: four independent units, four of four negative, sign test
`p = 0.0625`. That is a result you cannot defend at a conference, and no amount
of extra seeds fixes it — seeds are not independent units either, they share the
geometry.

Two more geometries (a **crack** — a thin slit from the wall with a singular tip
— and a **plate with two circular holes** — multiply connected, two obstacles)
were added for exactly that reason: six of six negative means gives a sign test
of `p = 0.0156`, and the two new geometries turned out to carry the largest and
tightest effects (`-0.031 ± 0.018` and `-0.034 ± 0.005`).

The second design decision: the headline comparison (`chart-eq` vs FNO) is *not*
the experiment that identifies the mechanism, because the chart adds parameters,
changes the input parameterisation and changes the sampling. The identifying
experiment is `chart-eq` vs `chart-fixed`: identical architecture, identical
parameter count, identical seed, identical budget, and the only difference is the
equidistribution term. That comparison is `-0.0073` in L2 (`p = 4e-4`, 17 of 18
runs) and `-0.0211` in H1 (18 of 18). It is printed as a row *inside* the paired
table, so a reader cannot see the headline without seeing the ablation.

Known weakness, stated rather than hidden: `λ_eq = 10` drives the learned chart
to `min det J ≈ 0.033` on four of the six geometries. Nothing folds (the floor is
`ε²`), but a cap on the term, or a penalty on the log-determinant, is the natural
next regularisation, and the honest statement is that we have not implemented it.
