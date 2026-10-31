# Theory and guarantees

This note states precisely what is proved, what is merely encouraged, and what is
only observed for the objects implemented in `dagnese_fno`.  Every statement
below is exercised by the test suite (`tests/`) under the same name.

---

## 0. Setting

Let $\Omega_{\mathrm{ref}} = [-1,1]^2$ be the reference square, discretised on a
structured grid $\xi_{ij}$ with NumPy's ``'ij'`` ordering, so that tensor axis 0
carries spatial component 0 and axis 1 component 1.  DIF-FNO learns

$$
\phi_\psi : \Omega_{\mathrm{ref}} \to \Omega_{\mathrm{comp}}, \qquad
\phi_\psi(\xi) = \xi + s \, r_\psi(\xi, z), \qquad
J_\phi[a,b] = \frac{\partial \phi_a}{\partial \xi_b},
$$

and then evaluates a Fourier Neural Operator $\mathcal F_\theta$ on the deformed
coordinates: $u = \mathcal F_\theta(\phi_\psi(\xi), a(\xi))$.

**Admissibility.** A deformation is admissible iff it is an orientation
preserving diffeomorphism, i.e. $\phi \in C^1$ is injective and

$$
\det J_\phi(\xi) > 0 \quad \forall \xi \in \Omega_{\mathrm{ref}} .
$$

$\det J_\phi \le 0$ somewhere is exactly *grid folding*: the computational cells
overlap and the operator is no longer evaluated on a valid chart.

---

## P1 — The model starts admissible

**Claim.** With the last layer of $r_\psi$ initialised to zero,
$\phi_\psi = \mathrm{Id}$, hence $J_\phi = I$ and $\det J_\phi \equiv 1$ at
$t = 0$.

*Proof.* $r_\psi = W_L \sigma(\cdots)$ with $W_L = 0$ and $b_L = 0$ gives
$r_\psi \equiv 0$ identically, so $\phi_\psi(\xi) = \xi$. $\square$

**Consequence.** Optimisation starts inside the admissible set; every later
violation is produced by the training dynamics, not by the initialisation.
Test: `test_zero_init_gives_the_identity`.

---

## P2 — Properties of the D'Agnese barrier

$$
\mathcal L_{\text{barrier}}(d) = \tau \, \mathrm{softplus}\!\left(\frac{\varepsilon - d}{\tau}\right),
\qquad d = \det J_\phi,\quad \tau > 0,\quad \varepsilon \ge 0 .
$$

Write $z = (\varepsilon - d)/\tau$. Since
$\mathrm{softplus}(z) = \log(1 + e^{z})$:

1. **Smoothness.** $\mathcal L \in C^\infty(\mathbb R)$; in particular there is
   no kink at $d = \varepsilon$ (unlike a hinge) and no pole (unlike a log
   barrier). $\displaystyle \frac{\partial \mathcal L}{\partial d}
   = -\sigma(z) \in (-1, 0)$ with $\sigma$ the logistic function, and
   $\displaystyle \frac{\partial^2 \mathcal L}{\partial d^2}
   = \tfrac{1}{\tau}\sigma(z)\sigma(-z) \le \tfrac{1}{4\tau}$, so the Hessian of
   the objective stays bounded.
2. **Monotonicity.** $\partial_d \mathcal L < 0$: the penalty increases as the
   cell folds, and it is the only term that does.
3. **No dead zone and no saturation.** The gradient is strictly negative for
   every finite $d$ (it decays like $e^{-(d-\varepsilon)/\tau}$ in the admissible
   region) and tends to $-1$ as $d \to -\infty$.  A hinge
   $\max(0, \varepsilon - d)$ has an exactly zero gradient wherever
   $d > \varepsilon$ (nothing pushes the map away from the fold) and a kink at
   $d = \varepsilon$; a log barrier $-\tau\log(d/\tau)$ has gradient
   $-\tau/d$, which explodes and forces trust-region safeguards.
4. **Linear growth.** $\mathcal L(d) = (\varepsilon - d) + \tau e^{-(\varepsilon - d)/\tau} + o(\cdot)$,
   i.e. $\mathcal L(d) \to \varepsilon - d$ with unit slope as
   $d \to -\infty$: the penalty grows linearly, so a deeply folded cell is not
   "rewarded" by a vanishing gradient, and it is never infinite, so the update
   cannot blow up.
5. **Neutral in the admissible region.** For $d \ge \varepsilon + 20\tau$ the
   penalty is below $10^{-9}$: once the map is safely injective the regulariser
   stops interfering with the data term.

Tests: `tests/test_barrier.py` (one test per item).

> **Scope.** P2 makes the barrier a *good* penalty. It does **not** make
> $\det J_\phi > 0$ a theorem: for any finite $\lambda_{\text{barrier}}$ a
> sufficiently large data gradient can still fold the grid.  The only guarantee
> in this library is P3.

---

## P3 — Certified injectivity (Lipschitz-constrained residual)

**Claim.** Let $\phi = \mathrm{Id} + r$ with $r$ differentiable, let the inputs
of $r$ depend on $\xi$ through a map whose Jacobian has spectral norm
$\|D\|_2 \le \kappa$, and let $\mathrm{Lip}(r) \le L$. If $\kappa L < 1$ then

$$
\sigma_{\min}(J_\phi) \ge 1 - \kappa L > 0
\quad\text{and}\qquad
\det J_\phi \ge (1 - \kappa L)^2 > 0 .
$$

*Proof.* $J_\phi = I + \partial_\xi r$ and, by the chain rule,
$\|J_\phi - I\|_2 = \|\partial_z r \, D\|_2 \le \mathrm{Lip}(r)\,\|D\|_2 \le \kappa L$.
For any matrix $A$ and any unit vector $v$,
$\| (I+A)v \| \ge 1 - \|A\|_2$, hence
$\sigma_{\min}(J_\phi) \ge 1 - \kappa L$.  In $\mathbb R^2$,
$\det J_\phi = \sigma_1 \sigma_2 \ge \sigma_{\min}^2$. $\square$

**How $L$ is enforced.** For $r = s\, W_L \sigma \cdots \sigma W_1$ with
$1$-Lipschitz activations ($\tanh$), $\mathrm{Lip}(r) \le s \prod_i \|W_i\|_2$.
`SpectralLinear` caps each factor to $\gamma = (L/s)^{1/K}$ through a one-sided
spectral normalisation
$W_{\mathrm{eff}} = W \min(1, \gamma/\sigma(W))$, so
$\sigma(W_{\mathrm{eff}}) \le \gamma$ at all times and therefore
$\mathrm{Lip}(r) \le s \gamma^K = L$.  The cap is one-sided on purpose: a zero
weight (the initialisation of P1) is left untouched instead of being amplified
by $1/\varepsilon$.

**How $\kappa$ is computed.**
* *global conditioning* $r(\xi, z(a))$: $z$ does not depend on $\xi$, so
  $D = I_2$ and $\kappa = 1$;
* *pointwise conditioning* $r(\xi, a(\xi))$: the input is
  $(\xi, a(\xi)) \in \mathbb R^3$ and
  $D = \begin{bmatrix} I_2 \\ \nabla a^\top \end{bmatrix}$, so
  $\|D\|_2 = \sqrt{1 + \|\nabla a\|^2}$ and one must take
  $\kappa = \sqrt{1 + \max_\xi \|\nabla a(\xi)\|^2}$.

  **This matters:** for the log-normal coefficients used in the benchmark
  ($a \in [e^{-2}, e^{2}]$ varying on a grid with $h = 2/63$) the measured
  $\max\|\nabla a\|$ is of order $10^2$, hence $\kappa \approx 3\cdot 10^2$ and a
  certificate would require $L < 3\cdot10^{-3}$ — i.e. a *provably*
  folding-free pointwise map can only be an infinitesimal perturbation of the
  identity.  Global conditioning removes that obstruction ($\kappa = 1$) and is
  the configuration used by the `dif-certified` variant.

Tests: `test_spectral_mode_enforces_the_lipschitz_bound`,
`test_lipschitz_bound_survives_training`,
`test_certified_model_never_folds_empirically`,
`test_certificate_is_sharp_for_a_linear_map`.

---

## P4 — Scale consistency (and the defect it fixes)

$\det J_\phi$ is a derivative.  A forward-difference stencil that is *not*
divided by the grid spacing returns $h^2 \det J_\phi$ on a uniform grid, so:

* the identity map reports $\det J = h^2 \approx 10^{-3}$ at $n = 64$ instead of
  $1$ — the quantity called "min det(J)" in the v4 tables is off by a factor
  $h^2 = 1.008 \cdot 10^{-3}$, and its *reported* value $3.45\cdot10^{-3}$
  corresponds to $\det J \approx 3.4$, i.e. a strongly area-expanding map, not a
  near-identity one;
* the barrier becomes resolution dependent: with $d_{\text{raw}} = h^2 d$,

  $$
  \tau\,\mathrm{softplus}\!\left(-\frac{h^2 d}{\tau}\right)
  = h^2\;\underbrace{\left[\frac{\tau}{h^2}\,\mathrm{softplus}\!\left(-\frac{d}{\tau/h^{2}}\right)\right]}_{\text{barrier with temperature } \tau/h^{2}}
$$

  so the *effective* temperature is $\tau_{\text{eff}} = \tau/h^{2} \approx 4.96$
  and the whole penalty is multiplied by $h^{2}$.  Over the physically relevant
  range $d \in [-1, 1]$ the penalty then varies by only $\sim 10^{-3}$ and its
  gradient by $\sim 5\cdot 10^{-4}$: the regulariser is numerically inert, and
  changing $n$ silently changes its strength.

`dagnese_fno.geometry.jacobian` therefore divides every difference by the
*local* increment (central differences in the interior, second-order one-sided
on uniform boundaries), so that `jacobian_det(identity) == 1` **at every
resolution** and $\tau$ is a length expressed in physical units.

Tests: `test_identity_has_unit_determinant_at_every_resolution`,
`test_result_is_independent_of_the_resolution`,
`test_non_uniform_grid_uses_local_spacing`.

---

---

# Part II — the D'Agnese Chart Operator

Part I is about *penalising* a defective chart.  Part II removes the defect:
the chart is a member of a class in which $\det J>0$ and injectivity are
*structural*, and it is trained by an objective that gives it a well-defined
job.  The v1 objects (P1–P4) remain the ablation baseline.

## 0'. Setting

Write $\hat u = (u,v) \in [0,1]^2$ for the *normalised* reference coordinates of
an $N\times N$ grid, $\Delta u = 1/(N-1)$, and $\Delta \xi = 2/(N-1)$ for the
physical grid spacing.  A chart is a map $\Phi : [-1,1]^2 \to [-1,1]^2$.
The transported field is

$$
\tilde a = P_\Phi a = a \circ \Phi, \qquad
P_\Phi^{-1} \tilde a = \tilde a \circ \Phi^{-1},
$$

and the operator is $\;u = T^{\Phi}_{\theta}(a) = P^{-1}_\Phi\,
\mathcal F_\theta\!\left(\Phi, P_\Phi a\right)$, where $\mathcal F_\theta$ is the
same FNO as in Part I, evaluated on the chart nodes.

---

## P5 — The D'Agnese chart: an exact, globally injective, parameter-free chart

**Definition.** A *D'Agnese triangular chart* is

$$
\Phi(\xi_1,\xi_2) = \big(2\,g(\hat u) - 1,\; 2\,\hat h(\hat u, v) - 1\big),
$$

where $g : [0,1] \to [0,1]$ is continuous, piecewise linear on the uniform grid
with $g(0)=0$, $g(1)=1$, and for every $\hat u$, $\hat h(\hat u,\cdot)$ is
strictly increasing piecewise linear with $\hat h(\cdot,0)=0$,
$\hat h(\cdot,1)=1$; $\hat h$ is bilinear in $(\hat u,v)$ between grid rows.

Both are generated from *strictly positive* increments,

$$
w_i = (1-\varepsilon)\,\mathrm{softmax}(z)_i + \varepsilon/K, \qquad
K = N-1,\;\; 0<\varepsilon<1,
$$

by a cumulative sum, where $z$ is the output of a small MLP.  Positivity is
therefore a property of the parametrisation, not of the optimiser.

**P5.i (Global injectivity).**  If $\Phi(\xi)=\Phi(\xi')$ then $g(\hat u)=g(\hat u')$,
and $g$ strictly increasing gives $\hat u=\hat u'$; then
$\hat h(\hat u,v)=\hat h(\hat u,v')$ and $\hat h(\hat u,\cdot)$ strictly
increasing gives $v=v'$.  Hence $\Phi$ is injective on the closed square, not
merely a local diffeomorphism.  $\;\square$

**P5.ii (Orientation, exactly).**  On the cell $C_{ij}$ the map is the bilinear
interpolant of its four corner images, so $J_\Phi$ is lower triangular,

$$
J_\Phi = \begin{pmatrix} \dfrac{2\,\Delta g_i}{\Delta\xi} & 0\\[6pt]
\partial_{\xi_1}\Phi_2 & \dfrac{2\,\Delta_v h_{ij}}{\Delta\xi}\end{pmatrix},
\qquad
\det J_\Phi = \frac{4\,\Delta g_i\,\Delta_v h_{ij}}{\Delta\xi^{2}} > 0 ,
$$

and it is computed as a *product*, never as $ad-bc$: the sign cannot be destroyed
by cancellation even when $|ad|$ and $|bc|$ are $10^{6}$ apart.  $\;\square$

**P5.iii (Exact inverse).**
$\Phi^{-1}(y_1,y_2) = \big(g^{-1}(y_1),\; \hat h(g^{-1}(y_1),\cdot)^{-1}(y_2)\big)$:
two monotone 1-D inversions in closed form, no iteration and no Newton step.
`DagneseTriangularChart.inverse` evaluates it on a grid and
`DagneseTriangularChart.inverse_at` at *arbitrary* points, which is what an
adaptive solver needs in order to locate a physical point in the mesh.

**P5.iv (A parameter-free floor).**  Because every increment is at least
$\varepsilon/K$ and $2/(K\Delta\xi) = 1/(\Delta\xi K)\cdot 2$,

$$
\min \Delta g_i \ge \frac{\varepsilon}{K}, \qquad
\min_{ij}\Delta_v h_{ij} \ge \frac{\varepsilon}{K}
\;\Longrightarrow\;
\det J_\Phi \;\ge\; \Big(\frac{2\varepsilon}{K\,\Delta\xi}\Big)^{2} = \varepsilon^{2},
$$

independently of $N$, of $\theta$, and of the size of the deformation.

Three numbers from the v1 line show what this replaces:

* with pointwise conditioning $\kappa \approx 3\cdot10^{2}$, so the P3
  certificate $(1-\kappa L)^2$ is positive only for
  $L < 3\cdot10^{-3}$ — an infinitesimal perturbation of the identity.  Getting a
  usable $\kappa = 1$ required switching to *global* conditioning;
* the v1 barrier configurations (`dif-barrier`, `dif-barrier-large`) report a
  certified bound of exactly $\mathbf{0}$: there is no certificate at all,
  because a softplus penalty is not a constraint (P2 scope note);
* the v1 certified configuration (`dif-certified`) does report
  $(1-L)^2 \in [0.42, 0.52]$, but only because it *caps* $L$, which also caps
  the deformation it can represent.

The D'Agnese chart reports $\varepsilon^2 = 10^{-6} > 0$ for **every** parameter
value with **no** constraint on the deformation.  $\;\square$

**P5.v (Identity at initialisation).**  A zeroed output layer gives constant
logits, hence a uniform softmax and $\Phi = \mathrm{Id}$ to the last bit; the
mixture $\varepsilon/K$ preserves exactness of this statement for every
$\varepsilon$.  This is what makes the operator a *strict superset* of the
baseline: at $t=0$ it computes the FNO output bit-for-bit (measured: $3\cdot10^{-8}$).

**P5.vi (Closure).**  Compositions of D'Agnese charts are D'Agnese charts
(each is injective and orientation preserving; the second layer is evaluated at
the images of the first by bilinear interpolation, which *is* the piecewise
bilinear map it represents).  The class contains the identity and is closed under
composition: a monoid of admissible charts, not a ball around the identity.

> **Why the mixture $\varepsilon/K$?**  A pure softmax is strictly positive in
> exact arithmetic but *underflows to zero* in float32 once the logits span
> $\sim 90$: the measured fold rate of a pure-softmax chart at parameter scale
> $10^{2}$ was 24–99 %.  The floor $\varepsilon = 10^{-3}$ keeps every increment
> above $\sim 250$ ulps at $N=64$, so the cumulative sum cannot absorb it and
> cannot produce a repeated knot.  The price is a cap $K/\varepsilon \sim 6\cdot
> 10^{4}$ on the cell-size ratio, far outside anything a learned chart needs.

**Verification** (`scripts/run_chart_study.py`, `results/study/STUDY.md`).
Three random inits per parameter scale, $N=32$, one layer, 32 hidden units:

| parameter scale | $\min\det J$ | folded cells | certified lb | parameter-free floor |
|---|---|---|---|---|
| 0 (identity) | 1.000 | 0.00 % | 1.000 | $10^{-6}$ |
| 0.1 | 0.886 | 0.00 % | 0.880 | $10^{-6}$ |
| 1 | $1.4\cdot10^{-6}$ | 0.00 % | $1.2\cdot10^{-6}$ | $10^{-6}$ |
| 3 | $9.98\cdot10^{-7}$ | 0.00 % | $9.98\cdot10^{-7}$ | $10^{-6}$ |
| 10 | $9.97\cdot10^{-7}$ | 0.00 % | $9.97\cdot10^{-7}$ | $10^{-6}$ |
| 30 | $9.97\cdot10^{-7}$ | 0.00 % | $9.97\cdot10^{-7}$ | $10^{-6}$ |
| $10^{2}$ | $9.99\cdot10^{-7}$ | 0.00 % | $9.99\cdot10^{-7}$ | $10^{-6}$ |
| $10^{3}$ | $9.99\cdot10^{-7}$ | 0.00 % | $9.99\cdot10^{-7}$ | $10^{-6}$ |

The certificate is *tighter* than the parameter-free floor at moderate scales and
equals it in the adversarial limit; the fold rate is zero in every row.  The
parameter-free floor $(2\varepsilon/(K\Delta\xi))^2$ is a statement about $\theta$;
in float32 the *computed* knots give $9.98\cdot10^{-7}$, a relative deficit of
$2\cdot10^{-3}$ caused by the accumulation of $O(\varepsilon_{32})$ absolute
errors in the cumulative sum.  Both numbers are reported: the theorem is about
the parametrisation, the measurement is about the arithmetic.  Tests:
`tests/test_monotone.py` (P5.i–vi, conditioning, gradients, resolution
consistency, bad inputs).

---

## P6 — The round trip is second order

**Claim.**  Let $f \in C^2(\Omega)$ and let $h$ be the largest chart cell
diameter.  Then

$$
\| P^{-1}_\Phi P_\Phi f - f \|_{\infty} \;\le\; C\,h^{2}\,\|D^{2} f\|_{\infty},
$$

where $C$ depends on the cell aspect ratio only.  Indeed $P^{-1}_\Phi P_\Phi f$ is
the piecewise-bilinear interpolant of $f$ on the chart cells, and a bilinear
interpolant reproduces linear functions exactly with the classical
$\tfrac{1}{8}h^2\|D^2f\|_\infty$ bound; pulling back and pushing forward only
re-evaluates the same interpolant at different points.

**Verification.**  One random chart ($\text{scale}=0.4$), a smooth field
$\sin 3x\cos 2y + x/2$:

| $N$ | $\|P^{-1}P f - f\|_\infty$ | rms | $\|\Phi^{-1}\Phi-\mathrm{Id}\|_\infty$ |
|---|---|---|---|
| 32 | $5.60\cdot10^{-2}$ | $1.23\cdot10^{-2}$ | $4.2\cdot10^{-7}$ |
| 64 | $1.43\cdot10^{-2}$ | $3.16\cdot10^{-3}$ | $6.6\cdot10^{-7}$ |
| 128 | $3.46\cdot10^{-3}$ | $8.15\cdot10^{-4}$ | $6.6\cdot10^{-7}$ |
| 256 | $9.04\cdot10^{-4}$ | $2.04\cdot10^{-4}$ | $1.2\cdot10^{-6}$ |

Observed order **1.97 (rms) / 1.99 (max)**.  The inverse is exact to
$10^{-6}$ at every resolution — in particular it does *not* degrade with $N$,
because it is a closed-form inversion of the same knots and not a numerical
solve.  For a field with an unresolved internal layer the observed order drops
to $1.7$: the transport is not the bottleneck there, the discretisation of the
field is.  Tests: `tests/test_transport.py`, `tests/test_monotone.py`
(`test_round_trip_*`, `test_convergence_order_*`).

---

## P7 — The D'Agnese Equidistribution Loss

**Definition.**

$$
\mathcal L_{\mathrm{eq}}(\Phi) \;=\; \frac{1}{N_c}\sum_{C}
\Big(\frac{M_C}{\bar M} - 1\Big)^{2},
\qquad
M_C = m\big|_{\Phi(C)}\,\lvert\det J_\Phi\rvert_C\,\Delta\xi^{2},
\qquad
\bar M = \frac{1}{N_c}\sum_C M_C ,
$$

with $m>0$ a monitor function (default $m = (\varepsilon_0 + |\nabla a|^2)^{1/2}$).

**(i) Non-negativity and characterisation.**  $\mathcal L_{\mathrm{eq}} \ge 0$ and
$\mathcal L_{\mathrm{eq}} = 0 \iff M_C = \bar M$ for every cell: the loss vanishes
*exactly* on the equidistributing meshes and nowhere else.

**(ii) Scale invariance.**  $\mathcal L_{\mathrm{eq}}$ is invariant under
$m \mapsto c\,m$ and under a rescaling of the domain, because it is normalised by
$\bar M$.  A loss without the normalisation could be driven to zero by shrinking
every cell, which is the sense in which naive "uniform cell size" objectives are
gameable.

**(iii) Worst-cell control.**  With $x_C = M_C/\bar M$ and $N_c$ cells,
Cauchy–Schwarz gives

$$
\max_C \frac{M_C}{\bar M} \le 1 + \sqrt{N_c\,\mathcal L_{\mathrm{eq}}},
\qquad
\min_C \frac{M_C}{\bar M} \ge 1 - \sqrt{N_c\,\mathcal L_{\mathrm{eq}}} .
$$

So a *small* loss bounds the mass of the worst cell, which is the quantity the
classical theory controls (de Boor 1973: the mesh minimising
$\max_C \int_C m$ equidistributes the monitor).

**(iv) Optimality for the representation error.**  If $m \ge m_0 > 0$ and
$|D^2 f| \le M_f$, then on a chart whose cells have diameter $h_C$,

$$
\|f - I_h f\|_\infty \le \frac{M_f}{8}\max_C h_C^2
\le \frac{M_f}{8\,m_0^{2}}\max_C \Big(\int_C m\Big)^{2},
$$

because $\int_C m \ge m_0 |C| \ge m_0 h_C$.  The right-hand side is minimised,
with value $(I_m/N_c)^2$ where $I_m=\int_\Omega m$, exactly when the monitor is
equidistributed.  Hence the equidistributed chart minimises a *certified upper
bound* on the representation error at fixed resolution.  The chain is a
statement about the chart interpolant; composing with $\Phi$ adds the $O(h^2)$
factor of P6.

**(v) What it does not say.**  Equidistribution minimises the worst cell mass,
not the $L^\infty$ error of the network's output, and not the PDE residual.  It is
an auxiliary objective, and §P8 measures how much it actually buys.

**(vi) Differentiability.**  $M_C$ is smooth in $\theta$ wherever the bracket of
the monotone inversion is locally constant, and $\bar M > 0$ by P5; the loss is
therefore differentiable almost everywhere, and its gradient reaches the chart
parameters (tested).

**Verification.**  A one-layer chart ($N=64$, 400 Adam steps, equidistribution
loss only, no data) compresses the mesh at a boundary layer, a ridge, and a
smooth field:

| field | loss (init $\to$ final) | cell-mass ratio | repr. error, uniform | repr. error, chart | gain |
|---|---|---|---|---|---|
| boundary layer | $0.43 \to 9.0\cdot10^{-4}$ | 1.30 | $4.06\cdot10^{-2}$ | $9.9\cdot10^{-3}$ | **4.1×** |
| ridge | $0.18 \to 1.2\cdot10^{-3}$ | 1.49 | $1.34\cdot10^{-2}$ | $6.7\cdot10^{-3}$ | **2.0×** |
| smooth | $0.029 \to 2.2\cdot10^{-3}$ | 1.81 | $1.71\cdot10^{-3}$ | $3.2\cdot10^{-3}$ | 0.54× |

The last row is the honest one and it is not a bug: equidistribution minimises
the worst *mass*, and for a field whose monitor is already nearly uniform the
redistribution has nothing to gain while the moment-to-moment jitter of the
optimiser costs a little.  Two consequences are drawn in the paper and in the
API: (a) the loss is an *auxiliary* term in the operator, weighted by
$\lambda_{\mathrm{eq}}$, never the only objective; (b) the chart can always fall
back to the identity, which is in the class by P5.v, so the operator cannot be
worse than its own baseline in the model class — only in an individual
finite-precision run.

---

## P8 — The chart is a gauge (a consistency remark, not a theorem)

Two different charts represent the same physical field, and the transported
operator is the same operator seen through a different discretisation:

$$
\big\|P^{-1}_{\Phi_a} T P_{\Phi_a} f \;-\; P^{-1}_{\Phi_b} T P_{\Phi_b} f\big\|
\;\le\; \|P^{-1}_{\Phi_a}TP_{\Phi_a}f - Tf\| + \|T f - P^{-1}_{\Phi_b}TP_{\Phi_b}f\|,
$$

and each term is bounded by the transport error of the corresponding chart
(P6).  The chart therefore changes the *representation* — where the degrees of
freedom sit — not the physical problem.  This is what makes "the same network at
every resolution" meaningful, and it is why the chart must be trained with the
task loss: the gauge freedom is invisible to the equidistribution term alone.

Measured on a deliberately strong chart (mean displacement 0.46, i.e. half the
domain): the relative difference between two such charts is
$1.7\cdot10^{-1}$, $2.7\cdot10^{-1}$, $9.3\cdot10^{-2}$ at $N=32,64,128$.  It is
*not* a clean $O(h^{2})$ sequence, because a violent chart makes the transport
error dominated by its smallest cells rather than by the mean spacing — this is
precisely the regime in which one should not use a violent chart, and it is
reported here so that the reader does not over-read P8.  No theorem is claimed.

---
## What is *not* claimed

* The softplus barrier (P2) is a soft penalty. It makes folding *expensive*, not
  impossible, and it does not bound $\det J_\phi$ away from zero.
* Zero initialisation (P1) is a statement about $t = 0$ only.
* $\det J_\phi > 0$ on a discrete grid does not imply global injectivity of the
  continuous map (a local diffeomorphism need not be injective); the certificate
  of P3 is the standard, local-to-global caveat included.
* The empirical statement "0.00% grid folding" is a measurement on the
  validation samples at the end of training, not a theorem — unless the run used
  `lipschitz_mode="spectral"`, in which case P3 applies, or the model is a
  `DagneseChartOperator`, in which case P5 applies and the guarantee is
  structural.
* P5 gives $\det J>0$ **on the discretised cells**.  The continuous piecewise
  bilinear interpolant of a D'Agnese chart is globally injective because the
  construction is separable and monotone in each argument — that part is a
  theorem, not a discretisation statement — but the *composed* multi-layer chart
  is injective only in the sense that a composition of injective maps is
  injective, and each layer is evaluated at the previous layer's images by
  bilinear interpolation whose exactness is asserted on the grid nodes.
* The lower bound is $\varepsilon^2$: it is *positive*, and it is *not tight*.
  A trained chart can have $\min\det J \approx 0.8$ while the certificate reports
  $10^{-6}$.  The certificate is a worst-case statement over the whole parameter
  space, and it should be reported as such.
* $\det J > 0$ everywhere does not make the *network* well conditioned, does not
  bound the Lipschitz constant of the operator, and does not bound the PDE error.
* Nothing in P5–P8 *implies* that the chart improves accuracy.  P7(iv) bounds a
  representation error; §P7's table shows that the bound is productive for
  feature-dominated fields and neutral otherwise.  The accuracy claims live or
  die in `results/v2/TABLE.md`, with paired seeds and the seed noise reported —
  and there the chart wins: pooled $\Delta L^2 = -0.0182$ ($p=4\cdot10^{-4}$),
  $\Delta H^1 = -0.0413$ ($p<10^{-4}$) over eighteen paired runs on six
  geometries, with an ablation that isolates the loss ($-0.0073$, 17/18 runs).
  That is a measurement on one protocol, not a theorem.
* The `dif*` rows of `results/repro` were produced with the original v1 data
  pipeline; they are not numerically comparable to the `results/v2` sweep, which
  is internally consistent.  Do not mix the two tables.
