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

  **This matters:** for the log-normal coefficients of the benchmark
  ($a \in [e^{-2}, e^{2}]$, $h = 2/63$, contrast $\gamma = 2$) the measured
  $\max\|\nabla a\|$ is $35$–$41$ (star $40.9$, L-shape $36.1$, annulus $38.2$,
  NACA 0012 $35.0$), hence $\kappa \approx 4\cdot 10^{1}$: a pointwise
  certificate would require $L < 1/\kappa \approx 0.025$ — a *provably*
  folding-free pointwise map can only be a small perturbation of the identity.
  (The maximum is taken over the cells whose forward stencil is entirely active:
  one-sided differences in the boundary layer mix $a$ with the constant fill
  value used outside the mask, which is an artefact of the discretisation
  rather than a property of the permeability.)  Global conditioning removes the
  obstruction altogether ($\kappa = 1$) and is the configuration used by the
  `dif-certified` variant.

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

## What is *not* claimed

* The softplus barrier (P2) is a soft penalty. It makes folding *expensive*, not
  impossible, and it does not bound $\det J_\phi$ away from zero.
* Zero initialisation (P1) is a statement about $t = 0$ only.
* $\det J_\phi > 0$ on a discrete grid does not imply global injectivity of the
  continuous map (a local diffeomorphism need not be injective); the certificate
  of P3 is the standard, local-to-global caveat included.
* The empirical statement "0.00% grid folding" is a measurement on the
  validation samples at the end of training, not a theorem — unless the run used
  `lipschitz_mode="spectral"`, in which case P3 applies and the reported
  `certified_det_lower_bound` is a valid lower bound.
