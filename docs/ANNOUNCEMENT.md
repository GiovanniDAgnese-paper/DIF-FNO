# Announcement pack

Everything below is derived from committed numbers: `results/v2/STATS.json`,
`results/v2/TABLE.md` and `paper/tables/benchmark.tex`. If you change the
benchmark, regenerate them with `make stats paper-tables` and update the text.

The rule for this pack: **no claim may be stronger than the numbers, and every
number is checkable in one command.**

---

## 1. One line

> A neural operator whose grid is a *certified diffeomorphism*: it cannot fold,
> it has a closed-form inverse, it costs 1 602 parameters, and on a six-geometry
> Darcy benchmark it cuts the relative L2 error by 8.6 % and H1 by 9.7 %
> (18 paired runs, p < 10⁻³).

## 2. Abstract (150 words, portable)

Fourier neural operators assume a regular grid. Deforming that grid to the
geometry is an old idea, and it has a fatal flaw: a learned deformation has no
reason to be injective, and a folded grid silently destroys the representation
while the training loss still looks fine. Existing methods penalise folding; a
penalty is not a guarantee, and the uncertified ablation in this repository
folds 12–22 % of its cells while scoring as well as the penalised model.

We replace the penalty with a construction. The **D'Agnese chart** is a
piecewise-bilinear diffeomorphism of the reference square built from strictly
positive increments: injective, `det J > 0` as a product of positive factors, a
closed-form inverse at arbitrary points, a parameter-free floor `ε²` that holds
for *every* parameter value, and exactly the identity at initialisation, so the
operator is a strict super-set of the FNO it starts from.

The **D'Agnese Equidistribution Loss** gives the learned mesh an objective:
its minimiser is the optimal transport map for the monitor density, and it
bounds the representation error of a given field. With the same architecture and
the same budget, adding it improves 17 of 18 runs.

## 3. Thread (7 posts)

1. Neural operators hate irregular geometry. The usual fix is to warp the grid
   — and then hope the warp is a diffeomorphism. It often isn't. When it folds,
   nothing crashes: the loss goes down, the results go wrong.
2. We deleted the hope. The D'Agnese chart is built from *positive increments*,
   so `det J > 0` is a product of positive numbers, not a softmax that might
   underflow. Injective for every parameter value. Closed-form inverse. Exactly
   the identity at init.
3. Consequence: the operator starts *bit-for-bit* as the FNO (measured 3e-8) and
   can only leave it by learning a valid mesh. The ablation is therefore honest
   by construction, not by effort.
4. Then we gave the mesh a criterion instead of a heuristic: the D'Agnese
   Equidistribution Loss, whose minimiser is the optimal transport map for the
   monitor. Classical theory (de Boor–Babuška–Rheinboldt) does the rest.
5. Benchmark: 6 geometries (star, L-shape, annulus, airfoil, a crack, a plate
   with two holes) × 3 seeds, paired. L2 −8.6 %, H1 −9.7 %, p < 10⁻³, 16/18 runs,
   6/6 geometries. Cost ×2.8 per step, +0.5 % parameters.
6. The load-bearing experiment is the ablation, not the headline: the *same*
   chart without the equidistribution term improves only 11/18 runs. Adding the
   loss moves 17/18 by a further 0.0073. The gain is the loss, not the extra
   parameters.
7. Everything is reproducible in three commands and the paper says what it does
   *not* show (the certificate is a worst case; the effect is small; two runs
   lose). Repo + paper + tests: <link>.

## 4. Explainer hooks (for a blog post or a lab page)

* **The failure mode nobody plots.** Folded grids do not raise errors; they
  quietly change the function being learned. Show the v1 ablation: 12–22 % folded
  cells, identical loss curve.
* **"Certified" should mean certified.** A bound that depends on the parameters
  is a hope. The floor here is `ε² = (2ε/(KΔξ))²`, independent of θ, N and the
  deformation size — and it is the *worst case*, which is why the paper also
  reports the measured `min det J` (0.03–0.9).
* **A mesh is a model.** If the chart is conditioned on the input field, the
  network outputs a mesh, not just a solution. That is the sentence to put on a
  slide.

## 5. Where to send it

| venue | why | what to emphasise |
|---|---|---|
| NeurIPS / ICML (main) | method + theory + benchmark, all present | the construction, the ablation, the honest statistics |
| ICLR | the ablation and the "strict super-set of the baseline" framing | the initialisation argument |
| SIAM J. Sci. Comput. / J. Comput. Phys. | the mesh reading is classical; the audience will recognise de Boor–Babuška | the certificate and the equidistribution theorem |
| arXiv (cs.LG + math.NA) first | priority + visibility | the whole package is self-contained |

Two practical notes. (i) The paper has **never been compiled** in this
repository — no TeX engine was available — so the first action before submission
is `make paper-pdf` on a machine with LaTeX and a read-through of the log for
overfull boxes. (ii) `paper/tables/benchmark.tex` is generated; never edit it by
hand.

## 6. Numbers, one command each

```bash
make stats          # every statistic in the paper -> results/v2/STATS.{json,md}
make paper-tables   # paper/tables/benchmark.tex + results/v2/TABLE.md + figure
make study          # deterministic verification of P5-P8 (no seeds involved)
make timing         # the x2.8 cost, measured
pytest tests        # 178 tests: every proposition, under its own name
```
