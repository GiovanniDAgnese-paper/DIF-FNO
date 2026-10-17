# Contributing

Thank you for taking the time to improve DIF-FNO.

## Getting set up

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests            # should be green before you start
ruff check src tests scripts
```

Useful entry points (see `make help`):

| command | what it does |
|---|---|
| `make data` | regenerate the Darcy datasets (64 x 64, 100 samples per domain) |
| `make benchmark` | the full reproduction sweep (3 variants x 4 domains x 3 seeds) |
| `make figures` | rebuild the figures from `results/repro` |
| `make test-cov` | tests with a coverage report |

## Ground rules

1. **Tests are part of the claim.** Every mathematical statement in
   `docs/THEORY.md` has a matching test in `tests/`. If you change the geometry,
   the losses or the models, add or update the corresponding test.
2. **Do not break reproducibility.** Runs are seeded; if you change a default
   that affects numbers (`tau`, `normalize_by_mask`, the split, the initialisation),
   say so in `CHANGELOG.md` and regenerate `results/repro`.
3. **Report what you measure, not what you hope.** The benchmark reports folding
   percentage, `min det(J)`, the *displacement* of the learned map and the
   certified bound. A map that stays at the identity trivially satisfies
   `det J > 0`; the displacement column exists so that this can be seen.
4. **Distinguish guarantees from heuristics.** Only the Lipschitz-constrained
   configuration is a guarantee (`det J >= (1 - kappa L)^2`). Please do not
   describe the softplus barrier as a guarantee anywhere in the docs.
5. **Style.** `ruff check` must pass; public functions carry a docstring with the
   maths in LaTeX.

## Adding a domain

1. Add a mask function to `src/dagnese_fno/data/masks.py` and register it in
   `DOMAINS` (with `simply_connected` and a default seed).
2. Add a geometry test in `tests/test_data.py` (fill fraction, topology).
3. Generate the data and, if you want it in the tables, add the domain to
   `scripts/run_benchmark.py --domains`.

## Reporting a bug

Open an issue with: the exact command, the versions (`python -c "from dagnese_fno import describe_environment; print(describe_environment())"`),
and the run JSON from `results/` when relevant.
