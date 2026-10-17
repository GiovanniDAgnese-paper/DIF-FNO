#!/usr/bin/env python
"""Run the full DIF-FNO benchmark sweep and consolidate the results.

Variants
--------
``fno``
    Standard FNO on the physical grid (baseline).
``dif-barrier``
    DIF-FNO with the D'Agnese softplus barrier (the v4 configuration:
    pointwise conditioning, deformation scale 0.1, lambda = 10).
``dif-certified``
    DIF-FNO with spectral-norm-capped residual and global conditioning:
    :math:`\\mathrm{Lip}(r) \\le L < 1`, hence :math:`\\det J \\ge (1-L)^2 > 0`
    **by construction**.
``dif-nobar``
    Ablation: unconstrained diffeomorphism (lambda = 0) at a large deformation
    scale -- shows what the barrier is supposed to prevent.
``dif-barrier-large``
    Same large deformation scale, barrier switched on.

Usage
-----
::

    python scripts/run_benchmark.py --domains star lshape annulus --out results/repro
    python scripts/run_benchmark.py --variants dif-nobar dif-barrier-large --domains star
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dagnese_fno.evaluation import paired_test, significance_stars, summarise  # noqa: E402
from dagnese_fno.training import TrainConfig, save_result, train_one  # noqa: E402

# --------------------------------------------------------------------------- #
VARIANTS: dict[str, dict] = {
    "fno": {
        "model": "fno",
        "lambda_barrier": 0.0,
        "barrier": "none",
        "description": "Standard FNO baseline (physical grid, identical backbone)",
    },
    "dif-barrier": {
        "model": "dif",
        "lambda_barrier": 10.0,
        "barrier": "softplus",
        "tau": 0.05,
        "deformation_scale": 0.1,
        "conditioning": "pointwise",
        "lipschitz_mode": "none",
        "description": "DIF-FNO + D'Agnese softplus barrier (v4 configuration)",
    },
    "dif-certified": {
        "model": "dif",
        "lambda_barrier": 1.0,
        "barrier": "softplus",
        "tau": 0.05,
        "deformation_scale": 0.5,
        "conditioning": "global",
        "lipschitz_mode": "spectral",
        "lipschitz_const": 0.9,
        "latent_dim": 8,
        "description": "DIF-FNO with Lipschitz-capped residual: det J >= (1-L)^2 by construction",
    },
    "dif-nobar": {
        "model": "dif",
        "lambda_barrier": 0.0,
        "barrier": "none",
        "deformation_scale": 1.0,
        "conditioning": "pointwise",
        "description": "Ablation: unconstrained diffeomorphism at large deformation scale",
    },
    "dif-barrier-large": {
        "model": "dif",
        "lambda_barrier": 10.0,
        "barrier": "softplus",
        "tau": 0.05,
        "deformation_scale": 1.0,
        "conditioning": "pointwise",
        "description": "Barrier at large deformation scale (what the barrier buys you)",
    },
}

DEFAULT_DOMAINS = ["star", "lshape", "annulus"]


# --------------------------------------------------------------------------- #
def build_config(domain: str, variant: str, seed: int, **overrides) -> TrainConfig:
    kwargs = dict(VARIANTS[variant])
    kwargs.pop("description", None)
    cfg = TrainConfig(domain=domain, seed=seed, tag=f"{variant}", **kwargs)
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise ValueError(f"unknown TrainConfig field '{k}'")
        setattr(cfg, k, v)
    return cfg


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--domains", nargs="+", default=DEFAULT_DOMAINS)
    p.add_argument("--variants", nargs="+", default=["fno", "dif-barrier", "dif-certified"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="results/repro")
    p.add_argument("--device", default="cpu")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--list-variants", action="store_true")
    p.add_argument("--consolidate-only", action="store_true",
                   help="skip training: consolidate every *.json already in --out")
    return p.parse_args(argv)


def _domains(summary: dict) -> list[str]:
    """Union of the domains covered by any variant, in canonical order."""
    found = {d for v in summary.values() for d in v if d not in ("AVG", "AVG*")}
    order = [d for d in DEFAULT_DOMAINS if d in found]
    return order + sorted(found - set(order))


def _fmt(value, digits: int = 4, plus: bool = False) -> str:
    """Format a :func:`summarise` summary as ``mean +- s.d.``."""
    if not value:
        return "--"
    mean, std = value["mean"], value["std"]
    if std is None or not np.isfinite(std) or value.get("n", 0) < 2:
        return f"{mean:{'+' if plus else ''}.{digits}f}"
    return f"{mean:{'+' if plus else ''}.{digits}f} $\\pm$ {std:.{digits}f}"


def _fmt_plain(value, digits: int = 4, plus: bool = False) -> str:
    if not value:
        return "--"
    return f"{value['mean']:{'+' if plus else ''}.{digits}f}"


def consolidate(runs: list[dict], out_dir: Path, n_bootstrap: int = 5000) -> dict:
    """Aggregate runs into ``CONSOLIDATED.json`` + markdown/LaTeX tables."""
    table: dict[tuple[str, str], list[dict]] = {}
    for r in runs:
        table.setdefault((r["config"]["tag"], r["config"]["domain"]), []).append(r["final"])

    summary: dict[str, dict] = {}
    for (variant, domain), finals in table.items():
        summary.setdefault(variant, {})[domain] = {
            field: summarise([f.get(field) for f in finals], n_bootstrap=n_bootstrap)
            for field in ("rel_l2", "rel_h1", "rel_H1", "min_detJ", "fold_pct",
                          "displacement_mean", "certified_det_lower_bound", "lipschitz_bound")
        }
        members = [r for r in runs
                   if r["config"]["tag"] == variant and r["config"]["domain"] == domain]
        summary[variant][domain]["n_runs"] = len(finals)
        summary[variant][domain]["wall_time_s"] = float(np.mean([r["wall_time_s"] for r in members]))
        summary[variant][domain]["n_params"] = members[0]["n_params"]

    # Averages: over every domain a variant ran on, and over the *common* subset
    # (only the latter is an apples-to-apples comparison across variants).
    common = set.intersection(*[{d for d in v if d != "AVG"} for v in summary.values()])
    for _variant, doms in summary.items():
        for key, pool in (("AVG", set(doms)), ("AVG*", common)):
            avg: dict = {}
            for field in ("rel_l2", "rel_h1", "rel_H1", "fold_pct", "displacement_mean",
                          "certified_det_lower_bound", "lipschitz_bound"):
                vals = [doms[d][field]["mean"] for d in pool
                        if doms[d].get(field) and np.isfinite(doms[d][field]["mean"])]
                avg[field] = ({"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                               "n": len(vals), "ci_low": float("nan"), "ci_high": float("nan")}
                              if vals else None)
            worst = [doms[d]["min_detJ"]["min"] for d in pool
                     if doms[d].get("min_detJ") and np.isfinite(doms[d]["min_detJ"]["min"])]
            avg["min_detJ"] = ({"mean": float(np.min(worst)), "std": float("nan"),
                                "n": len(worst), "ci_low": float("nan"),
                                "ci_high": float("nan")} if worst else None)
            avg["wall_time_s"] = float(np.mean([doms[d]["wall_time_s"] for d in pool]))
            doms[key] = avg

    paired = paired_comparison(runs)
    consolidated = {
        "summary": summary,
        "paired_vs_fno": paired,
        "variant_descriptions": {k: v.get("description", "") for k, v in VARIANTS.items()},
        "n_runs": len(runs),
        "seeds": sorted({r["config"]["seed"] for r in runs}),
        "environment": runs[0]["environment"] if runs else {},
        "provenance": {
            "git_commit": runs[0].get("git_commit") if runs else None,
            "git_dirty": runs[0].get("git_dirty") if runs else None,
            "dataset_sha256": sorted({r.get("data_sha") for r in runs if r.get("data_sha")}),
        },
        "runs": [
            {"run_name": r["run_name"], "domain": r["config"]["domain"],
             "variant": r["config"]["tag"], "seed": r["config"]["seed"],
             "final": r["final"], "best": r.get("best"),
             "wall_time_s": r["wall_time_s"], "n_params": r["n_params"],
             "n_train": r.get("n_train"), "n_val": r.get("n_val"), "kappa": r.get("kappa")}
            for r in runs
        ],
    }
    (out_dir / "CONSOLIDATED.json").write_text(json.dumps(consolidated, indent=2, default=str))
    (out_dir / "table.md").write_text(markdown_table(summary, paired, consolidated))
    (out_dir / "table.tex").write_text(latex_table(summary, consolidated))
    return consolidated


def paired_comparison(runs: list[dict], reference: str = "fno",
                      fields: tuple[str, ...] = ("rel_l2", "rel_h1")) -> dict:
    """Paired (same seed, same domain) tests against the reference variant.

    Two levels are reported: per domain, and **pooled** over all domains.  The
    pairing removes the domain difficulty from the comparison, so pooling is
    legitimate and buys statistical power.  ``t_pvalue`` is the paired
    *t*-test; ``wilcoxon_pvalue`` is the signed-rank test, the one to trust at
    small ``n``.
    """
    table: dict[tuple[str, str, int], dict] = {}
    for r in runs:
        table[(r["config"]["tag"], r["config"]["domain"], r["config"]["seed"])] = r["final"]

    variants = sorted({k[0] for k in table} - {reference})
    domains = sorted({k[1] for k in table})
    seeds = sorted({k[2] for k in table})

    out: dict[str, dict] = {}
    for variant in variants:
        per_domain: dict[str, dict] = {}
        for domain in domains:
            entry: dict[str, dict] = {}
            for f in fields:
                a = [table[(variant, domain, s)].get(f)
                     for s in seeds if (variant, domain, s) in table]
                b = [table[(reference, domain, s)].get(f)
                     for s in seeds if (reference, domain, s) in table]
                if a and all(x is not None for x in a) and all(x is not None for x in b):
                    entry[f] = paired_test(a, b)
            if entry:
                per_domain[domain] = entry
        pooled: dict[str, dict] = {}
        for f in fields:
            keys = [k for k in table if k[0] == variant and (reference, k[1], k[2]) in table]
            a = [table[k].get(f) for k in keys]
            b = [table[(reference, k[1], k[2])].get(f) for k in keys]
            if a and all(x is not None for x in a) and all(x is not None for x in b):
                pooled[f] = paired_test(a, b)
        if per_domain or pooled:
            out[variant] = dict(per_domain, POOLED=pooled)
    return out


def _paired_cell(e: dict | None) -> str:
    if not e:
        return "--"
    p = e.get("wilcoxon_pvalue")
    if p is None or not np.isfinite(p):
        p = e.get("t_pvalue")
    stars = significance_stars(p) if p is not None else ""
    return (f"{e['mean_diff']:+.4f} [{e['ci_low']:+.4f}, {e['ci_high']:+.4f}] "
            f"{stars} (n={e['n']})")


def markdown_table(summary: dict, paired: dict | None, consolidated: dict) -> str:
    variants = list(summary)
    domains = _domains(summary)
    labels = {"AVG*": "**Average (common domains)**", "AVG": "**Average (all domains)**"}
    rows = list(domains) + (["AVG*"] if any("AVG*" in v for v in summary.values()) else []) + ["AVG"]

    lines = ["# DIF-FNO benchmark", "",
             f"*{consolidated['n_runs']} runs, seeds {consolidated['seeds']}; "
             "mean $\\pm$ s.d. over the seeds.*", ""]
    head = ["Domain"] + [f"{v} $L^2$" for v in variants] + [f"{v} $H^1$" for v in variants]
    lines += ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for d in rows:
        row = [labels.get(d, d)]
        row += [_fmt((summary[v].get(d) or {}).get("rel_l2")) for v in variants]
        row += [_fmt((summary[v].get(d) or {}).get("rel_h1")) for v in variants]
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## Topology", ""]
    head2 = (["Domain"] + [f"{v} fold %" for v in variants]
             + [f"{v} min det J" for v in variants] + [f"{v} certified" for v in variants])
    lines += ["| " + " | ".join(head2) + " |", "|" + "|".join(["---"] * len(head2)) + "|"]
    for d in rows:
        row = [labels.get(d, d)]
        row += [_fmt((summary[v].get(d) or {}).get("fold_pct"), 2) for v in variants]
        row += [_fmt((summary[v].get(d) or {}).get("min_detJ"), 3, plus=True) for v in variants]
        row += [_fmt((summary[v].get(d) or {}).get("certified_det_lower_bound"), 3) for v in variants]
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## Displacement", ""]
    head3 = ["Domain"] + [f"{v} $|\\phi-\\mathrm{{Id}}|$" for v in variants]
    lines += ["| " + " | ".join(head3) + " |", "|" + "|".join(["---"] * len(head3)) + "|"]
    for d in rows:
        row = [labels.get(d, d)]
        row += [_fmt((summary[v].get(d) or {}).get("displacement_mean"), 3) for v in variants]
        lines.append("| " + " | ".join(row) + " |")

    if paired:
        lines += ["", "## Paired differences against `fno` (same seed, same domain)", "",
                  "Negative = better than the baseline. Brackets: 95% Student-t interval of "
                  "the paired difference (the bootstrap interval is narrower but undercovers "
                  "at n = 6 and is only reported in `CONSOLIDATED.json`). "
                  "Stars: paired *t*-test, or the exact Wilcoxon signed-rank test when "
                  "n >= 6; neither is corrected for the number of comparisons.", ""]
        head4 = (["Domain"] + [f"{v} $\\Delta L^2$" for v in paired]
                 + [f"{v} $\\Delta H^1$" for v in paired])
        lines += ["| " + " | ".join(head4) + " |", "|" + "|".join(["---"] * len(head4)) + "|"]
        for d in list(domains) + ["POOLED"]:
            if d != "POOLED" and not any(d in paired[v] for v in paired):
                continue
            row = ["**pooled**" if d == "POOLED" else d]
            for v in paired:
                row.append(_paired_cell((paired[v].get(d) or {}).get("rel_l2")))
            for v in paired:
                row.append(_paired_cell((paired[v].get(d) or {}).get("rel_h1")))
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def latex_table(summary: dict, consolidated: dict) -> str:
    variants = list(summary)
    domains = _domains(summary)
    cols = "l|" + "cc|" * len(variants)
    labels = {"AVG*": "\\textbf{Average (common)}", "AVG": "\\textbf{Average (all)}"}
    out = [
        "% Auto-generated by scripts/run_benchmark.py -- do not edit by hand.",
        f"% {consolidated['n_runs']} runs, seeds {consolidated['seeds']}.",
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{DIF-FNO benchmark reproduced with this repository (Darcy flow, "
        "60 epochs, 80/20 split). Relative $L^2$ and $H^1$ errors "
        "(mean $\\pm$ s.d. over the seeds).}",
        "\\label{tab:benchmark}",
        f"\\begin{{tabular}}{{{cols}}}",
        "\\hline",
        "\\textbf{Domain} & " + " & ".join(
            f"\\multicolumn{{2}}{{c|}}{{\\textbf{{{v}}}}}" for v in variants) + " \\\\",
        " & " + " & ".join("$L^2$ & $H^1$" for _ in variants) + " \\\\",
        "\\hline",
    ]
    for d in domains + ["AVG*", "AVG"]:
        row = [labels.get(d, d)]
        for v in variants:
            e = summary[v].get(d) or {}
            row.append(_fmt_plain(e.get("rel_l2")))
            row.append(_fmt_plain(e.get("rel_h1")))
        out.append(" & ".join(row) + " \\\\")
    out += ["\\hline", "\\end{tabular}", "\\end{table}", ""]

    out += [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Topological diagnostics. ``Certified'' is the proved lower bound "
        "$\\det J_\\phi \\ge (1-\\kappa L)^2$; it is positive only for the "
        "Lipschitz-constrained (\\texttt{spectral}) variant.}",
        "\\label{tab:topology}",
        f"\\begin{{tabular}}{{{cols}}}",
        "\\hline",
        "\\textbf{Domain} & " + " & ".join(
            f"\\multicolumn{{2}}{{c|}}{{\\textbf{{{v}}}}}" for v in variants) + " \\\\",
        " & " + " & ".join("fold \\% & certified" for _ in variants) + " \\\\",
        "\\hline",
    ]
    for d in domains + ["AVG*", "AVG"]:
        row = [labels.get(d, d)]
        for v in variants:
            e = summary[v].get(d) or {}
            row.append(_fmt_plain(e.get("fold_pct"), 2))
            row.append(_fmt_plain(e.get("certified_det_lower_bound"), 3))
        out.append(" & ".join(row) + " \\\\")
    out += ["\\hline", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(out)


def load_runs(out_dir: Path) -> list[dict]:
    """Load every run already stored in ``out_dir``."""
    runs = []
    for path in sorted(out_dir.glob("*.json")):
        if path.name == "CONSOLIDATED.json":
            continue
        data = json.loads(path.read_text())
        if "final" in data and "config" in data:
            runs.append(data)
    return runs


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.list_variants:
        for k, v in VARIANTS.items():
            print(f"{k:20s} {v.get('description','')}")
        return 0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.consolidate_only:
        runs = load_runs(out_dir)
        if not runs:
            print(f"no runs found in {out_dir}")
            return 1
        consolidate(runs, out_dir)
        print((out_dir / "table.md").read_text())
        return 0

    runs = []
    failures: list[str] = []
    t0 = time.time()
    for domain, variant, seed in product(args.domains, args.variants, args.seeds):
        cfg = build_config(domain, variant, seed, epochs=args.epochs,
                           data_dir=args.data_dir, out_dir=str(out_dir),
                           device=args.device, log_every=args.log_every)
        if not args.quiet:
            print(f"\n=== {variant} | {domain} | seed {seed} ===", flush=True)
        try:
            result = train_one(cfg, verbose=not args.quiet)
        except Exception as exc:  # keep the sweep alive; report at the end
            print(f"  FAILED {variant}|{domain}|seed{seed}: {exc!r}", flush=True)
            failures.append(f"{variant}|{domain}|seed{seed}: {exc!r}")
            continue
        path = save_result(result, str(out_dir))
        runs.append(result)
        if not args.quiet:
            print(f"  final: {json.dumps(result['final'])}")
            print(f"  saved: {path}", flush=True)

    if not runs:
        print("no successful runs", file=sys.stderr)
        return 1
    consolidated = consolidate(runs, out_dir)
    print(f"\n=== {len(runs)} runs in {time.time() - t0:.0f}s "
          f"({len(failures)} failed) ===")
    for f in failures:
        print("  FAILED", f)
    print(consolidated and (out_dir / "table.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
