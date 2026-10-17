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


# --------------------------------------------------------------------------- #
def paired_comparison(runs: list[dict], reference: str = "fno",
                      fields: tuple[str, ...] = ("rel_l2", "rel_h1")) -> dict:
    """Paired (same seed, same domain) differences against the reference variant.

    Seed-to-seed variability is the dominant source of noise in this benchmark
    (the train/validation split changes with the seed), so the paired difference
    is a far more sensitive statistic than the difference of the means.
    """
    table: dict[tuple[str, str, int], dict] = {}
    for r in runs:
        key = (r["config"]["tag"], r["config"]["domain"], r["config"]["seed"])
        table[key] = r["final"]

    out: dict[str, dict] = {}
    variants = sorted({k[0] for k in table} - {reference})
    domains = sorted({k[1] for k in table})
    for variant in variants:
        per_domain: dict[str, dict] = {}
        for domain in domains:
            diffs: dict[str, list[float]] = {f: [] for f in fields}
            for seed in sorted({k[2] for k in table}):
                a = table.get((variant, domain, seed))
                b = table.get((reference, domain, seed))
                if not a or not b:
                    continue
                for f in fields:
                    if a.get(f) is not None and b.get(f) is not None:
                        diffs[f].append(a[f] - b[f])
            if any(diffs.values()):
                per_domain[domain] = {
                    f: (float(np.mean(v)), float(np.std(v)) if len(v) > 1 else 0.0, len(v))
                    for f, v in diffs.items() if v
                }
        if per_domain:
            out[variant] = per_domain
    return out


def _domains(summary: dict) -> list[str]:
    """Union of the domains covered by any variant, in canonical order."""
    found = {d for v in summary.values() for d in v if d != "AVG"}
    order = [d for d in DEFAULT_DOMAINS if d in found]
    return order + sorted(found - set(order))


def consolidate(runs: list[dict], out_dir: Path) -> dict:
    """Aggregate runs into ``CONSOLIDATED.json`` + markdown/LaTeX tables."""
    table: dict[str, dict] = {}
    for r in runs:
        dom = r["config"]["domain"]
        var = r["config"]["tag"]
        key = f"{var}|{dom}"
        vals = table.setdefault(key, {"_runs": []})
        vals["_runs"].append(r)

    summary: dict[str, dict] = {}
    for key, entry in table.items():
        var, dom = key.split("|")
        runs_ = entry["_runs"]

        def agg(field):
            v = [r["final"].get(field) for r in runs_]
            v = [x for x in v if x is not None]
            return (float(np.mean(v)), float(np.std(v)) if len(v) > 1 else 0.0, int(len(v))) if v else None

        summary.setdefault(var, {})[dom] = {
            "rel_l2": agg("rel_l2"),
            "rel_h1": agg("rel_h1"),
            "rel_H1": agg("rel_H1"),
            "min_detJ": agg("min_detJ"),
            "fold_pct": agg("fold_pct"),
            "displacement_mean": agg("displacement_mean"),
            "certified_det_lower_bound": agg("certified_det_lower_bound"),
            "lipschitz_bound": agg("lipschitz_bound"),
            "n_runs": len(runs_),
            "n_params": runs_[0]["n_params"],
            "wall_time_s": float(np.mean([r["wall_time_s"] for r in runs_])),
        }
    # averages across domains
    for _var, doms in summary.items():
        avg = {}
        for field in ("rel_l2", "rel_h1", "rel_H1", "fold_pct", "displacement_mean", "certified_det_lower_bound"):
            vals = [d[field][0] for d in doms.values() if d[field] is not None]
            avg[field] = float(np.mean(vals)) if vals else None
        avg["min_detJ"] = min((d["min_detJ"][0] for d in doms.values() if d["min_detJ"] is not None), default=None)
        avg["wall_time_s"] = float(np.mean([d["wall_time_s"] for d in doms.values()]))
        doms["AVG"] = avg

    consolidated = {
        "summary": summary,
        "paired_vs_fno": paired_comparison(runs),
        "variant_descriptions": {k: v.get("description", "") for k, v in VARIANTS.items()},
        "runs": [
            {"run_name": r["run_name"], "domain": r["config"]["domain"],
             "variant": r["config"]["tag"], "seed": r["config"]["seed"],
             "final": r["final"], "wall_time_s": r["wall_time_s"],
             "n_params": r["n_params"]}
            for r in runs
        ],
        "environment": runs[0]["environment"] if runs else {},
        "n_runs": len(runs),
    }
    (out_dir / "CONSOLIDATED.json").write_text(json.dumps(consolidated, indent=2, default=str))
    (out_dir / "table.md").write_text(markdown_table(summary, consolidated["paired_vs_fno"]))
    (out_dir / "table.tex").write_text(latex_table(summary))
    return consolidated


def _fmt(value, digits=4, plus=False):
    if value is None:
        return "--"
    if isinstance(value, (tuple, list)):
        return f"{value[0]:.{digits}f}"
    return f"{value:.{digits}f}"


def markdown_table(summary: dict, paired: dict | None = None) -> str:
    variants = list(summary)
    domains = _domains(summary)
    head = ["Domain"] + [f"{v} L2" for v in variants] + [f"{v} H1" for v in variants]
    lines = ["| " + " | ".join(head) + " |",
             "|" + "|".join(["---"] * len(head)) + "|"]
    for d in domains + ["AVG"]:
        row = [d]
        for v in variants:
            row.append(_fmt((summary[v].get(d) or {}).get("rel_l2")))
        for v in variants:
            row.append(_fmt((summary[v].get(d) or {}).get("rel_h1")))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("Topological metrics")
    lines.append("")
    head2 = ["Domain"] + [f"{v} fold%" for v in variants] + [f"{v} min detJ" for v in variants] \
        + [f"{v} cert." for v in variants]
    lines += ["| " + " | ".join(head2) + " |", "|" + "|".join(["---"] * len(head2)) + "|"]
    for d in domains + ["AVG"]:
        row = [d]
        for v in variants:
            row.append(_fmt((summary[v].get(d) or {}).get("fold_pct"), 2))
        for v in variants:
            row.append(_fmt((summary[v].get(d) or {}).get("min_detJ"), 5, plus=True))
        for v in variants:
            row.append(_fmt((summary[v].get(d) or {}).get("certified_det_lower_bound"), 5))
        lines.append("| " + " | ".join(row) + " |")

    if paired:
        lines += ["", "Paired differences against `fno` (same seed; negative = better)", ""]
        head3 = ["Domain"] + [f"{v} dL2" for v in paired] + [f"{v} dH1" for v in paired]
        lines += ["| " + " | ".join(head3) + " |", "|" + "|".join(["---"] * len(head3)) + "|"]
        for d in domains:
            row = [d]
            for v in paired:
                e = paired[v].get(d, {}).get("rel_l2")
                row.append("--" if e is None else f"{e[0]:+.4f} +- {e[1]:.4f}")
            for v in paired:
                e = paired[v].get(d, {}).get("rel_h1")
                row.append("--" if e is None else f"{e[0]:+.4f} +- {e[1]:.4f}")
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def latex_table(summary: dict) -> str:
    variants = list(summary)
    domains = _domains(summary)
    cols = "l|" + "cc|" * len(variants)
    out = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{DIF-FNO benchmark reproduced with this repository "
        "(Darcy flow, 60 epochs, 80/20 split).}",
        "\\label{tab:benchmark}",
        f"\\begin{{tabular}}{{{cols}}}",
        "\\hline",
        "\\textbf{Domain} & " + " & ".join(
            f"\\multicolumn{{2}}{{c|}}{{\\textbf{{{v}}}}}" for v in variants) + " \\\\",
        " & " + " & ".join("$L^2$ & $H^1$" for _ in variants) + " \\\\",
        "\\hline",
    ]
    for d in domains + ["AVG"]:
        row = [d if d != "AVG" else "\\textbf{Average}"]
        for v in variants:
            e = summary[v].get(d) or {}
            row.append(_fmt(e.get("rel_l2")))
            row.append(_fmt(e.get("rel_h1")))
        out.append(" & ".join(row) + " \\\\")
    out += ["\\hline", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(out)


# --------------------------------------------------------------------------- #
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
