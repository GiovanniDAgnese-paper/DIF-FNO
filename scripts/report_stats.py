#!/usr/bin/env python
"""Every statistic quoted in the paper, recomputed from the per-run JSON.

The benchmark writes one JSON per (variant, domain, seed).  This script reads
them, forms the *paired* differences against the reference variant at the same
seed, and reports:

* the pooled mean, standard deviation, Student-t and Wilcoxon p-values,
  the number of improving runs, and both definitions of the relative gain
  (ratio of means and mean of per-run ratios);
* the per-domain means, and the domain-level sign test, which is the honest
  test when the runs are nested in geometries;
* the worst admissibility numbers over all runs.

It writes ``STATS.json`` next to the runs and prints a markdown summary.

Usage::

    python scripts/report_stats.py --results results/v2 --reference fno
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dagnese_fno.evaluation import paired_test, sign_test, significance_stars  # noqa: E402

METRICS = ("rel_l2", "rel_h1")
TOPOLOGY = ("fold_pct", "min_detJ", "certified_det_lower_bound", "chart_inverse_error")


def load_runs(root: Path) -> list[dict]:
    runs = []
    for path in sorted(root.glob("*.json")):
        if path.name.startswith("CONSOLIDATED") or path.name in ("STATS.json",):
            continue
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or "final" not in payload:
            continue
        cfg = payload.get("config", {})
        runs.append({
            "path": path.name,
            "variant": cfg.get("tag") or cfg.get("variant"),
            "domain": cfg.get("domain"),
            "seed": cfg.get("seed"),
            "final": payload["final"],
        })
    return [r for r in runs if r["variant"] and r["domain"]]


def paired(runs, variant, domains, metric, reference):
    """Per-(domain, seed) paired differences against ``reference``."""
    index: dict[tuple, dict] = {}
    for r in runs:
        index.setdefault((r["domain"], r["seed"]), {})[r["variant"]] = r["final"]
    diffs, refs, vals = [], [], []
    for d in domains:
        for (dom, _seed), per in sorted(index.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
            if dom != d or variant not in per or reference not in per:
                continue
            a, b = per[variant], per[reference]
            if a.get(metric) is None or b.get(metric) is None:
                continue
            diffs.append(float(a[metric] - b[metric]))
            refs.append(float(b[metric]))
            vals.append(float(a[metric]))
    return np.asarray(diffs), np.asarray(refs), np.asarray(vals)


def tests(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired inference, delegated to :mod:`dagnese_fno.evaluation.statistics`.

    The library function is the single implementation of the paired test in the
    repository (it also backs ``run_benchmark.py``); this wrapper only renames
    the keys and adds the t statistic, which is the number the paper quotes.
    """
    res = paired_test(list(a), list(b))
    d = a - b
    n = int(len(d))
    out = {
        "n": n,
        "mean": float(d.mean()),
        "sd": float(d.std(ddof=1)) if n > 1 else 0.0,
        "se": float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0,
        "negative": int((d < 0).sum()),
        "p_two_sided": res["t_pvalue"],
        "wilcoxon_p_less": float("nan"),
        "ci_low": res["ci_low"],
        "ci_high": res["ci_high"],
        "stars": significance_stars(res["t_pvalue"]),
    }
    out["t"] = out["mean"] / out["se"] if out["se"] > 0 else float("nan")
    try:  # one-sided Wilcoxon, the confirmatory direction (variant better = negative)
        from scipy import stats as _st
        if n >= 6 and not np.allclose(d, d[0]):
            out["wilcoxon_p_less"] = float(_st.wilcoxon(d, alternative="less").pvalue)
    except Exception:  # pragma: no cover - scipy optional
        pass
    return out


def _extreme(values, how):
    """min/max that ignores ``None``; returns ``None`` for an empty selection."""
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    return getattr(np, how)(vals) if vals else None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="results/v2")
    p.add_argument("--reference", default="fno")
    p.add_argument("--out", default=None, help="default: <results>/STATS.json")
    args = p.parse_args(argv)

    root = Path(args.results)
    runs = load_runs(root)
    if not runs:
        print(f"no runs under {root}")
        return 1
    domains = sorted({r["domain"] for r in runs})
    variants = sorted({r["variant"] for r in runs if r["variant"] != args.reference})

    report: dict = {"reference": args.reference, "domains": domains,
                    "n_runs": len(runs), "variants": {}}
    lines = [f"# Statistics, {len(domains)} domains, reference `{args.reference}`",
             "", f"Runs: {len(runs)}  |  domains: {', '.join(domains)}", ""]

    for v in variants:
        entry: dict = {}
        for metric in METRICS:
            diffs, refs, vals = paired(runs, v, domains, metric, args.reference)
            if len(diffs) < 2:
                continue
            stats = tests(vals, refs)
            stats["ratio_of_means"] = float(vals.mean() / refs.mean())
            stats["mean_of_ratios"] = float((vals / refs).mean())
            per_domain = []
            for d in domains:
                dd, _, _ = paired(runs, v, [d], metric, args.reference)
                per_domain.append({"domain": d, "n": int(len(dd)),
                                   "mean": float(dd.mean()) if len(dd) else float("nan")})
            # domains without a complete pair set carry no information: drop them
            per_domain = [x for x in per_domain if x["n"] > 0 and np.isfinite(x["mean"])]
            stats["per_domain"] = per_domain
            neg_domains = sum(1 for x in per_domain if x["mean"] < 0)
            stats["domains_negative"] = neg_domains
            stats["domains_total"] = len(per_domain)
            stats["domain_sign_test_p"] = sign_test(neg_domains, len(per_domain))
            entry[metric] = stats
            lines += [
                f"## `{v}` vs `{args.reference}`, {metric}",
                "",
                f"- pooled: n = {stats['n']}, mean = {stats['mean']:+.4f}, "
                f"sd = {stats['sd']:.4f}, t = {stats['t']:+.2f}, "
                f"p(two-sided) = {stats['p_two_sided']:.4f}, "
                f"p(Wilcoxon, less) = {stats['wilcoxon_p_less']:.4f} {stats['stars']}",
                f"- 95 % CI of the mean difference: [{stats['ci_low']:+.4f}, "
                f"{stats['ci_high']:+.4f}]",
                f"- improving runs: {stats['negative']}/{stats['n']}  "
                f"|  gain: {100 * (1 - stats['mean_of_ratios']):+.2f} % "
                f"(mean of ratios), {100 * (1 - stats['ratio_of_means']):+.2f} % (ratio of means)",
                f"- domains negative: {neg_domains}/{len(per_domain)} "
                f"(sign-test p = {stats['domain_sign_test_p']:.4f})",
                "",
                "| domain | n | paired mean |",
                "|---|---|---|",
            ]
            for x in per_domain:
                lines.append(f"| {x['domain']} | {x['n']} | {x['mean']:+.4f} |")
            lines.append("")
        report["variants"][v] = entry

    # admissibility: worst case per variant
    topo: dict = {}
    for v in variants:
        sel = [r["final"] for r in runs if r["variant"] == v]
        if not sel or "fold_pct" not in sel[0]:
            continue
        topo[v] = {
            "fold_pct_max": _extreme((x.get("fold_pct") for x in sel), "max"),
            "min_detJ_min": _extreme((x.get("min_detJ") for x in sel), "min"),
            "certified_lb_min": _extreme((x.get("certified_det_lower_bound") for x in sel), "min"),
            "inverse_error_max": _extreme((x.get("chart_inverse_error") for x in sel), "max"),
        }
    report["admissibility"] = topo
    if topo:
        lines += ["## Admissibility (worst case over all runs)", "",
                  "| variant | fold [%] | min det J | cert. lower bound | inverse error |",
                  "|---|---|---|---|---|"]
        for v, t in topo.items():
            def g(key, spec):
                return "--" if t[key] is None else format(t[key], spec)
            lines.append(f"| `{v}` | {g('fold_pct_max', '.2f')} | {g('min_detJ_min', '.4f')} | "
                         f"{g('certified_lb_min', '.4f')} | {g('inverse_error_max', '.2e')} |")
        lines.append("")

    out = Path(args.out) if args.out else root / "STATS.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    text = "\n".join(lines)
    (root / "STATS.md").write_text(text + "\n")
    print(text)
    print(f"\nwrote {out} and {root / 'STATS.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
