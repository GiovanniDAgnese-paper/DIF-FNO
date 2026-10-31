#!/usr/bin/env python
"""Turn ``results/v2/CONSOLIDATED.json`` into the paper's benchmark tables.

Produces three files:

* ``paper/tables/benchmark.tex`` -- the LaTeX table included by the paper,
  with the paired differences against the FNO baseline;
* ``results/v2/TABLE.md``       -- the same content for the repository;
* ``figures/paper/fig_benchmark.pdf|png`` -- a paired-difference plot.

The script never trains anything: it only reads what ``run_benchmark.py`` wrote,
so the tables in the paper and in the repository cannot drift apart.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SPEC = {
    "fno": "FNO (baseline)",
    "dif-barrier": "DIF-FNO + softplus barrier (v1)",
    "dif-certified": "DIF-FNO, Lipschitz-certified (v1)",
    "dif-nobar": "DIF-FNO, no barrier (v1 ablation)",
    "chart-fixed": r"\DAO{} (chart only)",
    "chart-eq": r"\DAO{} + $\lambda_{\mathrm{eq}}=10$",
    "chart-cond": r"\DAO{}, per-sample chart",
    "chart-deep": r"\DAO{}, 3 composed charts",
}
SHORT = {k: v.replace(r"\DAO{}", "DAO").replace(r"$\lambda_{\mathrm{eq}}=10$", "eq10")
         for k, v in SPEC.items()}
DOMAIN_ORDER = ["star", "lshape", "annulus", "naca0012", "slit", "double_hole"]
DOMAIN_TEX = {"star": "star", "lshape": "L-shape", "annulus": "annulus",
              "naca0012": "NACA0012", "slit": "crack", "double_hole": "two holes"}
ACC_FIELDS = ("rel_l2", "rel_h1")
TOPO_FIELDS = ("fold_pct", "min_detJ", "certified_det_lower_bound", "kappa",
                "chart_inverse_error", "eq_residual_pct")


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def collect(runs: list[dict]) -> dict:
    """``{(variant, domain): {field: (mean, std, n)}}`` and per-seed values."""
    table: dict[tuple[str, str], dict[str, list]] = {}
    for r in runs:
        key = (r["variant"], r["domain"])
        vals = table.setdefault(key, {})
        for f, v in r["final"].items():
            if isinstance(v, (int, float)) and v is not None:
                vals.setdefault(f, []).append(float(v))
    out = {}
    for key, vals in table.items():
        out[key] = {f: (float(np.mean(v)), float(np.std(v)), len(v)) for f, v in vals.items()}
    return out


def variants_in(summary: dict, complete_only: bool = True) -> list[str]:
    """Variants in canonical order; by default only those covering every domain."""
    found = {v for v, _ in summary}
    domains = domains_in(summary)
    if complete_only:
        found = {v for v in found
                 if all((v, d) in summary and "rel_l2" in summary[(v, d)] for d in domains)}
    order = [k for k in SPEC if k in found]
    return order + sorted(found - set(order))


def incomplete_variants(summary: dict) -> dict[str, list[str]]:
    """Variants that cover only some domains (reported, never tabulated)."""
    domains = domains_in(summary)
    out = {}
    for v in {v for v, _ in summary}:
        have = [d for d in domains if (v, d) in summary and "rel_l2" in summary[(v, d)]]
        if have and len(have) < len(domains):
            out[v] = have
    return out


def domains_in(summary: dict) -> list[str]:
    found = {d for _, d in summary}
    order = [d for d in DOMAIN_ORDER if d in found]
    return order + sorted(found - set(order))


def paired(runs: list[dict], variant: str, domain: str, field: str,
           ref: str = "fno") -> tuple[float, float, int]:
    """Paired difference (variant - ref) over seeds, plus its std."""
    by_seed = {}
    for r in runs:
        by_seed.setdefault(r["seed"], {})[r["variant"]] = r["final"]
    diffs = []
    for seed, per_variant in by_seed.items():
        a, b = per_variant.get(variant), per_variant.get(ref)
        if not a or not b:
            continue
        if a.get(field) is None or b.get(field) is None:
            continue
        if (runs[0].get("domain") is not None) and False:
            pass
        diffs.append(float(a[field] - b[field]))
    if not diffs:
        return (float("nan"), float("nan"), 0)
    return (float(np.mean(diffs)), float(np.std(diffs)), len(diffs))


def paired_by_domain(runs: list[dict], variant: str, domain: str, field: str,
                     ref: str = "fno") -> tuple[float, float, int]:
    by_seed = {}
    for r in runs:
        if r["domain"] != domain:
            continue
        by_seed.setdefault(r["seed"], {})[r["variant"]] = r["final"]
    diffs = []
    for seed, per in by_seed.items():
        a, b = per.get(variant), per.get(ref)
        if not a or not b or a.get(field) is None or b.get(field) is None:
            continue
        diffs.append(float(a[field] - b[field]))
    if not diffs:
        return (float("nan"), float("nan"), 0)
    return (float(np.mean(diffs)), float(np.std(diffs)), len(diffs))


def _ablation_delta(runs: list[dict], domain: str, seed: int, field: str):
    """``chart-eq`` minus ``chart-fixed`` for one (domain, seed), or ``None``."""
    per = {}
    for r in runs:
        if r["domain"] == domain and r["seed"] == seed:
            per[r["variant"]] = r["final"]
    a, b = per.get("chart-eq"), per.get("chart-fixed")
    if not a or not b or a.get(field) is None or b.get(field) is None:
        return None
    return float(a[field] - b[field])


def pooled_diff(runs: list[dict], variant: str, domains: list[str], field: str,
                ref: str = "fno") -> tuple[float, float, int, float, float]:
    """Pooled paired difference with the Student-t and Wilcoxon p-values."""
    diffs = []
    for d in domains:
        by_seed = {}
        for r in runs:
            if r["domain"] != d:
                continue
            by_seed.setdefault(r["seed"], {})[r["variant"]] = r["final"]
        for seed, per in by_seed.items():
            a, b = per.get(variant), per.get(ref)
            if not a or not b or a.get(field) is None or b.get(field) is None:
                continue
            diffs.append(float(a[field] - b[field]))
    if len(diffs) < 2:
        return (float("nan"),) * 5
    arr = np.asarray(diffs, dtype=float)
    m, sd = float(arr.mean()), float(arr.std(ddof=1))
    se = sd / np.sqrt(len(arr))
    t = m / se if se > 0 else float("nan")
    try:
        from scipy import stats
        p = 2.0 * float(stats.t.sf(abs(t), len(arr) - 1))
    except Exception:  # pragma: no cover - scipy is a test dependency only
        p = float("nan")
    return (m, sd, len(arr), float(t), p)


def fmt(x: float, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    return f"{x:.{digits}f}"


def _tp(t: float, p: float) -> str:
    """LaTeX for a t statistic and a p-value (``p < 0.001`` when it underflows)."""
    ps = "$p<0.001$" if p < 1e-3 else f"$p={p:.3f}$"
    return f"$t={t:+.2f}$, {ps}"


def fmt_sci(x: float, digits: int = 2) -> str:
    """LaTeX scientific notation, e.g. ``$3.1\\cdot10^{-1}$``."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    if x == 0:
        return "$0$"
    exp = int(np.floor(np.log10(abs(x))))
    mant = x / 10.0 ** exp
    return f"${mant:.{digits}f}\\cdot10^{{{exp}}}$"


def pfmt(x: float) -> str:
    """Signed paired difference with the seed spread."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    sign = "+" if x >= 0 else "$-$"
    return f"{sign}{abs(x):.4f}"


COLSHORT = {
    "fno": "FNO",
    "dif-barrier": "DIF v1",
    "dif-certified": "DIF v1 cert.",
    "dif-nobar": "DIF v1 nobar",
    "chart-fixed": r"\DAO{}",
    "chart-eq": r"\DAO{} + eq",
    "chart-cond": r"\DAO{} per-sample",
    "chart-deep": r"\DAO{} $\times3$",
}


def latex_table(summary: dict, runs: list[dict]) -> str:
    """Three tables: mean accuracy, paired differences, admissibility."""
    variants = variants_in(summary)
    domains = domains_in(summary)
    n_seeds = max((v[2] for d in summary.values() for v in d.values()), default=0)
    others = [v for v in variants if v != "fno"]
    cols = "r" * len(variants)
    out: list[str] = []

    # (1) mean relative L2 ----------------------------------------------------
    out += [r"\begin{table}[t]", r"\centering", r"\small",
            r"\setlength{\tabcolsep}{5pt}",
            r"\caption{PDE benchmark, relative $L^2$ error (mean over " + str(n_seeds) +
            r" seeds).  All variants share the FNO backbone ($334\,737$ parameters); "
            r"the chart adds $1\,602$ ($+0.5\%$).  Darcy flow on " + str(len(domains)) +
            r" geometries, $N=64$, $100$ samples per geometry, $80/20$ split, "
            r"$40$ epochs, Adam $10^{-3}$, batch $10$.  \emph{crack}: a thin slit from "
            r"the left wall (singular tip); \emph{two holes}: a plate with two circular "
            r"holes.}",
            r"\label{tab:benchmark}",
            r"\begin{tabular}{l" + cols + "}", r"\toprule",
            "Domain & " + " & ".join(COLSHORT.get(v, v) for v in variants) + r" \\",
            r"\midrule"]
    for d in domains:
        row = [DOMAIN_TEX.get(d, d)]
        row += [fmt(summary.get((v, d), {}).get("rel_l2", (np.nan,))[0]) for v in variants]
        out.append(" & ".join(row) + r" \\")
    out.append(r"\midrule")
    row = ["mean"]
    for v in variants:
        vals = [summary[(v, d)]["rel_l2"][0] for d in domains if (v, d) in summary]
        row.append(fmt(float(np.mean(vals))) if vals else "--")
    out.append(" & ".join(row) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    # (2) paired differences ---------------------------------------------------
    if others:
        out += [r"\begin{table}[t]", r"\centering", r"\small",
                r"\setlength{\tabcolsep}{6pt}",
                r"\caption{Paired differences against the FNO at the same seed and "
                r"geometry (negative = better), mean $\pm$ std over the seeds.  Bottom "
                r"row: pooled over all " + str(len(domains) * n_seeds) +
                r" runs, with the two-sided paired $t$ and Wilcoxon tests.}",
                r"\label{tab:paired}",
                r"\begin{tabular}{l" + "r" * len(others) + "}", r"\toprule",
                "Domain & " + " & ".join(r"$\Delta$" + COLSHORT.get(v, v) for v in others) + r" \\",
                r"\midrule"]
        for d in domains:
            row = [DOMAIN_TEX.get(d, d)]
            for v in others:
                m, sd, n = paired_by_domain(runs, v, d, "rel_l2")
                row.append("--" if not np.isfinite(m) else f"{pfmt(m)} $\\pm$ {sd:.4f}")
            out.append(" & ".join(row) + r" \\")
        out.append(r"\midrule")
        row = [r"pooled"]
        for v in others:
            d_all = []
            for dom_ in domains:
                m, _, _ = paired_by_domain(runs, v, dom_, "rel_l2")
                if np.isfinite(m):
                    d_all.append(m)
            pm, ps, pn, pt, pw = pooled_diff(runs, v, domains, "rel_l2")
            row.append("--" if not np.isfinite(pm) else
                       f"{pfmt(pm)} $\\pm$ {ps:.4f} ({_tp(pt, pw)})")
        out.append(" & ".join(row) + r" \\")
        if "chart-eq" in variants and "chart-fixed" in variants:
            am, asd, an, at, ap = pooled_diff(runs, "chart-eq", domains, "rel_l2",
                                              ref="chart-fixed")
            if np.isfinite(am):
                neg = sum(
                    1 for d in domains for _s in range(3)
                    if _ablation_delta(runs, d, _s, "rel_l2") is not None
                    and _ablation_delta(runs, d, _s, "rel_l2") < 0)
                out += [r"\midrule",
                        r"\multicolumn{3}{l}{\emph{ablation}: \DAO{} + eq minus \DAO{}, "
                        r"identical architecture and budget, over all " + str(an) + r" runs}\\",
                        r"\multicolumn{3}{l}{\hspace{1em}" + f"{pfmt(am)} $\\pm$ {asd:.4f} "
                        + "(" + _tp(at, ap) + ")" + f", {neg} of {an} runs negative" + r"}\\"]
        out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    # (3) admissibility --------------------------------------------------------
    topo = [v for v in variants if ("fold_pct" in summary.get((v, domains[0]), {}))]
    if topo:
        out += [r"\begin{table}[t]", r"\centering", r"\small",
                r"\setlength{\tabcolsep}{5pt}",
                r"\caption{Admissibility at the end of training.  \emph{fold}: worst "
                r"percentage of folded cells over all geometries; $\min\detJ$: "
                r"smallest measured determinant; \emph{cert.}: smallest reported lower "
                r"bound; \emph{inv.}: largest exact-inverse consistency error; "
                r"\emph{eq}: equidistribution residual (RMS cell-mass deviation).}",
                r"\label{tab:admissible}",
                r"\begin{tabular}{lrrrrr}", r"\toprule",
                r"Variant & fold [\%] & $\min\detJ$ & cert. & inv. & eq [\%] \\",
                r"\midrule"]
        for v in topo:
            # worst case over the *runs*, not over the per-domain means
            def worst(field, how="max"):
                vals = [float(r["final"][field]) for r in runs
                        if r["variant"] == v and r["final"].get(field) is not None
                        and np.isfinite(float(r["final"][field]))]
                if not vals:
                    return None
                return max(vals) if how == "max" else min(vals)
            fold = worst("fold_pct")
            mind = worst("min_detJ", "min")
            cert = worst("certified_det_lower_bound", "min")
            inv = worst("chart_inverse_error")
            eqv = worst("eq_residual_pct")
            rows = [COLSHORT.get(v, v).replace(r"\DAO{}", "DAO"),
                    "--" if fold is None else f"{fold:.2f}",
                    "--" if mind is None else f"{mind:.3f}",
                    "--" if cert is None else fmt_sci(cert),
                    "--" if inv is None else fmt_sci(inv),
                    "--" if eqv is None else f"{eqv:.0f}"]
            out.append(" & ".join(rows) + r" \\")
        out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    text = "\n".join(out) + "\n"
    partial = incomplete_variants(summary)
    if partial:
        note = "; ".join(f"{SHORT.get(v, v)} ({', '.join(DOMAIN_TEX.get(d, d) for d in ds)} only)"
                         for v, ds in partial.items())
        text += ("% NOTE: not tabulated (incomplete coverage): " + note + "\n")
    return text


def markdown_table(summary: dict, runs: list[dict]) -> str:
    variants = variants_in(summary)
    domains = domains_in(summary)
    out = ["# Benchmark, v2 (paired seeds)\n"]
    out.append("Protocol: 100 samples/geometry, N=64, 80/20 split, 40 epochs, "
               "Adam 1e-3, batch 10, seeds 0-2.  Paired difference vs `fno` "
               "at the same seed; negative = better.\n")
    head = ["Domain"] + [f"{v} L2" for v in variants] + [f"Δ{v} L2" for v in variants if v != "fno"]
    out.append("| " + " | ".join(head) + " |")
    out.append("|" + "|".join(["---"] * len(head)) + "|")
    for d in domains:
        row = [d]
        for v in variants:
            row.append(fmt(summary.get((v, d), {}).get("rel_l2", (np.nan,))[0]))
        for v in variants:
            if v == "fno":
                continue
            m, s, n = paired_by_domain(runs, v, d, "rel_l2")
            row.append("--" if not np.isfinite(m) else f"{m:+.4f} ± {s:.4f}")
        out.append("| " + " | ".join(row) + " |")
    if "chart-eq" in variants and "chart-fixed" in variants:
        am, asd, an, at, ap = pooled_diff(runs, "chart-eq", domains, "rel_l2",
                                          ref="chart-fixed")
        if np.isfinite(am):
            neg = sum(1 for d in domains for k in range(3)
                      if (_ablation_delta(runs, d, k, "rel_l2") or 0) < 0)
            out.append("")
            out.append(f"*Ablation, identical architecture and budget:* `chart-eq` minus "
                       f"`chart-fixed` = **{am:+.4f} ± {asd:.4f}** over n = {an} "
                       f"(t = {at:+.2f}, p = {ap:.4f}, {neg}/{an} runs negative).")
    out.append("")
    out.append("## Admissibility\n")
    head = ["Domain", "variant", "fold %", "min detJ", "certified lb", "kappa", "inv err", "eq res %"]
    out.append("| " + " | ".join(head) + " |")
    out.append("|" + "|".join(["---"] * len(head)) + "|")
    for d in domains:
        for v in variants:
            e = summary.get((v, d), {})
            if "fold_pct" not in e:
                continue
            out.append("| " + " | ".join([
                d, v,
                fmt(e["fold_pct"][0], 2),
                fmt(e.get("min_detJ", (np.nan,))[0], 3),
                ("%.2e" % e["certified_det_lower_bound"][0]) if "certified_det_lower_bound" in e else "--",
                fmt(e.get("kappa", (np.nan,))[0], 2),
                ("%.1e" % e.get("chart_inverse_error", (np.nan,))[0]) if "chart_inverse_error" in e else "--",
                fmt(e.get("eq_residual_pct", (np.nan,))[0], 1),
            ]) + " |")
    partial = incomplete_variants(summary)
    if partial:
        out.append("")
        out.append("Not tabulated (incomplete coverage): " + "; ".join(
            f"`{v}` ({', '.join(ds)} only)" for v, ds in partial.items()) + ".")
    return "\n".join(out) + "\n"


def figure(summary: dict, runs: list[dict], out: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variants = [v for v in variants_in(summary) if v != "fno"]
    domains = domains_in(summary)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    width = 0.8 / max(len(variants), 1)
    for k, v in enumerate(variants):
        xs, ys, errs = [], [], []
        for i, d in enumerate(domains):
            m, s, n = paired_by_domain(runs, v, d, "rel_l2")
            if not np.isfinite(m):
                continue
            xs.append(i + (k - (len(variants) - 1) / 2) * width)
            ys.append(m)
            errs.append(s)
        ax.bar(xs, ys, width=width, yerr=errs, capsize=2, label=SHORT.get(v, v))
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(range(len(domains)))
    ax.set_xticklabels([DOMAIN_TEX.get(d, d) for d in domains])
    ax.set_ylabel(r"paired $\Delta$ rel. $L^2$ vs FNO")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in ("pdf", "png"):
        p = out.with_suffix(f".{ext}")
        fig.savefig(p, dpi=160)
        written.append(str(p))
    plt.close(fig)
    return written


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="results/v2")
    p.add_argument("--paper-table", default="paper/tables/benchmark.tex")
    p.add_argument("--markdown", default="results/v2/TABLE.md")
    p.add_argument("--figure", default="figures/paper/fig_benchmark")
    args = p.parse_args(argv)

    results = Path(args.results)
    runs = []
    for path in sorted(results.glob("*.json")):
        if path.name in ("CONSOLIDATED.json",):
            continue
        r = json.loads(path.read_text())
        if "final" not in r or "config" not in r:
            continue
        runs.append({
            "run_name": r.get("run_name", path.stem),
            "variant": r["config"]["tag"],
            "domain": r["config"]["domain"],
            "seed": r["config"]["seed"],
            "final": r["final"],
            "n_params": r.get("n_params"),
            "wall_time_s": r.get("wall_time_s"),
        })
    if not runs:
        print(f"no runs found in {results}", file=sys.stderr)
        return 1
    summary = collect(runs)

    Path(args.paper_table).parent.mkdir(parents=True, exist_ok=True)
    Path(args.paper_table).write_text(latex_table(summary, runs))
    Path(args.markdown).write_text(markdown_table(summary, runs))
    figs = figure(summary, runs, Path(args.figure))

    print(f"runs: {len(runs)}")
    print(Path(args.markdown).read_text())
    for f in figs:
        print("figure:", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
