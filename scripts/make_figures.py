#!/usr/bin/env python
"""Rebuild the benchmark figures from a consolidated result folder.

Usage::

    python scripts/make_figures.py --results results/repro --out figures/repro

Produces (PDF + PNG, publication ready):

* ``fig_l2_h1.pdf``      -- relative L2 and H1 error per domain and variant,
  with the standard deviation over the seeds;
* ``fig_topology.pdf``   -- grid folding and min det(J) (log scale), together
  with the certified lower bound where available;
* ``fig_displacement.pdf`` -- how far the learned map actually moves the grid,
  against the certified injectivity bound;
* ``fig_curves_<domain>.pdf`` -- validation L2 / H1 / min det(J) against the epoch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
HATCH = ["", "//", "..", "xx", "++"]


def _style(ax, ylabel, title=None):
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("  ->", out / f"{name}.pdf")


def fig_errors(summary, variants, domains, out):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharex=True)
    x = np.arange(len(domains))
    width = 0.8 / max(len(variants), 1)
    for ax, field, label in zip(axes, ("rel_l2", "rel_h1"),
                                ("relative $L^2$ error", "relative $H^1$ error")):
        for k, v in enumerate(variants):
            means, stds = [], []
            for d in domains:
                e = (summary.get(v, {}).get(d) or {}).get(field)
                means.append(e[0] if e else np.nan)
                stds.append(e[1] if e else 0.0)
            ax.bar(x + k * width - 0.4 + width / 2, means, width, yerr=stds,
                   label=v, color=PALETTE[k % len(PALETTE)],
                   hatch=HATCH[k % len(HATCH)], capsize=3, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(domains)
        _style(ax, label)
    axes[0].legend(fontsize=7, ncol=1, frameon=False)
    fig.suptitle("DIF-FNO benchmark (mean $\\pm$ s.d. over seeds)")
    fig.tight_layout()
    _save(fig, out, "fig_l2_h1")


def fig_topology(summary, variants, domains, out):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharex=True)
    x = np.arange(len(domains))
    width = 0.8 / max(len(variants), 1)

    # Folded cells (%)
    for k, v in enumerate(variants):
        vals = []
        for d in domains:
            e = (summary.get(v, {}).get(d) or {}).get("fold_pct")
            vals.append(e[0] if e else np.nan)
        axes[0].bar(x + k * width - 0.4 + width / 2, vals, width, label=v,
                    color=PALETTE[k % len(PALETTE)], hatch=HATCH[k % len(HATCH)],
                    edgecolor="white")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(domains)
    _style(axes[0], "grid folding (%)")

    # min det J and certified bound
    for k, v in enumerate(variants):
        mins, certs = [], []
        for d in domains:
            e = summary.get(v, {}).get(d) or {}
            mins.append(e["min_detJ"][0] if e.get("min_detJ") else np.nan)
            certs.append(e["certified_det_lower_bound"][0]
                         if e.get("certified_det_lower_bound") else np.nan)
        pos = x + k * width - 0.4 + width / 2
        # Bottom at 0: a negative bar means folded cells.
        axes[1].bar(pos, np.nan_to_num(mins, nan=0.0), width, label=f"{v}: min det(J)",
                    color=PALETTE[k % len(PALETTE)], hatch=HATCH[k % len(HATCH)],
                    edgecolor="white")
        if np.isfinite(certs).any():
            axes[1].scatter(pos, np.nan_to_num(certs, nan=0.0), marker="_", s=220,
                            color="black", zorder=5,
                            label=f"{v}: certified" if k == 0 else None)
    # symlog: behaves linearly near zero (so negative = folded values are visible)
    # and logarithmically away from it (min det J spans ~3 decades across variants).
    axes[1].set_yscale("symlog", linthresh=1e-2)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(domains)
    _style(axes[1], "$\\min \\det(J_\\phi)$ (log)")
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    _save(fig, out, "fig_topology")


def fig_displacement(summary, variants, domains, out):
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    x = np.arange(len(domains))
    width = 0.8 / max(len(variants), 1)
    for k, v in enumerate(variants):
        vals, certs = [], []
        for d in domains:
            e = summary.get(v, {}).get(d) or {}
            vals.append(e["displacement_mean"][0] if e.get("displacement_mean") else np.nan)
            certs.append(e["certified_det_lower_bound"][0]
                         if e.get("certified_det_lower_bound") else np.nan)
        if all(np.isnan(vals)):
            continue
        pos = x + k * width - 0.4 + width / 2
        ax.bar(pos, vals, width, label=v, color=PALETTE[k % len(PALETTE)],
               hatch=HATCH[k % len(HATCH)], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(domains)
    _style(ax, "mean $\\|\\phi(\\xi) - \\xi\\|$", "How much the learned map actually deforms")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    _save(fig, out, "fig_displacement")


def fig_curves(runs, domain, out):
    by_variant: dict[str, list[list[dict]]] = {}
    for r in runs:
        if r["domain"] != domain:
            continue
        by_variant.setdefault(r["variant"], []).append(r["history"])
    if not by_variant:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for k, (variant, histories) in enumerate(sorted(by_variant.items())):
        n = min(len(h) for h in histories)
        for field, ax, label in zip(
            ("val_rel_l2", "val_rel_h1", "val_min_detJ"), axes,
            ("val relative $L^2$", "val relative $H^1$", "val $\\min\\det(J)$")):
            series = np.array([[h[i].get(field) for i in range(n)] for h in histories],
                              dtype=float)
            if np.all(np.isnan(series)):      # metric not defined for this variant
                continue
            epochs = np.arange(n)
            mu, sd = np.nanmean(series, 0), np.nanstd(series, 0)
            ax.plot(epochs, mu, label=variant, color=PALETTE[k % len(PALETTE)])
            ax.fill_between(epochs, mu - sd, mu + sd, alpha=0.15,
                            color=PALETTE[k % len(PALETTE)])
            _style(ax, label)
            ax.set_xlabel("epoch")
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"Training curves -- {domain}")
    fig.tight_layout()
    _save(fig, out, f"fig_curves_{domain}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="results/repro")
    p.add_argument("--out", default="figures/repro")
    p.add_argument("--consolidated", default=None,
                   help="path to CONSOLIDATED.json (default: <results>/CONSOLIDATED.json)")
    args = p.parse_args(argv)

    results = Path(args.results)
    path = Path(args.consolidated) if args.consolidated else results / "CONSOLIDATED.json"
    if not path.exists():
        print(f"{path} not found: run scripts/run_benchmark.py first", file=sys.stderr)
        return 1
    data = json.loads(path.read_text())
    summary, runs = data["summary"], data["runs"]
    variants = [v for v in summary if v in {r["variant"] for r in runs}]
    domains = [d for d in summary[variants[0]] if d != "AVG"]

    out = Path(args.out)
    print(f"Figures for {len(variants)} variants x {len(domains)} domains -> {out}")
    fig_errors(summary, variants, domains, out)
    fig_topology(summary, variants, domains, out)
    fig_displacement(summary, variants, domains, out)

    list(json.loads((results / "CONSOLIDATED.json").read_text())["runs"])
    full = []
    for r in runs:
        candidate = results / f"{r['variant']}_{'dif' if r['variant'] != 'fno' else 'fno'}_{r['domain']}_seed{r['seed']}.json"
        if candidate.exists():
            full.append({**r, "history": json.loads(candidate.read_text())["history"]})
    for d in domains:
        fig_curves(full, d, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
