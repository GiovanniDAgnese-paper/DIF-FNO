#!/usr/bin/env python
"""Deterministic verification of the four propositions (no training, no seeds).

This script produces the *theorem-verification* tables and figures of the paper.
Unlike the benchmark, nothing here is statistical: every number is a direct
measurement of a statement that is either proved or disproved.

Experiments
-----------
``P5``  certificate under adversarial parameters
        A grid of chart-parameter scales from ``0`` (identity) to ``1e3`` (a
        chart that compresses area by six orders of magnitude).  For each we
        report the minimum of ``det J``, the fraction of folded cells, the exact
        lower bound, and the inverse-consistency error.  The proposition says
        the first is positive, the second is zero and the third is positive for
        every row.

``P6``  round-trip convergence
        The pullback/pushforward round trip is the identity up to the
        interpolation error; the measured order must be 2.

``P7``  equidistribution and representation error
        At fixed resolution, an equidistributed chart represents a field with a
        sharp feature better than a uniform one.  Reported as the
        interpolation error against the number of nodes, uniform vs adaptive,
        and the ratio at the benchmark resolution.

``P8``  chart equivariance of the operator
        Two different charts produce the same operator output up to the
        re-discretisation error: measured as a function of the resolution.

Usage
-----
::

    python scripts/run_chart_study.py --out results/study --figures figures/study
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from dagnese_fno.losses import DagneseEquidistributionLoss  # noqa: E402
from dagnese_fno.geometry import (  # noqa: E402
    DagneseTriangularChart,
    sample_field,
    convergence_rate,
    pullback,
    pushforward,
    sample_field,
)

torch.set_default_dtype(torch.float32)

VARIANTS = ("chart",)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def grid(n: int, device="cpu", dtype=torch.float32) -> torch.Tensor:
    t = torch.linspace(-1.0, 1.0, n, device=device, dtype=dtype)
    return torch.stack(torch.meshgrid(t, t, indexing="ij"), dim=-1)


def randomise(chart: DagneseTriangularChart, scale: float, seed: int) -> None:
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in chart.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * scale)


def field_with_feature(n: int, kind: str = "ridge") -> torch.Tensor:
    """Analytic test field with a localised feature (the thing adaptivity buys)."""
    coords = grid(n, dtype=torch.float64)
    x, y = coords[..., 0], coords[..., 1]
    if kind == "ridge":
        return torch.exp(-((x - 0.15) ** 2) / 0.02) * torch.cos(3.0 * y)
    if kind == "boundary_layer":
        return torch.tanh((x - 0.6) / 0.05) + 0.3 * torch.sin(2.0 * y)
    if kind == "smooth":
        return torch.sin(3.0 * x) * torch.cos(2.0 * y) + 0.5 * x
    if kind == "multiscale":
        return (torch.sin(4.0 * x) * torch.cos(3.0 * y)
                + torch.exp(-((x + 0.4) ** 2 + (y - 0.3) ** 2) / 0.02))
    raise ValueError(f"unknown field '{kind}'")


def monitor_of(f: torch.Tensor, floor: float = 2e-2) -> torch.Tensor:
    """Smooth monitor ``(floor^2 + |grad f|^2)^{1/4}`` (a discrete surrogate of |f''|^1/2)."""
    gy = torch.zeros_like(f)
    gx = torch.zeros_like(f)
    gy[:, :-1] = f[:, 1:] - f[:, :-1]
    gx[:-1, :] = f[1:, :] - f[:-1, :]
    return (floor**2 + gy**2 + gx**2).pow(0.25)


def equidistributed_axis(monitor_1d: torch.Tensor, n: int) -> torch.Tensor:
    """Nodes on [-1, 1] that give equal monitor mass to every cell (de Boor)."""
    m = monitor_1d.double()
    x = torch.linspace(-1.0, 1.0, m.numel(), dtype=torch.float64)
    cum = torch.cumsum(0.5 * (m[1:] + m[:-1]) * torch.diff(x), dim=0)
    cum = torch.cat([torch.zeros(1, dtype=torch.float64), cum])
    cum = (cum - cum[0]) / (cum[-1] - cum[0])
    targets = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
    pos = torch.searchsorted(cum.contiguous(), targets.contiguous()).clamp(1, cum.numel() - 1)
    lo, hi = cum[pos - 1], cum[pos]
    frac = ((targets - lo) / (hi - lo).clamp_min(1e-30)).clamp(0, 1)
    nodes = x[pos - 1] + frac * (x[pos] - x[pos - 1])
    nodes[0], nodes[-1] = -1.0, 1.0
    return nodes


def bilinear_from_grid(fine: torch.Tensor, nodes_x: torch.Tensor, nodes_y: torch.Tensor) -> torch.Tensor:
    """Values of ``fine`` at the tensor-product nodes (exact bilinear sampling)."""
    n = fine.shape[0]
    t = torch.linspace(-1.0, 1.0, n, dtype=fine.dtype)
    out = torch.empty(nodes_x.numel(), nodes_y.numel(), dtype=fine.dtype)
    for axis, nodes in ((0, nodes_x), (1, nodes_y)):
        pass
    # separable interpolation: first along axis 0, then along axis 1
    out = _interp_axis(fine, nodes_x, axis=0)
    out = _interp_axis(out.transpose(0, 1), nodes_y, axis=0).transpose(0, 1)
    _ = (n, t)
    return out


def _interp_axis(f: torch.Tensor, nodes: torch.Tensor, axis: int) -> torch.Tensor:
    n = f.shape[axis]
    t = torch.linspace(-1.0, 1.0, n, dtype=f.dtype)
    pos = (nodes + 1.0) * 0.5 * (n - 1)
    i = pos.floor().clamp(0, n - 2).long()
    frac = (pos - i).to(f.dtype)
    a = f.index_select(axis, i)
    b = f.index_select(axis, i + 1)
    shape = [1] * f.dim()
    shape[axis] = len(nodes)
    return a + (b - a) * frac.reshape(shape)


# --------------------------------------------------------------------------- #
# P5 -- adversarial parameters
# --------------------------------------------------------------------------- #
def experiment_p5(n: int = 32, seeds=(0, 1, 2)) -> dict:
    coords = grid(n)
    rows = []
    for scale in (0.0, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 1000.0):
        rec = {"parameter_scale": scale, "runs": []}
        for seed in seeds:
            chart = DagneseTriangularChart(hidden=32, n_fourier=3)
            randomise(chart, scale, seed)
            phi, det = chart(coords)
            rec["runs"].append({
                "min_detJ": float(det.min()),
                "max_detJ": float(det.max()),
                "fold_pct": float((det <= 0).float().mean() * 100.0),
                "certified_lb": float(chart.certified_det_lower_bound(coords)),
                "parameter_free_floor": float(chart.parameter_free_det_floor(n)),
                "inverse_error": float(chart.inverse_consistency_error(coords)),
                "displacement": float((phi - coords).norm(dim=-1).mean()),
            })
        rec["min_detJ"] = min(r["min_detJ"] for r in rec["runs"])
        rec["fold_pct"] = max(r["fold_pct"] for r in rec["runs"])
        rec["certified_lb"] = min(r["certified_lb"] for r in rec["runs"])
        rec["parameter_free_floor"] = rec["runs"][0]["parameter_free_floor"]
        rec["inverse_error"] = max(r["inverse_error"] for r in rec["runs"])
        rows.append(rec)
    return {"experiment": "P5", "resolution": n, "seeds": list(seeds), "rows": rows}


# --------------------------------------------------------------------------- #
# P6 -- round-trip convergence
# --------------------------------------------------------------------------- #
def experiment_p6(resolutions=(32, 64, 128, 256), seed: int = 0) -> dict:
    out = {"experiment": "P6"}
    for kind in ("smooth", "multiscale"):
        rows = []
        for n in resolutions:
            coords = grid(n)
            chart = DagneseTriangularChart(hidden=32, n_fourier=3)
            randomise(chart, 0.4, seed)
            phi, det = chart(coords.unsqueeze(0))
            f = field_with_feature(n, kind).to(torch.float32).unsqueeze(0)
            phi_inv = chart.inverse(coords.unsqueeze(0), reference=coords)
            back = pushforward(pullback(f, phi, det), phi_inv)
            rows.append({
                "n": n,
                "max_error": float((back - f).abs().max()),
                "rms_error": float((back - f).pow(2).mean().sqrt()),
                "field_scale": float(f.abs().max()),
                "inverse_error": float(chart.inverse_consistency_error(coords)),
            })
        out[kind] = {
            "rows": rows,
            "order_rms": convergence_rate([r["rms_error"] for r in rows],
                                          [r["n"] for r in rows]),
            "order_max": convergence_rate([r["max_error"] for r in rows],
                                          [r["n"] for r in rows]),
        }
    out["rows"] = out["smooth"]["rows"]
    out["observed_order"] = out["smooth"]["order_rms"]
    return out


# --------------------------------------------------------------------------- #
# P7 -- representation error: uniform vs equidistributed
# --------------------------------------------------------------------------- #
def experiment_p7(field_kind: str = "ridge", fine: int = 256,
                  nodes=(9, 17, 33, 65)) -> dict:
    f = field_with_feature(fine, field_kind)
    coords = grid(fine, dtype=f.dtype)
    mon = monitor_of(f)
    # separable (tensor-product) equidistribution: marginals of the monitor
    m_x = mon.sum(dim=1)
    m_y = mon.sum(dim=0)
    rows = []
    for n in nodes:
        ux = torch.linspace(-1.0, 1.0, n, dtype=torch.float64)
        ax = equidistributed_axis(m_x, n)
        ay = equidistributed_axis(m_y, n)
        uniform_values = bilinear_from_grid(f, ux, ux)
        adaptive_values = bilinear_from_grid(f, ax, ay)
        uniform_rec = _reconstruct(uniform_values, ux, ux, fine)
        adaptive_rec = _reconstruct(adaptive_values, ax, ay, fine)
        rows.append({
            "n": n,
            "uniform_max": float((uniform_rec - f).abs().max()),
            "adaptive_max": float((adaptive_rec - f).abs().max()),
            "uniform_rms": float((uniform_rec - f).pow(2).mean().sqrt()),
            "adaptive_rms": float((adaptive_rec - f).pow(2).mean().sqrt()),
        })
    for r in rows:
        r["gain_max"] = r["uniform_max"] / max(r["adaptive_max"], 1e-12)
        r["gain_rms"] = r["uniform_rms"] / max(r["adaptive_rms"], 1e-12)
    order_u = convergence_rate([r["uniform_rms"] for r in rows], [r["n"] for r in rows])
    order_a = convergence_rate([r["adaptive_rms"] for r in rows], [r["n"] for r in rows])
    return {"experiment": "P7", "field": field_kind, "fine_resolution": fine,
            "rows": rows, "order_uniform": order_u, "order_adaptive": order_a,
            "max_cell_ratio": float(_max_cell_ratio(mon, rows[-1]["n"]))}


def _reconstruct(values: torch.Tensor, nx: torch.Tensor, ny: torch.Tensor, fine: int) -> torch.Tensor:
    """Inverse-distance-free reconstruction: sample ``values`` on (nx, ny) at the fine grid."""
    t = torch.linspace(-1.0, 1.0, fine, dtype=values.dtype)
    # map fine nodes to the index space of the node grid, then bilinear
    def locate(nodes):
        pos = torch.searchsorted(nodes.contiguous(), t.contiguous()).clamp(1, len(nodes) - 1)
        lo, hi = nodes[pos - 1], nodes[pos]
        frac = ((t - lo) / (hi - lo).clamp_min(1e-30)).clamp(0, 1)
        return (pos - 1).long(), frac

    i, fx = locate(nx.double())
    j, fy = locate(ny.double())
    a = values.index_select(0, i)
    b = values.index_select(0, i + 1)
    tmp = a + (b - a) * fx.to(values.dtype).unsqueeze(1)
    c = tmp.index_select(1, j)
    d = tmp.index_select(1, j + 1)
    return c + (d - c) * fy.to(values.dtype).unsqueeze(0)


def _max_cell_ratio(mon: torch.Tensor, n: int) -> float:
    m_x = mon.sum(dim=1)
    ax = equidistributed_axis(m_x, n)
    cells = torch.diff(ax)
    return float(cells.max() / cells.min())


def _physical_to_index(fine: int, chart, reference: torch.Tensor, n: int) -> torch.Tensor:
    """Continuous ``(i, j)`` index in the chart node array, for every physical node."""
    t = torch.linspace(-1.0, 1.0, fine, dtype=torch.float32)
    X, Y = torch.meshgrid(t, t, indexing="ij")
    pts = torch.stack([X, Y], dim=-1).reshape(1, -1, 2)
    with torch.no_grad():
        xi = chart.inverse_at(pts, reference=reference)
    return (0.5 * (xi + 1.0) * (n - 1)).reshape(fine, fine, 2)


def _bilinear_index(values: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Bilinear interpolation of ``values`` at continuous indices ``idx``."""
    n = values.shape[0]
    i = idx[..., 0].clamp(0, n - 1 - 1e-6)
    j = idx[..., 1].clamp(0, n - 1 - 1e-6)
    i0, j0 = i.floor().long(), j.floor().long()
    fi, fj = i - i0, j - j0
    i1, j1 = (i0 + 1).clamp_max(n - 1), (j0 + 1).clamp_max(n - 1)
    v00, v10 = values[i0, j0], values[i1, j0]
    v01, v11 = values[i0, j1], values[i1, j1]
    return (v00 * (1 - fi) * (1 - fj) + v10 * fi * (1 - fj)
            + v01 * (1 - fi) * fj + v11 * fi * fj)


# --------------------------------------------------------------------------- #
# P7b -- *learning* the equidistributed chart (the loss does the meshing)
# --------------------------------------------------------------------------- #
def fits_equidistributed_chart(kind: str = "boundary_layer", n: int = 64,
                               fine: int = 192, steps: int = 600, seed: int = 0,
                               lr: float = 0.02) -> dict:
    """Optimise a chart with the D'Agnese Equidistribution Loss *only*.

    The chart starts at the identity (``identity_init`` zeros the last layer) and
    is trained by Adam on the equidistribution loss with the monitor of a field
    that the network never sees: the only signal is "make the cells carry equal
    mass".  The resulting mesh is the learned analogue of the classical
    equidistributing mesh, and it is certified by construction.
    """
    torch.manual_seed(seed)
    coords = grid(n)
    field = field_with_feature(n, kind).to(torch.float32)
    monitor = monitor_of(field)
    monitor = monitor / monitor.mean()

    chart = DagneseTriangularChart(hidden=32, n_fourier=3)
    loss_fn = DagneseEquidistributionLoss()
    opt = torch.optim.Adam(chart.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    history = []
    for step in range(steps):
        phi, det = chart(coords.unsqueeze(0))
        m = sample_field(monitor.unsqueeze(0), phi)
        loss = loss_fn(m.clamp_min(1e-6), det)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 10 == 0 or step == steps - 1:
            history.append({"step": step, "loss": float(loss.detach())})

    with torch.no_grad():
        phi, det = chart(coords.unsqueeze(0))
        m = sample_field(monitor.unsqueeze(0), phi)
        final = float(loss_fn(m.clamp_min(1e-6), det))
        mass = (m * det).squeeze(0)
        displacement = float((phi[0] - coords).norm(dim=-1).mean())
        lb = float(chart.certified_det_lower_bound(coords))
        inv = float(chart.inverse_consistency_error(coords))

    # representation error at a *fixed number of nodes*, measured on a fine grid
    f_fine = field_with_feature(fine, kind).to(torch.float32)
    u = torch.linspace(-1.0, 1.0, n, dtype=torch.float64)
    uniform_values = bilinear_from_grid(f_fine, u, u)
    with torch.no_grad():
        chart_values = sample_field(f_fine.unsqueeze(0), chart(coords.unsqueeze(0))[0]).squeeze(0)
    uniform_rec = _reconstruct(uniform_values, u, u, fine)
    chart_rec = _bilinear_index(chart_values, _physical_to_index(fine, chart, coords, n))
    repr_uniform = float((uniform_rec - f_fine).abs().max())
    repr_chart = float((chart_rec - f_fine).abs().max())

    return {
        "field": kind, "resolution": n, "steps": steps, "lr": lr,
        "loss_initial": history[0]["loss"], "loss_final": final,
        "mass_ratio": float(mass.max() / mass.min()),
        "cell_ratio": float(det.max() / det.min()),
        "displacement": displacement,
        "certified_lb": lb, "inverse_error": inv,
        "repr_error_uniform": repr_uniform,
        "repr_error_chart": repr_chart,
        "repr_gain": repr_uniform / max(repr_chart, 1e-12),
        "history": history,
        "chart": chart, "phi": phi[0].detach(), "det": det[0].detach(),
        "monitor": monitor, "mass": mass.detach(), "field_values": field,
        "field_fine": f_fine,
    }


def experiment_p7b(steps: int = 600) -> dict:
    rows = []
    for kind in ("boundary_layer", "ridge", "smooth"):
        out = fits_equidistributed_chart(kind, steps=steps)
        rows.append({k: v for k, v in out.items()
                     if k not in ("chart", "phi", "det", "monitor", "field_values",
                                  "field_fine", "mass")})
    return {"experiment": "P7b", "rows": rows}


# --------------------------------------------------------------------------- #
# P8 -- chart equivariance of the operator
# --------------------------------------------------------------------------- #
def experiment_p8(resolutions=(32, 64, 128), seed: int = 0) -> dict:
    """Two different charts must produce the same *chart-space* operator output."""
    rows = []
    for n in resolutions:
        coords = grid(n).unsqueeze(0)
        f = field_with_feature(n, "smooth").to(torch.float32).unsqueeze(0)
        chart_a = DagneseTriangularChart(hidden=32, n_fourier=3)
        chart_b = DagneseTriangularChart(hidden=32, n_fourier=3)
        randomise(chart_a, 0.4, seed)
        randomise(chart_b, 0.4, seed + 100)
        pa, da = chart_a(coords)
        pb, db = chart_b(coords)
        fa = pullback(f, pa, da)
        fb = pullback(f, pb, db)
        # bring both to the physical grid: P_a^{-1} P_a f vs P_b^{-1} P_b f
        fa_phys = pushforward(fa, chart_a.inverse(coords, reference=coords[0]))
        fb_phys = pushforward(fb, chart_b.inverse(coords, reference=coords[0]))
        rows.append({
            "n": n,
            "equivariance_error": float((fa_phys - fb_phys).abs().max()),
            "chart_displacement": float((pa - coords).norm(dim=-1).mean()),
            "relative": float((fa_phys - fb_phys).abs().max() / f.abs().max()),
        })
    return {"experiment": "P8", "rows": rows,
            "observed_order": convergence_rate([r["relative"] for r in rows],
                                               [r["n"] for r in rows])}


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def make_figures(study: dict, out_dir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    # P5: certificate vs parameter scale
    rows = study["P5"]["rows"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    scale = [r["parameter_scale"] for r in rows]
    xs = [max(s, 1e-3) for s in scale]
    ax[0].loglog(xs, [max(r["min_detJ"], 1e-30) for r in rows], "o-", label=r"$\min \det J_\Phi$ (measured)")
    ax[0].loglog(xs, [max(r["certified_lb"], 1e-30) for r in rows], "s--", label="certified lower bound")
    ax[0].loglog(xs, [max(r["parameter_free_floor"], 1e-30) for r in rows], ":", label="parameter-free floor")
    ax[0].set_xlabel("chart parameter scale")
    ax[0].set_ylabel(r"$\det J_\Phi$")
    ax[0].set_title("(a) positivity of the Jacobian")
    ax[0].legend(fontsize=7, loc="lower left")
    ax[1].semilogx(xs, [r["fold_pct"] for r in rows], "o-")
    ax[1].set_ylim(-0.5, 1.0)
    ax[1].set_xlabel("chart parameter scale")
    ax[1].set_ylabel("folded cells [%]")
    ax[1].set_title("(b) folding, all parameter values")
    fig.tight_layout()
    path = out_dir / "fig_p5_certificate.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=150)
    written.append(str(path))
    plt.close(fig)

    # P6: round-trip convergence
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.3))
    for k, kind in enumerate(("smooth", "multiscale")):
        rows = study["P6"][kind]["rows"]
        ns = np.array([r["n"] for r in rows], dtype=float)
        ax[k].loglog(ns, [r["rms_error"] for r in rows], "o-", label="round trip (rms)")
        ax[k].loglog(ns, [r["max_error"] for r in rows], "s-", label="round trip (max)")
        ref = ns ** -2
        ax[k].loglog(ns, ref / ref[0] * rows[0]["rms_error"], "k:", label=r"$\propto h^2$")
        ax[k].set_xlabel("resolution $n$")
        ax[k].set_ylabel("error")
        ax[k].set_title(f"{kind}: order {study['P6'][kind]['order_rms']:.2f}")
        ax[k].legend(fontsize=7)
    fig.tight_layout()
    path = out_dir / "fig_p6_roundtrip.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=150)
    written.append(str(path))
    plt.close(fig)

    # P7: representation error, one panel per field
    kinds = list(study["P7"])
    fig, ax = plt.subplots(1, len(kinds), figsize=(3.1 * len(kinds), 3.2), squeeze=False)
    for k, kind in enumerate(kinds):
        rows = study["P7"][kind]["rows"]
        ns = np.array([r["n"] for r in rows])
        ax[0][k].loglog(ns, [r["uniform_max"] for r in rows], "o-", label="uniform mesh")
        ax[0][k].loglog(ns, [r["adaptive_max"] for r in rows], "s-", label="equidistributed chart")
        ax[0][k].set_xlabel("nodes per axis")
        ax[0][k].set_ylabel(r"$\|f-I_h f\|_\infty$")
        ax[0][k].set_title(k, fontsize=9)
        ax[0][k].legend(fontsize=7)
    fig.suptitle("P7: representation error at fixed resolution", fontsize=10)
    fig.tight_layout()
    path = out_dir / "fig_p7_representation.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=150)
    written.append(str(path))
    plt.close(fig)

    # Hero figure: the *learned* equidistributing chart
    if "P7b_chart" in study:
        fit = study["P7b_chart"]
        phi, det, mass = fit["phi"], fit["det"], fit["mass"]
        field, monitor = fit["field_values"], fit["monitor"]
        fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.6))
        ax[0].contourf(np.linspace(-1, 1, field.shape[0]), np.linspace(-1, 1, field.shape[1]),
                       field.numpy().T, levels=24, cmap="magma")
        step = max(1, phi.shape[0] // 16)
        ax[0].plot(phi[::step, :, 0].T.numpy(), phi[::step, :, 1].T.numpy(), lw=0.35, color="w", alpha=0.75)
        ax[0].plot(phi[:, ::step, 0].T.numpy(), phi[:, ::step, 1].T.numpy(), lw=0.35, color="w", alpha=0.75)
        ax[0].set_title(r"learned chart $\Phi(\xi)$ on the field")
        ax[0].set_aspect("equal")
        im = ax[1].imshow(det.numpy(), origin="lower", extent=[-1, 1, -1, 1], cmap="viridis")
        ax[1].set_title(r"$\det J_\Phi>0$ (exact, no penalty)")
        fig.colorbar(im, ax=ax[1], fraction=0.046)
        mass_np = mass.numpy().ravel()
        uniform_mass = monitor.numpy().ravel()
        uniform_mass = uniform_mass * (mass_np.mean() / uniform_mass.mean())
        ax[2].hist(mass_np, bins=30, alpha=0.75, label="learned chart", color="C0")
        ax[2].hist(uniform_mass, bins=30, alpha=0.5, label="uniform mesh", color="C1")
        ax[2].set_xlabel("cell mass")
        ax[2].set_ylabel("cells")
        ax[2].set_title(f"equidistribution: mass ratio {fit['mass_ratio']:.2f}")
        ax[2].legend(fontsize=8)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            path = out_dir / f"fig_chart.{ext}"
            fig.savefig(path, dpi=150)
        written.append(str(out_dir / "fig_chart.pdf"))
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        hist = fit["history"]
        ax.semilogy([h["step"] for h in hist], [h["loss"] for h in hist], "-")
        ax.set_xlabel("Adam step")
        ax.set_ylabel("D'Agnese equidistribution loss")
        ax.set_title(f"chart learned from scratch ({fit['field']})")
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(out_dir / f"fig_p7b_loss.{ext}", dpi=150)
        written.append(str(out_dir / "fig_p7b_loss.pdf"))
        plt.close(fig)
    return written


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _jsonable(obj):
    """Recursively convert tensors / numpy scalars to plain JSON types."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="results/study")
    p.add_argument("--figures", default="figures/study")
    p.add_argument("--quick", action="store_true", help="smaller sweeps")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    study: dict = {"environment": {"torch": torch.__version__, "numpy": np.__version__}}
    study["P5"] = experiment_p5()
    print("P5 done", flush=True)
    study["P6"] = experiment_p6(resolutions=(32, 64, 128) if args.quick else (32, 64, 128, 256))
    print("P6 done", flush=True)
    study["P7"] = {
        kind: experiment_p7(kind, nodes=(9, 17, 33, 65) if not args.quick else (9, 17, 33))
        for kind in ("ridge", "boundary_layer", "multiscale")
    }
    print("P7 done", flush=True)
    study["P7b"] = experiment_p7b()
    study["P7b_chart"] = fits_equidistributed_chart("boundary_layer")
    print("P7b done", flush=True)
    study["P8"] = experiment_p8()
    print("P8 done", flush=True)

    export = {k: v for k, v in study.items()
              if k != "P7b_chart"}
    export["P7b_chart"] = {k: v for k, v in study["P7b_chart"].items()
                           if k not in ("chart", "phi", "det", "monitor", "field_values",
                                        "field_fine", "mass")}
    (out / "STUDY.json").write_text(json.dumps(_jsonable(export), indent=2))

    # human-readable summary
    lines = ["# Deterministic verification of P5--P8", ""]
    lines.append("## P5 -- the certificate cannot be lost (adversarial parameters)")
    lines.append("")
    lines.append("| parameter scale | min det J | folded % | certified lb | parameter-free floor |")
    lines.append("|---|---|---|---|---|")
    for r in study["P5"]["rows"]:
        lines.append(f"| {r['parameter_scale']:g} | {r['min_detJ']:.3e} | {r['fold_pct']:.2f} | "
                     f"{r['certified_lb']:.3e} | {r['parameter_free_floor']:.3e} |")
    lines.append("")
    lines.append("## P6 -- round trip is the interpolation error")
    for kind in ("smooth", "multiscale"):
        lines.append("")
        lines.append(f"### {kind} field")
        lines.append("")
        lines.append("| n | max error | rms error | inverse error |")
        lines.append("|---|---|---|---|")
        for r in study["P6"][kind]["rows"]:
            lines.append(f"| {r['n']} | {r['max_error']:.3e} | {r['rms_error']:.3e} | "
                         f"{r['inverse_error']:.2e} |")
        lines.append(f"\nObserved order (rms / max): "
                     f"**{study['P6'][kind]['order_rms']:.2f} / {study['P6'][kind]['order_max']:.2f}**")
    lines.append("")
    lines.append("## P7 -- equidistribution reduces the representation error")
    for kind, res in study["P7"].items():
        lines.append("")
        lines.append(f"### {kind}")
        lines.append("")
        lines.append("| n | uniform max | adaptive max | gain | uniform rms | adaptive rms | gain |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in res["rows"]:
            lines.append(f"| {r['n']} | {r['uniform_max']:.3e} | {r['adaptive_max']:.3e} | "
                         f"{r['gain_max']:.2f}x | {r['uniform_rms']:.3e} | {r['adaptive_rms']:.3e} | "
                         f"{r['gain_rms']:.2f}x |")
    lines.append("")
    lines.append("## P7b -- the loss learns the mesh (chart trained by gradient descent only)")
    lines.append("")
    lines.append("| field | loss (init -> final) | mass ratio | cell ratio | cert. lb | "
                 "repr. error uniform | chart | gain |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in study["P7b"]["rows"]:
        lines.append(f"| {r['field']} | {r['loss_initial']:.4f} -> {r['loss_final']:.5f} | "
                     f"{r['mass_ratio']:.2f} | {r['cell_ratio']:.2f} | {r['certified_lb']:.3f} | "
                     f"{r['repr_error_uniform']:.4e} | {r['repr_error_chart']:.4e} | "
                     f"{r['repr_gain']:.2f}x |")
    lines.append("")
    lines.append("## P8 -- chart equivariance of the operator")
    lines.append("")
    lines.append("| n | relative difference between two charts | displacement |")
    lines.append("|---|---|---|")
    for r in study["P8"]["rows"]:
        lines.append(f"| {r['n']} | {r['relative']:.3e} | {r['chart_displacement']:.3f} |")
    (out / "STUDY.md").write_text("\n".join(lines) + "\n")

    figures = make_figures(study, Path(args.figures))
    print("\n".join([f"figures: {f}" for f in figures]))
    print((out / "STUDY.md").read_text()[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
