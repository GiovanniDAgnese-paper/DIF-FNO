#!/usr/bin/env python
"""Wall-clock cost of the chart: one training step at 64x64, batch 10, 2 threads.

The numbers quoted in the README, the paper and ``docs/DESIGN_NOTES.md`` come
from this script.  It measures a complete training step (forward, backward and
optimiser update) and reports the median of the steady-state steps, so that the
first-call overhead of the FFT plans and of ``grid_sample`` is excluded.

Usage::

    python scripts/bench_timing.py --resolution 64 --batch 10 --threads 2
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from dagnese_fno.models import DagneseChartOperator, FNO2d  # noqa: E402


def bench(model, coords: torch.Tensor, a: torch.Tensor, target: torch.Tensor,
          chart: bool, steps: int = 15) -> float:
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    times = []
    for _ in range(steps):
        t0 = time.perf_counter()
        if chart:
            u, _aux = model(coords, a)
        else:
            u = model(torch.cat([coords, a.unsqueeze(-1)], dim=-1))
        loss = ((u.squeeze(1) - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        times.append(time.perf_counter() - t0)
    return statistics.median(times[3:])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--resolution", type=int, default=64)
    p.add_argument("--batch", type=int, default=10)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--steps", type=int, default=15)
    args = p.parse_args(argv)

    torch.set_num_threads(args.threads)
    n, b = args.resolution, args.batch
    t = torch.linspace(-1.0, 1.0, n)
    coords = torch.stack(torch.meshgrid(t, t, indexing="ij"), dim=-1)
    coords = coords.unsqueeze(0).expand(b, -1, -1, -1).contiguous()
    a = torch.rand(b, n, n)
    target = torch.rand(b, n, n)

    variants = {
        "fno": (FNO2d(in_channels=3, out_channels=1, width=24, modes1=12, modes2=12,
                      n_layers=2, fc_channels=64), False),
        "dao": (DagneseChartOperator(), True),
        "dao+global": (DagneseChartOperator(chart_conditioning="global"), True),
    }
    base = None
    print(f"{'model':12s} {'ms/step':>8s} {'x':>6s} {'params':>9s}")
    for name, (model, chart) in variants.items():
        ms = 1000.0 * bench(model, coords, a, target, chart, args.steps)
        base = ms if base is None else base
        print(f"{name:12s} {ms:8.1f} {ms / base:6.2f} "
              f"{sum(p.numel() for p in model.parameters()):9d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
