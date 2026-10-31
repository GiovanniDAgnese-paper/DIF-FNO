"""Train one model on one domain (FNO baseline or DIF-FNO).

Examples
--------
Baseline::

    python scripts/train.py --model fno --domain star --epochs 60 --seed 0

D'Agnese Chart Operator (certified, adaptive chart)::

    python scripts/train.py --model chart --domain star --lambda-eq 10

DIF-FNO with the D'Agnese barrier::

Certified (Lipschitz-constrained, folding-free by construction)::

    python scripts/train.py --model dif --domain star --lipschitz-mode spectral \
        --lipschitz-const 0.9 --conditioning global --deformation-scale 0.5

Every run writes ``results/<tag>_<model>_<domain>_seed<seed>.json`` containing
the full configuration, the per-epoch history and the final metrics.
"""

from __future__ import annotations

import argparse
import json

from ..data import DOMAINS
from ..training import TrainConfig, save_result, train_one


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = TrainConfig()
    # experiment
    p.add_argument("--domain", default=d.domain, choices=sorted(DOMAINS),
                   help="computational domain (default: %(default)s)")
    p.add_argument("--model", default=d.model,
                   choices=["fno", "dif", "chart", "dao", "chart-fno", "cpo"])
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--tag", default=d.tag)
    # data / optimisation
    p.add_argument("--data-dir", default=d.data_dir)
    p.add_argument("--n-samples", type=int, default=d.n_samples)
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--weight-decay", type=float, default=d.weight_decay)
    p.add_argument("--grad-clip", type=float, default=d.grad_clip)
    p.add_argument("--scheduler", default=d.scheduler, choices=["none", "cosine"])
    # objective
    p.add_argument("--alpha-h1", type=float, default=d.alpha_h1)
    p.add_argument("--lambda-barrier", type=float, default=d.lambda_barrier)
    p.add_argument("--barrier", default=d.barrier, choices=["softplus", "log", "hinge", "none"])
    p.add_argument("--tau", type=float, default=d.tau)
    p.add_argument("--margin", type=float, default=d.margin)
    p.add_argument("--no-mask-normalisation", dest="normalize_by_mask", action="store_false",
                   help="reproduce the v4 convention (MSE averaged over the bounding box)")
    # backbone
    p.add_argument("--width", type=int, default=d.width)
    p.add_argument("--modes1", type=int, default=d.modes1)
    p.add_argument("--modes2", type=int, default=d.modes2)
    p.add_argument("--n-layers", type=int, default=d.n_layers)
    p.add_argument("--fc-channels", type=int, default=d.fc_channels)
    p.add_argument("--init", default=d.init, choices=["v4", "normal"])
    # diffeo
    p.add_argument("--diffeo-hidden", type=int, default=d.diffeo_hidden)
    p.add_argument("--diffeo-layers", type=int, default=d.diffeo_layers)
    p.add_argument("--deformation-scale", type=float, default=d.deformation_scale)
    p.add_argument("--lipschitz-mode", default=d.lipschitz_mode, choices=["none", "spectral"])
    p.add_argument("--lipschitz-const", type=float, default=d.lipschitz_const)
    p.add_argument("--conditioning", default=d.conditioning, choices=["pointwise", "global"])
    p.add_argument("--latent-dim", type=int, default=d.latent_dim)
    # certified chart branch (D'Agnese Chart Operator)
    p.add_argument("--chart-hidden", type=int, default=d.chart_hidden)
    p.add_argument("--chart-layers", type=int, default=d.chart_layers)
    p.add_argument("--chart-fourier", type=int, default=d.chart_fourier)
    p.add_argument("--chart-activation", default=d.chart_activation)
    p.add_argument("--chart-conditioning", default=d.chart_conditioning,
                   choices=["none", "global"])
    p.add_argument("--chart-alpha", type=float, default=d.chart_alpha)
    p.add_argument("--chart-min-increment", type=float, default=d.chart_min_increment)
    p.add_argument("--lambda-eq", type=float, default=d.lambda_eq)
    p.add_argument("--eq-beta", type=float, default=d.eq_beta)
    p.add_argument("--eq-floor", type=float, default=d.eq_floor)
    p.add_argument("--eq-smooth", type=int, default=d.eq_smooth)
    p.add_argument("--loss-space", default=d.loss_space, choices=["physical", "chart"])
    # bookkeeping
    p.add_argument("--device", default=d.device)
    p.add_argument("--out-dir", default=d.out_dir)
    p.add_argument("--log-every", type=int, default=d.log_every)
    p.add_argument("--eval-every", type=int, default=d.eval_every)
    p.add_argument("--save-checkpoint", action="store_true")
    p.add_argument("--early-stopping-patience", type=int, default=d.early_stopping_patience)
    p.add_argument("--use-wandb", action="store_true")
    p.add_argument("--wandb-project", default=d.wandb_project)
    p.add_argument("--wandb-mode", default=d.wandb_mode)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def config_from_args(args) -> TrainConfig:
    fields = {k: v for k, v in vars(args).items()
              if k not in ("quiet", "no_mask_normalisation")}
    return TrainConfig(**fields)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = config_from_args(args)
    result = train_one(cfg, verbose=not args.quiet)
    path = save_result(result, cfg.out_dir)
    if not args.quiet:
        print("\nFinal metrics:")
        print(json.dumps(result["final"], indent=2))
        print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
