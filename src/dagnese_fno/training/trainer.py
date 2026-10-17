"""Training loop for the DIF-FNO benchmark.

One function, :func:`train_one`, runs a complete experiment from a
:class:`TrainConfig` and returns every metric needed to reproduce the tables:
relative :math:`L^2`/:math:`H^1` errors, :math:`\\det J_\\phi` statistics, the
grid-folding percentage, the amount of deformation actually used, and -- for the
Lipschitz-constrained model -- the *certified* lower bound on
:math:`\\det J_\\phi`.

The baseline and DIF-FNO go through the same code path with the same backbone
hyper-parameters, so the two columns of the benchmark differ only by the
diffeomorphic branch.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from ..data import load_dataset
from ..geometry import (
    certified_det_lower_bound,
    det_jacobian_stats,
    displacement_stats,
    jacobian_det,
)
from ..losses import DiffeomorphicOperatorLoss
from ..metrics import relative_H1, relative_h1, relative_l2
from ..models import build_model
from ..utils import describe_environment, set_seed

__all__ = ["TrainConfig", "train_one", "max_gradient_norm"]


@dataclass
class TrainConfig:
    """Full description of one experiment."""

    # experiment
    domain: str = "star"
    model: str = "dif"                    # "fno" | "dif"
    seed: int = 0
    tag: str = "v4-repro"

    # data
    data_dir: str = "data"
    n_samples: int | None = None
    train_fraction: float = 0.8

    # optimisation
    epochs: int = 60
    batch_size: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float | None = None
    scheduler: str = "none"               # "none" | "cosine"

    # objective
    alpha_h1: float = 0.0
    lambda_barrier: float = 10.0
    barrier: str = "softplus"             # "softplus" | "log" | "hinge" | "none"
    tau: float = 0.05
    margin: float = 0.0
    normalize_by_mask: bool = True

    # backbone (identical for both models)
    width: int = 24
    modes1: int = 12
    modes2: int = 12
    n_layers: int = 2
    fc_channels: int = 64
    init: str = "v4"

    # diffeomorphic branch
    diffeo_hidden: int = 32
    diffeo_layers: int = 2
    deformation_scale: float = 0.1
    lipschitz_mode: str = "none"          # "none" | "spectral"
    lipschitz_const: float = 0.9
    conditioning: str = "pointwise"       # "pointwise" | "global"
    latent_dim: int = 8

    # bookkeeping
    device: str = "cpu"
    out_dir: str = "results"
    log_every: int = 10
    eval_every: int = 1
    save_checkpoint: bool = False
    early_stopping_patience: int | None = None
    use_wandb: bool = False
    wandb_project: str = "dagnese-dif-fno"
    wandb_mode: str = "offline"

    def model_kwargs(self) -> dict:
        return {
            "width": self.width, "modes1": self.modes1, "modes2": self.modes2,
            "n_layers": self.n_layers, "fc_channels": self.fc_channels,
            "init": self.init, "diffeo_hidden": self.diffeo_hidden,
            "diffeo_layers": self.diffeo_layers,
            "deformation_scale": self.deformation_scale,
            "lipschitz_mode": self.lipschitz_mode,
            "lipschitz_const": self.lipschitz_const,
            "conditioning": self.conditioning, "latent_dim": self.latent_dim,
        }

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def is_diffeomorphic(self) -> bool:
        return self.model.lower() in ("dif", "dif-fno", "dif_fno")


def max_gradient_norm(coeffs: torch.Tensor, mask: torch.Tensor, spacing: float) -> float:
    """:math:`\\max\\|\\nabla a\\|` over the dataset -- the :math:`\\kappa` of pointwise conditioning."""
    gy = torch.zeros_like(coeffs)
    gx = torch.zeros_like(coeffs)
    gy[:, :-1, :] = (coeffs[:, 1:, :] - coeffs[:, :-1, :]) / spacing
    gx[:, :, :-1] = (coeffs[:, :, 1:] - coeffs[:, :, :-1]) / spacing
    g = (gy ** 2 + gx ** 2).sqrt()
    m = mask.to(torch.bool).reshape(1, *mask.shape).expand_as(g)
    return float(g[m].max())


# --------------------------------------------------------------------------- #
def train_one(cfg: TrainConfig, verbose: bool = True) -> dict:
    """Run one experiment and return a JSON-serialisable result dictionary."""
    t_start = time.time()
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    if device.type == "cpu":
        n_threads = int(os.environ.get("OMP_NUM_THREADS", 0) or 0)
        if n_threads <= 0:
            n_threads = min(4, os.cpu_count() or 1)
        torch.set_num_threads(max(1, n_threads))

    # ---------------- data ---------------- #
    path = Path(cfg.data_dir) / f"darcy_{cfg.domain}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"dataset not found: {path}. Generate it with "
            f"`python scripts/generate_data.py --domains {cfg.domain}`"
        )
    data = load_dataset(path)
    coeffs = data["coeffs"].float().to(device)
    targets = data["targets"].float().to(device)
    coords = data["coords"].float().to(device)
    mask = data["mask"].bool().to(device)
    spacing = float(data["spacing"])
    if cfg.n_samples is not None:
        coeffs, targets = coeffs[: cfg.n_samples], targets[: cfg.n_samples]

    n_total = coeffs.shape[0]
    n_train = int(round(cfg.train_fraction * n_total))
    perm = torch.randperm(n_total, generator=torch.Generator().manual_seed(cfg.seed))
    train_idx, val_idx = perm[:n_train], perm[n_train:]
    kappa = (1.0 + max_gradient_norm(coeffs[train_idx], mask, spacing) ** 2) ** 0.5

    # ---------------- model ---------------- #
    model = build_model(cfg.model, **cfg.model_kwargs()).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
             if cfg.scheduler == "cosine" else None)
    criterion = DiffeomorphicOperatorLoss(
        alpha_h1=cfg.alpha_h1, lambda_barrier=cfg.lambda_barrier if cfg.is_diffeomorphic else 0.0,
        barrier=cfg.barrier, tau=cfg.tau, margin=cfg.margin,
        normalize_by_mask=cfg.normalize_by_mask,
    )

    run_name = f"{cfg.tag}_{cfg.model}_{cfg.domain}_seed{cfg.seed}"
    wandb_run = None
    if cfg.use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(project=cfg.wandb_project, name=run_name,
                                   config=cfg.to_dict(), mode=cfg.wandb_mode, reinit=True)
        except Exception as exc:  # pragma: no cover - optional dependency
            print(f"[warn] wandb unavailable ({exc}); continuing without it")
            wandb_run = None

    # ---------------- helpers ---------------- #
    def forward_batch(idx: torch.Tensor):
        a_b = coeffs[idx]
        u_t = targets[idx]
        c_b = coords.unsqueeze(0).expand(len(idx), -1, -1, -1).contiguous()
        if cfg.is_diffeomorphic:
            u_p, xi = model(c_b, a_b, mask=mask)
            det_j = jacobian_det(xi, spacing=spacing)
            return u_p, u_t, xi, det_j
        x = torch.cat([c_b, a_b.unsqueeze(-1)], dim=-1)
        return model(x), u_t, None, None

    @torch.no_grad()
    def evaluate(idx: torch.Tensor) -> dict:
        model.eval()
        l2, h1, H1, dets, folds, disp = [], [], [], [], [], []
        for i in range(0, len(idx), cfg.batch_size):
            b = idx[i:i + cfg.batch_size]
            u_p, u_t, xi, det_j = forward_batch(b)
            up, ut = u_p * mask, u_t * mask
            l2.append(float(relative_l2(up, ut, mask)))
            h1.append(float(relative_h1(up, ut, mask, spacing)))
            H1.append(float(relative_H1(up, ut, mask, spacing)))
            if det_j is not None:
                stats = det_jacobian_stats(det_j, mask)
                dets.append(stats["min"])
                folds.append(stats["fold_pct"])
                disp.append(displacement_stats(xi, coords, mask)["mean"])
        out = {
            "rel_l2": float(np.mean(l2)),
            "rel_h1": float(np.mean(h1)),
            "rel_H1": float(np.mean(H1)),
        }
        if dets:
            out.update({
                "min_detJ": float(np.min(dets)),
                "fold_pct": float(np.mean(folds)),
                "displacement_mean": float(np.mean(disp)),
            })
        else:
            out.update({"min_detJ": None, "fold_pct": None, "displacement_mean": None})
        if cfg.is_diffeomorphic:
            lip = model.lipschitz_bound()
            k = kappa if cfg.conditioning == "pointwise" else 1.0
            out["lipschitz_bound"] = float(lip)
            out["certified_det_lower_bound"] = float(certified_det_lower_bound(lip, k))
            out["is_certified"] = bool(model.diffeo.is_certified)
            out["kappa"] = float(k)
        model.train()
        return out

    # ---------------- loop ---------------- #
    history: list[dict] = []
    best = {"rel_l2": float("inf"), "epoch": -1}
    epochs_without_improvement = 0
    model.train()
    for epoch in range(cfg.epochs):
        t0 = time.time()
        running = {"total": 0.0, "mse": 0.0, "h1": 0.0, "barrier": 0.0}
        order = torch.randperm(n_train)
        n_batches = 0
        for i in range(0, n_train, cfg.batch_size):
            b = train_idx[order[i:i + cfg.batch_size]]
            u_p, u_t, xi, det_j = forward_batch(b)
            loss, comps = criterion(u_p * mask, u_t * mask, mask, det_j, spacing=spacing)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            running["total"] += float(loss.detach())
            running["mse"] += float(comps["mse"].detach())
            running["h1"] += float(comps["h1"].detach())
            running["barrier"] += float(comps["barrier"].detach())
            n_batches += 1
        if sched is not None:
            sched.step()

        record = {
            "epoch": epoch,
            "lr": opt.param_groups[0]["lr"],
            "train_loss": running["total"] / max(n_batches, 1),
            "train_mse": running["mse"] / max(n_batches, 1),
            "train_h1": running["h1"] / max(n_batches, 1),
            "train_barrier": running["barrier"] / max(n_batches, 1),
            "epoch_time_s": time.time() - t0,
        }
        if cfg.eval_every > 0 and (epoch % cfg.eval_every == 0 or epoch == cfg.epochs - 1):
            record.update({f"val_{k}": v for k, v in evaluate(val_idx).items()})
            if record["val_rel_l2"] < best["rel_l2"] - 1e-12:
                best = {"rel_l2": record["val_rel_l2"], "epoch": epoch}
                epochs_without_improvement = 0
                if cfg.save_checkpoint:
                    ckpt_dir = Path(cfg.out_dir) / "checkpoints"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    torch.save({"model": model.state_dict(), "config": cfg.to_dict(),
                                "epoch": epoch}, ckpt_dir / f"{run_name}.pt")
            else:
                epochs_without_improvement += 1
        history.append(record)

        if wandb_run is not None:  # pragma: no cover - requires wandb
            wandb_run.log(record)
        if verbose and (epoch % cfg.log_every == 0 or epoch == cfg.epochs - 1):
            extra = ""
            if record.get("val_min_detJ") is not None:
                extra = (f" bar={record['train_barrier']:.2e} "
                         f"minDetJ={record['val_min_detJ']:+.3e} "
                         f"fold={record['val_fold_pct']:.2f}% "
                         f"|d|={record['val_displacement_mean']:.3f}")
            print(f"[{run_name}] ep {epoch:3d}  mse={record['train_mse']:.3e}{extra}  "
                  f"L2={record['val_rel_l2']:.4f}  H1={record['val_rel_h1']:.4f}", flush=True)

        if (cfg.early_stopping_patience is not None
                and epochs_without_improvement >= cfg.early_stopping_patience):
            if verbose:
                print(f"[{run_name}] early stopping at epoch {epoch}")
            break

    final = dict(evaluate(val_idx))
    result = {
        "run_name": run_name,
        "config": cfg.to_dict(),
        "final": final,
        "best": best,
        "history": history,
        "n_train": int(n_train),
        "n_val": int(len(val_idx)),
        "n_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "wall_time_s": time.time() - t_start,
        "environment": describe_environment(),
        "data_sha": None,
        "kappa": kappa if cfg.is_diffeomorphic else None,
        "grid_spacing": spacing,
    }
    if wandb_run is not None:  # pragma: no cover
        wandb_run.summary.update(dict(final.items()))
        wandb_run.finish()
    return result


def save_result(result: dict, out_dir: str = "results", filename: str | None = None) -> Path:
    """Write ``result`` to ``<out_dir>/<filename>.json`` (history included)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = filename or f"{result['run_name']}.json"
    path = out / name
    path.write_text(json.dumps(result, indent=2, default=str))
    return path
