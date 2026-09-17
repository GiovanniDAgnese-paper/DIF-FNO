"""Honest training pipeline for FNO and DIF-FNO on REAL Darcy flow data."""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath("src"))
from dagnese_fno.barrier_loss import DAgneseBarrierLoss

import wandb

DEVICE = torch.device("cpu")


# ---------------- MODELS ----------------
class SpectralConv2d(nn.Module):
    def __init__(self, in_c, out_c, m1, m2):
        super().__init__()
        self.m1, self.m2 = m1, m2
        scale = 1.0 / (in_c * out_c)
        self.w1 = nn.Parameter(scale * torch.rand(in_c, out_c, m1, m2, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.rand(in_c, out_c, m1, m2, dtype=torch.cfloat))

    def forward(self, x):
        B = x.shape[0]
        xf = torch.fft.rfft2(x)
        out = torch.zeros(B, self.w1.shape[1], x.size(-2), x.size(-1)//2 + 1,
                          dtype=torch.cfloat, device=x.device)
        out[:, :, :self.m1, :self.m2] = torch.einsum(
            "bixy,ioxy->boxy", xf[:, :, :self.m1, :self.m2], self.w1)
        out[:, :, -self.m1:, :self.m2] = torch.einsum(
            "bixy,ioxy->boxy", xf[:, :, -self.m1:, :self.m2], self.w2)
        return torch.fft.irfft2(out, s=(x.size(-2), x.size(-1)))


class FNO2d(nn.Module):
    def __init__(self, in_c=3, width=24, m1=12, m2=12):
        super().__init__()
        self.fc0 = nn.Linear(in_c, width)
        self.c0 = SpectralConv2d(width, width, m1, m2)
        self.c1 = SpectralConv2d(width, width, m1, m2)
        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.fc1 = nn.Linear(width, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):  # x: (B, H, W, C)
        x = self.fc0(x).permute(0, 3, 1, 2)
        x = F.gelu(self.c0(x) + self.w0(x))
        x = F.gelu(self.c1(x) + self.w1(x))
        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        return self.fc2(x).squeeze(-1)


class DiffeomorphicMap(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 2),
        )

    def forward(self, coords):  # coords: (B,H,W,2) -> (B,H,W,2)
        return coords + 0.1 * self.net(coords)


class DIFFNO2d(nn.Module):
    def __init__(self, width=24, m1=12, m2=12):
        super().__init__()
        self.diffeo = DiffeomorphicMap()
        self.fno = FNO2d(in_c=3, width=width, m1=m1, m2=m2)

    def forward(self, coords, a):
        xi = self.diffeo(coords)                    # (B,H,W,2)
        x = torch.cat([xi, a.unsqueeze(-1)], -1)    # (B,H,W,3)
        u = self.fno(x)
        return u, xi


# ---------------- METRICS ----------------
def rel_l2(pred, true, mask):
    m = mask.unsqueeze(0)
    num = ((pred - true)**2 * m).sum()
    den = (true**2 * m).sum() + 1e-12
    return (num / den).sqrt().item()


def rel_h1(pred, true, mask):
    def grad(f):
        gy = torch.zeros_like(f); gx = torch.zeros_like(f)
        gy[:, :-1, :] = f[:, 1:, :] - f[:, :-1, :]
        gx[:, :, :-1] = f[:, :, 1:] - f[:, :, :-1]
        return gy, gx
    py, px = grad(pred); ty, tx = grad(true)
    m = mask.unsqueeze(0)
    num = (((py-ty)**2 + (px-tx)**2) * m).sum()
    den = ((ty**2 + tx**2) * m).sum() + 1e-12
    return (num / den).sqrt().item()


def jacobian_det(xi):
    """xi: (B,H,W,2) -> det(J) per pixel (B,H,W)."""
    gy = torch.zeros_like(xi); gx = torch.zeros_like(xi)
    gy[:, :-1, :, :] = xi[:, 1:, :, :] - xi[:, :-1, :, :]
    gx[:, :, :-1, :] = xi[:, :, 1:, :] - xi[:, :, :-1, :]
    # xi[...,0]=xi_x, xi[...,1]=xi_y
    # det = d(xi_x)/dx * d(xi_y)/dy - d(xi_x)/dy * d(xi_y)/dx
    dx_x = gx[..., 0]; dy_x = gy[..., 0]
    dx_y = gx[..., 1]; dy_y = gy[..., 1]
    return dx_x * dy_y - dy_x * dx_y


# ---------------- TRAINING ----------------
def train_one(domain, model_type, epochs=60, batch_size=10, lr=1e-3,
              lambda_barrier=0.01, seed=0, wandb_mode="offline"):
    torch.manual_seed(seed)
    np.random.seed(seed)

    d = torch.load(f"real_data/darcy_{domain}.pt", weights_only=False)
    coeffs = d["coeffs"].float()   # (N,H,W)
    targets = d["targets"].float() # (N,H,W)
    coords = d["coords"]           # (H,W,2)
    mask = d["mask"]               # (H,W) bool

    N, H, W = coeffs.shape
    n_train = int(N * 0.8)
    idx = torch.randperm(N, generator=torch.Generator().manual_seed(seed))
    tr, va = idx[:n_train], idx[n_train:]

    coords_b = coords.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()

    if model_type == "fno":
        model = FNO2d(in_c=3, width=24, m1=12, m2=12).to(DEVICE)
    else:
        model = DIFFNO2d(width=24, m1=12, m2=12).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    barrier_fn = DAgneseBarrierLoss(eps=1e-5)

    cfg = {
        "domain": domain, "model_type": model_type, "epochs": epochs,
        "batch_size": batch_size, "lr": lr, "lambda_barrier": lambda_barrier,
        "n_train": n_train, "n_val": N - n_train, "H": H, "W": W,
        "seed": seed, "device": "cpu",
    }
    run_name = f"{model_type}_{domain}_seed{seed}"
    wandb.init(project="dagnese-dif-fno", name=run_name, config=cfg, mode=wandb_mode,
               reinit=True)

    def forward_batch(idxs):
        a_b = coeffs[idxs].to(DEVICE)          # (B,H,W)
        u_t = targets[idxs].to(DEVICE)         # (B,H,W)
        c_b = coords.unsqueeze(0).expand(len(idxs), -1, -1, -1).contiguous().to(DEVICE)
        if model_type == "fno":
            x = torch.cat([c_b, a_b.unsqueeze(-1)], -1)
            u_p = model(x)
            return u_p, u_t, None
        else:
            u_p, xi = model(c_b, a_b)
            return u_p, u_t, xi

    t0 = time.time()
    best_val = float("inf")
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        ep_loss = 0.0; nb = 0
        for i in range(0, n_train, batch_size):
            b = tr[perm[i:i+batch_size]]
            u_p, u_t, xi = forward_batch(b)
            loss = F.mse_loss(u_p * mask, u_t * mask)
            if xi is not None:
                detJ = jacobian_det(xi)
                lb = barrier_fn(detJ[mask.unsqueeze(0).expand_as(detJ)])
                loss = loss + lambda_barrier * lb
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item(); nb += 1
        ep_loss /= max(nb, 1)

        # validation
        model.eval()
        with torch.no_grad():
            v_l2, v_h1, v_ndet, v_fold = [], [], [], []
            for i in range(0, len(va), batch_size):
                b = va[i:i+batch_size]
                u_p, u_t, xi = forward_batch(b)
                v_l2.append(rel_l2(u_p*mask, u_t*mask, mask))
                v_h1.append(rel_h1(u_p*mask, u_t*mask, mask))
                if xi is not None:
                    detJ = jacobian_det(xi)
                    inside = detJ[mask.unsqueeze(0).expand_as(detJ)]
                    v_ndet.append(inside.min().item())
                    v_fold.append((inside <= 0).float().mean().item() * 100.0)
            mean_l2 = float(np.mean(v_l2))
            mean_h1 = float(np.mean(v_h1))
            mean_ndet = float(np.mean(v_ndet)) if v_ndet else None
            mean_fold = float(np.mean(v_fold)) if v_fold else None

        log = {"epoch": ep, "train_loss": ep_loss, "val_rel_l2": mean_l2,
               "val_rel_h1": mean_h1, "wall_time_s": time.time()-t0}
        if mean_ndet is not None:
            log["val_min_detJ"] = mean_ndet
            log["val_fold_pct"] = mean_fold
        wandb.log(log)

        if ep % 10 == 0 or ep == epochs-1:
            extra = f" minDetJ={mean_ndet:.3e} fold={mean_fold:.2f}%" if mean_ndet is not None else ""
            print(f"[{run_name}] ep {ep:3d}  loss={ep_loss:.5e}  L2={mean_l2:.4e}  H1={mean_h1:.4e}{extra}")

        if mean_l2 < best_val:
            best_val = mean_l2

    # final metrics
    final = {"domain": domain, "model_type": model_type, "seed": seed,
             "val_rel_l2": mean_l2, "val_rel_h1": mean_h1}
    if mean_ndet is not None:
        final["val_min_detJ"] = mean_ndet
        final["val_fold_pct"] = mean_fold
    wandb.summary.update(final)
    wandb.finish()
    return final


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="star", choices=["star","lshape","annulus"])
    ap.add_argument("--model", default="fno", choices=["fno","dif"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lambda_barrier", type=float, default=0.01)
    ap.add_argument("--wandb_mode", default="offline")
    args = ap.parse_args()

    res = train_one(args.domain, args.model, epochs=args.epochs,
                    lambda_barrier=args.lambda_barrier, seed=args.seed,
                    wandb_mode=args.wandb_mode)
    os.makedirs("results", exist_ok=True)
    out = f"results/real_{args.model}_{args.domain}_seed{args.seed}.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print("Saved:", out)
    print(json.dumps(res, indent=2))
