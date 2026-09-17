"""DIF-FNO v2: fixed barrier (softplus) + zero-init diffeo for guaranteed det(J)>0."""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath("src"))

import wandb
DEVICE = torch.device("cpu")


# ---------- Models ----------
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
        out[:, :, :self.m1, :self.m2] = torch.einsum("bixy,ioxy->boxy", xf[:, :, :self.m1, :self.m2], self.w1)
        out[:, :, -self.m1:, :self.m2] = torch.einsum("bixy,ioxy->boxy", xf[:, :, -self.m1:, :self.m2], self.w2)
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
    def forward(self, x):
        x = self.fc0(x).permute(0, 3, 1, 2)
        x = F.gelu(self.c0(x) + self.w0(x))
        x = F.gelu(self.c1(x) + self.w1(x))
        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        return self.fc2(x).squeeze(-1)


class DiffeomorphicMap(nn.Module):
    """Input-conditioned zero-init residual: phi(x, a) = x + scale * net(x, a).
    At t=0, net=0 so phi = identity and det(J)=1. As training proceeds, the
    map adapts to each coefficient field a(x), providing per-sample alignment."""
    def __init__(self, hidden=32, scale=0.1):
        super().__init__()
        self.scale = scale
        # in: (coord_x, coord_y, a) = 3
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    def forward(self, coords, a):
        # coords: (B,H,W,2), a: (B,H,W)
        inp = torch.cat([coords, a.unsqueeze(-1)], dim=-1)  # (B,H,W,3)
        return coords + self.scale * self.net(inp)


class DIFFNO2d(nn.Module):
    def __init__(self, width=24, m1=12, m2=12):
        super().__init__()
        self.diffeo = DiffeomorphicMap()
        self.fno = FNO2d(in_c=3, width=width, m1=m1, m2=m2)
    def forward(self, coords, a):
        xi = self.diffeo(coords, a)
        x = torch.cat([xi, a.unsqueeze(-1)], -1)
        u = self.fno(x)
        return u, xi


# ---------- Metrics ----------
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
    gy = torch.zeros_like(xi); gx = torch.zeros_like(xi)
    gy[:, :-1, :, :] = xi[:, 1:, :, :] - xi[:, :-1, :, :]
    gx[:, :, :-1, :] = xi[:, :, 1:, :] - xi[:, :, :-1, :]
    # Jacobian of phi: J[a,b] = d(phi_a)/d(coord_b)
    # gy = diff along axis 0, gx = diff along axis 1
    # J[0,0] = d(phi_x)/d(coord_0) = gy[...,0]
    # J[0,1] = d(phi_x)/d(coord_1) = gx[...,0]
    # J[1,0] = d(phi_y)/d(coord_0) = gy[...,1]
    # J[1,1] = d(phi_y)/d(coord_1) = gx[...,1]
    # det(J) = J[0,0]*J[1,1] - J[0,1]*J[1,0]
    J00 = gy[..., 0]; J01 = gx[..., 0]
    J10 = gy[..., 1]; J11 = gx[..., 1]
    return J00 * J11 - J01 * J10


# ---------- Barrier v2 (SOFT, no saturation) ----------
def barrier_loss_v2(det_J, tau=0.005):
    """Smooth softplus barrier: ~0 when det>0, linear growth when det<0.
    Always differentiable, no clamp saturation."""
    return F.softplus(-det_J / tau).mean() * tau


# ---------- Training ----------
def train_one(domain, model_type, epochs=60, batch_size=10, lr=1e-3,
              lambda_barrier=1.0, seed=0, wandb_mode="offline", tag="v2"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    d = torch.load(f"real_data/darcy_{domain}.pt", weights_only=False)
    coeffs = d["coeffs"].float()
    targets = d["targets"].float()
    coords = d["coords"]
    mask = d["mask"]
    N, H, W = coeffs.shape
    n_train = int(N * 0.8)
    idx = torch.randperm(N, generator=torch.Generator().manual_seed(seed))
    tr, va = idx[:n_train], idx[n_train:]

    if model_type == "fno":
        model = FNO2d(in_c=3, width=24, m1=12, m2=12).to(DEVICE)
    else:
        model = DIFFNO2d(width=24, m1=12, m2=12).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    cfg = {"domain": domain, "model_type": model_type, "epochs": epochs,
           "batch_size": batch_size, "lr": lr, "lambda_barrier": lambda_barrier,
           "n_train": n_train, "n_val": N - n_train, "H": H, "W": W,
           "seed": seed, "device": "cpu", "barrier": "softplus_v2",
           "diffeo_init": "zero"}
    run_name = f"{tag}_{model_type}_{domain}_seed{seed}"
    wandb.init(project="dagnese-dif-fno", name=run_name, config=cfg,
               mode=wandb_mode, reinit=True)

    def forward_batch(idxs):
        a_b = coeffs[idxs].to(DEVICE)
        u_t = targets[idxs].to(DEVICE)
        c_b = coords.unsqueeze(0).expand(len(idxs), -1, -1, -1).contiguous().to(DEVICE)
        if model_type == "fno":
            x = torch.cat([c_b, a_b.unsqueeze(-1)], -1)
            return model(x), u_t, None
        else:
            u_p, xi = model(c_b, a_b)
            return u_p, u_t, xi

    t0 = time.time()
    m_l2 = m_h1 = m_ndet = m_fold = 0.0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        ep_loss = ep_mse = ep_bar = 0.0; nb = 0
        for i in range(0, n_train, batch_size):
            b = tr[perm[i:i+batch_size]]
            u_p, u_t, xi = forward_batch(b)
            mse = F.mse_loss(u_p * mask, u_t * mask)
            loss = mse
            if xi is not None:
                detJ = jacobian_det(xi)
                lb = barrier_loss_v2(detJ, tau=0.005)
                loss = loss + lambda_barrier * lb
            opt.zero_grad(); loss.backward(); opt.step()
            ep_mse += mse.item(); ep_bar += (lb.item() if xi is not None else 0.0)
            ep_loss += loss.item(); nb += 1
        ep_mse /= max(nb, 1); ep_bar /= max(nb, 1); ep_loss /= max(nb, 1)

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
            m_l2 = float(np.mean(v_l2)); m_h1 = float(np.mean(v_h1))
            m_ndet = float(np.mean(v_ndet)) if v_ndet else None
            m_fold = float(np.mean(v_fold)) if v_fold else None

        log = {"epoch": ep, "train_loss": ep_loss, "train_mse": ep_mse,
               "train_barrier": ep_bar, "val_rel_l2": m_l2, "val_rel_h1": m_h1,
               "wall_time_s": time.time()-t0}
        if m_ndet is not None:
            log["val_min_detJ"] = m_ndet; log["val_fold_pct"] = m_fold
        wandb.log(log)

        if ep % 10 == 0 or ep == epochs-1:
            extra = f" bar={ep_bar:.3e} minDetJ={m_ndet:.3e} fold={m_fold:.2f}%" if m_ndet is not None else ""
            print(f"[{run_name}] ep {ep:3d}  mse={ep_mse:.3e}{extra}  L2={m_l2:.4e}  H1={m_h1:.4e}")

    final = {"domain": domain, "model_type": model_type, "seed": seed,
             "val_rel_l2": m_l2, "val_rel_h1": m_h1}
    if m_ndet is not None:
        final["val_min_detJ"] = m_ndet; final["val_fold_pct"] = m_fold
    wandb.summary.update(final); wandb.finish()
    return final


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="star", choices=["star","lshape","annulus"])
    ap.add_argument("--model", default="dif", choices=["fno","dif"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lambda_barrier", type=float, default=1.0)
    ap.add_argument("--wandb_mode", default="offline")
    ap.add_argument("--tag", default="v4")
    args = ap.parse_args()
    res = train_one(args.domain, args.model, epochs=args.epochs,
                    lambda_barrier=args.lambda_barrier, seed=args.seed,
                    wandb_mode=args.wandb_mode, tag=args.tag)
    os.makedirs("results", exist_ok=True)
    out = f"results/real_{args.model}_{args.domain}_seed{args.seed}_{args.tag}.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print("Saved:", out); print(json.dumps(res, indent=2))
