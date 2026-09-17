"""Generate REAL Darcy flow datasets on non-convex domains via finite differences."""
import os, sys, time
import numpy as np
import torch
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

def make_star_mask(n):
    x = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, x, indexing='ij')
    R = np.sqrt(X**2 + Y**2)
    Theta = np.arctan2(Y, X)
    R_theta = 0.6 + 0.15 * np.cos(5 * Theta)
    return R <= R_theta

def make_lshape_mask(n):
    mask = np.ones((n, n), dtype=bool)
    m = n // 2
    mask[m:, m:] = False
    # thin margin to avoid degenerate boundary
    mask[:, 0] = mask[0, :] = mask[:, -1] = mask[-1, :] = False
    return mask

def make_annulus_mask(n, r_in=0.35, r_out=0.85):
    x = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, x, indexing='ij')
    R = np.sqrt(X**2 + Y**2)
    m = (R >= r_in) & (R <= r_out)
    m[:, 0] = m[0, :] = m[:, -1] = m[-1, :] = False
    return m

def sample_coeff_field(n, mask, rng, n_modes=6):
    """Smooth positive GRF via spectral synthesis."""
    x = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, x, indexing='ij')
    a = np.zeros((n, n))
    for k1 in range(-n_modes, n_modes + 1):
        for k2 in range(-n_modes, n_modes + 1):
            amp = rng.standard_normal() / (1 + k1 * k1 + k2 * k2)
            phase = rng.uniform(0, 2 * np.pi)
            a += amp * np.cos(np.pi * (k1 * X + k2 * Y) + phase)
    a = a / (np.abs(a).max() + 1e-8)
    a = np.exp(2.0 * a)
    return a * mask + (1 - mask) * 1.0

def solve_darcy(mask, a, f, h):
    """Solve -div(a grad u) = f, homogeneous Dirichlet on mask boundary."""
    n = mask.shape[0]
    interior = np.zeros_like(mask)
    interior[1:-1, 1:-1] = (mask[1:-1, 1:-1] & mask[:-2, 1:-1] & mask[2:, 1:-1]
                             & mask[1:-1, :-2] & mask[1:-1, 2:])
    idx = -np.ones_like(mask, dtype=np.int64)
    n_int = int(interior.sum())
    idx[interior] = np.arange(n_int)
    A = lil_matrix((n_int, n_int))
    b = np.zeros(n_int)
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            if not interior[i, j]:
                continue
            k = idx[i, j]
            a_e = 2 * a[i, j] * a[i, j + 1] / (a[i, j] + a[i, j + 1] + 1e-12)
            a_w = 2 * a[i, j] * a[i, j - 1] / (a[i, j] + a[i, j - 1] + 1e-12)
            a_n = 2 * a[i, j] * a[i + 1, j] / (a[i, j] + a[i + 1, j] + 1e-12)
            a_s = 2 * a[i, j] * a[i - 1, j] / (a[i, j] + a[i - 1, j] + 1e-12)
            A[k, k] = (a_e + a_w + a_n + a_s) / h**2
            if interior[i, j + 1]: A[k, idx[i, j + 1]] = -a_e / h**2
            if interior[i, j - 1]: A[k, idx[i, j - 1]] = -a_w / h**2
            if interior[i + 1, j]: A[k, idx[i + 1, j]] = -a_n / h**2
            if interior[i - 1, j]: A[k, idx[i - 1, j]] = -a_s / h**2
            b[k] = f[i, j]
    A = A.tocsr()
    u_int = spsolve(A, b)
    u = np.zeros_like(a)
    u[interior] = u_int
    return u

def generate_dataset(name, mask_fn, n=64, n_samples=100, seed=42):
    rng = np.random.default_rng(seed)
    mask = mask_fn(n)
    h = 2.0 / (n - 1)
    x = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, x, indexing='ij')
    coords = np.stack([X, Y], axis=-1)
    coeffs = np.zeros((n_samples, n, n), dtype=np.float32)
    targets = np.zeros((n_samples, n, n), dtype=np.float32)
    t0 = time.time()
    for k in range(n_samples):
        a = sample_coeff_field(n, mask, rng)
        f = mask.astype(np.float64)
        u = solve_darcy(mask, a, f, h)
        coeffs[k] = (a * mask).astype(np.float32)
        targets[k] = (u * mask).astype(np.float32)
        if (k + 1) % 20 == 0:
            print(f"  [{name}] {k+1}/{n_samples} ({time.time()-t0:.1f}s)")
    return {
        'coeffs': torch.tensor(coeffs),
        'targets': torch.tensor(targets),
        'coords': torch.tensor(coords, dtype=torch.float32),
        'mask': torch.tensor(mask, dtype=torch.bool),
    }

if __name__ == "__main__":
    os.makedirs("real_data", exist_ok=True)
    configs = [
        ("star",    make_star_mask,    64, 100, 42),
        ("lshape",  make_lshape_mask,  64, 100, 43),
        ("annulus", make_annulus_mask, 64, 100, 44),
    ]
    for name, fn, n, ns, seed in configs:
        print(f"\n=== Generating {name.upper()} n={n} samples={ns} ===")
        data = generate_dataset(name, fn, n=n, n_samples=ns, seed=seed)
        out = f"real_data/darcy_{name}.pt"
        torch.save(data, out)
        print(f"  Saved -> {out}  (coeffs {tuple(data['coeffs'].shape)})")
    print("\nDONE.")
