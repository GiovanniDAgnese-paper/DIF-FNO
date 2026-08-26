import torch
import numpy as np

def generate_star_domain(grid_size=64, num_samples=100):
    """Genera domini a stella (Star-shaped non-convex domain)."""
    theta = np.linspace(0, 2*np.pi, grid_size)
    r_boundary = 0.5 + 0.2 * np.sin(5 * theta)
    
    grids = []
    for _ in range(num_samples):
        r = np.linspace(0.05, 1.0, grid_size)
        R, TH = np.meshgrid(r, theta)
        X = R * np.cos(TH) * r_boundary
        Y = R * np.sin(TH) * r_boundary
        grid = np.stack([X, Y], axis=-1)
        grids.append(grid)
        
    return torch.tensor(np.array(grids), dtype=torch.float32)

def generate_annulus_domain(grid_size=64, num_samples=100, r_in=0.3, r_out=1.0):
    """Genera domini ad anello (Annulus domain)."""
    theta = np.linspace(0, 2*np.pi, grid_size)
    r = np.linspace(r_in, r_out, grid_size)
    R, TH = np.meshgrid(r, theta)
    X = R * np.cos(TH)
    Y = R * np.sin(TH)
    grid = np.stack([X, Y], axis=-1)
    return torch.tensor(np.tile(grid, (num_samples, 1, 1, 1)), dtype=torch.float32)
