import torch
import numpy as np

class ParametricDomainGenerator:
    r"""
    Generatore stocastico di domini 2D lisci per deformazione polare:
    r(\theta) = r_0 + \sum_{k=1}^K (a_k \cos(k\theta) + b_k \sin(k\theta))
    Garantisce differenti topologie e geometrie mai viste per la valutazione zero-shot.
    """
    def __init__(self, res=128, max_harmonics=4):
        self.res = res
        self.max_harmonics = max_harmonics

    def sample_domain(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        r0 = 0.5 + 0.1 * torch.rand(1).item()
        harmonics = self.max_harmonics
        coeffs_a = (torch.rand(harmonics) - 0.5) * 0.15 / torch.arange(1, harmonics + 1)
        coeffs_b = (torch.rand(harmonics) - 0.5) * 0.15 / torch.arange(1, harmonics + 1)

        xi = torch.linspace(-1, 1, self.res)
        eta = torch.linspace(-1, 1, self.res)
        GRID_XI, GRID_ETA = torch.meshgrid(xi, eta, indexing='ij')

        R_grid = torch.sqrt(GRID_XI**2 + GRID_ETA**2) / np.sqrt(2)
        THETA_grid = torch.atan2(GRID_ETA, GRID_XI)

        r_mesh = torch.full_like(THETA_grid, r0)
        for k in range(1, harmonics + 1):
            r_mesh += coeffs_a[k-1] * torch.cos(k * THETA_grid) + coeffs_b[k-1] * torch.sin(k * THETA_grid)

        X_phys = R_grid * r_mesh * torch.cos(THETA_grid)
        Y_phys = R_grid * r_mesh * torch.sin(THETA_grid)
        
        mask = (R_grid <= 1.0).float()

        return {
            'X_phys': X_phys,
            'Y_phys': Y_phys,
            'mask': mask,
            'grid_lat': torch.stack([GRID_XI, GRID_ETA], dim=-1)
        }
