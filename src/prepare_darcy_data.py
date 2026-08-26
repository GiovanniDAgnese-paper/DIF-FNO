import torch
import numpy as np

def generate_deformed_darcy_batch(batch_size: int, res: int, device: torch.device):
    """
    Simula un dataset di Darcy Flow su domini deformati non-lineari.
    - Permeabilità a(x,y): Campo stocastico GRF.
    - Geometria: Dominio deformato tramite trasformazione sinusoidale ai bordi.
    """
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(0, 1, res, device=device),
        torch.linspace(0, 1, res, device=device),
        indexing="ij"
    )
    
    # Deformazione non-lineare del dominio fisico (onda sui bordi)
    x_phys = grid_x + 0.1 * torch.sin(2 * np.pi * grid_y)
    y_phys = grid_y + 0.1 * torch.cos(2 * np.pi * grid_x)
    phys_grid = torch.stack([x_phys, y_phys], dim=-1).unsqueeze(0).repeat(batch_size, 1, 1, 1)

    # Campo di permeabilità a(x,y)
    freq_a = torch.randint(1, 4, (batch_size, 1, 1, 1), device=device).float()
    coeff_a = torch.exp(torch.sin(freq_a * np.pi * x_phys) * torch.cos(freq_a * np.pi * y_phys))
    
    # Soluzione u(x,y) approssimata per la PDE di Darcy: -div(a grad u) = f
    sol_u = torch.sin(np.pi * x_phys) * torch.sin(np.pi * y_phys) / (coeff_a + 0.5)

    return coeff_a, sol_u, phys_grid

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a, u, grid = generate_deformed_darcy_batch(4, 64, device)
    print(f"Dataset Darcy Deformed generato con successo:")
    print(f" - Permeabilità a(x,y): {a.shape}")
    print(f" - Soluzione u(x,y):    {u.shape}")
    print(f" - Griglia Fisica:      {grid.shape}")
