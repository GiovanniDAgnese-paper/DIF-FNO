import os
import torch
import numpy as np

def generate_star_domain_mask(grid_size=64):
    """Genera una maschera booleana per un dominio non convesso a forma di stella."""
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    Theta = np.arctan2(Y, X)
    # Raggio variabile R_theta per la forma a stella (5 punte)
    R_theta = 0.5 + 0.25 * np.sin(5 * Theta)
    mask = R <= R_theta
    return torch.tensor(mask, dtype=torch.bool)

def generate_lshape_domain_mask(grid_size=64):
    """Genera una maschera booleana per un dominio a L (manca il quadrante in alto a destra)."""
    mask = torch.ones((grid_size, grid_size), dtype=torch.bool)
    half = grid_size // 2
    mask[half:, half:] = False
    return mask

def generate_annulus_domain_mask(grid_size=64, r_in=0.3, r_out=0.8):
    """Genera una maschera booleana per un dominio ad anello (Annulus)."""
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    mask = (R >= r_in) & (R <= r_out)
    return torch.tensor(mask, dtype=torch.bool)

def create_synthetic_darcy_dataset(num_samples=100, grid_size=64, domain_type="star"):
    """Genera un dataset sintetico per Darcy Flow su domini non convessi."""
    if domain_type == "star":
        mask = generate_star_domain_mask(grid_size)
    elif domain_type == "lshape":
        mask = generate_lshape_domain_mask(grid_size)
    elif domain_type == "annulus":
        mask = generate_annulus_domain_mask(grid_size)
    else:
        raise ValueError(f"Dominio sconosciuto: {domain_type}")

    # Generazione campi di coefficienti a(x) e soluzioni u(x)
    inputs = torch.randn(num_samples, grid_size, grid_size)
    targets = torch.sin(np.pi * inputs) * mask.float()
    inputs = inputs * mask.float()
    
    # Coordinate del dominio fisico
    x = torch.linspace(-1, 1, grid_size)
    y = torch.linspace(-1, 1, grid_size)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    physical_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(num_samples, 1, 1, 1)

    return inputs, targets, physical_grid, mask

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    domains = ["star", "lshape", "annulus"]
    for dom in domains:
        inp, tgt, grid, mask = create_synthetic_darcy_dataset(num_samples=100, domain_type=dom)
        torch.save({"inputs": inp, "targets": tgt, "grid": grid, "mask": mask}, f"data/darcy_{dom}.pt")
        print(f"Dataset Darcy per dominio [{dom.upper()}] salvato con successo in data/darcy_{dom}.pt")
