import torch
import matplotlib.pyplot as plt
import numpy as np
from benchmark_models import DIFFNO2d

def generate_diffeo_figures():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Estrazione visualizzazioni geometriche su {device}...")

    # Inizializza il modello DIF-FNO
    model = DIFFNO2d().to(device)
    model.eval()

    resolution = 32  # Risoluzione per la visualizzazione pulita della griglia
    batch_size = 1

    # Genera la griglia latente (B, H, W, 2)
    latent_grid = model.diffeo.get_grid(batch_size, resolution, resolution, device)
    
    # Calcola la mappa diffeomorfa e lo Jacobiano
    mapped_grid, det_J, barrier_loss = model.diffeo.compute_jacobian_and_barrier(latent_grid)

    # Conversione in NumPy per Matplotlib
    grid_lat = latent_grid[0].detach().cpu().numpy()
    grid_map = mapped_grid[0].detach().cpu().numpy()
    det_J_np = det_J[0].detach().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)

    # 1. Griglia Latente Canonica (Omega_l)
    for i in range(resolution):
        axes[0].plot(grid_lat[i, :, 0], grid_lat[i, :, 1], 'k-', alpha=0.4, linewidth=0.8)
        axes[0].plot(grid_lat[:, i, 0], grid_lat[:, i, 1], 'k-', alpha=0.4, linewidth=0.8)
    axes[0].set_title(r"Dominio Latente $\Omega_l$ (Canonico)", fontsize=12)
    axes[0].set_aspect('equal')
    axes[0].axis('off')

    # 2. Griglia Fisica Deformata phi(Omega_l)
    for i in range(resolution):
        axes[1].plot(grid_map[i, :, 0], grid_map[i, :, 1], 'b-', alpha=0.6, linewidth=0.8)
        axes[1].plot(grid_map[:, i, 0], grid_map[:, i, 1], 'b-', alpha=0.6, linewidth=0.8)
    axes[1].set_title(r"Dominio Fisico Mappato $\phi(\Omega_l)$", fontsize=12)
    axes[1].set_aspect('equal')
    axes[1].axis('off')

    # 3. Heatmap Determinante Jacobiano det(J)
    im = axes[2].imshow(det_J_np, cmap='viridis', origin='lower')
    cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label(r"$\det(J)$", fontsize=11)
    axes[2].set_title(r"Garanzia Topologica: $\det(J) > 0$", fontsize=12)
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig("fig1_diffeo_mapping.png", bbox_inches='tight')
    plt.savefig("fig1_diffeo_mapping.pdf", bbox_inches='tight')
    print("Figura 'fig1_diffeo_mapping.png' e 'fig1_diffeo_mapping.pdf' salvate con successo.")

if __name__ == "__main__":
    generate_diffeo_figures()
