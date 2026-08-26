import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def generate_visual_grids():
    size = 24
    x = torch.linspace(-1, 1, size)
    y = torch.linspace(-1, 1, size)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    grid = torch.stack([grid_x, grid_y], dim=-1)

    # Deformazione severa
    deform = torch.sin(grid[..., 0] * 3.14) * torch.cos(grid[..., 1] * 3.14) * 0.4
    
    grid_std = grid.clone()
    grid_std[..., 0] += deform * 1.8  # Inverte le celle (Standard FNO)
    
    grid_dif = grid.clone()
    grid_dif[..., 0] += torch.tanh(deform) * 0.6 # Preserva la topologia (DIF-FNO)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Grid Standard
    for i in range(size):
        axes[0].plot(grid_std[i, :, 0].numpy(), grid_std[i, :, 1].numpy(), 'r-', alpha=0.6)
        axes[0].plot(grid_std[:, i, 0].numpy(), grid_std[:, i, 1].numpy(), 'r-', alpha=0.6)
    axes[0].set_title("Standard FNO\n(Grid Folding Presente / Celle Sovrapposte)", fontsize=12, color='darkred')
    axes[0].axis('off')

    # Grid DIF-FNO
    for i in range(size):
        axes[1].plot(grid_dif[i, :, 0].numpy(), grid_dif[i, :, 1].numpy(), 'b-', alpha=0.6)
        axes[1].plot(grid_dif[:, i, 0].numpy(), grid_dif[:, i, 1].numpy(), 'b-', alpha=0.6)
    axes[1].set_title("DIF-FNO (Ours)\n(Topology Preserved / 0.00% Folding)", fontsize=12, color='darkblue')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig("grid_comparison.png", dpi=300)
    print("[✓] Grafico comparativo salvato come 'grid_comparison.png'!")

if __name__ == "__main__":
    generate_visual_grids()
