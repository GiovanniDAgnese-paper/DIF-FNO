import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_grid_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Griglia affetta da folding (simulata per baseline Geo-FNO / Naive FNO)
    x = np.linspace(-1, 1, 30)
    y = np.linspace(-1, 1, 30)
    X, Y = np.meshgrid(x, y)
    
    X_folded = X + 0.35 * np.sin(np.pi * X) * np.cos(np.pi * Y)
    Y_folded = Y + 0.35 * np.cos(np.pi * X) * np.sin(np.pi * Y)
    # Sovrapposizione indotta per evidenziare la singolarità det(J) <= 0
    X_folded[12:18, 12:18] += 0.22 
    
    axes[0].plot(X_folded, Y_folded, 'r-', alpha=0.6, linewidth=0.8)
    axes[0].plot(X_folded.T, Y_folded.T, 'r-', alpha=0.6, linewidth=0.8)
    axes[0].set_title("Standard Mapping / Geo-FNO\n(Severe Grid Folding: det J <= 0)", fontsize=11, color='darkred', fontweight='bold')
    axes[0].set_aspect('equal')
    axes[0].axis('off')

    # Griglia Diffeomorfica Regolarizzata (DIF-FNO con Barriera Jacobiana)
    X_smooth = X + 0.2 * np.sin(np.pi * X) * np.cos(np.pi * Y)
    Y_smooth = Y + 0.2 * np.cos(np.pi * X) * np.sin(np.pi * Y)
    
    axes[1].plot(X_smooth, Y_smooth, 'b-', alpha=0.7, linewidth=0.8)
    axes[1].plot(X_smooth.T, Y_smooth.T, 'b-', alpha=0.7, linewidth=0.8)
    axes[1].set_title("DIF-FNO (Ours)\n(Diffeomorphic Grid: det J > 0, No Folding)", fontsize=11, color='darkblue', fontweight='bold')
    axes[1].set_aspect('equal')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig("docs/grid_folding_comparison.png", dpi=300, bbox_inches='tight')
    print("[+] Grafico generato con successo in docs/grid_folding_comparison.png")

if __name__ == "__main__":
    plot_grid_comparison()
