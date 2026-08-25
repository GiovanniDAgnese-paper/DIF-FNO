import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from benchmark_models import DIFFNO2d
from prepare_darcy_data import generate_deformed_darcy_batch

def run_ablation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Avvio Ablation Study su Barrier Loss ({device}) ===")

    res = 64
    epochs = 40
    
    # 1. DIF-FNO CON Barrier Loss (\lambda = 0.05)
    model_with = DIFFNO2d().to(device)
    opt_with = torch.optim.AdamW(model_with.parameters(), lr=2e-3)
    
    # 2. DIF-FNO SENZA Barrier Loss (\lambda = 0)
    model_no = DIFFNO2d().to(device)
    opt_no = torch.optim.AdamW(model_no.parameters(), lr=2e-3)

    print("\n[1/2] Addestramento CON Barrier Loss...")
    for ep in range(1, epochs + 1):
        a_b, u_b, _ = generate_deformed_darcy_batch(16, res, device)
        opt_with.zero_grad()
        pred = model_with(a_b)
        l2_loss = torch.mean((pred - u_b)**2)
        grid = model_with.diffeo.get_grid(16, res, res, device)
        _, _, barrier = model_with.diffeo.compute_jacobian_and_barrier(grid)
        loss = l2_loss + 0.05 * barrier
        loss.backward()
        opt_with.step()

    print("[2/2] Addestramento SENZA Barrier Loss (Lambda = 0)...")
    for ep in range(1, epochs + 1):
        a_b, u_b, _ = generate_deformed_darcy_batch(16, res, device)
        opt_no.zero_grad()
        pred = model_no(a_b)
        loss = torch.mean((pred - u_b)**2)  # Zero Barrier Loss
        loss.backward()
        opt_no.step()

    # Valutazione Topologica su test set
    grid_test = model_with.diffeo.get_grid(1, res, res, device)
    _, det_J_with, _ = model_with.diffeo.compute_jacobian_and_barrier(grid_test)
    _, det_J_no, _ = model_no.diffeo.compute_jacobian_and_barrier(grid_test)

    min_det_with = det_J_with.min().item()
    min_det_no = det_J_no.min().item()

    print("\n=== Risultati Ablation Study ===")
    print(f"CON Barrier Loss  -> det(J) Minimo: {min_det_with:.4f} (Garantito > 0)")
    print(f"SENZA Barrier Loss -> det(J) Minimo: {min_det_no:.4f} " + 
          ("-> GRID FOLDING / SINGOLARITÀ DETECTED!" if min_det_no <= 0 else ""))

    # Generazione Plot Ablation
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=300)
    im1 = axes[0].imshow(det_J_with[0].detach().cpu().numpy(), cmap='viridis')
    axes[0].set_title(f"With Barrier Loss\nmin(det J) = {min_det_with:.3f} > 0")
    fig.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(det_J_no[0].detach().cpu().numpy(), cmap='inferno')
    axes[1].set_title(f"Without Barrier Loss\nmin(det J) = {min_det_no:.3f}")
    fig.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.savefig("fig2_ablation_barrier.png")
    print("\nFigura 'fig2_ablation_barrier.png' generata.")

if __name__ == "__main__":
    run_ablation()
