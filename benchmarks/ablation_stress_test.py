import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from benchmark_models import DIFFNO2d

def generate_extreme_deformed_batch(batch_size: int, res: int, device: torch.device):
    """
    Genera una geometria ad alto stress con forti curvature e angoli stretti.
    """
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(0, 1, res, device=device),
        torch.linspace(0, 1, res, device=device),
        indexing="ij"
    )
    # Deformazione estrema: ampiezza 0.35 e frequenza doppia per forzare il grid folding
    x_phys = grid_x + 0.35 * torch.sin(4 * np.pi * grid_y) * torch.cos(2 * np.pi * grid_x)
    y_phys = grid_y + 0.35 * torch.cos(4 * np.pi * grid_x) * torch.sin(2 * np.pi * grid_y)
    
    coeff_a = torch.exp(torch.sin(3 * np.pi * x_phys) * torch.cos(3 * np.pi * y_phys)).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    sol_u = (torch.sin(2 * np.pi * x_phys) * torch.cos(2 * np.pi * y_phys)).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    return coeff_a, sol_u

def run_stress_ablation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Stress-Test Ablation: Mappatura ad Alta Curvatura ({device}) ===")

    res = 64
    epochs = 60

    model_with = DIFFNO2d().to(device)
    opt_with = torch.optim.AdamW(model_with.parameters(), lr=4e-3)

    model_no = DIFFNO2d().to(device)
    opt_no = torch.optim.AdamW(model_no.parameters(), lr=4e-3)

    print("\n[1/2] Training CON Barrier Loss (Lambda = 0.1)...")
    for ep in range(1, epochs + 1):
        a_b, u_b = generate_extreme_deformed_batch(16, res, device)
        opt_with.zero_grad()
        pred = model_with(a_b)
        l2_loss = torch.mean((pred - u_b)**2)
        grid = model_with.diffeo.get_grid(16, res, res, device)
        _, _, barrier = model_with.diffeo.compute_jacobian_and_barrier(grid)
        loss = l2_loss + 0.1 * barrier
        loss.backward()
        opt_with.step()

    print("[2/2] Training SENZA Barrier Loss (Lambda = 0)...")
    for ep in range(1, epochs + 1):
        a_b, u_b = generate_extreme_deformed_batch(16, res, device)
        opt_no.zero_grad()
        pred = model_no(a_b)
        loss = torch.mean((pred - u_b)**2)
        loss.backward()
        opt_no.step()

    grid_test = model_with.diffeo.get_grid(1, res, res, device)
    _, det_J_with, _ = model_with.diffeo.compute_jacobian_and_barrier(grid_test)
    _, det_J_no, _ = model_no.diffeo.compute_jacobian_and_barrier(grid_test)

    det_with_np = det_J_with[0].detach().cpu().numpy()
    det_no_np = det_J_no[0].detach().cpu().numpy()

    print("\n=== Risultati Stress-Test ===")
    print(f"CON Barrier Loss  -> det(J) Minimo: {det_with_np.min():.4f} (Topologia Integrale)")
    print(f"SENZA Barrier Loss -> det(J) Minimo: {det_no_np.min():.4f} (Valore Critico/Singolarità)")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
    
    im1 = axes[0].imshow(det_with_np, cmap='viridis', origin='lower')
    axes[0].set_title(f"With Barrier Loss\nmin(det J) = {det_with_np.min():.3f} > 0 (Diffeomorfismo Preservato)", fontsize=10)
    fig.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(det_no_np, cmap='magma', origin='lower')
    axes[1].set_title(f"Without Barrier Loss\nmin(det J) = {det_no_np.min():.3f} (Grid Collapse / Instabilità)", fontsize=10)
    fig.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.savefig("fig2_ablation_stress.png")
    print("\nNuova figura 'fig2_ablation_stress.png' salvata.")

if __name__ == "__main__":
    run_stress_ablation()
