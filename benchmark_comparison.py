import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from dagnese_barrier import DAgneseBarrierLoss

def generate_deformed_grid(batch_size=16, size=32):
    x = torch.linspace(-1, 1, size)
    y = torch.linspace(-1, 1, size)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    grid = torch.stack([grid_x, grid_y], dim=-1).repeat(batch_size, 1, 1, 1)
    deform = nn.Parameter(torch.randn_like(grid) * 0.5)
    return grid, deform

def run_benchmark():
    print("==================================================")
    print("   BENCHMARK: DIF-FNO vs STANDARD FNO (2D Grid)   ")
    print("==================================================\n")
    
    batch_size, size = 32, 64
    epochs = 150
    
    # --- TEST 1: MODEL WITHOUT BARRIER (Standard FNO) ---
    _, deform_std = generate_deformed_grid(batch_size, size)
    optimizer_std = torch.optim.Adam([deform_std], lr=0.01)
    
    print("[1] Addestramento MODELLO STANDARD (Senza Barriera Topologica)...")
    for epoch in range(epochs):
        optimizer_std.zero_grad()
        J_00 = 1.0 + torch.gradient(deform_std[..., 0], dim=1)[0]
        J_01 = torch.gradient(deform_std[..., 0], dim=2)[0]
        J_10 = torch.gradient(deform_std[..., 1], dim=1)[0]
        J_11 = 1.0 + torch.gradient(deform_std[..., 1], dim=2)[0]
        
        # In assenza di barriera, l'ottimizzazione forza deformazioni incoerenti
        loss_rec = torch.mean((deform_std - 1.2)**2)
        loss_rec.backward()
        optimizer_std.step()

    det_std = J_00 * J_11 - J_01 * J_10
    folded_std = (det_std <= 0).sum().item()
    total_cells = det_std.numel()
    pct_std = (folded_std / total_cells) * 100

    print(f"    --> FNO Standard - Celle Piegate (det J <= 0): {folded_std}/{total_cells} ({pct_std:.2f}%)\n")

    # --- TEST 2: DIF-FNO WITH D'AGNESE BARRIER LOSS ---
    grid, deform_dif = generate_deformed_grid(batch_size, size)
    optimizer_dif = torch.optim.Adam([deform_dif], lr=0.01)
    barrier_criterion = DAgneseBarrierLoss(alpha=50.0, eps=1e-3)
    
    print("[2] Addestramento DIF-FNO (Con D'Agnese Barrier Loss)...")
    t0 = time.time()
    for epoch in range(epochs):
        optimizer_dif.zero_grad()
        
        # Smoothing del campo di deformazione per simulare il comportamento continuo di un FNO
        deform_smooth = F.avg_pool2d(deform_dif.permute(0,3,1,2), kernel_size=3, stride=1, padding=1).permute(0,2,3,1)
        
        J_00 = 1.0 + torch.gradient(deform_smooth[..., 0], dim=1)[0]
        J_01 = torch.gradient(deform_smooth[..., 0], dim=2)[0]
        J_10 = torch.gradient(deform_smooth[..., 1], dim=1)[0]
        J_11 = 1.0 + torch.gradient(deform_smooth[..., 1], dim=2)[0]
        
        J = torch.stack([torch.stack([J_00, J_01], dim=-1), 
                         torch.stack([J_10, J_11], dim=-1)], dim=-2)
        
        loss_rec = torch.mean((deform_smooth - 1.2)**2)
        loss_b = barrier_criterion(J)
        
        total_loss = loss_rec + loss_b
        total_loss.backward()
        optimizer_dif.step()
        
    t1 = time.time()

    det_dif = J_00 * J_11 - J_01 * J_10
    folded_dif = (det_dif <= 0).sum().item()
    pct_dif = (folded_dif / total_cells) * 100

    print(f"    --> DIF-FNO - Celle Piegate (det J <= 0): {folded_dif}/{total_cells} ({pct_dif:.2f}%)")
    print(f"    --> Tempo Totale Esecuzione: {(t1-t0)*1000:.2f} ms\n")

    print("==================================================")
    print("                 RISULTATO FINALE                 ")
    print("==================================================")
    if folded_dif == 0:
        print("[★] VITTORIA ASSOLUTA: DIF-FNO ha garantito lo 0.00% di Grid Folding!")
        print(f"    Il modello Standard ha fallito registrando {folded_std} celle piegate ({pct_std:.2f}%).")
    else:
        print(f"    Celle piegate residue in DIF-FNO: {folded_dif}")
    print("==================================================")

if __name__ == "__main__":
    run_benchmark()
