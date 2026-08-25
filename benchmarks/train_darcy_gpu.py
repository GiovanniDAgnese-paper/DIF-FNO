import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from benchmark_models import DIFFNO2d, GeoFNO2d, FNOMask2d, ensure_channel_last
from prepare_darcy_data import generate_deformed_darcy_batch

def compute_h1_physical(pred: torch.Tensor, target: torch.Tensor) -> tuple:
    pred_c = ensure_channel_last(pred)
    target_c = ensure_channel_last(target)
    N = pred_c.shape[1]
    h = 1.0 / N

    grad_p_y = (pred_c[:, 1:, :, :] - pred_c[:, :-1, :, :]) / h
    grad_p_x = (pred_c[:, :, 1:, :] - pred_c[:, :, :-1, :]) / h
    grad_t_y = (target_c[:, 1:, :, :] - target_c[:, :-1, :, :]) / h
    grad_t_x = (target_c[:, :, 1:, :] - target_c[:, :, :-1, :]) / h

    l2_err = torch.mean(torch.norm(pred_c.reshape(pred_c.shape[0], -1) - target_c.reshape(target_c.shape[0], -1), p=2, dim=1) / 
                        (torch.norm(target_c.reshape(target_c.shape[0], -1), p=2, dim=1) + 1e-8))
    
    h1_y = torch.mean(torch.norm(grad_p_y.reshape(grad_p_y.shape[0], -1) - grad_t_y.reshape(grad_t_y.shape[0], -1), p=2, dim=1) / 
                      (torch.norm(grad_t_y.reshape(grad_t_y.shape[0], -1), p=2, dim=1) + 1e-8))
    
    h1_x = torch.mean(torch.norm(grad_p_x.reshape(grad_p_x.shape[0], -1) - grad_t_x.reshape(grad_t_x.shape[0], -1), p=2, dim=1) / 
                      (torch.norm(grad_t_x.reshape(grad_t_x.shape[0], -1), p=2, dim=1) + 1e-8))
    
    return l2_err, (l2_err + h1_y + h1_x) / 3.0

def train_darcy_model(model: nn.Module, epochs: int = 100, resolution: int = 64, device: torch.device = None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    model.train()
    
    for epoch in range(1, epochs + 1):
        a_batch, u_batch, phys_grid = generate_deformed_darcy_batch(batch_size=16, res=resolution, device=device)
        optimizer.zero_grad()
        
        if isinstance(model, GeoFNO2d):
            pred = model(a_batch, phys_grid=phys_grid)
        else:
            pred = model(a_batch)

        l2_loss, h1_loss = compute_h1_physical(pred, u_batch)
        loss = l2_loss + 0.1 * h1_loss
        
        # Barrier Loss topologica per DIF-FNO
        if hasattr(model, 'diffeo'):
            latent_grid = model.diffeo.get_grid(a_batch.shape[0], resolution, resolution, device)
            _, _, barrier_loss = model.diffeo.compute_jacobian_and_barrier(latent_grid)
            loss += 0.05 * barrier_loss
            
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 20 == 0 or epoch == epochs:
            print(f"   Epoca [{epoch:3d}/{epochs}] | Loss: {loss.item():.4f} | L2: {l2_loss.item():.4f}", flush=True)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Addestramento Avanzato Darcy Deformed su {device} ===")
    
    seeds = [42, 123, 456]
    resolutions = [64, 128]
    epochs_count = 100 if torch.cuda.is_available() else 40
    
    models_cls = {
        "DIF-FNO (Ours)": DIFFNO2d,
        "Geo-FNO": GeoFNO2d,
        "Masked-FNO": FNOMask2d
    }

    results = {name: {res: {"l2": [], "h1": []} for res in resolutions} for name in models_cls.keys()}

    for name, m_cls in models_cls.items():
        print(f"\n--- Avvio modelli per: {name} ---")
        for res in resolutions:
            for seed in seeds:
                print(f"Res: {res}x{res} | Seed: {seed} | Epoche: {epochs_count}")
                torch.manual_seed(seed)
                np.random.seed(seed)
                
                model = m_cls().to(device)
                train_darcy_model(model, epochs=epochs_count, resolution=res, device=device)
                
                model.eval()
                with torch.no_grad():
                    a_test, u_test, grid_test = generate_deformed_darcy_batch(batch_size=32, res=res, device=device)
                    if isinstance(model, GeoFNO2d):
                        pred = model(a_test, phys_grid=grid_test)
                    else:
                        pred = model(a_test)
                    l2_err, h1_err = compute_h1_physical(pred, u_test)
                    
                results[name][res]["l2"].append(l2_err.item())
                results[name][res]["h1"].append(h1_err.item())

    # Formattazione Tabella LaTeX
    latex_str = [
        "\\begin{table}[h]", "\\centering",
        "\\caption{Deformed Darcy Flow Benchmark: Convergence Results}",
        "\\label{tab:darcy_results}",
        "\\begin{tabular}{lcccc}", "\\toprule",
        " & \\multicolumn{2}{c}{$64 \\times 64$} & \\multicolumn{2}{c}{$128 \\times 128$} \\\\",
        "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}",
        "Model & $L^2$ Error & $H^1$ Error & $L^2$ Error & $H^1$ Error \\\\",
        "\\midrule"
    ]

    for name in models_cls.keys():
        row = f"{name}"
        for res in resolutions:
            l2_m, l2_s = np.mean(results[name][res]["l2"]), np.std(results[name][res]["l2"])
            h1_m, h1_s = np.mean(results[name][res]["h1"]), np.std(results[name][res]["h1"])
            row += f" & {l2_m:.4f} \\pm {l2_s:.4f} & {h1_m:.4f} \\pm {h1_s:.4f}"
        row += " \\\\"
        latex_str.append(row)

    latex_str.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    
    table_out = "\n".join(latex_str)
    with open("table_results.tex", "w") as f:
        f.write(table_out)
    print("\nBenchmark completato con successo. File 'table_results.tex' aggiornato.\n")
    print(table_out)

if __name__ == "__main__":
    main()
