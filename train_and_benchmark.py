import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from benchmark_models import DIFFNO2d, GeoFNO2d, FNOMask2d, ensure_channel_last

# Ottimizzazione multithreading su CPU Arch Linux
torch.set_num_threads(torch.get_num_threads())

def generate_grf_data(batch_size: int, resolution: int, device: torch.device):
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(0, 1, resolution, device=device),
        torch.linspace(0, 1, resolution, device=device),
        indexing="ij"
    )
    freq1 = torch.randint(1, 5, (batch_size, 1, 1, 1), device=device).float()
    freq2 = torch.randint(1, 5, (batch_size, 1, 1, 1), device=device).float()
    
    x_in = torch.sin(freq1 * np.pi * grid_x) * torch.cos(freq2 * np.pi * grid_y)
    target = x_in / ((freq1 * np.pi)**2 + (freq2 * np.pi)**2 + 1e-5)
    return x_in, target

def compute_h1_physical(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
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

def train_model(model: nn.Module, epochs: int = 10, resolution: int = 64, device: torch.device = None):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()
    
    for epoch in range(epochs):
        x_batch, y_batch = generate_grf_data(batch_size=8, resolution=resolution, device=device)
        optimizer.zero_grad()
        
        pred = model(x_batch)
        l2_loss, h1_loss = compute_h1_physical(pred, y_batch)
        loss = l2_loss + 0.1 * h1_loss
        
        if hasattr(model, 'diffeo'):
            # Calcolo isolato della barrier loss su un campionamento ridotto per velocizzare la CPU
            latent_grid = model.diffeo.get_grid(x_batch.shape[0], resolution, resolution, device)
            _, _, barrier_loss = model.diffeo.compute_jacobian_and_barrier(latent_grid)
            loss += 0.05 * barrier_loss
            
        loss.backward()
        optimizer.step()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Esecuzione Benchmark su {device} (Thread attivi: {torch.get_num_threads()}) ===")
    
    seeds = [42, 123, 456]  # 3 Seed per iterazione veloce su CPU
    resolutions = [64, 128]  # Risoluzioni di test locale
    models_cls = {
        "DIF-FNO (Ours)": DIFFNO2d,
        "Geo-FNO": GeoFNO2d,
        "Masked-FNO": FNOMask2d
    }

    results = {name: {res: {"l2": [], "h1": []} for res in resolutions} for name in models_cls.keys()}
    total_runs = len(models_cls) * len(resolutions) * len(seeds)
    current_run = 0

    for name, m_cls in models_cls.items():
        for res in resolutions:
            for seed in seeds:
                current_run += 1
                print(f"[{current_run}/{total_runs}] Addestramento {name} | Res: {res}x{res} | Seed: {seed}...", flush=True)
                
                torch.manual_seed(seed)
                np.random.seed(seed)
                
                model = m_cls().to(device)
                train_model(model, epochs=10, resolution=res, device=device)
                
                model.eval()
                with torch.no_grad():
                    x_test, y_test = generate_grf_data(batch_size=16, resolution=res, device=device)
                    pred = model(x_test)
                    l2_err, h1_err = compute_h1_physical(pred, y_test)
                    
                results[name][res]["l2"].append(l2_err.item())
                results[name][res]["h1"].append(h1_err.item())

    # Generazione output LaTeX
    latex_str = [
        "\\begin{table}[h]", "\\centering",
        "\\caption{Trained Benchmark Comparison: Relative $L^2$ and $H^1$ Errors}",
        "\\label{tab:dif_fno_results_trained}",
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
    print("\nBenchmark completato con successo. File 'table_results.tex' generato.\n")
    print(table_out)

if __name__ == "__main__":
    main()
