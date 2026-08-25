import os
import time
import numpy as np
import torch
from benchmark_models import DIFFNO2d, GeoFNO2d, FNOMask2d, ensure_channel_last

def relative_l2_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    diff_norms = torch.norm(pred.reshape(pred.shape[0], -1) - target.reshape(target.shape[0], -1), p=2, dim=1)
    target_norms = torch.norm(target.reshape(target.shape[0], -1), p=2, dim=1)
    return torch.mean(diff_norms / (target_norms + 1e-8)).item()

def relative_h1_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_c = ensure_channel_last(pred)
    target_c = ensure_channel_last(target)

    grad_pred_y = pred_c[:, 1:, :, :] - pred_c[:, :-1, :, :]
    grad_pred_x = pred_c[:, :, 1:, :] - pred_c[:, :, :-1, :]

    grad_target_y = target_c[:, 1:, :, :] - target_c[:, :-1, :, :]
    grad_target_x = target_c[:, :, 1:, :] - target_c[:, :, :-1, :]

    l2_val = relative_l2_error(pred_c, target_c)
    h1_y = relative_l2_error(grad_pred_y, grad_target_y)
    h1_x = relative_l2_error(grad_pred_x, grad_target_x)

    return (l2_val + h1_y + h1_x) / 3.0

def generate_synthetic_pde_data(batch_size: int, resolution: int, device: torch.device):
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(0, 1, resolution, device=device),
        torch.linspace(0, 1, resolution, device=device),
        indexing="ij"
    )
    x_in = torch.sin(2 * np.pi * grid_x).unsqueeze(0).repeat(batch_size, 1, 1).unsqueeze(-1)
    target = (1.0 / (8 * (np.pi**2))) * torch.sin(2 * np.pi * grid_x).unsqueeze(0).repeat(batch_size, 1, 1).unsqueeze(-1)

    return x_in, target

def run_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing DIF-FNO Benchmark on Device: {device}")

    seeds = [42, 123, 456, 789, 999]
    resolutions = [64, 128, 256]
    models = {
        "DIF-FNO (Ours)": DIFFNO2d(),
        "Geo-FNO": GeoFNO2d(),
        "Masked-FNO": FNOMask2d()
    }

    results = {name: {res: {"l2": [], "h1": []} for res in resolutions} for name in models.keys()}

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        for res in resolutions:
            x_in, target = generate_synthetic_pde_data(batch_size=8, resolution=res, device=device)

            for name, model in models.items():
                model.to(device)
                model.eval()

                with torch.no_grad():
                    start_time = time.time()
                    pred = model(x_in)
                    _ = time.time() - start_time

                l2_err = relative_l2_error(pred, target)
                h1_err = relative_h1_error(pred, target)

                results[name][res]["l2"].append(l2_err)
                results[name][res]["h1"].append(h1_err)

    latex_str = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Full Benchmark Comparison: Relative $L^2$ and $H^1$ Errors across Resolutions}",
        "\\label{tab:dif_fno_results}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        " & \\multicolumn{2}{c}{$64 \\times 64$} & \\multicolumn{2}{c}{$128 \\times 128$} & \\multicolumn{2}{c}{$256 \\times 256$} \\\\",
        "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7}",
        "Model & $L^2$ Error & $H^1$ Error & $L^2$ Error & $H^1$ Error & $L^2$ Error & $H^1$ Error \\\\",
        "\\midrule"
    ]

    for name in models.keys():
        row = f"{name}"
        for res in resolutions:
            l2_mean = np.mean(results[name][res]["l2"])
            l2_std = np.std(results[name][res]["l2"])
            h1_mean = np.mean(results[name][res]["h1"])
            h1_std = np.std(results[name][res]["h1"])
            row += f" & {l2_mean:.4f} \\pm {l2_std:.4f} & {h1_mean:.4f} \\pm {h1_std:.4f}"
        row += " \\\\"
        latex_str.append(row)

    latex_str.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])

    table_content = "\n".join(latex_str)
    with open("table_results.tex", "w") as f:
        f.write(table_content)

    print("\nBenchmark completato con successo. File 'table_results.tex' generato.\n")
    print(table_content)

if __name__ == "__main__":
    run_evaluation()
