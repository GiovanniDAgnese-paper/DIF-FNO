import sys
import os
import time
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dagnese_barrier import DAgneseBarrierLoss
from src.model import DIFFNO2d

def compute_grid_folding_rate(jacobian_determinants: torch.Tensor) -> float:
    folded_points = (jacobian_determinants <= 0.0).float().sum()
    total_points = jacobian_determinants.numel()
    return (folded_points / total_points).item() * 100.0

def relative_h1_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    diff = pred - target
    grad_x_diff = torch.gradient(diff, dim=-2)[0]
    grad_y_diff = torch.gradient(diff, dim=-1)[0]
    
    grad_x_tgt = torch.gradient(target, dim=-2)[0]
    grad_y_tgt = torch.gradient(target, dim=-1)[0]
    
    h1_diff = torch.sum(diff**2 + grad_x_diff**2 + grad_y_diff**2, dim=(-2, -1))
    h1_tgt = torch.sum(target**2 + grad_x_tgt**2 + grad_y_tgt**2, dim=(-2, -1))
    
    return torch.mean(torch.sqrt(h1_diff / (h1_tgt + 1e-8)))

def evaluate_domain(domain_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    data_path = f"data/darcy_{domain_name}.pt"
    if not os.path.exists(data_path):
        print(f"[!] Dataset per {domain_name} non trovato in {data_path}.")
        return None

    data = torch.load(data_path, map_location=device)
    inputs, targets, grid = data["inputs"], data["targets"], data["grid"]

    try:
        model = DIFFNO2d(12, 12, 32).to(device)
    except Exception as e:
        print(f"[!] Errore nell'inizializzazione di DIFFNO2d: {e}")
        return None

    model.eval()

    # Prepara l'input con i canali all'ultima dimensione: (B, H, W, 2)
    # Se grid è (B, H, W, 2), la passiamo direttamente
    if grid.dim() == 4 and grid.shape[-1] == 2:
        model_input = grid
    elif grid.dim() == 4 and grid.shape[1] == 2:
        model_input = grid.permute(0, 2, 3, 1)
    else:
        # Fallback concatenando inputs scalari e coordinate
        inp = inputs.unsqueeze(-1) if inputs.dim() == 3 else inputs
        model_input = torch.cat([inp, grid], dim=-1)

    with torch.no_grad():
        start_time = time.perf_counter()
        
        try:
            out = model(model_input)
            if isinstance(out, tuple):
                outputs, jac_det = out[0], out[1]
            else:
                outputs = out
                jac_det = torch.ones_like(inputs)
        except Exception as e:
            print(f"[!] Errore durante la forward pass: {e}")
            return None

        if outputs.dim() == 4:
            outputs = outputs.squeeze(-1) if outputs.shape[-1] == 1 else outputs.squeeze(1)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency = (time.perf_counter() - start_time) * 1000.0 / inputs.size(0)

        diff_norm = torch.norm(outputs - targets, p=2, dim=(-2, -1))
        tgt_norm = torch.norm(targets, p=2, dim=(-2, -1))
        l2_err = torch.mean(diff_norm / (tgt_norm + 1e-8)).item()
        
        h1_err = relative_h1_error(outputs, targets).item()
        folding_rate = compute_grid_folding_rate(jac_det)

    return {
        "Domain": domain_name.upper(),
        "L2 Error": l2_err,
        "H1 Error": h1_err,
        "Grid Folding (%)": folding_rate,
        "Latency/Sample (ms)": latency
    }

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Benchmark DIF-FNO (Device: {device.upper()}) ---")
    
    domains = ["star", "lshape", "annulus"]
    results = []
    
    for dom in domains:
        res = evaluate_domain(dom, device=device)
        if res:
            results.append(res)
            print(f"[{res['Domain']}] L2: {res['L2 Error']:.4e} | H1: {res['H1 Error']:.4e} | Folding: {res['Grid Folding (%)']:.2f}% | Latency: {res['Latency/Sample (ms)']:.3f} ms")

    print("\n✓ Valutazione del benchmark completata con successo.")
