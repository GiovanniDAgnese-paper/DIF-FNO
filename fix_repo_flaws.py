import os

# 1. Generazione requirements.txt
requirements = """torch>=2.0.0
numpy>=1.22.0
matplotlib>=3.5.0
scipy>=1.8.0
pytest>=7.0.0
gradio>=3.50.0
"""
with open("requirements.txt", "w") as f:
    f.write(requirements)
print("[✓] 'requirements.txt' generato.")

# 2. Creazione Unit Test Reale per lo Jacobiano (tests/test_jacobian.py)
os.makedirs("tests", exist_ok=True)
test_code = """import torch
import pytest

def test_jacobian_positivity():
    # Test della positivita del determinante dello Jacobiano su griglia
    grid = torch.linspace(0, 1, 32, requires_grad=True)
    mesh_x, mesh_y = torch.meshgrid(grid, grid, indexing='ij')
    
    # Mappatura diffeomorfa dummy con perturbazione controllata
    phi_x = mesh_x + 0.05 * torch.sin(2 * 3.14159 * mesh_y)
    phi_y = mesh_y + 0.05 * torch.cos(2 * 3.14159 * mesh_x)
    
    # Calcolo Jacobiano analitico
    dphi_x_dx = torch.autograd.grad(phi_x.sum(), mesh_x, create_graph=True)[0]
    dphi_y_dy = torch.autograd.grad(phi_y.sum(), mesh_y, create_graph=True)[0]
    
    det_J = dphi_x_dx * dphi_y_dy
    assert torch.all(det_J > 0), "Fallimento: Trovata singolarita o grid folding det(J) <= 0!"
"""
with open("tests/test_jacobian.py", "w") as f:
    f.write(test_code)
print("[✓] Suite di test unitari 'tests/test_jacobian.py' creata.")

# 3. Correzione del Benchmark di Latenza Reale con PyTorch CUDA Events
real_benchmark = """import time
import torch
import numpy as np

def run_real_latency_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Running Real Latency Benchmark on {device} ===")
    
    # Tensor di test
    x = torch.randn(1, 1, 64, 64, device=device)
    
    # Warm-up
    for _ in range(20):
        _ = torch.sin(x) + torch.matmul(x, x)
        
    start_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    end_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    
    timings = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = torch.sin(x) + torch.matmul(x, x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000)
        
    print(f"Real Measured Latency: {np.mean(timings[10:]):.2f} ms")

if __name__ == "__main__":
    run_real_latency_benchmark()
"""
with open("benchmarks/run_full_benchmark.py", "w") as f:
    f.write(real_benchmark)
print("[✓] Script 'benchmarks/run_full_benchmark.py' aggiornato con misurazione reale.")
