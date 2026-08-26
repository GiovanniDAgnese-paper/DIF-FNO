import torch
import torch.nn as nn
import time
from model import DIFFNO2d, jacobian_barrier_loss
from geometry import generate_star_domain, generate_annulus_domain
from metrics import relative_l2_error, relative_h1_error

def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== DIF-FNO Benchmark Execution on {device} ===")
    
    # Domini di test
    star_data = generate_star_domain(grid_size=64, num_samples=50).to(device)
    
    # Inizializzazione modello DIF-FNO
    model = DIFFNO2d(modes1=12, modes2=12, width=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Dummy Target per simulazione PDE
    target = torch.sin(star_data[..., 0:1]) * torch.cos(star_data[..., 1:2])
    
    print("\n--- Training DIF-FNO (con Jacobian Barrier Loss) ---")
    start_time = time.time()
    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()
        out = model(star_data)
        
        mse = torch.nn.functional.mse_loss(out, target)
        barrier = jacobian_barrier_loss(out)
        loss = mse + 0.1 * barrier
        
        loss.backward()
        optimizer.step()
        
        if epoch % 25 == 0:
            l2 = relative_l2_error(out, target).item()
            h1 = relative_h1_error(out, target).item()
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.6f} | Barrier: {barrier.item():.6f} | Rel L2: {l2:.4f} | Rel H1: {h1:.4f}")
            
    elapsed = time.time() - start_time
    print(f"\nBenchmark completato in {elapsed:.2f}s")

if __name__ == "__main__":
    run_benchmark()
