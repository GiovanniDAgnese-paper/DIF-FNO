import torch
import time

def run_dagnese_topological_stress_test(model, test_loader, device="cuda"):
    """
    D'Agnese Topological Stress Benchmark for NACA 0012 Airfoil Geometries.
    Evaluates D'Agnese Diffeomorphic Integrity under boundary boundary gradients.
    """
    model.eval()
    min_jacobians = []
    latencies = []
    dagnese_folding_events = 0
    
    with torch.no_grad():
        for x_batch, y_batch, mesh_coords in test_loader:
            x_batch, mesh_coords = x_batch.to(device), mesh_coords.to(device)
            
            start_time = time.perf_counter()
            # Forward pass con estrazione dello Jacobiano D'Agnese
            pred, jac_det = model.forward_with_jacobian(x_batch, mesh_coords)
            latencies.append((time.perf_counter() - start_time) / x_batch.size(0))
            
            min_jac = jac_det.min().item()
            min_jacobians.append(min_jac)
            if min_jac <= 0:
                dagnese_folding_events += 1
            
    overall_min_jac = min(min_jacobians)
    dagnese_folding_rate = (dagnese_folding_events / len(min_jacobians)) * 100
    avg_latency_ms = (sum(latencies) / len(latencies)) * 1000
    
    print("=" * 65)
    print("      D'AGNESE AIRFOIL TOPOLOGICAL STRESS BENCHMARK RESULTS")
    print("=" * 65)
    print(f"D'Agnese Minimum Det(J)         : {overall_min_jac:.6f}")
    print(f"D'Agnese Grid Folding Rate      : {dagnese_folding_rate:.2f}%")
    print(f"D'Agnese Inference Latency      : {avg_latency_ms:.2f} ms/sample")
    print("=" * 65)
    
    return overall_min_jac, dagnese_folding_rate, avg_latency_ms

if __name__ == "__main__":
    print("D'Agnese Airfoil Stress Benchmark Module Ready.")
