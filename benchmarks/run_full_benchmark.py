import time
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
