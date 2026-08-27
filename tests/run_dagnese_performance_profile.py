import sys
import os
import torch
import time
import inspect

sys.path.append(os.path.abspath("src"))
import model as model_module
from dagnese_fno.dagnese_mesh_generator import generate_dagnese_naca0012_mesh

def profile_dagnese_dif_fno_performance():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo di esecuzione: {device}")
    
    model_class = getattr(model_module, "DIFFNO2d", None)
    if model_class is None:
        print("DIFFNO2d non trovato in src/model.py")
        return

    print(f"Modello individuato: {model_class.__name__}")

    resolutions = [(128, 128), (256, 256), (512, 512)]
    
    print("=" * 80)
    print("   DIF-FNO: NACA 0012 MODEL INFERENCE & MEMORY BENCHMARK")
    print("=" * 80)
    
    # Parametri di default standard per modelli FNO/DIF-FNO 2D
    default_kwargs = {
        "modes1": 12,
        "modes2": 12,
        "width": 32
    }
    
    sig = inspect.signature(model_class.__init__)
    params = sig.parameters
    if "in_channels" in params:
        default_kwargs["in_channels"] = 2
    if "in_dim" in params:
        default_kwargs["in_dim"] = 2
        
    for ny, nx in resolutions:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            
        mesh = generate_dagnese_naca0012_mesh(nx, ny).unsqueeze(0).to(device)
        
        try:
            net = model_class(**default_kwargs).to(device)
        except Exception as e:
            print(f"Inizializzazione fallita per {nx}x{ny}: {e}")
            continue
            
        net.eval()
        
        # Gestione del layout dei tensori (B, H, W, C) vs (B, C, H, W)
        input_tensor = mesh
        with torch.no_grad():
            try:
                _ = net(input_tensor)
            except Exception:
                input_tensor = mesh.permute(0, 3, 1, 2)
                try:
                    _ = net(input_tensor)
                except Exception as e:
                    print(f"Forward pass fallito per {nx}x{ny}: {e}")
                    continue
                
        # Warmup pass
        with torch.no_grad():
            for _ in range(3):
                _ = net(input_tensor)
                
        # Misurazione latenza forward pass
        start_time = time.perf_counter()
        iterations = 20
        with torch.no_grad():
            for _ in range(iterations):
                _ = net(input_tensor)
                if device.type == "cuda":
                    torch.cuda.synchronize()
        
        elapsed_time_ms = ((time.perf_counter() - start_time) / iterations) * 1000
        
        if device.type == "cuda":
            mem_usage_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            mem_label = "VRAM Peak"
        else:
            param_size = sum(p.nelement() * p.element_size() for p in net.parameters()) / (1024 ** 2)
            mem_usage_mb = param_size
            mem_label = "Model RAM"
            
        print(f"Risoluzione: {nx:3d}x{ny:3d} | Latenza Forward: {elapsed_time_ms:6.2f} ms | {mem_label}: {mem_usage_mb:6.2f} MB")

    print("=" * 80)
    print("Profilatura reale del modello completata.")

if __name__ == "__main__":
    profile_dagnese_dif_fno_performance()
