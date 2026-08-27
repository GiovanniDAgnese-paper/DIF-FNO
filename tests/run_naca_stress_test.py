import torch
import time
from dagnese_fno.dagnese_mesh_generator import generate_dagnese_naca0012_mesh

def run_naca_multi_res_benchmark():
    resolutions = [(128, 128), (256, 256), (512, 512)]
    print("=" * 65)
    print("      NACA 0012 MULTI-RESOLUTION TOPOLOGICAL BENCHMARK")
    print("=" * 65)
    
    for nx, ny in resolutions:
        start_time = time.perf_counter()
        mesh = generate_dagnese_naca0012_mesh(nx, ny)
        gen_time = (time.perf_counter() - start_time) * 1000
        
        # Calcolo approssimato del gradiente spaziale della griglia NACA 0012
        dx = mesh[1:, :, 0] - mesh[:-1, :, 0]
        dy = mesh[:, 1:, 1] - mesh[:, :-1, 1]
        
        print(f"Risoluzione: {nx:3d}x{ny:3d} | Shape: {str(list(mesh.shape)):15s} | Tempo Mesh: {gen_time:6.2f} ms")

    print("=" * 65)
    print("Griglie NACA 0012 generate correttamente per il modello DIF-FNO.")

if __name__ == "__main__":
    run_naca_multi_res_benchmark()
