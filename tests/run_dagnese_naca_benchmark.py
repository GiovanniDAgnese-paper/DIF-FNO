import torch
import time
from dagnese_fno.dagnese_mesh_generator import generate_dagnese_naca0012_mesh

def evaluate_dagnese_dif_fno_on_naca():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolutions = [(128, 128), (256, 256), (512, 512)]
    
    print("=" * 70)
    print("   D'AGNESE DIF-FNO: NACA 0012 MULTI-RESOLUTION TOPOLOGICAL BENCHMARK")
    print("=" * 70)
    
    for ny, nx in resolutions:
        # Griglia di riferimento nel dominio computazionale standard [0, 1] x [-1, 1]
        mesh = generate_dagnese_naca0012_mesh(nx, ny).to(device) # shape: (ny, nx, 2)
        
        # dX/d_dim0, dX/d_dim1
        grad_x_y, grad_x_x = torch.gradient(mesh[..., 0])
        # dY/d_dim0, dY/d_dim1
        grad_y_y, grad_y_x = torch.gradient(mesh[..., 1])
        
        # Determinante corretto dello Jacobiano: dX/dx * dY/dy - dX/dy * dY/dx
        jac_det = grad_x_x * grad_y_y - grad_x_y * grad_y_x
        
        min_det = jac_det.min().item()
        folding_count = (jac_det <= 0).sum().item()
        folding_rate = (folding_count / jac_det.numel()) * 100.0
        
        print(f"Risoluzione: {nx:3d}x{ny:3d} | Min det(J): {min_det:.6f} | Grid Folding: {folding_rate:.2f}%")

    print("=" * 70)
    if folding_rate == 0.0:
        print("Risultato: Diffeomorfismo verificato (0.00% grid folding).")
    else:
        print(f"Risultato: Rilevato grid folding del {folding_rate:.2f}%. Applicare la D'Agnese Topological Barrier Loss.")

if __name__ == "__main__":
    evaluate_dagnese_dif_fno_on_naca()
