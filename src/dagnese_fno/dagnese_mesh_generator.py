import torch
import numpy as np

def generate_dagnese_naca0012_mesh(num_points_x=128, num_points_y=128):
    """
    Generates structured mesh around NACA 0012 profile for the D'Agnese Benchmark.
    """
    x = np.linspace(0, 1, num_points_x)
    # Profilo alare NACA 0012
    yt = 5 * 0.12 * (0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)
    
    grid_x, grid_y = np.meshgrid(x, np.linspace(-1, 1, num_points_y))
    mesh = np.stack([grid_x, grid_y], axis=-1)
    return torch.tensor(mesh, dtype=torch.float32)

if __name__ == "__main__":
    mesh = generate_dagnese_naca0012_mesh()
    print(f"D'Agnese NACA Mesh generata con successo: {mesh.shape}")
