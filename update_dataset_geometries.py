import torch
import numpy as np

def generate_multi_geometry_dataset(num_samples=200, res=64, geom_type='star'):
    grid_x = torch.linspace(-1, 1, res)
    grid_y = torch.linspace(-1, 1, res)
    x1, x2 = torch.meshgrid(grid_x, grid_y, indexing='ij')
    coords = torch.stack([x1, x2], dim=-1).reshape(-1, 2)
    
    r = torch.sqrt(coords[:, 0]**2 + coords[:, 1]**2)
    theta = torch.atan2(coords[:, 1], coords[:, 0])
    
    if geom_type == 'star':
        radius_boundary = 0.6 + 0.25 * torch.cos(5 * theta) + 0.1 * torch.sin(3 * theta)
        mask = (r <= radius_boundary).float()
    elif geom_type == 'l_shape':
        mask = ((coords[:, 0] <= 0.5) & (coords[:, 1] <= 0.5) & ~((coords[:, 0] > 0) & (coords[:, 1] > 0))).float()
    elif geom_type == 'annulus':
        mask = ((r >= 0.2) & (r <= 0.7)).float()
    else:
        raise ValueError("Geometria non supportata")

    f_list, u_list, x_list, mask_list = [], [], [], []
    for _ in range(num_samples):
        k1, k2 = np.random.randint(1, 4), np.random.randint(1, 4)
        f_val = torch.sin(k1 * np.pi * coords[:, 0]) * torch.cos(k2 * np.pi * coords[:, 1])
        u_val = f_val / ((k1 * np.pi)**2 + (k2 * np.pi)**2)
        
        f_list.append(f_val.reshape(res, res))
        u_list.append(u_val.reshape(res, res))
        x_list.append(coords.reshape(res, res, 2))
        mask_list.append(mask.reshape(res, res))
        
    return torch.stack(x_list), torch.stack(f_list), torch.stack(u_list), torch.stack(mask_list)

print("Modulo geometrie avanzate generato correttamente.")
