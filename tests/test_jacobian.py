import torch
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
