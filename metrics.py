import torch

def compute_jacobian_and_physical_h1(pred, grid_phys):
    """
    Audit Diffeomorfico & Physical H1 Norm via Chain Rule:
    1. Calcola J = d(x,y)/d(xi,eta) via differenze finite sulla griglia latente.
    2. Calcola det(J) per verificare la condizione di non-folding (det(J) > 0).
    3. Inverte J per applicare J^{-T} * grad_latente(u) ottenendo il gradiente fisico reale.
    """
    if pred.ndim == 4 and pred.shape[1] == 1:
        pred = pred.permute(0, 2, 3, 1)

    B, H, W, _ = pred.shape
    
    d_xi = 2.0 / (H - 1)
    d_eta = 2.0 / (W - 1)
    
    X = grid_phys[..., 0]
    Y = grid_phys[..., 1]
    
    # Derivate parziali delle coordinate fisiche (elementi di J)
    dx_dxi = torch.gradient(X, spacing=d_xi, dim=1)[0]
    dx_deta = torch.gradient(X, spacing=d_eta, dim=2)[0]
    dy_dxi = torch.gradient(Y, spacing=d_xi, dim=1)[0]
    dy_deta = torch.gradient(Y, spacing=d_eta, dim=2)[0]
    
    # Determinante dello Jacobiano
    det_J = dx_dxi * dy_deta - dx_deta * dy_dxi
    
    # Derivate latenti della predizione
    u = pred[..., 0]
    du_dxi = torch.gradient(u, spacing=d_xi, dim=1)[0]
    du_deta = torch.gradient(u, spacing=d_eta, dim=2)[0]
    
    # Inversione analitica 2x2 e applicazione Chain Rule J^{-T}
    det_J_safe = torch.clamp(det_J, min=1e-7)
    
    du_dx = (dy_deta * du_dxi - dy_dxi * du_deta) / det_J_safe
    du_dy = (-dx_deta * du_dxi + dx_dxi * du_deta) / det_J_safe
    
    # Norma H1 integrata nello spazio fisico
    grad_sq_phys = du_dx**2 + du_dy**2
    h1_norm = torch.sqrt(torch.mean(grad_sq_phys * torch.abs(det_J_safe)))
    
    return det_J, h1_norm
