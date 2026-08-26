import torch
import torch.nn as nn

class DiffeomorphicMapping(nn.Module):
    r"""
    Explicit Diffeomorphic Mapping phi: \Omega_l -> \Omega_p
    Enforces det(J) > 0 and provides geometric chain-rule gradient transformation.
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2)
        )
        nn.init.normal_(self.net[-1].weight, std=1e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, xi_eta):
        r"""
        xi_eta: Tensor of shape (B, H, W, 2) or (N, 2) in latent space [-1, 1]^2
        """
        shape = xi_eta.shape
        is_grad_enabled = torch.is_grad_enabled()
        
        # Explicitly enable grad locally to compute Jacobian J even during inference
        with torch.enable_grad():
            grid_flat = xi_eta.reshape(-1, 2).detach().requires_grad_(True)
            disp = self.net(grid_flat)
            phys_flat = grid_flat + disp

            # Compute Jacobian components with retain_graph=True to preserve graph between dx and dy
            dx = torch.autograd.grad(
                phys_flat[:, 0].sum(), 
                grid_flat, 
                create_graph=is_grad_enabled, 
                retain_graph=True
            )[0]
            
            dy = torch.autograd.grad(
                phys_flat[:, 1].sum(), 
                grid_flat, 
                create_graph=is_grad_enabled, 
                retain_graph=is_grad_enabled
            )[0]

            J = torch.stack([dx, dy], dim=-2) # shape: (N, 2, 2)
            detJ = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]

            phys = phys_flat.reshape(shape)
            J = J.reshape(shape[:-1] + (2, 2))
            detJ = detJ.reshape(shape[:-1])

        if not is_grad_enabled:
            phys = phys.detach()
            J = J.detach()
            detJ = detJ.detach()

        return phys, J, detJ

    def compute_physical_gradients(self, grad_u_latent, J):
        r"""
        Transforms latent gradients \nabla_{(\xi,\eta)} u to physical gradients \nabla_{(x,y)} u
        via Inverse Transposed Jacobian: \nabla_{(x,y)} u = J^{-T} \nabla_{(\xi,\eta)} u
        """
        detJ = J[..., 0, 0] * J[..., 1, 1] - J[..., 0, 1] * J[..., 1, 0]
        detJ_safe = torch.clamp(detJ, min=1e-7).unsqueeze(-1)

        inv_J_T = torch.stack([
            torch.stack([J[..., 1, 1], -J[..., 0, 1]], dim=-1),
            torch.stack([-J[..., 1, 0], J[..., 0, 0]], dim=-1)
        ], dim=-2) / detJ_safe.unsqueeze(-1)

        grad_u_phys = torch.matmul(inv_J_T, grad_u_latent.unsqueeze(-1)).squeeze(-1)
        return grad_u_phys

    @staticmethod
    def jacobian_regularization_loss(detJ, eps=1e-3):
        r"""
        Barrier loss enforcing det(J) > eps > 0 everywhere to prevent grid folding.
        """
        violations = torch.clamp(eps - detJ, min=0.0)
        return torch.mean(violations ** 2)
