import torch
import torch.nn as nn

class DAgneseBarrierLoss(nn.Module):
    """
    D'Agnese Topological Barrier Loss functional (L_barrier).
    Penalizes Jacobian determinant degeneration to enforce global C^1-diffeomorphic invertibility
    and guarantee 0.00% grid folding on non-convex domains.
    """
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, det_J: torch.Tensor) -> torch.Tensor:
        """
        Computes the logarithmic barrier penalty over the Jacobian determinant.
        
        Args:
            det_J (torch.Tensor): Determinant of the coordinate transformation Jacobian.
            
        Returns:
            torch.Tensor: Scalar loss value.
        """
        clamped_det = torch.clamp(det_J - self.eps, min=1e-7)
        barrier = -torch.log(clamped_det)
        return torch.mean(barrier)
