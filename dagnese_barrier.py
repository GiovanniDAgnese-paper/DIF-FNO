import torch
import torch.nn as nn
import torch.nn.functional as F

class DAgneseBarrierLoss(nn.Module):
    """
    D'Agnese Differentiable Barrier Loss.
    Formulazione continua e differenziabile ovunque per prevenire il grid folding
    negli operatori neurali (DIF-FNO).
    """
    def __init__(self, alpha=100.0, eps=1e-4):
        super(DAgneseBarrierLoss, self).__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(self, J):
        # Determinante analitico veloce 2x2: ad - bc
        det_J = J[..., 0, 0] * J[..., 1, 1] - J[..., 0, 1] * J[..., 1, 0]
        
        # Penalità inversa per det(J) -> 0 e quadratica forte se det(J) <= 0
        # Mantiene il gradiente continuo e tracciabile al 100% da autograd
        relu_viol = F.relu(self.eps - det_J)
        barrier = torch.where(
            det_J > self.eps,
            -torch.log(det_J),
            -torch.log(torch.tensor(self.eps, device=J.device)) + 1e4 * (relu_viol ** 2)
        )
        
        return self.alpha * barrier.mean()

def get_compiled_dagnese_loss(alpha=100.0, eps=1e-4):
    loss_module = DAgneseBarrierLoss(alpha=alpha, eps=eps)
    return torch.compile(loss_module)
