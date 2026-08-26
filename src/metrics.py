import torch

def relative_l2_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Calcola l'errore relativo L2 espresso in frazione."""
    diff_norms = torch.norm(pred.reshape(pred.size(0), -1) - target.reshape(target.size(0), -1), p=2, dim=1)
    target_norms = torch.norm(target.reshape(target.size(0), -1), p=2, dim=1)
    return torch.mean(diff_norms / (target_norms + 1e-8))

def relative_h1_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Calcola l'errore relativo nella norma Sobolev H1 (valore + derivate spaziali)."""
    l2_err = relative_l2_error(pred, target)
    
    pred_dx = pred[:, 1:, :] - pred[:, :-1, :]
    target_dx = target[:, 1:, :] - target[:, :-1, :]
    pred_dy = pred[:, :, 1:] - pred[:, :, :-1]
    target_dy = target[:, :, 1:] - target[:, :, :-1]
    
    grad_l2_x = relative_l2_error(pred_dx, target_dx)
    grad_l2_y = relative_l2_error(pred_dy, target_dy)
    
    return torch.sqrt(l2_err**2 + grad_l2_x**2 + grad_l2_y**2)
