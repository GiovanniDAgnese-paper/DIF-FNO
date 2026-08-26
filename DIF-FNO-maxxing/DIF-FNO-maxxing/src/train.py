import torch
import torch.nn as nn
import torch.nn.functional as F
from model import DIFFNO2d, jacobian_barrier_loss

def train_step(model, optimizer, x, y, lambda_barrier=0.1):
    model.train()
    optimizer.zero_grad()
    
    out = model(x)
    mse_loss = F.mse_loss(out, y)
    
    # Grid diffeomorphism regularization
    barrier = jacobian_barrier_loss(out)
    total_loss = mse_loss + lambda_barrier * barrier
    
    total_loss.backward()
    optimizer.step()
    return total_loss.item(), mse_loss.item()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inizializzazione training su dispositivo: {device}")
    model = DIFFNO2d(modes1=12, modes2=12, width=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print("Modello DIF-FNO pronto per il benchmark.")
