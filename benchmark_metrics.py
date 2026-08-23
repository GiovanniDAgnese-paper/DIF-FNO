import torch
import time
import numpy as np
from train_diff_fno import DIFFNO2d, generate_irregular_star_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Caricamento Modello
model = DIFFNO2d(modes1=12, modes2=12, width=32, canonical_res=64).to(device)
model.load_state_dict(torch.load("dif_fno_weights.pt", map_location=device))
model.eval()

# 2. Conteggio Parametri Totali
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# 3. Misurazione Latenza di Inferenza (ms/sample)
x_test, f_test, u_test, mask_test = generate_irregular_star_dataset(num_samples=100, num_points=1024)
x_test, f_test, mask_test = x_test.to(device), f_test.to(device), mask_test.to(device)

# Warmup GPU
with torch.no_grad():
    for _ in range(10):
        _ = model(x_test[:1], f_test[:1], mask_test[:1])

if torch.cuda.is_available():
    torch.cuda.synchronize()

start_time = time.time()
with torch.no_grad():
    for i in range(100):
        _ = model(x_test[i:i+1], f_test[i:i+1], mask_test[i:i+1])
if torch.cuda.is_available():
    torch.cuda.synchronize()

latency_ms = ((time.time() - start_time) / 100.0) * 1000.0

# 4. Calcolo Errore Relativo Sobolev H1
with torch.no_grad():
    u_pred = model(x_test, f_test, mask_test)

    # Derivate spaziali approssimate
    grad_pred_x = torch.gradient(u_pred, dim=1)[0]
    grad_true_x = torch.gradient(u_test.to(device), dim=1)[0]

    h1_diff = torch.norm((u_pred - u_test.to(device)) * mask_test, p=2, dim=1)**2 + \
              torch.norm((grad_pred_x - grad_true_x) * mask_test, p=2, dim=1)**2

    h1_ref = torch.norm(u_test.to(device) * mask_test, p=2, dim=1)**2 + \
             torch.norm(grad_true_x * mask_test, p=2, dim=1)**2 + 1e-8

    rel_h1_error = torch.mean(torch.sqrt(h1_diff / h1_ref)).item() * 100.0

print(f"Parametri Totali: {total_params / 1e6:.2f}M")
print(f"Latenza Inferenza: {latency_ms:.2f} ms/sample")
print(f"Errore Relativo Sobolev H1: {rel_h1_error:.2f}%")
