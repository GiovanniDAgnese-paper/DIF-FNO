import torch
import matplotlib.pyplot as plt
import numpy as np
from train_diff_fno import DIFFNO2d, generate_irregular_star_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DIFFNO2d(modes1=12, modes2=12, width=32, canonical_res=64).to(device)
model.load_state_dict(torch.load("dif_fno_weights.pt", map_location=device))
model.eval()

# Generazione di 1 campione di test
x, f, u_true, mask = generate_irregular_star_dataset(num_samples=1, num_points=2048)
x, f, u_true, mask = x.to(device), f.to(device), u_true.to(device), mask.to(device)

with torch.no_grad():
    u_pred = model(x, f, mask)

# Normalizzazione delle forme dimensionali per evitare broadcasting non voluto
coords = x[0].cpu().numpy()                       # Dimensione: (2048, 2)
true_sol = u_true[0].cpu().numpy().reshape(-1)   # Dimensione: (2048,)
pred_sol = u_pred[0].cpu().numpy().reshape(-1)   # Dimensione: (2048,)
error = np.abs(true_sol - pred_sol)              # Dimensione: (2048,)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

sc0 = axes[0].scatter(coords[:, 0], coords[:, 1], c=true_sol, cmap='viridis', s=15)
axes[0].set_title("Ground Truth (Poisson Sol.)", fontsize=12, fontweight='bold')
fig.colorbar(sc0, ax=axes[0])

sc1 = axes[1].scatter(coords[:, 0], coords[:, 1], c=pred_sol, cmap='viridis', s=15)
axes[1].set_title("Predizione DIF-FNO", fontsize=12, fontweight='bold')
fig.colorbar(sc1, ax=axes[1])

sc2 = axes[2].scatter(coords[:, 0], coords[:, 1], c=error, cmap='inferno', s=15)
axes[2].set_title("Errore Assoluto |u_true - u_pred|", fontsize=12, fontweight='bold')
fig.colorbar(sc2, ax=axes[2])

for ax in axes:
    ax.set_aspect('equal')
    ax.axis('off')

plt.tight_layout()
plt.savefig("figure1_dif_fno_results.png", dpi=300, bbox_inches='tight')
print("--> Immagine 'figure1_dif_fno_results.png' generata con successo a 300 DPI.")
