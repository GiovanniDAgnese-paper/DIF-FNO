import torch
import torch.optim as optim
import torch.nn.functional as F
from train_diff_fno import DIFFNO2d
from update_dataset_geometries import generate_multi_geometry_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Dispositivo in uso: {device}")

model = DIFFNO2d(modes1=8, modes2=8, width=32, canonical_res=64).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

print("Inizio Addestramento Reale su Geometria Irregolare...")

# Genera dataset di addestramento
x, f, u_true, mask = generate_multi_geometry_dataset(num_samples=300, res=64, geom_type='star')
B, H, W, C = x.shape
x_flat = x.reshape(B, H * W, C).to(device)
f_flat = f.reshape(B, H * W, 1).to(device)
mask_flat = mask.reshape(B, H * W, 1).to(device)
u_true = u_true.reshape(B, H * W).to(device)

model.train()
for epoch in range(1, 151):
    optimizer.zero_grad()
    u_pred = model(x_flat, f_flat, mask_flat)
    
    # Loss L2 pesata sulla maschera del dominio
    diff = (u_pred - u_true) * mask_flat.squeeze(-1)
    loss = torch.mean(diff**2) / (torch.mean(u_true**2) + 1e-8)
    
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    if epoch % 25 == 0 or epoch == 1:
        print(f"Epoca [{epoch:3d}/150] | Relative L2 Loss: {loss.item()*100:.2f}%")

# Salva i pesi reali e funzionanti
torch.save(model.state_dict(), "dif_fno_weights.pt")
print("\nAddestramento completato! Pesi salvati in 'dif_fno_weights.pt'.")
