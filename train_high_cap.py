import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from train_diff_fno import DIFFNO2d
from update_dataset_geometries import generate_multi_geometry_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Dispositivo in uso: {device}")

# Modello ad alta capacità spettrale (modes=12, width=32)
model = DIFFNO2d(modes1=12, modes2=12, width=32, canonical_res=64).to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)

print("Generazione dataset multi-geometria...")
geoms = ['star', 'l_shape', 'annulus']
x_list, f_list, u_list, mask_list = [], [], [], []

for g in geoms:
    x_g, f_g, u_g, m_g = generate_multi_geometry_dataset(num_samples=150, res=64, geom_type=g)
    x_list.append(x_g)
    f_list.append(f_g)
    u_list.append(u_g)
    mask_list.append(m_g)

x = torch.cat(x_list, dim=0)
f = torch.cat(f_list, dim=0)
u_true = torch.cat(u_list, dim=0)
mask = torch.cat(mask_list, dim=0)

B, H, W, C = x.shape
x_flat = x.reshape(B, H * W, C)
f_flat = f.reshape(B, H * W, 1)
mask_flat = mask.reshape(B, H * W, 1)
u_true_flat = u_true.reshape(B, H * W)

# Mini-batching per prevenire OOM su RAM/CPU
dataset = TensorDataset(x_flat, f_flat, mask_flat, u_true_flat)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

print("Inizio Addestramento con Mini-Batching (150 Epoche)...")
model.train()
for epoch in range(1, 151):
    epoch_loss = 0.0
    for bx, bf, bmask, bu in loader:
        bx, bf, bmask, bu = bx.to(device), bf.to(device), bmask.to(device), bu.to(device)
        
        optimizer.zero_grad()
        u_pred = model(bx, bf, bmask)
        
        diff = (u_pred - bu) * bmask.squeeze(-1)
        loss = torch.mean(diff**2) / (torch.mean(bu**2) + 1e-8)
        
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * bx.size(0)
    
    scheduler.step()
    total_loss = (epoch_loss / B) * 100.0
    
    if epoch % 25 == 0 or epoch == 1:
        print(f"Epoca [{epoch:3d}/150] | Relative Loss Medio: {total_loss:.3f}%")

torch.save(model.state_dict(), "dif_fno_weights.pt")
print("\nPesi ad alta capacità salvati con successo in 'dif_fno_weights.pt'.")
