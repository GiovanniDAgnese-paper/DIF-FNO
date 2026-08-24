import torch
import torch.optim as optim
from train_diff_fno import DIFFNO2d
from update_dataset_geometries import generate_multi_geometry_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Dispositivo in uso: {device}")

model = DIFFNO2d(modes1=8, modes2=8, width=32, canonical_res=64).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)

print("Inizio Addestramento Multi-Geometria...")

# Generazione dati combinati
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
x_flat = x.reshape(B, H * W, C).to(device)
f_flat = f.reshape(B, H * W, 1).to(device)
mask_flat = mask.reshape(B, H * W, 1).to(device)
u_true = u_true.reshape(B, H * W).to(device)

model.train()
for epoch in range(1, 121):
    optimizer.zero_grad()
    u_pred = model(x_flat, f_flat, mask_flat)
    
    diff = (u_pred - u_true) * mask_flat.squeeze(-1)
    loss = torch.mean(diff**2) / (torch.mean(u_true**2) + 1e-8)
    
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    if epoch % 20 == 0 or epoch == 1:
        print(f"Epoca [{epoch:3d}/120] | Loss Multi-Geometria: {loss.item()*100:.2f}%")

torch.save(model.state_dict(), "dif_fno_weights.pt")
print("\nNuovi pesi multi-geometria salvati in 'dif_fno_weights.pt'.")
