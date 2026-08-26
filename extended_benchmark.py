import torch
from train_diff_fno import DIFFNO2d
from update_dataset_geometries import generate_multi_geometry_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Allineamento rigoroso dei parametri
model = DIFFNO2d(modes1=12, modes2=12, width=32, canonical_res=64).to(device)

try:
    model.load_state_dict(torch.load("dif_fno_weights.pt", map_location=device))
    print("Peso dif_fno_weights.pt (High-Cap) caricato con successo!")
except Exception as e:
    print(f"Errore nel caricamento: {e}")

model.eval()

geometries = ['star', 'l_shape', 'annulus']
resolutions = [64, 128]

print("\n=== RISULTATI BENCHMARK FINALE (HIGH CAPACITY) ===")
print(f"{'Geometria':<12} | {'Risoluzione':<12} | {'Rel L2 Err (%)':<15} | {'Rel H1 Err (%)':<15}")
print("-" * 62)

for geom in geometries:
    for res in resolutions:
        x, f, u_true, mask = generate_multi_geometry_dataset(num_samples=50, res=res, geom_type=geom)
        
        B, H, W, C = x.shape
        x_flat = x.reshape(B, H * W, C).to(device)
        f_flat = f.reshape(B, H * W, 1).to(device)
        mask_flat = mask.reshape(B, H * W, 1).to(device)
        u_true = u_true.to(device)
        mask = mask.to(device)
        
        with torch.no_grad():
            u_pred_flat = model(x_flat, f_flat, mask_flat)
            u_pred = u_pred_flat.reshape(B, H, W)
            
            diff_l2 = torch.norm((u_pred - u_true) * mask, p=2, dim=(1,2))
            ref_l2 = torch.norm(u_true * mask, p=2, dim=(1,2)) + 1e-8
            rel_l2 = torch.mean(diff_l2 / ref_l2).item() * 100.0
            
            grad_pred_x = torch.gradient(u_pred, dim=1)[0]
            grad_true_x = torch.gradient(u_true, dim=1)[0]
            diff_h1 = torch.norm((grad_pred_x - grad_true_x) * mask, p=2, dim=(1,2))
            ref_h1 = torch.norm(grad_true_x * mask, p=2, dim=(1,2)) + 1e-8
            rel_h1 = torch.mean(diff_h1 / ref_h1).item() * 100.0
            
        print(f"{geom:<12} | {f'{res}x{res}':<12} | {rel_l2:<15.2f} | {rel_h1:<15.2f}")
