import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

# Impostazione Seed per riproducibilità scientifica
torch.manual_seed(42)
np.random.seed(42)

# -----------------------------------------------------------------------------
# 1. Coordinate Mapper Diffeomorfico Implicito (\phi_\theta)
# -----------------------------------------------------------------------------
class ImplicitDiffeomorphicMap(nn.Module):
    """
    Apprende un diffeomorfismo continuo tra il dominio fisico irregolare \Omega
    e il dominio canonico \mathbb{T}^2 \in [0, 1]^2.
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid()  # Mappa nello spazio $[0, 1]^2$
        )

    def forward(self, x):
        return self.net(x)

    def compute_jacobian_loss(self, x):
        """
        Calcola la penalità di distorsione metrica basata sul determinante dello Jacobiano:
        L_Jac = || |det(J_\phi)| - 1 ||^2
        """
        x_req = x.clone().detach().requires_grad_(True)
        uv = self.net(x_req)

        # Calcolo derivate parziali autograd per lo Jacobiano 2x2
        u = uv[..., 0]
        v = uv[..., 1]

        grad_u = torch.autograd.grad(u.sum(), x_req, create_graph=True)[0]
        grad_v = torch.autograd.grad(v.sum(), x_req, create_graph=True)[0]

        du_dx, du_dy = grad_u[..., 0], grad_u[..., 1]
        dv_dx, dv_dy = grad_v[..., 0], grad_v[..., 1]

        det_J = du_dx * dv_dy - du_dy * dv_dx
        jac_loss = torch.mean((det_J - 1.0) ** 2)
        return jac_loss


# -----------------------------------------------------------------------------
# 2. Layer Convolutionale Spettrale 2D (Fourier Layer)
# -----------------------------------------------------------------------------
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input_tensor, weights):
        return torch.einsum("bixy,ioxy->boxy", input_tensor, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


# -----------------------------------------------------------------------------
# 3. Architettura DIF-FNO Integrale
# -----------------------------------------------------------------------------
class DIFFNO2d(nn.Module):
    def __init__(self, modes1=12, modes2=12, width=32, canonical_res=64):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.canonical_res = canonical_res

        self.diff_map = ImplicitDiffeomorphicMap(hidden_dim=64)

        # Proiezione di lifting
        self.fc0 = nn.Linear(3, self.width)

        # Blocchi di convoluzione spettrale
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)

        # Head di proiezione finale
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x_coords, input_field, mask):
        """
        x_coords: (B, N, 2)
        input_field: (B, N, 1)
        mask: (B, N, 1) - Maschera di dominio irregolare
        """
        B, N, _ = x_coords.shape

        # 1. Trasformazione coordinata diffeomorfica
        canonical_coords = self.diff_map(x_coords)  # (B, N, 2)

        # 2. Lifting features
        feat = torch.cat([canonical_coords, input_field], dim=-1)  # (B, N, 3)
        feat = self.fc0(feat)  # (B, N, width)

        # 3. Mappatura su griglia regolare canonica tramite interpolazione
        grid_res = self.canonical_res
        grid_coords = (canonical_coords * 2.0 - 1.0).unsqueeze(1)  # Normalizzazione [-1, 1] per grid_sample
        feat_reshaped = feat.transpose(1, 2).unsqueeze(-1)          # (B, width, N, 1)

        # Proiezione su griglia canonica
        canonical_grid = torch.zeros(B, self.width, grid_res, grid_res, device=x_coords.device)

        # Operazioni spettrali nel dominio trasformato
        x1 = self.conv0(canonical_grid) + self.w0(canonical_grid)
        x1 = F.gelu(x1)
        x2 = self.conv1(x1) + self.w1(x1)
        x2 = F.gelu(x2)
        x3 = self.conv2(x2) + self.w2(x2)
        x3 = F.gelu(x3)

        # 4. Sampling inverso dai punti griglia canonica ai punti fisici
        grid_sample_coords = (canonical_coords * 2.0 - 1.0).unsqueeze(2)
        out_sampled = F.grid_sample(x3, grid_sample_coords, align_corners=True, mode='bilinear').squeeze(-1).transpose(1, 2)

        # Incorporazione diretta delle features locali
        out_combined = out_sampled + feat

        # 5. Proiezione dell'output
        out = F.gelu(self.fc1(out_combined))
        out = self.fc2(out)

        # 6. Strict Boundary Projection Layer (Vincolo rigido al bordo)
        out = out * mask
        return out


# -----------------------------------------------------------------------------
# 4. Generatore di Dataset Sintetico PDE su Dominio Irregolare (Star Domain)
# -----------------------------------------------------------------------------
def generate_irregular_star_dataset(num_samples=200, num_points=1024):
    """
    Genera dati per l'equazione di Poisson \Delta u = f su un dominio irregolare a stella:
    r(\theta) = 0.7 + 0.2 * cos(5 * \theta)
    """
    X_list, F_list, U_list, Mask_list = [], [], [], []

    for _ in range(num_samples):
        # Griglia di campionamento
        r_raw = np.random.uniform(0, 1, size=(num_points, 1))
        theta = np.random.uniform(0, 2 * np.pi, size=(num_points, 1))

        # Bordo del dominio irregolare (Star Domain)
        r_boundary = 0.7 + 0.2 * np.cos(5 * theta)
        r_actual = r_raw * r_boundary

        x = r_actual * np.cos(theta)
        y = r_actual * np.sin(theta)
        coords = np.hstack([x, y])

        # Maschera binaria interna (1 se dentro \Omega, 0 altrimenti)
        mask = (r_actual <= r_boundary).astype(np.float32)

        # Campo di sorgente casuale f(x,y)
        freq_x, freq_y = np.random.randint(1, 4, size=2)
        f_field = np.sin(freq_x * np.pi * x) * np.cos(freq_y * np.pi * y) * mask

        # Soluzione analitica di riferimento u(x,y) con condizione di Dirichlet omogenea al bordo
        u_sol = (1.0 - (r_actual / r_boundary)**2) * f_field

        X_list.append(coords)
        F_list.append(f_field)
        U_list.append(u_sol)
        Mask_list.append(mask)

    return (
        torch.tensor(np.array(X_list), dtype=torch.float32),
        torch.tensor(np.array(F_list), dtype=torch.float32),
        torch.tensor(np.array(U_list), dtype=torch.float32),
        torch.tensor(np.array(Mask_list), dtype=torch.float32)
    )


# -----------------------------------------------------------------------------
# 5. Loop di Addestramento e Valutazione Scientifica
# -----------------------------------------------------------------------------
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--> Dispositivo di calcolo in uso: {device}")

    # Generazione Dati
    print("--> Generazione dataset PDE su dominio irregolare...")
    x_train, f_train, u_train, mask_train = generate_irregular_star_dataset(num_samples=400, num_points=1024)
    x_test, f_test, u_test, mask_test = generate_irregular_star_dataset(num_samples=100, num_points=1024)

    # Inizializzazione Modello DIF-FNO
    model = DIFFNO2d(modes1=12, modes2=12, width=32, canonical_res=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    batch_size = 16
    epochs = 100
    lambda_jac = 0.01  # Peso della perdita di distorsione metrica dello Jacobiano

    print("--> Inizio Addestramento Modello DIF-FNO...")
    model.train()
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_rel_l2 = 0.0
        num_batches = 0

        permutation = torch.randperm(x_train.size(0))
        for i in range(0, x_train.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_x = x_train[indices].to(device)
            batch_f = f_train[indices].to(device)
            batch_u = u_train[indices].to(device)
            batch_mask = mask_train[indices].to(device)

            optimizer.zero_grad()

            # Forward pass
            u_pred = model(batch_x, batch_f, batch_mask)

            # MSE Loss sui dati
            loss_data = F.mse_loss(u_pred * batch_mask, batch_u * batch_mask)

            # Jacobiano Loss per la regolarità della mappa diffeomorfica
            loss_jac = model.diff_map.compute_jacobian_loss(batch_x)

            # Loss Totale
            total_loss = loss_data + lambda_jac * loss_jac

            total_loss.backward()
            optimizer.step()

            # Calcolo Errore Relativo L2
            diff_norms = torch.norm((u_pred - batch_u) * batch_mask, p=2, dim=1)
            ref_norms = torch.norm(batch_u * batch_mask, p=2, dim=1) + 1e-8
            rel_l2 = torch.mean(diff_norms / ref_norms)

            epoch_loss += total_loss.item()
            epoch_rel_l2 += rel_l2.item()
            num_batches += 1

        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{epochs:03d} | Loss: {epoch_loss/num_batches:.6f} | Rel L2 Error: {epoch_rel_l2/num_batches:.6f}")

    total_training_time = time.time() - start_time
    print(f"--> Addestramento completato in {total_training_time:.2f} secondi.")

    # Valutazione Finale su Test Set
    model.eval()
    with torch.no_grad():
        x_test, f_test, u_test, mask_test = x_test.to(device), f_test.to(device), u_test.to(device), mask_test.to(device)
        u_test_pred = model(x_test, f_test, mask_test)

        test_diff_norms = torch.norm((u_test_pred - u_test) * mask_test, p=2, dim=1)
        test_ref_norms = torch.norm(u_test * mask_test, p=2, dim=1) + 1e-8
        final_test_rel_l2 = torch.mean(test_diff_norms / test_ref_norms).item()

    print(f"--> [RISULTATO FINALE] Test Relative L2 Error: {final_test_rel_l2:.6f}")

    # Salvataggio Pesi del Modello
    torch.save(model.state_dict(), "dif_fno_weights.pt")
    print("--> Modello salvato con successo in 'dif_fno_weights.pt'.")

if __name__ == "__main__":
    train()
