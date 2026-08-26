import torch
import torch.nn as nn
import torch.nn.functional as F

def ensure_channel_last(x: torch.Tensor, expected_channels: int = 1) -> torch.Tensor:
    """
    Normalizza la forma del tensore in (B, H, W, C).
    Accetta (B, H, W), (B, C, H, W) e (B, H, W, C).
    """
    if x.ndim == 3:
        return x.unsqueeze(-1)
    elif x.ndim == 4:
        if x.shape[1] == expected_channels and x.shape[-1] != expected_channels:
            return x.permute(0, 2, 3, 1)
        elif x.shape[-1] == expected_channels:
            return x
        elif x.shape[1] < x.shape[2] and x.shape[1] < x.shape[3]:
            return x.permute(0, 2, 3, 1)
    return x

class SpectralConv2d(nn.Module):
    """
    Layer Spettrale 2D FNO con parametri complessi e trasformata di Fourier.
    """
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )

    def _compl_mul2d(self, input_tensor: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", input_tensor, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)
        batchsize = x.shape[0]
        size_h = x.shape[2]
        size_w = x.shape[3]

        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(2), x.size(3) // 2 + 1,
            device=x.device, dtype=torch.cfloat
        )

        out_ft[:, :, :self.modes1, :self.modes2] = self._compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = self._compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )

        x_out = torch.fft.irfft2(out_ft, s=(size_h, size_w))
        return x_out.permute(0, 2, 3, 1)


class DiffeomorphicMapping(nn.Module):
    """
    Modulo di Mappatura Diffeomorfa phi: Omega_l -> Omega_p.
    Calcola lo Jacobiano J e impone la Barrier Loss per det(J) > eps > 0.
    """
    def __init__(self, hidden_dim: int = 64, eps_barrier: float = 1e-3):
        super().__init__()
        self.eps_barrier = eps_barrier
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2)
        )

    def get_grid(self, batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, height, device=device),
            torch.linspace(0, 1, width, device=device),
            indexing="ij"
        )
        grid = torch.stack([grid_x, grid_y], dim=-1)
        return grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)

    def compute_jacobian_and_barrier(self, grid_latent: torch.Tensor):
        grid_req = grid_latent.detach().clone().requires_grad_(True)
        mapped = grid_req + self.net(grid_req)

        with torch.enable_grad():
            u_pos = mapped[..., 0]
            v_pos = mapped[..., 1]

            grad_u = torch.autograd.grad(u_pos.sum(), grid_req, create_graph=True, retain_graph=True)[0]
            grad_v = torch.autograd.grad(v_pos.sum(), grid_req, create_graph=True, retain_graph=True)[0]

            du_dxi, du_deta = grad_u[..., 0], grad_u[..., 1]
            dv_dxi, dv_deta = grad_v[..., 0], grad_v[..., 1]

            det_J = du_dxi * dv_deta - du_deta * dv_dxi

        barrier_loss = torch.mean(F.relu(self.eps_barrier - det_J) ** 2)
        return mapped, det_J, barrier_loss

    def forward(self, grid_latent: torch.Tensor):
        offset = self.net(grid_latent)
        mapped_grid = grid_latent + offset
        return mapped_grid


class DIFFNO2d(nn.Module):
    """
    Diffeomorphic Fourier Neural Operator 2D.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1, width: int = 64, modes1: int = 12, modes2: int = 12):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width

        self.diffeo = DiffeomorphicMapping(hidden_dim=64)
        self.fc0 = nn.Linear(in_channels + 2, self.width)

        self.conv0 = SpectralConv2d(self.width, self.width, modes1, modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, modes1, modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, modes1, modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, modes1, modes2)

        self.w0 = nn.Linear(self.width, self.width)
        self.w1 = nn.Linear(self.width, self.width)
        self.w2 = nn.Linear(self.width, self.width)
        self.w3 = nn.Linear(self.width, self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x: torch.Tensor):
        x_in = ensure_channel_last(x, self.in_channels)
        B, H, W, _ = x_in.shape

        latent_grid = self.diffeo.get_grid(B, H, W, x_in.device)
        mapped_grid = self.diffeo(latent_grid)

        x_cat = torch.cat([x_in, mapped_grid], dim=-1)
        x_proj = self.fc0(x_cat)

        x1 = self.conv0(x_proj) + self.w0(x_proj)
        x1 = F.gelu(x1)

        x2 = self.conv1(x1) + self.w1(x1)
        x2 = F.gelu(x2)

        x3 = self.conv2(x2) + self.w2(x2)
        x3 = F.gelu(x3)

        x4 = self.conv3(x3) + self.w3(x3)
        x4 = F.gelu(x4)

        out = self.fc1(x4)
        out = F.gelu(out)
        out = self.fc2(out)

        return ensure_channel_last(out, self.out_channels)


class GeoFNO2d(nn.Module):
    """
    Geo-FNO Baseline Model.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1, width: int = 64, modes1: int = 12, modes2: int = 12):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width

        self.fc0 = nn.Linear(in_channels + 2, self.width)
        self.conv0 = SpectralConv2d(self.width, self.width, modes1, modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, modes1, modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, modes1, modes2)
        self.w0 = nn.Linear(self.width, self.width)
        self.w1 = nn.Linear(self.width, self.width)
        self.w2 = nn.Linear(self.width, self.width)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x: torch.Tensor, phys_grid: torch.Tensor = None):
        x_in = ensure_channel_last(x, self.in_channels)
        B, H, W, _ = x_in.shape
        if phys_grid is None:
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, 1, H, device=x_in.device),
                torch.linspace(0, 1, W, device=x_in.device),
                indexing="ij"
            )
            phys_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)

        phys_grid = ensure_channel_last(phys_grid, expected_channels=2)
        x_cat = torch.cat([x_in, phys_grid], dim=-1)
        x_proj = self.fc0(x_cat)

        x1 = F.gelu(self.conv0(x_proj) + self.w0(x_proj))
        x2 = F.gelu(self.conv1(x1) + self.w1(x1))
        x3 = F.gelu(self.conv2(x2) + self.w2(x2))

        out = F.gelu(self.fc1(x3))
        out = self.fc2(out)
        return ensure_channel_last(out, self.out_channels)


class FNOMask2d(nn.Module):
    """
    Masked FNO Baseline Model.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1, width: int = 64, modes1: int = 12, modes2: int = 12):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width

        self.fc0 = nn.Linear(in_channels + 3, self.width)
        self.conv0 = SpectralConv2d(self.width, self.width, modes1, modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, modes1, modes2)
        self.w0 = nn.Linear(self.width, self.width)
        self.w1 = nn.Linear(self.width, self.width)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        x_in = ensure_channel_last(x, self.in_channels)
        B, H, W, _ = x_in.shape

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, H, device=x_in.device),
            torch.linspace(0, 1, W, device=x_in.device),
            indexing="ij"
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)

        if mask is None:
            mask = torch.ones((B, H, W, 1), device=x_in.device)
        else:
            mask = ensure_channel_last(mask, expected_channels=1)

        x_cat = torch.cat([x_in, grid, mask], dim=-1)
        x_proj = self.fc0(x_cat)

        x1 = F.gelu(self.conv0(x_proj) + self.w0(x_proj))
        x2 = F.gelu(self.conv1(x1) + self.w1(x1))

        out = F.gelu(self.fc1(x2))
        out = self.fc2(out)
        return ensure_channel_last(out, self.out_channels)
