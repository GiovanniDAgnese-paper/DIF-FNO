import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DiffeomorphicMap(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=64, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x):
        return x + 0.1 * self.net(x)

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

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class DIFFNO2d(nn.Module):
    def __init__(self, modes1=12, modes2=12, width=48, canonical_res=64):
        super().__init__()
        self.diff_map = DiffeomorphicMap()
        self.fc0 = nn.Linear(3, width)
        self.conv0 = SpectralConv2d(width, width, modes1, modes2)
        self.conv1 = SpectralConv2d(width, width, modes1, modes2)
        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)
        self.res = canonical_res

    def forward(self, x_coords, f_in, mask):
        B, N, _ = x_coords.shape
        xi = self.diff_map(x_coords)
        h = torch.cat([xi, f_in], dim=-1)
        h = self.fc0(h)
        
        H = W = int(np.sqrt(N))
        h = h.permute(0, 2, 1).reshape(B, -1, H, W)
        
        x1 = self.conv0(h) + self.w0(h)
        x1 = F.gelu(x1)
        x2 = self.conv1(x1) + self.w1(x1)
        x2 = F.gelu(x2)
        
        x2 = x2.reshape(B, -1, H*W).permute(0, 2, 1)
        out = self.fc2(F.gelu(self.fc1(x2)))
        return out.squeeze(-1)

print("Architettura aggiornata con width dinamico.")
