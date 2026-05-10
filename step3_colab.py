# ── Google Colab setup ────────────────────────────────────────────────────────
# Run this file in a Colab notebook cell:  !python step3_colab.py
# Or paste the cells into a notebook manually.
#
# Before running: upload dvl_dataset.npz to your Google Drive under:
#   My Drive / PhD / dvl paper / dvl_dataset.npz
# Adjust DRIVE_ROOT below if you put it somewhere else.

from google.colab import drive
drive.mount("/content/drive")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

# ── Paths ─────────────────────────────────────────────────────────────────────

DRIVE_ROOT   = "/content/drive/MyDrive/PhD/dvl paper"
DATASET_PATH = f"{DRIVE_ROOT}/dvl_dataset.npz"

# ── Device ────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ── Load dataset ──────────────────────────────────────────────────────────────

class DVLDataset(Dataset):
    def __init__(self, path):
        data             = np.load(path)
        self.signals     = torch.tensor(data["signals"],   dtype=torch.float32)  # (W, 3, N)
        self.peak_maps   = torch.tensor(data["peak_maps"], dtype=torch.float32)  # (W, 3, N)
        self.means       = torch.tensor(data["means"],     dtype=torch.float32)  # (W, 3)
        self.stds        = torch.tensor(data["stds"],      dtype=torch.float32)  # (W, 3)
        self.kurtoses    = torch.tensor(data["kurtoses"],  dtype=torch.float32)  # (W, 3)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return self.signals[idx], self.peak_maps[idx], self.means[idx], self.stds[idx], self.kurtoses[idx]


# ── EDM preconditioning constants ─────────────────────────────────────────────

SIGMA_MIN  = 0.002
SIGMA_MAX  = 80.0
SIGMA_DATA = 1.0  # signals are normalized to unit variance per window


def c_skip(sigma):
    return SIGMA_DATA**2 / (sigma**2 + SIGMA_DATA**2)

def c_out(sigma):
    return sigma * SIGMA_DATA / (sigma**2 + SIGMA_DATA**2)**0.5

def c_in(sigma):
    return 1.0 / (sigma**2 + SIGMA_DATA**2)**0.5

def c_noise(sigma):
    return torch.log(sigma) / 4.0

def loss_weight(sigma):
    return (sigma**2 + SIGMA_DATA**2) / (sigma * SIGMA_DATA)**2


# ── σ embedding: scalar → 64-dim via small MLP ────────────────────────────────

class SigmaEmbedding(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, sigma):
        # sigma: (B,) → (B, embed_dim)
        noise = c_noise(sigma).unsqueeze(-1)   # (B, 1)
        return self.mlp(noise)                 # (B, embed_dim)


# ── 1D ResBlock ────────────────────────────────────────────────────────────────

class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim=64):
        super().__init__()
        self.conv1  = nn.Conv1d(in_channels,  out_channels, kernel_size=3, padding=1)
        self.conv2  = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1  = nn.GroupNorm(8, out_channels)
        self.norm2  = nn.GroupNorm(8, out_channels)
        self.act    = nn.SiLU()
        self.sigma_proj = nn.Linear(embed_dim, out_channels)
        # 1x1 conv to match channels for residual if needed
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, sigma_emb):
        # x: (B, in_channels, L)
        # sigma_emb: (B, embed_dim)
        h = self.act(self.norm1(self.conv1(x)))
        h = h + self.sigma_proj(sigma_emb).unsqueeze(-1)  # inject σ
        h = self.act(self.norm2(self.conv2(h)))
        return h + self.residual_conv(x)


# ── 1D U-Net ──────────────────────────────────────────────────────────────────

class UNet1D(nn.Module):
    """
    Input channels: 3 (signal) + 3 (peak_map) + 3 (mean) + 3 (std) + 3 (kurtosis) = 15
    Output channels: 3 (predicted x_0)
    Peak map is also injected additively at each encoder scale for temporal control.
    """
    def __init__(self, in_channels=15, out_channels=3, base_channels=64, embed_dim=64):
        super().__init__()
        C = base_channels  # 64

        self.sigma_emb = SigmaEmbedding(embed_dim)

        # encoder
        self.enc1 = ResBlock1D(in_channels, C,    embed_dim)
        self.enc2 = ResBlock1D(C,           C*2,  embed_dim)
        self.enc3 = ResBlock1D(C*2,         C*4,  embed_dim)

        self.down = nn.AvgPool1d(kernel_size=2, stride=2)

        # bottleneck
        self.bottleneck = ResBlock1D(C*4, C*4, embed_dim)

        # multi-scale peak map projections — re-inject peak_map at each encoder scale
        # so temporal position info survives U-Net downsampling
        self.peak_proj1 = nn.Conv1d(3, C,   kernel_size=1)
        self.peak_proj2 = nn.Conv1d(3, C*2, kernel_size=1)
        self.peak_proj3 = nn.Conv1d(3, C*4, kernel_size=1)
        self.peak_projb = nn.Conv1d(3, C*4, kernel_size=1)

        # decoder — upsample to match skip connection size exactly
        self.dec3 = ResBlock1D(C*4 + C*4, C*4, embed_dim)
        self.dec2 = ResBlock1D(C*4 + C*2, C*2, embed_dim)
        self.dec1 = ResBlock1D(C*2 + C,   C,   embed_dim)

        self.out_conv = nn.Conv1d(C, out_channels, kernel_size=1)

    def forward(self, x, sigma, peak_map, mean, std, kurtosis):
        B, _, L = x.shape

        x_scaled = c_in(sigma).view(B, 1, 1) * x

        mean_exp = mean.unsqueeze(-1).expand(B, 3, L)
        std_exp  = std.unsqueeze(-1).expand(B, 3, L)
        kurt_exp = kurtosis.unsqueeze(-1).expand(B, 3, L)
        inp = torch.cat([x_scaled, peak_map, mean_exp, std_exp, kurt_exp], dim=1)  # (B, 15, L)

        sigma_emb = self.sigma_emb(sigma)  # (B, 64)

        # encoder — inject peak_map additively at each level (re-downsampled to match)
        e1 = self.enc1(inp, sigma_emb)
        e1 = e1 + self.peak_proj1(peak_map)

        e2 = self.enc2(self.down(e1), sigma_emb)
        e2 = e2 + self.peak_proj2(F.interpolate(peak_map, size=e2.shape[-1], mode='linear', align_corners=False))

        e3 = self.enc3(self.down(e2), sigma_emb)
        e3 = e3 + self.peak_proj3(F.interpolate(peak_map, size=e3.shape[-1], mode='linear', align_corners=False))

        b = self.bottleneck(self.down(e3), sigma_emb)
        b = b + self.peak_projb(F.interpolate(peak_map, size=b.shape[-1], mode='linear', align_corners=False))

        # decoder with skip connections
        d3 = self.dec3(torch.cat([F.interpolate(b,  size=e3.shape[-1], mode='nearest'), e3], dim=1), sigma_emb)
        d2 = self.dec2(torch.cat([F.interpolate(d3, size=e2.shape[-1], mode='nearest'), e2], dim=1), sigma_emb)
        d1 = self.dec1(torch.cat([F.interpolate(d2, size=e1.shape[-1], mode='nearest'), e1], dim=1), sigma_emb)

        return self.out_conv(d1)


# ── EDM wrapper: applies preconditioning around Gϕ ───────────────────────────

class EDMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = UNet1D()

    def forward(self, x_noisy, sigma, peak_map, mean, std, kurtosis):
        # x_noisy: (B, 3, L) — z = x0 + ε
        # returns x̂: (B, 3, L) — denoised estimate of x0

        skip  = c_skip(sigma).view(-1, 1, 1) * x_noisy
        scale = c_out(sigma).view(-1, 1, 1)

        net_out = self.net(x_noisy, sigma, peak_map, mean, std, kurtosis)

        return skip + scale * net_out  # x̂ = cskip·z + cout·Gϕ(cin·z, cnoise, c)


# ── Training loop ────────────────────────────────────────────────────────────

P_UNCOND = 0.15   # fraction of samples trained unconditionally (CFG dropout)

def sample_sigma(batch_size, device, P_mean=-1.2, P_std=1.2):
    # sample σ from log-normal distribution as in EDM paper
    log_sigma = torch.randn(batch_size, device=device) * P_std + P_mean
    return torch.exp(log_sigma)

def train(epochs=15000, batch_size=32, lr=1e-4):
    dataset    = DVLDataset(DATASET_PATH)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model     = EDMModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Training on {len(dataset)} windows for {epochs} epochs")
    print(f"Parameters: {n_params:,}")
    print(f"Batch size: {batch_size}  ({len(dataloader)} batches/epoch)\n")

    # save training config to a timestamped log file
    run_time   = datetime.now()
    timestamp  = run_time.strftime('%Y%m%d_%H%M%S')
    log_path   = f"{DRIVE_ROOT}/training_log_{timestamp}.txt"
    model_path = f"{DRIVE_ROOT}/edm_model_{timestamp}.pt"
    with open(log_path, "w") as f:
        f.write(f"Training run: {run_time.strftime('%Y-%m-%d  %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")
        f.write("[EDM constants]\n")
        f.write(f"  SIGMA_MIN   = {SIGMA_MIN}\n")
        f.write(f"  SIGMA_MAX   = {SIGMA_MAX}\n")
        f.write(f"  SIGMA_DATA  = {SIGMA_DATA}\n\n")
        f.write("[Noise schedule sampling]\n")
        f.write(f"  P_mean      = -1.2\n")
        f.write(f"  P_std       =  1.2\n\n")
        f.write("[Model architecture]\n")
        f.write(f"  in_channels   = 15  (3 signal + 3 peak_map + 3 mean + 3 std + 3 kurtosis)\n")
        f.write(f"  out_channels  = 3\n")
        f.write(f"  base_channels = 64\n")
        f.write(f"  embed_dim     = 64\n")
        f.write(f"  peak_map      = multi-scale injection (enc1/enc2/enc3/bottleneck)\n")
        f.write(f"  total params  = {n_params:,}\n\n")
        f.write("[Training]\n")
        f.write(f"  epochs        = {epochs}\n")
        f.write(f"  batch_size    = {batch_size}\n")
        f.write(f"  lr            = {lr}\n")
        f.write(f"  optimizer     = Adam\n")
        f.write(f"  grad_clip     = 1.0  (clip_grad_norm)\n")
        f.write(f"  loss_weight   = clamped at max 10\n")
        f.write(f"  cfg_dropout   = {P_UNCOND}  (per-sample condition dropout)\n")
        f.write(f"  device        = {device}\n\n")
        f.write("[Dataset]\n")
        f.write(f"  path          = {DATASET_PATH}\n")
        f.write(f"  n_windows     = {len(dataset)}\n")
        f.write(f"  window_size   = {dataset.signals.shape[-1]}\n")
        f.write(f"  conditions    = peak_map (K=3 peaks, sigma=10), mean, std, kurtosis\n\n")
        f.write(f"[Files]\n")
        f.write(f"  model         = {model_path}\n")
        f.write(f"  log           = {log_path}\n")
    print(f"Training config saved to {log_path}\n")

    with open(log_path, "a") as f:
        f.write("[Loss curve]\n")

    losses = []
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for signals, peak_maps, means, stds, kurtoses in dataloader:
            signals   = signals.to(device)
            peak_maps = peak_maps.to(device)
            means     = means.to(device)
            stds      = stds.to(device)
            kurtoses  = kurtoses.to(device)
            B = len(signals)

            # sample σ and noise
            sigma   = sample_sigma(B, device)                                   # (B,)
            epsilon = torch.randn_like(signals) * sigma.view(B, 1, 1)
            z       = signals + epsilon                                         # z = x + ε

            # compute target: (x0 - cskip·z) / cout
            cs     = c_skip(sigma).view(B, 1, 1)
            co     = c_out(sigma).view(B, 1, 1)
            target = (signals - cs * z) / co                                   # (B, 3, L)

            # CFG condition dropout: per sample, zero all conditions with prob P_UNCOND
            keep = (torch.rand(B, device=device) > P_UNCOND).float()
            peak_maps_in = peak_maps * keep.view(B, 1, 1)
            means_in     = means     * keep.view(B, 1)
            stds_in      = stds      * keep.view(B, 1)
            kurtoses_in  = kurtoses  * keep.view(B, 1)

            # forward pass
            x_hat = model(z, sigma, peak_maps_in, means_in, stds_in, kurtoses_in)  # (B, 3, L)

            # weighted loss — clamp weight to prevent extreme values at small σ
            w    = loss_weight(sigma).clamp(max=10.0).view(B, 1, 1)
            loss = (w * (x_hat - target) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)

        if epoch % 250 == 0:
            print(f"Epoch {epoch:>6} / {epochs} — loss: {avg_loss:.6f}")
            with open(log_path, "a") as f:
                f.write(f"  epoch {epoch:>6} / {epochs}  loss = {avg_loss:.6f}\n")

        if epoch % 500 == 0:
            torch.save(model.state_dict(), model_path)

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    end_time = datetime.now()
    with open(log_path, "a") as f:
        f.write("[Results]\n")
        f.write(f"  final loss    = {losses[-1]:.6f}\n")
        f.write(f"  min loss      = {min(losses):.6f}  (epoch {losses.index(min(losses)) + 1})\n")
        f.write(f"  training time = {str(end_time - run_time).split('.')[0]}\n")
        f.write(f"  finished at   = {end_time.strftime('%Y-%m-%d  %H:%M:%S')}\n")
    print(f"Results appended to {log_path}")

    return model, losses


if __name__ == "__main__":
    model, losses = train()
