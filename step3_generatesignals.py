import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

DATASET_PATH = "/Users/michal/Desktop/PhD/dvl paper/DATA/dvl_dataset.npz"


# ── Load dataset ──────────────────────────────────────────────────────────────

class DVLDataset(Dataset):
    def __init__(self, path):
        data           = np.load(path)
        self.signals    = torch.tensor(data["signals"],    dtype=torch.float32)  # (13, 3, N)
        self.curvatures = torch.tensor(data["curvatures"], dtype=torch.float32)  # (13, 3, N)
        self.means      = torch.tensor(data["means"],      dtype=torch.float32)  # (13, 3)
        self.stds       = torch.tensor(data["stds"],       dtype=torch.float32)  # (13, 3)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return self.signals[idx], self.curvatures[idx], self.means[idx], self.stds[idx]


# ── EDM preconditioning constants ─────────────────────────────────────────────

SIGMA_MIN  = 0.002
SIGMA_MAX  = 80.0
SIGMA_DATA = 0.5


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
    Input channels: 3 (signal) + 3 (curvature) + 3 (mean) + 3 (std) = 12
    Output channels: 3 (predicted x_0)
    """
    def __init__(self, in_channels=12, out_channels=3, base_channels=64, embed_dim=64):
        super().__init__()
        C = base_channels  # 64

        self.sigma_emb = SigmaEmbedding(embed_dim)

        # encoder
        self.enc1 = ResBlock1D(in_channels, C,    embed_dim)   # (B, 64,  400)
        self.enc2 = ResBlock1D(C,           C*2,  embed_dim)   # (B, 128, 200)
        self.enc3 = ResBlock1D(C*2,         C*4,  embed_dim)   # (B, 256, 100)

        self.down = nn.AvgPool1d(kernel_size=2, stride=2)

        # bottleneck
        self.bottleneck = ResBlock1D(C*4, C*4, embed_dim)      # (B, 256, 50)

        # decoder
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

        self.dec3 = ResBlock1D(C*4 + C*4, C*4, embed_dim)     # (B, 256, 100)
        self.dec2 = ResBlock1D(C*4 + C*2, C*2, embed_dim)     # (B, 128, 200)
        self.dec1 = ResBlock1D(C*2 + C,   C,   embed_dim)     # (B, 64,  400)

        self.out_conv = nn.Conv1d(C, out_channels, kernel_size=1)

    def forward(self, x, sigma, curvature, mean, std):
        # x:         (B, 3, 400)
        # sigma:     (B,)
        # curvature: (B, 3, 400)
        # mean:      (B, 3)
        # std:       (B, 3)

        B, _, L = x.shape

        # expand mean and std to (B, 3, 400) and concatenate all conditions
        mean_exp = mean.unsqueeze(-1).expand(B, 3, L)
        std_exp  = std.unsqueeze(-1).expand(B, 3, L)
        inp = torch.cat([x, curvature, mean_exp, std_exp], dim=1)  # (B, 12, 400)

        # apply cin preconditioning
        inp = c_in(sigma).view(B, 1, 1) * inp

        sigma_emb = self.sigma_emb(sigma)  # (B, 64)

        # encoder
        e1 = self.enc1(inp,          sigma_emb)  # (B, 64,  400)
        e2 = self.enc2(self.down(e1), sigma_emb)  # (B, 128, 200)
        e3 = self.enc3(self.down(e2), sigma_emb)  # (B, 256, 100)

        # bottleneck
        b = self.bottleneck(self.down(e3), sigma_emb)  # (B, 256, 50)

        # decoder with skip connections
        d3 = self.dec3(torch.cat([self.up(b),  e3], dim=1), sigma_emb)  # (B, 256, 100)
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1), sigma_emb)  # (B, 128, 200)
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1), sigma_emb)  # (B, 64,  400)

        return self.out_conv(d1)  # (B, 3, 400)


# ── EDM wrapper: applies preconditioning around Gϕ ───────────────────────────

class EDMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = UNet1D()

    def forward(self, x_noisy, sigma, curvature, mean, std):
        # x_noisy: (B, 3, 400) — z = x0 + ε
        # returns x̂: (B, 3, 400) — denoised estimate of x0

        skip  = c_skip(sigma).view(-1, 1, 1) * x_noisy
        scale = c_out(sigma).view(-1, 1, 1)

        net_out = self.net(x_noisy, sigma, curvature, mean, std)

        return skip + scale * net_out  # x̂ = cskip·z + cout·Gϕ(cin·z, cnoise, c)


# ── Training loop ────────────────────────────────────────────────────────────

def sample_sigma(batch_size, P_mean=-1.2, P_std=1.2):
    # sample σ from log-normal distribution as in EDM paper
    log_sigma = torch.randn(batch_size) * P_std + P_mean
    return torch.exp(log_sigma)

def train(epochs=10000, batch_size=4, lr=3e-5):
    dataset    = DVLDataset(DATASET_PATH)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model     = EDMModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Training on {len(dataset)} trajectories for {epochs} epochs")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    losses = []
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for signals, curvatures, means, stds in dataloader:
            B = len(signals)

            # sample σ and noise
            sigma   = sample_sigma(B)                          # (B,)
            epsilon = torch.randn_like(signals) * sigma.view(B, 1, 1)
            z       = signals + epsilon                        # z = x + ε

            # compute target: (x0 - cskip·z) / cout
            cs  = c_skip(sigma).view(B, 1, 1)
            co  = c_out(sigma).view(B, 1, 1)
            target = (signals - cs * z) / co                  # (B, 3, 400)

            # forward pass
            x_hat = model(z, sigma, curvatures, means, stds)  # (B, 3, 400)

            # weighted loss
            w    = loss_weight(sigma).view(B, 1, 1)
            loss = (w * (x_hat - target) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)

        if epoch % 500 == 0:
            print(f"Epoch {epoch:>6} / {epochs} — loss: {avg_loss:.6f}")

    return model, losses


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    dataset    = DVLDataset(DATASET_PATH)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    print(f"Dataset size: {len(dataset)} trajectories")

    signals, curvatures, means, stds = next(iter(dataloader))
    print(f"signals:    {signals.shape}")
    print(f"curvatures: {curvatures.shape}")
    print(f"means:      {means.shape}")
    print(f"stds:       {stds.shape}")

    model = EDMModel()
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # simulate one forward pass with random σ
    sigma   = torch.rand(len(signals)) * (SIGMA_MAX - SIGMA_MIN) + SIGMA_MIN
    epsilon = torch.randn_like(signals)
    z       = signals + sigma.view(-1, 1, 1) * epsilon   # EDM forward: z = x + ε

    x_hat = model(z, sigma, curvatures, means, stds)
    print(f"Input shape:  {z.shape}")
    print(f"Output shape: {x_hat.shape}")   # should be (4, 3, 400)

    # plot clean signal, noisy input, and untrained model output for first sample
    import matplotlib.pyplot as plt
    labels = ["vx", "vy", "vz"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    for i in range(3):
        axes[i].plot(signals[0, i].numpy(),          label="clean",         color="red",       linewidth=1.2)
        axes[i].plot(z[0, i].detach().numpy(),       label=f"noisy σ={sigma[0]:.2f}", color="steelblue", linewidth=0.8)
        axes[i].plot(x_hat[0, i].detach().numpy(),   label="model output (untrained)", color="green",     linewidth=0.8)
        axes[i].set_ylabel(labels[i])
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.3)
    fig.suptitle("Sanity check — clean vs noisy vs untrained model output")
    plt.tight_layout()
    plt.show()

    # ── Train ─────────────────────────────────────────────────────────────────
    model, losses = train()

    # loss curve
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(losses)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss")
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()
