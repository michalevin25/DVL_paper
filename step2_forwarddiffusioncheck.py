import numpy as np
import matplotlib.pyplot as plt

DATASET_PATH = "/Users/michal/Desktop/PhD/dvl paper/DATA/dvl_dataset.npz"

SIGMA_MIN = 0.002
SIGMA_MAX = 80.0

data     = np.load(DATASET_PATH)
signals  = data["signals"]   # (W, 3, N) — window-normalized
traj_ids = data["traj_ids"]  # (W,)

print(f"signals:  {signals.shape}")
print(f"windows:  {len(signals)}")


# ── Forward diffusion (EDM): x_noisy = x0 + sigma * epsilon ──────────────────

def forward_diffusion(x0, sigma):
    eps = np.random.randn(*x0.shape)
    return x0 + sigma * eps


# ── Plot: one window, 6 sigma levels ─────────────────────────────────────────

np.random.seed(42)

win_idx     = np.where(traj_ids == 3)[0][0]   # first window of trajectory 3
x0          = signals[win_idx]                 # (3, N)
vel_labels  = ["vx", "vy", "vz"]

sigmas      = np.exp(np.linspace(np.log(SIGMA_MIN), np.log(SIGMA_MAX), 6))

fig, axes = plt.subplots(3, len(sigmas), figsize=(18, 7), sharex=True)

for col, sigma in enumerate(sigmas):
    x_noisy = forward_diffusion(x0, sigma)
    for row in range(3):
        axes[row, col].plot(x0[row],      color="red",       linewidth=1.0, label="clean")
        axes[row, col].plot(x_noisy[row], color="steelblue", linewidth=0.7, alpha=0.85, label=f"σ={sigma:.3f}")
        axes[row, col].grid(True, alpha=0.3)
        if row == 0:
            axes[row, col].set_title(f"σ = {sigma:.3f}", fontsize=9)
        if col == 0:
            axes[row, col].set_ylabel(vel_labels[row])
        if row == 0 and col == 0:
            axes[row, col].legend(fontsize=7)

fig.suptitle("Forward diffusion (EDM) — trajectory 3, window 0\nclean signal → pure noise as σ increases")
plt.tight_layout()
plt.show()
