import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

DATA_PATH = "/Users/michal/Desktop/PhD/dvl paper/A-KIT-main/Data"
SAVE_PATH = "/Users/michal/Desktop/PhD/dvl paper/DATA/dvl_dataset"
N_TRAJECTORIES = 13
SMOOTHING = 0.5


# ── 1. Load raw signals ────────────────────────────────────────────────────────

def load_signals():
    signals = []
    times = []
    for i in range(1, N_TRAJECTORIES + 1):
        path = f"{DATA_PATH}/Trajectory{i}/DVL_trajectory{i}.csv"
        data = pd.read_csv(path)
        time = data.iloc[:, 0].values
        vx   = data.iloc[:, 1].values
        vy   = data.iloc[:, 2].values
        vz   = data.iloc[:, 3].values
        signals.append(np.stack([vx, vy, vz], axis=0))  # (3, N)
        times.append(time)
    return signals, times


# ── 2. Compute curvature maps ──────────────────────────────────────────────────

# Extracts the spline curvature of the signal to use as a conditioning input for the diffusion model.
# The full signal spline (per axis) serves as the condition that guides the denoising process.
def compute_curvature(signal_3axis, time):
    curvature = np.zeros_like(signal_3axis)
    for axis in range(3):
        s   = signal_3axis[axis]
        spl = UnivariateSpline(time, s, s=SMOOTHING)
        d1  = spl.derivative(n=1)(time)
        d2  = spl.derivative(n=2)(time)
        curvature[axis] = d2 / (1 + d1**2)**1.5
    return curvature  # (3, N)


# ── 3. Compute per-trajectory mean and std ────────────────────────────────────

# Per-trajectory mean and std per axis — used as scalar conditions for the diffusion model.
def compute_stats(signal_3axis):
    mean = signal_3axis.mean(axis=1)  # (3,)
    std  = signal_3axis.std(axis=1)   # (3,)
    return mean, std


# ── 4. Normalize ──────────────────────────────────────────────────────────────

def normalize(signal_3axis, mean, std):
    return (signal_3axis - mean[:, None]) / std[:, None]


# ── 4. PyTorch Dataset ─────────────────────────────────────────────────────────

class DVLDataset(Dataset):
    def __init__(self, signals, curvatures, means, stds):
        self.samples = list(zip(signals, curvatures, means, stds))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        signal, curvature, mean, std = self.samples[idx]
        return (
            torch.tensor(signal,    dtype=torch.float32),  # (3, N)
            torch.tensor(curvature, dtype=torch.float32),  # (3, N)
            torch.tensor(mean,      dtype=torch.float32),  # (3,)
            torch.tensor(std,       dtype=torch.float32),  # (3,)
        )


# ── 5. Build pipeline ──────────────────────────────────────────────────────────

def build_pipeline(batch_size=4, save_path=None):
    print("Loading signals...")
    signals, times = load_signals()
    print(f"  {len(signals)} trajectories, shape {signals[0].shape}")

    print("Computing curvature maps...")
    curvatures = [compute_curvature(s, t) for s, t in zip(signals, times)]

    print("Computing per-trajectory stats...")
    means, stds = zip(*[compute_stats(s) for s in signals])

    print("Normalizing signals...")
    signals_norm = [normalize(s, m, sd) for s, m, sd in zip(signals, means, stds)]

    if save_path is not None:
        np.savez(
            save_path,
            signals=np.stack(signals_norm),  # (13, 3, N) — normalized
            curvatures=np.stack(curvatures), # (13, 3, N)
            means=np.stack(means),           # (13, 3)
            stds=np.stack(stds),             # (13, 3)
        )
        print(f"  Saved dataset to {save_path}.npz")

    dataset = DVLDataset(signals_norm, curvatures, means, stds)
    print(f"  Dataset size: {len(dataset)} samples")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("Done.")
    return dataloader


# ── 6. Sanity check ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dataloader = build_pipeline(batch_size=4, save_path=SAVE_PATH)

    signals, times = load_signals()
    curvatures = [compute_curvature(s, t) for s, t in zip(signals, times)]

    fig, axes = plt.subplots(2, 3, figsize=(14, 5))
    labels = ["vx", "vy", "vz"]
    for j, (sig, curv, t) in enumerate(zip(signals, curvatures, times)):
        for i in range(3):
            axes[0, i].plot(t, sig[i], alpha=0.6, label=f"Traj {j+1}")
            axes[1, i].plot(t, curv[i], alpha=0.6)
    for i in range(3):
        axes[0, i].set_title(f"Signal — {labels[i]}")
        axes[1, i].set_title(f"Curvature — {labels[i]}")
    axes[0, 2].legend(loc="upper right", fontsize=6)
    plt.tight_layout()
    plt.show()
