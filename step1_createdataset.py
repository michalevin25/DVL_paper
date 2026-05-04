import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import UnivariateSpline
from scipy.stats import kurtosis as compute_kurtosis
import matplotlib.pyplot as plt

DATA_PATH      = "/Users/michal/Desktop/PhD/dvl paper/A-KIT-main/Data"
SAVE_PATH      = "/Users/michal/Desktop/PhD/dvl paper/DATA/dvl_dataset"
N_TRAJECTORIES = 13
SMOOTHING      = 0.5
WINDOW_SIZE    = 206
STRIDE         = 50
N_BINS         = 20   # bins in the spike histogram


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


# ── 2b. Compute spike histogram from curvature ───────────────────────────────

def compute_spike_histogram(curvature, n_bins=N_BINS):
    """
    Coarse temporal histogram of curvature spike amplitude.
    Each bin = max |curvature| over that time slice.
    Input:  (3, N)  Output: (3, n_bins)
    """
    _, N  = curvature.shape
    hist  = np.zeros((3, n_bins))
    edges = np.linspace(0, N, n_bins + 1, dtype=int)
    for b in range(n_bins):
        hist[:, b] = np.abs(curvature[:, edges[b]:edges[b + 1]]).max(axis=1)
    return hist


# ── 3. Compute per-trajectory mean and std ────────────────────────────────────

# Per-trajectory mean and std per axis — used as scalar conditions for the diffusion model.
def compute_stats(signal_3axis):
    mean = signal_3axis.mean(axis=1)  # (3,)
    std  = signal_3axis.std(axis=1)   # (3,)
    return mean, std


# ── 4. Normalize ──────────────────────────────────────────────────────────────

def normalize(signal_3axis, mean, std):
    return (signal_3axis - mean[:, None]) / std[:, None]


# ── 5. Windowing ──────────────────────────────────────────────────────────────

def create_windows(signals, curvatures):
    win_signals     = []
    win_spike_hists = []
    win_means       = []
    win_stds        = []
    win_kurtoses    = []
    win_traj_ids    = []
    for traj_idx, (sig, curv) in enumerate(zip(signals, curvatures)):
        N = sig.shape[1]
        for start in range(0, N - WINDOW_SIZE + 1, STRIDE):
            end           = start + WINDOW_SIZE
            w_sig         = sig[:, start:end]                              # (3, WINDOW_SIZE)
            w_curv        = curv[:, start:end]                             # (3, WINDOW_SIZE)
            w_spike_hist  = compute_spike_histogram(w_curv)                # (3, N_BINS)
            w_mean        = w_sig.mean(axis=1)                             # (3,)
            w_std         = w_sig.std(axis=1)                              # (3,)
            w_kurt        = compute_kurtosis(w_sig, axis=1, fisher=True)   # (3,)
            win_signals.append(w_sig)
            win_spike_hists.append(w_spike_hist)
            win_means.append(w_mean)
            win_stds.append(w_std)
            win_kurtoses.append(w_kurt)
            win_traj_ids.append(traj_idx + 1)
    return win_signals, win_spike_hists, win_means, win_stds, win_kurtoses, win_traj_ids


# ── 4. PyTorch Dataset ─────────────────────────────────────────────────────────

class DVLDataset(Dataset):
    def __init__(self, signals, spike_hists, means, stds, kurtoses):
        self.samples = list(zip(signals, spike_hists, means, stds, kurtoses))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        signal, spike_hist, mean, std, kurt = self.samples[idx]
        return (
            torch.tensor(signal,     dtype=torch.float32),  # (3, N)
            torch.tensor(spike_hist, dtype=torch.float32),  # (3, N_BINS)
            torch.tensor(mean,       dtype=torch.float32),  # (3,)
            torch.tensor(std,        dtype=torch.float32),  # (3,)
            torch.tensor(kurt,       dtype=torch.float32),  # (3,)
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

    print("Creating windows...")
    win_signals, win_spike_hists, win_means, win_stds, win_kurtoses, win_traj_ids = create_windows(signals_norm, curvatures)
    print(f"  {len(win_signals)} windows (window={WINDOW_SIZE}, stride={STRIDE}, bins={N_BINS})")

    if save_path is not None:
        np.savez(
            save_path,
            signals=np.stack(win_signals),            # (W, 3, WINDOW_SIZE)
            spike_hists=np.stack(win_spike_hists),    # (W, 3, N_BINS)
            means=np.stack(win_means),                # (W, 3)
            stds=np.stack(win_stds),                  # (W, 3)
            kurtoses=np.stack(win_kurtoses),          # (W, 3)
            traj_ids=np.array(win_traj_ids),          # (W,) trajectory number 1–13
        )
        print(f"  Saved dataset to {save_path}.npz")

    dataset = DVLDataset(win_signals, win_spike_hists, win_means, win_stds, win_kurtoses)
    print(f"  Dataset size: {len(dataset)} windows")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("Done.")
    return dataloader


# ── 6. Sanity check ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dataloader = build_pipeline(batch_size=4, save_path=SAVE_PATH)

    signals, times = load_signals()
    curvatures = [compute_curvature(s, t) for s, t in zip(signals, times)]

    # plot signals and their spike histograms for the first trajectory
    fig, axes = plt.subplots(2, 3, figsize=(14, 5))
    labels = ["vx", "vy", "vz"]
    sig0  = signals[0]
    curv0 = curvatures[0]
    hist0 = compute_spike_histogram(curv0)
    bin_centers = np.linspace(0, sig0.shape[1], N_BINS)
    for i in range(3):
        axes[0, i].plot(times[0], sig0[i], color="steelblue", linewidth=0.8)
        axes[0, i].set_title(f"Signal — {labels[i]}")
        axes[1, i].bar(bin_centers, hist0[i], width=sig0.shape[1] / N_BINS * 0.8,
                       color="darkorange", alpha=0.8)
        axes[1, i].set_title(f"Spike histogram — {labels[i]}")
    plt.tight_layout()
    plt.show()
