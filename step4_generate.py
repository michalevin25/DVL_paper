import torch
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from step3_generatesignals import EDMModel, SIGMA_MIN, SIGMA_MAX, N_BINS

DATA_DIR     = "/Users/michal/Desktop/PhD/dvl paper/DATA"
DATASET_PATH = f"{DATA_DIR}/dvl_dataset.npz"


# ── Load latest model by timestamp ───────────────────────────────────────────

candidates = sorted(glob.glob(f"{DATA_DIR}/edm_model_*.pt"))
if not candidates:
    raise FileNotFoundError(f"No edm_model_*.pt found in {DATA_DIR}")
MODEL_PATH = candidates[-1]  # lexicographic sort = chronological for YYYYMMDD_HHMMSS
print(f"Loading model: {os.path.basename(MODEL_PATH)}")

model = EDMModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()
print("Model loaded.")


# ── EDM sampler (deterministic Euler-Heun) ────────────────────────────────────

def generate(spike_hist, mean, std, kurtosis, signal_length=206, n_steps=200, seed=None, return_trajectory=False):
    """
    Generate a signal conditioned on a spike histogram and scalar statistics.
    spike_hist:    (1, 3, N_BINS)  — maneuver timing and intensity
    mean:          (1, 3)
    std:           (1, 3)
    kurtosis:      (1, 3)
    signal_length: number of time samples to generate
    returns:       (3, signal_length)
    """
    if seed is not None:
        torch.manual_seed(seed)

    x      = torch.randn(1, 3, signal_length) * SIGMA_MAX
    sigmas = torch.exp(torch.linspace(np.log(SIGMA_MAX), np.log(SIGMA_MIN), n_steps + 1))

    snapshots = []

    with torch.no_grad():
        for i in range(n_steps):
            sigma_cur  = sigmas[i].expand(1)
            sigma_next = sigmas[i + 1].expand(1)
            dt         = (sigma_next - sigma_cur).view(1, 1, 1)

            x_denoised = model(x, sigma_cur, spike_hist, mean, std, kurtosis)
            d_cur      = (x - x_denoised) / sigma_cur.view(1, 1, 1)
            x_next     = x + dt * d_cur

            if i < n_steps - 1:
                x_denoised_next = model(x_next, sigma_next, spike_hist, mean, std, kurtosis)
                d_next          = (x_next - x_denoised_next) / sigma_next.view(1, 1, 1)
                x_next          = x + dt * (d_cur + d_next) / 2

            x = x_next

            if return_trajectory:
                snapshots.append((i, sigmas[i].item(), x.squeeze(0).clone()))

    if return_trajectory:
        return x.squeeze(0), snapshots
    return x.squeeze(0)


def make_hist(bin_indices, amplitudes, n_bins=N_BINS):
    """
    Build a (1, 3, N_BINS) spike histogram manually.
    bin_indices: list of bin positions (0–N_BINS-1) where spikes occur
    amplitudes:  corresponding amplitude values
    All three axes get the same histogram.
    """
    hist = torch.zeros(1, 3, n_bins)
    for b, a in zip(bin_indices, amplitudes):
        hist[0, :, b] = a
    return hist


def plot_3d_trajectory(signals_list, labels_list, colors_list, title, dt=1.0):
    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection='3d')
    for sig, label, color in zip(signals_list, labels_list, colors_list):
        v = sig.numpy() if isinstance(sig, torch.Tensor) else sig
        x = np.cumsum(v[0] * dt)
        y = np.cumsum(v[1] * dt)
        z = np.cumsum(v[2] * dt)
        ax.plot(x, y, z, label=label, color=color, linewidth=1.5)
        ax.scatter(x[0], y[0], z[0], color=color, s=40, zorder=5)
    ax.set_xlabel("X (integrated vx)")
    ax.set_ylabel("Y (integrated vy)")
    ax.set_zlabel("Z (integrated vz)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_denoising_steps(spike_hist, mean, std, kurtosis, signal_length=206, n_steps=200, seed=42):
    _, snapshots = generate(spike_hist, mean, std, kurtosis, signal_length, n_steps, seed, return_trajectory=True)
    indices  = np.linspace(0, len(snapshots) - 1, 6, dtype=int)
    selected = [snapshots[i] for i in indices]
    fig, axes = plt.subplots(3, 6, figsize=(20, 7), sharex=True)
    vel_labels = ["vx", "vy", "vz"]
    for col, (step, sigma, x) in enumerate(selected):
        for row in range(3):
            axes[row, col].plot(x[row].numpy(), linewidth=0.8, color="steelblue")
            axes[row, col].grid(True, alpha=0.3)
            if row == 0:
                axes[row, col].set_title(f"step {step}\nσ={sigma:.2f}", fontsize=8)
            if col == 0:
                axes[row, col].set_ylabel(vel_labels[row])
    fig.suptitle("Denoising process — pure noise → generated trajectory")
    plt.tight_layout()
    plt.show()


# ── Load dataset ──────────────────────────────────────────────────────────────

data        = np.load(DATASET_PATH)
spike_hists = torch.tensor(data["spike_hists"], dtype=torch.float32)  # (W, 3, N_BINS)
means       = torch.tensor(data["means"],       dtype=torch.float32)  # (W, 3)
stds        = torch.tensor(data["stds"],        dtype=torch.float32)  # (W, 3)
kurtoses    = torch.tensor(data["kurtoses"],    dtype=torch.float32)  # (W, 3)
signals     = torch.tensor(data["signals"],     dtype=torch.float32)  # (W, 3, N)
traj_ids    = data["traj_ids"]                                         # (W,)

N            = signals.shape[-1]   # signal length (206)
vel_labels   = ["vx", "vy", "vz"]
cond_idx     = 0
real_signal  = signals[cond_idx]
spike_hist_0 = spike_hists[cond_idx].unsqueeze(0)  # (1, 3, N_BINS)
mean_0       = means[cond_idx].unsqueeze(0)
std_0        = stds[cond_idx].unsqueeze(0)
kurt_0       = kurtoses[cond_idx].unsqueeze(0)


# ── Test 1: designed spike histograms — controlled maneuver scenarios ─────────
# Bins 0–19 map to the 206-sample window (~10 samples per bin).
# Bin k corresponds to samples k*10 – (k+1)*10.

mean_syn = means[0].unsqueeze(0)
std_syn  = stds[0].unsqueeze(0)
kurt_syn = kurtoses[0].unsqueeze(0)

hist_none  = make_hist([], [])                          # no maneuvers
hist_early = make_hist([3],     [1.0])                  # one maneuver early  (~sample 30)
hist_mid   = make_hist([10],    [1.0])                  # one maneuver middle (~sample 100)
hist_late  = make_hist([17],    [1.0])                  # one maneuver late   (~sample 170)
hist_two   = make_hist([4, 15], [1.0, 1.0])             # two maneuvers
hist_heavy = make_hist([4, 10, 15], [1.5, 1.0, 1.5])   # three maneuvers, outer ones stronger

scenarios = [
    (hist_none,  "no maneuvers"),
    (hist_early, "early maneuver (~t=30)"),
    (hist_mid,   "mid maneuver (~t=100)"),
    (hist_late,  "late maneuver (~t=170)"),
    (hist_two,   "two maneuvers (~t=40, 150)"),
    (hist_heavy, "three maneuvers"),
]

fig, axes = plt.subplots(3, len(scenarios), figsize=(22, 8), sharex=True)
for col, (hist, title) in enumerate(scenarios):
    gen = generate(hist, mean_syn, std_syn, kurt_syn, signal_length=N, seed=42)
    for row in range(3):
        axes[row, col].plot(gen[row].numpy(), color="steelblue", linewidth=0.9)
        axes[row, col].grid(True, alpha=0.3)
        if col == 0:
            axes[row, col].set_ylabel(vel_labels[row])
    axes[0, col].set_title(title, fontsize=8)
fig.suptitle("Test 1: designed spike histograms — maneuver timing control")
plt.tight_layout()
plt.show()


# ── Test 2: same histogram, different seeds → diversity check ─────────────────

seeds  = [0, 42, 123, 999, 2024]
colors = ["steelblue", "darkorange", "green", "purple", "brown"]

fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
for i in range(3):
    for j, seed in enumerate(seeds):
        gen = generate(hist_two, mean_syn, std_syn, kurt_syn, signal_length=N, seed=seed)
        axes[i].plot(gen[i].numpy(), label=f"seed={seed}", color=colors[j], linewidth=0.9, alpha=0.8)
    axes[i].set_ylabel(vel_labels[i])
    axes[i].legend(fontsize=7, ncol=5)
    axes[i].grid(True, alpha=0.3)
fig.suptitle("Test 2: same conditions (two maneuvers), 5 different seeds — diversity check")
plt.tight_layout()
plt.show()


# ── Test 3: real spike histograms from dataset vs generated ───────────────────

compare_indices = [0, 15, 30]
fig, axes = plt.subplots(3, len(compare_indices) * 2, figsize=(22, 8), sharex=True)

for col, idx in enumerate(compare_indices):
    hist_real = spike_hists[idx].unsqueeze(0)
    hist_zero = make_hist([], [])
    mean_t    = means[idx].unsqueeze(0)
    std_t     = stds[idx].unsqueeze(0)
    kurt_t    = kurtoses[idx].unsqueeze(0)

    gen_real = generate(hist_real, mean_t, std_t, kurt_t, signal_length=N, seed=42)
    gen_zero = generate(hist_zero, mean_t, std_t, kurt_t, signal_length=N, seed=42)

    for row in range(3):
        axes[row, col * 2].plot(gen_real[row].numpy(), color="steelblue",  linewidth=0.9)
        axes[row, col * 2 + 1].plot(gen_zero[row].numpy(), color="darkorange", linewidth=0.9)
        axes[row, col * 2].grid(True, alpha=0.3)
        axes[row, col * 2 + 1].grid(True, alpha=0.3)
        if col == 0:
            axes[row, col * 2].set_ylabel(vel_labels[row])

    axes[0, col * 2].set_title(f"win {idx} — real hist", fontsize=8)
    axes[0, col * 2 + 1].set_title(f"win {idx} — no maneuvers", fontsize=8)

fig.suptitle("Test 3: real spike histogram vs flat histogram (same stats)")
plt.tight_layout()
plt.show()


# ── Test 4: unseen trajectories 12 and 13 ────────────────────────────────────

for test_traj in [12, 13]:
    test_mask    = np.where(traj_ids == test_traj)[0]
    test_win_idx = test_mask[len(test_mask) // 2]

    hist_test = spike_hists[test_win_idx].unsqueeze(0)
    mean_test = means[test_win_idx].unsqueeze(0)
    std_test  = stds[test_win_idx].unsqueeze(0)
    kurt_test = kurtoses[test_win_idx].unsqueeze(0)
    real_test = signals[test_win_idx]

    gen_test = generate(hist_test, mean_test, std_test, kurt_test, signal_length=N, seed=42)

    fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
    for i in range(3):
        axes[i].plot(real_test[i].numpy(), label="real (unseen)", color="red",       linewidth=1.4)
        axes[i].plot(gen_test[i].numpy(),  label="generated",     color="steelblue", linewidth=1.0)
        axes[i].set_ylabel(vel_labels[i])
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.3)
    fig.suptitle(f"Test 4: unseen trajectory {test_traj} — real vs generated")
    plt.tight_layout()
    plt.show()

    plot_3d_trajectory(
        [real_test, gen_test],
        ["real (unseen)", "generated"],
        ["red", "steelblue"],
        f"3D trajectory — unseen trajectory {test_traj}  (real vs generated)"
    )


# ── Denoising visualisation ───────────────────────────────────────────────────

plot_denoising_steps(spike_hist_0, mean_0, std_0, kurt_0, signal_length=N)
