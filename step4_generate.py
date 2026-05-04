import torch
import numpy as np
import matplotlib.pyplot as plt
from step3_generatesignals import EDMModel, SIGMA_MIN, SIGMA_MAX

MODEL_PATH   = "/Users/michal/Desktop/PhD/dvl paper/DATA/edm_model.pt"
DATASET_PATH = "/Users/michal/Desktop/PhD/dvl paper/DATA/dvl_dataset.npz"


# ── Load trained model ────────────────────────────────────────────────────────

model = EDMModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()
print("Model loaded.")


# ── EDM sampler (deterministic Euler-Heun) ────────────────────────────────────

def generate(curvature, mean, std, kurtosis, n_steps=200, seed=None, return_trajectory=False):
    """
    Generate a new trajectory conditioned on curvature, mean, std, kurtosis.
    curvature: (1, 3, 206)
    mean:      (1, 3)
    std:       (1, 3)
    kurtosis:  (1, 3)
    returns:   (3, 206) generated signal
    """
    if seed is not None:
        torch.manual_seed(seed)

    # step 1: start from pure noise at σ_max
    x = torch.randn(1, 3, curvature.shape[-1]) * SIGMA_MAX

    # step 2: build σ schedule from σ_max → σ_min
    sigmas = torch.exp(torch.linspace(np.log(SIGMA_MAX), np.log(SIGMA_MIN), n_steps + 1))

    snapshots = []  # store x at selected steps for visualization

    # step 3: iterative denoising with Heun correction
    with torch.no_grad():
        for i in range(n_steps):
            sigma_cur  = sigmas[i].expand(1)
            sigma_next = sigmas[i + 1].expand(1)
            dt         = (sigma_next - sigma_cur).view(1, 1, 1)

            # first derivative (at current point)
            x_denoised = model(x, sigma_cur, curvature, mean, std, kurtosis)
            d_cur      = (x - x_denoised) / sigma_cur.view(1, 1, 1)

            # Euler step to get preliminary x_next
            x_next = x + dt * d_cur

            # skip Heun correction at last step
            if i < n_steps - 1:
                # second derivative (at next point)
                x_denoised_next = model(x_next, sigma_next, curvature, mean, std, kurtosis)
                d_next          = (x_next - x_denoised_next) / sigma_next.view(1, 1, 1)

                # corrected step using average slope
                x_next = x + dt * (d_cur + d_next) / 2

            x = x_next

            if return_trajectory:
                snapshots.append((i, sigmas[i].item(), x.squeeze(0).clone()))

    if return_trajectory:
        return x.squeeze(0), snapshots
    return x.squeeze(0)  # (3, 206)


def plot_3d_trajectory(signals_list, labels_list, colors_list, title, dt=1.0):
    """Integrate velocity windows → 3D paths and overlay them."""
    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection='3d')

    for sig, label, color in zip(signals_list, labels_list, colors_list):
        v = sig.numpy() if isinstance(sig, torch.Tensor) else sig  # (3, N)
        x = np.cumsum(v[0] * dt)
        y = np.cumsum(v[1] * dt)
        z = np.cumsum(v[2] * dt)
        ax.plot(x, y, z, label=label, color=color, linewidth=1.5)
        ax.scatter(x[0], y[0], z[0], color=color, s=40, zorder=5)  # mark start

    ax.set_xlabel("X (integrated vx)")
    ax.set_ylabel("Y (integrated vy)")
    ax.set_zlabel("Z (integrated vz)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_denoising_steps(curvature, mean, std, kurtosis, n_steps=200, seed=42):
    _, snapshots = generate(curvature, mean, std, kurtosis, n_steps=n_steps, seed=seed, return_trajectory=True)

    # pick 6 evenly spaced snapshots
    indices  = np.linspace(0, len(snapshots) - 1, 6, dtype=int)
    selected = [snapshots[i] for i in indices]

    fig, axes = plt.subplots(3, 6, figsize=(20, 7), sharex=True)
    labels = ["vx", "vy", "vz"]
    for col, (step, sigma, x) in enumerate(selected):
        for row in range(3):
            axes[row, col].plot(x[row].numpy(), linewidth=0.8, color="steelblue")
            axes[row, col].grid(True, alpha=0.3)
            if row == 0:
                axes[row, col].set_title(f"step {step}\nσ={sigma:.2f}", fontsize=8)
            if col == 0:
                axes[row, col].set_ylabel(labels[row])
    fig.suptitle("Denoising process — pure noise → generated trajectory")
    plt.tight_layout()
    plt.show()


# ── Use conditions from an existing trajectory ────────────────────────────────

data       = np.load(DATASET_PATH)
curvatures = torch.tensor(data["curvatures"], dtype=torch.float32)  # (W, 3, 206)
means      = torch.tensor(data["means"],      dtype=torch.float32)  # (W, 3)
stds       = torch.tensor(data["stds"],       dtype=torch.float32)  # (W, 3)
kurtoses   = torch.tensor(data["kurtoses"],   dtype=torch.float32)  # (W, 3)
signals    = torch.tensor(data["signals"],    dtype=torch.float32)  # (W, 3, 206)
traj_ids   = data["traj_ids"]                                        # (W,) values 1–13

# pick first window as the condition
cond_idx    = 0
curvature   = curvatures[cond_idx].unsqueeze(0)  # (1, 3, 206)
mean        = means[cond_idx].unsqueeze(0)        # (1, 3)
std         = stds[cond_idx].unsqueeze(0)         # (1, 3)
kurtosis    = kurtoses[cond_idx].unsqueeze(0)     # (1, 3)
real_signal = signals[cond_idx]                   # (3, 206)

labels = ["vx", "vy", "vz"]

# ── Test 2: synthesized curvature conditions ──────────────────────────────────

N = signals.shape[-1]  # window size from dataset

# use mean/std/kurtosis from window 0 as the velocity regime
mean_syn = means[0].unsqueeze(0)     # (1, 3)
std_syn  = stds[0].unsqueeze(0)      # (1, 3)
kurt_syn = kurtoses[0].unsqueeze(0)  # (1, 3)

# scenario A: flat curvature — smooth trajectory, no maneuvers
curv_flat = torch.zeros(1, 3, N)

# scenario B: single maneuver peak at sample 100
curv_peak = torch.zeros(1, 3, N)
peak      = torch.exp(-0.5 * ((torch.arange(N) - 100) / 10.0) ** 2)
curv_peak[0, 0] = peak   # maneuver only in vx
curv_peak[0, 1] = peak   # and vy

# generate both
gen_flat = generate(curv_flat, mean_syn, std_syn, kurt_syn, n_steps=200, seed=42)
gen_peak = generate(curv_peak, mean_syn, std_syn, kurt_syn, n_steps=200, seed=42)

fig, axes = plt.subplots(3, 2, figsize=(16, 8), sharex=True)
for i in range(3):
    axes[i, 0].plot(gen_flat[i].numpy(), color="steelblue", linewidth=1.0)
    axes[i, 0].set_ylabel(labels[i])
    axes[i, 0].grid(True, alpha=0.3)

    axes[i, 1].plot(gen_peak[i].numpy(), color="darkorange", linewidth=1.0)
    axes[i, 1].grid(True, alpha=0.3)

axes[0, 0].set_title("Scenario A: flat curvature (smooth trajectory)")
axes[0, 1].set_title("Scenario B: single maneuver peak at sample 100")
plt.tight_layout()
plt.show()

# ── Comparison: full conditions vs mean/std only ─────────────────────────────

compare_indices = [0, 15, 30]
fig, axes = plt.subplots(3, len(compare_indices) * 2, figsize=(22, 8), sharex=True)

for col, idx in enumerate(compare_indices):
    mean_t    = means[idx].unsqueeze(0)
    std_t     = stds[idx].unsqueeze(0)
    kurt_t    = kurtoses[idx].unsqueeze(0)
    curv_real = curvatures[idx].unsqueeze(0)
    curv_zero = torch.zeros(1, 3, N)

    gen_full = generate(curv_real, mean_t, std_t, kurt_t, n_steps=200, seed=42)
    gen_stat = generate(curv_zero, mean_t, std_t, kurt_t, n_steps=200, seed=42)

    col_full = col * 2
    col_stat = col * 2 + 1

    for row in range(3):
        axes[row, col_full].plot(gen_full[row].numpy(), color="steelblue",  linewidth=0.9)
        axes[row, col_stat].plot(gen_stat[row].numpy(), color="darkorange", linewidth=0.9)
        axes[row, col_full].grid(True, alpha=0.3)
        axes[row, col_stat].grid(True, alpha=0.3)
        if col == 0:
            axes[row, col_full].set_ylabel(labels[row])

    axes[0, col_full].set_title(f"win {idx} — full conditions",   fontsize=8)
    axes[0, col_stat].set_title(f"win {idx} — mean/std only",     fontsize=8)

fig.suptitle("Full conditions (curvature + mean + std + kurtosis)  vs  mean/std/kurtosis only")
plt.tight_layout()
plt.show()

# ── Test 1: mix conditions from different windows ────────────────────────────

# curvature from window 0, mean/std from window 20 (different trajectory character)
curv_idx  = 0
stats_idx = 2

curvature_mix  = curvatures[curv_idx].unsqueeze(0)   # (1, 3, 206)
mean_mix       = means[stats_idx].unsqueeze(0)        # (1, 3)
std_mix        = stds[stats_idx].unsqueeze(0)         # (1, 3)
kurt_mix       = kurtoses[stats_idx].unsqueeze(0)     # (1, 3)
real_curv_src  = signals[curv_idx]                    # real signal that owns the curvature
real_stats_src = signals[stats_idx]                   # real signal that owns the mean/std

gen_mix = generate(curvature_mix, mean_mix, std_mix, kurt_mix, n_steps=200, seed=42)

fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
for i in range(3):
    axes[i].plot(real_curv_src[i].numpy(),  label=f"real (curvature src, win {curv_idx})",  color="red",        linewidth=1.2)
    axes[i].plot(real_stats_src[i].numpy(), label=f"real (stats src, win {stats_idx})",      color="darkorange", linewidth=1.2, alpha=0.7)
    axes[i].plot(gen_mix[i].numpy(),        label="generated (mixed conditions)",             color="steelblue",  linewidth=1.0)
    axes[i].set_ylabel(labels[i])
    axes[i].legend(fontsize=7)
    axes[i].grid(True, alpha=0.3)
fig.suptitle(f"Test 1: curvature from win {curv_idx}, mean/std from win {stats_idx}")
plt.tight_layout()
plt.show()


# ── Test 3: multiple samples with same conditions, different seeds ────────────

n_samples = 5
seeds     = [0, 42, 123, 999, 2024]
labels    = ["vx", "vy", "vz"]
colors    = ["steelblue", "darkorange", "green", "purple", "brown"]

fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
for i in range(3):
    axes[i].plot(real_signal[i].numpy(), label="real", color="red", linewidth=1.5, zorder=10)
    for j, seed in enumerate(seeds):
        gen = generate(curvature, mean, std, kurtosis, n_steps=200, seed=seed)
        axes[i].plot(gen[i].numpy(), label=f"gen seed={seed}", color=colors[j],
                     linewidth=0.9, alpha=0.7)
    axes[i].set_ylabel(labels[i])
    axes[i].legend(fontsize=7, ncol=3)
    axes[i].grid(True, alpha=0.3)

fig.suptitle(f"5 generated samples — same conditions (window {cond_idx}), different seeds")
plt.tight_layout()
plt.show()

# 3D trajectory — real vs generated for window cond_idx
gen_3d = generate(curvature, mean, std, kurtosis, n_steps=200, seed=42)
plot_3d_trajectory(
    [real_signal, gen_3d],
    ["real", "generated"],
    ["red", "steelblue"],
    f"3D trajectory — window {cond_idx}  (real vs generated)"
)

# ── Visualize denoising steps ─────────────────────────────────────────────────
plot_denoising_steps(curvature, mean, std, kurtosis, n_steps=200, seed=42)


# ── Test 4: unseen conditions from test trajectories (12 and 13) ──────────────
# Trajectories 12–13 were never seen during training.
# We pick one window from each and generate with full conditions.

for test_traj in [12, 13]:
    test_mask    = np.where(traj_ids == test_traj)[0]
    test_win_idx = test_mask[len(test_mask) // 2]         # middle window of this trajectory

    curv_test   = curvatures[test_win_idx].unsqueeze(0)    # (1, 3, 206)
    mean_test   = means[test_win_idx].unsqueeze(0)         # (1, 3)
    std_test    = stds[test_win_idx].unsqueeze(0)          # (1, 3)
    kurt_test   = kurtoses[test_win_idx].unsqueeze(0)      # (1, 3)
    real_test   = signals[test_win_idx]                    # (3, 206)

    gen_test = generate(curv_test, mean_test, std_test, kurt_test, n_steps=200, seed=42)

    fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
    for i in range(3):
        axes[i].plot(real_test[i].numpy(), label="real (unseen)", color="red",       linewidth=1.4)
        axes[i].plot(gen_test[i].numpy(),  label="generated",     color="steelblue", linewidth=1.0)
        axes[i].set_ylabel(labels[i])
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
