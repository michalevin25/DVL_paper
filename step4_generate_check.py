# %% [markdown]
# # Step 4: Generate & Evaluate

# %%
%matplotlib inline

# %% Imports & config
import torch
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy import stats as scipy_stats
from scipy.signal import welch
from step3_generatesignals import EDMModel, SIGMA_MIN, SIGMA_MAX

PEAK_SIGMA  = 10   # must match step1
K_PEAKS     = 3

DATA_DIR     = "/Users/michal/Desktop/PhD/dvl paper/DATA"
DATASET_PATH = f"{DATA_DIR}/dvl_dataset.npz"


# %% Load model
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

# %% Helper functions
def generate(peak_map, mean, std, kurtosis, signal_length=206, n_steps=200, seed=None,
             cfg_scale=1.0, return_trajectory=False):
    """
    Generate a signal conditioned on a peak map and scalar statistics.
    peak_map:      (1, 3, signal_length)
    mean:          (1, 3)
    std:           (1, 3)
    kurtosis:      (1, 3)
    cfg_scale:     classifier-free guidance strength (1.0 = no CFG, 2-4 = typical)
    returns:       (3, signal_length)
    """
    if seed is not None:
        torch.manual_seed(seed)

    x      = torch.randn(1, 3, signal_length) * SIGMA_MAX
    sigmas = torch.exp(torch.linspace(np.log(SIGMA_MAX), np.log(SIGMA_MIN), n_steps + 1))

    # null conditions for CFG unconditional pass
    null_peak_map = torch.zeros_like(peak_map)
    null_mean     = torch.zeros_like(mean)
    null_std      = torch.zeros_like(std)
    null_kurtosis = torch.zeros_like(kurtosis)

    snapshots = []

    with torch.no_grad():
        for i in range(n_steps):
            sigma_cur  = sigmas[i].expand(1)
            sigma_next = sigmas[i + 1].expand(1)
            dt         = (sigma_next - sigma_cur).view(1, 1, 1)

            x_denoised_cond = model(x, sigma_cur, peak_map, mean, std, kurtosis)
            if cfg_scale != 1.0:
                x_denoised_uncond = model(x, sigma_cur, null_peak_map, null_mean, null_std, null_kurtosis)
                x_denoised = x_denoised_uncond + cfg_scale * (x_denoised_cond - x_denoised_uncond)
            else:
                x_denoised = x_denoised_cond

            d_cur  = (x - x_denoised) / sigma_cur.view(1, 1, 1)
            x_next = x + dt * d_cur

            if i < n_steps - 1:
                x_denoised_next_cond = model(x_next, sigma_next, peak_map, mean, std, kurtosis)
                if cfg_scale != 1.0:
                    x_denoised_next_uncond = model(x_next, sigma_next, null_peak_map, null_mean, null_std, null_kurtosis)
                    x_denoised_next = x_denoised_next_uncond + cfg_scale * (x_denoised_next_cond - x_denoised_next_uncond)
                else:
                    x_denoised_next = x_denoised_next_cond
                d_next = (x_next - x_denoised_next) / sigma_next.view(1, 1, 1)
                x_next = x + dt * (d_cur + d_next) / 2

            x = x_next

            if return_trajectory:
                snapshots.append((i, sigmas[i].item(), x.squeeze(0).clone()))

    result = x.squeeze(0)  # (3, N) — zero mean, unit variance (window-normalized space)

    # denormalize: reverse the window-level normalization applied in step1
    target_std  = std.squeeze(0).unsqueeze(1)   # (3, 1)
    target_mean = mean.squeeze(0).unsqueeze(1)  # (3, 1)
    result      = result * target_std + target_mean

    if return_trajectory:
        return result, snapshots
    return result


def make_peak_map(peak_times, amplitudes, signal_length=206, sigma=PEAK_SIGMA):
    """
    Build a (1, 3, signal_length) peak map from specified maneuver events.
    peak_times:  list of sample indices where peaks occur
    amplitudes:  corresponding amplitude values (positive = forward, negative = reverse)
    All three axes get the same map; adjust per-axis after if needed.
    """
    t   = np.arange(signal_length, dtype=np.float32)
    out = np.zeros(signal_length, dtype=np.float32)
    for loc, amp in zip(peak_times, amplitudes):
        out += amp * np.exp(-((t - loc) ** 2) / (2 * sigma ** 2))
    pm = torch.tensor(out).unsqueeze(0).expand(3, -1)  # (3, L)
    return pm.unsqueeze(0).clone()                      # (1, 3, L)


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


def plot_denoising_steps(peak_map, mean, std, kurtosis, signal_length=206, n_steps=200, seed=42):
    _, snapshots = generate(peak_map, mean, std, kurtosis, signal_length, n_steps, seed, return_trajectory=True)
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


# %% Load dataset

data      = np.load(DATASET_PATH)
peak_maps = torch.tensor(data["peak_maps"], dtype=torch.float32)  # (W, 3, N)
means     = torch.tensor(data["means"],     dtype=torch.float32)  # (W, 3)
stds      = torch.tensor(data["stds"],      dtype=torch.float32)  # (W, 3)
kurtoses  = torch.tensor(data["kurtoses"],  dtype=torch.float32)  # (W, 3)
signals   = torch.tensor(data["signals"],   dtype=torch.float32)  # (W, 3, N)
traj_ids  = data["traj_ids"]                                       # (W,)

N           = signals.shape[-1]   # signal length (206)
vel_labels  = ["vx", "vy", "vz"]
cond_idx    = 0
real_signal = signals[cond_idx]
peak_map_0  = peak_maps[cond_idx].unsqueeze(0)  # (1, 3, N)
mean_0      = means[cond_idx].unsqueeze(0)
std_0       = stds[cond_idx].unsqueeze(0)
kurt_0      = kurtoses[cond_idx].unsqueeze(0)


# %% Shared conditions

mean_syn = means[0].unsqueeze(0)
std_syn  = stds[0].unsqueeze(0)
kurt_syn = kurtoses[0].unsqueeze(0)

# peak times are sample indices (0–205); amplitudes are arbitrary signed floats
peak_none  = make_peak_map([], [])
peak_early = make_peak_map([30],      [2.0])
peak_mid   = make_peak_map([103],     [2.0])
peak_late  = make_peak_map([175],     [2.0])
peak_two   = make_peak_map([40, 155], [2.0, 2.0])
peak_heavy = make_peak_map([40, 103, 155], [2.0, 1.5, 2.0])

# %% Test 1: designed peak maps
scenarios = [
    (peak_none,  "no maneuvers"),
    (peak_early, "early maneuver (~t=30)"),
    (peak_mid,   "mid maneuver (~t=103)"),
    (peak_late,  "late maneuver (~t=175)"),
    (peak_two,   "two maneuvers (~t=40, 155)"),
    (peak_heavy, "three maneuvers"),
]
fig, axes = plt.subplots(3, len(scenarios), figsize=(22, 8), sharex=True)
for col, (pm, title) in enumerate(scenarios):
    gen = generate(pm, mean_syn, std_syn, kurt_syn, signal_length=N, seed=42)
    for row in range(3):
        axes[row, col].plot(gen[row].numpy(), color="steelblue", linewidth=0.9)
        axes[row, col].grid(True, alpha=0.3)
        if col == 0:
            axes[row, col].set_ylabel(vel_labels[row])
    axes[0, col].set_title(title, fontsize=8)
fig.suptitle("Test 1: designed peak maps — maneuver timing control")
plt.tight_layout()
plt.show()


# %% Test 2: same peak map, different seeds
seeds  = [0, 42, 123, 999, 2024]
colors = ["steelblue", "darkorange", "green", "purple", "brown"]
fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
for i in range(3):
    for j, seed in enumerate(seeds):
        gen = generate(peak_two, mean_syn, std_syn, kurt_syn, signal_length=N, seed=seed)
        axes[i].plot(gen[i].numpy(), label=f"seed={seed}", color=colors[j], linewidth=0.9, alpha=0.8)
    axes[i].set_ylabel(vel_labels[i])
    axes[i].legend(fontsize=7, ncol=5)
    axes[i].grid(True, alpha=0.3)
fig.suptitle("Test 2: same conditions (two maneuvers), 5 different seeds — diversity check")
plt.tight_layout()
plt.show()


# %% Test 3: real peak map vs empty peak map
compare_indices = [0, 15, 30]
fig, axes = plt.subplots(3, len(compare_indices) * 2, figsize=(22, 8), sharex=True)
for col, idx in enumerate(compare_indices):
    pm_real  = peak_maps[idx].unsqueeze(0)
    pm_empty = make_peak_map([], [])
    mean_t   = means[idx].unsqueeze(0)
    std_t    = stds[idx].unsqueeze(0)
    kurt_t   = kurtoses[idx].unsqueeze(0)
    gen_real  = generate(pm_real,  mean_t, std_t, kurt_t, signal_length=N, seed=42)
    gen_empty = generate(pm_empty, mean_t, std_t, kurt_t, signal_length=N, seed=42)
    for row in range(3):
        axes[row, col * 2].plot(gen_real[row].numpy(),  color="steelblue",  linewidth=0.9)
        axes[row, col * 2 + 1].plot(gen_empty[row].numpy(), color="darkorange", linewidth=0.9)
        axes[row, col * 2].grid(True, alpha=0.3)
        axes[row, col * 2 + 1].grid(True, alpha=0.3)
        if col == 0:
            axes[row, col * 2].set_ylabel(vel_labels[row])
    axes[0, col * 2].set_title(f"win {idx} — real peaks", fontsize=8)
    axes[0, col * 2 + 1].set_title(f"win {idx} — no maneuvers", fontsize=8)
fig.suptitle("Test 3: real peak map vs empty peak map (same stats)")
plt.tight_layout()
plt.show()


# %% Test 4: unseen trajectories 12 and 13
for test_traj in [12, 13]:
    test_mask    = np.where(traj_ids == test_traj)[0]
    test_win_idx = test_mask[len(test_mask) // 2]
    pm_test   = peak_maps[test_win_idx].unsqueeze(0)
    mean_test = means[test_win_idx].unsqueeze(0)
    std_test  = stds[test_win_idx].unsqueeze(0)
    kurt_test = kurtoses[test_win_idx].unsqueeze(0)
    real_test = signals[test_win_idx]
    gen_test = generate(pm_test, mean_test, std_test, kurt_test, signal_length=N, seed=42)
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


# %% Denoising visualisation
plot_denoising_steps(peak_map_0, mean_0, std_0, kurt_0, signal_length=N)


# %% Evaluation helpers
def window_stats(sig_tensor):
    """(3, N) tensor → (mean, std, kurtosis) each shape (3,)."""
    s = sig_tensor.numpy()
    return s.mean(axis=1), s.std(axis=1), scipy_stats.kurtosis(s, axis=1, fisher=True)


# %% Evaluation 1: Statistical fidelity
real_m, real_s, real_k = [], [], []
gen_m,  gen_s,  gen_k  = [], [], []
is_test_flag            = []
print("Evaluation 1: statistical fidelity — generating for all windows...")
for idx in range(len(signals)):
    hi = peak_maps[idx].unsqueeze(0)
    mi = means[idx].unsqueeze(0)
    si = stds[idx].unsqueeze(0)
    ki = kurtoses[idx].unsqueeze(0)
    real_denorm = signals[idx] * stds[idx].unsqueeze(1) + means[idx].unsqueeze(1)
    rm, rs, rk  = window_stats(real_denorm)
    gen_i       = generate(hi, mi, si, ki, signal_length=N, seed=42)
    gm, gs, gk  = window_stats(gen_i)
    real_m.extend(rm);  real_s.extend(rs);  real_k.extend(rk)
    gen_m.extend(gm);   gen_s.extend(gs);   gen_k.extend(gk)
    is_test_flag.extend([traj_ids[idx] >= 12] * 3)
real_m, real_s, real_k = np.array(real_m), np.array(real_s), np.array(real_k)
gen_m,  gen_s,  gen_k  = np.array(gen_m),  np.array(gen_s),  np.array(gen_k)
is_test_flag            = np.array(is_test_flag)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
stat_pairs = [(real_m, gen_m, "mean"), (real_s, gen_s, "std"), (real_k, gen_k, "kurtosis")]
for ax, (real, gen, label) in zip(axes, stat_pairs):
    tr = ~is_test_flag; te = is_test_flag
    ax.scatter(real[tr], gen[tr], color="steelblue", alpha=0.7, s=30, label="train")
    ax.scatter(real[te], gen[te], color="red",       alpha=0.9, s=60, label="test", zorder=5)
    lo = min(real.min(), gen.min()); hi = max(real.max(), gen.max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.5)
    r, _ = scipy_stats.pearsonr(real, gen)
    ax.text(0.05, 0.92, f"r = {r:.3f}", transform=ax.transAxes, fontsize=9)
    ax.set_xlabel(f"real {label}"); ax.set_ylabel(f"generated {label}")
    ax.set_title(label); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
fig.suptitle("Evaluation 1: statistical fidelity")
plt.tight_layout(); plt.show()


# %% Evaluation 2: Power spectral density
test_indices = np.where(traj_ids >= 12)[0]
nperseg      = min(64, N)
real_psds, gen_psds = [], []
print("Evaluation 2: power spectral density...")
for idx in test_indices:
    hi = peak_maps[idx].unsqueeze(0); mi = means[idx].unsqueeze(0)
    si = stds[idx].unsqueeze(0);      ki = kurtoses[idx].unsqueeze(0)
    gen_i = generate(hi, mi, si, ki, signal_length=N, seed=42)
    real_denorm = signals[idx] * stds[idx].unsqueeze(1) + means[idx].unsqueeze(1)
    for ax in range(3):
        freqs, Pr = welch(real_denorm[ax].numpy(), nperseg=nperseg)
        _,     Pg = welch(gen_i[ax].numpy(),       nperseg=nperseg)
        real_psds.append(Pr); gen_psds.append(Pg)
real_psds = np.array(real_psds); gen_psds = np.array(gen_psds)
fig, ax = plt.subplots(figsize=(10, 5))
ax.semilogy(freqs, real_psds.mean(axis=0), color="red",       linewidth=1.5, label="real")
ax.fill_between(freqs, real_psds.mean(axis=0)-real_psds.std(axis=0),
                       real_psds.mean(axis=0)+real_psds.std(axis=0), color="red", alpha=0.15)
ax.semilogy(freqs, gen_psds.mean(axis=0),  color="steelblue", linewidth=1.5, label="generated")
ax.fill_between(freqs, gen_psds.mean(axis=0)-gen_psds.std(axis=0),
                       gen_psds.mean(axis=0)+gen_psds.std(axis=0), color="steelblue", alpha=0.15)
ax.set_xlabel("Frequency (normalized)"); ax.set_ylabel("Power")
ax.set_title("Evaluation 2: power spectral density"); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()


# %% Evaluation 3: Diversity
test_indices = np.where(traj_ids >= 12)[0]
div_seeds  = [0, 42, 123, 999, 2024]
div_colors = ["steelblue", "darkorange", "green", "purple", "brown"]
print("Evaluation 3: diversity...")
for idx in test_indices[:2]:
    hi = peak_maps[idx].unsqueeze(0); mi = means[idx].unsqueeze(0)
    si = stds[idx].unsqueeze(0);      ki = kurtoses[idx].unsqueeze(0)
    samples = np.stack([generate(hi, mi, si, ki, signal_length=N, seed=s).numpy() for s in div_seeds])
    smean = samples.mean(axis=0); sstd = samples.std(axis=0); t = np.arange(N)
    fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
    for ax_i in range(3):
        axes[ax_i].plot(signals[idx, ax_i].numpy(), color="red", linewidth=1.4, label="real", zorder=10)
        for j, (s, c) in enumerate(zip(samples, div_colors)):
            axes[ax_i].plot(s[ax_i], color=c, linewidth=0.8, alpha=0.55, label=f"seed {div_seeds[j]}")
        axes[ax_i].fill_between(t, smean[ax_i]-sstd[ax_i], smean[ax_i]+sstd[ax_i],
                                color="steelblue", alpha=0.18, label="±1 std")
        axes[ax_i].set_ylabel(vel_labels[ax_i]); axes[ax_i].legend(fontsize=7, ncol=4)
        axes[ax_i].grid(True, alpha=0.3)
    pairwise = [np.sqrt(np.mean((samples[i]-samples[j])**2))
                for i in range(len(samples)) for j in range(i+1, len(samples))]
    fig.suptitle(f"Evaluation 3: diversity — trajectory {traj_ids[idx]}, window {idx} — "
                 f"5 seeds  (mean pairwise RMS = {np.mean(pairwise):.4f})")
    plt.tight_layout(); plt.show()


# %% Evaluation 4: Controllability
ctrl_scenarios = [
    (peak_none,  "no maneuvers"), (peak_early, "early (~t=30)"),
    (peak_mid,   "mid (~t=103)"), (peak_late,  "late (~t=175)"),
    (peak_two,   "two maneuvers"),
]
N_BINS_EVAL = 20
print("Evaluation 4: controllability...")
fig, axes = plt.subplots(2, len(ctrl_scenarios), figsize=(20, 6))
bin_edges   = np.linspace(0, N, N_BINS_EVAL + 1, dtype=int)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
for col, (pm, title) in enumerate(ctrl_scenarios):
    gen    = generate(pm, mean_syn, std_syn, kurt_syn, signal_length=N, seed=42)
    gen_np = gen.numpy()
    pm_np  = pm[0].numpy().mean(axis=0)
    gen_rms = np.array([np.sqrt(np.mean(gen_np[:, bin_edges[b]:bin_edges[b+1]]**2)) for b in range(N_BINS_EVAL)])
    axes[0, col].plot(pm_np, color="darkorange", linewidth=0.9)
    axes[0, col].set_title(title, fontsize=8)
    if col == 0: axes[0, col].set_ylabel("input peak map")
    axes[1, col].bar(bin_centers, gen_rms, width=N/N_BINS_EVAL*0.8, color="steelblue", alpha=0.8)
    axes[1, col].set_ylim(0, None)
    if col == 0: axes[1, col].set_ylabel("generated signal RMS")
fig.suptitle("Evaluation 4: controllability — input peak map vs generated signal energy per bin")
plt.tight_layout(); plt.show()


# %% Evaluation 5: condition ablation

print("Evaluation 5: condition ablation...")

abl_idx  = np.where(traj_ids == 3)[0][len(np.where(traj_ids == 3)[0]) // 2]   # middle window of trajectory 3
abl_pm   = peak_maps[abl_idx].unsqueeze(0)
abl_mean = means[abl_idx].unsqueeze(0)
abl_std  = stds[abl_idx].unsqueeze(0)
abl_kurt = kurtoses[abl_idx].unsqueeze(0)

ablation_scenarios = [
    (abl_pm,               abl_mean,                   abl_std,                  abl_kurt,                   "baseline\n(all conditions)"),
    (make_peak_map([], []), abl_mean,                   abl_std,                  abl_kurt,                   "no peak map\n(empty)"),
    (abl_pm,               torch.zeros_like(abl_mean), torch.ones_like(abl_std), abl_kurt,                   "no mean+std\n(mean→0, std→1)"),
    (abl_pm,               abl_mean,                   abl_std,                  torch.zeros_like(abl_kurt), "no kurtosis\n(kurt → 0)"),
    (make_peak_map([], []), torch.zeros_like(abl_mean), torch.ones_like(abl_std), torch.zeros_like(abl_kurt), "all ablated\n(all default)"),
]

fig, axes = plt.subplots(3, len(ablation_scenarios), figsize=(22, 8), sharex=True)
colors_abl = ["steelblue", "darkorange", "green", "purple", "brown", "gray"]

for col, (hist, mean, std, kurt, title) in enumerate(ablation_scenarios):
    gen = generate(hist, mean, std, kurt, signal_length=N, seed=42)
    for row in range(3):
        axes[row, col].plot(gen[row].numpy(), color=colors_abl[col], linewidth=0.9)
        axes[row, col].grid(True, alpha=0.3)
        if col == 0:
            axes[row, col].set_ylabel(vel_labels[row])
    axes[0, col].set_title(title, fontsize=8)

fig.suptitle(
    f"Evaluation 5: condition ablation — trajectory {traj_ids[abl_idx]}, window {abl_idx}\n"
    "each column removes one conditioning signal"
)
plt.tight_layout()
plt.show()
