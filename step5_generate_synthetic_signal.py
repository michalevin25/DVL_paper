# %% [markdown]
# # Generate Synthetic DVL Dataset
# 13 trajectories × N_SEEDS signals × 206 samples, with mixed conditions.
# Stats (mean, std, kurtosis) are taken from different real trajectories and
# combined to create diverse but physically plausible synthetic trajectories.
# Generates at signal_length=206 to match training length.

# %% Imports & setup
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from scipy.stats import kurtosis as compute_kurtosis
from step3_train import EDMModel, SIGMA_MIN, SIGMA_MAX

%matplotlib inline

PEAK_SIGMA    = 10     # must match step1
SIGNAL_LENGTH = 206    # matches training length
N_SEEDS       = 5      # signals generated per trajectory (different seeds)
TRAIN_LENGTH  = 206    # length model was trained on — used to scale peak_times

DATA_DIR = "/Users/michal/Desktop/PhD/dvl paper/DATA"
OUT_DIR  = "/Users/michal/Desktop/PhD/dvl paper/GENERATED DATA"
os.makedirs(OUT_DIR, exist_ok=True)

candidates = sorted(glob.glob(f"{DATA_DIR}/edm_model_*.pt"))
if not candidates:
    raise FileNotFoundError(f"No edm_model_*.pt found in {DATA_DIR}")
MODEL_PATH = candidates[-1]
print(f"Loading model: {os.path.basename(MODEL_PATH)}")
model = EDMModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()
print("Model loaded.")


# %% Core functions

def generate(peak_map, mean, std, kurtosis, signal_length=SIGNAL_LENGTH,
             n_steps=200, seed=None, cfg_scale=1.0):
    """
    peak_map : (1, 3, signal_length)
    mean/std/kurtosis : (1, 3)  — conditions passed to the model; model outputs m/s directly
    returns  : (3, signal_length) — velocity signal in m/s
    """
    if seed is not None:
        torch.manual_seed(seed)

    x      = torch.randn(1, 3, signal_length) * SIGMA_MAX
    sigmas = torch.exp(torch.linspace(np.log(SIGMA_MAX), np.log(SIGMA_MIN), n_steps + 1))

    null_peak_map = torch.zeros_like(peak_map)
    null_mean     = torch.zeros_like(mean)
    null_std      = torch.zeros_like(std)
    null_kurtosis = torch.zeros_like(kurtosis)

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
                x_dn_next_cond = model(x_next, sigma_next, peak_map, mean, std, kurtosis)
                if cfg_scale != 1.0:
                    x_dn_next_uncond = model(x_next, sigma_next, null_peak_map, null_mean, null_std, null_kurtosis)
                    x_dn_next = x_dn_next_uncond + cfg_scale * (x_dn_next_cond - x_dn_next_uncond)
                else:
                    x_dn_next = x_dn_next_cond
                d_next = (x_next - x_dn_next) / sigma_next.view(1, 1, 1)
                x_next = x + dt * (d_cur + d_next) / 2

            x = x_next

    return x.squeeze(0)  # (3, N) — model trained on raw m/s, output is m/s directly


def make_peak_map(peak_times, amplitudes, signal_length=SIGNAL_LENGTH, sigma=PEAK_SIGMA):
    """Build a (1, 3, signal_length) peak map — same map applied to all 3 axes."""
    t   = np.arange(signal_length, dtype=np.float32)
    out = np.zeros(signal_length, dtype=np.float32)
    for loc, amp in zip(peak_times, amplitudes):
        out += amp * np.exp(-((t - loc) ** 2) / (2 * sigma ** 2))
    pm = torch.tensor(out).unsqueeze(0).expand(3, -1)
    return pm.unsqueeze(0).clone()  # (1, 3, L)


# %% Trajectory definitions
#
# Stats are taken from real trajectories and mixed to create diverse scenarios.
# Physical meaning:
#   mean_vx  : forward(+) / backward(-) velocity
#   mean_vy  : starboard(+) / port(-) lateral velocity
#   mean_vz  : vertical drift (sign depends on vehicle convention)
#   std      : velocity variability per axis — higher = more dynamic motion
#   kurtosis : impulsiveness — high = rare sharp velocity bursts; ~0 = Gaussian / smooth
#
# Real trajectory stat reference (window-averaged):
#   T1 : mean(-0.29,-0.06,-0.08) std(0.96,0.78,1.05) kurt(+0.3,+1.6,+16.7)  ← vz-spiky
#   T2 : mean(-0.46,+0.31,-0.08) std(0.62,0.96,1.06) kurt(+3.8,+1.4,-0.0)
#   T3 : mean(+0.02,+0.06,+0.14) std(0.80,1.10,1.12) kurt(+0.2,+3.2,+2.5)
#   T4 : mean(-0.10,+0.17,+0.01) std(0.80,0.96,0.78) kurt(-0.4,+1.5,+1.6)   ← Gaussian
#   T5 : mean(-0.07,+0.05,-0.03) std(1.01,0.98,0.98) kurt(+14.0,+0.3,-0.4)  ← vx-spiky
#   T6 : mean(+0.01,-0.30,-0.07) std(0.56,0.72,0.79) kurt(+22.8,+10.1,+10.9) ← very spiky
#   T7 : mean(-0.31,-0.13,+0.00) std(0.86,0.67,0.93) kurt(+11.6,+0.0,+5.0)  ← vx-spiky
#   T8 : mean(+0.17,-0.07,+0.08) std(0.95,1.00,0.78) kurt(-0.6,+0.2,+9.1)   ← vz-spiky
#   T9 : mean(+0.01,+0.05,+0.00) std(1.02,1.01,1.04) kurt(-0.3,+1.1,+12.3)  ← vz-spiky
#  T10 : mean(+0.33,-0.10,+0.07) std(0.92,1.05,1.07) kurt(+0.1,+1.6,+0.3)   ← forward
#  T11 : mean(+0.40,+0.13,+0.02) std(0.57,0.85,0.68) kurt(+11.8,+9.2,+34.3) ← extreme
#  T12 : mean(+0.04,+0.27,-0.00) std(0.94,0.85,0.97) kurt(+0.4,-0.7,-0.1)   ← Gaussian
#  T13 : mean(+0.03,+0.20,-0.19) std(0.72,0.61,0.87) kurt(+1.6,-0.9,-0.2)

TRAJECTORIES = [
    # ── id  description                          mean(vx,vy,vz)            std(vx,vy,vz)        kurt(vx,vy,vz)        peak_times   amplitudes
    dict(id=1,  desc="Forward cruise, Gaussian",           mean=[+0.331,-0.103,+0.069], std=[0.804,0.956,0.782], kurt=[+0.44,-0.67,-0.06], peak_times=[],         amps=[],              peak_label="none"),
    dict(id=2,  desc="Lateral + backward-vz, smooth",      mean=[+0.029,+0.199,-0.185], std=[0.716,0.614,0.867], kurt=[+1.60,-0.95,+3.00], peak_times=[],         amps=[],              peak_label="none"),
    dict(id=3,  desc="Backward-lateral, spiky vz, early",  mean=[-0.455,+0.310,-0.082], std=[1.015,1.013,1.035], kurt=[+0.34,+1.59,+16.68],peak_times=[40],       amps=[2.0],           peak_label="early(t=40)"),
    dict(id=4,  desc="Near-hover, explosive all axes",     mean=[+0.012,+0.048,+0.000], std=[0.562,0.716,0.786], kurt=[+22.83,+10.12,+10.92],peak_times=[40],     amps=[2.0],           peak_label="early(t=40)"),
    dict(id=5,  desc="High-speed forward, spiky vx",       mean=[+0.331,-0.103,+0.069], std=[1.006,0.983,0.981], kurt=[+14.01,+0.30,-0.39], peak_times=[103],      amps=[2.0],           peak_label="mid(t=103)"),
    dict(id=6,  desc="Backward, mild, smooth",             mean=[-0.286,-0.057,-0.083], std=[0.716,0.614,0.867], kurt=[+0.25,+3.16,+2.49],  peak_times=[],         amps=[],              peak_label="none"),
    dict(id=7,  desc="Forward-starboard, spiky vx, late",  mean=[+0.398,+0.133,+0.023], std=[0.949,1.003,0.780], kurt=[+11.63,+0.04,+4.97], peak_times=[165],      amps=[2.0],           peak_label="late(t=165)"),
    dict(id=8,  desc="Port drift, two maneuvers",          mean=[+0.012,-0.296,-0.067], std=[0.803,0.880,0.900], kurt=[+3.78,+1.37,-0.04],  peak_times=[40,165],   amps=[2.0,2.0],       peak_label="two(t=40,165)"),
    dict(id=9,  desc="Slow creep, low variability, late",  mean=[-0.095,+0.171,+0.015], std=[0.567,0.680,0.678], kurt=[+1.60,-0.95,-0.21],  peak_times=[165],      amps=[2.0],           peak_label="late(t=165)"),
    dict(id=10, desc="Backward-vz, spiky vz, mid+late",    mean=[+0.029,+0.199,-0.185], std=[0.863,0.670,0.931], kurt=[-0.60,+0.21,+9.11],  peak_times=[103,165],  amps=[2.0,2.0],       peak_label="two(t=103,165)"),
    dict(id=11, desc="Lateral, very spiky, three turns",   mean=[+0.044,+0.266,-0.005], std=[0.957,0.780,1.047], kurt=[+11.77,+9.17,+34.29], peak_times=[35,103,170],amps=[2.0,1.5,2.0], peak_label="three(t=35,103,170)"),
    dict(id=12, desc="Forward, spiky vz, mid maneuver",    mean=[+0.169,-0.069,+0.078], std=[0.917,1.052,1.067], kurt=[-0.26,+1.08,+12.33], peak_times=[103],      amps=[2.0],           peak_label="mid(t=103)"),
    dict(id=13, desc="Near-hover, spiky vx, three turns",  mean=[-0.073,+0.051,-0.026], std=[0.616,0.720,0.720], kurt=[+11.63,+0.04,+4.97], peak_times=[35,103,170],amps=[2.0,1.5,2.0], peak_label="three(t=35,103,170)"),
]


# %% Generate dataset

all_signals   = []
all_peak_maps = []
all_means     = []
all_stds      = []
all_kurtoses  = []
all_traj_ids  = []

print(f"Generating {len(TRAJECTORIES)} trajectories × {N_SEEDS} seeds × {SIGNAL_LENGTH} samples...")
print()

for traj in TRAJECTORIES:
    # Scale peak_times from training length (206) to generation length (400)
    scaled_times = [int(t * SIGNAL_LENGTH / TRAIN_LENGTH) for t in traj["peak_times"]]
    pm      = make_peak_map(scaled_times, traj["amps"], signal_length=SIGNAL_LENGTH)
    mean_t  = torch.tensor([traj["mean"]],  dtype=torch.float32)  # (1, 3)
    std_t   = torch.tensor([traj["std"]],   dtype=torch.float32)
    kurt_t  = torch.tensor([traj["kurt"]],  dtype=torch.float32)
    pm_np   = pm.squeeze(0).numpy()  # (3, 400)

    for seed in range(N_SEEDS):
        sig = generate(pm, mean_t, std_t, kurt_t,
                       signal_length=SIGNAL_LENGTH, seed=seed)  # (3, 400) denormalized

        # Recompute actual stats from generated signal (matches step1 approach)
        sig_np      = sig.numpy()
        actual_mean = sig_np.mean(axis=1)           # (3,)
        actual_std  = sig_np.std(axis=1).clip(1e-8) # (3,)
        sig_norm    = (sig_np - actual_mean[:, None]) / actual_std[:, None]
        actual_kurt = compute_kurtosis(sig_norm, axis=1, fisher=True)  # (3,)

        all_signals.append(sig_np)
        all_peak_maps.append(pm_np)
        all_means.append(actual_mean)
        all_stds.append(actual_std)
        all_kurtoses.append(actual_kurt)
        all_traj_ids.append(traj["id"])

    print(f"  Traj {traj['id']:2d} — {traj['desc']:<40s}  peaks: {traj['peak_label']}")

out_path = f"{OUT_DIR}/synthetic_dataset"
np.savez(
    out_path,
    signals   = np.stack(all_signals),    # (65, 3, 400) — denormalized m/s
    peak_maps = np.stack(all_peak_maps),  # (65, 3, 400)
    means     = np.stack(all_means),      # (65, 3)
    stds      = np.stack(all_stds),       # (65, 3)
    kurtoses  = np.stack(all_kurtoses),   # (65, 3)
    traj_ids  = np.array(all_traj_ids),   # (65,)
)
print(f"\nSaved → {out_path}.npz")
print(f"Shape: {np.stack(all_signals).shape}  ({len(TRAJECTORIES)} trajs × {N_SEEDS} seeds × 3 axes × {SIGNAL_LENGTH} samples)")


# %% Visualize — one window per trajectory (vx, vy, vz)
data_gen = np.load(f"{OUT_DIR}/synthetic_dataset.npz")
gen_sigs = data_gen["signals"]    # (65, 3, 400)
gen_ids  = data_gen["traj_ids"]
vel_labels = ["vx", "vy", "vz"]

fig, axes = plt.subplots(13, 3, figsize=(18, 30), sharex=True)
for row, traj in enumerate(TRAJECTORIES):
    idx = np.where(gen_ids == traj["id"])[0][0]  # first window
    sig = gen_sigs[idx]
    for col in range(3):
        axes[row, col].plot(sig[col], linewidth=0.8, color="steelblue")
        axes[row, col].grid(True, alpha=0.3)
        if row == 0:
            axes[row, col].set_title(vel_labels[col], fontsize=10)
        if col == 0:
            axes[row, col].set_ylabel(f"T{traj['id']}", fontsize=8, rotation=0, labelpad=25)
fig.suptitle("Synthetic dataset — one window per trajectory (m/s)", fontsize=12)
plt.tight_layout()
plt.show()


# %% Visualize — peak maps for all 13 trajectories
fig, axes = plt.subplots(13, 1, figsize=(14, 20), sharex=True)
gen_pms = data_gen["peak_maps"]  # (65, 3, 400)
for row, traj in enumerate(TRAJECTORIES):
    idx = np.where(gen_ids == traj["id"])[0][0]
    pm  = gen_pms[idx].mean(axis=0)  # average across axes for display
    axes[row].plot(pm, color="darkorange", linewidth=0.9)
    axes[row].axhline(0, color="k", linewidth=0.4, alpha=0.4)
    axes[row].set_ylabel(f"T{traj['id']}", fontsize=8, rotation=0, labelpad=25)
    axes[row].set_title(f"{traj['desc']}  [{traj['peak_label']}]", fontsize=8, loc="right")
    axes[row].grid(True, alpha=0.2)
fig.suptitle("Synthetic dataset — peak maps (maneuver descriptors)", fontsize=12)
plt.tight_layout()
plt.show()


# %% Visualize — velocities (m/s) for all 13 trajectories
fig, axes = plt.subplots(13, 3, figsize=(18, 30), sharex=True)
traj_colors = plt.cm.tab20(np.linspace(0, 1, 13))
for row, traj in enumerate(TRAJECTORIES):
    idx  = np.where(gen_ids == traj["id"])[0][0]
    vel  = gen_sigs[idx]                                           # (3, 400) m/s
    for col in range(3):
        axes[row, col].plot(vel[col], linewidth=0.8, color=traj_colors[row])
        axes[row, col].grid(True, alpha=0.3)
        if row == 0:
            axes[row, col].set_title(vel_labels[col], fontsize=10)
        if col == 0:
            axes[row, col].set_ylabel(f"T{traj['id']}\n{traj['desc'][:18]}",
                                      fontsize=7, rotation=0, labelpad=70)
fig.suptitle("Synthetic dataset — velocities (m/s)", fontsize=12)
plt.tight_layout()
plt.show()


# %% Visualize — 3D trajectories (integrated positions) — all 13 overlaid
fig = plt.figure(figsize=(12, 9))
ax3d = fig.add_subplot(111, projection="3d")
dt = 1.0
for row, traj in enumerate(TRAJECTORIES):
    idx = np.where(gen_ids == traj["id"])[0][0]
    vel = gen_sigs[idx]                          # (3, 400) m/s
    x = np.cumsum(vel[0] * dt)
    y = np.cumsum(vel[1] * dt)
    z = np.cumsum(vel[2] * dt)
    ax3d.plot(x, y, z, color=traj_colors[row], linewidth=1.2, label=f"T{traj['id']}")
    ax3d.scatter(x[0], y[0], z[0], color=traj_colors[row], s=30, zorder=5)
ax3d.set_xlabel("X (∫vx dt)"); ax3d.set_ylabel("Y (∫vy dt)"); ax3d.set_zlabel("Z (∫vz dt)")
ax3d.set_title("Synthetic dataset — all 13 trajectories (integrated positions)")
ax3d.legend(fontsize=7, ncol=2, loc="upper left")
plt.tight_layout()
plt.show()


# %% Visualize — 3D trajectories — individual subplots per trajectory
fig = plt.figure(figsize=(20, 16))
for row, traj in enumerate(TRAJECTORIES):
    idx = np.where(gen_ids == traj["id"])[0][0]
    vel = gen_sigs[idx]                          # (3, 400) m/s
    x = np.cumsum(vel[0] * dt)
    y = np.cumsum(vel[1] * dt)
    z = np.cumsum(vel[2] * dt)
    ax = fig.add_subplot(4, 4, row + 1, projection="3d")
    ax.plot(x, y, z, color=traj_colors[row], linewidth=1.0)
    ax.scatter(x[0], y[0], z[0], color=traj_colors[row], s=20)
    ax.set_title(f"T{traj['id']}: {traj['desc'][:22]}\n[{traj['peak_label']}]",
                 fontsize=6.5)
    ax.tick_params(labelsize=5)
    ax.set_xlabel("X", fontsize=6); ax.set_ylabel("Y", fontsize=6); ax.set_zlabel("Z", fontsize=6)
fig.suptitle("Synthetic dataset — 3D trajectories (one window per trajectory)", fontsize=12)
plt.tight_layout()
plt.show()


# %% Condition fidelity check
# Compare what we ASKED for (conditioned mean/std/kurt) vs what the model GENERATED.
# Perfect fidelity = all points on the diagonal.
# Off-diagonal = model ignoring the condition for that axis/trajectory.

data_fid   = np.load(f"{OUT_DIR}/synthetic_dataset.npz")
actual_means = data_fid["means"]     # (65, 3) — recomputed from generated signal
actual_stds  = data_fid["stds"]      # (65, 3)
actual_kurts = data_fid["kurtoses"]  # (65, 3)
fid_ids      = data_fid["traj_ids"]  # (65,)

# Build target (conditioned) arrays aligned to the 65 signals
target_means = np.array([traj["mean"] for traj in TRAJECTORIES for _ in range(N_SEEDS)])  # (65, 3)
target_stds  = np.array([traj["std"]  for traj in TRAJECTORIES for _ in range(N_SEEDS)])  # (65, 3)
target_kurts = np.array([traj["kurt"] for traj in TRAJECTORIES for _ in range(N_SEEDS)])  # (65, 3)

ax_labels = ["vx", "vy", "vz"]

# Scatter: conditioned vs actual for mean, std, kurtosis
fig, axes = plt.subplots(3, 3, figsize=(14, 12))
for col, (tgt, act, stat_name) in enumerate([
    (target_means, actual_means, "mean"),
    (target_stds,  actual_stds,  "std"),
    (target_kurts, actual_kurts, "kurtosis"),
]):
    for row in range(3):
        ax  = axes[row, col]
        x   = tgt[:, row]
        y   = act[:, row]
        err = np.abs(y - x)
        sc  = ax.scatter(x, y, c=err, cmap="RdYlGn_r", vmin=0, vmax=err.max(), s=30, alpha=0.8)
        lo  = min(x.min(), y.min()) - 0.05
        hi  = max(x.max(), y.max()) + 0.05
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel(f"conditioned {stat_name} ({ax_labels[row]})")
        ax.set_ylabel(f"actual {stat_name} ({ax_labels[row]})")
        mae = np.mean(err)
        ax.set_title(f"{stat_name} — {ax_labels[row]}  (MAE={mae:.3f})", fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.colorbar(sc, ax=ax, label="|error|")

fig.suptitle("Condition fidelity: conditioned vs actual generated stats\n(on diagonal = model follows condition)", fontsize=11)
plt.tight_layout()
plt.show()

# Per-trajectory fidelity table
print("\nPer-trajectory condition fidelity (mean absolute error across seeds):")
fid_rows = []
for traj in TRAJECTORIES:
    mask = fid_ids == traj["id"]
    for ax_i, ax_lbl in enumerate(ax_labels):
        err_mean = np.abs(actual_means[mask, ax_i] - traj["mean"][ax_i]).mean()
        err_std  = np.abs(actual_stds[mask,  ax_i] - traj["std"][ax_i]).mean()
        err_kurt = np.abs(actual_kurts[mask, ax_i] - traj["kurt"][ax_i]).mean()
        fid_rows.append({
            "traj": traj["id"], "axis": ax_lbl,
            "MAE mean": round(err_mean, 4),
            "MAE std":  round(err_std,  4),
            "MAE kurt": round(err_kurt, 4),
        })

df_fid = pd.DataFrame(fid_rows)
print(df_fid.to_string(index=False))

# Bar chart: mean MAE per trajectory (averaged across axes)
traj_mae = df_fid.groupby("traj")[["MAE mean", "MAE std", "MAE kurt"]].mean()
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for col, stat in enumerate(["MAE mean", "MAE std", "MAE kurt"]):
    axes[col].bar(traj_mae.index, traj_mae[stat], color="steelblue", alpha=0.8)
    axes[col].set_xlabel("Trajectory ID")
    axes[col].set_ylabel("MAE")
    axes[col].set_title(f"Condition fidelity — {stat} (avg across axes)")
    axes[col].grid(True, axis="y", alpha=0.3)
plt.suptitle("Condition fidelity per trajectory — lower = model follows condition better", fontsize=10)
plt.tight_layout()
plt.show()

# %% Diversity check — 5 seeds per trajectory for traj 5 and 11
fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
colors = ["steelblue", "darkorange", "green", "purple", "brown"]
for row, tid in enumerate([5, 11]):
    mask = np.where(gen_ids == tid)[0]
    for seed_i, idx in enumerate(mask):
        for col in range(3):
            axes[row, col].plot(gen_sigs[idx, col], color=colors[seed_i],
                                linewidth=0.8, alpha=0.8, label=f"seed {seed_i}")
    for col in range(3):
        axes[row, col].set_ylabel(vel_labels[col])
        axes[row, col].grid(True, alpha=0.3)
        if row == 0:
            axes[row, col].legend(fontsize=7)
    axes[row, 0].set_title(f"Traj {tid} — {TRAJECTORIES[tid-1]['desc']}", loc="left", fontsize=9)
fig.suptitle("Diversity check: 5 seeds for two trajectories (400 samples each)")
plt.tight_layout()
plt.show()
