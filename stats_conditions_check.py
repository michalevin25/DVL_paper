# %% [markdown]
# # Stats Conditions Check
# Are the conditioning variables (mean, std, kurtosis) correlated with each other?
# We compute stats on 50-sample non-overlapping windows of the raw DVL signals
# (before any normalization) across all 13 trajectories.

# %%
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kurtosis as compute_kurtosis, pearsonr
from scipy.interpolate import UnivariateSpline

DATA_PATH   = "/Users/michal/Desktop/PhD/dvl paper/A-KIT-main/Data"
WINDOW_SIZE = 50
STRIDE      = 50   # non-overlapping
N_TRAJ      = 13


# %% Load raw signals

def load_signals():
    signals, times = [], []
    for i in range(1, N_TRAJ + 1):
        path = f"{DATA_PATH}/Trajectory{i}/DVL_trajectory{i}.csv"
        data = pd.read_csv(path)
        times.append(data.iloc[:, 0].values)
        vx, vy, vz = data.iloc[:, 1].values, data.iloc[:, 2].values, data.iloc[:, 3].values
        signals.append(np.stack([vx, vy, vz], axis=0))  # (3, N)
    return signals, times

signals, times = load_signals()
print(f"Loaded {len(signals)} trajectories")
for i, s in enumerate(signals):
    print(f"  Traj {i+1:2d}: {s.shape[1]} samples  |  "
          f"vx [{s[0].min():.3f}, {s[0].max():.3f}]  "
          f"vy [{s[1].min():.3f}, {s[1].max():.3f}]  "
          f"vz [{s[2].min():.3f}, {s[2].max():.3f}]")


# %% Compute per-window stats

records = []
for traj_idx, sig in enumerate(signals):
    N = sig.shape[1]
    for start in range(0, N - WINDOW_SIZE + 1, STRIDE):
        end   = start + WINDOW_SIZE
        w     = sig[:, start:end]          # (3, 50) — raw, not normalized
        mean  = w.mean(axis=1)             # (3,)
        std   = w.std(axis=1).clip(1e-8)   # (3,)
        w_norm = (w - mean[:, None]) / std[:, None]
        kurt  = compute_kurtosis(w_norm, axis=1, fisher=True)  # (3,)
        records.append({
            "traj":    traj_idx + 1,
            "win_start": start,
            "mean_vx": mean[0], "mean_vy": mean[1], "mean_vz": mean[2],
            "std_vx":  std[0],  "std_vy":  std[1],  "std_vz":  std[2],
            "kurt_vx": kurt[0], "kurt_vy": kurt[1], "kurt_vz": kurt[2],
        })

df = pd.DataFrame(records)
print(f"\n{len(df)} windows total ({WINDOW_SIZE}-sample, stride={STRIDE})")
print(df.describe().round(3))


# %% Correlation heatmap — all 9 stats

stat_cols = ["mean_vx","mean_vy","mean_vz","std_vx","std_vy","std_vz","kurt_vx","kurt_vy","kurt_vz"]
corr = df[stat_cols].corr()

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
plt.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
ax.set_xticks(range(len(stat_cols))); ax.set_xticklabels(stat_cols, rotation=45, ha="right")
ax.set_yticks(range(len(stat_cols))); ax.set_yticklabels(stat_cols)
for i in range(len(stat_cols)):
    for j in range(len(stat_cols)):
        ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center",
                fontsize=7, color="black" if abs(corr.values[i,j]) < 0.7 else "white")
ax.set_title(f"Pearson correlation between conditioning stats\n"
             f"({len(df)} windows × {WINDOW_SIZE} samples, all 13 trajectories)")
plt.tight_layout()
plt.show()


# %% Scatter matrix — mean vs std vs kurtosis, per axis

axes_labels = ["vx", "vy", "vz"]
pair_labels = [("mean", "std"), ("mean", "kurt"), ("std", "kurt")]

fig, axes = plt.subplots(3, 3, figsize=(14, 12))
colors = plt.cm.tab10(np.linspace(0, 1, N_TRAJ))

for col, ax_name in enumerate(axes_labels):
    for row, (xstat, ystat) in enumerate(pair_labels):
        ax = axes[row, col]
        xcol = f"{xstat}_{ax_name}"
        ycol = f"{ystat}_{ax_name}"
        for tid in range(1, N_TRAJ + 1):
            sub = df[df["traj"] == tid]
            ax.scatter(sub[xcol], sub[ycol], color=colors[tid-1], s=18, alpha=0.7,
                       label=f"T{tid}" if col == 0 and row == 0 else None)
        r, p = pearsonr(df[xcol], df[ycol])
        ax.set_xlabel(f"{xstat} ({ax_name})", fontsize=9)
        ax.set_ylabel(f"{ystat} ({ax_name})", fontsize=9)
        ax.set_title(f"{ax_name}: {xstat} vs {ystat}  (r={r:+.2f})", fontsize=9)
        ax.grid(True, alpha=0.3)

axes[0, 0].legend(fontsize=6, ncol=2, loc="upper right")
fig.suptitle(f"Pairwise scatter: mean / std / kurtosis per axis\n"
             f"({len(df)} windows, {WINDOW_SIZE} samples each, colored by trajectory)",
             fontsize=11)
plt.tight_layout()
plt.show()


# %% Cross-axis correlations — does vx std predict vy std?

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
cross_pairs = [
    ("std_vx",  "std_vy",  "std: vx vs vy"),
    ("std_vx",  "std_vz",  "std: vx vs vz"),
    ("std_vy",  "std_vz",  "std: vy vs vz"),
    ("kurt_vx", "kurt_vy", "kurt: vx vs vy"),
    ("kurt_vx", "kurt_vz", "kurt: vx vs vz"),
    ("kurt_vy", "kurt_vz", "kurt: vy vs vz"),
]
for ax, (xcol, ycol, title) in zip(axes.flat, cross_pairs):
    for tid in range(1, N_TRAJ + 1):
        sub = df[df["traj"] == tid]
        ax.scatter(sub[xcol], sub[ycol], color=colors[tid-1], s=18, alpha=0.7)
    r, _ = pearsonr(df[xcol], df[ycol])
    ax.set_xlabel(xcol, fontsize=9)
    ax.set_ylabel(ycol, fontsize=9)
    ax.set_title(f"{title}  (r={r:+.2f})", fontsize=9)
    ax.grid(True, alpha=0.3)
fig.suptitle("Cross-axis correlations: do axes behave together?", fontsize=11)
plt.tight_layout()
plt.show()


# %% Summary table — strongest correlations

print("=== Top correlations (|r| > 0.3) ===")
corr_flat = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
corr_flat = corr_flat.reindex(corr_flat.abs().sort_values(ascending=False).index)
for (a, b), r in corr_flat.items():
    if abs(r) > 0.3:
        strength = "strong" if abs(r) > 0.6 else "moderate"
        print(f"  {a:12s} ↔ {b:12s}  r = {r:+.3f}  ({strength})")
