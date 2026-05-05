import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH    = "/Users/michal/Desktop/PhD/dvl paper/A-KIT-main/Data"
DATASET_PATH = "/Users/michal/Desktop/PhD/dvl paper/DATA/dvl_dataset.npz"
WINDOW_SIZE  = 206
STRIDE       = 50

data     = np.load(DATASET_PATH)
signals  = data["signals"]    # (W, 3, N) — normalized
traj_ids = data["traj_ids"]   # (W,)

N = signals.shape[-1]
vel_labels = ["vx", "vy", "vz"]
dot_labels = ["vx·vy", "vx·vz", "vy·vz"]

def dot_products(sig):
    """sig: (3, N) normalized → dot products (3, N)"""
    return np.stack([sig[0]*sig[1], sig[0]*sig[2], sig[1]*sig[2]], axis=0)


# ── Plot 1: a few hand-picked windows to see dot product vs signal ────────────

sample_indices = []
for traj in [1, 3, 7, 11]:
    mask = np.where(traj_ids == traj)[0]
    if len(mask):
        sample_indices.append(mask[len(mask) // 2])

fig, axes = plt.subplots(len(sample_indices), 6, figsize=(24, 3 * len(sample_indices)), sharex=True)
if len(sample_indices) == 1:
    axes = axes[np.newaxis, :]

for row, idx in enumerate(sample_indices):
    sig  = signals[idx]       # (3, N)
    dots = dot_products(sig)  # (3, N)
    for col in range(3):
        axes[row, col].plot(sig[col], color="steelblue", linewidth=0.9)
        axes[row, col].axhline(0, color="k", linewidth=0.4, alpha=0.4)
        axes[row, col].grid(True, alpha=0.3)
        if row == 0:
            axes[row, col].set_title(vel_labels[col])
        if col == 0:
            axes[row, col].set_ylabel(f"traj {traj_ids[idx]}\nwin {idx}")
    for col in range(3):
        axes[row, col + 3].plot(dots[col], color="darkorange", linewidth=0.9)
        axes[row, col + 3].axhline(0, color="k", linewidth=0.4, alpha=0.4)
        axes[row, col + 3].grid(True, alpha=0.3)
        if row == 0:
            axes[row, col + 3].set_title(dot_labels[col])

fig.suptitle("Normalized signals (blue) vs pairwise dot products (orange)")
plt.tight_layout()
plt.show()


# ── Plot 2: distribution of mean dot products across all windows ──────────────

mean_dots = np.stack([dot_products(signals[i]).mean(axis=1) for i in range(len(signals))])
# mean_dots: (W, 3)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for col, label in enumerate(dot_labels):
    axes[col].hist(mean_dots[:, col], bins=40, color="darkorange", alpha=0.8, edgecolor="k", linewidth=0.4)
    axes[col].axvline(0, color="k", linewidth=1, linestyle="--")
    axes[col].set_xlabel(f"mean({label})")
    axes[col].set_ylabel("count")
    axes[col].set_title(label)
    axes[col].grid(True, alpha=0.3)
fig.suptitle("Distribution of mean dot products across all windows")
plt.tight_layout()
plt.show()


# ── Plot 3: compare windows with extreme vs near-zero mean dot product ────────

fig, axes = plt.subplots(2, 6, figsize=(24, 6), sharex=True)
col_idx = 0  # look at vx·vy
vals = mean_dots[:, col_idx]
high_idx = np.argsort(vals)[-1]
low_idx  = np.argsort(vals)[0]
zero_idx = np.argsort(np.abs(vals))[0]

for row_i, (idx, row_label) in enumerate([(high_idx, f"max vx·vy = {vals[high_idx]:.3f}  (traj {traj_ids[high_idx]})"),
                                           (zero_idx, f"≈0 vx·vy = {vals[zero_idx]:.3f}  (traj {traj_ids[zero_idx]})")]):
    sig  = signals[idx]
    dots = dot_products(sig)
    for col in range(3):
        axes[row_i, col].plot(sig[col], color="steelblue", linewidth=0.9)
        axes[row_i, col].axhline(0, color="k", linewidth=0.4, alpha=0.4)
        axes[row_i, col].grid(True, alpha=0.3)
        if row_i == 0:
            axes[row_i, col].set_title(vel_labels[col])
        if col == 0:
            axes[row_i, col].set_ylabel(row_label, fontsize=8)
    for col in range(3):
        axes[row_i, col + 3].plot(dots[col], color="darkorange", linewidth=0.9)
        axes[row_i, col + 3].axhline(0, color="k", linewidth=0.4, alpha=0.4)
        axes[row_i, col + 3].grid(True, alpha=0.3)
        if row_i == 0:
            axes[row_i, col + 3].set_title(dot_labels[col])

fig.suptitle("Extreme vs near-zero vx·vy windows")
plt.tight_layout()
plt.show()


# ── Plot 4: full raw trajectories with highlighted windows ────────────────────

def load_raw(traj_id):
    path = f"{DATA_PATH}/Trajectory{traj_id}/DVL_trajectory{traj_id}.csv"
    df   = pd.read_csv(path)
    time = df.iloc[:, 0].values
    vx   = df.iloc[:, 1].values
    vy   = df.iloc[:, 2].values
    vz   = df.iloc[:, 3].values
    return time, np.stack([vx, vy, vz], axis=0)  # (3, N_raw)

def window_start(win_idx, traj_id):
    """Sample index where this window starts within its trajectory."""
    first = np.where(traj_ids == traj_id)[0][0]
    return (win_idx - first) * STRIDE

# collect (window_index, traj_id) pairs to highlight
highlight = {}
for idx in sample_indices:
    tid = traj_ids[idx]
    highlight.setdefault(tid, []).append(idx)
for idx in [high_idx, zero_idx]:
    tid = traj_ids[idx]
    highlight.setdefault(tid, []).append(idx)

traj_list = sorted(highlight.keys())
fig, axes = plt.subplots(len(traj_list), 3, figsize=(20, 3 * len(traj_list)), sharex=False)
if len(traj_list) == 1:
    axes = axes[np.newaxis, :]

highlight_colors = ["tab:orange", "tab:green", "tab:red", "tab:purple"]

for row, tid in enumerate(traj_list):
    time, raw = load_raw(tid)
    for col in range(3):
        axes[row, col].plot(time, raw[col], color="steelblue", linewidth=0.7, alpha=0.8)
        axes[row, col].axhline(0, color="k", linewidth=0.4, alpha=0.3)
        axes[row, col].grid(True, alpha=0.3)
        if row == 0:
            axes[row, col].set_title(vel_labels[col])
        if col == 0:
            axes[row, col].set_ylabel(f"traj {tid}")
        for k, win_idx in enumerate(highlight[tid]):
            start = window_start(win_idx, tid)
            t0    = time[start] if start < len(time) else time[-1]
            end   = min(start + WINDOW_SIZE, len(time)) - 1
            t1    = time[end]
            color = highlight_colors[k % len(highlight_colors)]
            axes[row, col].axvspan(t0, t1, alpha=0.25, color=color, label=f"win {win_idx}" if col == 0 else None)
    axes[row, 0].legend(fontsize=7, loc="upper left")

fig.suptitle("Raw trajectories — highlighted windows used in the plots above")
plt.tight_layout()
plt.show()
