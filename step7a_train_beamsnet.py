# %% [markdown]
# # Step 7a — Train BeamsNetV2 on A-KIT, Leave-2-Out Evaluation
# Two runs per config:
#   Baseline  — train on A-KIT only (11 trajs), test on held-out 2
#   Augmented — train on A-KIT + synthetic (excluding traj_ids matching test set)

# %% Imports & paths
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from numpy import linalg as LA
%matplotlib inline

torch.manual_seed(0)
np.random.seed(0)

BEAMS_PATH     = "/Users/michal/Desktop/PhD/dvl paper/DATA/AKIT_beams_dataset.npz"
DATA_PATH      = "/Users/michal/Desktop/PhD/dvl paper/A-KIT-main/Data"
SYN_BEAMS_PATH = "/Users/michal/Desktop/PhD/dvl paper/GENERATED DATA/synthetic_beams_dataset.npz"
SYN_GT_PATH    = "/Users/michal/Desktop/PhD/dvl paper/GENERATED DATA/synthetic_dataset.npz"
SAVE_PATH      = "/Users/michal/Desktop/PhD/dvl paper/BeamsNetV2_AKIT.pkl"

N_TRAJ = 13

# Each entry is a pair of trajectory IDs held out for testing
TEST_CONFIGS = [
    (1, 13),
]

T          = 3
BATCH_SIZE = 4
EPOCHS     = 150
LR         = 1e-3

# %% Beam geometry and LS pseudoinverse (BeamsNet geometry, no mounting correction)

def make_beam_matrix():
    rows = []
    for k in range(4):
        psi   = (45 + k * 90) * np.pi / 180
        alpha = 20 * np.pi / 180
        rows.append([np.cos(psi) * np.sin(alpha),
                     np.sin(psi) * np.sin(alpha),
                     np.cos(alpha)])
    return np.array(rows)   # (4, 3)

A     = make_beam_matrix()
P_INV = np.linalg.lstsq(A, np.eye(4), rcond=None)[0]   # (3, 4)

# %% BeamsNetV2 model (exact original architecture)

class BeamsNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv1d(in_channels=T, out_channels=6, kernel_size=2, stride=1),
            nn.Tanh(),
        )
        self.FC_ConvToFc = nn.Sequential(
            nn.Linear(18, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
            nn.ReLU(),
        )
        self.FC_output = nn.Sequential(nn.Linear(6, 3))

    def forward(self, x, y):
        """x: (B, 4) current beam  |  y: (B, T, 4) past beams"""
        y = self.conv_layer(y)
        y = torch.flatten(y, 1)
        y = self.FC_ConvToFc(y)
        x = torch.column_stack((x, y))
        return self.FC_output(x)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)

# %% Metric helpers

def rmse(true, pred):
    return np.sqrt(mean_squared_error(LA.norm(true, axis=1), LA.norm(pred, axis=1)))

def mae(true, pred):
    return np.mean(np.abs(LA.norm(pred, axis=1) - LA.norm(true, axis=1)))

def nse(true, pred):
    t = LA.norm(true, axis=1)
    p = LA.norm(pred, axis=1)
    return 1 - np.sum((p - t) ** 2) / np.sum((t - np.mean(t)) ** 2)

def vaf(true, pred):
    t = LA.norm(true, axis=1)
    p = LA.norm(pred, axis=1)
    return (1 - np.var(t - p) / np.var(t)) * 100

# %% Window builder

def make_windows(beams_noisy, gt_body):
    """
    beams_noisy : (4, N) — noisy beam observations
    gt_body     : (3, N) — body-frame GT velocity
    Returns X (N-T, 4), Y (N-T, T, 4), Z (N-T, 3)  all float32
    """
    N = beams_noisy.shape[1]
    X = np.zeros((N - T, 4),    dtype=np.float32)
    Y = np.zeros((N - T, T, 4), dtype=np.float32)
    Z = np.zeros((N - T, 3),    dtype=np.float32)
    for t in range(N - T):
        X[t] = beams_noisy[:, t + T]
        Y[t] = beams_noisy[:, t:t + T].T    # (T, 4)
        Z[t] = gt_body[:, t + T]
    return X, Y, Z

# %% Load all A-KIT data once

data_beams   = np.load(BEAMS_PATH)
akit_windows = []
for i in range(N_TRAJ):
    csv = pd.read_csv(f"{DATA_PATH}/Trajectory{i+1}/DVL_trajectory{i+1}.csv")
    vx, vy, vz = csv.iloc[:, 1].values, csv.iloc[:, 2].values, csv.iloc[:, 3].values
    gt_body = np.stack([vx, vy, vz], axis=0)
    X, Y, Z = make_windows(data_beams[f"traj{i+1}_beams_noisy"], gt_body)
    akit_windows.append({"traj_id": i + 1, "X": X, "Y": Y, "Z": Z})

# %% Load synthetic data once

syn_data     = np.load(SYN_BEAMS_PATH)
syn_gt       = np.load(SYN_GT_PATH)
_syn_beams   = syn_data["beams_noisy"]   # (65, 4, N)
_syn_signals = syn_gt["signals"]         # (65, 3, N)
_syn_ids     = syn_gt["traj_ids"]        # (65,)

syn_windows = []
for i in range(len(_syn_beams)):
    X, Y, Z = make_windows(_syn_beams[i], _syn_signals[i])
    syn_windows.append({"traj_id": int(_syn_ids[i]), "X": X, "Y": Y, "Z": Z})

print(f"Synthetic windows loaded: {len(syn_windows)} signals, "
      f"{sum(len(d['X']) for d in syn_windows):,} total windows")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def make_loader(X, Y, Z, shuffle=True):
    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(Z),
    )
    return torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

def train_and_eval(test_traj_ids, aug_windows=None):
    """
    Train on A-KIT minus test_traj_ids (+ optional aug_windows), evaluate on test_traj_ids.
    Validation is always pure A-KIT so val loss is comparable across runs.
    """
    test_set  = [d for d in akit_windows if d["traj_id"] in test_traj_ids]
    train_set = [d for d in akit_windows if d["traj_id"] not in test_traj_ids]

    akit_X = np.concatenate([d["X"] for d in train_set])
    akit_Y = np.concatenate([d["Y"] for d in train_set])
    akit_Z = np.concatenate([d["Z"] for d in train_set])

    N_all   = len(akit_X)
    N_val   = N_all // 4
    N_train = N_all - N_val

    # Val always comes from A-KIT only
    val_X, val_Y, val_Z = akit_X[N_train:], akit_Y[N_train:], akit_Z[N_train:]

    # Training: A-KIT train split + optional synthetic augmentation
    if aug_windows:
        aug_X = np.concatenate([d["X"] for d in aug_windows])
        aug_Y = np.concatenate([d["Y"] for d in aug_windows])
        aug_Z = np.concatenate([d["Z"] for d in aug_windows])
        train_X = np.concatenate([akit_X[:N_train], aug_X])
        train_Y = np.concatenate([akit_Y[:N_train], aug_Y])
        train_Z = np.concatenate([akit_Z[:N_train], aug_Z])
    else:
        train_X, train_Y, train_Z = akit_X[:N_train], akit_Y[:N_train], akit_Z[:N_train]

    train_loader = make_loader(train_X, train_Y, train_Z, shuffle=True)
    val_loader   = make_loader(val_X,   val_Y,   val_Z,   shuffle=False)

    model = BeamsNetV2().to(device)
    model.initialize_weights()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val, best_state = float("inf"), None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb, zb in train_loader:
            xb, yb, zb = xb.to(device), yb.to(device), zb.to(device)
            optimizer.zero_grad()
            criterion(model(xb, yb), zb).backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb, zb in val_loader:
                xb, yb, zb = xb.to(device), yb.to(device), zb.to(device)
                val_loss += criterion(model(xb, yb), zb).item() * len(xb)
        val_loss /= N_val
        if val_loss < best_val:
            best_val  = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()

    results = []
    with torch.no_grad():
        for d in test_set:
            X_t = torch.from_numpy(d["X"]).to(device)
            Y_t = torch.from_numpy(d["Y"]).to(device)
            pred_bn = model(X_t, Y_t).cpu().numpy()
            pred_ls = (P_INV @ d["X"].T).T
            results.append({"traj_id": d["traj_id"], "gt": d["Z"],
                            "beamsnet": pred_bn, "ls": pred_ls})
    return results

# %% Run leave-2-out — baseline vs augmented

summary_rows = []

for test_pair in TEST_CONFIGS:
    test_ids = set(test_pair)
    aug      = [d for d in syn_windows if d["traj_id"] not in test_ids]

    print(f"\n{'='*60}")
    print(f"Test trajs : {sorted(test_ids)}")
    print(f"Train trajs: {sorted(set(range(1, N_TRAJ+1)) - test_ids)}")
    print(f"Synthetic  : {len(aug)} signals ({sum(len(d['X']) for d in aug):,} windows) "
          f"[traj_ids {sorted(test_ids)} excluded]")

    # ── Baseline: A-KIT only ──────────────────────────────────────
    print("\n  [Baseline — A-KIT only]")
    results_base = train_and_eval(test_ids, aug_windows=None)
    gt_b  = np.concatenate([r["gt"]       for r in results_base])
    bn_b  = np.concatenate([r["beamsnet"] for r in results_base])
    ls_b  = np.concatenate([r["ls"]       for r in results_base])
    rmse_bn_base = rmse(gt_b, bn_b)
    rmse_ls_base = rmse(gt_b, ls_b)
    improv_base  = (rmse_ls_base - rmse_bn_base) / rmse_ls_base * 100
    for r in results_base:
        print(f"    Traj {r['traj_id']}: BN={rmse(r['gt'], r['beamsnet']):.4f}  LS={rmse(r['gt'], r['ls']):.4f}")
    print(f"    → Improvement: {improv_base:.2f}%")

    # ── Augmented: A-KIT + synthetic ─────────────────────────────
    print("\n  [Augmented — A-KIT + synthetic]")
    results_aug = train_and_eval(test_ids, aug_windows=aug)
    gt_a  = np.concatenate([r["gt"]       for r in results_aug])
    bn_a  = np.concatenate([r["beamsnet"] for r in results_aug])
    ls_a  = np.concatenate([r["ls"]       for r in results_aug])
    rmse_bn_aug = rmse(gt_a, bn_a)
    rmse_ls_aug = rmse(gt_a, ls_a)
    improv_aug  = (rmse_ls_aug - rmse_bn_aug) / rmse_ls_aug * 100
    for r in results_aug:
        print(f"    Traj {r['traj_id']}: BN={rmse(r['gt'], r['beamsnet']):.4f}  LS={rmse(r['gt'], r['ls']):.4f}")
    print(f"    → Improvement: {improv_aug:.2f}%")

    delta = improv_aug - improv_base
    print(f"\n  Δ improvement from augmentation: {delta:+.2f}pp")

    summary_rows.append({
        "Test trajs":       str(sorted(test_ids)),
        "LS RMSE":          round(rmse_ls_base, 4),
        "BN RMSE (base)":   round(rmse_bn_base, 4),
        "Improv base (%)":  round(improv_base,  2),
        "BN RMSE (aug)":    round(rmse_bn_aug,  4),
        "Improv aug (%)":   round(improv_aug,   2),
        "Δ (pp)":           round(delta,        2),
    })

# %% Summary

print(f"\n{'='*60}")
print("Summary:")
print(pd.DataFrame(summary_rows).to_string(index=False))

# ============================================================
# %% [markdown]
# ## Synthetic data diagnostic
# Train on ALL A-KIT (13 trajs), test on 65 synthetic signals.
# Per-signal improvement ranking + per-axis RMSE breakdown
# reveals which synthetic signals are closest to real A-KIT
# and which velocity axis drives the domain gap.
# ============================================================

# %% Train on all A-KIT

print("\nTraining on all A-KIT trajectories for synthetic diagnostic...")
all_X = np.concatenate([d["X"] for d in akit_windows])
all_Y = np.concatenate([d["Y"] for d in akit_windows])
all_Z = np.concatenate([d["Z"] for d in akit_windows])

N_all   = len(all_X)
N_val   = N_all // 4
N_train = N_all - N_val

train_loader = make_loader(all_X[:N_train], all_Y[:N_train], all_Z[:N_train], shuffle=True)
val_loader   = make_loader(all_X[N_train:], all_Y[N_train:], all_Z[N_train:], shuffle=False)

diag_model = BeamsNetV2().to(device)
diag_model.initialize_weights()
criterion  = nn.MSELoss()
optimizer  = torch.optim.Adam(diag_model.parameters(), lr=LR)

best_val, best_state = float("inf"), None
for epoch in range(1, EPOCHS + 1):
    diag_model.train()
    for xb, yb, zb in train_loader:
        xb, yb, zb = xb.to(device), yb.to(device), zb.to(device)
        optimizer.zero_grad()
        criterion(diag_model(xb, yb), zb).backward()
        optimizer.step()

    diag_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb, zb in val_loader:
            xb, yb, zb = xb.to(device), yb.to(device), zb.to(device)
            val_loss += criterion(diag_model(xb, yb), zb).item() * len(xb)
    val_loss /= N_val
    if val_loss < best_val:
        best_val   = val_loss
        best_state = {k: v.clone() for k, v in diag_model.state_dict().items()}

diag_model.load_state_dict(best_state)
diag_model.eval()
print(f"Done. Best val loss: {best_val:.6f}")

# %% Run on all synthetic signals

N_SYN = len(syn_windows)

syn_results = []
with torch.no_grad():
    for i, d in enumerate(syn_windows):
        X, Y, Z = d["X"], d["Y"], d["Z"]
        pred_bn = diag_model(torch.from_numpy(X).to(device),
                             torch.from_numpy(Y).to(device)).cpu().numpy()
        pred_ls = (P_INV @ X.T).T
        syn_results.append({
            "signal_idx": i,
            "traj_id":    d["traj_id"],
            "gt": Z, "beamsnet": pred_bn, "ls": pred_ls,
        })

# %% Per-axis RMSE helper

def rmse_ax(true, pred):
    """Per-axis RMSE — returns (3,) array for vx, vy, vz."""
    return np.sqrt(np.mean((true - pred) ** 2, axis=0))

# %% Per-signal improvement table (sorted best → worst)

rows = []
for r in syn_results:
    gt, bn, ls = r["gt"], r["beamsnet"], r["ls"]
    improv = (rmse(gt, ls) - rmse(gt, bn)) / rmse(gt, ls) * 100
    ax_bn  = rmse_ax(gt, bn)
    ax_ls  = rmse_ax(gt, ls)
    rows.append({
        "sig":      r["signal_idx"],
        "traj":     r["traj_id"],
        "improv":   round(improv, 2),
        "BN|vx":    round(ax_bn[0], 4),
        "BN|vy":    round(ax_bn[1], 4),
        "BN|vz":    round(ax_bn[2], 4),
        "LS|vx":    round(ax_ls[0], 4),
        "LS|vy":    round(ax_ls[1], 4),
        "LS|vz":    round(ax_ls[2], 4),
    })

df_syn = pd.DataFrame(rows).sort_values("improv", ascending=False)
print("\nPer-signal improvement (best → worst):")
print(df_syn.to_string(index=False))

# %% Per-axis aggregate summary

all_gt = np.concatenate([r["gt"]       for r in syn_results])
all_bn = np.concatenate([r["beamsnet"] for r in syn_results])
all_ls = np.concatenate([r["ls"]       for r in syn_results])

ax_bn_all = rmse_ax(all_gt, all_bn)
ax_ls_all = rmse_ax(all_gt, all_ls)
ax_improv = (ax_ls_all - ax_bn_all) / ax_ls_all * 100

print("\nPer-axis aggregate (all 65 synthetic signals):")
df_ax = pd.DataFrame({
    "axis":    ["vx", "vy", "vz"],
    "BN RMSE": ax_bn_all.round(4),
    "LS RMSE": ax_ls_all.round(4),
    "improv %": ax_improv.round(2),
})
print(df_ax.to_string(index=False))

overall_improv = (rmse(all_gt, all_ls) - rmse(all_gt, all_bn)) / rmse(all_gt, all_ls) * 100
print(f"\nOverall improvement: {overall_improv:.2f}%")

# %% Plot — per-axis RMSE: BN vs LS across all synthetic signals

fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
x = np.arange(N_SYN)
labels = ["vx", "vy", "vz"]
for j in range(3):
    bn_j = [rmse_ax(r["gt"], r["beamsnet"])[j] for r in syn_results]
    ls_j = [rmse_ax(r["gt"], r["ls"])[j]       for r in syn_results]
    axes[j].plot(x, bn_j, label="BeamsNetV2", color="steelblue", linewidth=0.9, marker="o", markersize=3)
    axes[j].plot(x, ls_j, label="LS",         color="tomato",    linewidth=0.9, marker="o", markersize=3)
    axes[j].set_ylabel(f"RMSE {labels[j]}")
    axes[j].legend(fontsize=8)
    axes[j].grid(True, alpha=0.3)
axes[0].set_title("Per-axis RMSE across 65 synthetic signals (A-KIT-trained model)")
axes[2].set_xlabel("Signal index (sorted by traj ID)")
plt.tight_layout()
plt.show()

# %% Plot — improvement distribution

improvs = df_syn["improv"].values
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(np.arange(N_SYN), df_syn["improv"].values, color=["steelblue" if v > 0 else "tomato" for v in improvs], alpha=0.8)
ax.axhline(0, color="black", linewidth=0.8)
ax.axhline(overall_improv, color="navy", linewidth=1.2, linestyle="--", label=f"mean={overall_improv:.1f}%")
ax.set_xlabel("Signal (sorted best → worst)")
ax.set_ylabel("Improvement over LS (%)")
ax.set_title("Per-signal improvement — A-KIT model on synthetic data")
ax.legend()
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# %%
