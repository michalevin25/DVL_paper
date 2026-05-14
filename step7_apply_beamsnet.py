# %% [markdown]
# # Apply BeamsNet to A-KIT Beam Data (Zero-Shot)
# Loads AKIT_beams_dataset.npz (step6) and applies the pre-trained BeamsNetV2
# to each trajectory with no retraining.
# Beams are computed as H @ v_body (no mounting rotation), matching BeamsNet's
# training assumption. GT is body-frame velocity directly.
# Trajectory 12 is excluded (degenerate velocity profile).

# %% Imports & paths
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from numpy import linalg as LA
%matplotlib inline

torch.manual_seed(0)
np.random.seed(0)

BEAMS_PATH  = "/Users/michal/Desktop/PhD/dvl paper/DATA/AKIT_beams_dataset.npz"
MODEL_PATH  = "/Users/michal/Desktop/PhD/dvl paper/BeamsNet-main/code/BeamsNetV2.pkl"
DATA_PATH   = "/Users/michal/Desktop/PhD/dvl paper/A-KIT-main/Data"
N_TRAJ      = 13
T           = 3        # history window — must match BeamsNetV2 training
SKIP_TRAJS  = {12}     # excluded from evaluation

# %% B-matrix and LS pseudoinverse (BeamsNet geometry — no mounting correction)
# This is the same simple A matrix BeamsNet uses internally.

def make_beam_matrix():
    rows = []
    for k in range(4):
        psi = (45 + k * 90) * np.pi / 180
        alpha = 20 * np.pi / 180
        rows.append([np.cos(psi) * np.sin(alpha),
                     np.sin(psi) * np.sin(alpha),
                     np.cos(alpha)])
    return np.array(rows)  # (4, 3)

A     = make_beam_matrix()                           # (4, 3)
P_INV = np.linalg.lstsq(A, np.eye(4), rcond=None)[0]  # (3, 4)  LS pseudo-inverse

# %% BeamsNetV2 model definition (exact copy from BeamsNet-main/code/BeamsNetV2_Test.py)

class BeamsNetV2(nn.Module):
    def __init__(self):
        super(BeamsNetV2, self).__init__()
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
        self.FC_output = nn.Sequential(
            nn.Linear(6, 3),
        )

    def forward(self, x, y):
        y = self.conv_layer(y)
        y = torch.flatten(y, 1)
        y = self.FC_ConvToFc(y)
        x = torch.column_stack((x, y))
        x = self.FC_output(x)
        return x

# %% Load model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model  = BeamsNetV2()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.to(device)
model.eval()
print(f"BeamsNetV2 loaded on {device}")

# %% Load beam dataset and GT body velocities

data_beams      = np.load(BEAMS_PATH)
beams_list      = [data_beams[f"traj{i+1}_beams"]       for i in range(N_TRAJ)]  # (4, N) clean
noisy_beams_list = [data_beams[f"traj{i+1}_beams_noisy"] for i in range(N_TRAJ)]  # (4, N) with error model

gt_body_list = []
for i in range(N_TRAJ):
    csv = pd.read_csv(f"{DATA_PATH}/Trajectory{i+1}/DVL_trajectory{i+1}.csv")
    vx, vy, vz = csv.iloc[:, 1].values, csv.iloc[:, 2].values, csv.iloc[:, 3].values
    gt_body_list.append(np.stack([vx, vy, vz], axis=0))  # (3, N) body frame

print("Loaded data:")
for i in range(N_TRAJ):
    tag = " [SKIP]" if (i + 1) in SKIP_TRAJS else ""
    print(f"  Traj {i+1}: beams {beams_list[i].shape}  noisy {noisy_beams_list[i].shape}  gt {gt_body_list[i].shape}{tag}")

# %% Build per-trajectory inference inputs

def make_windows(noisy_beams, gt_body):
    """
    noisy_beams : (4, N)  — error-model beams, used as model input
    gt_body     : (3, N)  — body-frame GT, used for evaluation only
    Returns X (N-T, 4), Y (N-T, T, 4), Z (N-T, 3)
    """
    N = noisy_beams.shape[1]
    X = np.zeros((N - T, 4),    dtype=np.float32)
    Y = np.zeros((N - T, T, 4), dtype=np.float32)
    Z = np.zeros((N - T, 3),    dtype=np.float32)
    for t in range(N - T):
        X[t] = noisy_beams[:, t + T]          # current noisy beam
        Y[t] = noisy_beams[:, t:t + T].T      # (T, 4) noisy history
        Z[t] = gt_body[:, t + T]              # body-frame GT at t+T
    return X, Y, Z

# %% Metrics helpers (same as BeamsNet original)

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

# %% Run BeamsNet and LS on all trajectories

results = []

with torch.no_grad():
    for i in range(N_TRAJ):
        if (i + 1) in SKIP_TRAJS:
            continue
        X, Y_win, Z = make_windows(noisy_beams_list[i], gt_body_list[i])

        X_t = torch.from_numpy(X).to(device)         # (M, 4)
        Y_t = torch.from_numpy(Y_win).to(device)     # (M, T, 4)

        pred_bn = model(X_t, Y_t).cpu().numpy()      # (M, 3)

        # Least-squares baseline: P_INV @ current_beam (in DVL frame)
        pred_ls = (P_INV @ X.T).T                    # (M, 3)

        results.append({
            "traj_id":  i + 1,
            "gt":       Z,          # (M, 3) DVL frame
            "beamsnet": pred_bn,    # (M, 3)
            "ls":       pred_ls,    # (M, 3)
        })
        print(f"Traj {i+1}: {len(Z)} samples  "
              f"BN RMSE={rmse(Z, pred_bn):.4f}  LS RMSE={rmse(Z, pred_ls):.4f}")

# %% Summary table

rows = []
for r in results:
    gt, bn, ls = r["gt"], r["beamsnet"], r["ls"]
    rows.append({
        "Traj":        r["traj_id"],
        "BN RMSE":     rmse(gt, bn),
        "LS RMSE":     rmse(gt, ls),
        "BN MAE":      mae(gt, bn),
        "LS MAE":      mae(gt, ls),
        "BN R2":       nse(gt, bn),
        "LS R2":       nse(gt, ls),
        "BN VAF":      vaf(gt, bn),
        "LS VAF":      vaf(gt, ls),
    })

df = pd.DataFrame(rows).set_index("Traj")
print("\nPer-trajectory results:")
print(df.round(4).to_string())

# Overall (concatenate all)
all_gt = np.concatenate([r["gt"]       for r in results], axis=0)
all_bn = np.concatenate([r["beamsnet"] for r in results], axis=0)
all_ls = np.concatenate([r["ls"]       for r in results], axis=0)

overall_bn_rmse = rmse(all_gt, all_bn)
overall_ls_rmse = rmse(all_gt, all_ls)
improvement = (overall_ls_rmse - overall_bn_rmse) / overall_ls_rmse * 100

print("\nOverall:")
print(f"  BeamsNetV2 RMSE={overall_bn_rmse:.4f}  MAE={mae(all_gt, all_bn):.4f}  "
      f"R2={nse(all_gt, all_bn):.4f}  VAF={vaf(all_gt, all_bn):.2f}%")
print(f"  LS         RMSE={overall_ls_rmse:.4f}  MAE={mae(all_gt, all_ls):.4f}  "
      f"R2={nse(all_gt, all_ls):.4f}  VAF={vaf(all_gt, all_ls):.2f}%")
print(f"  Improvement (RMSE): {improvement:.2f}%")

# %% Plot — predicted vs GT for each trajectory (speed magnitude)

fig, axes = plt.subplots(N_TRAJ, 1, figsize=(14, 3 * N_TRAJ), sharex=False)
for i, r in enumerate(results):
    gt_speed = LA.norm(r["gt"],       axis=1)
    bn_speed = LA.norm(r["beamsnet"], axis=1)
    ls_speed = LA.norm(r["ls"],       axis=1)
    ax = axes[i]
    ax.plot(gt_speed, label="GT",         linewidth=0.9, color="black")
    ax.plot(bn_speed, label="BeamsNetV2", linewidth=0.8, color="steelblue", alpha=0.8)
    ax.plot(ls_speed, label="LS",         linewidth=0.8, color="tomato",    alpha=0.8, linestyle="--")
    ax.set_ylabel(f"T{r['traj_id']}", fontsize=9, rotation=0, labelpad=30)
    ax.set_xlabel("Sample")
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend(loc="upper right", fontsize=8)
fig.suptitle("BeamsNetV2 vs LS vs GT — speed magnitude (DVL frame) per trajectory", fontsize=12)
plt.tight_layout()
plt.show()

# %% Plot — RMSE and R² per trajectory (BeamsNet vs LS)

traj_ids  = [r["traj_id"] for r in results]
bn_rmses  = [rmse(r["gt"], r["beamsnet"]) for r in results]
ls_rmses  = [rmse(r["gt"], r["ls"])       for r in results]

x     = np.arange(N_TRAJ)
width = 0.35

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(x - width / 2, bn_rmses, width, label="BeamsNetV2", color="steelblue", alpha=0.85)
ax.bar(x + width / 2, ls_rmses, width, label="LS",         color="tomato",    alpha=0.85)
ax.axhline(np.mean(bn_rmses), color="steelblue", linestyle="--", linewidth=1.2, label=f"BN mean={np.mean(bn_rmses):.4f}")
ax.axhline(np.mean(ls_rmses), color="tomato",    linestyle="--", linewidth=1.2, label=f"LS mean={np.mean(ls_rmses):.4f}")
ax.set_xticks(x)
ax.set_xticklabels([f"T{i}" for i in traj_ids])
ax.set_ylabel("RMSE [m/s]")
ax.set_title("BeamsNetV2 vs LS — RMSE per trajectory (DVL frame)")
ax.legend()
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# %% Plot — R² per trajectory (BeamsNet vs LS)

bn_r2s = [nse(r["gt"], r["beamsnet"]) for r in results]
ls_r2s = [nse(r["gt"], r["ls"])       for r in results]

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(x - width / 2, bn_r2s, width, label="BeamsNetV2", color="steelblue", alpha=0.85)
ax.bar(x + width / 2, ls_r2s, width, label="LS",         color="tomato",    alpha=0.85)
ax.axhline(np.mean(bn_r2s), color="steelblue", linestyle="--", linewidth=1.2, label=f"BN mean={np.mean(bn_r2s):.4f}")
ax.axhline(np.mean(ls_r2s), color="tomato",    linestyle="--", linewidth=1.2, label=f"LS mean={np.mean(ls_r2s):.4f}")
ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f"T{i}" for i in traj_ids])
ax.set_ylabel("R²")
ax.set_title("BeamsNetV2 vs LS — R² per trajectory (DVL frame)")
ax.legend()
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# %% Plot — per-axis predictions for trajectory 1

fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
labels = ["vx", "vy", "vz"]
r = results[0]
for j in range(3):
    axes[j].plot(r["gt"][:, j],       label="GT",         linewidth=0.9, color="black")
    axes[j].plot(r["beamsnet"][:, j], label="BeamsNetV2", linewidth=0.8, color="steelblue", alpha=0.8)
    axes[j].plot(r["ls"][:, j],       label="LS",         linewidth=0.8, color="tomato", linestyle="--", alpha=0.8)
    axes[j].set_ylabel(labels[j])
    axes[j].grid(True, alpha=0.3)
axes[0].legend(loc="upper right", fontsize=9)
axes[0].set_title("Trajectory 1 — BeamsNetV2 vs LS vs GT (DVL frame, per axis)")
axes[2].set_xlabel("Sample")
plt.tight_layout()
plt.show()
