# %% [markdown]
# # Apply BeamsNet to A-KIT Beam Data (Zero-Shot)
# Loads AKIT_beams_dataset.npz (step6) and applies the pre-trained BeamsNetV2
# to each of the 13 trajectories with no retraining.
# Our beams were computed as H @ T_body_to_DVL @ v_body, so the LS baseline
# and BeamsNet output are both in DVL frame. GT is transformed to DVL frame
# for fair comparison.
# Note: BeamsNetV1 (also available) additionally requires IMU windows — see
# BeamsNetV1_Test.py for the input format; adapting it is a natural next step.

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
T           = 3   # history window — must match BeamsNetV2 training

# %% DVL geometry helpers (same as step6)

def T_ref_to_body_rad(Euler_body_to_ref_rad):
    phi, theta, psi = Euler_body_to_ref_rad
    T1 = np.array([[1,  0,            0           ],
                   [0,  np.cos(phi),  np.sin(phi) ],
                   [0, -np.sin(phi),  np.cos(phi) ]])
    T2 = np.array([[ np.cos(theta), 0, -np.sin(theta)],
                   [ 0,             1,  0            ],
                   [ np.sin(theta), 0,  np.cos(theta)]])
    T3 = np.array([[ np.cos(psi), np.sin(psi), 0],
                   [-np.sin(psi), np.cos(psi), 0],
                   [ 0,          0,            1]])
    return T1 @ T2 @ T3

def T_body_to_ref_rad(euler):
    return T_ref_to_body_rad(euler).T

GT_body_to_DVL_deg = np.array([-179.9845, 0.2162, -44.3146])
T_BODY_TO_DVL = T_body_to_ref_rad(GT_body_to_DVL_deg * np.pi / 180)  # (3, 3)

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

data_beams = np.load(BEAMS_PATH)
beams_list = [data_beams[f"traj{i+1}_beams"] for i in range(N_TRAJ)]  # each (4, N)

gt_body_list = []
for i in range(N_TRAJ):
    csv = pd.read_csv(f"{DATA_PATH}/Trajectory{i+1}/DVL_trajectory{i+1}.csv")
    vx, vy, vz = csv.iloc[:, 1].values, csv.iloc[:, 2].values, csv.iloc[:, 3].values
    gt_body_list.append(np.stack([vx, vy, vz], axis=0))  # (3, N)

# Transform GT to DVL frame for comparison with LS/BeamsNet output
gt_dvl_list = [T_BODY_TO_DVL @ gt for gt in gt_body_list]   # each (3, N)

print("Loaded data:")
for i in range(N_TRAJ):
    print(f"  Traj {i+1}: beams {beams_list[i].shape}  gt_dvl {gt_dvl_list[i].shape}")

# %% Build per-trajectory inference inputs

def make_windows(beams, gt_dvl):
    """
    beams  : (4, N)
    gt_dvl : (3, N)
    Returns X (N-T, 4), Y (N-T, T, 4), Z (N-T, 3)
    """
    N = beams.shape[1]
    X = np.zeros((N - T, 4),    dtype=np.float32)
    Y = np.zeros((N - T, T, 4), dtype=np.float32)
    Z = np.zeros((N - T, 3),    dtype=np.float32)
    for t in range(N - T):
        X[t] = beams[:, t + T]          # current beam
        Y[t] = beams[:, t:t + T].T      # (T, 4) history
        Z[t] = gt_dvl[:, t + T]         # GT at t+T
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
        X, Y_win, Z = make_windows(beams_list[i], gt_dvl_list[i])

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
        "BN NSE":      nse(gt, bn),
        "LS NSE":      nse(gt, ls),
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

print("\nOverall:")
print(f"  BeamsNetV2 RMSE={rmse(all_gt, all_bn):.4f}  MAE={mae(all_gt, all_bn):.4f}  "
      f"NSE={nse(all_gt, all_bn):.4f}  VAF={vaf(all_gt, all_bn):.2f}%")
print(f"  LS         RMSE={rmse(all_gt, all_ls):.4f}  MAE={mae(all_gt, all_ls):.4f}  "
      f"NSE={nse(all_gt, all_ls):.4f}  VAF={vaf(all_gt, all_ls):.2f}%")

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
