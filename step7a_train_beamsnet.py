# %% [markdown]
# # Step 7a — Train BeamsNetV2 on A-KIT Data
# Retrains BeamsNetV2 on A-KIT beam measurements (step6 output) instead of
# the original Snapir data. Follows the original BeamsNet training pipeline:
# same architecture, same noise model already applied in step6, same 75/25
# train/val split (applied to the concatenated training trajectories).
# Held-out test trajectories are evaluated per-trajectory and overall.

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

BEAMS_PATH = "/Users/michal/Desktop/PhD/dvl paper/DATA/AKIT_beams_dataset.npz"
DATA_PATH  = "/Users/michal/Desktop/PhD/dvl paper/A-KIT-main/Data"
SAVE_PATH  = "/Users/michal/Desktop/PhD/dvl paper/BeamsNetV2_AKIT.pkl"

N_TRAJ      = 13
SKIP_TRAJS  = {12}
TRAIN_TRAJS = set(range(1, 12))   # trajectories 1–11
TEST_TRAJS  = {13}                # trajectory 13 held out

T          = 3      # DVL history window — must match architecture
BATCH_SIZE = 32
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

# %% Load beam data and GT velocities

data_beams = np.load(BEAMS_PATH)

noisy_beams_list = []
gt_body_list     = []

for i in range(N_TRAJ):
    noisy_beams_list.append(data_beams[f"traj{i+1}_beams_noisy"])   # (4, 400)
    csv = pd.read_csv(f"{DATA_PATH}/Trajectory{i+1}/DVL_trajectory{i+1}.csv")
    vx, vy, vz = csv.iloc[:, 1].values, csv.iloc[:, 2].values, csv.iloc[:, 3].values
    gt_body_list.append(np.stack([vx, vy, vz], axis=0))             # (3, 400)

print("Loaded trajectories:")
for i in range(N_TRAJ):
    tag = " [SKIP]" if (i+1) in SKIP_TRAJS else \
          " [TEST]"  if (i+1) in TEST_TRAJS  else " [TRAIN]"
    print(f"  Traj {i+1}: beams {noisy_beams_list[i].shape}  gt {gt_body_list[i].shape}{tag}")

# %% Build windows and form train / test sets

train_X, train_Y, train_Z = [], [], []
test_data = []   # list of dicts for per-trajectory test eval

for i in range(N_TRAJ):
    tid = i + 1
    if tid in SKIP_TRAJS:
        continue
    X, Y, Z = make_windows(noisy_beams_list[i], gt_body_list[i])
    if tid in TEST_TRAJS:
        test_data.append({"traj_id": tid, "X": X, "Y": Y, "Z": Z})
    else:
        train_X.append(X)
        train_Y.append(Y)
        train_Z.append(Z)

train_X = np.concatenate(train_X, axis=0)
train_Y = np.concatenate(train_Y, axis=0)
train_Z = np.concatenate(train_Z, axis=0)

N_all   = len(train_X)
N_val   = N_all // 4
N_train = N_all - N_val

X_tr, X_va = train_X[:N_train], train_X[N_train:]
Y_tr, Y_va = train_Y[:N_train], train_Y[N_train:]
Z_tr, Z_va = train_Z[:N_train], train_Z[N_train:]

print(f"\nTrain: {N_train} windows  |  Val: {N_val} windows")
print(f"Test trajectories: {sorted(TEST_TRAJS)}  ({sum(len(d['Z']) for d in test_data)} windows)")

# %% DataLoaders

def make_loader(X, Y, Z, shuffle=True):
    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(Y),
        torch.from_numpy(Z),
    )
    return torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

train_loader = make_loader(X_tr, Y_tr, Z_tr, shuffle=True)
val_loader   = make_loader(X_va, Y_va, Z_va, shuffle=False)

# %% Initialise model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model  = BeamsNetV2().to(device)
model.initialize_weights()
print(f"Training on {device}")

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# %% Training loop

train_losses, val_losses = [], []
best_val_loss = float("inf")

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    for xb, yb, zb in train_loader:
        xb, yb, zb = xb.to(device), yb.to(device), zb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb, yb), zb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(xb)
    train_losses.append(epoch_loss / N_train)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb, zb in val_loader:
            xb, yb, zb = xb.to(device), yb.to(device), zb.to(device)
            val_loss += criterion(model(xb, yb), zb).item() * len(xb)
    val_losses.append(val_loss / N_val)

    if val_losses[-1] < best_val_loss:
        best_val_loss = val_losses[-1]
        torch.save(model.state_dict(), SAVE_PATH)
        tag = " *"
    else:
        tag = ""

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{EPOCHS}  train={train_losses[-1]:.6f}  val={val_losses[-1]:.6f}{tag}")

print(f"\nBest val loss: {best_val_loss:.6f}  → {SAVE_PATH}")

# %% Plot training curves

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(train_losses, label="Train", linewidth=1.2)
ax.plot(val_losses,   label="Val",   linewidth=1.2)
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss")
ax.set_title("BeamsNetV2 — Training on A-KIT data")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% Load best model and evaluate on test trajectories

model.load_state_dict(torch.load(SAVE_PATH, map_location=device, weights_only=True))
model.eval()

results = []
with torch.no_grad():
    for d in test_data:
        X_t = torch.from_numpy(d["X"]).to(device)
        Y_t = torch.from_numpy(d["Y"]).to(device)
        pred_bn = model(X_t, Y_t).cpu().numpy()
        pred_ls = (P_INV @ d["X"].T).T
        results.append({
            "traj_id":  d["traj_id"],
            "gt":       d["Z"],
            "beamsnet": pred_bn,
            "ls":       pred_ls,
        })
        print(f"Traj {d['traj_id']}: BN RMSE={rmse(d['Z'], pred_bn):.4f}  "
              f"LS RMSE={rmse(d['Z'], pred_ls):.4f}")

# %% Summary table

rows = []
for r in results:
    gt, bn, ls = r["gt"], r["beamsnet"], r["ls"]
    rows.append({
        "Traj":    r["traj_id"],
        "BN RMSE": rmse(gt, bn), "LS RMSE": rmse(gt, ls),
        "BN MAE":  mae(gt, bn),  "LS MAE":  mae(gt, ls),
        "BN R²":   nse(gt, bn),  "LS R²":   nse(gt, ls),
        "BN VAF":  vaf(gt, bn),  "LS VAF":  vaf(gt, ls),
    })

df = pd.DataFrame(rows).set_index("Traj")
print("\nPer-trajectory results:")
print(df.round(4).to_string())

all_gt = np.concatenate([r["gt"]       for r in results], axis=0)
all_bn = np.concatenate([r["beamsnet"] for r in results], axis=0)
all_ls = np.concatenate([r["ls"]       for r in results], axis=0)

overall_bn = rmse(all_gt, all_bn)
overall_ls = rmse(all_gt, all_ls)
print(f"\nOverall  BN RMSE={overall_bn:.4f}  LS RMSE={overall_ls:.4f}  "
      f"Improvement={( overall_ls - overall_bn) / overall_ls * 100:.2f}%")

# %% Plot — speed magnitude per test trajectory

for r in results:
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(LA.norm(r["gt"],       axis=1), label="GT",         linewidth=0.9, color="black")
    ax.plot(LA.norm(r["beamsnet"], axis=1), label="BeamsNetV2", linewidth=0.8, color="steelblue", alpha=0.8)
    ax.plot(LA.norm(r["ls"],       axis=1), label="LS",         linewidth=0.8, color="tomato",   alpha=0.8, linestyle="--")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Speed [m/s]")
    ax.set_title(f"Traj {r['traj_id']} — BeamsNetV2 vs LS vs GT (speed magnitude)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# %% Plot — per-axis for first test trajectory

r = results[0]
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
for j, label in enumerate(["vx", "vy", "vz"]):
    axes[j].plot(r["gt"][:, j],       label="GT",         linewidth=0.9, color="black")
    axes[j].plot(r["beamsnet"][:, j], label="BeamsNetV2", linewidth=0.8, color="steelblue", alpha=0.8)
    axes[j].plot(r["ls"][:, j],       label="LS",         linewidth=0.8, color="tomato",   alpha=0.8, linestyle="--")
    axes[j].set_ylabel(label)
    axes[j].grid(True, alpha=0.3)
axes[0].legend(loc="upper right", fontsize=9)
axes[0].set_title(f"Traj {r['traj_id']} — per-axis predictions")
axes[2].set_xlabel("Sample")
plt.tight_layout()
plt.show()
