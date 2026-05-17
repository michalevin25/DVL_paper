# %% [markdown]
# # BeamsNet on Original Snapir Data
# Supports two modes via SKIP_TRAINING:
#   False → train from scratch (or from initial weights) and save
#   True  → load existing pre-trained weights and evaluate only
#
# Preprocessing follows BeamsNet-main exactly: same beam geometry, noise model,
# and train/val split convention.

# %% Imports & paths
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import cos, sin
from numpy.linalg import inv
from sklearn.metrics import mean_squared_error
from numpy import linalg as LA
%matplotlib inline

torch.manual_seed(0)
np.random.seed(0)

TRAIN_V_PATH     = "/Users/michal/Desktop/PhD/dvl paper/BeamsNet-main/dataset/TrainAndValidation/V.npy"
TEST_V_PATH      = "/Users/michal/Desktop/PhD/dvl paper/BeamsNet-main/dataset/Test/V_test.npy"
INIT_W_PATH      = "/Users/michal/Desktop/PhD/dvl paper/BeamsNet-main/code/Initial Weights/BeamsNetV2_InitialWeights.pkl"
PRETRAINED_PATH  = "/Users/michal/Desktop/PhD/dvl paper/BeamsNet-main/code/BeamsNetV2.pkl"
SAVE_PATH        = "/Users/michal/Desktop/PhD/dvl paper/BeamsNetV2_trained.pkl"

# ── mode switch ──────────────────────────────────────────────────────────────
SKIP_TRAINING = True   # True → use PRETRAINED_PATH; False → train from scratch
# ─────────────────────────────────────────────────────────────────────────────

T           = 3      # DVL history window (must match architecture)
BATCH_SIZE  = 32
EPOCHS      = 150
LR          = 1e-3

# %% Beam geometry (same as BeamsNet paper, no mounting correction)

def make_beam_matrix():
    rows = []
    for k in range(4):
        psi   = (45 + k * 90) * np.pi / 180
        alpha = 20 * np.pi / 180
        rows.append([cos(psi) * sin(alpha), sin(psi) * sin(alpha), cos(alpha)])
    return np.array(rows)   # (4, 3)

A     = make_beam_matrix()
AT_A  = A.T @ A
P_INV = np.linalg.lstsq(A, np.eye(4), rcond=None)[0]   # (3, 4)  LS pseudoinverse

# %% BeamsNetV2 model

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

# %% Preprocessing helper — velocity → noisy beams → windows

def velocity_to_noisy_beams(V_col):
    """V_col: (3, N) — returns beams_noisy (N, 4)"""
    N = V_col.shape[1]
    beams = np.array([A @ (V_col[:, i] * (1 + 0.007)) for i in range(N)])   # scale 0.7%
    beams_noisy = beams + (0.042 ** 2) * np.random.randn(N, 4) + 0.0001
    return beams_noisy

def make_windows(beams_noisy, V_col):
    """
    beams_noisy : (N, 4)
    V_col       : (3, N)  — GT velocities
    Returns X (N-T, 4), Y (N-T, T, 4), Z (N-T, 3)
    """
    N  = beams_noisy.shape[0]
    X  = np.zeros((N - T, 4),    dtype=np.float32)
    Y  = np.zeros((N - T, T, 4), dtype=np.float32)
    Z  = np.zeros((N - T, 3),    dtype=np.float32)
    for t in range(N - T):
        X[t] = beams_noisy[t + T, :]
        Y[t] = beams_noisy[t:t + T, :]
        Z[t] = V_col[:, t + T]
    return X, Y, Z

# %% Load and preprocess training data

V_train_raw    = np.load(TRAIN_V_PATH)          # (3, 13886)
beams_train    = velocity_to_noisy_beams(V_train_raw)
X_all, Y_all, Z_all = make_windows(beams_train, V_train_raw)

N_all    = len(X_all)
N_val    = N_all // 4
N_train  = N_all - N_val

# First 75 % → train, last 25 % → validation (original convention)
X_tr,  X_va  = X_all[:N_train],  X_all[N_train:]
Y_tr,  Y_va  = Y_all[:N_train],  Y_all[N_train:]
Z_tr,  Z_va  = Z_all[:N_train],  Z_all[N_train:]

print(f"Train: {N_train} samples  |  Val: {N_val} samples")

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

# %% Initialise model and optionally train

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model  = BeamsNetV2().to(device)

if SKIP_TRAINING:
    model.load_state_dict(torch.load(PRETRAINED_PATH, map_location=device, weights_only=True))
    print(f"Loaded pre-trained weights from {PRETRAINED_PATH}")
    model.eval()

    # Quick validation-split check with pre-trained weights
    preds_va = []
    with torch.no_grad():
        for xb, yb, _ in val_loader:
            preds_va.append(model(xb.to(device), yb.to(device)).cpu().numpy())
    preds_va = np.concatenate(preds_va, axis=0)
    ls_va    = (P_INV @ X_va.T).T
    print(f"\nVal split ({N_val} samples) with pre-trained weights:")
    print(f"  BN  RMSE={rmse(Z_va, preds_va):.5f}  MAE={mae(Z_va, preds_va):.5f}  "
          f"R²={nse(Z_va, preds_va):.4f}  VAF={vaf(Z_va, preds_va):.2f}%")
    print(f"  LS  RMSE={rmse(Z_va, ls_va):.5f}  MAE={mae(Z_va, ls_va):.5f}  "
          f"R²={nse(Z_va, ls_va):.4f}  VAF={vaf(Z_va, ls_va):.2f}%")

else:
    try:
        model.load_state_dict(torch.load(INIT_W_PATH, map_location=device, weights_only=True))
        print(f"Loaded initial weights from {INIT_W_PATH}")
    except Exception as e:
        print(f"Using Kaiming init ({e})")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

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

    print(f"\nBest val loss: {best_val_loss:.6f}  → saved to {SAVE_PATH}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train_losses, label="Train", linewidth=1.2)
    ax.plot(val_losses,   label="Val",   linewidth=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("BeamsNetV2 — Training on Snapir data")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    model.load_state_dict(torch.load(SAVE_PATH, map_location=device, weights_only=True))
    model.eval()

V_test_raw    = np.load(TEST_V_PATH)             # (3, 2001)
beams_test    = velocity_to_noisy_beams(V_test_raw)
X_te, Y_te, Z_te = make_windows(beams_test, V_test_raw)
N_te = len(X_te)

X_te_t = torch.from_numpy(X_te).to(device)
Y_te_t = torch.from_numpy(Y_te).to(device)

with torch.no_grad():
    pred_bn = model(X_te_t, Y_te_t).cpu().numpy()   # (N_te, 3)

pred_ls = (P_INV @ X_te.T).T                        # (N_te, 3)

print(f"\nTest set: {N_te} samples")
df = pd.DataFrame(
    [[rmse(Z_te, pred_bn), mae(Z_te, pred_bn), nse(Z_te, pred_bn), vaf(Z_te, pred_bn)],
     [rmse(Z_te, pred_ls), mae(Z_te, pred_ls), nse(Z_te, pred_ls), vaf(Z_te, pred_ls)]],
    index=["BeamsNetV2 (pre-trained)" if SKIP_TRAINING else "BeamsNetV2 (retrained)", "Least Squares"],
    columns=["RMSE", "MAE", "R²", "VAF"],
)
print(df.round(5))
improvement = (rmse(Z_te, pred_ls) - rmse(Z_te, pred_bn)) / rmse(Z_te, pred_ls) * 100
print(f"\nRMSE improvement over LS: {improvement:.2f}%")

# %% Plot — speed magnitude on test set

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(LA.norm(Z_te,    axis=1), label="GT",         linewidth=0.9,  color="black")
ax.plot(LA.norm(pred_bn, axis=1), label="BeamsNetV2", linewidth=0.8,  color="steelblue", alpha=0.8)
ax.plot(LA.norm(pred_ls, axis=1), label="LS",         linewidth=0.8,  color="tomato",    alpha=0.8, linestyle="--")
ax.set_xlabel("Sample")
ax.set_ylabel("Speed [m/s]")
ax.set_title("Test set — speed magnitude")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% Plot — per-axis predictions on test set

fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
labels = ["vx", "vy", "vz"]
for j in range(3):
    axes[j].plot(Z_te[:, j],    label="GT",         linewidth=0.9, color="black")
    axes[j].plot(pred_bn[:, j], label="BeamsNetV2", linewidth=0.8, color="steelblue", alpha=0.8)
    axes[j].plot(pred_ls[:, j], label="LS",         linewidth=0.8, color="tomato",    alpha=0.8, linestyle="--")
    axes[j].set_ylabel(labels[j])
    axes[j].grid(True, alpha=0.3)
axes[0].legend(loc="upper right", fontsize=9)
axes[0].set_title("Test set — BeamsNetV2 vs LS vs GT (per axis)")
axes[2].set_xlabel("Sample")
plt.tight_layout()
plt.show()
