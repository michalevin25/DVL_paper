# %% [markdown]
# # Convert Body-Frame DVL Velocities → Beam Velocities
# Loads the 13 raw trajectories from step1 and converts each (3, N) body-frame
# velocity signal to (4, N) DVL beam velocities using the physical B-matrix model.

# %% Imports & paths
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
DATA_PATH      = "/Users/michal/Desktop/PhD/dvl paper/A-KIT-main/Data"
SAVE_PATH      = "/Users/michal/Desktop/PhD/dvl paper/DATA/AKIT_beams_dataset"
N_TRAJECTORIES = 13

# Error model parameters (BeamsNet eq. 7)
SCALE = 0.007    # 0.7% scale factor
BIAS  = 0.0001   # m/s constant bias per beam
SIGMA = 0.042    # m/s Gaussian noise std




# %% DVL geometry helpers

def T_ref_to_body_rad(Euler_body_to_ref_rad):
    phi_rad   = Euler_body_to_ref_rad[0]
    theta_rad = Euler_body_to_ref_rad[1]
    psi_rad   = Euler_body_to_ref_rad[2]
    T1 = np.array([[1.0,  0.0,               0.0             ],
                   [0.0,  np.cos(phi_rad),    np.sin(phi_rad) ],
                   [0.0, -np.sin(phi_rad),    np.cos(phi_rad) ]])
    T2 = np.array([[ np.cos(theta_rad), 0.0, -np.sin(theta_rad)],
                   [ 0.0,               1.0,  0.0              ],
                   [ np.sin(theta_rad), 0.0,  np.cos(theta_rad)]])
    T3 = np.array([[ np.cos(psi_rad), np.sin(psi_rad), 0.0],
                   [-np.sin(psi_rad), np.cos(psi_rad), 0.0],
                   [ 0.0,             0.0,             1.0]])
    return T1 @ T2 @ T3

def T_body_to_ref_rad(Euler_body_to_ref_rad):
    return T_ref_to_body_rad(Euler_body_to_ref_rad).T

def B_mat(psi_angs_deg=np.array([45, 135, 225, 315]), alpha_deg=20):
    psi1, psi2, psi3, psi4 = psi_angs_deg * np.pi / 180
    alpha_rad = alpha_deg * np.pi / 180

    h1 = np.array([np.cos(psi1) * np.sin(alpha_rad),
                   np.sin(psi1) * np.sin(alpha_rad), np.cos(alpha_rad)])
    h2 = np.array([np.cos(psi2) * np.sin(alpha_rad),
                   np.sin(psi2) * np.sin(alpha_rad), np.cos(alpha_rad)])
    h3 = np.array([np.cos(psi3) * np.sin(alpha_rad),
                   np.sin(psi3) * np.sin(alpha_rad), np.cos(alpha_rad)])
    h4 = np.array([np.cos(psi4) * np.sin(alpha_rad),
                   np.sin(psi4) * np.sin(alpha_rad), np.cos(alpha_rad)])

    H = np.array([h1, h2, h3, h4]).reshape((4, 3))
    return H

def apply_error_model(beams):
    """
    Apply DVL error model (eq. 7) to clean beams.
    beams : (4, N)  →  noisy beams : (4, N)
    """
    return beams * (1 + SCALE) + SIGMA * np.random.randn(*beams.shape) + BIAS

def body_to_beams(v_body):
    """
    Project body-frame velocity directly to 4 beam velocities (no mounting rotation).
    Matches BeamsNet's training assumption: beams = H @ v_body.
    v_body : (3,) or (N, 3)  →  beams : (4,) or (N, 4)
    """
    H      = B_mat()
    v_body = np.asarray(v_body)
    single = v_body.ndim == 1
    if single:
        v_body = v_body[np.newaxis, :]

    beams = v_body @ H.T   # (N, 4)

    return beams[0] if single else beams

# %% Load raw trajectories

def load_signals():
    signals, times = [], []
    for i in range(1, N_TRAJECTORIES + 1):
        path = f"{DATA_PATH}/Trajectory{i}/DVL_trajectory{i}.csv"
        data = pd.read_csv(path)
        time = data.iloc[:, 0].values
        vx   = data.iloc[:, 1].values
        vy   = data.iloc[:, 2].values
        vz   = data.iloc[:, 3].values
        signals.append(np.stack([vx, vy, vz], axis=0))  # (3, N)
        times.append(time)
    return signals, times

signals, times = load_signals()
print(f"Loaded {len(signals)} trajectories")
for i, s in enumerate(signals):
    print(f"  Traj {i+1}: shape {s.shape}")

# %% Convert body → beams

beams_list = []
for i, sig in enumerate(signals):
    # sig: (3, N)  →  body_to_beams expects (N, 3)
    v_body = sig.T                      # (N, 3)
    beams  = body_to_beams(v_body)      # (N, 4)
    beams_list.append(beams.T)          # store as (4, N)
    print(f"  Traj {i+1}: body {sig.shape} → beams {beams_list[-1].shape}")

noisy_beams_list = [apply_error_model(b) for b in beams_list]

# %% Save dataset

np.savez(
    SAVE_PATH,
    **{f"traj{i+1}_beams":       beams_list[i]       for i in range(N_TRAJECTORIES)},
    **{f"traj{i+1}_beams_noisy": noisy_beams_list[i] for i in range(N_TRAJECTORIES)},
    **{f"traj{i+1}_time":        times[i]             for i in range(N_TRAJECTORIES)},
)
print(f"Saved to {SAVE_PATH}.npz")

# %% Sanity check — plot trajectory 1: body vs beams

fig, axes = plt.subplots(2, 1, figsize=(14, 6))

body_labels = ["vx", "vy", "vz"]
beam_labels = ["beam1", "beam2", "beam3", "beam4"]
t = times[0]

for j, lbl in enumerate(body_labels):
    axes[0].plot(t, signals[0][j], label=lbl)
axes[0].set_title("Trajectory 1 — body frame (vx, vy, vz)")
axes[0].legend()
axes[0].set_xlabel("Time [s]")
axes[0].set_ylabel("Velocity [m/s]")

for j, lbl in enumerate(beam_labels):
    axes[1].plot(t, beams_list[0][j], label=lbl)
axes[1].set_title("Trajectory 1 — DVL beams")
axes[1].legend()
axes[1].set_xlabel("Time [s]")
axes[1].set_ylabel("Beam velocity [m/s]")

plt.tight_layout()
plt.show()

# %% Round-trip sanity check (single sample)

v_test   = signals[0][:, 0]            # (3,)
beams_rt = body_to_beams(v_test)       # (4,)
H        = B_mat()
v_rec    = np.linalg.pinv(H) @ beams_rt

print("Round-trip check (first sample of traj 1):")
print(f"  original  : {v_test}")
print(f"  recovered : {v_rec}")
print(f"  match     : {np.allclose(v_test, v_rec)}")

# %% Sanity check 3 — Pure vertical motion gives equal beams

v_vertical = np.array([0.0, 0.0, 1.0])
beams_vert = body_to_beams(v_vertical)
print("\nCheck 3 — pure vertical motion (expect all 4 beams equal):")
print(f"  beams : {beams_vert}")
print(f"  all equal : {np.allclose(beams_vert, beams_vert[0])}")

# %% Sanity check 4 — Magnitude is preserved (Parseval-like)

v_mag = np.array([1.0, 0.5, 0.2])
beams_mag = body_to_beams(v_mag)
print("\nCheck 4 — magnitude scaling (should not explode or vanish):")
print(f"  input  ||v||    : {np.linalg.norm(v_mag):.4f}")
print(f"  output ||beams||: {np.linalg.norm(beams_mag):.4f}")
print(f"  ratio           : {np.linalg.norm(beams_mag) / np.linalg.norm(v_mag):.4f}")

# %% Convert synthetic signals → beam velocities

GEN_PATH         = "/Users/michal/Desktop/PhD/dvl paper/GENERATED DATA/synthetic_dataset.npz"
SYNTHETIC_SAVE   = "/Users/michal/Desktop/PhD/dvl paper/GENERATED DATA/synthetic_beams_dataset"

data = np.load(GEN_PATH)
gen_sigs   = data["signals"]   # (65, 3, 400) — normalized
gen_means  = data["means"]     # (65, 3)
gen_stds   = data["stds"]      # (65, 3)
gen_ids    = data["traj_ids"]  # (65,)
N_GEN      = len(gen_sigs)

print(f"Loaded synthetic dataset: {gen_sigs.shape}")

syn_beams_list = []
for i in range(N_GEN):
    beams = body_to_beams(gen_sigs[i].T)  # (400, 4)
    syn_beams_list.append(beams.T)        # (4, 400)

syn_beams = np.stack(syn_beams_list)  # (65, 4, 400)
print(f"Beam dataset shape: {syn_beams.shape}")

syn_beams_noisy = np.stack([apply_error_model(b) for b in syn_beams_list])  # (65, 4, 400)

np.savez(
    SYNTHETIC_SAVE,
    beams       = syn_beams,        # (65, 4, 400) clean
    beams_noisy = syn_beams_noisy,  # (65, 4, 400) with error model applied
    means    = gen_means,
    stds     = gen_stds,
    traj_ids = gen_ids,
)
print(f"Saved → {SYNTHETIC_SAVE}.npz")

# %% Sanity check — plot first synthetic trajectory: body vs beams

fig, axes = plt.subplots(2, 1, figsize=(14, 6))
i = 0
vel0 = gen_sigs[i]  # (3, 400) m/s

for j, lbl in enumerate(["vx", "vy", "vz"]):
    axes[0].plot(vel0[j], label=lbl)
axes[0].set_title(f"Synthetic traj {gen_ids[i]} — body frame (denormalized m/s)")
axes[0].legend(); axes[0].set_xlabel("Sample"); axes[0].set_ylabel("Velocity [m/s]")

for j, lbl in enumerate(["beam1", "beam2", "beam3", "beam4"]):
    axes[1].plot(syn_beams[i, j], label=lbl)
axes[1].set_title(f"Synthetic traj {gen_ids[i]} — DVL beams")
axes[1].legend(); axes[1].set_xlabel("Sample"); axes[1].set_ylabel("Beam velocity [m/s]")

plt.tight_layout()
plt.show()
