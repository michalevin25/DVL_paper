import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

DATASET_PATH = "/Users/michal/Desktop/PhD/dvl paper/DATA/dvl_dataset.npz"

data       = np.load(DATASET_PATH)
signals    = data["signals"]     # (13, 3, N)
curvatures = data["curvatures"]  # (13, 3, N)


# ── 1. Peak width analysis → window size ─────────────────────────────────────

def get_peak_widths(data_array):
    widths = []
    for traj_idx in range(len(data_array)):
        for axis in range(3):
            s = np.abs(data_array[traj_idx, axis])
            threshold = s.mean() + s.std()
            peaks, props = find_peaks(s, height=threshold, width=1)
            if len(peaks) > 0:
                widths.extend(props["widths"].tolist())
    return np.array(widths)

signal_widths    = get_peak_widths(signals)
curvature_widths = get_peak_widths(curvatures)

for name, widths in [("Signal", signal_widths), ("Curvature", curvature_widths)]:
    print(f"{name} peak widths:")
    print(f"  min:    {widths.min():.1f} samples")
    print(f"  mean:   {widths.mean():.1f} samples")
    print(f"  max:    {widths.max():.1f} samples")
    print(f"  → suggested window size: {int(widths.max() * 2)} samples (2× max peak width)\n")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, name, widths, color in zip(axes,
                                    ["Signal", "Curvature"],
                                    [signal_widths, curvature_widths],
                                    ["steelblue", "darkorange"]):
    ax.hist(widths, bins=20, color=color, edgecolor="white")
    ax.axvline(widths.mean(), color="red",    linestyle="--", label=f"mean={widths.mean():.1f}")
    ax.axvline(widths.max(),  color="black",  linestyle="--", label=f"max={widths.max():.1f}")
    ax.set_xlabel("Peak width (samples)")
    ax.set_ylabel("Count")
    ax.set_title(f"{name} peak widths — all trajectories & axes")
    ax.legend()
    ax.grid(True, alpha=0.4)

plt.tight_layout()
plt.show()


# ── 2. Autocorrelation analysis → stride ─────────────────────────────────────

decorr_lengths = []
max_lag = 200

for traj_idx in range(len(signals)):
    for axis in range(3):
        s = signals[traj_idx, axis]
        s = s - s.mean()
        # normalized autocorrelation
        acf = np.correlate(s, s, mode="full")
        acf = acf[len(acf)//2:]
        acf = acf / acf[0]
        # find first lag where autocorrelation drops below 0.5
        below = np.where(acf[:max_lag] < 0.5)[0]
        if len(below) > 0:
            decorr_lengths.append(below[0])

decorr_lengths = np.array(decorr_lengths)
print(f"\nAutocorrelation decorrelation length (drop below 0.5):")
print(f"  min:    {decorr_lengths.min()} samples")
print(f"  mean:   {decorr_lengths.mean():.1f} samples")
print(f"  max:    {decorr_lengths.max()} samples")
print(f"  → suggested stride: {int(decorr_lengths.mean())} samples")

# plot autocorrelation for trajectory 1, vx
s  = signals[0, 0]
s  = s - s.mean()
acf = np.correlate(s, s, mode="full")
acf = acf[len(acf)//2:]
acf = acf / acf[0]

axes[1].plot(acf[:max_lag], color="steelblue")
axes[1].axhline(0.5,  color="red",    linestyle="--", label="0.5 threshold")
axes[1].axhline(0.0,  color="gray",   linestyle="--", linewidth=0.8)
axes[1].set_xlabel("Lag (samples)")
axes[1].set_ylabel("Autocorrelation")
axes[1].set_title("Autocorrelation — Trajectory 1, vx")
axes[1].legend()
axes[1].grid(True, alpha=0.4)

plt.tight_layout()
plt.show()


# ── 3. Summary ────────────────────────────────────────────────────────────────

suggested_window = int(max(signal_widths.max(), curvature_widths.max()) * 2)
suggested_stride = int(decorr_lengths.mean())
n_windows        = sum(
    max(0, (signals.shape[2] - suggested_window) // suggested_stride + 1)
    for _ in range(len(signals))
)

print(f"\nSuggested parameters:")
print(f"  window size: {suggested_window}")
print(f"  stride:      {suggested_stride}")
print(f"  total windows from 13 trajectories: {n_windows}")
