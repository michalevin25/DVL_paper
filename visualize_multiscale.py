"""
Visualizes the multi-scale peak_map injection change.

Top section — Before vs After.
  "Before": peak_map flows through successive AvgPool (no fresh injection
             after enc1).  Shown at each encoder scale in original-time coords.
  "After":  peak_map freshly re-sampled and injected at every scale.

Bottom half — architecture schematic: OLD vs NEW.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Synthetic peak map ─────────────────────────────────────────────────────────

L     = 206
SIGMA = 5   # sharper peaks so compression is visually obvious

def make_peak_map(positions, amplitudes, length=L, sigma=SIGMA):
    t   = np.arange(length, dtype=np.float32)
    out = np.zeros(length, dtype=np.float32)
    for pos, amp in zip(positions, amplitudes):
        out += amp * np.exp(-((t - pos) ** 2) / (2 * sigma ** 2))
    return out

pm_full = make_peak_map([40, 160], [1.0, -0.85])
pm_t    = torch.tensor(pm_full).view(1, 1, -1)

pool = nn.AvgPool1d(kernel_size=2, stride=2)

# ── "Before" path: peak_map passes through successive average-pooling ──────────
# (this is what the model has access to at each encoder level without fresh injection)

before_103 = pool(pm_t).squeeze().numpy()                  # 1× pooled → L=103
before_51  = pool(pool(pm_t)).squeeze().numpy()            # 2× pooled → L=51
before_25  = pool(pool(pool(pm_t))).squeeze().numpy()      # 3× pooled → L=25

before_signals = {
    206: pm_full,
    103: before_103,
    51:  before_51,
    25:  before_25,
}

# ── "After" path: fresh linear interpolation to each scale ────────────────────

def interp(pm, size):
    return F.interpolate(pm, size=size, mode='linear', align_corners=False).squeeze().numpy()

after_signals = {
    206: pm_full,
    103: interp(pm_t, 103),
    51:  interp(pm_t, 51),
    25:  interp(pm_t, 25),
}

sizes  = [206, 103, 51, 25]
labels = ["enc1  (L=206)", "enc2  (L=103)", "enc3  (L=51)", "bottleneck  (L=25)"]

EARLY_POS  = 40
LATE_POS   = 160
POS_COLORS = ["#2ecc71", "#e74c3c"]

# ── Figure layout ─────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(16, 11))
fig.patch.set_facecolor("#f9f9f9")

gs = fig.add_gridspec(3, 4, hspace=0.15, wspace=0.32,
                      height_ratios=[1, 1, 1.3],
                      top=0.92, bottom=0.06, left=0.06, right=0.97)

# ── Helper: draw one scale subplot ────────────────────────────────────────────

def draw_scale(ax, size, pm, title, active, is_before):
    """
    active=True  → fresh injection at this level
    active=False → only pooled info available (before case, levels 2-4)
    """
    # x-axis in original time coordinates (0..205) so peaks are comparable
    x = np.linspace(0, L - 1, size)

    # stem plot: vertical bars from zero
    col  = "#2980b9" if active else "#999"
    lw   = 1.4 if size > 50 else 2.0  # wider bars for coarser grids

    markerline, stemlines, baseline = ax.stem(x, pm, linefmt=col,
                                               markerfmt=" ", basefmt="k-")
    plt.setp(stemlines, linewidth=lw, alpha=0.85 if active else 0.55)
    plt.setp(baseline,  linewidth=0.5, alpha=0.4)

    # smooth envelope for reference
    x_full = np.linspace(0, L - 1, 500)
    # reconstruct smooth envelope from the pm values
    pm_env = np.interp(x_full, x, pm)
    ax.plot(x_full, pm_env, color=col, linewidth=0.8,
            alpha=0.45 if active else 0.3, linestyle="--")

    # true peak position markers
    for pos, c in [(EARLY_POS, POS_COLORS[0]), (LATE_POS, POS_COLORS[1])]:
        ax.axvline(pos, color=c, linewidth=1.3, linestyle="--", alpha=0.85, zorder=4)

    # injection / state label
    if active:
        ax.text(0.5, 0.91, "✓ injected fresh", transform=ax.transAxes,
                ha="center", fontsize=8.5, color="#27ae60", fontweight="bold")
    else:
        ax.text(0.5, 0.91, "✗ only avg-pooled signal", transform=ax.transAxes,
                ha="center", fontsize=8.5, color="#c0392b", fontweight="bold")

    ax.set_xlim(-3, L + 2)
    ax.set_ylim(-1.2, 1.45)
    ax.axhline(0, color="k", linewidth=0.4, alpha=0.35)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.18)

    ax.text(0.97, 0.04, f"{size} samples", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7.5, color="#666")

    # resolution annotation
    res = L / size
    ax.text(0.97, 0.14, f"1 sample = {res:.1f} steps", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7, color="#888")

    if title:
        ax.set_title(title, fontsize=9.5, fontweight="bold", pad=3)


# ── Row 0: Before ─────────────────────────────────────────────────────────────

fig.text(0.005, 0.845, "Before", va="center", rotation=90,
         fontsize=11, fontweight="bold", color="#c0392b")

for col, (size, lab) in enumerate(zip(sizes, labels)):
    ax     = fig.add_subplot(gs[0, col])
    active = (size == 206)
    draw_scale(ax, size, before_signals[size], lab, active=active, is_before=True)
    if col > 0:
        ax.set_yticklabels([])

# ── Row 1: After ──────────────────────────────────────────────────────────────

fig.text(0.005, 0.655, "After", va="center", rotation=90,
         fontsize=11, fontweight="bold", color="#27ae60")

for col, (size, lab) in enumerate(zip(sizes, labels)):
    ax = fig.add_subplot(gs[1, col])
    draw_scale(ax, size, after_signals[size], "", active=True, is_before=False)
    if col > 0:
        ax.set_yticklabels([])

# shared legend
early_patch = mpatches.Patch(color=POS_COLORS[0], label="early peak  (t=40)")
late_patch  = mpatches.Patch(color=POS_COLORS[1], label="late peak   (t=160)")
fig.legend(handles=[early_patch, late_patch], loc="upper center",
           ncol=2, fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, 0.97))

fig.add_artist(plt.Line2D([0.03, 0.97], [0.413, 0.413],
                           transform=fig.transFigure,
                           color="#bbb", linewidth=1.2, linestyle="--"))

# ── Row 2: architecture schematic ─────────────────────────────────────────────

ax_old = fig.add_subplot(gs[2, :2])
ax_new = fig.add_subplot(gs[2, 2:])

def draw_arch(ax, inject_all=False, title=""):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)

    BOX_H    = 0.55
    BN_COL   = "#8e44ad"
    ENC_COLS = ["#3498db", "#2980b9", "#1f618d"]
    DEC_COLS = ["#e67e22", "#d35400", "#ba4a00"]

    enc_boxes = [
        (0.3,  3.8, 2.2, BOX_H, ENC_COLS[0], "enc1\n(B,64,206)"),
        (0.3,  2.7, 2.2, BOX_H, ENC_COLS[1], "enc2\n(B,128,103)"),
        (0.3,  1.6, 2.2, BOX_H, ENC_COLS[2], "enc3\n(B,256,51)"),
        (3.7,  0.6, 2.6, BOX_H, BN_COL,      "bottleneck\n(B,256,25)"),
    ]
    dec_boxes = [
        (7.5,  1.6, 2.2, BOX_H, DEC_COLS[2], "dec3\n(B,256,51)"),
        (7.5,  2.7, 2.2, BOX_H, DEC_COLS[1], "dec2\n(B,128,103)"),
        (7.5,  3.8, 2.2, BOX_H, DEC_COLS[0], "dec1\n(B,64,206)"),
    ]

    def box(ax, x, y, w, h, col, lab):
        rect = mpatches.FancyBboxPatch((x, y), w, h,
                                        boxstyle="round,pad=0.06",
                                        facecolor=col, edgecolor="white",
                                        linewidth=1.2, alpha=0.88)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, lab, ha="center", va="center",
                fontsize=7.5, color="white", fontweight="bold")

    def arrow(ax, x0, y0, x1, y1, col="gray"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.2))

    for b in enc_boxes + dec_boxes:
        box(ax, *b)

    arrow(ax, 1.4, 3.8,  1.4, 3.35)
    arrow(ax, 1.4, 2.7,  1.4, 2.25)
    arrow(ax, 1.4, 1.6,  3.7, 0.93)
    arrow(ax, 7.0, 0.93, 8.6, 1.6,  col="#e67e22")
    arrow(ax, 8.6, 2.25, 8.6, 2.7,  col="#e67e22")
    arrow(ax, 8.6, 3.35, 8.6, 3.8,  col="#e67e22")

    for sy in [4.075, 2.975, 1.875]:
        ax.annotate("", xy=(7.5, sy), xytext=(2.5, sy),
                    arrowprops=dict(arrowstyle="->", color="#aaa",
                                   lw=0.9, linestyle="dashed"))

    if not inject_all:
        ax.text(-0.1, 4.35, "peak_map\n(B,3,206)", ha="center", va="center",
                fontsize=7.5, color="#27ae60", fontweight="bold")
        arrow(ax, 0.45, 4.25, 0.7, 4.07, col="#27ae60")
        ax.text(5.0, 0.22, "peak position lost by bottleneck  ✗",
                ha="center", fontsize=8, color="#c0392b", style="italic")
    else:
        inject_pts = [(1.4, 4.075, "L=206"), (1.4, 2.975, "L=103"),
                      (1.4, 1.875, "L=51"),  (5.0, 0.875, "L=25")]
        for (ix, iy, ilab) in inject_pts:
            ax.annotate("", xy=(ix, iy), xytext=(ix - 1.1, iy),
                        arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.5))
            ax.text(ix - 1.15, iy, ilab, ha="right", va="center",
                    fontsize=7.5, color="#27ae60", fontweight="bold")
        ax.text(-0.6, 5.0, "peak_map\n(re-sampled\nper level)", ha="center", va="center",
                fontsize=7.5, color="#27ae60", fontweight="bold")
        ax.text(5.0, 0.22, "peak position preserved at every scale  ✓",
                ha="center", fontsize=8, color="#27ae60", style="italic")

draw_arch(ax_old, inject_all=False, title="Before — single injection at input")
draw_arch(ax_new, inject_all=True,  title="After — multi-scale injection")

plt.suptitle("Multi-scale peak map injection for temporal controllability",
             fontsize=13, fontweight="bold", y=0.99)

plt.savefig("/Users/michal/Desktop/PhD/dvl paper/DATA/multiscale_injection_explainer.png",
            dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.show()
print("Saved.")
