# %%
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

fig, ax = plt.subplots(figsize=(12, 9))
ax.set_xlim(0, 12); ax.set_ylim(0, 9); ax.axis("off")

# ── colour scheme ─────────────────────────────────────────────────────────────
C_HUB   = "#2c7bb6"   # blue  — hub node (std_vx)
C_TIGHT = "#d7191c"   # red   — tightly coupled
C_MED   = "#fdae61"   # amber — moderately coupled
C_FREE  = "#1a9641"   # green — free / independent
C_ARROW_TIGHT = "#d7191c"
C_ARROW_MED   = "#f4a442"
C_ARROW_LOOSE = "#aaaaaa"

def node(ax, x, y, label, sublabel, color, step=None, w=2.0, h=0.75):
    box = mpatches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                                   boxstyle="round,pad=0.08",
                                   facecolor=color, edgecolor="white",
                                   linewidth=2, zorder=3)
    ax.add_patch(box)
    if step:
        ax.text(x - w/2 + 0.12, y + h/2 - 0.12, str(step),
                fontsize=7, color="white", fontweight="bold",
                va="top", ha="left", zorder=4)
    ax.text(x, y + 0.08, label, ha="center", va="center",
            fontsize=11, fontweight="bold", color="white", zorder=4)
    ax.text(x, y - 0.22, sublabel, ha="center", va="center",
            fontsize=7.5, color="white", alpha=0.9, zorder=4)

def arrow(ax, x1, y1, x2, y2, label, color, lw=2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=14),
                zorder=2)
    mx, my = (x1+x2)/2, (y1+y2)/2
    # offset label slightly perpendicular to arrow
    dx, dy = x2-x1, y2-y1
    length = (dx**2+dy**2)**0.5
    ox, oy = -dy/length*0.28, dx/length*0.28
    ax.text(mx+ox, my+oy, label, ha="center", va="center",
            fontsize=8, color=color, fontweight="bold", zorder=5)

# ── node positions ────────────────────────────────────────────────────────────
#  Row 1: hub
ax.text(6, 8.6, "Tutorial order: set tightly coupled conditions first",
        ha="center", va="center", fontsize=13, fontweight="bold", color="#333333")

node(ax, 6,   7.5, "std_vx",   "SET FIRST — hub variable", C_HUB,   step=1)

#  Row 2: directly coupled to std_vx
node(ax, 2.5, 5.5, "std_vy",   "must be ≈ std_vx (±0.25)", C_TIGHT, step=2)
node(ax, 6,   5.5, "mean_vz",  "commits kurt_vz range",     C_TIGHT, step=3)
node(ax, 9.5, 5.5, "std_vz",   "low vx → low vz (funnel)",  C_MED,   step=5)

#  Row 3: kurt_vz depends on both std_vx and mean_vz
node(ax, 6,   3.5, "kurt_vz",  "constrained by std_vx & mean_vz", C_MED, step=4)

#  Row 4: free variables
node(ax, 1,   1.5, "mean_vx",  "fully free",      C_FREE, step=6)
node(ax, 3.5, 1.5, "mean_vy",  "fully free",      C_FREE, step=7)
node(ax, 6.5, 1.5, "kurt_vx",  "fully free",      C_FREE, step=8)
node(ax, 9.5, 1.5, "kurt_vy",  "fully free",      C_FREE, step=9)

# ── arrows ────────────────────────────────────────────────────────────────────
# std_vx → std_vy  (tight)
arrow(ax, 5.0, 7.2, 3.1, 5.85, "r=0.66", C_ARROW_TIGHT, lw=2.5)
# std_vx → std_vz  (medium, funnel)
arrow(ax, 7.0, 7.2, 8.9, 5.85, "r=0.55\n(funnel)", C_ARROW_MED, lw=2)
# std_vx → kurt_vz (medium, main driver)
arrow(ax, 6.0, 7.12, 6.0, 5.88, "r=0.60\nmain driver", C_ARROW_MED, lw=2)
# mean_vz → kurt_vz
arrow(ax, 6.0, 5.12, 6.0, 3.88, "r=0.62", C_ARROW_TIGHT, lw=2.5)

# ── legend ────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(color=C_HUB,   label="Hub — most connected"),
    mpatches.Patch(color=C_TIGHT, label="Tightly coupled  (r > 0.60)"),
    mpatches.Patch(color=C_MED,   label="Moderately coupled  (r ≈ 0.40–0.60)"),
    mpatches.Patch(color=C_FREE,  label="Free — independent"),
]
ax.legend(handles=legend_items, loc="lower left", fontsize=9,
          framealpha=0.9, edgecolor="#cccccc")

# ── footnote ─────────────────────────────────────────────────────────────────
ax.text(6, 0.3,
        "kurt_vx, kurt_vy, kurt_vz are mutually independent  •  "
        "std_vy ↔ std_vz are decoupled",
        ha="center", va="center", fontsize=8.5, color="#666666", style="italic")

plt.tight_layout()
plt.savefig("/Users/michal/Desktop/PhD/dvl paper/conditioning_flowchart.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("Saved → conditioning_flowchart.png")

# %%
