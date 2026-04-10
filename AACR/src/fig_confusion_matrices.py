"""
AACR Figure: Confusion Matrices (Nature style)
A. Cancer Screening (2x2)
B. Cancer Type Identification (5x5)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from nature_style import (
    apply_style, save_figure, CANCER_COLORS,
    DOUBLE_COL, FONT_SIZE, LINE_WIDTH,
)

apply_style()

# ── Load data ──
AACR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data = np.load(os.path.join(AACR_DIR, "data", "confusion_matrices.npz"),
               allow_pickle=True)
cm1 = data["cm1"]
cm2 = data["cm2"]
auc_mean = float(data["auc"])
f1_mean = float(data["f1"])

# Labels
S1_LABELS = ["Non-Cancer", "Cancer"]
S2_LABELS = ["PRC", "OVC", "LC", "PAC", "CRC"]
S2_KEYS = ["PRO", "OVA", "LUN", "PAN", "CRC"]

# Muted diagonal colors
S1_DIAG = ["#6B8EAD", "#3D6E99"]
S2_DIAG = [CANCER_COLORS[k] for k in S2_KEYS]

# Percentages
cm1_pct = cm1 / cm1.sum(axis=1, keepdims=True) * 100
cm2_pct = cm2 / cm2.sum(axis=1, keepdims=True) * 100


def draw_cm(ax, pct, labels, diag_colors, fs_main, fs_sub):
    n = len(labels)
    for i in range(n):
        for j in range(n):
            p = pct[i, j]
            if i == j:
                fc = diag_colors[i]
                tc = "white"
            else:
                intensity = min(p / 25.0, 1.0)
                g = 0.97 - intensity * 0.22
                fc = (g, g, g)
                tc = "#444444" if p > 1.0 else "#bbbbbb"

            rect = plt.Rectangle((j, n - 1 - i), 1, 1,
                                  facecolor=fc, edgecolor="white", linewidth=1.5)
            ax.add_patch(rect)

            txt = f"{p:.1f}%" if p >= 0.5 else ("0%" if p == 0 else f"{p:.1f}%")
            ax.text(j + 0.5, n - 1 - i + 0.5, txt,
                    ha="center", va="center",
                    fontsize=fs_main, fontweight="bold", color=tc)

    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect("equal")
    ax.set_xticks([i + 0.5 for i in range(n)])
    ax.set_xticklabels(labels, fontsize=fs_sub, rotation=0, ha="center")
    ax.set_yticks([i + 0.5 for i in range(n)])
    ax.set_yticklabels(reversed(labels), fontsize=fs_sub)
    ax.set_xlabel("Predicted", fontsize=FONT_SIZE["axis_label"], labelpad=6)
    ax.set_ylabel("Actual", fontsize=FONT_SIZE["axis_label"], labelpad=6)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)


# ── Figure ──
fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.45))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 2.2], wspace=0.45,
                      left=0.08, right=0.95, top=0.88, bottom=0.20)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

# Panel A: Cancer Screening
draw_cm(ax1, cm1_pct, S1_LABELS, S1_DIAG,
        fs_main=FONT_SIZE["title"], fs_sub=FONT_SIZE["axis_label"])
ax1.set_title("Cancer Screening", fontsize=FONT_SIZE["title"],
              fontweight="bold", pad=8)

sens = cm1[1,1] / cm1[1].sum()
spec = cm1[0,0] / cm1[0].sum()
ax1.text(0.5, -0.28, f"AUC = {auc_mean:.3f}   Sens = {sens:.1%}   Spec = {spec:.1%}",
         transform=ax1.transAxes, ha="center",
         fontsize=FONT_SIZE["annotation"], color="#666666")

# Panel B: Cancer Type ID
draw_cm(ax2, cm2_pct, S2_LABELS, S2_DIAG,
        fs_main=FONT_SIZE["axis_label"], fs_sub=FONT_SIZE["axis_label"])
ax2.set_title("Cancer Type Identification", fontsize=FONT_SIZE["title"],
              fontweight="bold", pad=8)
ax2.text(0.5, -0.28, f"Macro F1 = {f1_mean:.3f}",
         transform=ax2.transAxes, ha="center",
         fontsize=FONT_SIZE["annotation"], color="#666666")

save_figure(fig, "fig_confusion_matrices")
plt.close()
print("Done.")
