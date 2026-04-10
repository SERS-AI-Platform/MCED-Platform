"""
Figure 1: Study pipeline schematic.
Urine collection → SERS measurement + AI model → Cancer-specific detection
with mini spectral panels showing discriminative peaks per cancer type.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from nature_style import (
    apply_style, apply_nature_style, save_figure,
    load_spectra, compute_group_means,
    CANCER_COLORS, AACR_LABELS, NON_CANCER_COLOR, NON_CANCER_GROUPS, PEAK_REGIONS,
    DOUBLE_COL, FONT_SIZE, LINE_WIDTH,
)

apply_style()

# ── Load real spectral data for mini-insets ──
spec, wavenumbers, wn_cols = load_spectra()

cancer_order = ["PRO", "OVA", "LUN", "PAN", "CRC"]
groups_map = {g: "Non-Cancer" for g in NON_CANCER_GROUPS}
groups_map["YNOR"] = "Non-Cancer"
groups_map["CPAN"] = "PAN"
for g in cancer_order:
    groups_map[g] = g

stats = compute_group_means(spec, wn_cols, groups_map)

peak_map = {p[0]: p for p in PEAK_REGIONS}

# ── Figure layout ──
fig_w = DOUBLE_COL
fig_h = DOUBLE_COL * 0.48
fig = plt.figure(figsize=(fig_w, fig_h))

# Main schematic axis (top portion)
ax_main = fig.add_axes([0.0, 0.38, 1.0, 0.62])
ax_main.set_xlim(0, 10)
ax_main.set_ylim(0, 3.5)
ax_main.axis("off")

# ── Style params ──
box_fc = "#F5F5F5"
box_ec = "#CCCCCC"
text_color = "#333333"
sub_color = "#777777"
arrow_color = "#999999"

# ── Box 1: Urine Collection ──
b1 = FancyBboxPatch((0.2, 1.2), 2.2, 1.6, boxstyle="round,pad=0.15",
                     facecolor=box_fc, edgecolor=box_ec, linewidth=0.8)
ax_main.add_patch(b1)
ax_main.text(1.3, 2.55, "Urine Collection", ha="center", va="center",
             fontsize=FONT_SIZE["axis_label"], fontweight="bold", color=text_color)
ax_main.text(1.3, 2.05, "N = 1,240 patients\n9 cohorts\n5 cancer types + 4 controls",
             ha="center", va="center", fontsize=FONT_SIZE["annotation"],
             color=sub_color, linespacing=1.4)

# ── Arrow 1→2 ──
ax_main.annotate("", xy=(3.0, 2.0), xytext=(2.5, 2.0),
                 arrowprops=dict(arrowstyle="->", color=arrow_color,
                                 lw=1.0, mutation_scale=12))

# ── Box 2: SERS Measurement ──
b2 = FancyBboxPatch((3.0, 1.2), 2.5, 1.6, boxstyle="round,pad=0.15",
                     facecolor=box_fc, edgecolor=box_ec, linewidth=0.8)
ax_main.add_patch(b2)
ax_main.text(4.25, 2.55, "SERS Measurement", ha="center", va="center",
             fontsize=FONT_SIZE["axis_label"], fontweight="bold", color=text_color)
ax_main.text(4.25, 2.05, "Gold nanoparticle substrate\nPortable Raman (785 nm)\n5 replicates / sample",
             ha="center", va="center", fontsize=FONT_SIZE["annotation"],
             color=sub_color, linespacing=1.4)

# ── Arrow 2→3 ──
ax_main.annotate("", xy=(6.1, 2.0), xytext=(5.6, 2.0),
                 arrowprops=dict(arrowstyle="->", color=arrow_color,
                                 lw=1.0, mutation_scale=12))

# ── Box 3: AI Classification ──
b3 = FancyBboxPatch((6.1, 1.2), 3.5, 1.6, boxstyle="round,pad=0.15",
                     facecolor=box_fc, edgecolor=box_ec, linewidth=0.8)
ax_main.add_patch(b3)
ax_main.text(7.85, 2.55, "Two-Stage AI Classifier", ha="center", va="center",
             fontsize=FONT_SIZE["axis_label"], fontweight="bold", color=text_color)

# Two sub-boxes inside Box 3
s1_box = FancyBboxPatch((6.35, 1.6), 1.5, 0.6, boxstyle="round,pad=0.08",
                        facecolor="#E8EAF6", edgecolor="#9FA8DA", linewidth=0.5)
ax_main.add_patch(s1_box)
ax_main.text(7.1, 1.9, "Stage 1\nCancer vs Normal", ha="center", va="center",
             fontsize=FONT_SIZE["annotation"], color="#3F51B5", linespacing=1.3)

s2_box = FancyBboxPatch((8.1, 1.6), 1.3, 0.6, boxstyle="round,pad=0.08",
                        facecolor="#FFF3E0", edgecolor="#FFCC80", linewidth=0.5)
ax_main.add_patch(s2_box)
ax_main.text(8.75, 1.9, "Stage 2\nCancer Type", ha="center", va="center",
             fontsize=FONT_SIZE["annotation"], color="#E65100", linespacing=1.3)

# Small arrow between sub-boxes
ax_main.annotate("", xy=(8.05, 1.9), xytext=(7.9, 1.9),
                 arrowprops=dict(arrowstyle="->", color="#999", lw=0.6, mutation_scale=8))

# ── Step numbers ──
for i, (x, y) in enumerate([(0.35, 2.95), (3.15, 2.95), (6.25, 2.95)]):
    circle = plt.Circle((x, y), 0.15, color="#AAAAAA", zorder=5)
    ax_main.add_patch(circle)
    ax_main.text(x, y, str(i + 1), ha="center", va="center",
                 fontsize=FONT_SIZE["annotation"], fontweight="bold",
                 color="white", zorder=6)

# ── Downward arrow from Box 3 to spectra panels ──
ax_main.annotate("", xy=(7.85, 0.85), xytext=(7.85, 1.15),
                 arrowprops=dict(arrowstyle="->", color=arrow_color,
                                 lw=1.0, mutation_scale=12))
ax_main.text(7.85, 0.7, "Cancer-Specific Spectral Signatures", ha="center",
             va="center", fontsize=FONT_SIZE["annotation"], color=sub_color,
             fontstyle="italic")

# ── Bottom row: 5 mini spectral panels ──
panel_width = 0.17
panel_height = 0.30
panel_y = 0.04
panel_starts = [0.03 + i * 0.195 for i in range(5)]

for i, cancer in enumerate(cancer_order):
    ax_sp = fig.add_axes([panel_starts[i], panel_y, panel_width, panel_height])

    color = CANCER_COLORS[cancer]
    nc = stats["Non-Cancer"]
    cs = stats[cancer]

    # Downsample for cleaner mini-plot
    step = 3
    wn_ds = wavenumbers[::step]
    nc_ds = nc["mean"][::step]
    cs_ds = cs["mean"][::step]

    ax_sp.plot(wn_ds, nc_ds, color=NON_CANCER_COLOR, linewidth=0.4, alpha=0.5)
    ax_sp.plot(wn_ds, cs_ds, color=color, linewidth=0.7)

    # Shade peak region
    _, lo, hi, wn_label = peak_map[cancer]
    ax_sp.axvspan(lo, hi, color=color, alpha=0.25, zorder=0)

    # Minimal formatting
    ax_sp.set_title(AACR_LABELS[cancer], fontsize=FONT_SIZE["annotation"],
                    fontweight="bold", color=color, pad=2)
    ax_sp.set_xlim(wavenumbers.min(), wavenumbers.max())
    ax_sp.tick_params(labelsize=5, length=1.5, width=0.3, pad=1)
    ax_sp.spines["top"].set_visible(False)
    ax_sp.spines["right"].set_visible(False)
    ax_sp.spines["left"].set_linewidth(0.3)
    ax_sp.spines["bottom"].set_linewidth(0.3)

    # Only label leftmost
    if i == 0:
        ax_sp.set_ylabel("a.u.", fontsize=5, labelpad=1)
    else:
        ax_sp.set_yticklabels([])

    # X label
    if i == 2:
        ax_sp.set_xlabel("Raman shift (cm$^{-1}$)", fontsize=5.5, labelpad=1)

save_figure(fig, "fig1_pipeline")
plt.close()
print("Figure 1 complete.")
