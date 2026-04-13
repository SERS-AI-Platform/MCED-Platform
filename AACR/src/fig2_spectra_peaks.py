"""
Figure 2: Cancer-specific SERS spectra with discriminative peak shading.
Each panel shows one cancer type (colored) vs Non-Cancer control (gray),
with the cancer-specific peak region highlighted.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from nature_style import (
    apply_style, apply_nature_style, add_panel_label, save_figure,
    load_spectra, compute_group_means,
    CANCER_COLORS, CANCER_LABELS, NON_CANCER_COLOR, PEAK_REGIONS,
    DOUBLE_COL, FONT_SIZE, LINE_WIDTH,
)

apply_style()

# ── Load data ──
spec, wavenumbers, wn_cols = load_spectra()
stats = compute_group_means(spec, wn_cols)

# ── Panel order ──
cancer_order = ["PRO", "OVA", "LUN", "PAN", "CRC"]
peak_map = {p[0]: p for p in PEAK_REGIONS}

# ── Create figure: 1 row x 5 columns ──
fig, axes = plt.subplots(1, 5, figsize=(DOUBLE_COL, DOUBLE_COL * 0.28),
                         sharey=True)
fig.subplots_adjust(wspace=0.08)

for i, (cancer, ax) in enumerate(zip(cancer_order, axes)):
    label = chr(ord("a") + i)
    add_panel_label(ax, label)

    # Non-cancer (gray background)
    nc = stats["Non-Cancer"]
    ax.fill_between(wavenumbers, nc["mean"] - nc["sem"], nc["mean"] + nc["sem"],
                    color=NON_CANCER_COLOR, alpha=0.12, linewidth=0)
    ax.plot(wavenumbers, nc["mean"], color=NON_CANCER_COLOR,
            linewidth=LINE_WIDTH["thin"], alpha=0.6, label="Non-Cancer")

    # Cancer spectrum
    cs = stats[cancer]
    color = CANCER_COLORS[cancer]
    ax.fill_between(wavenumbers, cs["mean"] - cs["sem"], cs["mean"] + cs["sem"],
                    color=color, alpha=0.15, linewidth=0)
    ax.plot(wavenumbers, cs["mean"], color=color,
            linewidth=LINE_WIDTH["spectrum"], label=CANCER_LABELS[cancer])

    # Shade discriminative peak region
    _, lo, hi, wn_label = peak_map[cancer]
    ax.axvspan(lo, hi, color=color, alpha=0.20, zorder=0)
    # Peak annotation arrow
    mid_wn = (lo + hi) / 2
    # Find intensity at peak for this cancer
    idx = np.argmin(np.abs(wavenumbers - mid_wn))
    peak_y = cs["mean"][idx]
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0] if ax.get_ylim()[1] > ax.get_ylim()[0] else 1
    ax.annotate(
        f"{wn_label} cm$^{{-1}}$",
        xy=(mid_wn, peak_y),
        xytext=(mid_wn, peak_y + 0.8),
        fontsize=FONT_SIZE["annotation"],
        color=color, ha="center",
        arrowprops=dict(arrowstyle="-", color=color, lw=0.5),
    )

    # Panel title
    ax.set_title(CANCER_LABELS[cancer], fontsize=FONT_SIZE["title"],
                 fontweight="bold", color=color, pad=8)

    # X-axis label only on bottom
    ax.set_xlabel("Raman shift (cm$^{-1}$)")
    ax.set_xlim(wavenumbers.min(), wavenumbers.max())

    # Y-axis label on first panel only
    if i == 0:
        ax.set_ylabel("Intensity (a.u.)")

    # Legend on first panel
    if i == 0:
        ax.legend(loc="upper right", frameon=False, fontsize=FONT_SIZE["annotation"])

    apply_nature_style(ax)

# Adjust y limits after all panels plotted
all_means = np.concatenate([stats[c]["mean"] for c in cancer_order] + [stats["Non-Cancer"]["mean"]])
y_lo = all_means.min() - 0.3
y_hi = all_means.max() + 1.5
for ax in axes:
    ax.set_ylim(y_lo, y_hi)

# Re-annotate peaks with correct y positioning
for i, (cancer, ax) in enumerate(zip(cancer_order, axes)):
    _, lo, hi, wn_label = peak_map[cancer]
    mid_wn = (lo + hi) / 2
    idx = np.argmin(np.abs(wavenumbers - mid_wn))
    peak_y = stats[cancer]["mean"][idx]
    # Clear old annotations and re-add
    for child in list(ax.texts):
        if "cm" in child.get_text():
            child.remove()
    for child in list(ax.patches):
        pass  # keep patches
    ax.annotate(
        f"{wn_label} cm$^{{-1}}$",
        xy=(mid_wn, peak_y),
        xytext=(mid_wn, peak_y + (y_hi - y_lo) * 0.15),
        fontsize=FONT_SIZE["annotation"],
        color=CANCER_COLORS[cancer], ha="center",
        arrowprops=dict(arrowstyle="-", color=CANCER_COLORS[cancer], lw=0.5),
    )

save_figure(fig, "fig2_spectra_peaks")
plt.close()
print("Figure 2 complete.")
