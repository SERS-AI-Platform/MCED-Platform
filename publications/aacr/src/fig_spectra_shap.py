"""
AACR Figure: Representative SERS spectra per cancer type vs Non-Cancer control,
with top-5 SHAP-important wavenumber peaks highlighted.

SHAP is computed from the binary LR model (cancer vs non-cancer).
For linear models: SHAP_i = coef_i * (x_i - E[x_i])
Per-cancer-type averaging gives cancer-specific discriminative peaks.

Cancer types: PRO(PRC), OVA(OVC), LUN(LC), CRC, CPAN(PAC)
Non-Cancer: NOR, DIA, HBP, H.D.
Layout: vertical (6 rows x 1 col)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from nature_style import (
    apply_style, apply_nature_style, add_panel_label, save_figure,
    load_spectra, compute_group_means,
    CANCER_COLORS, NON_CANCER_COLOR, NON_CANCER_GROUPS,
    DOUBLE_COL, FONT_SIZE, LINE_WIDTH,
)

apply_style()

# ── Config (AACR 5-cancer) ──
CANCER_ORDER = ["PRO", "OVA", "LUN", "CRC", "PAN"]
DISPLAY_LABELS = {
    "PRO": "PRC",
    "OVA": "OVC",
    "LUN": "LC",
    "CRC": "CRC",
    "PAN": "PAC",
}
N_TOP_PEAKS = 5
PEAK_GROUP_RADIUS = 15  # cm-1: merge peaks within this distance
SERS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MODEL_DIR = os.path.join(
    SERS_ROOT,
    "models", "results", "01_benchmarks", "main_5models",
    "logistic_regression", "v004", "checkpoints",
)
N_FOLDS = 5

# ── Load spectra ──
spec, wavenumbers, wn_cols = load_spectra()

# Groups map: CPAN only for PAC (no YPAN, no SPAN)
groups_map = {}
for g in NON_CANCER_GROUPS:
    groups_map[g] = "Non-Cancer"
groups_map["YNOR"] = "Non-Cancer"
groups_map["CPAN"] = "PAN"
for g in CANCER_ORDER:
    groups_map[g] = g

stats = compute_group_means(spec, wn_cols, groups_map)

# ── Compute per-cancer SHAP from binary LR ──
model_wn_cols = wn_cols[1:-1]
model_wavenumbers = wavenumbers[1:-1]

coefs = []
for fold in range(N_FOLDS):
    path = os.path.join(MODEL_DIR, f"fold_{fold}_binary.joblib")
    pipe = joblib.load(path)
    lr = pipe.named_steps["logisticregression"]
    coefs.append(lr.coef_[0])
avg_coef = np.mean(coefs, axis=0)

# Global mean spectrum
spec_copy = spec.copy()
spec_copy["display_group"] = spec_copy["group"].map(groups_map)
spec_copy = spec_copy.dropna(subset=["display_group"])
spec_med = spec_copy.groupby(["display_group", "sample_id"]).first().reset_index()
global_mean = spec_med[model_wn_cols].values.mean(axis=0)


def find_top_peak_regions(shap_vals, wavenumbers, n_peaks=5, merge_radius=15):
    """Find top N peak regions, merging adjacent high-SHAP wavenumbers."""
    abs_shap = np.abs(shap_vals)
    sorted_idx = np.argsort(abs_shap)[::-1]

    regions = []
    used = set()
    for idx in sorted_idx:
        if len(regions) >= n_peaks:
            break
        wn = wavenumbers[idx]
        # Skip if too close to an already-selected region
        if any(abs(wn - r["wn"]) < merge_radius for r in regions):
            continue
        # Find the extent of this peak cluster
        cluster = [idx]
        for j in sorted_idx:
            if j in used:
                continue
            if abs(wavenumbers[j] - wn) < merge_radius:
                cluster.append(j)
                used.add(j)
        # Representative = highest |SHAP| in cluster
        best = max(cluster, key=lambda j: abs_shap[j])
        regions.append({
            "idx": best,
            "wn": wavenumbers[best],
            "shap": shap_vals[best],
            "cluster_lo": wavenumbers[min(cluster, key=lambda j: wavenumbers[j])],
            "cluster_hi": wavenumbers[max(cluster, key=lambda j: wavenumbers[j])],
        })
        used.add(idx)
    return regions


# Per-cancer SHAP
cancer_shap = {}
for cancer in CANCER_ORDER:
    if cancer not in stats:
        continue
    cancer_mean = stats[cancer]["mean"][1:-1]
    shap_vals = avg_coef * (cancer_mean - global_mean)
    regions = find_top_peak_regions(shap_vals, model_wavenumbers, N_TOP_PEAKS, PEAK_GROUP_RADIUS)
    cancer_shap[cancer] = regions

# ── Create figure: 6 rows x 1 col (vertical) ──
n_panels = len(CANCER_ORDER) + 1
row_height = 1.1  # inches per row
fig, axes = plt.subplots(n_panels, 1,
                         figsize=(DOUBLE_COL, row_height * n_panels),
                         sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.15)

panel_labels = [chr(ord("a") + i) for i in range(n_panels)]

# Helper: place annotations avoiding overlap
def annotate_peaks(ax, regions, color, y_top, y_range, cancer_mean_full):
    """Place peak labels at the top of the panel, spread to avoid overlap."""
    if not regions:
        return
    # Sort by wavenumber for left-to-right layout
    regions_sorted = sorted(regions, key=lambda r: r["wn"])

    # Place text at fixed y near top, stagger if needed
    text_y = y_top - y_range * 0.05
    min_text_gap = 120  # minimum cm-1 between text centers

    # Compute text x positions with repulsion
    raw_x = [r["wn"] for r in regions_sorted]
    text_x = list(raw_x)

    # Simple repulsion: push apart if too close
    for _ in range(20):
        for j in range(len(text_x) - 1):
            gap = text_x[j + 1] - text_x[j]
            if gap < min_text_gap:
                push = (min_text_gap - gap) / 2
                text_x[j] -= push
                text_x[j + 1] += push

    for r, tx in zip(regions_sorted, text_x):
        wn = r["wn"]
        # Find y at peak position in full spectrum
        full_idx = np.argmin(np.abs(wavenumbers - wn))
        peak_y = cancer_mean_full[full_idx]

        # Draw vertical line from peak to near top
        ax.plot([wn, wn], [peak_y + y_range * 0.03, text_y - y_range * 0.08],
                color=color, lw=0.5, alpha=0.5)
        # Horizontal connector if text shifted
        if abs(tx - wn) > 5:
            connector_y = text_y - y_range * 0.08
            ax.plot([wn, tx], [connector_y, connector_y],
                    color=color, lw=0.4, alpha=0.4)

        # Direction indicator
        direction = "+" if r["shap"] > 0 else "-"

        ax.text(tx, text_y, f'{wn:.0f}',
                fontsize=FONT_SIZE["annotation"], color=color,
                ha="center", va="top", fontweight="bold")


# Plot 5 cancer panels
for i, cancer in enumerate(CANCER_ORDER):
    ax = axes[i]
    add_panel_label(ax, panel_labels[i], x=-0.06, y=1.05)

    color = CANCER_COLORS[cancer]
    cs = stats[cancer]
    nc = stats["Non-Cancer"]
    dlabel = DISPLAY_LABELS[cancer]

    # Non-cancer (gray)
    ax.fill_between(wavenumbers, nc["mean"] - nc["sem"], nc["mean"] + nc["sem"],
                    color=NON_CANCER_COLOR, alpha=0.10, linewidth=0)
    ax.plot(wavenumbers, nc["mean"], color=NON_CANCER_COLOR,
            linewidth=LINE_WIDTH["thin"], alpha=0.5, label="NOR")

    # Cancer spectrum
    ax.fill_between(wavenumbers, cs["mean"] - cs["sem"], cs["mean"] + cs["sem"],
                    color=color, alpha=0.15, linewidth=0)
    ax.plot(wavenumbers, cs["mean"], color=color,
            linewidth=LINE_WIDTH["spectrum"],
            label=f'{dlabel} (n={cs["n"]})')

    # Shade SHAP peak regions
    regions = cancer_shap.get(cancer, [])
    for r in regions:
        lo = r["cluster_lo"] - 5
        hi = r["cluster_hi"] + 5
        ax.axvspan(lo, hi, color=color, alpha=0.12, zorder=0)

    # Legend inside panel (right side)
    ax.legend(loc="upper right", frameon=False, fontsize=FONT_SIZE["legend"],
              handlelength=1.2)

    # Y label with cancer name
    ax.set_ylabel("Intensity (a.u.)", fontsize=FONT_SIZE["axis_label"])

    # Panel title on right
    ax.text(0.98, 0.92, dlabel, transform=ax.transAxes,
            fontsize=FONT_SIZE["title"] + 1, fontweight="bold",
            color=color, ha="right", va="top")

    apply_nature_style(ax)

# Last panel: Non-Cancer reference
ax = axes[-1]
add_panel_label(ax, panel_labels[-1], x=-0.06, y=1.05)
nc = stats["Non-Cancer"]
ax.fill_between(wavenumbers, nc["mean"] - nc["sem"], nc["mean"] + nc["sem"],
                color=NON_CANCER_COLOR, alpha=0.15, linewidth=0)
ax.plot(wavenumbers, nc["mean"], color=NON_CANCER_COLOR,
        linewidth=LINE_WIDTH["spectrum"],
        label=f'NOR (n={nc["n"]})')
ax.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=FONT_SIZE["axis_label"])
ax.set_ylabel("Intensity (a.u.)", fontsize=FONT_SIZE["axis_label"])
ax.legend(loc="upper right", frameon=False, fontsize=FONT_SIZE["legend"],
          handlelength=1.2)
ax.text(0.98, 0.92, "NOR", transform=ax.transAxes,
        fontsize=FONT_SIZE["title"] + 1, fontweight="bold",
        color=NON_CANCER_COLOR, ha="right", va="top")
apply_nature_style(ax)

# Axis limits
for ax in axes:
    ax.set_xlim(wavenumbers.min(), wavenumbers.max())

all_means = np.concatenate([stats[c]["mean"] for c in CANCER_ORDER if c in stats]
                           + [stats["Non-Cancer"]["mean"]])
y_lo = all_means.min() - 0.3
y_hi = all_means.max() + 2.0
for ax in axes:
    ax.set_ylim(y_lo, y_hi)

# Add peak annotations (after y limits set)
for i, cancer in enumerate(CANCER_ORDER):
    if cancer not in cancer_shap:
        continue
    ax = axes[i]
    regions = cancer_shap[cancer]
    color = CANCER_COLORS[cancer]
    cs = stats[cancer]
    annotate_peaks(ax, regions, color, y_hi, y_hi - y_lo, cs["mean"])

save_figure(fig, "fig_spectra_shap")
plt.close()

# Print summary
print("\n=== Top-5 SHAP peak regions per cancer type ===")
for cancer in CANCER_ORDER:
    if cancer not in cancer_shap:
        continue
    dlabel = DISPLAY_LABELS[cancer]
    print(f"\n{dlabel}:")
    for rank, r in enumerate(sorted(cancer_shap[cancer], key=lambda x: -abs(x["shap"]))):
        d = "+" if r["shap"] > 0 else "-"
        print(f"  {rank+1}. {r['wn']:.0f} cm-1 [{r['cluster_lo']:.0f}-{r['cluster_hi']:.0f}] "
              f"(SHAP={r['shap']:+.4f}, {d})")
