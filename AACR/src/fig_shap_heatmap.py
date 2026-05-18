"""
Figure: SHAP Heatmap — Common vs Cancer-Specific Spectral Signatures

Heatmap showing SHAP values (wavenumber peaks × cancer types).
Reveals which spectral features are shared across cancers (common metabolic
signatures) and which are unique to specific cancer types.

Replaces fig_shap_bars.py for AACR poster.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from nature_style import (
    apply_style, apply_nature_style, save_figure,
    load_spectra, compute_group_means,
    CANCER_COLORS, NON_CANCER_COLOR, NON_CANCER_GROUPS,
    AACR_LABELS, AACR_INTERNAL_ORDER, AACR_CANCER_ORDER,
    DOUBLE_COL, SINGLE_COL, FONT_SIZE, LINE_WIDTH,
)

apply_style()

# ── Config ──
CANCER_ORDER = AACR_INTERNAL_ORDER  # PRO, OVA, LUN, PAN, CRC
N_TOP_PER_CANCER = 5
N_DISPLAY_PEAKS = 15  # total unique peaks to show
PEAK_MERGE_RADIUS = 15  # cm-1
SERS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(SERS_ROOT, "models", "production", "stage1_fusion.joblib")
N_SERS_FEATURES = 933  # matches current processed_spectra.csv

# ── Load spectra ──
PRE_QC_PATH = os.path.join(SERS_ROOT, "results", "processed_spectra.csv")
spec = pd.read_csv(PRE_QC_PATH)
wn_cols = [c for c in spec.columns if c.startswith("x_")]
wavenumbers = np.array([float(c.replace("x_", "")) for c in wn_cols])

groups_map = {}
for g in NON_CANCER_GROUPS:
    groups_map[g] = "Non-Cancer"
groups_map["YNOR"] = "Non-Cancer"
groups_map["CPAN"] = "PAN"
for g in CANCER_ORDER:
    groups_map[g] = g

stats = compute_group_means(spec, wn_cols, groups_map)

# ── SHAP from Stage 1 Fusion LR (SERS portion only) ──
model_wn_cols = wn_cols  # all SERS features
model_wavenumbers = wavenumbers

pipe = joblib.load(MODEL_PATH)
coef_all = pipe.named_steps["logisticregression"].coef_[0]
coef_sers = coef_all[:N_SERS_FEATURES]

# Global mean
spec_copy = spec.copy()
spec_copy["display_group"] = spec_copy["group"].map(groups_map)
spec_copy = spec_copy.dropna(subset=["display_group"])
spec_med = spec_copy.groupby(["display_group", "sample_id"]).first().reset_index()
global_mean = spec_med[model_wn_cols].values.mean(axis=0)

# ── Compute SHAP per cancer ──
shap_matrix = {}  # cancer -> full SHAP array (933,)
for cancer in CANCER_ORDER:
    if cancer not in stats:
        continue
    cancer_mean = stats[cancer]["mean"]
    shap_matrix[cancer] = coef_sers * (cancer_mean - global_mean)


def find_top_peaks(shap_vals, wns, n=5, radius=15):
    """Top N peaks by absolute SHAP magnitude with merging."""
    abs_shap = np.abs(shap_vals)
    regions = []
    for idx in np.argsort(abs_shap)[::-1]:
        if len(regions) >= n:
            break
        wn = wns[idx]
        if any(abs(wn - r["wn"]) < radius for r in regions):
            continue
        regions.append({"idx": idx, "wn": wn})
    return regions


# ── Collect unique peak wavenumbers across all cancers ──
all_peaks = []
seen_wns = set()
for cancer in CANCER_ORDER:
    peaks = find_top_peaks(shap_matrix[cancer], model_wavenumbers,
                           n=N_TOP_PER_CANCER, radius=PEAK_MERGE_RADIUS)
    for p in peaks:
        # Check if this peak is already represented (within merge radius)
        wn = p["wn"]
        if any(abs(wn - s) < PEAK_MERGE_RADIUS for s in seen_wns):
            continue
        seen_wns.add(wn)
        all_peaks.append(p)

# Sort by wavenumber for spectral order
all_peaks.sort(key=lambda x: x["wn"])

# Limit to N_DISPLAY_PEAKS (pick those with highest max |SHAP| across cancers)
if len(all_peaks) > N_DISPLAY_PEAKS:
    for p in all_peaks:
        p["max_shap"] = max(abs(shap_matrix[c][p["idx"]]) for c in CANCER_ORDER)
    all_peaks.sort(key=lambda x: -x["max_shap"])
    all_peaks = all_peaks[:N_DISPLAY_PEAKS]
    all_peaks.sort(key=lambda x: x["wn"])

# ── Build heatmap matrix ──
n_peaks = len(all_peaks)
n_cancers = len(CANCER_ORDER)
heatmap = np.zeros((n_peaks, n_cancers))

for j, cancer in enumerate(CANCER_ORDER):
    for i, peak in enumerate(all_peaks):
        heatmap[i, j] = shap_matrix[cancer][peak["idx"]]

# ── Classify peaks: shared vs cancer-specific ──
# A peak is "shared" if |SHAP| > threshold in 3+ cancers
SHARE_THRESHOLD = 0.05
peak_counts = np.sum(np.abs(heatmap) > SHARE_THRESHOLD, axis=1)

shared_idx = [i for i in range(n_peaks) if peak_counts[i] >= 3]
specific_idx = [i for i in range(n_peaks) if peak_counts[i] < 3]

# Reorder: shared first (sorted by wn), then specific (sorted by wn)
row_order = shared_idx + specific_idx
n_shared = len(shared_idx)

heatmap_ordered = heatmap[row_order, :]
peaks_ordered = [all_peaks[i] for i in row_order]

# ── Use absolute SHAP values ──
heatmap_abs = np.abs(heatmap_ordered)

# ── Sort rows by total importance (most discriminative at top) ──
row_totals = heatmap_abs.sum(axis=1)
row_order2 = np.argsort(-row_totals)
heatmap_abs = heatmap_abs[row_order2, :]
peaks_ordered = [peaks_ordered[i] for i in row_order2]

# ── Build per-cancer colormaps (white → cancer color) ──
from matplotlib.colors import LinearSegmentedColormap

def make_cancer_cmap(hex_color):
    """White → cancer color sequential colormap."""
    import matplotlib.colors as mc
    rgb = mc.to_rgb(hex_color)
    return LinearSegmentedColormap.from_list("", ["#FFFFFF", hex_color], N=256)

cancer_cmaps = {c: make_cancer_cmap(CANCER_COLORS[c]) for c in CANCER_ORDER}

# ── Figure ──
fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.62, SINGLE_COL * 0.65))

vmax = np.max(heatmap_abs)

# Transpose: x=wavenumber, y=cancer
heatmap_T = heatmap_abs.T  # (n_cancers, n_peaks)

# Draw each cell individually with its cancer-specific colormap
for i, cancer in enumerate(CANCER_ORDER):
    for j in range(len(peaks_ordered)):
        val = heatmap_T[i, j]
        normed = val / vmax
        color = cancer_cmaps[cancer](normed)
        rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                              facecolor=color, edgecolor="none",
                              linewidth=0)
        ax.add_patch(rect)

ax.set_xlim(-0.5, len(peaks_ordered) - 0.5)
ax.set_ylim(n_cancers - 0.5, -0.5)

# ── Axis labels ──
# X-axis: wavenumbers
wn_labels = [f"{p['wn']:.0f}" for p in peaks_ordered]
ax.set_xticks(np.arange(len(peaks_ordered)))
ax.set_xticklabels(wn_labels, fontsize=FONT_SIZE["axis_label"] + 1.5, rotation=45, ha="right")
ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=FONT_SIZE["axis_label"] + 2)

# Y-axis: cancer types
cancer_labels = [AACR_LABELS[c] for c in CANCER_ORDER]
ax.set_yticks(np.arange(n_cancers))
ax.set_yticklabels(cancer_labels, fontsize=FONT_SIZE["axis_label"] + 2, fontweight="bold")

for tick_idx, cancer in enumerate(CANCER_ORDER):
    ax.get_yticklabels()[tick_idx].set_color(CANCER_COLORS[cancer])

# ── SHAP value text inside cells ──
for i in range(n_cancers):
    for j in range(len(peaks_ordered)):
        val = heatmap_T[i, j]
        if val > 0.03:
            normed = val / vmax
            text_color = "white" if normed > 0.5 else "#333333"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=FONT_SIZE["tick"] + 1, color=text_color)

# ── Clean up ──
ax.tick_params(axis="both", which="both", length=0)
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
save_figure(fig, "fig_shap_heatmap")
plt.close()

# ── Summary ──
print(f"\n=== SHAP Heatmap Summary ===")
print(f"Total peaks displayed: {len(peaks_ordered)}")
print(f"Shared (≥3 cancers): {n_shared}")
print(f"Cancer-specific (<3 cancers): {len(peaks_ordered) - n_shared}")
shared_str = ', '.join(f"{p['wn']:.0f} cm⁻¹" for p in peaks_ordered[:n_shared])
specific_str = ', '.join(f"{p['wn']:.0f} cm⁻¹" for p in peaks_ordered[n_shared:])
print(f"\nShared peaks: {shared_str}")
print(f"Specific peaks: {specific_str}")
