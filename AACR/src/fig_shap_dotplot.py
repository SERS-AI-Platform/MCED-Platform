"""
Figure: SHAP Dot Plot — Discriminative Features Across Cancer Types

Dot (bubble) chart: wavenumber (y) × cancer type (x), dot size = |SHAP|.
Common features show large dots across all columns.
Cancer-specific features show one large dot.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from nature_style import (
    apply_style, apply_nature_style, save_figure,
    compute_group_means,
    CANCER_COLORS, NON_CANCER_GROUPS,
    AACR_LABELS, AACR_INTERNAL_ORDER,
    DOUBLE_COL, SINGLE_COL, FONT_SIZE,
)

apply_style()

# ── Config ──
CANCER_ORDER = AACR_INTERNAL_ORDER
N_TOP_PER_CANCER = 5
N_DISPLAY_PEAKS = 10
PEAK_MERGE_RADIUS = 15
SERS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(SERS_ROOT, "models", "production", "stage1_fusion.joblib")
N_SERS_FEATURES = 933

# ── Load & compute SHAP (shared logic) ──
PRE_QC_PATH = os.path.join(SERS_ROOT, "results", "qc_experiment", "processed_spectra.csv")
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

model_wn_cols = wn_cols[1:-1]
model_wavenumbers = wavenumbers[1:-1]

pipe = joblib.load(MODEL_PATH)
coef_all = pipe.named_steps["logisticregression"].coef_[0]
coef_sers = coef_all[:N_SERS_FEATURES]

spec_copy = spec.copy()
spec_copy["display_group"] = spec_copy["group"].map(groups_map)
spec_copy = spec_copy.dropna(subset=["display_group"])
spec_med = spec_copy.groupby(["display_group", "sample_id"]).first().reset_index()
global_mean = spec_med[model_wn_cols].values.mean(axis=0)

shap_matrix = {}
for cancer in CANCER_ORDER:
    if cancer not in stats:
        continue
    cancer_mean = stats[cancer]["mean"][1:-1]
    shap_matrix[cancer] = coef_sers * (cancer_mean - global_mean)


def find_top_peaks(shap_vals, wns, n=5, radius=15):
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


# ── Collect unique peaks ──
all_peaks = []
seen_wns = set()
for cancer in CANCER_ORDER:
    peaks = find_top_peaks(shap_matrix[cancer], model_wavenumbers,
                           n=N_TOP_PER_CANCER, radius=PEAK_MERGE_RADIUS)
    for p in peaks:
        wn = p["wn"]
        if any(abs(wn - s) < PEAK_MERGE_RADIUS for s in seen_wns):
            continue
        seen_wns.add(wn)
        all_peaks.append(p)

# Select top N by max importance
for p in all_peaks:
    p["max_shap"] = max(abs(shap_matrix[c][p["idx"]]) for c in CANCER_ORDER)
all_peaks.sort(key=lambda x: -x["max_shap"])
all_peaks = all_peaks[:N_DISPLAY_PEAKS]
all_peaks.sort(key=lambda x: x["wn"])

# Build importance matrix
n_peaks = len(all_peaks)
n_cancers = len(CANCER_ORDER)
imp_matrix = np.zeros((n_peaks, n_cancers))
for j, cancer in enumerate(CANCER_ORDER):
    for i, peak in enumerate(all_peaks):
        imp_matrix[i, j] = abs(shap_matrix[cancer][peak["idx"]])

# Sort rows by total importance (most important at top)
row_totals = imp_matrix.sum(axis=1)
row_order = np.argsort(-row_totals)
imp_matrix = imp_matrix[row_order, :]
all_peaks = [all_peaks[i] for i in row_order]

# ── Figure ──
fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.15, DOUBLE_COL * 0.35))

# Dot sizing
max_imp = imp_matrix.max()
SIZE_MIN = 8
SIZE_MAX = 220

for i in range(n_peaks):
    for j in range(n_cancers):
        val = imp_matrix[i, j]
        size = SIZE_MIN + (val / max_imp) * (SIZE_MAX - SIZE_MIN)
        alpha = 0.3 + 0.7 * (val / max_imp)
        color = CANCER_COLORS[CANCER_ORDER[j]]
        ax.scatter(j, i, s=size, c=color, alpha=alpha, edgecolors="white",
                   linewidths=0.3, zorder=3)

# Grid lines
for i in range(n_peaks):
    ax.axhline(y=i, color="#E8E8E8", linewidth=0.3, zorder=1)
for j in range(n_cancers):
    ax.axvline(x=j, color="#E8E8E8", linewidth=0.3, zorder=1)

# Axes
cancer_labels = [AACR_LABELS[c] for c in CANCER_ORDER]
ax.set_xticks(np.arange(n_cancers))
ax.set_xticklabels(cancer_labels, fontsize=FONT_SIZE["tick"] + 1, fontweight="bold")
for tick_idx, cancer in enumerate(CANCER_ORDER):
    ax.get_xticklabels()[tick_idx].set_color(CANCER_COLORS[cancer])

wn_labels = [f"{p['wn']:.0f} cm⁻¹" for p in all_peaks]
ax.set_yticks(np.arange(n_peaks))
ax.set_yticklabels(wn_labels, fontsize=FONT_SIZE["tick"] + 0.5)

ax.set_xlim(-0.6, n_cancers - 0.4)
ax.set_ylim(n_peaks - 0.6, -0.6)

# Size legend
legend_vals = [0.1, 0.25, 0.4]
legend_x = n_cancers + 0.3
for k, v in enumerate(legend_vals):
    s = SIZE_MIN + (v / max_imp) * (SIZE_MAX - SIZE_MIN)
    ax.scatter(legend_x, k + 0.5, s=s, c="#888888", alpha=0.6,
               edgecolors="white", linewidths=0.3, clip_on=False)
    ax.text(legend_x + 0.35, k + 0.5, f"{v:.2f}",
            fontsize=FONT_SIZE["annotation"], va="center", color="#555555",
            clip_on=False)
ax.text(legend_x + 0.15, -0.3, "|SHAP|",
        fontsize=FONT_SIZE["annotation"], fontweight="bold", va="center",
        ha="center", color="#555555", clip_on=False)

ax.set_title("Discriminative Spectral Features Across Cancer Types",
             fontsize=FONT_SIZE["title"] + 1, fontweight="bold",
             color="#2A2A2A", pad=8)

apply_nature_style(ax)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.tick_params(axis="both", length=0)

plt.tight_layout()
save_figure(fig, "fig_shap_dotplot")
plt.close()
print("Done: fig_shap_dotplot")
