"""
Figure: SHAP Grouped Lollipop — Discriminative Features Across Cancer Types

Grouped lollipop chart: each wavenumber peak has 5 stems (one per cancer).
Common features = all stems tall. Specific features = one stem tall.
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
    DOUBLE_COL, SINGLE_COL, FONT_SIZE, LINE_WIDTH,
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

# ── Load & compute SHAP ──
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

for p in all_peaks:
    p["max_shap"] = max(abs(shap_matrix[c][p["idx"]]) for c in CANCER_ORDER)
all_peaks.sort(key=lambda x: -x["max_shap"])
all_peaks = all_peaks[:N_DISPLAY_PEAKS]

# Sort by total importance (most important first)
for p in all_peaks:
    p["total"] = sum(abs(shap_matrix[c][p["idx"]]) for c in CANCER_ORDER)
all_peaks.sort(key=lambda x: -x["total"])

# Build importance matrix
n_peaks = len(all_peaks)
n_cancers = len(CANCER_ORDER)
imp_matrix = np.zeros((n_peaks, n_cancers))
for j, cancer in enumerate(CANCER_ORDER):
    for i, peak in enumerate(all_peaks):
        imp_matrix[i, j] = abs(shap_matrix[cancer][peak["idx"]])

# ── Figure: horizontal grouped lollipop ──
fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.55, DOUBLE_COL * 0.40))

group_height = 1.0
stem_spacing = group_height / (n_cancers + 1)

for i in range(n_peaks):
    y_center = i * group_height
    # Light background band for alternating rows
    if i % 2 == 0:
        ax.axhspan(y_center - group_height * 0.45, y_center + group_height * 0.45,
                    color="#F5F5F5", zorder=0)

    for j, cancer in enumerate(CANCER_ORDER):
        val = imp_matrix[i, j]
        y = y_center + (j - (n_cancers - 1) / 2) * stem_spacing
        color = CANCER_COLORS[cancer]

        # Stem
        ax.plot([0, val], [y, y], color=color, linewidth=1.0,
                alpha=0.7, zorder=2, solid_capstyle="round")
        # Dot
        ax.scatter(val, y, s=25 + 100 * (val / imp_matrix.max()),
                   c=color, alpha=0.85, edgecolors="white",
                   linewidths=0.4, zorder=3)

# Y labels (wavenumber)
y_positions = [i * group_height for i in range(n_peaks)]
wn_labels = [f"{p['wn']:.0f} cm⁻¹" for p in all_peaks]
ax.set_yticks(y_positions)
ax.set_yticklabels(wn_labels, fontsize=FONT_SIZE["tick"] + 0.5)
ax.invert_yaxis()

ax.set_xlabel("Feature Importance (|SHAP|)", fontsize=FONT_SIZE["axis_label"])
ax.set_xlim(0, imp_matrix.max() * 1.15)

# Legend
legend_handles = []
for cancer in CANCER_ORDER:
    label = AACR_LABELS[cancer]
    color = CANCER_COLORS[cancer]
    h = ax.scatter([], [], s=40, c=color, label=label,
                   edgecolors="white", linewidths=0.4)
    legend_handles.append(h)
ax.legend(handles=legend_handles, loc="lower right",
          fontsize=FONT_SIZE["legend"], ncol=1, frameon=False,
          handletextpad=0.3, columnspacing=0.8)

ax.set_title("Discriminative Spectral Features Across Cancer Types",
             fontsize=FONT_SIZE["title"] + 1, fontweight="bold",
             color="#2A2A2A", pad=8)

apply_nature_style(ax)
ax.tick_params(axis="y", length=0)

plt.tight_layout()
save_figure(fig, "fig_shap_lollipop")
plt.close()
print("Done: fig_shap_lollipop")
