"""
Figure: Feature Importance Bar Chart per Cancer Type

Horizontal bar chart showing top-5 discriminative wavenumber peaks
per cancer type, based on Stage 1 Fusion LR SHAP values.
SHAP for linear model: coef_i * (x_i - E[x_i]), SERS features only.

Uses SAME model and method as fig_spectra_shap.py for peak consistency.
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
    load_spectra, compute_group_means,
    CANCER_COLORS, NON_CANCER_COLOR, NON_CANCER_GROUPS,
    AACR_LABELS, AACR_INTERNAL_ORDER,
    DOUBLE_COL, FONT_SIZE, LINE_WIDTH,
)

apply_style()

# ── Config ──
CANCER_ORDER = AACR_INTERNAL_ORDER  # PRO, OVA, LUN, PAN, CRC
N_TOP_PEAKS = 5
PEAK_MERGE_RADIUS = 15  # cm-1
SERS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(SERS_ROOT, "models", "production", "stage1_fusion.joblib")
N_SERS_FEATURES = 933

# ── Load spectra ──
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

# ── SHAP from Stage 1 Fusion LR (SERS portion only) ──
model_wn_cols = wn_cols[1:-1]  # 933 features
model_wavenumbers = wavenumbers[1:-1]

pipe = joblib.load(MODEL_PATH)
coef_all = pipe.named_steps["logisticregression"].coef_[0]  # (936,)
coef_sers = coef_all[:N_SERS_FEATURES]

# Global mean (medoid per patient)
spec_copy = spec.copy()
spec_copy["display_group"] = spec_copy["group"].map(groups_map)
spec_copy = spec_copy.dropna(subset=["display_group"])
spec_med = spec_copy.groupby(["display_group", "sample_id"]).first().reset_index()
global_mean = spec_med[model_wn_cols].values.mean(axis=0)


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
        regions.append({"idx": idx, "wn": wn, "shap": shap_vals[idx],
                        "importance": abs_shap[idx],
                        "sign": "+" if shap_vals[idx] > 0 else "-"})
    return sorted(regions, key=lambda x: -x["importance"])


cancer_shap = {}
for cancer in CANCER_ORDER:
    if cancer not in stats:
        continue
    cancer_mean = stats[cancer]["mean"][1:-1]
    shap_vals = coef_sers * (cancer_mean - global_mean)
    cancer_shap[cancer] = find_top_peaks(shap_vals, model_wavenumbers, N_TOP_PEAKS, PEAK_MERGE_RADIUS)

# ── Figure: 5 panels (one per cancer), vertical layout ──
n_cancers = len(CANCER_ORDER)
fig, axes = plt.subplots(n_cancers, 1,
                         figsize=(DOUBLE_COL * 0.55, DOUBLE_COL * 0.80),
                         sharey=False, sharex=True)
fig.subplots_adjust(hspace=0.70, left=0.22, right=0.88, top=0.92, bottom=0.08)

# Global x-axis range
all_imp = [r["importance"] for peaks in cancer_shap.values() for r in peaks]
x_max = max(all_imp) * 1.35

for i, cancer in enumerate(CANCER_ORDER):
    ax = axes[i]
    color = CANCER_COLORS[cancer]
    dlabel = AACR_LABELS[cancer]
    peaks = cancer_shap[cancer]

    wns = [f"{r['wn']:.0f}" for r in peaks]
    abs_vals = [r["importance"] for r in peaks]
    directions = [r["sign"] for r in peaks]
    y_pos = np.arange(len(peaks))

    bars = ax.barh(y_pos, abs_vals, height=0.55,
                   color=color, edgecolor=color,
                   linewidth=0.5, alpha=0.80, zorder=3)

    # Value labels with direction arrow
    for j, (val, yp, d) in enumerate(zip(abs_vals, y_pos, directions)):
        arrow = "\u2191" if d == "+" else "\u2193"
        ax.text(val + x_max * 0.03, yp, f"{val:.3f} {arrow}",
                ha="left", va="center",
                fontsize=FONT_SIZE["annotation"],
                color="#555555")

    # Wavenumber labels on y-axis
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{w} cm\u207b\u00b9" for w in wns],
                       fontsize=FONT_SIZE["tick"] + 0.5)
    ax.invert_yaxis()

    ax.set_xlim(0, x_max)
    ax.tick_params(axis="x", labelsize=FONT_SIZE["tick"])

    # Cancer label ABOVE the panel (left-aligned)
    ax.set_title(dlabel, fontsize=FONT_SIZE["title"] + 1, fontweight="bold",
                 color=color, loc="left", pad=4)

    apply_nature_style(ax)

# X label on bottom panel only
axes[-1].set_xlabel("Feature Importance (|SHAP|)", fontsize=FONT_SIZE["axis_label"])

# Suptitle
fig.suptitle("Feature Importance (Top-5 SHAP Peaks per Cancer Type)",
             fontsize=FONT_SIZE["title"] + 1.5, fontweight="bold",
             color="#2A2A2A", y=0.97)

fig.text(0.55, 0.01,
         "Stage 1 Fusion LR  |  SHAP = coef \u00d7 (x \u2013 E[x])  |  "
         "\u2191 = positive  |  \u2193 = negative",
         ha="center", va="bottom",
         fontsize=FONT_SIZE["annotation"], color="#999999", style="italic")

save_figure(fig, "fig_shap_bars")
plt.close()

# ── Summary table ──
print("\n=== Stage 1 SHAP Feature Importance (Fusion LR) ===")
print(f"{'Cancer':<6} {'Rank':<5} {'Wavenumber':<12} {'|SHAP|':>8}  {'Direction'}")
print("-" * 50)
for cancer in CANCER_ORDER:
    dl = AACR_LABELS[cancer]
    for k, r in enumerate(cancer_shap[cancer]):
        d = "\u2191 positive" if r["sign"] == "+" else "\u2193 negative"
        print(f"{dl:<6} {k+1:<5} {r['wn']:>8.0f} cm\u207b\u00b9  {r['importance']:>8.4f}  {d}")
    print()
