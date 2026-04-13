"""
AACR Figure: Representative SERS spectra per cancer type vs Non-Cancer control,
with top-5 SHAP-important wavenumber peaks highlighted.

Model: Production Fusion LR (SERS + age/sex/BMI), sex constraint.
SHAP for linear model: coef_i * (x_i - E[x_i]), SERS features only.

Cancer types: PRO(PRC), OVA(OVC), LUN(LC), CRC, CPAN(PAC)
Non-Cancer: NOR, DIA, HBP, H.D.
Layout: 6 rows x 1 col (vertical)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import joblib
import matplotlib.pyplot as plt
from nature_style import (
    apply_style, apply_nature_style, add_panel_label, save_figure,
    load_spectra, compute_group_means,
    CANCER_COLORS, NON_CANCER_COLOR, NON_CANCER_GROUPS,
    AACR_LABELS, AACR_INTERNAL_ORDER,
    DOUBLE_COL, FONT_SIZE, LINE_WIDTH,
)

apply_style()

# ── Config ──
CANCER_ORDER = AACR_INTERNAL_ORDER
DISPLAY_LABELS = AACR_LABELS
N_TOP_PEAKS = 5
PEAK_MERGE_RADIUS = 15  # cm-1
SERS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(SERS_ROOT, "models", "production", "stage1_fusion.joblib")
N_SERS_FEATURES = 933  # first 933 of 936 total (last 3 = age, sex, bmi)

# Demographic n-counts (pre-QC, matching Table 1)
DEMO_N = {"PRO": 100, "OVA": 70, "LUN": 300, "CRC": 300, "PAN": 70, "Non-Cancer": 400}

# ── Load spectra (pre-QC, all samples) ──
PRE_QC_PATH = os.path.join(SERS_ROOT, "results", "qc_experiment", "processed_spectra.csv")
import pandas as pd
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

# ── SHAP from Fusion LR (SERS part only) ──
model_wn_cols = wn_cols[1:-1]  # 933 features
model_wavenumbers = wavenumbers[1:-1]

pipe = joblib.load(MODEL_PATH)
coef_all = pipe.named_steps["logisticregression"].coef_[0]  # (936,)
coef_sers = coef_all[:N_SERS_FEATURES]  # SERS portion only

# Global mean (medoid per patient)
spec_copy = spec.copy()
spec_copy["display_group"] = spec_copy["group"].map(groups_map)
spec_copy = spec_copy.dropna(subset=["display_group"])
spec_med = spec_copy.groupby(["display_group", "sample_id"]).first().reset_index()
global_mean = spec_med[model_wn_cols].values.mean(axis=0)


def find_top_peaks(shap_vals, wns, n=5, radius=15):
    """Top N peaks with merging of adjacent wavenumbers."""
    abs_shap = np.abs(shap_vals)
    regions = []
    for idx in np.argsort(abs_shap)[::-1]:
        if len(regions) >= n:
            break
        wn = wns[idx]
        if any(abs(wn - r["wn"]) < radius for r in regions):
            continue
        regions.append({"idx": idx, "wn": wn, "shap": shap_vals[idx]})
    return regions


cancer_shap = {}
for cancer in CANCER_ORDER:
    if cancer not in stats:
        continue
    cancer_mean = stats[cancer]["mean"][1:-1]
    shap_vals = coef_sers * (cancer_mean - global_mean)
    cancer_shap[cancer] = find_top_peaks(shap_vals, model_wavenumbers, N_TOP_PEAKS, PEAK_MERGE_RADIUS)


def spread_positions(raw_xs, min_gap=110, lo=450, hi=2150):
    """Push label positions apart to prevent text overlap."""
    xs = sorted(raw_xs)
    for _ in range(50):
        for j in range(len(xs) - 1):
            gap = xs[j + 1] - xs[j]
            if gap < min_gap:
                push = (min_gap - gap) / 2
                xs[j] -= push
                xs[j + 1] += push
    for j in range(len(xs)):
        xs[j] = max(lo, min(hi, xs[j]))
    return xs


# ── Figure ──
n_panels = len(CANCER_ORDER) + 1
row_h = 1.2
fig, axes = plt.subplots(n_panels, 1,
                         figsize=(DOUBLE_COL, row_h * n_panels),
                         sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.12, left=0.10, right=0.96)

# Y limits
all_means = np.concatenate(
    [stats[c]["mean"] for c in CANCER_ORDER if c in stats]
    + [stats["Non-Cancer"]["mean"]]
)
y_lo = all_means.min() - 0.2
y_hi = all_means.max() + 1.2

for i, cancer in enumerate(CANCER_ORDER):
    ax = axes[i]
    color = CANCER_COLORS[cancer]
    cs = stats[cancer]
    nc = stats["Non-Cancer"]
    dlabel = DISPLAY_LABELS[cancer]

    # Non-cancer (gray)
    ax.fill_between(wavenumbers, nc["mean"] - nc["sem"], nc["mean"] + nc["sem"],
                    color=NON_CANCER_COLOR, alpha=0.10, linewidth=0)
    ax.plot(wavenumbers, nc["mean"], color=NON_CANCER_COLOR,
            linewidth=LINE_WIDTH["thin"], alpha=0.5)

    # Cancer spectrum
    ax.fill_between(wavenumbers, cs["mean"] - cs["sem"], cs["mean"] + cs["sem"],
                    color=color, alpha=0.15, linewidth=0)
    ax.plot(wavenumbers, cs["mean"], color=color, linewidth=LINE_WIDTH["spectrum"])

    # SHAP peak shading + annotations
    regions = cancer_shap.get(cancer, [])
    if regions:
        for r in regions:
            ax.axvspan(r["wn"] - 10, r["wn"] + 10, color=color, alpha=0.13, zorder=0)

        raw_wns = [r["wn"] for r in regions]
        idx_order = sorted(range(len(raw_wns)), key=lambda k: raw_wns[k])
        sorted_raw = [raw_wns[k] for k in idx_order]
        spread = spread_positions(sorted_raw)

        text_y = y_hi - (y_hi - y_lo) * 0.04
        line_top = y_hi - (y_hi - y_lo) * 0.18

        for orig_wn, text_x in zip(sorted_raw, spread):
            full_idx = np.argmin(np.abs(wavenumbers - orig_wn))
            peak_y = cs["mean"][full_idx]

            ax.plot([orig_wn, orig_wn],
                    [peak_y + (y_hi - y_lo) * 0.02, line_top],
                    color=color, lw=0.4, alpha=0.4, zorder=5)
            if abs(text_x - orig_wn) > 8:
                ax.plot([orig_wn, text_x], [line_top, line_top],
                        color=color, lw=0.35, alpha=0.3, zorder=5)
                ax.plot([text_x, text_x],
                        [line_top, line_top + (y_hi - y_lo) * 0.03],
                        color=color, lw=0.35, alpha=0.3, zorder=5)

            ax.text(text_x, text_y, f"{orig_wn:.0f}",
                    fontsize=FONT_SIZE["annotation"], color=color,
                    ha="center", va="top", fontweight="bold", zorder=6)

    # Label inside panel (right side, vertically centered)
    ax.text(0.99, 0.50, f'{dlabel}  (n={DEMO_N.get(cancer, cs["n"])})',
            transform=ax.transAxes,
            fontsize=FONT_SIZE["title"] + 0.5, fontweight="bold",
            color=color, ha="right", va="center", alpha=0.85)

    ax.set_ylabel("Intensity (a.u.)", fontsize=FONT_SIZE["axis_label"], labelpad=2)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlim(wavenumbers.min(), wavenumbers.max())
    apply_nature_style(ax)

# NOR panel
ax = axes[-1]
nc = stats["Non-Cancer"]
ax.fill_between(wavenumbers, nc["mean"] - nc["sem"], nc["mean"] + nc["sem"],
                color=NON_CANCER_COLOR, alpha=0.15, linewidth=0)
ax.plot(wavenumbers, nc["mean"], color=NON_CANCER_COLOR,
        linewidth=LINE_WIDTH["spectrum"])
ax.text(0.99, 0.50, f'Non-Cancer  (n={DEMO_N["Non-Cancer"]})',
        transform=ax.transAxes,
        fontsize=FONT_SIZE["title"] + 0.5, fontweight="bold",
        color=NON_CANCER_COLOR, ha="right", va="center", alpha=0.85)
ax.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=FONT_SIZE["axis_label"])
ax.set_ylabel("Intensity (a.u.)", fontsize=FONT_SIZE["axis_label"], labelpad=2)
ax.set_ylim(y_lo, y_hi)
ax.set_xlim(wavenumbers.min(), wavenumbers.max())
apply_nature_style(ax)

save_figure(fig, "fig_spectra_shap")
plt.close()

# Summary
print("\n=== Top-5 SHAP peak regions (Fusion LR, SERS portion) ===")
for cancer in CANCER_ORDER:
    if cancer not in cancer_shap:
        continue
    dl = DISPLAY_LABELS[cancer]
    print(f"\n{dl}:")
    for k, r in enumerate(sorted(cancer_shap[cancer], key=lambda x: -abs(x["shap"]))):
        d = "+" if r["shap"] > 0 else "-"
        print(f"  {k+1}. {r['wn']:.0f} cm-1 (SHAP={r['shap']:+.4f} {d})")
