"""
Figure 4: TNM stage-wise cancer detection sensitivity.
Dot plot showing Early (Stage I-II) vs Late (Stage III-IV) sensitivity per cancer type.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from nature_style import (
    apply_style, apply_nature_style, add_panel_label, save_figure,
    CANCER_COLORS, CANCER_LABELS, SINGLE_COL, DOUBLE_COL, FONT_SIZE, ROOT,
)

apply_style()

# ── Config ──
CANCER_ORDER = ["CRC", "LUN", "PAN", "PRO", "OVA"]  # by n staged, descending
FOLD_GROUP_MAP = {"PAN": "CPAN"}  # clinical -> fold prediction group name
ALPHA = 0.8  # ensemble blend weight


def map_to_major_stage(s):
    """Map stage string to roman numeral I/II/III/IV."""
    if pd.isna(s):
        return None
    s = str(s).strip().upper()
    if s.startswith("IV"):
        return "IV"
    if s.startswith("III"):
        return "III"
    if s.startswith("II"):
        return "II"
    if s.startswith("I"):
        return "I"
    return None


def stage_to_binary(stage):
    """Map I/II -> 'Early', III/IV -> 'Late'."""
    if stage in ("I", "II"):
        return "Early (I-II)"
    if stage in ("III", "IV"):
        return "Late (III-IV)"
    return None


def wilson_ci(p, n, z=1.96):
    """Wilson score 95% CI."""
    if n == 0:
        return 0, 0
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return max(0, center - spread), min(1, center + spread)


# ── Load ensemble predictions ──
npz_path = os.path.join(ROOT, "results", "training", "step5_ensemble", "fold_predictions.npz")
d = np.load(npz_path, allow_pickle=True)

binary_prob = ALPHA * d["val_bp_lr"] + (1 - ALPHA) * d["val_bp_rn"]
binary_labels = d["binary_labels"]
groups = d["groups"]
sample_ids = d["sample_ids"]

# Compute optimal threshold via Youden's J
fpr, tpr, thresholds = roc_curve(binary_labels, binary_prob)
j_scores = tpr - fpr
opt_threshold = thresholds[np.argmax(j_scores)]
print(f"Optimal threshold: {opt_threshold:.4f}")

# Predictions
preds = (binary_prob >= opt_threshold).astype(int)

# ── Build (group, sample_id) -> stage mapping ──
# Load sers_patient_features for patient_id lookup
feat = pd.read_csv(os.path.join(ROOT, "data", "clinical_data", "standardized",
                                "sers_patient_features.csv"))

stage_map = {}  # (fold_group, sample_id) -> major_stage

for cancer in CANCER_ORDER:
    fold_group = FOLD_GROUP_MAP.get(cancer, cancer)
    clin_path = os.path.join(ROOT, "data", "clinical_data", "standardized",
                             f"{cancer}_clinical_standardized.csv")
    clin = pd.read_csv(clin_path)

    if cancer in ("PRO", "CRC"):
        # patient_id = "PRO 1" format -> extract int
        for _, row in clin.iterrows():
            if pd.notna(row.get("stage")):
                try:
                    sid = int(str(row["patient_id"]).split()[-1])
                    stage = map_to_major_stage(row["stage"])
                    if stage:
                        stage_map[(fold_group, sid)] = stage
                except (ValueError, IndexError):
                    pass

    elif cancer == "PAN":
        # patient_id = "CPAN 1" format
        for _, row in clin.iterrows():
            if pd.notna(row.get("stage")):
                try:
                    sid = int(str(row["patient_id"]).split()[-1])
                    stage = map_to_major_stage(row["stage"])
                    if stage:
                        stage_map[("CPAN", sid)] = stage
                except (ValueError, IndexError):
                    pass

    elif cancer in ("OVA", "LUN"):
        # Use sers_patient_features to map patient_id -> sample_id
        feat_sub = feat[feat["group"] == fold_group][["sample_id", "patient_id"]]
        pid_to_sid = dict(zip(feat_sub["patient_id"].astype(str), feat_sub["sample_id"]))

        for _, row in clin.iterrows():
            if pd.notna(row.get("stage")):
                pid = str(row["patient_id"])
                # Try direct patient_id match (e.g., "LUN 1" format)
                if pid in pid_to_sid:
                    sid = pid_to_sid[pid]
                else:
                    # Try extracting int if it's "LUN X" format
                    try:
                        sid = int(pid.split()[-1])
                    except (ValueError, IndexError):
                        continue

                stage = map_to_major_stage(row["stage"])
                if stage:
                    stage_map[(fold_group, sid)] = stage

print(f"Total staged samples mapped: {len(stage_map)}")

# ── Compute per-cancer, per-stage sensitivity ──
results = []

for cancer in CANCER_ORDER:
    fold_group = FOLD_GROUP_MAP.get(cancer, cancer)

    for stage_bin_label in ["Early (I-II)", "Late (III-IV)"]:
        # Find samples matching this cancer + stage
        n_correct = 0
        n_total = 0

        for i in range(len(groups)):
            if groups[i] != fold_group:
                continue
            sid = sample_ids[i]
            stage = stage_map.get((fold_group, sid))
            if stage is None:
                continue
            stage_bin = stage_to_binary(stage)
            if stage_bin != stage_bin_label:
                continue

            # Count unique patients only (take first replicate prediction)
            # Since fold predictions already aggregate, each row is unique
            n_total += 1
            if preds[i] == 1:  # correctly detected as cancer
                n_correct += 1

        if n_total > 0:
            sens = n_correct / n_total
            ci_lo, ci_hi = wilson_ci(sens, n_total)
        else:
            sens, ci_lo, ci_hi = np.nan, np.nan, np.nan

        results.append({
            "cancer": cancer,
            "stage": stage_bin_label,
            "sensitivity": sens,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "n": n_total,
        })

df = pd.DataFrame(results)
print("\nStage-wise results:")
print(df.to_string(index=False))

# ── Plot: Dot plot ──
fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.3, SINGLE_COL * 1.0))
add_panel_label(ax, "")

x_positions = np.arange(len(CANCER_ORDER))
early_color = "#4A90D9"
late_color = "#D94A4A"
offset = 0.12

for stage_label, color, dx, marker in [
    ("Early (I-II)", early_color, -offset, "o"),
    ("Late (III-IV)", late_color, offset, "s"),
]:
    sub = df[df["stage"] == stage_label]
    sub = sub.set_index("cancer").reindex(CANCER_ORDER).reset_index()

    x = x_positions + dx
    y = sub["sensitivity"].values
    err_lo = y - sub["ci_lo"].values
    err_hi = sub["ci_hi"].values - y

    # Handle NaN
    valid = ~np.isnan(y)

    ax.errorbar(x[valid], y[valid],
                yerr=[err_lo[valid], err_hi[valid]],
                fmt=marker, color=color, markersize=6,
                markeredgecolor="white", markeredgewidth=0.6,
                capsize=3, capthick=0.8, elinewidth=0.8,
                label=stage_label, zorder=5)

    # Annotate n values
    for j in range(len(CANCER_ORDER)):
        n_val = sub.iloc[j]["n"]
        s_val = sub.iloc[j]["sensitivity"]
        if not np.isnan(s_val) and n_val > 0:
            ax.text(x[j], s_val + err_hi[j] + 0.03,
                    f"n={int(n_val)}", ha="center",
                    fontsize=FONT_SIZE["annotation"] - 0.5,
                    color=color, alpha=0.8)

ax.set_xticks(x_positions)
ax.set_xticklabels([CANCER_LABELS.get(c, c) for c in CANCER_ORDER])
ax.set_ylabel("Detection Sensitivity")
ax.set_title("Stage-wise Cancer Detection Performance", fontsize=FONT_SIZE["title"], pad=10)
ax.set_ylim(0.5, 1.08)
ax.axhline(y=0.9, color="#DDDDDD", linewidth=0.5, linestyle="--", zorder=0)
ax.legend(loc="lower left", frameon=True, edgecolor="#DDDDDD",
          fancybox=False, framealpha=0.9)
apply_nature_style(ax, ylabel="Detection Sensitivity")

# Add footnote for OVA
ova_early_n = df[(df["cancer"] == "OVA") & (df["stage"] == "Early (I-II)")]["n"].values
ova_late_n = df[(df["cancer"] == "OVA") & (df["stage"] == "Late (III-IV)")]["n"].values
if len(ova_early_n) > 0 and len(ova_late_n) > 0:
    fig.text(0.5, -0.02,
             f"* Ovarian staging data limited (Early n={int(ova_early_n[0])}, Late n={int(ova_late_n[0])})",
             ha="center", fontsize=FONT_SIZE["annotation"], color="#999999", fontstyle="italic")

save_figure(fig, "fig4_stage_performance")
plt.close()
print("Figure 4 complete.")
