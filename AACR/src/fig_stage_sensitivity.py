"""
Figure: Stage-stratified Detection Sensitivity (uSERS-Net)

Shows sensitivity for Early (I–II) vs Late (III–IV) stage per cancer type.
PRO uses T-stage proxy (T1–T2 = early, T3–T4 = advanced).
OVA excluded due to insufficient staging data (10/70).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_curve
from nature_style import (
    apply_style, apply_nature_style, add_panel_label, save_figure,
    AACR_LABELS, AACR_COLORS, OUR_MODEL,
    DOUBLE_COL, FONT_SIZE, LINE_WIDTH,
)

apply_style()

SERS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AACR_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FUSION_PATH = os.path.join(AACR_DIR, "data", "early_fusion", "fold_predictions.npz")

# ── Load fold predictions ──
fus    = np.load(FUSION_PATH, allow_pickle=True)
groups = fus["groups"]
sids   = fus["sample_ids"]
bp     = fus["val_binary_prob"]
bl     = fus["binary_labels"]

fpr, tpr, thresholds = roc_curve(bl, bp)
thr  = thresholds[np.argmax(tpr - fpr)]
pred = (bp >= thr).astype(int)

# Patient-level detection: majority vote across 5 replicates
pat_pred = {}
for i in range(len(groups)):
    key = (str(groups[i]), int(sids[i]))
    pat_pred.setdefault(key, []).append(pred[i])
pat_detected = {k: int(np.mean(v) >= 0.5) for k, v in pat_pred.items()}


def get_sensitivity(group, sample_ids):
    """Compute sensitivity for a list of (group, sample_id) patients."""
    results = [pat_detected.get((group, sid), 0) for sid in sample_ids]
    return np.mean(results) if results else np.nan


# ── Staging mappings ──

def classify_ajcc(stage_str):
    """Return 'Early', 'Late', or None based on AJCC stage string."""
    if pd.isna(stage_str):
        return None
    s = str(stage_str).strip().upper()
    if s in ("UNKNOWN", ""):
        return None
    if s.startswith("III") or s.startswith("IV"):
        return "Late"
    if s.startswith("I") or s.startswith("II"):
        return "Early"
    return None


def classify_tstage(t_str):
    """T1–T2 = Early, T3–T4 = Late (PRO proxy)."""
    if pd.isna(t_str):
        return None
    s = str(t_str).strip().upper()
    if s.startswith("T1") or s.startswith("T2"):
        return "Early"
    if s.startswith("T3") or s.startswith("T4"):
        return "Late"
    return None


# ── Build per-cancer stage → sensitivity ──
results = {}  # cancer_display → {"Early": sens, "Late": sens, "n_early": n, "n_late": n}

# ── PRC (PRO, T-stage proxy) ──
pro = pd.read_csv(os.path.join(AACR_DIR, "data", "PRO_with_staging.csv"))
pro["_sid"] = pro["patient_id"].str.replace("PRO ", "").astype(int)
pro["_stage_group"] = pro["t_stage"].apply(classify_tstage)
early_ids = pro.loc[pro["_stage_group"] == "Early", "_sid"].tolist()
late_ids  = pro.loc[pro["_stage_group"] == "Late",  "_sid"].tolist()
results["PRC"] = {
    "Early": get_sensitivity("PRO", early_ids), "n_early": len(early_ids),
    "Late":  get_sensitivity("PRO", late_ids),  "n_late":  len(late_ids),
    "stage_label": "T1–T2 vs T3–T4",
}

# ── LC (LUN) — positional matching (row 0 → sample_id 1) ──
lun = pd.read_csv(os.path.join(SERS_ROOT, "data", "clinical_data", "standardized",
                               "LUN_clinical_standardized.csv"))
lun["_sid"] = np.arange(1, len(lun) + 1)   # positional: row i → sample_id i+1
lun["_sg"] = lun["stage"].apply(classify_ajcc)
early_ids = lun.loc[lun["_sg"] == "Early", "_sid"].tolist()
late_ids  = lun.loc[lun["_sg"] == "Late",  "_sid"].tolist()
results["LC"] = {
    "Early": get_sensitivity("LUN", early_ids), "n_early": len(early_ids),
    "Late":  get_sensitivity("LUN", late_ids),  "n_late":  len(late_ids),
    "stage_label": "Stage I–II vs III–IV",
}

# ── PAC (CPAN) ──
pan = pd.read_csv(os.path.join(AACR_DIR, "data", "PAN_with_ajcc_stage.csv"))
pan["_sid"] = np.arange(1, len(pan) + 1)   # positional
pan["_sg"] = pan["ajcc_stage"].apply(classify_ajcc)
early_ids = pan.loc[pan["_sg"] == "Early", "_sid"].tolist()
late_ids  = pan.loc[pan["_sg"] == "Late",  "_sid"].tolist()
results["PAC"] = {
    "Early": get_sensitivity("CPAN", early_ids), "n_early": len(early_ids),
    "Late":  get_sensitivity("CPAN", late_ids),  "n_late":  len(late_ids),
    "stage_label": "Stage I–II vs III–IV",
}

# ── CRC — positional matching (patient_ids start at 271, not 1) ──
crc = pd.read_csv(os.path.join(AACR_DIR, "data", "CRC_with_ajcc_stage.csv"))
crc["_sid"] = np.arange(1, len(crc) + 1)   # positional: row i → sample_id i+1
crc["_sg"] = crc["ajcc_stage"].apply(classify_ajcc)
early_ids = crc.loc[crc["_sg"] == "Early", "_sid"].tolist()
late_ids  = crc.loc[crc["_sg"] == "Late",  "_sid"].tolist()
results["CRC"] = {
    "Early": get_sensitivity("CRC", early_ids), "n_early": len(early_ids),
    "Late":  get_sensitivity("CRC", late_ids),  "n_late":  len(late_ids),
    "stage_label": "Stage I–II vs III–IV",
}

# Print summary
print("\n=== Stage-stratified Detection Sensitivity (uSERS-Net) ===")
for cancer, r in results.items():
    print(f"  {cancer} ({r['stage_label']}):")
    print(f"    Early (n={r['n_early']}): {r['Early']:.1%}")
    print(f"    Late  (n={r['n_late']}):  {r['Late']:.1%}")

# ── Figure: dot plot with connecting line ──
DISPLAY_ORDER = ["PRC", "LC", "PAC", "CRC"]
EARLY_COLOR = "#4A90D9"   # blue
LATE_COLOR  = "#E8604C"   # orange-red

fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.7, DOUBLE_COL * 0.45), facecolor="white")

y_positions = np.arange(len(DISPLAY_ORDER))
dot_size = 60

for yi, cancer in enumerate(DISPLAY_ORDER):
    r = results[cancer]
    color = AACR_COLORS[cancer]
    e_val = r["Early"]
    l_val = r["Late"]

    # Connecting line
    ax.plot([e_val, l_val], [yi, yi],
            color=color, lw=1.6, alpha=0.5, zorder=1)

    # Dots
    ax.scatter(e_val, yi, s=dot_size + 20, color=EARLY_COLOR,
               edgecolors=color, linewidths=1.2, zorder=3)
    ax.scatter(l_val, yi, s=dot_size + 20, color=LATE_COLOR,
               edgecolors=color, linewidths=1.2, zorder=3)

    # Labels — position to the right of dots to avoid overlap
    x_right = max(e_val, l_val) + 0.015
    fs = FONT_SIZE["annotation"]

    gap = abs(e_val - l_val)
    if gap < 0.04:
        # Dots nearly overlap: single combined annotation to the right
        ax.text(x_right, yi + 0.08,
                f"Early {e_val:.0%} (n={r['n_early']})",
                ha="left", va="bottom", fontsize=fs, color=EARLY_COLOR)
        ax.text(x_right, yi - 0.08,
                f"Late {l_val:.0%} (n={r['n_late']})",
                ha="left", va="top", fontsize=fs, color=LATE_COLOR)
    else:
        # Enough space: label each dot individually above
        ax.text(e_val, yi + 0.22,
                f"Early {e_val:.0%}\n(n={r['n_early']})",
                ha="center", va="bottom", fontsize=fs, color=EARLY_COLOR)
        ax.text(l_val, yi + 0.22,
                f"Late {l_val:.0%}\n(n={r['n_late']})",
                ha="center", va="bottom", fontsize=fs, color=LATE_COLOR)

    # Cancer label
    ax.text(0.44, yi, cancer, ha="right", va="center",
            fontsize=FONT_SIZE["title"], fontweight="bold",
            color=AACR_COLORS[cancer])

# Grid
for yi in y_positions:
    ax.axhline(yi, color="#F0F0F0", lw=0.5, zorder=0)
for xv in [0.6, 0.7, 0.8, 0.9, 1.0]:
    ax.axvline(xv, color="#F0F0F0", lw=0.5, zorder=0)

ax.set_xlim(0.45, 1.18)
ax.set_ylim(-0.6, len(DISPLAY_ORDER) - 0.4)
ax.set_yticks([])
ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_xticklabels(["50%", "60%", "70%", "80%", "90%", "100%"],
                   fontsize=FONT_SIZE["tick"])
ax.set_xlabel("Detection Sensitivity", fontsize=FONT_SIZE["axis_label"], labelpad=6)
ax.set_title(f"{OUR_MODEL}: Stage-Stratified Cancer Detection Sensitivity",
             fontsize=FONT_SIZE["title"] + 1, fontweight="bold",
             color="#2A2A2A", pad=10)

# Legend
handles = [
    mpatches.Patch(color=EARLY_COLOR, label="Early stage (I–II)"),
    mpatches.Patch(color=LATE_COLOR,  label="Late stage (III–IV)"),
]
ax.legend(handles=handles, loc="lower right",
          fontsize=FONT_SIZE["legend"] + 0.5,
          frameon=True, framealpha=0.92, edgecolor="#DDDDDD")

ax.text(0.99, -0.18,
        "PRC: T1–T2 vs T3–T4  |  LC, PAC, CRC: AJCC Stage I–II vs III–IV  |  "
        "OVC excluded (staging data <15%)",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=FONT_SIZE["annotation"], color="#999999", style="italic")

apply_nature_style(ax)
ax.spines["left"].set_visible(False)

save_figure(fig, "fig_stage_sensitivity")
plt.close()
