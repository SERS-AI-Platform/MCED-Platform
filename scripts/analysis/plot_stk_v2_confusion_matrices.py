"""
STK-V2 Confusion Matrix Visualization (publication-quality)
Each figure = one type of information only.
Colors follow config/config.yaml group_colors.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, f1_score, roc_curve,
    precision_recall_fscore_support
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HOLDOUT_REPORT = PROJECT_ROOT / "results" / "training" / "stacking_v2_holdout" / "holdout_report.json"
OOF_PATH = PROJECT_ROOT / "results" / "training" / "stacking_v2" / "oof_predictions.npz"
OUTPUT_DIR = PROJECT_ROOT / "figures" / "training" / "stacking_v2_holdout"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Config colors ──
GROUP_COLORS = {
    "PRO": "#E91E63", "BRE": "#FF69B4", "OVA": "#AB47BC",
    "LUN": "#42A5F5", "CRC": "#EF5350", "PAN": "#FFA726", "BLC": "#7E57C2",
}
CAT_COLORS = {"cancer": "#E53935", "non_cancer": "#43A047"}

CANCER_TYPES = ["PRO", "LUN", "CRC", "PAN", "OVA", "BRE", "BLC"]
CANCER_SHORT = ["PRO", "LUN", "CRC", "PAN", "OVA", "BRE", "BLC"]
CANCER_FULL = [
    "Prostate", "Lung", "Colorectal",
    "Pancreatic", "Ovarian", "Breast", "Bladder"
]


# ═══════════════════════════════════════════════
# Plotting helpers
# ═══════════════════════════════════════════════

def plot_s2_cm(ax, cm, title, subtitle="", fontscale=1.0):
    """Plot 7-class cancer type confusion matrix."""
    n = len(CANCER_TYPES)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    for i in range(n):
        for j in range(n):
            frac = cm_norm[i, j]
            if i == j:
                rgb = mcolors.to_rgb(GROUP_COLORS[CANCER_TYPES[i]])
                alpha = 0.45 + 0.55 * frac
                bg = (*rgb, alpha)
            else:
                if frac > 0:
                    bg = (0.88 - 0.35 * frac,) * 3 + (1.0,)
                else:
                    bg = (0.96, 0.96, 0.96, 1.0)

            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                  facecolor=bg, edgecolor="white", lw=1.5)
            ax.add_patch(rect)

            val = cm[i, j]
            if val == 0:
                continue
            pct = frac * 100
            text_color = "white" if frac > 0.35 else "#333333"
            fs = 9.5 * fontscale
            if i == j:
                ax.text(j, i, f"{val}\n({pct:.0f}%)", ha="center", va="center",
                        fontsize=fs, fontweight="bold", color=text_color)
            else:
                ax.text(j, i, f"{val}", ha="center", va="center",
                        fontsize=fs * 0.85, color=text_color)

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))

    ax.set_xticklabels(CANCER_SHORT, fontsize=9 * fontscale, fontweight="bold", ha="center")
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    for i, tick in enumerate(ax.get_xticklabels()):
        tick.set_color(GROUP_COLORS[CANCER_TYPES[i]])

    ax.set_yticklabels(CANCER_SHORT, fontsize=9.5 * fontscale, fontweight="bold")
    for i, tick in enumerate(ax.get_yticklabels()):
        tick.set_color(GROUP_COLORS[CANCER_TYPES[i]])

    for i in range(n):
        recall = cm_norm[i, i]
        ax.text(n - 0.28, i, f"{recall:.1%}", ha="left", va="center",
                fontsize=8 * fontscale, fontweight="bold",
                color=GROUP_COLORS[CANCER_TYPES[i]])

    ax.set_xlabel("Predicted", fontsize=10.5 * fontscale, fontweight="bold", labelpad=6)
    ax.set_ylabel("True", fontsize=10.5 * fontscale, fontweight="bold")

    if subtitle:
        ax.set_title(f"{title}\n{subtitle}",
                     fontsize=11.5 * fontscale, fontweight="bold", pad=8,
                     linespacing=1.6)
    else:
        ax.set_title(title, fontsize=11.5 * fontscale, fontweight="bold", pad=8)


def plot_s1_cm(ax, cm, title, subtitle="", fontscale=1.0):
    """Plot 2-class binary confusion matrix."""
    labels = ["Non-cancer", "Cancer"]
    diag_colors = [CAT_COLORS["non_cancer"], CAT_COLORS["cancer"]]
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    for i in range(2):
        for j in range(2):
            frac = cm_norm[i, j]
            if i == j:
                rgb = mcolors.to_rgb(diag_colors[i])
                alpha = 0.45 + 0.55 * frac
                bg = (*rgb, alpha)
            else:
                bg = (0.92,) * 3 + (1.0,) if frac > 0 else (0.96,) * 3 + (1.0,)

            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                  facecolor=bg, edgecolor="white", lw=2)
            ax.add_patch(rect)

            val = cm[i, j]
            pct = frac * 100
            text_color = "white" if frac > 0.5 else "#333333"
            ax.text(j, i, f"{val}\n({pct:.0f}%)", ha="center", va="center",
                    fontsize=14 * fontscale, fontweight="bold", color=text_color)

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(1.5, -0.5)
    ax.set_xticks(range(2))
    ax.set_yticks(range(2))
    ax.set_xticklabels(labels, fontsize=11 * fontscale, fontweight="bold")
    ax.set_yticklabels(labels, fontsize=11 * fontscale, fontweight="bold")
    ax.get_yticklabels()[0].set_color(CAT_COLORS["non_cancer"])
    ax.get_yticklabels()[1].set_color(CAT_COLORS["cancer"])
    ax.set_xlabel("Predicted", fontsize=12 * fontscale, fontweight="bold", labelpad=6)
    ax.set_ylabel("True", fontsize=12 * fontscale, fontweight="bold")

    if subtitle:
        ax.set_title(f"{title}\n{subtitle}",
                     fontsize=13 * fontscale, fontweight="bold", pad=8,
                     linespacing=1.5)
    else:
        ax.set_title(title, fontsize=13 * fontscale, fontweight="bold", pad=8)


def build_s1_cm(result):
    """Reconstruct binary CM from balanced threshold sens/spec."""
    sens = result["s1_balanced_sens"]
    spec = result["s1_balanced_spec"]
    n_cancer = sum(result.get(f"s2_{ct}_n", 0) for ct in CANCER_TYPES)
    n_total = result["n_samples"]
    n_noncancer = n_total - n_cancer
    tp = round(sens * n_cancer)
    fn = n_cancer - tp
    tn = round(spec * n_noncancer)
    fp = n_noncancer - tn
    return np.array([[tn, fp], [fn, tp]])


# ═══════════════════════════════════════════════
# Load holdout data
# ═══════════════════════════════════════════════
with open(HOLDOUT_REPORT) as f:
    holdout = json.load(f)

split_titles = ["Train (OOF)", "Validation", "Test"]


# ═══════════════════════════════════════════════
# Fig 1: Holdout — Stage 1 Cancer Screening ONLY (3 panels)
# ═══════════════════════════════════════════════
fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5.5))
fig1.suptitle("STK-V2  Holdout — Stage 1: Cancer Screening",
              fontsize=17, fontweight="bold", y=1.01)

for idx, result in enumerate(holdout["results"]):
    cm_bin = build_s1_cm(result)
    auc = result["s1_auc"]
    sens = result["s1_balanced_sens"]
    spec = result["s1_balanced_spec"]
    plot_s1_cm(axes1[idx], cm_bin, split_titles[idx],
               f"AUC={auc:.4f}  Sens={sens:.3f}  Spec={spec:.3f}")

fig1.tight_layout(rect=[0, 0, 1, 0.92])
fig1.savefig(OUTPUT_DIR / "stk_v2_cm_cancer_screening.png", dpi=200,
             bbox_inches="tight", facecolor="white")
print("Saved: stk_v2_cm_cancer_screening.png")


# ═══════════════════════════════════════════════
# Fig 2: Holdout — Stage 2 Cancer Type ONLY (3 panels)
# ═══════════════════════════════════════════════
fig2, axes2 = plt.subplots(1, 3, figsize=(24, 7))
fig2.suptitle("STK-V2  Holdout — Stage 2: Cancer Type Classification",
              fontsize=17, fontweight="bold", y=1.01)

for idx, result in enumerate(holdout["results"]):
    cm = np.array(result["s2_confusion_matrix"])
    n = result["n_samples"]
    f1 = result["s2_f1_macro"]
    plot_s2_cm(axes2[idx], cm, split_titles[idx],
               f"n = {n}  |  Macro F1 = {f1:.3f}", fontscale=0.95)

fig2.tight_layout(rect=[0, 0, 1, 0.96])
fig2.savefig(OUTPUT_DIR / "stk_v2_cm_cancer_type.png", dpi=200,
             bbox_inches="tight", facecolor="white")
print("Saved: stk_v2_cm_cancer_type.png")


# ═══════════════════════════════════════════════
# Nested CV — reconstruct from OOF
# ═══════════════════════════════════════════════
print("\n--- Reconstructing Nested CV ---")
oof = np.load(OOF_PATH, allow_pickle=True)

MODEL_NAMES = [
    "lr_raw", "lr_d1", "lr_d2", "lr_concat", "lr_peak",
    "xgb_raw", "xgb_d1", "rf_raw", "rf_d1", "ridge_concat"
]

binary_labels = oof["binary_labels"].astype(int)
cancer_type_labels = oof["cancer_type_labels"].astype(int)
sample_ids = oof["sample_ids"]
n_samples = len(binary_labels)

meta_s1 = np.nan_to_num(
    np.column_stack([oof[f"{m}_s1"] for m in MODEL_NAMES]), nan=0.0)
meta_s2 = np.nan_to_num(
    np.hstack([oof[f"{m}_s2"] for m in MODEL_NAMES]), nan=0.0)

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
all_s1_pred = np.full(n_samples, np.nan)
all_s2_pred = np.full(n_samples, -1, dtype=int)

for fold, (tr_idx, te_idx) in enumerate(sgkf.split(meta_s1, binary_labels, sample_ids)):
    ml_s1 = LogisticRegression(C=0.5, penalty="elasticnet", l1_ratio=0.5,
                                solver="saga", max_iter=5000, random_state=42)
    ml_s1.fit(meta_s1[tr_idx], binary_labels[tr_idx])
    all_s1_pred[te_idx] = ml_s1.predict_proba(meta_s1[te_idx])[:, 1]

    cancer_tr = binary_labels[tr_idx] == 1
    cancer_te = binary_labels[te_idx] == 1
    te_cancer_idx = te_idx[cancer_te]
    if cancer_tr.sum() > 0 and cancer_te.sum() > 0:
        ml_s2 = LogisticRegression(C=0.5, penalty="elasticnet", l1_ratio=0.5,
                                    solver="saga", max_iter=5000, random_state=42)
        ml_s2.fit(meta_s2[tr_idx][cancer_tr], cancer_type_labels[tr_idx][cancer_tr])
        all_s2_pred[te_cancer_idx] = ml_s2.predict(meta_s2[te_cancer_idx])

# S1 metrics
fpr, tpr, thresholds = roc_curve(binary_labels, all_s1_pred)
best_thresh = thresholds[np.argmax(tpr - fpr)]
s1_pred_bin = (all_s1_pred >= best_thresh).astype(int)
s1_auc = roc_auc_score(binary_labels, all_s1_pred)
cm_s1_ncv = confusion_matrix(binary_labels, s1_pred_bin, labels=[0, 1])
tn, fp, fn, tp = cm_s1_ncv.ravel()
ncv_sens, ncv_spec = tp / (tp + fn), tn / (tn + fp)

# S2 metrics
cancer_mask = cancer_type_labels >= 0
s2_true = cancer_type_labels[cancer_mask]
s2_pred = all_s2_pred[cancer_mask]
valid_s2 = s2_pred >= 0
cm_s2_ncv = confusion_matrix(s2_true[valid_s2], s2_pred[valid_s2], labels=list(range(7)))
ncv_f1 = f1_score(s2_true[valid_s2], s2_pred[valid_s2], average="macro")

print(f"S1: AUC={s1_auc:.4f}, Sens={ncv_sens:.3f}, Spec={ncv_spec:.3f}")
print(f"S2: Macro F1={ncv_f1:.4f}")


# ═══════════════════════════════════════════════
# Fig 3: Nested CV — Stage 1 ONLY
# ═══════════════════════════════════════════════
fig3, ax3 = plt.subplots(figsize=(6, 5.5))
plot_s1_cm(ax3, cm_s1_ncv,
           f"STK-V2  Nested 5-Fold CV (n={n_samples})\nStage 1: Cancer Screening",
           f"AUC={s1_auc:.4f}  Sens={ncv_sens:.3f}  Spec={ncv_spec:.3f}",
           fontscale=1.05)
fig3.tight_layout()
fig3.savefig(OUTPUT_DIR / "stk_v2_cm_nested_cv_s1.png", dpi=200,
             bbox_inches="tight", facecolor="white")
print("Saved: stk_v2_cm_nested_cv_s1.png")


# ═══════════════════════════════════════════════
# Fig 4: Nested CV — Stage 2 ONLY
# ═══════════════════════════════════════════════
fig4, ax4 = plt.subplots(figsize=(8.5, 7))
plot_s2_cm(ax4, cm_s2_ncv,
           f"STK-V2  Nested 5-Fold CV (n={n_samples})",
           f"Stage 2: Cancer Type ID  |  Macro F1 = {ncv_f1:.4f}",
           fontscale=1.15)
fig4.tight_layout()
fig4.savefig(OUTPUT_DIR / "stk_v2_cm_nested_cv_s2.png", dpi=200,
             bbox_inches="tight", facecolor="white")
print("Saved: stk_v2_cm_nested_cv_s2.png")


# ═══════════════════════════════════════════════
# Fig 5: Per-cancer bar chart (Nested CV)
# ═══════════════════════════════════════════════
prec, rec, f1_per, sup = precision_recall_fscore_support(
    s2_true[valid_s2], s2_pred[valid_s2], labels=list(range(7)), zero_division=0
)

fig5, ax5 = plt.subplots(figsize=(13, 6))

x = np.arange(7)
w = 0.22
offsets = [-w, 0, w]
metric_names = ["Precision", "Recall", "F1"]
metric_short = ["P", "R", "F1"]
metric_vals = [prec, rec, f1_per]
alphas = [0.55, 0.75, 1.0]

for mi, (mname, mval, al) in enumerate(zip(metric_names, metric_vals, alphas)):
    bars = ax5.bar(x + offsets[mi], mval, w,
                   color=[GROUP_COLORS[ct] for ct in CANCER_TYPES],
                   alpha=al, edgecolor="white", lw=0.8)
    for bi, bar in enumerate(bars):
        h = bar.get_height()
        # Value label above bar
        ax5.text(bar.get_x() + bar.get_width() / 2, h + 0.015,
                 f".{int(h*1000):03d}" if h < 1.0 else "1.00",
                 ha="center", va="bottom",
                 fontsize=6.5, fontweight="bold", color="#444444")
        # Metric name inside first cancer's bars only
        if bi == 0:
            ax5.text(bar.get_x() + bar.get_width() / 2, h - 0.06,
                     metric_short[mi],
                     ha="center", va="top",
                     fontsize=8.5, fontweight="bold", color="white")

ax5.set_xticks(x)
xlabels_bar = [f"{ct} ({f})\nn={n}" for ct, f, n in
               zip(CANCER_SHORT, CANCER_FULL, sup)]
ax5.set_xticklabels(xlabels_bar, fontsize=8.5, fontweight="bold")
for i, tick in enumerate(ax5.get_xticklabels()):
    tick.set_color(GROUP_COLORS[CANCER_TYPES[i]])

ax5.set_ylim(0, 1.12)
ax5.set_ylabel("Score", fontsize=12, fontweight="bold")
ax5.set_title(
    f"STK-V2 Nested CV — Per-Cancer Type Metrics    (Macro F1 = {ncv_f1:.4f})",
    fontsize=14, fontweight="bold", pad=10)
ax5.grid(axis="y", alpha=0.2, lw=0.5)
ax5.spines["top"].set_visible(False)
ax5.spines["right"].set_visible(False)

fig5.tight_layout()
fig5.savefig(OUTPUT_DIR / "stk_v2_per_cancer_metrics_nested_cv.png", dpi=200,
             bbox_inches="tight", facecolor="white")
print("Saved: stk_v2_per_cancer_metrics_nested_cv.png")

plt.close("all")
print("\nDone — 5 figures generated.")
