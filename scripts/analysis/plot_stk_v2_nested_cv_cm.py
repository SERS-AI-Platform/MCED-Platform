"""
STK-V2 Nested CV Confusion Matrices
Reconstruct from oof_predictions.npz using ElasticNet meta-learner
with 5-fold StratifiedGroupKFold — identical to experiment_summary.json.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, f1_score, classification_report
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OOF_PATH = PROJECT_ROOT / "results" / "training" / "stacking_v2" / "oof_predictions.npz"
OUTPUT_DIR = PROJECT_ROOT / "figures" / "training" / "stacking_v2_holdout"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load OOF predictions ──
oof = np.load(OOF_PATH, allow_pickle=True)

MODEL_NAMES = [
    "lr_raw", "lr_d1", "lr_d2", "lr_concat", "lr_peak",
    "xgb_raw", "xgb_d1", "rf_raw", "rf_d1", "ridge_concat"
]
CANCER_TYPES = ["PRO", "LUN", "CRC", "PAN", "OVA", "BRE", "BLC"]
CANCER_LABELS = [
    "PRO\n(Prostate)", "LUN\n(Lung)", "CRC\n(Colorectal)",
    "PAN\n(Pancreatic)", "OVA\n(Ovarian)", "BRE\n(Breast)", "BLC\n(Bladder)"
]

binary_labels = oof["binary_labels"].astype(int)
cancer_type_labels = oof["cancer_type_labels"].astype(int)
sample_ids = oof["sample_ids"]
n_samples = len(binary_labels)

# Build meta-feature matrices
meta_s1 = np.column_stack([oof[f"{m}_s1"] for m in MODEL_NAMES])  # (1628, 10)
meta_s2 = np.hstack([oof[f"{m}_s2"] for m in MODEL_NAMES])        # (1628, 70)
meta_s1 = np.nan_to_num(meta_s1, nan=0.0)
meta_s2 = np.nan_to_num(meta_s2, nan=0.0)

# ── Nested CV: 5-fold, ElasticNet meta ──
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

all_s1_pred = np.full(n_samples, np.nan)
all_s1_true = binary_labels.copy()
all_s2_pred = np.full(n_samples, -1, dtype=int)
all_s2_true = cancer_type_labels.copy()

for fold, (tr_idx, te_idx) in enumerate(sgkf.split(meta_s1, binary_labels, sample_ids)):
    # Stage 1: binary
    ml_s1 = LogisticRegression(
        C=0.5, penalty="elasticnet", l1_ratio=0.5,
        solver="saga", max_iter=5000, random_state=42
    )
    ml_s1.fit(meta_s1[tr_idx], binary_labels[tr_idx])
    s1_prob = ml_s1.predict_proba(meta_s1[te_idx])[:, 1]
    all_s1_pred[te_idx] = s1_prob

    # Stage 2: cancer type (cancer samples only)
    cancer_tr = binary_labels[tr_idx] == 1
    cancer_te = binary_labels[te_idx] == 1
    te_cancer_idx = te_idx[cancer_te]

    if cancer_tr.sum() > 0 and cancer_te.sum() > 0:
        ml_s2 = LogisticRegression(
            C=0.5, penalty="elasticnet", l1_ratio=0.5,
            solver="saga", max_iter=5000, random_state=42
        )
        ml_s2.fit(meta_s2[tr_idx][cancer_tr], cancer_type_labels[tr_idx][cancer_tr])
        s2_pred = ml_s2.predict(meta_s2[te_cancer_idx])
        all_s2_pred[te_cancer_idx] = s2_pred

    print(f"Fold {fold}: te={len(te_idx)}, cancer_te={cancer_te.sum()}")

# ── Compute metrics ──
# Stage 1: threshold at Youden's J
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(all_s1_true, all_s1_pred)
j_scores = tpr - fpr
best_thresh = thresholds[np.argmax(j_scores)]
s1_pred_binary = (all_s1_pred >= best_thresh).astype(int)
s1_auc = roc_auc_score(all_s1_true, all_s1_pred)

# Stage 1 CM
cm_s1 = confusion_matrix(all_s1_true, s1_pred_binary, labels=[0, 1])
tn, fp, fn, tp = cm_s1.ravel()
sens = tp / (tp + fn)
spec = tn / (tn + fp)

print(f"\n=== Stage 1 (Cancer Screening) ===")
print(f"AUC: {s1_auc:.4f}")
print(f"Threshold (Youden): {best_thresh:.4f}")
print(f"Sensitivity: {sens:.4f}, Specificity: {spec:.4f}")
print(f"CM:\n{cm_s1}")

# Stage 2: cancer samples only
cancer_mask = all_s2_true >= 0
s2_true_cancer = all_s2_true[cancer_mask]
s2_pred_cancer = all_s2_pred[cancer_mask]
valid_s2 = s2_pred_cancer >= 0
s2_true_v = s2_true_cancer[valid_s2]
s2_pred_v = s2_pred_cancer[valid_s2]

cm_s2 = confusion_matrix(s2_true_v, s2_pred_v, labels=list(range(7)))
s2_f1 = f1_score(s2_true_v, s2_pred_v, average="macro")

print(f"\n=== Stage 2 (Cancer Type ID) ===")
print(f"Macro F1: {s2_f1:.4f}")
print(f"CM:\n{cm_s2}")
print(classification_report(s2_true_v, s2_pred_v, target_names=CANCER_TYPES, digits=4))


# ═══════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════

def plot_cm(ax, cm, labels, title, cmap="Blues"):
    n_classes = len(labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1, aspect="equal")

    for i in range(n_classes):
        for j in range(n_classes):
            val = cm[i, j]
            pct = cm_norm[i, j] * 100
            color = "white" if cm_norm[i, j] > 0.6 else "black"
            if val > 0:
                text = f"{val}\n({pct:.0f}%)"
            else:
                text = "0"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=10 if n_classes <= 2 else 7.5,
                    fontweight="bold" if i == j else "normal",
                    color=color)

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=11, fontweight="bold")
    ax.set_ylabel("True", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    # Recall on right
    for i in range(n_classes):
        recall = cm_norm[i, i]
        ax.text(n_classes - 0.3, i, f"{recall:.1%}",
                ha="left", va="center", fontsize=8, color="#1565C0",
                fontweight="bold")


# ── Figure: Combined S1 + S2 ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6),
                                gridspec_kw={"width_ratios": [1, 2.5]})
fig.suptitle(
    f"STK-V2  |  Nested 5-Fold CV (n={n_samples}, ElasticNet meta) — Confusion Matrices",
    fontsize=14, fontweight="bold", y=1.02
)

# S1
plot_cm(ax1, cm_s1, ["Non-cancer", "Cancer"],
        f"Stage 1: Cancer Screening\nAUC={s1_auc:.4f}  Sens={sens:.3f}  Spec={spec:.3f}",
        cmap="Oranges")

# S2
plot_cm(ax2, cm_s2, CANCER_LABELS,
        f"Stage 2: Cancer Type ID\nMacro F1={s2_f1:.4f}",
        cmap="Blues")

fig.tight_layout()
out_path = OUTPUT_DIR / "stk_v2_cm_nested_cv.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"\nSaved: {out_path}")


# ── Figure: S2 per-cancer metrics bar chart ──
fig2, ax = plt.subplots(figsize=(10, 5))

from sklearn.metrics import precision_recall_fscore_support
prec, rec, f1_per, sup = precision_recall_fscore_support(
    s2_true_v, s2_pred_v, labels=list(range(7)), zero_division=0
)

x = np.arange(7)
w = 0.25
bars1 = ax.bar(x - w, prec, w, label="Precision", color="#1976D2", alpha=0.85)
bars2 = ax.bar(x, rec, w, label="Recall", color="#FF5722", alpha=0.85)
bars3 = ax.bar(x + w, f1_per, w, label="F1", color="#4CAF50", alpha=0.85)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.3f}",
                ha="center", va="bottom", fontsize=7, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels([f"{ct}\n(n={n})" for ct, n in zip(CANCER_TYPES, sup)], fontsize=9)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Score", fontsize=11, fontweight="bold")
ax.set_title(f"STK-V2 Nested CV — Per-Cancer Precision / Recall / F1  (Macro F1={s2_f1:.4f})",
             fontsize=12, fontweight="bold")
ax.legend(loc="lower right", fontsize=10)
ax.grid(axis="y", alpha=0.3)

fig2.tight_layout()
out_path2 = OUTPUT_DIR / "stk_v2_per_cancer_metrics_nested_cv.png"
fig2.savefig(out_path2, dpi=200, bbox_inches="tight")
print(f"Saved: {out_path2}")

plt.close("all")
print("\nDone — 2 figures generated.")
