"""
Figure 3: ROC curves for the Ensemble model.
(a) Stage 1 — Cancer vs Non-Cancer binary screening.
(b) Stage 2 — Per-cancer-type One-vs-Rest ROC.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
from sklearn.metrics import roc_curve, auc
from nature_style import (
    apply_style, apply_nature_style, add_panel_label, save_figure,
    CANCER_COLORS, CANCER_LABELS, NON_CANCER_COLOR,
    DOUBLE_COL, FONT_SIZE, LINE_WIDTH, ROOT,
)

apply_style()

# ── Load ensemble predictions ──
npz_path = os.path.join(ROOT, "results", "training", "step5_ensemble", "fold_predictions.npz")
d = np.load(npz_path, allow_pickle=True)

alpha = 0.8
binary_prob = alpha * d["val_bp_lr"] + (1 - alpha) * d["val_bp_rn"]
binary_labels = d["binary_labels"]

# Cancer logits: blend then softmax
cancer_logits = alpha * d["val_cl_lr"] + (1 - alpha) * d["val_cl_rn"]
cancer_type_labels = d["cancer_type_labels"]

# Cancer type mapping: ['PRO', 'LUN', 'CRC', 'CPAN', 'OVA'] → index 0-4
CANCER_TYPE_ORDER = ["PRO", "LUN", "CRC", "CPAN", "OVA"]
DISPLAY_ORDER = ["PRO", "LUN", "CRC", "PAN", "OVA"]

# ── Figure: 2 panels ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.42))
fig.subplots_adjust(wspace=0.35)

# ── Panel (a): Stage 1 Binary ROC ──
add_panel_label(ax1, "a")

fpr, tpr, thresholds = roc_curve(binary_labels, binary_prob)
roc_auc = auc(fpr, tpr)

# Youden's J
j_scores = tpr - fpr
opt_idx = np.argmax(j_scores)
opt_fpr, opt_tpr = fpr[opt_idx], tpr[opt_idx]
opt_t = thresholds[opt_idx]

ax1.plot(fpr, tpr, color="#2C3E50", linewidth=LINE_WIDTH["roc"],
         label=f"Ensemble (AUC = {roc_auc:.3f})")
ax1.plot([0, 1], [0, 1], "--", color="#BBBBBB", linewidth=LINE_WIDTH["thin"])
ax1.plot(opt_fpr, opt_tpr, "o", color="#D4A017", markersize=5, markeredgecolor="white",
         markeredgewidth=0.8, zorder=5)

# Annotate optimal point
sens = opt_tpr
spec = 1 - opt_fpr
ax1.annotate(
    f"Sens = {sens:.1%}\nSpec = {spec:.1%}",
    xy=(opt_fpr, opt_tpr),
    xytext=(opt_fpr + 0.15, opt_tpr - 0.12),
    fontsize=FONT_SIZE["annotation"],
    arrowprops=dict(arrowstyle="-", color="#666666", lw=0.5),
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CCCCCC", lw=0.5),
)

ax1.set_xlabel("1 - Specificity")
ax1.set_ylabel("Sensitivity")
ax1.set_title("Stage 1: Cancer Screening", fontsize=FONT_SIZE["title"], pad=8)
ax1.legend(loc="lower right", frameon=True, edgecolor="#DDDDDD",
           fancybox=False, framealpha=0.9)
ax1.set_xlim(-0.02, 1.02)
ax1.set_ylim(-0.02, 1.02)
ax1.set_aspect("equal")
apply_nature_style(ax1)

# ── Panel (b): Stage 2 Per-Cancer OvR ROC ──
add_panel_label(ax2, "b")

cancer_mask = cancer_type_labels >= 0
ct_labels = cancer_type_labels[cancer_mask]
ct_probs = softmax(cancer_logits[cancer_mask], axis=1)

for idx, (fold_name, display_name) in enumerate(zip(CANCER_TYPE_ORDER, DISPLAY_ORDER)):
    color = CANCER_COLORS[display_name]
    label_display = CANCER_LABELS[display_name]

    binary_ovr = (ct_labels == idx).astype(int)
    prob_ovr = ct_probs[:, idx]

    fpr_c, tpr_c, _ = roc_curve(binary_ovr, prob_ovr)
    auc_c = auc(fpr_c, tpr_c)
    n_samples = binary_ovr.sum()

    ax2.plot(fpr_c, tpr_c, color=color, linewidth=LINE_WIDTH["roc"],
             label=f"{label_display} (n={n_samples}, AUC = {auc_c:.3f})")

ax2.plot([0, 1], [0, 1], "--", color="#BBBBBB", linewidth=LINE_WIDTH["thin"])
ax2.set_xlabel("1 - Specificity")
ax2.set_ylabel("Sensitivity")
ax2.set_title("Stage 2: Cancer Type Identification", fontsize=FONT_SIZE["title"], pad=8)
ax2.legend(loc="lower right", frameon=True, edgecolor="#DDDDDD",
           fancybox=False, framealpha=0.9, fontsize=FONT_SIZE["annotation"])
ax2.set_xlim(-0.02, 1.02)
ax2.set_ylim(-0.02, 1.02)
ax2.set_aspect("equal")
apply_nature_style(ax2)

save_figure(fig, "fig3_roc_curves")
plt.close()
print(f"Figure 3 complete. Optimal threshold = {opt_t:.3f}")
