"""
Figure 9: ROC Curves — uSERS-Net vs Baselines

(a) Stage 1: Cancer Screening — S1 ROC per model
(b) Stage 2: Cancer Type ID — per-cancer S2 ROC for uSERS-Net
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_curve, auc, roc_auc_score
from nature_style import (
    apply_style, apply_nature_style, add_panel_label, save_figure,
    AACR_LABELS, AACR_COLORS, AACR_CANCER_ORDER,
    MODEL_COLORS, MODEL_ORDER, BENCHMARK_DIR, OUR_MODEL,
    DOUBLE_COL, FONT_SIZE, LINE_WIDTH,
)

apply_style()

AACR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FUSION_PATH = os.path.join(AACR_DIR, "data", "early_fusion", "fold_predictions.npz")

BASELINE_MODELS = {
    "Logistic Regression": ("logistic_regression", "v004"),
    "Random Forest":       ("random_forest",       "v002"),
    "XGBoost":             ("xgboost",             "v001"),
    "CNN1D":               ("cnn1d",               "v001"),
    "ResNet18":            ("resnet18",             "v001"),
}
CANCER_TYPES  = ["PRO", "LUN", "CRC", "CPAN", "OVA"]
_TO_INTERNAL  = {"CPAN": "PAN"}

# AACR display order: PRO→PRC, OVA→OVC, LUN→LC, CPAN→PAC, CRC→CRC
AACR_INDEX_ORDER = [0, 4, 1, 3, 2]   # indices into CANCER_TYPES

# ── Load all predictions ──
model_preds = {}   # name → (bl, bp)

for name, (mdir, ver) in BASELINE_MODELS.items():
    npz = os.path.join(BENCHMARK_DIR, mdir, ver, "fold_predictions.npz")
    d   = np.load(npz, allow_pickle=True)
    model_preds[name] = (d["binary_labels"], d["val_binary_prob"])

fus = np.load(FUSION_PATH, allow_pickle=True)
bl_fus  = fus["binary_labels"]
bp_fus  = fus["val_binary_prob"]
cl_fus  = fus["val_cancer_logits"]
ctl_fus = fus["cancer_type_labels"]
model_preds[OUR_MODEL] = (bl_fus, bp_fus)


# ── Figure ──
fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.48), facecolor="white")
ax1 = fig.add_axes([0.09, 0.13, 0.37, 0.76])
ax2 = fig.add_axes([0.60, 0.13, 0.37, 0.76])

OUR_ACCENT = "#C0392B"

# ══ Panel (a): S1 ROC ══
for model in MODEL_ORDER:
    bl, bp = model_preds[model]
    fpr, tpr, _ = roc_curve(bl, bp)
    auc_val = auc(fpr, tpr)

    is_ours = (model == OUR_MODEL)
    color   = OUR_ACCENT if is_ours else MODEL_COLORS[model]
    lw      = 1.8 if is_ours else 0.9
    zorder  = 10 if is_ours else 2
    alpha   = 1.0 if is_ours else 0.75
    ls      = "-"

    short = {
        "Logistic Regression": "LR",
        "Random Forest": "RF",
        "XGBoost": "XGB",
        "CNN1D": "CNN1D",
        "ResNet18": "ResNet18",
        OUR_MODEL: OUR_MODEL,
    }[model]

    label = f"{short}  (AUC={auc_val:.3f})"
    ax1.plot(fpr, tpr, color=color, lw=lw, alpha=alpha, zorder=zorder,
             label=label)

ax1.plot([0, 1], [0, 1], color="#CCCCCC", lw=0.6, ls="--", zorder=0)
ax1.set_xlim(-0.02, 1.02)
ax1.set_ylim(-0.02, 1.02)
ax1.set_xlabel("1 – Specificity (FPR)", fontsize=FONT_SIZE["axis_label"])
ax1.set_ylabel("Sensitivity (TPR)", fontsize=FONT_SIZE["axis_label"])
ax1.set_title("Stage 1: Cancer Screening", fontsize=FONT_SIZE["title"] + 1,
              fontweight="bold", color="#2A2A2A")

legend = ax1.legend(fontsize=FONT_SIZE["legend"],
                    loc="lower right", frameon=True,
                    framealpha=0.92, edgecolor="#DDDDDD",
                    handlelength=1.4, handleheight=0.8)
for text, model in zip(legend.get_texts(), MODEL_ORDER):
    if model == OUR_MODEL:
        text.set_fontweight("bold")
        text.set_color(OUR_ACCENT)

apply_nature_style(ax1)
add_panel_label(ax1, "a", x=-0.15, y=1.08)


# ══ Panel (b): S2 per-cancer ROC for uSERS-Net ══
cancer_mask = bl_fus == 1
ctl_cancer  = ctl_fus[cancer_mask]
cl_cancer   = cl_fus[cancer_mask]

for ci in AACR_INDEX_ORDER:
    cname    = CANCER_TYPES[ci]
    internal = _TO_INTERNAL.get(cname, cname)
    display  = AACR_LABELS[internal]
    color    = AACR_COLORS[display]

    y_true = (ctl_cancer == ci).astype(int)
    y_score = cl_cancer[:, ci]

    try:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_val = auc(fpr, tpr)
        ax2.plot(fpr, tpr, color=color, lw=1.4,
                 label=f"{display}  (AUC={auc_val:.3f})")
    except ValueError:
        pass

ax2.plot([0, 1], [0, 1], color="#CCCCCC", lw=0.6, ls="--", zorder=0)
ax2.set_xlim(-0.02, 1.02)
ax2.set_ylim(-0.02, 1.02)
ax2.set_xlabel("1 – Specificity (FPR)", fontsize=FONT_SIZE["axis_label"])
ax2.set_ylabel("Sensitivity (TPR)", fontsize=FONT_SIZE["axis_label"])
ax2.set_title(f"{OUR_MODEL}: Stage 2 Cancer Type ID",
              fontsize=FONT_SIZE["title"] + 1, fontweight="bold",
              color="#2A2A2A")

legend2 = ax2.legend(fontsize=FONT_SIZE["legend"],
                     loc="lower right", frameon=True,
                     framealpha=0.92, edgecolor="#DDDDDD",
                     handlelength=1.4, handleheight=0.8)
for text, ci in zip(legend2.get_texts(), AACR_INDEX_ORDER):
    cname    = CANCER_TYPES[ci]
    internal = _TO_INTERNAL.get(cname, cname)
    display  = AACR_LABELS[internal]
    text.set_color(AACR_COLORS[display])
    text.set_fontweight("bold")

apply_nature_style(ax2)
add_panel_label(ax2, "b", x=-0.15, y=1.08)


# ── Subtitle ──
fig.text(0.50, 0.01,
         f"5-fold stratified CV  |  n = 6,200 spectra (all replicates)  |  "
         f"{OUR_MODEL}: Ensemble (Fusion LR + ResNet18, α=0.8)",
         ha="center", va="bottom",
         fontsize=FONT_SIZE["annotation"] + 0.5, color="#888888", style="italic")

save_figure(fig, "fig9_auc_curves")
plt.close()

print("\n=== AUC Summary ===")
print(f"{'Model':<22}  S1 AUC")
print("-" * 35)
for model in MODEL_ORDER:
    bl, bp = model_preds[model]
    a = roc_auc_score(bl, bp)
    tag = " ***" if model == OUR_MODEL else ""
    short = {
        "Logistic Regression": "LR", "Random Forest": "RF", "XGBoost": "XGB",
        "CNN1D": "CNN1D", "ResNet18": "ResNet18", OUR_MODEL: OUR_MODEL,
    }[model]
    print(f"  {short:<20}  {a:.4f}{tag}")

print(f"\n{OUR_MODEL} S2 per-cancer AUC:")
for ci in AACR_INDEX_ORDER:
    cname   = CANCER_TYPES[ci]
    display = AACR_LABELS[_TO_INTERNAL.get(cname, cname)]
    y_true  = (ctl_cancer == ci).astype(int)
    y_score = cl_cancer[:, ci]
    a = roc_auc_score(y_true, y_score)
    print(f"  {display}: {a:.4f}")
