"""
Figure 8: Per-subgroup (cancer type) model benchmark — heatmap.

Mono-tone cells with cancer-type colors on column labels only.
uSERS-Net row highlighted with border + bold text.

(a) Stage 1 — Detection sensitivity per cancer type
(b) Stage 2 — Type identification F1 per cancer type
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from sklearn.metrics import roc_curve
from nature_style import (
    apply_style, apply_nature_style, add_panel_label, save_figure,
    AACR_LABELS, AACR_COLORS, AACR_CANCER_ORDER,
    MODEL_COLORS, MODEL_ORDER, BENCHMARK_DIR, OUR_MODEL,
    DOUBLE_COL, FONT_SIZE, LINE_WIDTH, SERS_ROOT,
)

apply_style()

# ── Model paths ──
BASELINE_MODELS = {
    "Logistic Regression": ("logistic_regression", "v004"),
    "Random Forest":       ("random_forest",       "v002"),
    "XGBoost":             ("xgboost",             "v001"),
    "CNN1D":               ("cnn1d",               "v001"),
    "ResNet18":            ("resnet18",             "v001"),
}
FUSION_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "data", "early_fusion")
CANCER_TYPES = ["PRO", "LUN", "CRC", "CPAN", "OVA"]
_DATA_TO_INTERNAL = {"CPAN": "PAN"}

AACR_INDEX_ORDER = [0, 4, 1, 3, 2]  # PRO, OVA, LUN, CPAN, CRC
AACR_DISPLAY = [
    AACR_LABELS[_DATA_TO_INTERNAL.get(CANCER_TYPES[i], CANCER_TYPES[i])]
    for i in AACR_INDEX_ORDER
]

MODEL_SHORT = {
    "Logistic Regression": "LR",
    "Random Forest": "RF",
    "XGBoost": "XGB",
    "CNN1D": "CNN1D",
    "ResNet18": "ResNet18",
    OUR_MODEL: OUR_MODEL,
}


def compute_metrics(data):
    bl = data["binary_labels"]
    ctl = data["cancer_type_labels"]
    bp = data["val_binary_prob"]
    cl = data["val_cancer_logits"]

    fpr, tpr, thresholds = roc_curve(bl, bp)
    thr = thresholds[np.argmax(tpr - fpr)]
    pred_binary = (bp >= thr).astype(int)

    det_sens, type_f1 = {}, {}
    cancer_mask = bl == 1
    pred_type = cl[cancer_mask].argmax(axis=1)
    true_type = ctl[cancer_mask]

    for ci in range(len(CANCER_TYPES)):
        mask = ctl == ci
        n = mask.sum()
        det_sens[ci] = pred_binary[mask].mean() if n > 0 else 0.0

        tp = ((pred_type == ci) & (true_type == ci)).sum()
        fp = ((pred_type == ci) & (true_type != ci)).sum()
        fn = ((pred_type != ci) & (true_type == ci)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        type_f1[ci] = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return det_sens, type_f1


def compute_fusion_metrics():
    data = np.load(os.path.join(FUSION_PATH, "fold_predictions.npz"),
                   allow_pickle=True)
    bl  = data["binary_labels"]
    ctl = data["cancer_type_labels"]
    bp  = data["val_binary_prob"]
    cl  = data["val_cancer_logits"]

    fpr, tpr, thresholds = roc_curve(bl, bp)
    thr = thresholds[np.argmax(tpr - fpr)]
    pred_binary = (bp >= thr).astype(int)

    det_sens, type_f1 = {}, {}
    cancer_mask = bl == 1
    pred_type = cl[cancer_mask].argmax(axis=1)
    true_type  = ctl[cancer_mask]

    for ci in range(len(CANCER_TYPES)):
        mask = ctl == ci
        n = mask.sum()
        det_sens[ci] = pred_binary[mask].mean() if n > 0 else 0.0

        tp = ((pred_type == ci) & (true_type == ci)).sum()
        fp = ((pred_type == ci) & (true_type != ci)).sum()
        fn = ((pred_type != ci) & (true_type == ci)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        type_f1[ci] = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return det_sens, type_f1


# ── Collect metrics ──
all_det, all_f1 = {}, {}

for model_name, (model_dir, version) in BASELINE_MODELS.items():
    npz = os.path.join(BENCHMARK_DIR, model_dir, version, "fold_predictions.npz")
    data = np.load(npz, allow_pickle=True)
    det, f1 = compute_metrics(data)
    all_det[model_name] = det
    all_f1[model_name] = f1

fus_det, fus_f1 = compute_fusion_metrics()
all_det[OUR_MODEL] = fus_det
all_f1[OUR_MODEL] = fus_f1


def build_matrix(metric_dict):
    mat = np.zeros((len(MODEL_ORDER), len(AACR_INDEX_ORDER)))
    for i, model_name in enumerate(MODEL_ORDER):
        for j, ci in enumerate(AACR_INDEX_ORDER):
            mat[i, j] = metric_dict[model_name][ci]
    return mat


det_mat = build_matrix(all_det)
f1_mat = build_matrix(all_f1)

# ── Style ──
OUR_IDX = MODEL_ORDER.index(OUR_MODEL)
OUR_ACCENT = "#C0392B"

# Mono colormap: very light, AACR poster style (white → medium gray)
vmin, vmax = 0.25, 1.0
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
cmap = mcolors.LinearSegmentedColormap.from_list(
    "mono", [(0, "#FFFFFF"), (0.4, "#E8E8E8"), (0.75, "#C0C0C0"), (1, "#808080")]
)


def draw_heatmap(ax, mat, title, panel_label, show_ylabel=True):
    n_rows, n_cols = mat.shape

    for i in range(n_rows):
        is_ours = (i == OUR_IDX)
        for j in range(n_cols):
            val = mat[i, j]
            color = cmap(norm(val))

            pad = 0.04
            rect = mpatches.FancyBboxPatch(
                (j + pad, n_rows - 1 - i + pad), 1 - 2 * pad, 1 - 2 * pad,
                boxstyle="round,pad=0.02",
                facecolor=color, edgecolor="white", linewidth=0.8,
            )
            ax.add_patch(rect)

            # Text contrast (lighter palette → dark text only)
            lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            text_color = "#F5F5F5" if lum < 0.40 else "#2A2A2A"

            fw = "bold" if is_ours else "regular"
            fs = FONT_SIZE["legend"] + 1.5 if is_ours else FONT_SIZE["legend"] + 0.5

            ax.text(j + 0.5, n_rows - 1 - i + 0.5, f"{val:.2f}",
                    ha="center", va="center", fontsize=fs,
                    fontweight=fw, color=text_color, zorder=3)

    # Our model row — accent border
    our_y = n_rows - 1 - OUR_IDX
    highlight = mpatches.FancyBboxPatch(
        (-0.06, our_y - 0.03), n_cols + 0.12, 1.06,
        boxstyle="round,pad=0.04",
        facecolor="none", edgecolor=OUR_ACCENT, linewidth=1.8, zorder=4,
    )
    ax.add_patch(highlight)

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_aspect("equal")

    # Column labels — cancer-type colors on text only
    ax.set_xticks([j + 0.5 for j in range(n_cols)])
    ax.set_xticklabels(AACR_DISPLAY, fontsize=FONT_SIZE["title"],
                       fontweight="bold")
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    for tick, name in zip(ax.get_xticklabels(), AACR_DISPLAY):
        tick.set_color(AACR_COLORS[name])

    # Row labels
    if show_ylabel:
        ax.set_yticks([n_rows - 1 - i + 0.5 for i in range(n_rows)])
        ax.set_yticklabels([MODEL_SHORT[m] for m in MODEL_ORDER],
                           fontsize=FONT_SIZE["axis_label"] + 0.5)
        for tick, model in zip(ax.get_yticklabels(), MODEL_ORDER):
            if model == OUR_MODEL:
                tick.set_fontweight("bold")
                tick.set_color(OUR_ACCENT)
            else:
                tick.set_color("#555555")
    else:
        ax.set_yticks([])

    # Title
    ax.set_title(title, fontsize=FONT_SIZE["title"] + 1, fontweight="bold",
                 pad=16, color="#2A2A2A")

    add_panel_label(ax, panel_label,
                    x=-0.20 if show_ylabel else -0.05, y=1.20)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0)


# ── Figure ──
fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.48), facecolor="white")
ax1 = fig.add_axes([0.13, 0.10, 0.34, 0.72])
ax2 = fig.add_axes([0.56, 0.10, 0.34, 0.72])

draw_heatmap(ax1, det_mat, "Stage 1: Cancer Screening", "a", show_ylabel=True)
draw_heatmap(ax2, f1_mat, "Stage 2: Cancer Type ID", "b", show_ylabel=False)

# Subtitle — model description
fig.text(0.50, 0.02,
         "5-fold stratified CV  |  n = 6,200 spectra (all replicates)  |  "
         f"{OUR_MODEL}: Ensemble (Fusion LR + ResNet18, α=0.8)  |  S1 AUC = 0.987",
         ha="center", va="bottom",
         fontsize=FONT_SIZE["annotation"] + 0.5, color="#888888",
         style="italic")

# Colorbar
cbar_ax = fig.add_axes([0.92, 0.14, 0.012, 0.56])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.ax.tick_params(labelsize=FONT_SIZE["tick"], length=2, width=0.3)
cbar.outline.set_linewidth(0.3)

save_figure(fig, "fig8_subgroup_benchmark")
plt.close()

# ── Summary ──
print(f"\n=== Per-Subgroup Benchmark: {OUR_MODEL} vs Baselines ===")
print(f"{'':22}  {'  '.join(f'{d:>5}' for d in AACR_DISPLAY)}")
print("-" * 58)
for label, metric in [("Detection Sensitivity", all_det), ("Type ID F1", all_f1)]:
    print(f"\n{label}:")
    for m in MODEL_ORDER:
        tag = " ***" if m == OUR_MODEL else ""
        print(f"  {MODEL_SHORT[m]:<20}  {'  '.join(f'{metric[m][ci]:5.3f}' for ci in AACR_INDEX_ORDER)}{tag}")
