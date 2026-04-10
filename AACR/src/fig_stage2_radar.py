"""
Figure: Stage 2 Cancer Type ID — Radar chart comparison (6 models × 5 cancer types).

uSERS-Net polygon highlighted in red; baselines in muted tones.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_curve
from nature_style import (
    apply_style, save_figure,
    AACR_LABELS, AACR_COLORS,
    MODEL_COLORS, MODEL_ORDER, BENCHMARK_DIR, OUR_MODEL,
    SINGLE_COL, DOUBLE_COL, FONT_SIZE, LINE_WIDTH,
)

apply_style()

AACR_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FUSION_PATH = os.path.join(AACR_DIR, "data", "early_fusion", "fold_predictions.npz")

BASELINE_MODELS = {
    "Logistic Regression": ("logistic_regression", "v004"),
    "Random Forest":       ("random_forest",       "v002"),
    "XGBoost":             ("xgboost",             "v001"),
    "CNN1D":               ("cnn1d",               "v001"),
    "ResNet18":            ("resnet18",             "v001"),
}
CANCER_TYPES     = ["PRO", "LUN", "CRC", "CPAN", "OVA"]
_DATA_TO_INTERNAL = {"CPAN": "PAN"}
AACR_INDEX_ORDER = [0, 4, 1, 3, 2]   # PRO, OVA, LUN, CPAN, CRC → PRC, OVC, LC, PAC, CRC
AACR_DISPLAY = [
    AACR_LABELS[_DATA_TO_INTERNAL.get(CANCER_TYPES[i], CANCER_TYPES[i])]
    for i in AACR_INDEX_ORDER
]


def compute_type_f1(data):
    bl  = data["binary_labels"]
    ctl = data["cancer_type_labels"]
    cl  = data["val_cancer_logits"]

    cancer_mask = bl == 1
    pred_type   = cl[cancer_mask].argmax(axis=1)
    true_type   = ctl[cancer_mask]

    f1 = {}
    for ci in range(len(CANCER_TYPES)):
        tp = ((pred_type == ci) & (true_type == ci)).sum()
        fp = ((pred_type == ci) & (true_type != ci)).sum()
        fn = ((pred_type != ci) & (true_type == ci)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1[ci] = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return f1


# ── Collect F1 per model ──
all_f1 = {}
for name, (mdir, ver) in BASELINE_MODELS.items():
    npz  = os.path.join(BENCHMARK_DIR, mdir, ver, "fold_predictions.npz")
    data = np.load(npz, allow_pickle=True)
    all_f1[name] = compute_type_f1(data)

fus_data = np.load(FUSION_PATH, allow_pickle=True)
all_f1[OUR_MODEL] = compute_type_f1(fus_data)

# ── Radar setup ──
N      = len(AACR_INDEX_ORDER)   # 5 axes
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]             # close the polygon

OUR_ACCENT   = "#C0392B"
OUR_ALPHA      = 0.30
BAND_ALPHA     = 0.18

MODEL_SHORT = {
    "Logistic Regression": "LR",
    "Random Forest":       "RF",
    "XGBoost":             "XGB",
    "CNN1D":               "CNN1D",
    "ResNet18":            "ResNet18",
    OUR_MODEL:             OUR_MODEL,
}

# ── Baseline min/max band ──
baseline_names = [m for m in MODEL_ORDER if m != OUR_MODEL]
baseline_matrix = np.array([
    [all_f1[m][ci] for ci in AACR_INDEX_ORDER]
    for m in baseline_names
])
band_min = baseline_matrix.min(axis=0).tolist() + [baseline_matrix.min(axis=0)[0]]
band_max = baseline_matrix.max(axis=0).tolist() + [baseline_matrix.max(axis=0)[0]]
band_best = baseline_matrix.max(axis=0).tolist() + [baseline_matrix.max(axis=0)[0]]

fig_size = DOUBLE_COL * 0.62
fig, ax = plt.subplots(figsize=(fig_size, fig_size),
                       subplot_kw=dict(polar=True), facecolor="white")

# ── Draw grid rings ──
ring_vals = [0.25, 0.50, 0.75, 1.00]
for rv in ring_vals:
    ring = [rv] * N + [rv]
    ax.plot(angles, ring, color="#DDDDDD", lw=0.5, zorder=0)
    if rv < 1.0:
        ax.text(angles[0], rv + 0.02, f"{rv:.2f}",
                ha="center", va="bottom",
                fontsize=FONT_SIZE["annotation"] - 0.5, color="#AAAAAA")

# ── Draw axis spokes ──
for ang in angles[:-1]:
    ax.plot([ang, ang], [0, 1.0], color="#DDDDDD", lw=0.5, zorder=0)

# ── Gray band: min–max range of all baselines ──
ax.fill_between(angles, band_min, band_max,
                color="#AAAAAA", alpha=BAND_ALPHA, zorder=1)
ax.plot(angles, band_max, color="#888888", lw=1.2,
        ls="--", alpha=0.7, zorder=2, label="Best baseline")
ax.plot(angles, band_min, color="#BBBBBB", lw=0.6,
        ls=":", alpha=0.5, zorder=2)

# ── uSERS-Net: bold red polygon ──
our_vals = [all_f1[OUR_MODEL][ci] for ci in AACR_INDEX_ORDER] + \
           [all_f1[OUR_MODEL][AACR_INDEX_ORDER[0]]]
ax.fill(angles, our_vals, color=OUR_ACCENT, alpha=OUR_ALPHA, zorder=4)
ax.plot(angles, our_vals, color=OUR_ACCENT, lw=2.4, zorder=5,
        label=OUR_MODEL)

# ── Cancer-type axis labels (colored, bold) ──
ax.set_xticks(angles[:-1])
ax.set_xticklabels([])

label_r = 1.18
for ang, display in zip(angles[:-1], AACR_DISPLAY):
    color = AACR_COLORS[display]
    ha = "center"
    if ang < np.pi / 2 or ang > 3 * np.pi / 2:
        ha = "left"
    elif np.pi / 2 < ang < 3 * np.pi / 2:
        ha = "right"
    ax.text(ang, label_r, display,
            ha=ha, va="center",
            fontsize=FONT_SIZE["title"] + 1, fontweight="bold",
            color=color, transform=ax.transData)

# ── Remove default radial ticks and spine ──
ax.set_yticks([])
ax.set_yticklabels([])
ax.spines["polar"].set_visible(False)
ax.set_rlim(0, 1.0)

# ── Legend ──
handles = [
    mpatches.Patch(facecolor=OUR_ACCENT, alpha=OUR_ALPHA,
                   edgecolor=OUR_ACCENT, linewidth=2.0, label=OUR_MODEL),
    mpatches.Patch(facecolor="#AAAAAA", alpha=BAND_ALPHA,
                   edgecolor="#888888", linewidth=1.0,
                   label="Baseline range (LR / RF / XGB / CNN1D / ResNet18)"),
]
ax.legend(handles=handles,
          loc="lower center",
          bbox_to_anchor=(0.50, -0.20),
          ncol=1,
          fontsize=FONT_SIZE["legend"] + 0.5,
          frameon=True, framealpha=0.92,
          edgecolor="#DDDDDD",
          handlelength=1.4, handleheight=1.0)

ax.set_title("Stage 2: Cancer Type Identification (F1)",
             fontsize=FONT_SIZE["title"] + 1, fontweight="bold",
             color="#2A2A2A", pad=24)

fig.text(0.50, -0.04,
         f"5-fold stratified CV  |  n = 6,200 spectra  |  "
         f"{OUR_MODEL}: Ensemble (Fusion LR + ResNet18, α=0.8)",
         ha="center", va="bottom",
         fontsize=FONT_SIZE["annotation"] + 0.5, color="#888888", style="italic")

save_figure(fig, "fig_stage2_radar")
plt.close()

# ── Summary ──
print(f"\n=== Stage 2 F1 per Cancer Type ===")
print(f"{'':22}  {'  '.join(f'{d:>5}' for d in AACR_DISPLAY)}")
print("-" * 60)
for model in MODEL_ORDER:
    tag = " ***" if model == OUR_MODEL else ""
    vals = "  ".join(f"{all_f1[model][ci]:5.3f}" for ci in AACR_INDEX_ORDER)
    print(f"  {MODEL_SHORT[model]:<20}  {vals}{tag}")
