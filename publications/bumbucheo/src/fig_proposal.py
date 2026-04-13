"""
범부처 제안서 Figure 생성

1. Mean Spectra (7-Cancer + Non-Cancer overlay)
2. ROC Curves (S1 Screening + S2 Per-Cancer Type ID)
3. Confusion Matrices (S1 2×2 + S2 7×7)

Data: fold_predictions_7cancer.npz (1,575 subjects, uSERS-Net ensemble)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score, f1_score, confusion_matrix

from nature_style import (
    apply_style, apply_nature_style, add_panel_label, save_figure,
    CANCER_COLORS, AACR_LABELS, OUR_MODEL, NON_CANCER_COLOR,
    DOUBLE_COL, SINGLE_COL, FONT_SIZE, LINE_WIDTH,
    MODEL_GRID, FULL_GRID,
)

apply_style()

BUMBU_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURE_DIR = os.path.join(BUMBU_DIR, "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)

# ── Load data ──
NPZ_PATH = os.path.join(BUMBU_DIR, "data", "fold_predictions_7cancer.npz")
data = np.load(NPZ_PATH, allow_pickle=True)

X = data["X"]                    # (1575, 933)
bl = data["binary_labels"]       # 0=non-cancer, 1=cancer
ctl = data["cancer_type_labels"]
groups = data["groups"]
bp = data["val_binary_prob"]
cl = data["val_cancer_logits"]
cancer_types = list(data["cancer_types"])

n_ct = len(cancer_types)
print(f"Loaded: {len(X)} subjects, {n_ct} cancer types: {cancer_types}")
print(f"Cancer: {(bl==1).sum()}, Non-cancer: {(bl==0).sum()}")

# Display labels
DISPLAY_LABELS = {ct: AACR_LABELS.get(ct, ct) for ct in cancer_types}
DISPLAY_COLORS = {DISPLAY_LABELS[ct]: CANCER_COLORS.get(ct, "#888") for ct in cancer_types}

# ── Metrics ──
fpr_all, tpr_all, thresholds = roc_curve(bl, bp)
thr = thresholds[np.argmax(tpr_all - fpr_all)]
s1_auc = roc_auc_score(bl, bp)
s1_sens = (bp[bl == 1] > thr).mean()
s1_spec = (bp[bl == 0] <= thr).mean()

cancer_mask = bl == 1
pred_type = cl[cancer_mask].argmax(axis=1)
true_type = ctl[cancer_mask]
s2_f1 = f1_score(true_type, pred_type, average="macro", zero_division=0)

print(f"\nS1 AUC: {s1_auc:.4f}, Sens: {s1_sens:.1%}, Spec: {s1_spec:.1%}")
print(f"S2 F1 macro: {s2_f1:.4f}")

# Wavenumbers for spectra (935 full grid)
wavenumbers = FULL_GRID  # 935 points


# ═══════════════════════════════════════════
# Figure 1: Mean Spectra (7-Cancer + Non-Cancer)
# ═══════════════════════════════════════════
print("\n--- Generating Mean Spectra ---")

# Non-cancer display order: gray background
NON_CANCER_DISPLAY = "Non-Cancer"

# Compute per-group mean ± SEM
group_stats = {}

# Cancer groups
for ci, ct in enumerate(cancer_types):
    mask = (bl == 1) & (groups == ct)
    if mask.sum() == 0:
        # Try matching by cancer_type_labels
        mask = (ctl == ci) & (bl == 1)
    if mask.sum() > 0:
        vals = X[mask]
        display = DISPLAY_LABELS[ct]
        group_stats[display] = {
            "mean": vals.mean(axis=0),
            "sem": vals.std(axis=0) / np.sqrt(len(vals)),
            "n": mask.sum(),
            "color": CANCER_COLORS.get(ct, "#888"),
            "order": ci,
        }

# Non-cancer (combined)
nc_mask = bl == 0
if nc_mask.sum() > 0:
    vals = X[nc_mask]
    group_stats[NON_CANCER_DISPLAY] = {
        "mean": vals.mean(axis=0),
        "sem": vals.std(axis=0) / np.sqrt(len(vals)),
        "n": nc_mask.sum(),
        "color": NON_CANCER_COLOR,
        "order": -1,
    }

# Sort: Non-Cancer first, then cancer types in order
sorted_groups = sorted(group_stats.keys(),
                       key=lambda g: group_stats[g]["order"])

# Panel layout: overlay plot
fig, ax = plt.subplots(figsize=(DOUBLE_COL, DOUBLE_COL * 0.45), facecolor="none")

for grp in sorted_groups:
    s = group_stats[grp]
    lw = LINE_WIDTH["thin"] if grp == NON_CANCER_DISPLAY else LINE_WIDTH["spectrum"]
    alpha_line = 0.5 if grp == NON_CANCER_DISPLAY else 0.9
    zorder = 1 if grp == NON_CANCER_DISPLAY else 2

    ax.plot(wavenumbers, s["mean"], color=s["color"], lw=lw,
            alpha=alpha_line, label=f'{grp} (n={s["n"]})', zorder=zorder)
    ax.fill_between(wavenumbers,
                    s["mean"] - s["sem"], s["mean"] + s["sem"],
                    color=s["color"], alpha=0.10, zorder=zorder - 0.5)

ax.set_xlabel("Raman Shift (cm⁻¹)", fontsize=FONT_SIZE["axis_label"])
ax.set_ylabel("Intensity (a.u.)", fontsize=FONT_SIZE["axis_label"])
ax.set_title("SERS Spectra: 7-Cancer + Non-Cancer Control",
             fontsize=FONT_SIZE["title"] + 1, fontweight="bold", color="#2A2A2A", pad=8)

ax.set_xlim(wavenumbers[0], wavenumbers[-1])

# Legend: 2 columns, outside right
leg = ax.legend(fontsize=FONT_SIZE["legend"], loc="upper right",
                frameon=True, framealpha=0.92, edgecolor="#DDDDDD",
                ncol=2, handlelength=1.4)
for text in leg.get_texts():
    # Color-code legend text
    label = text.get_text().split(" (")[0]
    if label in DISPLAY_COLORS:
        text.set_color(DISPLAY_COLORS[label])
    elif label == NON_CANCER_DISPLAY:
        text.set_color(NON_CANCER_COLOR)

apply_nature_style(ax)
ax.grid(True, alpha=0.06, zorder=0)

plt.tight_layout()
save_figure(fig, "fig_spectra_7c", formats=("png",))
plt.close()


# ═══════════════════════════════════════════
# Figure 1b: Vertical Stacked Spectra Panels (per cancer vs Non-Cancer)
# ═══════════════════════════════════════════
print("\n--- Generating Vertical Spectra Panels ---")

# Full cancer names for left labels
CANCER_FULL_NAMES = {
    "PRO": "Prostate\nCancer",
    "BRE": "Breast\nCancer",
    "OVA": "Ovarian\nCancer",
    "LUN": "Lung\nCancer",
    "CRC": "Colorectal\nCancer",
    "PAN": "Pancreatic\nCancer",
    "BLC": "Bladder\nCancer",
}

nc_stats = group_stats[NON_CANCER_DISPLAY]
n_panels = n_ct  # 7

fig, axes = plt.subplots(n_panels, 1,
                         figsize=(DOUBLE_COL, DOUBLE_COL * 1.15),
                         facecolor="none", sharex=True, sharey=True)

for ci, ct in enumerate(cancer_types):
    ax = axes[ci]
    display = DISPLAY_LABELS[ct]
    s = group_stats[display]

    # Non-cancer background
    ax.plot(wavenumbers, nc_stats["mean"], color=NON_CANCER_COLOR,
            lw=LINE_WIDTH["thin"], alpha=0.45)
    ax.fill_between(wavenumbers,
                    nc_stats["mean"] - nc_stats["sem"],
                    nc_stats["mean"] + nc_stats["sem"],
                    color=NON_CANCER_COLOR, alpha=0.06)

    # Cancer spectrum
    ax.plot(wavenumbers, s["mean"], color=s["color"],
            lw=LINE_WIDTH["spectrum"] + 0.3, alpha=0.9)
    ax.fill_between(wavenumbers,
                    s["mean"] - s["sem"], s["mean"] + s["sem"],
                    color=s["color"], alpha=0.15)

    ax.set_xlim(wavenumbers[0], wavenumbers[-1])

    # Left: cancer name label (outside y-axis)
    ax.set_ylabel(CANCER_FULL_NAMES.get(ct, ct),
                  fontsize=FONT_SIZE["axis_label"] + 0.5,
                  fontweight="bold", color=s["color"],
                  rotation=0, labelpad=52, va="center", ha="center")

    # Right: sample count annotation
    ax.text(1.01, 0.5, f'n={s["n"]}',
            transform=ax.transAxes, fontsize=FONT_SIZE["annotation"],
            color=s["color"], va="center", ha="left", alpha=0.7)

    # Remove x-axis labels except bottom
    if ci < n_panels - 1:
        ax.tick_params(axis="x", labelbottom=False)

    # Clean style
    apply_nature_style(ax)
    ax.tick_params(axis="y", labelsize=FONT_SIZE["tick"] - 0.5)
    ax.grid(True, alpha=0.05, zorder=0)

# Bottom x-axis label
axes[-1].set_xlabel("Raman Shift (cm⁻¹)", fontsize=FONT_SIZE["axis_label"] + 0.5)

# Shared legend at top
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color=NON_CANCER_COLOR, lw=1.2, alpha=0.6,
           label=f'Non-Cancer (n={nc_stats["n"]})'),
    Line2D([0], [0], color="#C0392B", lw=1.5,
           label="Cancer (per type)"),
]
fig.legend(handles=legend_elements, loc="upper center",
           ncol=2, fontsize=FONT_SIZE["legend"] + 0.5,
           frameon=True, framealpha=0.92, edgecolor="#DDD",
           bbox_to_anchor=(0.55, 0.995))

fig.suptitle("SERS Mean Spectra by Cancer Type",
             fontsize=FONT_SIZE["title"] + 2, fontweight="bold",
             color="#2A2A2A", y=1.01)

plt.subplots_adjust(hspace=0.08, left=0.15, right=0.95, top=0.96, bottom=0.04)
save_figure(fig, "fig_spectra_panels_7c", formats=("png",))
plt.close()


# ═══════════════════════════════════════════
# Figure 2: ROC Curves
# ═══════════════════════════════════════════
print("\n--- Generating ROC Curves ---")

OUR_ACCENT = "#C0392B"

fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.48), facecolor="none")
ax1 = fig.add_axes([0.09, 0.13, 0.37, 0.76])
ax2 = fig.add_axes([0.58, 0.13, 0.39, 0.76])

# Panel (a): S1 Screening ROC
fpr_s1, tpr_s1, _ = roc_curve(bl, bp)
auc_s1 = auc(fpr_s1, tpr_s1)

ax1.plot(fpr_s1, tpr_s1, color=OUR_ACCENT, lw=1.8,
         label=f"{OUR_MODEL}  (AUC = {auc_s1:.3f})")
ax1.plot([0, 1], [0, 1], color="#CCCCCC", lw=0.6, ls="--", zorder=0)
ax1.set_xlim(-0.02, 1.02)
ax1.set_ylim(-0.02, 1.02)
ax1.set_xlabel("1 – Specificity (FPR)", fontsize=FONT_SIZE["axis_label"])
ax1.set_ylabel("Sensitivity (TPR)", fontsize=FONT_SIZE["axis_label"])
ax1.set_title("Cancer Screening (7-Cancer + Control)",
              fontsize=FONT_SIZE["title"] + 1, fontweight="bold", color="#2A2A2A")

# Operating point marker
best_idx = np.argmax(tpr_all - fpr_all)
ax1.plot(fpr_all[best_idx], tpr_all[best_idx], 'o', color=OUR_ACCENT,
         markersize=5, zorder=5)
ax1.annotate(f"Sens={s1_sens:.1%}\nSpec={s1_spec:.1%}",
             xy=(fpr_all[best_idx], tpr_all[best_idx]),
             xytext=(fpr_all[best_idx] + 0.12, tpr_all[best_idx] - 0.10),
             fontsize=FONT_SIZE["annotation"] + 0.5, color=OUR_ACCENT,
             arrowprops=dict(arrowstyle="-", color=OUR_ACCENT, lw=0.5))

legend1 = ax1.legend(fontsize=FONT_SIZE["legend"],
                     loc="lower right", frameon=True,
                     framealpha=0.92, edgecolor="#DDDDDD")
for text in legend1.get_texts():
    text.set_fontweight("bold")
    text.set_color(OUR_ACCENT)

apply_nature_style(ax1)
add_panel_label(ax1, "a", x=-0.15, y=1.08)

# Panel (b): S2 Per-cancer ROC
per_cancer_auc = {}
for ci, ct in enumerate(cancer_types):
    display = DISPLAY_LABELS[ct]
    color = DISPLAY_COLORS[display]
    y_true = (true_type == ci).astype(int)
    y_score = cl[cancer_mask][:, ci]
    try:
        fpr_c, tpr_c, _ = roc_curve(y_true, y_score)
        auc_c = auc(fpr_c, tpr_c)
        per_cancer_auc[ct] = auc_c
        ax2.plot(fpr_c, tpr_c, color=color, lw=1.2,
                 label=f"{display}  (AUC = {auc_c:.3f})")
    except ValueError:
        pass

ax2.plot([0, 1], [0, 1], color="#CCCCCC", lw=0.6, ls="--", zorder=0)
ax2.set_xlim(-0.02, 1.02)
ax2.set_ylim(-0.02, 1.02)
ax2.set_xlabel("1 – Specificity (FPR)", fontsize=FONT_SIZE["axis_label"])
ax2.set_ylabel("Sensitivity (TPR)", fontsize=FONT_SIZE["axis_label"])
ax2.set_title(f"Cancer Type Identification (One-vs-Rest)",
              fontsize=FONT_SIZE["title"] + 1, fontweight="bold", color="#2A2A2A")

legend2 = ax2.legend(fontsize=FONT_SIZE["legend"] - 0.5,
                     loc="lower right", frameon=True,
                     framealpha=0.92, edgecolor="#DDDDDD",
                     handlelength=1.4)
for text, ct in zip(legend2.get_texts(), cancer_types):
    display = DISPLAY_LABELS[ct]
    text.set_color(DISPLAY_COLORS[display])
    text.set_fontweight("bold")

apply_nature_style(ax2)
add_panel_label(ax2, "b", x=-0.15, y=1.08)

fig.text(0.50, 0.01,
         f"n = {len(X):,} subjects  |  "
         f"{OUR_MODEL}: Ensemble (Fusion LR + ResNet18)",
         ha="center", va="bottom",
         fontsize=FONT_SIZE["annotation"] + 0.5, color="#888888", style="italic")

save_figure(fig, "fig_roc_proposal_7c", formats=("png",))
plt.close()


# ═══════════════════════════════════════════
# Figure 3: Confusion Matrices
# ═══════════════════════════════════════════
print("\n--- Generating Confusion Matrices ---")

pred_binary = (bp > thr).astype(int)
cm1 = confusion_matrix(bl, pred_binary, labels=[0, 1])
cm2 = confusion_matrix(true_type, pred_type, labels=list(range(n_ct)))

cm1_pct = cm1 / cm1.sum(axis=1, keepdims=True) * 100
cm2_pct = cm2 / cm2.sum(axis=1, keepdims=True) * 100

S1_LABELS = ["Non-Cancer", "Cancer"]
S2_LABELS = [DISPLAY_LABELS[ct] for ct in cancer_types]
S1_DIAG = ["#6B8EAD", "#3D6E99"]
S2_DIAG = [CANCER_COLORS.get(ct, "#888888") for ct in cancer_types]


def draw_cm(ax, pct, labels, diag_colors, fs_main, fs_sub):
    n = len(labels)
    for i in range(n):
        for j in range(n):
            p = pct[i, j]
            if i == j:
                fc = diag_colors[i]
                tc = "white"
            else:
                intensity = min(p / 25.0, 1.0)
                g = 0.97 - intensity * 0.22
                fc = (g, g, g)
                tc = "#444444" if p > 1.0 else "#bbbbbb"

            rect = plt.Rectangle((j, n - 1 - i), 1, 1,
                                  facecolor=fc, edgecolor="white", linewidth=1.5)
            ax.add_patch(rect)

            # Percentage only
            pct_txt = f"{p:.1f}%" if p >= 0.5 else ("0%" if p == 0 else f"{p:.1f}%")
            ax.text(j + 0.5, n - 1 - i + 0.5, pct_txt,
                    ha="center", va="center",
                    fontsize=fs_main, fontweight="bold", color=tc)

    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect("equal")
    ax.set_xticks([i + 0.5 for i in range(n)])
    ax.set_xticklabels(labels, fontsize=fs_sub,
                       rotation=45 if n > 4 else 0,
                       ha="center" if n <= 4 else "right")
    ax.set_yticks([i + 0.5 for i in range(n)])
    ax.set_yticklabels(reversed(labels), fontsize=fs_sub)
    ax.set_xlabel("Predicted", fontsize=FONT_SIZE["axis_label"], labelpad=6)
    ax.set_ylabel("Actual", fontsize=FONT_SIZE["axis_label"], labelpad=6)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)


fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.50), facecolor="none")
gs = fig.add_gridspec(1, 2, width_ratios=[1, 3], wspace=0.45,
                      left=0.06, right=0.97, top=0.85, bottom=0.18)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

# Panel A: Cancer Screening
draw_cm(ax1, cm1_pct, S1_LABELS, S1_DIAG,
        fs_main=FONT_SIZE["title"], fs_sub=FONT_SIZE["axis_label"])
ax1.set_title("Cancer Screening", fontsize=FONT_SIZE["title"],
              fontweight="bold", pad=8)
ax1.text(0.5, -0.28,
         f"AUC = {s1_auc:.3f}   Sens = {s1_sens:.1%}   Spec = {s1_spec:.1%}",
         transform=ax1.transAxes, ha="center",
         fontsize=FONT_SIZE["annotation"], color="#666666")
add_panel_label(ax1, "a", x=-0.20, y=1.08)

# Panel B: Cancer Type ID
draw_cm(ax2, cm2_pct, S2_LABELS, S2_DIAG,
        fs_main=FONT_SIZE["axis_label"] - 0.5, fs_sub=FONT_SIZE["axis_label"] - 0.5)
ax2.set_title("Cancer Type Identification (7-Cancer)",
              fontsize=FONT_SIZE["title"], fontweight="bold", pad=8)
ax2.text(0.5, -0.28, f"Macro F1 = {s2_f1:.3f}",
         transform=ax2.transAxes, ha="center",
         fontsize=FONT_SIZE["annotation"], color="#666666")
add_panel_label(ax2, "b", x=-0.10, y=1.08)

save_figure(fig, "fig_cm_proposal_7c", formats=("png",))
plt.close()


# ═══════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════
print(f"\n{'='*64}")
print(f"  범부처 제안서 Figure Summary")
print(f"{'='*64}")
print(f"  Subjects:       {len(X):,}")
print(f"  Cancer:         {(bl==1).sum():,} ({n_ct} types)")
print(f"  Non-Cancer:     {(bl==0).sum():,}")
print(f"  S1 AUC:         {s1_auc:.4f}")
print(f"  S1 Sensitivity: {s1_sens:.1%}")
print(f"  S1 Specificity: {s1_spec:.1%}")
print(f"  S2 F1 macro:    {s2_f1:.4f}")
print(f"\n  Per-cancer Type ID AUC:")
for ct in cancer_types:
    display = DISPLAY_LABELS[ct]
    a = per_cancer_auc.get(ct, float('nan'))
    print(f"    {display}: {a:.4f}")
print(f"\n  Generated figures:")
for name in ["fig_spectra_7c", "fig_spectra_panels_7c",
             "fig_roc_proposal_7c", "fig_cm_proposal_7c"]:
    print(f"    publications/bumbucheo/figures/{name}.png")
print(f"{'='*64}")
