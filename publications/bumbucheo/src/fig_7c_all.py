"""
7-Cancer uSERS-Net Figures for 범부처 PT

Generates:
  1. ROC Curves (S1 Screening + S2 Per-cancer)
  2. Confusion Matrix (S1 2x2 + S2 7x7)
  3. PCA/t-SNE distribution scatter
  4. Per-cancer sensitivity bar chart

Uses fold_predictions_7cancer.npz from gen_fusion_predictions_7c.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_curve, auc, roc_auc_score, f1_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from nature_style import (
    apply_style, apply_nature_style, add_panel_label, save_figure,
    CANCER_COLORS, AACR_LABELS, OUR_MODEL,
    DOUBLE_COL, SINGLE_COL, FONT_SIZE, LINE_WIDTH, NON_CANCER_COLOR,
)

apply_style()

BUMBU_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURE_DIR = os.path.join(BUMBU_DIR, "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)

# ── Load 7-cancer predictions ──
NPZ_PATH = os.path.join(BUMBU_DIR, "data", "fold_predictions_7cancer.npz")
data = np.load(NPZ_PATH, allow_pickle=True)

X = data["X"]
bl = data["binary_labels"]
ctl = data["cancer_type_labels"]
groups = data["groups"]
bp = data["val_binary_prob"]
cl = data["val_cancer_logits"]
cancer_types = list(data["cancer_types"])

n_ct = len(cancer_types)
print(f"Loaded: {len(X)} spectra, {n_ct} cancer types: {cancer_types}")
print(f"Cancer: {(bl==1).sum()}, Non-cancer: {(bl==0).sum()}")

# ── Threshold (Youden) ──
fpr_all, tpr_all, thresholds = roc_curve(bl, bp)
thr = thresholds[np.argmax(tpr_all - fpr_all)]
s1_auc = roc_auc_score(bl, bp)

cancer_mask = bl == 1
pred_type = cl[cancer_mask].argmax(axis=1)
true_type = ctl[cancer_mask]
s2_f1 = f1_score(true_type, pred_type, average="macro", zero_division=0)

s1_sens = (bp[bl == 1] > thr).mean()
s1_spec = (bp[bl == 0] <= thr).mean()

print(f"\nS1 AUC: {s1_auc:.4f}")
print(f"S1 Sensitivity: {s1_sens:.4f}")
print(f"S1 Specificity: {s1_spec:.4f}")
print(f"S2 F1 macro: {s2_f1:.4f}")

# Display labels and colors
DISPLAY_LABELS = {ct: AACR_LABELS.get(ct, ct) for ct in cancer_types}
DISPLAY_COLORS = {}
for ct in cancer_types:
    display = DISPLAY_LABELS[ct]
    DISPLAY_COLORS[display] = CANCER_COLORS.get(ct, "#888888")

# Display order for 7 cancers
DISPLAY_ORDER = [DISPLAY_LABELS[ct] for ct in cancer_types]

# ═══════════════════════════════════════════
# Figure 1: ROC Curves
# ═══════════════════════════════════════════
print("\n--- Generating ROC Curves ---")

fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.48), facecolor="none")
ax1 = fig.add_axes([0.09, 0.13, 0.37, 0.76])
ax2 = fig.add_axes([0.58, 0.13, 0.39, 0.76])

OUR_ACCENT = "#C0392B"

# Panel (a): S1 Screening ROC
fpr_s1, tpr_s1, _ = roc_curve(bl, bp)
auc_s1 = auc(fpr_s1, tpr_s1)

ax1.plot(fpr_s1, tpr_s1, color=OUR_ACCENT, lw=1.8,
         label=f"{OUR_MODEL}  (AUC={auc_s1:.3f})")
ax1.plot([0, 1], [0, 1], color="#CCCCCC", lw=0.6, ls="--", zorder=0)
ax1.set_xlim(-0.02, 1.02)
ax1.set_ylim(-0.02, 1.02)
ax1.set_xlabel("1 – Specificity (FPR)", fontsize=FONT_SIZE["axis_label"])
ax1.set_ylabel("Sensitivity (TPR)", fontsize=FONT_SIZE["axis_label"])
ax1.set_title("Cancer Screening (7-Cancer)", fontsize=FONT_SIZE["title"] + 1,
              fontweight="bold", color="#2A2A2A")

legend1 = ax1.legend(fontsize=FONT_SIZE["legend"],
                     loc="lower right", frameon=True,
                     framealpha=0.92, edgecolor="#DDDDDD")
for text in legend1.get_texts():
    text.set_fontweight("bold")
    text.set_color(OUR_ACCENT)

apply_nature_style(ax1)
add_panel_label(ax1, "a", x=-0.15, y=1.08)

# Panel (b): S2 Per-cancer ROC
for ci, ct in enumerate(cancer_types):
    display = DISPLAY_LABELS[ct]
    color = DISPLAY_COLORS[display]

    y_true = (true_type == ci).astype(int)
    y_score = cl[cancer_mask][:, ci]

    try:
        fpr_c, tpr_c, _ = roc_curve(y_true, y_score)
        auc_c = auc(fpr_c, tpr_c)
        ax2.plot(fpr_c, tpr_c, color=color, lw=1.2,
                 label=f"{display}  (AUC={auc_c:.3f})")
    except ValueError:
        pass

ax2.plot([0, 1], [0, 1], color="#CCCCCC", lw=0.6, ls="--", zorder=0)
ax2.set_xlim(-0.02, 1.02)
ax2.set_ylim(-0.02, 1.02)
ax2.set_xlabel("1 – Specificity (FPR)", fontsize=FONT_SIZE["axis_label"])
ax2.set_ylabel("Sensitivity (TPR)", fontsize=FONT_SIZE["axis_label"])
ax2.set_title(f"{OUR_MODEL}: Cancer Type ID (7-Cancer)",
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
         f"5-fold stratified CV  |  n = {len(X):,} spectra  |  "
         f"{OUR_MODEL}: Ensemble (Fusion LR + ResNet18, α=0.8)",
         ha="center", va="bottom",
         fontsize=FONT_SIZE["annotation"] + 0.5, color="#888888", style="italic")

save_figure(fig, "fig_roc_curves_7c", formats=("png",))
plt.close()


# ═══════════════════════════════════════════
# Figure 2: Confusion Matrices
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

            txt = f"{p:.1f}%" if p >= 0.5 else ("0%" if p == 0 else f"{p:.1f}%")
            ax.text(j + 0.5, n - 1 - i + 0.5, txt,
                    ha="center", va="center",
                    fontsize=fs_main, fontweight="bold", color=tc)

    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect("equal")
    ax.set_xticks([i + 0.5 for i in range(n)])
    ax.set_xticklabels(labels, fontsize=fs_sub, rotation=45 if n > 4 else 0, ha="center" if n <= 4 else "right")
    ax.set_yticks([i + 0.5 for i in range(n)])
    ax.set_yticklabels(reversed(labels), fontsize=fs_sub)
    ax.set_xlabel("Predicted", fontsize=FONT_SIZE["axis_label"], labelpad=6)
    ax.set_ylabel("Actual", fontsize=FONT_SIZE["axis_label"], labelpad=6)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)


fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.48), facecolor="none")
gs = fig.add_gridspec(1, 2, width_ratios=[1, 3], wspace=0.45,
                      left=0.06, right=0.97, top=0.88, bottom=0.18)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

# Panel A: Cancer Screening
draw_cm(ax1, cm1_pct, S1_LABELS, S1_DIAG,
        fs_main=FONT_SIZE["title"], fs_sub=FONT_SIZE["axis_label"])
ax1.set_title("Cancer Screening", fontsize=FONT_SIZE["title"],
              fontweight="bold", pad=8)
ax1.text(0.5, -0.28, f"AUC = {s1_auc:.3f}   Sens = {s1_sens:.1%}   Spec = {s1_spec:.1%}",
         transform=ax1.transAxes, ha="center",
         fontsize=FONT_SIZE["annotation"], color="#666666")

add_panel_label(ax1, "a", x=-0.20, y=1.08)

# Panel B: Cancer Type ID
draw_cm(ax2, cm2_pct, S2_LABELS, S2_DIAG,
        fs_main=FONT_SIZE["axis_label"] - 0.5, fs_sub=FONT_SIZE["axis_label"] - 0.5)
ax2.set_title("Cancer Type Identification (7-Cancer)", fontsize=FONT_SIZE["title"],
              fontweight="bold", pad=8)
ax2.text(0.5, -0.28, f"Macro F1 = {s2_f1:.3f}",
         transform=ax2.transAxes, ha="center",
         fontsize=FONT_SIZE["annotation"], color="#666666")

add_panel_label(ax2, "b", x=-0.10, y=1.08)

save_figure(fig, "fig_confusion_matrices_7c", formats=("png",))
plt.close()


# ═══════════════════════════════════════════
# Figure 3: PCA Distribution
# ═══════════════════════════════════════════
print("\n--- Generating PCA Distribution ---")

# Use medoid (one per sample) for cleaner visualization
sample_key = np.array([f"{g}_{s}" for g, s in zip(groups, data["sample_ids"])])
unique_keys, first_idx = np.unique(sample_key, return_index=True)
X_unique = X[first_idx]
groups_unique = groups[first_idx]
bl_unique = bl[first_idx]

# PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_unique)
var_ratio = pca.explained_variance_ratio_

# Assign display labels
display_groups = []
for g in groups_unique:
    if g in DISPLAY_LABELS:
        display_groups.append(DISPLAY_LABELS[g])
    else:
        display_groups.append("Non-Cancer")
display_groups = np.array(display_groups)

fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.48), facecolor="none")

# Panel (a): Cancer vs Non-Cancer
ax = axes[0]
nc_mask = bl_unique == 0
c_mask = bl_unique == 1
ax.scatter(X_pca[nc_mask, 0], X_pca[nc_mask, 1], s=8, alpha=0.3,
           color=NON_CANCER_COLOR, label=f"Non-Cancer (n={nc_mask.sum()})", zorder=1)
ax.scatter(X_pca[c_mask, 0], X_pca[c_mask, 1], s=8, alpha=0.3,
           color="#C0392B", label=f"Cancer (n={c_mask.sum()})", zorder=2)
ax.set_xlabel(f"PC1 ({var_ratio[0]:.1%})", fontsize=FONT_SIZE["axis_label"])
ax.set_ylabel(f"PC2 ({var_ratio[1]:.1%})", fontsize=FONT_SIZE["axis_label"])
ax.set_title("Cancer vs Non-Cancer", fontsize=FONT_SIZE["title"] + 1,
             fontweight="bold", color="#2A2A2A")
ax.legend(fontsize=FONT_SIZE["legend"], loc="best", frameon=True,
          framealpha=0.92, edgecolor="#DDDDDD", markerscale=2)
apply_nature_style(ax)
add_panel_label(ax, "a", x=-0.15, y=1.08)

# Panel (b): Per-cancer type
ax = axes[1]
# Plot non-cancer first (background)
nc_mask_disp = display_groups == "Non-Cancer"
ax.scatter(X_pca[nc_mask_disp, 0], X_pca[nc_mask_disp, 1], s=6, alpha=0.15,
           color=NON_CANCER_COLOR, label=f"Non-Cancer", zorder=1)
# Plot each cancer type
for ct in cancer_types:
    display = DISPLAY_LABELS[ct]
    mask = display_groups == display
    if mask.sum() > 0:
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], s=10, alpha=0.5,
                   color=DISPLAY_COLORS[display],
                   label=f"{display} (n={mask.sum()})", zorder=2)

ax.set_xlabel(f"PC1 ({var_ratio[0]:.1%})", fontsize=FONT_SIZE["axis_label"])
ax.set_ylabel(f"PC2 ({var_ratio[1]:.1%})", fontsize=FONT_SIZE["axis_label"])
ax.set_title("Per-Cancer Distribution", fontsize=FONT_SIZE["title"] + 1,
             fontweight="bold", color="#2A2A2A")
ax.legend(fontsize=FONT_SIZE["legend"] - 0.5, loc="best", frameon=True,
          framealpha=0.92, edgecolor="#DDDDDD", markerscale=2,
          ncol=2)
apply_nature_style(ax)
add_panel_label(ax, "b", x=-0.15, y=1.08)

plt.tight_layout()
save_figure(fig, "fig_pca_distribution_7c", formats=("png",))
plt.close()


# ═══════════════════════════════════════════
# Figure 4: Per-cancer Sensitivity Bar Chart
# ═══════════════════════════════════════════
print("\n--- Generating Sensitivity Bar Chart ---")

per_cancer_sens = {}
per_cancer_n = {}
for ci, ct in enumerate(cancer_types):
    mask = groups == ct
    if mask.sum() > 0:
        per_cancer_sens[ct] = (bp[mask] > thr).mean()
        per_cancer_n[ct] = mask.sum()

fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.65, DOUBLE_COL * 0.45), facecolor="none")

y_pos = np.arange(len(cancer_types))
bars_sens = [per_cancer_sens.get(ct, 0) for ct in cancer_types]
bars_labels = [DISPLAY_LABELS[ct] for ct in cancer_types]
bars_colors = [CANCER_COLORS.get(ct, "#888888") for ct in cancer_types]

bars = ax.barh(y_pos, bars_sens, color=bars_colors, edgecolor="white",
               height=0.6, alpha=0.85)

for i, (bar, ct) in enumerate(zip(bars, cancer_types)):
    w = bar.get_width()
    n = per_cancer_n.get(ct, 0)
    ax.text(w + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{w:.1%} (n={n})",
            va="center", ha="left",
            fontsize=FONT_SIZE["annotation"] + 0.5, color="#444444")

ax.set_yticks(y_pos)
ax.set_yticklabels(bars_labels, fontsize=FONT_SIZE["axis_label"])
ax.set_xlim(0, 1.15)
ax.set_xlabel("Detection Sensitivity", fontsize=FONT_SIZE["axis_label"])
ax.set_title(f"{OUR_MODEL}: Per-Cancer Detection Sensitivity",
             fontsize=FONT_SIZE["title"] + 1, fontweight="bold", color="#2A2A2A", pad=10)

# Add threshold line annotation
ax.axvline(x=s1_sens, color="#C0392B", ls="--", lw=0.8, alpha=0.5)
ax.text(s1_sens + 0.01, len(cancer_types) - 0.3,
        f"Overall: {s1_sens:.1%}",
        fontsize=FONT_SIZE["annotation"], color="#C0392B", alpha=0.7)

apply_nature_style(ax)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)

plt.tight_layout()
save_figure(fig, "fig_sensitivity_bar_7c", formats=("png",))
plt.close()


# ═══════════════════════════════════════════
# Figure 5: 2D Model Classification Space
# ═══════════════════════════════════════════
print("\n--- Generating 2D Model Classification Space ---")

from matplotlib.patches import Ellipse

# Medoid-level predictions
bp_unique = bp[first_idx]
cl_unique = cl[first_idx]

# PCA on model's 7-dim cancer type logits → 2D
pca2d = PCA(n_components=2, random_state=42)
logits_2d = pca2d.fit_transform(cl_unique)
var2d = pca2d.explained_variance_ratio_

fig, ax = plt.subplots(figsize=(DOUBLE_COL, DOUBLE_COL * 0.7), facecolor="none")

# Non-cancer background
nc_mask = display_groups == "Non-Cancer"
ax.scatter(logits_2d[nc_mask, 0], logits_2d[nc_mask, 1],
           s=12, alpha=0.12, color=NON_CANCER_COLOR, zorder=1,
           label=f"Non-Cancer (n={nc_mask.sum()})")

# Each cancer type with confidence ellipse
for ct in cancer_types:
    display = DISPLAY_LABELS[ct]
    color = DISPLAY_COLORS[display]
    mask = display_groups == display
    if mask.sum() < 3:
        continue

    pts = logits_2d[mask]
    ax.scatter(pts[:, 0], pts[:, 1],
               s=22, alpha=0.55, color=color, edgecolors="white", linewidths=0.3,
               label=f"{display} (n={mask.sum()})", zorder=3)

    # 95% confidence ellipse
    from numpy.linalg import eigh
    cov = np.cov(pts.T)
    vals, vecs = eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h = 2 * 1.96 * np.sqrt(vals)  # 95% CI
    ell = Ellipse(xy=pts.mean(axis=0), width=w, height=h, angle=angle,
                  facecolor=color, alpha=0.08, edgecolor=color, linewidth=1.2,
                  linestyle="--", zorder=2)
    ax.add_patch(ell)

    # Centroid label
    cx, cy = pts.mean(axis=0)
    ax.annotate(display, (cx, cy), fontsize=FONT_SIZE["title"] + 1,
                fontweight="bold", color=color, ha="center", va="center",
                zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor=color, alpha=0.85, linewidth=0.6))

ax.set_xlabel(f"Model PC1 ({var2d[0]:.1%} var)", fontsize=FONT_SIZE["axis_label"] + 1)
ax.set_ylabel(f"Model PC2 ({var2d[1]:.1%} var)", fontsize=FONT_SIZE["axis_label"] + 1)
ax.set_title(f"{OUR_MODEL}: Learned Classification Space",
             fontsize=FONT_SIZE["title"] + 2, fontweight="bold", color="#2A2A2A", pad=12)

ax.legend(fontsize=FONT_SIZE["legend"] + 0.5, loc="upper right", frameon=True,
          framealpha=0.92, edgecolor="#DDDDDD", markerscale=1.5, ncol=2)

apply_nature_style(ax)
ax.grid(True, alpha=0.08, zorder=0)

plt.tight_layout()
save_figure(fig, "fig_model_2d_7c", formats=("png",))
plt.close()


# ═══════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════
print(f"\n{'='*64}")
print(f"  7-Cancer uSERS-Net Figure Summary")
print(f"{'='*64}")
print(f"  S1 AUC:         {s1_auc:.4f}")
print(f"  S1 Sensitivity: {s1_sens:.4f}")
print(f"  S1 Specificity: {s1_spec:.4f}")
print(f"  S2 F1 macro:    {s2_f1:.4f}")
print(f"\n  Per-cancer S2 AUC:")
for ci, ct in enumerate(cancer_types):
    display = DISPLAY_LABELS[ct]
    y_t = (true_type == ci).astype(int)
    y_s = cl[cancer_mask][:, ci]
    try:
        a = roc_auc_score(y_t, y_s)
        print(f"    {display}: {a:.4f}")
    except Exception:
        print(f"    {display}: N/A")

print(f"\n  Per-cancer Detection Sensitivity:")
for ct in cancer_types:
    display = DISPLAY_LABELS[ct]
    sens = per_cancer_sens.get(ct, 0)
    n = per_cancer_n.get(ct, 0)
    print(f"    {display}: {sens:.4f} (n={n})")

print(f"\n  Generated figures:")
for name in ["fig_roc_curves_7c", "fig_confusion_matrices_7c",
             "fig_pca_distribution_7c", "fig_sensitivity_bar_7c",
             "fig_model_2d_7c"]:
    print(f"    publications/bumbucheo/figures/{name}.png")
print(f"{'='*64}")
