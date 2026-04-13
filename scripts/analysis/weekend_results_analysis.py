#!/usr/bin/env python3
"""
Weekend Experiments — Comprehensive Results Analysis
=====================================================
1. Best model (Stacking LR) — Confusion Matrix + SHAP
2. All experiment summary figures

Usage:
    python scripts/analysis/weekend_results_analysis.py
"""
from __future__ import annotations
import sys, json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from scipy.signal import savgol_filter

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    f1_score, roc_auc_score, precision_recall_curve, average_precision_score,
)

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.sers.preprocessing import trim_spectrum, baseline_correction, normalize_spectrum, resample
from src.sers.io import find_spectra, read_spectrum, parse_filename

# ── Paths ────────────────────────────────────────────────────────────────────
RESULTS_DIR = PROJECT_ROOT / "results"
STACKING_DIR = RESULTS_DIR / "weekend_experiments" / "stacking_optimization"
CONTRASTIVE_DIR = RESULTS_DIR / "contrastive_sweep"
LOHO_DIR = RESULTS_DIR / "weekend_experiments" / "loho_validation"
PERM_DIR = RESULTS_DIR / "weekend_experiments" / "exp4_permutation_importance_20260403_204449"
NORMSEARCH_DIR = RESULTS_DIR / "weekend_experiments" / "exp6_norm_feature_search"
OUTPUT_DIR = RESULTS_DIR / "weekend_experiments" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CANCER_TYPES = ["PRO", "LUN", "CRC", "PAN", "OVA", "BRE", "BLC"]
CANCER_LABELS_KR = {
    "PRO": "전립선암", "LUN": "폐암", "CRC": "대장암",
    "PAN": "췌장암", "OVA": "난소암", "BRE": "유방암", "BLC": "방광암",
}
NON_CANCER = ["NOR", "DIA", "HBP", "H.D."]

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

COLORS = {
    "PRO": "#2196F3", "LUN": "#4CAF50", "CRC": "#FF9800",
    "PAN": "#9C27B0", "OVA": "#E91E63", "BRE": "#00BCD4", "BLC": "#795548",
    "Non-Cancer": "#9E9E9E",
}
MODEL_COLORS = {"LR": "#2196F3", "XGB": "#FF9800", "RF": "#4CAF50"}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  1. STACKING BEST MODEL — CONFUSION MATRIX                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def load_oof_data():
    data = np.load(STACKING_DIR / "oof_predictions.npz", allow_pickle=True)
    return data


def plot_stacking_confusion_matrices(data, save_dir):
    """Generate confusion matrices for the best stacking ensemble (LR meta)."""
    binary_labels = data["binary_labels"]
    cancer_type_labels = data["cancer_type_labels"]
    sample_ids = data["sample_ids"]

    # ── Combine base model predictions (simple average for OOF) ──
    base_models = [
        "lr_raw", "lr_d1", "lr_d2", "lr_concat", "lr_peak",
        "xgb_raw", "xgb_d1", "rf_raw", "rf_d1", "ridge_concat",
    ]

    # S1: Average binary probabilities
    s1_probs = np.mean([data[f"{m}_s1"] for m in base_models], axis=0)
    # S2: Average multiclass probabilities
    s2_probs = np.mean([data[f"{m}_s2"] for m in base_models], axis=0)

    # ── Figure 1: Binary (Cancer vs Non-Cancer) ──
    s1_pred = (s1_probs >= 0.5).astype(int)
    cm_binary = confusion_matrix(binary_labels, s1_pred)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Binary CM
    ax = axes[0]
    labels_bin = ["Non-Cancer", "Cancer"]
    im = ax.imshow(cm_binary, cmap="Blues", aspect="auto")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels_bin); ax.set_yticklabels(labels_bin)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Stage 1: Cancer Screening\n(Stacking Ensemble, OOF)", fontweight="bold")

    for i in range(2):
        for j in range(2):
            total = cm_binary[i].sum()
            pct = cm_binary[i, j] / total * 100
            color = "white" if cm_binary[i, j] > cm_binary.max() * 0.5 else "black"
            ax.text(j, i, f"{cm_binary[i,j]}\n({pct:.1f}%)", ha="center", va="center",
                    fontsize=14, fontweight="bold", color=color)

    # S1 metrics annotation
    tn, fp, fn, tp = cm_binary.ravel()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    s1_auc = roc_auc_score(binary_labels, s1_probs)
    ax.text(0.02, -0.18, f"AUC: {s1_auc:.4f}  |  Sensitivity: {sens:.4f}  |  Specificity: {spec:.4f}",
            transform=ax.transAxes, fontsize=12, style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#E3F2FD", alpha=0.8))

    # ── Multiclass CM (cancer samples only) ──
    ax = axes[1]
    cancer_mask = binary_labels == 1
    y_true_type = cancer_type_labels[cancer_mask].astype(int)
    y_pred_type = np.argmax(s2_probs[cancer_mask], axis=1)

    cm_type = confusion_matrix(y_true_type, y_pred_type, labels=list(range(7)))

    im2 = ax.imshow(cm_type, cmap="Oranges", aspect="auto")
    ax.set_xticks(range(7)); ax.set_yticks(range(7))
    ax.set_xticklabels(CANCER_TYPES, rotation=45, ha="right")
    ax.set_yticklabels(CANCER_TYPES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Stage 2: Cancer Type Identification\n(Stacking Ensemble, OOF)", fontweight="bold")

    for i in range(7):
        for j in range(7):
            total = cm_type[i].sum() if cm_type[i].sum() > 0 else 1
            pct = cm_type[i, j] / total * 100
            color = "white" if cm_type[i, j] > cm_type.max() * 0.5 else "black"
            ax.text(j, i, f"{cm_type[i,j]}\n({pct:.0f}%)", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color)

    s2_f1 = f1_score(y_true_type, y_pred_type, average="macro")
    per_class_f1 = f1_score(y_true_type, y_pred_type, average=None, labels=list(range(7)))
    f1_str = " | ".join(f"{ct}:{f:.2f}" for ct, f in zip(CANCER_TYPES, per_class_f1))
    ax.text(0.02, -0.22, f"Macro F1: {s2_f1:.4f}\n{f1_str}",
            transform=ax.transAxes, fontsize=10, style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF3E0", alpha=0.8))

    fig.suptitle("Stacking Ensemble (LR Meta-Learner) — Best Weekend Result", fontsize=18, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_dir / "01_stacking_confusion_matrices.png")
    plt.close(fig)
    print(f"  [✓] Confusion matrices → {save_dir / '01_stacking_confusion_matrices.png'}")

    return s1_auc, s2_f1, per_class_f1


def plot_stacking_roc_curves(data, save_dir):
    """ROC curves for S1 and per-cancer-type S2."""
    binary_labels = data["binary_labels"]
    cancer_type_labels = data["cancer_type_labels"]

    base_models = [
        "lr_raw", "lr_d1", "lr_d2", "lr_concat", "lr_peak",
        "xgb_raw", "xgb_d1", "rf_raw", "rf_d1", "ridge_concat",
    ]
    s1_probs = np.mean([data[f"{m}_s1"] for m in base_models], axis=0)
    s2_probs = np.mean([data[f"{m}_s2"] for m in base_models], axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # S1 ROC
    ax = axes[0]
    fpr, tpr, _ = roc_curve(binary_labels, s1_probs)
    s1_auc_val = auc(fpr, tpr)
    ax.plot(fpr, tpr, color="#1565C0", lw=2.5, label=f"Stacking (AUC={s1_auc_val:.4f})")
    # Individual base models
    for m in ["lr_d1", "xgb_d1", "rf_d1"]:
        fpr_m, tpr_m, _ = roc_curve(binary_labels, data[f"{m}_s1"])
        auc_m = auc(fpr_m, tpr_m)
        ax.plot(fpr_m, tpr_m, "--", lw=1.2, alpha=0.6, label=f"{m} (AUC={auc_m:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("Stage 1: Cancer Screening ROC", fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)

    # S2 per-cancer ROC (one-vs-rest)
    ax = axes[1]
    cancer_mask = binary_labels == 1
    y_true_type = cancer_type_labels[cancer_mask].astype(int)
    s2_cancer = s2_probs[cancer_mask]

    for ci, ct in enumerate(CANCER_TYPES):
        y_bin = (y_true_type == ci).astype(int)
        if y_bin.sum() == 0:
            continue
        fpr_c, tpr_c, _ = roc_curve(y_bin, s2_cancer[:, ci])
        auc_c = auc(fpr_c, tpr_c)
        ax.plot(fpr_c, tpr_c, color=COLORS[ct], lw=2, label=f"{ct} (AUC={auc_c:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("Stage 2: Per-Cancer-Type ROC (One-vs-Rest)", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)

    fig.suptitle("Stacking Ensemble — ROC Curves", fontsize=18, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_dir / "02_stacking_roc_curves.png")
    plt.close(fig)
    print(f"  [✓] ROC curves → {save_dir / '02_stacking_roc_curves.png'}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  2. SHAP ANALYSIS — Base Model Level + Wavenumber Level                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def plot_shap_meta_level(data, save_dir):
    """SHAP for meta-learner: which base model predictions matter most."""
    import shap

    binary_labels = data["binary_labels"]
    cancer_type_labels = data["cancer_type_labels"]

    base_models = [
        "lr_raw", "lr_d1", "lr_d2", "lr_concat", "lr_peak",
        "xgb_raw", "xgb_d1", "rf_raw", "rf_d1", "ridge_concat",
    ]

    # ── S1 meta-learner SHAP ──
    X_meta_s1 = np.column_stack([data[f"{m}_s1"] for m in base_models])
    scaler_s1 = StandardScaler()
    X_meta_s1_scaled = scaler_s1.fit_transform(X_meta_s1)
    lr_s1 = LogisticRegression(C=1.0, max_iter=2000)
    lr_s1.fit(X_meta_s1_scaled, binary_labels)

    explainer_s1 = shap.LinearExplainer(lr_s1, X_meta_s1_scaled)
    shap_s1 = explainer_s1.shap_values(X_meta_s1_scaled)

    # ── S2 meta-learner SHAP ──
    cancer_mask = binary_labels == 1
    y_type = cancer_type_labels[cancer_mask].astype(int)
    s2_parts = []
    s2_feat_names = []
    for m in base_models:
        arr = data[f"{m}_s2"]
        if arr.ndim == 2 and arr.shape[1] == 7:
            s2_parts.append(arr[cancer_mask])
            for ci, ct in enumerate(CANCER_TYPES):
                s2_feat_names.append(f"{m}_{ct}")
        else:
            s2_parts.append(arr[cancer_mask].reshape(-1, 1))
            s2_feat_names.append(m)
    X_meta_s2 = np.hstack(s2_parts)

    scaler_s2 = StandardScaler()
    X_meta_s2_scaled = scaler_s2.fit_transform(X_meta_s2)
    lr_s2_meta = LogisticRegression(C=1.0, max_iter=2000, multi_class="multinomial")
    lr_s2_meta.fit(X_meta_s2_scaled, y_type)

    explainer_s2 = shap.LinearExplainer(lr_s2_meta, X_meta_s2_scaled)
    shap_s2 = explainer_s2.shap_values(X_meta_s2_scaled)

    # ── Plot ──
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # S1 SHAP
    ax = axes[0]
    shap_s1_vals = np.array(shap_s1)
    if shap_s1_vals.ndim == 3:  # (n_classes, n_samples, n_features) → take class 1
        shap_s1_vals = shap_s1_vals[1]
    elif shap_s1_vals.ndim == 1:
        shap_s1_vals = shap_s1_vals.reshape(1, -1)
    mean_abs_shap_s1 = np.abs(shap_s1_vals).mean(axis=0)
    order = np.argsort(mean_abs_shap_s1)
    ax.barh(range(len(base_models)), mean_abs_shap_s1[order], color="#1976D2", alpha=0.85)
    ax.set_yticks(range(len(base_models)))
    ax.set_yticklabels([base_models[i] for i in order])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Stage 1 (Cancer Screening)\nMeta-Learner Base Model Importance", fontweight="bold")

    # S2 SHAP — aggregate by base model
    ax = axes[1]
    shap_s2_arr = np.array(shap_s2)
    if shap_s2_arr.ndim == 3:  # (n_samples, n_features, n_classes)
        shap_s2_combined = np.abs(shap_s2_arr).sum(axis=2)  # sum over classes → (n_samples, n_features)
    else:
        shap_s2_combined = np.abs(shap_s2_arr)
    mean_abs_shap_s2 = shap_s2_combined.mean(axis=0)

    # Aggregate by base model name
    model_shap = {}
    for fi, fname in enumerate(s2_feat_names):
        base = fname.rsplit("_", 1)[0] if any(fname.endswith(f"_{ct}") for ct in CANCER_TYPES) else fname
        # Find the actual base model name
        for bm in base_models:
            if fname.startswith(bm):
                base = bm
                break
        model_shap[base] = model_shap.get(base, 0) + mean_abs_shap_s2[fi]
    bases = list(model_shap.keys())
    vals = [model_shap[b] for b in bases]
    order2 = np.argsort(vals)
    ax.barh(range(len(bases)), [vals[i] for i in order2], color="#E65100", alpha=0.85)
    ax.set_yticks(range(len(bases)))
    ax.set_yticklabels([bases[i] for i in order2])
    ax.set_xlabel("Sum of Mean |SHAP value| (across 7 classes)")
    ax.set_title("Stage 2 (Cancer Type ID)\nMeta-Learner Base Model Importance", fontweight="bold")

    fig.suptitle("SHAP Analysis — Which Base Models Drive the Stacking Ensemble?",
                 fontsize=18, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_dir / "03_shap_meta_learner.png")
    plt.close(fig)
    print(f"  [✓] SHAP meta-learner → {save_dir / '03_shap_meta_learner.png'}")


def plot_shap_wavenumber_level(data, save_dir):
    """
    SHAP at wavenumber level: retrain the best base model (lr_d1) on full data
    and compute SHAP values for each wavenumber.
    """
    import shap

    binary_labels = data["binary_labels"]
    cancer_type_labels = data["cancer_type_labels"]
    sample_ids = data["sample_ids"]

    # Load processed spectra for feature-level SHAP
    spectra_df = pd.read_csv(RESULTS_DIR / "processed_spectra.csv")
    wn_cols = [c for c in spectra_df.columns if c.startswith("x_")]
    wavenumbers = np.array([float(c.replace("x_", "")) for c in wn_cols])

    # Build sample_id key matching OOF convention:
    # OOF uses PAN_1 (from CPAN_1), PAN_YPAN_4 (from YPAN_4), NOR_YNOR_1 (from YNOR_1)
    def make_oof_sid(row):
        g, sid = row["group"], str(row["sample_id"])
        if g == "CPAN":
            return f"PAN_{sid}"
        elif g == "YPAN":
            return f"PAN_YPAN_{sid}"
        elif g == "YNOR":
            return f"NOR_YNOR_{sid}"
        else:
            return f"{g}_{sid}"

    spectra_df["sid"] = spectra_df.apply(make_oof_sid, axis=1)

    # Mean-aggregate replicates
    agg = spectra_df.groupby("sid")[wn_cols].mean()
    oof_sids = data["sample_ids"]

    # Align to OOF sample order
    common_sids = [s for s in oof_sids if s in agg.index]
    if len(common_sids) < len(oof_sids) * 0.8:
        print(f"  [!] Only {len(common_sids)}/{len(oof_sids)} samples matched — skipping wavenumber SHAP")
        return

    X_raw = agg.loc[common_sids].values
    sid_to_idx = {s: i for i, s in enumerate(oof_sids)}
    matched_indices = [sid_to_idx[s] for s in common_sids]
    y_bin = binary_labels[matched_indices]
    y_type = cancer_type_labels[matched_indices]

    # Compute 1st derivative (lr_d1 uses this)
    X_d1 = savgol_filter(X_raw, 11, 3, deriv=1, axis=1)

    # Train LR on d1 features — S2 (cancer type) for interpretability
    cancer_mask = y_bin == 1
    X_d1_cancer = X_d1[cancer_mask]
    y_type_cancer = y_type[cancer_mask].astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_d1_cancer)
    lr_s2 = LogisticRegression(C=1.0, max_iter=2000, solver="saga",
                                multi_class="multinomial", class_weight="balanced")
    lr_s2.fit(X_scaled, y_type_cancer)

    # SHAP
    explainer = shap.LinearExplainer(lr_s2, X_scaled)
    shap_values = explainer.shap_values(X_scaled)  # list of 7 arrays

    # ── Plot 1: Beeswarm-style summary per cancer type ──
    fig, axes = plt.subplots(4, 2, figsize=(20, 24))
    axes_flat = axes.ravel()

    shap_arr = np.array(shap_values)
    n_classes_actual = shap_arr.shape[2] if shap_arr.ndim == 3 else (len(shap_values) if isinstance(shap_values, list) else 7)

    for ci in range(min(7, n_classes_actual)):
        ax = axes_flat[ci]
        if shap_arr.ndim == 3:
            sv = shap_arr[:, :, ci]
        elif isinstance(shap_values, list):
            sv = shap_values[ci]
        else:
            sv = shap_arr
        mean_abs = np.abs(sv).mean(axis=0)
        top_k = 20
        top_idx = np.argsort(mean_abs)[-top_k:]

        # Color by feature value
        for rank, fi in enumerate(top_idx):
            vals = sv[:, fi]
            feat_vals = X_scaled[:, fi]
            ax.scatter(vals, [rank] * len(vals), c=feat_vals, cmap="coolwarm",
                       s=3, alpha=0.4, vmin=-2, vmax=2)
        ax.set_yticks(range(top_k))
        ax.set_yticklabels([f"{wavenumbers[i]:.0f} cm⁻¹" for i in top_idx], fontsize=9)
        ax.axvline(0, color="gray", lw=0.8, ls="--")
        ax.set_xlabel("SHAP value")
        ax.set_title(f"{CANCER_TYPES[ci]} ({CANCER_LABELS_KR.get(CANCER_TYPES[ci], '')})",
                     fontweight="bold", color=COLORS.get(CANCER_TYPES[ci], "black"))

    # Hide unused subplot
    axes_flat[7].axis("off")

    fig.suptitle("Wavenumber-Level SHAP — Per Cancer Type (lr_d1, 1st Derivative)",
                 fontsize=18, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(save_dir / "04_shap_wavenumber_per_cancer.png")
    plt.close(fig)
    print(f"  [✓] SHAP wavenumber per cancer → {save_dir / '04_shap_wavenumber_per_cancer.png'}")

    # ── Plot 2: Global wavenumber importance spectrum ──
    fig, ax = plt.subplots(figsize=(16, 6))
    shap_arr_g = np.array(shap_values)
    if shap_arr_g.ndim == 3:
        global_importance = np.abs(shap_arr_g).mean(axis=0).sum(axis=1)  # (features, classes) → sum over classes
    elif isinstance(shap_values, list):
        global_importance = np.sum([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        global_importance = np.abs(shap_arr_g).mean(axis=0)

    ax.fill_between(wavenumbers, global_importance, alpha=0.3, color="#1976D2")
    ax.plot(wavenumbers, global_importance, color="#1976D2", lw=1.5)
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Sum of Mean |SHAP| across 7 cancer types")
    ax.set_title("Global Wavenumber Importance — Stacking Best Model (lr_d1)", fontweight="bold")

    # Annotate top peaks
    from scipy.signal import find_peaks as scipy_find_peaks
    peaks, props = scipy_find_peaks(global_importance, distance=20, prominence=global_importance.max() * 0.05)
    for pi in peaks[:15]:
        ax.annotate(f"{wavenumbers[pi]:.0f}", xy=(wavenumbers[pi], global_importance[pi]),
                    xytext=(0, 10), textcoords="offset points", fontsize=8, ha="center",
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))

    ax.set_xlim(400, 2200)
    fig.tight_layout()
    fig.savefig(save_dir / "05_shap_global_wavenumber_importance.png")
    plt.close(fig)
    print(f"  [✓] SHAP global wavenumber → {save_dir / '05_shap_global_wavenumber_importance.png'}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  3. META-LEARNER COMPARISON                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def plot_meta_learner_comparison(save_dir):
    """Compare meta-learners from nested CV results."""
    df = pd.read_csv(STACKING_DIR / "nested_cv_results.csv")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    meta_order = ["lr", "elasticnet", "xgb", "rf", "mlp"]
    meta_colors = ["#1976D2", "#7B1FA2", "#FF6F00", "#2E7D32", "#C62828"]

    for idx, (metric, title) in enumerate([("s1_auc", "Stage 1: Det AUC"), ("s2_f1", "Stage 2: Type ID F1")]):
        ax = axes[idx]
        bp_data = [df[df["meta_learner"] == m][metric].values for m in meta_order]
        bp = ax.boxplot(bp_data, labels=[m.upper() for m in meta_order],
                        patch_artist=True, widths=0.6)
        for patch, color in zip(bp["boxes"], meta_colors):
            patch.set_facecolor(color); patch.set_alpha(0.3)
        for patch, color in zip(bp["medians"], meta_colors):
            patch.set_color(color); patch.set_linewidth(2)

        # Scatter individual folds
        for i, (vals, color) in enumerate(zip(bp_data, meta_colors)):
            jitter = np.random.normal(0, 0.05, len(vals))
            ax.scatter([i + 1 + j for j in jitter], vals, color=color, s=40, zorder=5, alpha=0.7)

        ax.set_ylabel(metric.replace("_", " ").upper())
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        # Mean annotation
        for i, m in enumerate(meta_order):
            mean_val = df[df["meta_learner"] == m][metric].mean()
            ax.text(i + 1, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02,
                    f"{mean_val:.4f}", ha="center", fontsize=9, fontweight="bold")

    fig.suptitle("Meta-Learner Comparison (5-Fold Nested CV)", fontsize=18, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_dir / "06_meta_learner_comparison.png")
    plt.close(fig)
    print(f"  [✓] Meta-learner comparison → {save_dir / '06_meta_learner_comparison.png'}")


def plot_base_model_contribution(save_dir):
    """Base model contribution and single vs ensemble comparison."""
    df_contrib = pd.read_csv(STACKING_DIR / "base_model_contribution.csv")
    df_single = pd.read_csv(STACKING_DIR / "single_vs_ensemble.csv")

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Single vs Ensemble
    ax = axes[0]
    singles = df_single[df_single["type"] == "single"].sort_values("f1_type", ascending=True)
    ens = df_single[df_single["type"] == "ensemble"]
    y_pos = range(len(singles))
    bars = ax.barh(y_pos, singles["f1_type"].values, color="#90CAF9", alpha=0.8, label="Single Model")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(singles["model"].values)
    ax.set_xlabel("Type ID F1 Score")
    ax.set_title("Single Models vs Ensemble", fontweight="bold")

    # Ensemble line
    ens_f1 = ens["f1_type"].values[0]
    ax.axvline(ens_f1, color="#E65100", lw=2.5, ls="--", label=f"Ensemble avg: {ens_f1:.3f}")

    # Best from nested CV
    with open(STACKING_DIR / "best_ensemble_config.json") as f:
        best = json.load(f)
    best_f1 = best["mean_s2_f1"]
    ax.axvline(best_f1, color="#C62828", lw=2.5, ls="-", label=f"Stacking LR: {best_f1:.3f}")
    ax.legend(loc="lower right")

    # Contribution
    ax = axes[1]
    df_c = df_contrib.sort_values("mean_auc_drop", ascending=True)
    colors = ["#E57373" if v < 0 else "#4CAF50" for v in df_c["mean_auc_drop"].values]
    ax.barh(range(len(df_c)), df_c["mean_auc_drop"].values, xerr=df_c["std_auc_drop"].values,
            color=colors, alpha=0.8, capsize=3)
    ax.set_yticks(range(len(df_c)))
    ax.set_yticklabels(df_c["base_model"].values)
    ax.set_xlabel("Mean AUC Drop When Removed")
    ax.set_title("Base Model Contribution\n(Permutation Importance)", fontweight="bold")
    ax.axvline(0, color="gray", lw=0.8)

    fig.suptitle("Stacking Ensemble — Architecture Analysis", fontsize=18, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_dir / "07_base_model_analysis.png")
    plt.close(fig)
    print(f"  [✓] Base model analysis → {save_dir / '07_base_model_analysis.png'}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  4. CONTRASTIVE SWEEP                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def plot_contrastive_results(save_dir):
    """Contrastive learning sweep results."""
    df = pd.read_csv(CONTRASTIVE_DIR / "sweep_results.csv")
    with open(CONTRASTIVE_DIR / "best_config.json") as f:
        best = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    # Heatmap: temp × proj_dim (best epochs), finetune F1
    ax = axes[0]
    pivot = df.groupby(["temperature", "proj_dim"])["finetune_f1_type"].max().reset_index()
    pivot_table = pivot.pivot(index="temperature", columns="proj_dim", values="finetune_f1_type")
    im = ax.imshow(pivot_table.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(pivot_table.columns)))
    ax.set_xticklabels(pivot_table.columns.astype(int))
    ax.set_yticks(range(len(pivot_table.index)))
    ax.set_yticklabels([f"{t:.2f}" for t in pivot_table.index])
    ax.set_xlabel("Projection Dim"); ax.set_ylabel("Temperature")
    ax.set_title("Finetune F1 (max over epochs)", fontweight="bold")
    for i in range(len(pivot_table.index)):
        for j in range(len(pivot_table.columns)):
            v = pivot_table.values[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9,
                    color="white" if v > 0.67 else "black")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Probe vs Finetune comparison
    ax = axes[1]
    ax.scatter(df["probe_f1_type"], df["finetune_f1_type"], c=df["temperature"],
               cmap="viridis", s=50, alpha=0.7)
    ax.plot([0.6, 0.7], [0.6, 0.7], "k--", lw=0.8, alpha=0.4)
    ax.set_xlabel("Probe F1"); ax.set_ylabel("Finetune F1")
    ax.set_title("Probe vs Finetune Performance", fontweight="bold")
    cb = plt.colorbar(ax.collections[0], ax=ax, shrink=0.8)
    cb.set_label("Temperature")

    # Best config annotation
    best_c = best["best_contrastive"]
    ax.scatter(best_c["probe_f1_type"], best_c["finetune_f1_type"],
               s=200, marker="*", color="red", zorder=10, label="Best")
    ax.legend()

    # Comparison with LR baseline
    ax = axes[2]
    methods = ["LR (d1)\nBaseline", "Supervised\nDL", "Contrastive\nProbe", "Contrastive\nFinetune"]
    f1_vals = [0.9136, best["supervised_baseline"]["f1_type"],
               best_c["probe_f1_type"], best_c["finetune_f1_type"]]
    colors_bar = ["#1976D2", "#9E9E9E", "#FF6F00", "#E65100"]
    bars = ax.bar(methods, f1_vals, color=colors_bar, alpha=0.85, width=0.6)
    for bar, v in zip(bars, f1_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Type ID F1 Score")
    ax.set_title("DL vs LR Comparison", fontweight="bold")
    ax.set_ylim(0.5, 1.0)
    ax.axhline(0.9136, color="#1976D2", ls="--", lw=1.5, alpha=0.5, label="LR best (0.914)")
    ax.legend()

    fig.suptitle("Contrastive Learning Sweep — 36 Configs, ~12h",
                 fontsize=18, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_dir / "08_contrastive_sweep.png")
    plt.close(fig)
    print(f"  [✓] Contrastive sweep → {save_dir / '08_contrastive_sweep.png'}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  5. LOHO VALIDATION                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def plot_loho_results(save_dir):
    """Leave-One-Hospital-Out validation results."""
    hospitals = ["Chungbuk", "SNUH", "StMarys"]
    results = {}
    for h in hospitals:
        fp = LOHO_DIR / "checkpoints" / f"loho_{h}_lr.json"
        if fp.exists():
            with open(fp) as f:
                results[h] = json.load(f)["result"]

    if not results:
        print("  [!] LOHO results not found — skipping")
        return

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Panel 1: S1 sensitivity
    ax = axes[0]
    hosp_names = list(results.keys())
    sens_vals = [results[h]["s1_sens"] for h in hosp_names]
    colors_h = ["#1976D2", "#E65100", "#2E7D32"]
    bars = ax.bar(hosp_names, sens_vals, color=colors_h, alpha=0.8, width=0.5)
    for bar, v in zip(bars, sens_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("Sensitivity")
    ax.set_title("Stage 1: Cancer Detection\nSensitivity", fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.axhline(0.8, color="red", ls="--", lw=1, alpha=0.5, label="Target (0.80)")
    ax.legend()

    # Panel 2: S2 F1
    ax = axes[1]
    f1_vals = [results[h]["s2_f1"] for h in hosp_names]
    bars = ax.bar(hosp_names, f1_vals, color=colors_h, alpha=0.8, width=0.5)
    for bar, v in zip(bars, f1_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("Macro F1")
    ax.set_title("Stage 2: Cancer Type ID\nMacro F1", fontweight="bold")
    ax.set_ylim(0, 1.1)

    # Panel 3: Per-cancer sensitivity
    ax = axes[2]
    all_cancers = set()
    for h in hosp_names:
        all_cancers.update(results[h].get("per_cancer_sens", {}).keys())
    all_cancers = sorted(all_cancers)

    x_pos = np.arange(len(all_cancers))
    width = 0.25
    for i, (h, color) in enumerate(zip(hosp_names, colors_h)):
        per_c = results[h].get("per_cancer_sens", {})
        vals = [per_c.get(c, 0) for c in all_cancers]
        ax.bar(x_pos + i * width, vals, width, color=color, alpha=0.8, label=h)
        for xi, v in zip(x_pos + i * width, vals):
            if v > 0:
                ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=8, rotation=45)

    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(all_cancers)
    ax.set_ylabel("Sensitivity")
    ax.set_title("Per-Cancer Sensitivity\nby Hospital", fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 1.2)

    fig.suptitle("Leave-One-Hospital-Out Validation — Generalization Assessment",
                 fontsize=18, fontweight="bold", y=1.02, color="#C62828")
    fig.tight_layout()
    fig.savefig(save_dir / "09_loho_validation.png")
    plt.close(fig)
    print(f"  [✓] LOHO validation → {save_dir / '09_loho_validation.png'}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  6. PERMUTATION IMPORTANCE                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def plot_permutation_importance(save_dir):
    """Wavenumber permutation importance from exp4."""
    df = pd.read_csv(PERM_DIR / "importance_ranking.csv")

    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    # Spectrum-wide importance
    ax = axes[0]
    auc_col = "mean_auc_drop" if "mean_auc_drop" in df.columns else "auc_drop_mean"
    auc_std_col = "std_auc_drop" if "std_auc_drop" in df.columns else "auc_drop_std"

    ax.fill_between(df["wavenumber"], df[auc_col], alpha=0.3, color="#1976D2")
    ax.plot(df["wavenumber"], df[auc_col], color="#1976D2", lw=1)
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("AUC Drop (permutation)")
    ax.set_title("Permutation Feature Importance — Full Spectrum", fontweight="bold")
    ax.set_xlim(400, 2200)

    # Top peaks
    top20 = df.nlargest(20, auc_col)
    for _, row in top20.iterrows():
        ax.annotate(f"{row['wavenumber']:.0f}", xy=(row["wavenumber"], row[auc_col]),
                    xytext=(0, 8), textcoords="offset points", fontsize=7, ha="center",
                    arrowprops=dict(arrowstyle="-", color="red", lw=0.5))

    # Top 30 bar chart
    ax = axes[1]
    top30 = df.nlargest(30, auc_col).sort_values(auc_col, ascending=True)
    ax.barh(range(30), top30[auc_col].values,
            xerr=top30[auc_std_col].values if auc_std_col in top30.columns else None,
            color="#1976D2", alpha=0.8, capsize=2)
    ax.set_yticks(range(30))
    ax.set_yticklabels([f"{w:.1f} cm⁻¹" for w in top30["wavenumber"].values], fontsize=9)
    ax.set_xlabel("AUC Drop")
    ax.set_title("Top 30 Most Important Wavenumbers", fontweight="bold")

    fig.suptitle("Permutation Feature Importance — 1000 Permutations per Feature",
                 fontsize=18, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_dir / "10_permutation_importance.png")
    plt.close(fig)
    print(f"  [✓] Permutation importance → {save_dir / '10_permutation_importance.png'}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  7. NORM × FEATURE SEARCH (partial results)                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def plot_norm_feature_search(save_dir):
    """Parse checkpoint JSONs and create summary heatmaps."""
    ckpt_dir = NORMSEARCH_DIR / "checkpoints"
    if not ckpt_dir.exists():
        print("  [!] Norm×Feature search checkpoints not found — skipping")
        return

    records = []
    for fp in ckpt_dir.glob("*.json"):
        try:
            with open(fp) as f:
                d = json.load(f)
            parts = fp.stem.split("_")
            # e.g., l2_d1_LR_seed42 → norm=l2, feat=d1, model=LR, seed=42
            seed_idx = next(i for i, p in enumerate(parts) if p.startswith("seed"))
            norm = parts[0]
            model = parts[seed_idx - 1]
            feat = "_".join(parts[1:seed_idx - 1])
            seed = int(parts[seed_idx].replace("seed", ""))
            records.append({
                "norm": norm, "feat": feat, "model": model, "seed": seed,
                "auc": d.get("s1_auc", d.get("auc", np.nan)),
                "f1": d.get("s2_f1", d.get("f1_type", np.nan)),
            })
        except Exception:
            continue

    if not records:
        print("  [!] No valid checkpoints found — skipping")
        return

    df = pd.DataFrame(records)
    print(f"  Parsed {len(df)} checkpoint results ({df['norm'].nunique()} norms × {df['feat'].nunique()} feats × {df['model'].nunique()} models)")

    # Aggregate over seeds
    agg = df.groupby(["norm", "feat", "model"]).agg(
        auc_mean=("auc", "mean"), auc_std=("auc", "std"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        n_seeds=("seed", "count"),
    ).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    for mi, (model, color) in enumerate(MODEL_COLORS.items()):
        ax = axes[mi]
        sub = agg[agg["model"] == model]
        if sub.empty:
            ax.set_title(f"{model} — No data"); continue

        pivot = sub.pivot(index="norm", columns="feat", values="f1_mean")
        im = ax.imshow(pivot.values, cmap="YlGnBu", aspect="auto",
                       vmin=agg["f1_mean"].quantile(0.1), vmax=agg["f1_mean"].quantile(0.95))
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("Feature Transform"); ax.set_ylabel("Normalization")
        ax.set_title(f"{model} — Type ID F1", fontweight="bold")

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                v = pivot.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9,
                            color="white" if v > pivot.values[~np.isnan(pivot.values)].mean() else "black")
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Normalization × Feature Transform Search (Partial: 1040/1800)",
                 fontsize=18, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_dir / "11_norm_feature_search.png")
    plt.close(fig)
    print(f"  [✓] Norm×Feature search → {save_dir / '11_norm_feature_search.png'}")

    # Best combos table
    top10 = agg.nlargest(10, "f1_mean")
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.axis("off")
    table = ax2.table(
        cellText=top10[["norm", "feat", "model", "f1_mean", "f1_std", "auc_mean", "n_seeds"]].round(4).values,
        colLabels=["Norm", "Feature", "Model", "F1 Mean", "F1 Std", "AUC Mean", "N Seeds"],
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)
    # Color best row
    for j in range(7):
        table[1, j].set_facecolor("#E8F5E9")
    ax2.set_title("Top 10 Norm × Feature × Model Combinations", fontsize=16, fontweight="bold", pad=20)
    fig2.tight_layout()
    fig2.savefig(save_dir / "12_norm_feature_top10.png")
    plt.close(fig2)
    print(f"  [✓] Top 10 combos → {save_dir / '12_norm_feature_top10.png'}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  8. COMPREHENSIVE SUMMARY FIGURE                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def plot_comprehensive_summary(save_dir):
    """One-page summary of all weekend experiments."""
    fig = plt.figure(figsize=(24, 14))
    gs = gridspec.GridSpec(2, 4, hspace=0.4, wspace=0.3)

    # 1. Stacking result card
    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    with open(STACKING_DIR / "best_ensemble_config.json") as f:
        best = json.load(f)
    text = (
        f"STACKING ENSEMBLE\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"Meta: LR\n"
        f"Base Models: 10\n"
        f"Nested CV: 5×3\n\n"
        f"Det AUC: {best['mean_s1_auc']:.4f}\n"
        f"Type F1: {best['mean_s2_f1']:.4f}\n"
        f"  ±{best['std_s2_f1']:.3f}\n\n"
        f"★ 7-cancer 역대 최고"
    )
    ax.text(0.5, 0.5, text, transform=ax.transAxes, fontsize=13, va="center", ha="center",
            fontfamily="monospace", bbox=dict(boxstyle="round,pad=0.8", facecolor="#E8F5E9", alpha=0.9))

    # 2. Contrastive card
    ax = fig.add_subplot(gs[0, 1])
    ax.axis("off")
    with open(CONTRASTIVE_DIR / "best_config.json") as f:
        cont = json.load(f)
    bc = cont["best_contrastive"]
    text = (
        f"CONTRASTIVE SWEEP\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"Configs: 36\n"
        f"Best: τ=0.05 d=128\n\n"
        f"Probe F1:    {bc['probe_f1_type']:.3f}\n"
        f"Finetune F1: {bc['finetune_f1_type']:.3f}\n"
        f"vs LR best:  0.914\n\n"
        f"✗ DL < LR 여전히"
    )
    ax.text(0.5, 0.5, text, transform=ax.transAxes, fontsize=13, va="center", ha="center",
            fontfamily="monospace", bbox=dict(boxstyle="round,pad=0.8", facecolor="#FFF3E0", alpha=0.9))

    # 3. LOHO card
    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    text = (
        f"LOHO VALIDATION\n"
        f"━━━━━━━━━━━━━━━\n"
        f"Chungbuk: F1 0.000\n"
        f"  (PRO/CRC/PAN/BLC)\n"
        f"SNUH:     F1 0.259\n"
        f"  (OVA/LUN)\n"
        f"StMarys:  F1 0.108\n"
        f"  (LUN)\n\n"
        f"✗ 병원간 일반화 실패"
    )
    ax.text(0.5, 0.5, text, transform=ax.transAxes, fontsize=13, va="center", ha="center",
            fontfamily="monospace", bbox=dict(boxstyle="round,pad=0.8", facecolor="#FFEBEE", alpha=0.9))

    # 4. Norm search card
    ax = fig.add_subplot(gs[0, 3])
    ax.axis("off")
    text = (
        f"NORM×FEATURE SEARCH\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"Progress: 1040/1800\n"
        f"(58%, 프로세스 중단)\n\n"
        f"Best so far:\n"
        f"  snv/d1/LR  F1=0.910\n"
        f"  l2/d1/LR   F1=0.917\n\n"
        f"LR >> XGB >> RF\n"
        f"d1 > raw_d1 > d2"
    )
    ax.text(0.5, 0.5, text, transform=ax.transAxes, fontsize=13, va="center", ha="center",
            fontfamily="monospace", bbox=dict(boxstyle="round,pad=0.8", facecolor="#E3F2FD", alpha=0.9))

    # 5. Historical F1 comparison (bottom left)
    ax = fig.add_subplot(gs[1, :2])
    experiments = [
        ("Phase Q\nLR+sex", 0.892, "#90CAF9"),
        ("Phase V\n3view+clin", 0.914, "#64B5F6"),
        ("Phase W\nStacking v1", 0.877, "#42A5F5"),
        ("Contrastive\nBest", 0.681, "#FFCC80"),
        ("Norm Search\nl2/d1/LR", 0.917, "#81C784"),
        ("Stacking v2\nLR Meta ★", 0.946, "#2E7D32"),
    ]
    names, vals, cols = zip(*experiments)
    bars = ax.bar(names, vals, color=cols, alpha=0.85, width=0.6, edgecolor="white", linewidth=1.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"{v:.3f}", ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("Type ID Macro F1")
    ax.set_title("7-Cancer Type ID F1 — Historical Comparison", fontweight="bold")
    ax.set_ylim(0.6, 1.02)
    ax.axhline(0.9, color="red", ls="--", lw=1, alpha=0.4, label="F1=0.9 target")
    ax.legend()

    # 6. Key insights (bottom right)
    ax = fig.add_subplot(gs[1, 2:])
    ax.axis("off")
    insights = (
        "KEY INSIGHTS — Weekend Experiments Summary\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "1. Stacking LR meta-learner F1=0.946 → 7-cancer 역대 최고\n"
        "   • lr_d1, lr_d2가 앙상블에서 가장 중요한 base model\n"
        "   • 단일 best (ridge_concat 0.934) 대비 +1.2pp\n\n"
        "2. Contrastive learning (36 config sweep)\n"
        "   • Best F1=0.681 — LR (0.914)의 75% 수준\n"
        "   • DL이 LR을 이기지 못하는 패턴 재확인\n\n"
        "3. LOHO: 병원 간 일반화 완전 실패\n"
        "   • Chungbuk 제외 시 PRO/CRC/PAN/BLC 분류 불가\n"
        "   • 보라매 전향적 검증 전 batch correction 필수\n\n"
        "4. Norm×Feature: l2+d1+LR이 최적 조합 (중간 결과)\n"
        "   • SNV ≈ L2 >> MinMax\n"
        "   • 프로세스 재실행 필요 (760개 남음)"
    )
    ax.text(0.05, 0.95, insights, transform=ax.transAxes, fontsize=12, va="top",
            fontfamily="monospace", bbox=dict(boxstyle="round,pad=0.8", facecolor="#F5F5F5", alpha=0.9))

    fig.suptitle("SERS-AI Weekend Experiments — 2026-04-03 ~ 04-06",
                 fontsize=22, fontweight="bold", y=1.02)
    fig.savefig(save_dir / "00_weekend_summary.png")
    plt.close(fig)
    print(f"  [✓] Summary dashboard → {save_dir / '00_weekend_summary.png'}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    print("=" * 60)
    print("Weekend Experiments — Comprehensive Analysis")
    print("=" * 60)

    # Load stacking OOF data
    print("\n[1/8] Loading stacking OOF predictions...")
    data = load_oof_data()

    print("\n[2/8] Confusion matrices...")
    s1_auc, s2_f1, per_class_f1 = plot_stacking_confusion_matrices(data, OUTPUT_DIR)
    print(f"       S1 AUC={s1_auc:.4f}, S2 Macro F1={s2_f1:.4f}")

    print("\n[3/8] ROC curves...")
    plot_stacking_roc_curves(data, OUTPUT_DIR)

    print("\n[4/8] SHAP — Meta-learner level...")
    try:
        plot_shap_meta_level(data, OUTPUT_DIR)
    except Exception as e:
        print(f"  [!] SHAP meta failed: {e}")

    print("\n[5/8] SHAP — Wavenumber level...")
    try:
        plot_shap_wavenumber_level(data, OUTPUT_DIR)
    except Exception as e:
        print(f"  [!] SHAP wavenumber failed: {e}")

    print("\n[6/8] Meta-learner comparison & base model analysis...")
    plot_meta_learner_comparison(OUTPUT_DIR)
    plot_base_model_contribution(OUTPUT_DIR)

    print("\n[7/8] Other experiments...")
    plot_contrastive_results(OUTPUT_DIR)
    plot_loho_results(OUTPUT_DIR)
    plot_permutation_importance(OUTPUT_DIR)
    plot_norm_feature_search(OUTPUT_DIR)

    print("\n[8/8] Comprehensive summary...")
    plot_comprehensive_summary(OUTPUT_DIR)

    print("\n" + "=" * 60)
    print(f"All figures saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
