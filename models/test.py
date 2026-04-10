"""
SERS Cancer Detection ??Evaluation & Visualization (ResNet18-1D)

Two-stage: Binary + Cancer Type (8: PRO, BRE, OVA, LUN, CRC, CPAN, SPAN, BLC)

Usage:
    python test.py
    python test.py -i results/training --no-shap

Output (results/training/evaluation/):
    metrics/   ??summary, per-group, threshold sweep
    stage1/    ??ROC, confusion, distribution, train vs val, SHAP
    stage2/    ??per-type ROC, confusion, bar chart, train vs val, SHAP
    embedding/ ??t-SNE
"""

from __future__ import annotations

import sys
import argparse
import logging
import os
import json
from pathlib import Path
from datetime import datetime
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, auc,
    confusion_matrix,
    precision_recall_curve, average_precision_score,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.manifold import TSNE

import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from models.model import ModelConfig, build_model
except ImportError:
    from model import ModelConfig, build_model
from src.sers.visualization import (
    build_mean_spectrum_profile,
    build_shap_spectrum_profile,
    compute_gradient_shap_values,
    plot_confusion_summary_bar,
    plot_binary_shap_summary,
    plot_class_shap_summary,
    plot_mean_spectrum,
    plot_mean_spectra_overlay,
    plot_multiclass_shap_summary,
    plot_peak_intensity_overview,
    plot_peak_intensity_profile,
    plot_group_peak_difference,
    plot_shap_feature_importance_bar,
    plot_shap_mean_magnitude_spectrum,
    plot_shap_mean_signed_spectrum,
    summarize_confusion_pairs,
    summarize_shap_feature_importance,
)

logger = logging.getLogger(__name__)

plt.rcParams.update({
    "figure.dpi": 180,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 13,
    "axes.titlesize": 18,
    "axes.labelsize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "axes.linewidth": 1.1,
})
GROUP_COLORS = {
    "PRO": "#E91E63", "BRE": "#F06292", "OVA": "#AB47BC",
    "LUN": "#42A5F5", "CRC": "#EF5350", "CPAN": "#FFA726", "SPAN": "#FF7043",
    "NOR": "#8D6E63", "DIA": "#66BB6A", "HBP": "#26A69A", "H.D.": "#78909C",
}


# =============================================================================
# 1. Load
# =============================================================================
def load_outputs(train_dir):
    npz = np.load(train_dir / "fold_predictions.npz", allow_pickle=True)
    with open(train_dir / "training_summary.json") as f:
        summary = json.load(f)
    data = {k: npz[k] for k in npz.files}
    if data["groups"].dtype.kind in ("U", "O"):
        data["groups"] = np.array(data["groups"], dtype=str)
    logger.info(f"Loaded: {data['X'].shape[0]} samples, {data['X'].shape[1]} features")
    logger.info(f"  Cancer types: {summary['cancer_types']}")
    return data, summary


# =============================================================================
# 2. Metrics
# =============================================================================
def binary_metrics(y_true, y_prob, threshold=0.5, prefix=""):
    y_pred = (y_prob > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        f"{prefix}auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        f"{prefix}pr_auc": average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        f"{prefix}accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0,
        f"{prefix}specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        f"{prefix}ppv": tp / (tp + fp) if (tp + fp) > 0 else 0,
        f"{prefix}npv": tn / (tn + fn) if (tn + fn) > 0 else 0,
        f"{prefix}f1": f1_score(y_true, y_pred, zero_division=0),
    }


def multiclass_metrics(y_true, y_logits, names, prefix=""):
    y_prob = torch.softmax(torch.tensor(y_logits), dim=-1).numpy()
    y_pred = y_prob.argmax(axis=1)
    m = {
        f"{prefix}accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    try:
        m[f"{prefix}auc_macro"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except ValueError:
        m[f"{prefix}auc_macro"] = np.nan

    try:
        per_class_ap = []
        for idx in sorted(set(y_true)):
            bg = (y_true == idx).astype(int)
            if bg.sum() > 0:
                per_class_ap.append(average_precision_score(bg, y_prob[:, idx]))
        m[f"{prefix}pr_auc_macro"] = float(np.mean(per_class_ap)) if per_class_ap else np.nan
    except ValueError:
        m[f"{prefix}pr_auc_macro"] = np.nan

    for idx in sorted(set(y_true)):
        name = names[idx] if idx < len(names) else f"cls_{idx}"
        bg = (y_true == idx).astype(int)
        if bg.sum() == 0: continue
        m[f"{prefix}{name}_sens"] = recall_score(y_true == idx, y_pred == idx, zero_division=0)
        m[f"{prefix}{name}_prec"] = precision_score(y_true == idx, y_pred == idx, zero_division=0)
        try:
            m[f"{prefix}{name}_auc"] = roc_auc_score(bg, y_prob[:, idx])
        except ValueError:
            m[f"{prefix}{name}_auc"] = np.nan
        try:
            m[f"{prefix}{name}_pr_auc"] = average_precision_score(bg, y_prob[:, idx])
        except ValueError:
            m[f"{prefix}{name}_pr_auc"] = np.nan
        m[f"{prefix}{name}_n"] = int(bg.sum())
    return m


def optimize_threshold(y_true, y_prob):
    fpr, tpr, th = roc_curve(y_true, y_prob)
    j = tpr - fpr
    best = j.argmax()
    return float(th[best]), float(j[best])


def per_group_metrics(y_true, y_prob, groups, threshold=0.5):
    rows = []
    for g in sorted(set(groups)):
        mask = groups == g
        gt, gp = y_true[mask], y_prob[mask]
        gpred = (gp > threshold).astype(int)
        row = {"group": g, "n": int(mask.sum()), "n_cancer": int((gt == 1).sum())}
        row["auc"] = roc_auc_score(gt, gp) if len(np.unique(gt)) > 1 else np.nan
        row["accuracy"] = accuracy_score(gt, gpred)
        row["sensitivity"] = recall_score(gt, gpred, zero_division=0) if (gt == 1).any() else np.nan
        neg = (gt == 0).sum()
        row["specificity"] = ((gpred == 0) & (gt == 0)).sum() / neg if neg > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# 3. Stage 1 Plots
# =============================================================================
def plot_roc_s1(yt, yp, path):
    fpr, tpr, th = roc_curve(yt, yp)
    a = auc(fpr, tpr)
    opt_t, j = optimize_threshold(yt, yp)
    oi = np.argmin(np.abs(th - opt_t))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr, tpr, color="#c0392b", lw=2.5, label=f"ROC (AUC = {a:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.scatter(fpr[oi], tpr[oi], s=120, c="gold", edgecolors="black",
               zorder=5, label=f"Optimal (J={j:.3f}, ?={opt_t:.3f})")
    ax.set(xlabel="1 ??Specificity", ylabel="Sensitivity",
           title="Stage 1: Cancer vs Non-cancer (ResNet18-1D)")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right", fontsize=11); ax.grid(True, alpha=0.3)
    ax.text(0.55, 0.12, f"?={opt_t:.3f}\nSens={tpr[oi]:.3f}\nSpec={1-fpr[oi]:.3f}",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9))
    plt.tight_layout(); fig.savefig(path); plt.close()
    return opt_t


def plot_pr_s1(yt, yp, path):
    precision, recall, _ = precision_recall_curve(yt, yp)
    ap = average_precision_score(yt, yp)
    prevalence = yt.sum() / len(yt)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(recall, precision, color="#c0392b", lw=2.5, label=f"PR (AP = {ap:.4f})")
    ax.axhline(prevalence, color="k", ls="--", alpha=0.3, label=f"Baseline (prev={prevalence:.3f})")
    ax.set(xlabel="Recall", ylabel="Precision",
           title="Stage 1: Cancer vs Non-cancer — Precision-Recall")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower left", fontsize=11); ax.grid(True, alpha=0.3)
    plt.tight_layout(); fig.savefig(path); plt.close()


def plot_tsne_3d(emb, groups, bl, path):
    logger.info("  t-SNE (3D)...")
    try:
        tsne = TSNE(n_components=3, random_state=42, perplexity=min(30, len(emb) - 1))
        co = tsne.fit_transform(emb)
    except Exception as exc:
        logger.warning(f"  t-SNE skipped: {exc}")
        return

    fig = plt.figure(figsize=(18, 8))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    for label, c, name in [(0, "#3498db", "Non-cancer"), (1, "#e74c3c", "Cancer")]:
        m = bl == label
        ax1.scatter(co[m, 0], co[m, 1], co[m, 2], c=c, s=12, alpha=0.5, label=f"{name} (n={m.sum()})")
    ax1.set_title("t-SNE - Binary")
    ax1.legend(fontsize=9)

    for g in sorted(set(groups)):
        m = groups == g
        ax2.scatter(co[m, 0], co[m, 1], co[m, 2], c=GROUP_COLORS.get(g, "#999"),
                    s=12, alpha=0.5, label=f"{g} (n={m.sum()})")
    ax2.set_title("t-SNE - All Groups")
    ax2.legend(fontsize=7, ncol=2)
    plt.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_cm_s1(yt, yp, t, path):
    ypr = (yp > t).astype(int)
    cm = confusion_matrix(yt, ypr, labels=[0, 1])
    cn = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    labels = ["Non-cancer", "Cancer"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=labels, yticklabels=labels, cbar=False, annot_kws={"size": 16})
    axes[0].set(xlabel="Predicted", ylabel="True", title=f"Counts (?={t:.3f})")
    sns.heatmap(cn, annot=True, fmt=".2%", cmap="Blues", ax=axes[1],
                xticklabels=labels, yticklabels=labels, cbar=False, annot_kws={"size": 16}, vmin=0, vmax=1)
    axes[1].set(xlabel="Predicted", ylabel="True", title="Normalized")
    plt.suptitle("Stage 1 ??Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout(); fig.savefig(path); plt.close()


def plot_dist(yt, yp, t, path):
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, 1, 60)
    ax.hist(yp[yt == 0], bins=bins, alpha=0.55, color="#3498db",
            label=f"Non-cancer (n={(yt==0).sum()})", density=True, edgecolor="white")
    ax.hist(yp[yt == 1], bins=bins, alpha=0.55, color="#e74c3c",
            label=f"Cancer (n={(yt==1).sum()})", density=True, edgecolor="white")
    ax.axvline(t, color="black", ls="--", lw=1.5, label=f"?={t:.3f}")
    ax.set(xlabel="P(Cancer)", ylabel="Density", title="Stage 1 ??Prediction Distribution")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); fig.savefig(path); plt.close()


def plot_train_val_roc(tbt, tbp, vbt, vbp, title, path):
    fig, ax = plt.subplots(figsize=(7, 7))
    for label, bt, bp, c in [("Train", tbt, tbp, "#3498db"), ("Val", vbt, vbp, "#e74c3c")]:
        fpr, tpr, _ = roc_curve(bt, bp)
        ax.plot(fpr, tpr, color=c, lw=2, label=f"{label} AUC={auc(fpr, tpr):.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set(xlabel="FPR", ylabel="TPR", title=title)
    ax.legend(fontsize=12); ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    plt.tight_layout(); fig.savefig(path); plt.close()


# =============================================================================
# 4. Stage 2 Plots
# =============================================================================
def plot_roc_s2(yt, yl, names, path):
    yp = torch.softmax(torch.tensor(yl), dim=-1).numpy()
    present = sorted(set(yt))
    fig, ax = plt.subplots(figsize=(8, 7))
    for idx in present:
        name = names[idx] if idx < len(names) else f"Type {idx}"
        bg = (yt == idx).astype(int)
        if bg.sum() == 0: continue
        fpr, tpr, _ = roc_curve(bg, yp[:, idx])
        try: a = roc_auc_score(bg, yp[:, idx])
        except: a = np.nan
        ax.plot(fpr, tpr, lw=2, color=GROUP_COLORS.get(name),
                label=f"{name} (n={bg.sum()}, AUC={a:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set(xlabel="FPR", ylabel="TPR", title="Stage 2 ??Per-Cancer ROC (OvR)")
    ax.legend(fontsize=9, loc="lower right"); ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    plt.tight_layout(); fig.savefig(path); plt.close()


def plot_pr_s2(yt, yl, names, path):
    yp = torch.softmax(torch.tensor(yl), dim=-1).numpy()
    present = sorted(set(yt))
    fig, ax = plt.subplots(figsize=(8, 7))
    for idx in present:
        name = names[idx] if idx < len(names) else f"Type {idx}"
        bg = (yt == idx).astype(int)
        if bg.sum() == 0: continue
        prec_arr, rec_arr, _ = precision_recall_curve(bg, yp[:, idx])
        try:
            ap = average_precision_score(bg, yp[:, idx])
        except ValueError:
            ap = np.nan
        ax.plot(rec_arr, prec_arr, lw=2, color=GROUP_COLORS.get(name),
                label=f"{name} (n={bg.sum()}, AP={ap:.3f})")
    ax.set(xlabel="Recall", ylabel="Precision",
           title="Stage 2 — Per-Cancer PR Curves (OvR)")
    ax.legend(fontsize=9, loc="lower left"); ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    plt.tight_layout(); fig.savefig(path); plt.close()


def plot_cm_s2(yt, yl, names, path):
    yp = torch.softmax(torch.tensor(yl), dim=-1).numpy()
    ypred = yp.argmax(axis=1)
    present = sorted(set(yt))
    ns = [names[i] for i in present]
    cm = confusion_matrix(yt, ypred, labels=present)
    cn = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", ax=axes[0],
                xticklabels=ns, yticklabels=ns, cbar=False, annot_kws={"size": 11})
    axes[0].set(xlabel="Predicted", ylabel="True", title="Counts")
    sns.heatmap(cn, annot=True, fmt=".1%", cmap="Oranges", ax=axes[1],
                xticklabels=ns, yticklabels=ns, cbar=False, annot_kws={"size": 11}, vmin=0, vmax=1)
    axes[1].set(xlabel="Predicted", ylabel="True", title="Normalized")
    plt.suptitle("Stage 2 ??Cancer Type Confusion (8-class)", fontsize=14, fontweight="bold")
    plt.tight_layout(); fig.savefig(path); plt.close()


def plot_bars_s2(yt, yl, names, path):
    yp = torch.softmax(torch.tensor(yl), dim=-1).numpy()
    ypred = yp.argmax(axis=1)
    present = sorted(set(yt))
    rows = []
    for idx in present:
        name = names[idx] if idx < len(names) else f"T{idx}"
        bg = (yt == idx).astype(int)
        if bg.sum() == 0: continue
        sens = recall_score(yt == idx, ypred == idx, zero_division=0)
        prec = precision_score(yt == idx, ypred == idx, zero_division=0)
        try: a = roc_auc_score(bg, yp[:, idx])
        except: a = np.nan
        rows.append({"type": name, "n": bg.sum(), "sensitivity": sens, "precision": prec, "auc": a})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(df)); w = 0.25
    ax.bar(x - w, df["sensitivity"], w, label="Sensitivity", color="#e74c3c", alpha=0.8)
    ax.bar(x, df["precision"], w, label="Precision", color="#3498db", alpha=0.8)
    ax.bar(x + w, df["auc"], w, label="AUC (OvR)", color="#2ecc71", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['type']}\n(n={r['n']})" for _, r in df.iterrows()])
    ax.set(ylabel="Score", title="Stage 2 ??Per-Type Metrics"); ax.legend()
    ax.grid(True, axis="y", alpha=0.3); ax.set_ylim(0, 1.05)
    plt.tight_layout(); fig.savefig(path); plt.close()
    return df


def plot_train_val_s2(tct, tcl, vct, vcl, names, path):
    present = sorted(set(vct))
    n = len(present)
    cols = min(4, n); rows_n = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(5 * cols, 4.5 * rows_n))
    axes = np.array(axes).flatten() if n > 1 else np.array([axes])

    for ai, idx in enumerate(present):
        if ai >= len(axes): break
        ax = axes[ai]
        name = names[idx] if idx < len(names) else f"T{idx}"
        for label, ct, cl, c in [("Train", tct, tcl, "#3498db"), ("Val", vct, vcl, "#e74c3c")]:
            prob = torch.softmax(torch.tensor(cl), dim=-1).numpy()
            bg = (ct == idx).astype(int)
            if bg.sum() == 0: continue
            fpr, tpr, _ = roc_curve(bg, prob[:, idx])
            ax.plot(fpr, tpr, color=c, lw=1.5, label=f"{label} AUC={auc(fpr, tpr):.3f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_title(name); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    for j in range(len(present), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Stage 2 ??Train vs Val per Cancer Type", fontsize=14, fontweight="bold")
    plt.tight_layout(); fig.savefig(path); plt.close()


# =============================================================================
# 5. SHAP
# =============================================================================
def run_shap_s1(model, X_bg, X_exp, device, out_dir, feature_names):
    logger.info("  SHAP Stage 1...")
    import torch.nn as nn

    class W(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
        def forward(self, x):
            return self.base(x)["binary_prob"].squeeze(-1)

    w = W(model).to(device)
    try:
        sv = compute_gradient_shap_values(w, X_bg, X_exp, device)
        sv = plot_binary_shap_summary(
            sv,
            X_exp,
            out_dir / "shap_summary.png",
            feature_names=feature_names,
            top_k=30,
            title="Stage 1 - SHAP (Top 30 Wavenumbers)",
        )
        np.save(out_dir / "shap_values_s1.npy", sv)
        logger.info(f"    Saved: {np.asarray(sv).shape}")
    except Exception as e:
        logger.warning(f"    Failed: {e}")


def build_feature_names(n_features):
    return [f"x_{i}" for i in range(n_features)]


def resolve_feature_names(processed_csv, n_features):
    if not processed_csv:
        return build_feature_names(n_features)

    csv_path = Path(processed_csv)
    if not csv_path.is_absolute():
        csv_path = PROJECT_ROOT / csv_path

    if not csv_path.exists():
        logger.warning(f"  Processed spectra CSV not found: {csv_path}. Falling back to feature indices.")
        return build_feature_names(n_features)

    try:
        columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
    except Exception as exc:
        logger.warning(f"  Failed to read feature names from {csv_path}: {exc}")
        return build_feature_names(n_features)

    feature_names = [col for col in columns if str(col).startswith("x_")]
    if len(feature_names) != n_features:
        logger.warning(
            f"  Feature count mismatch between SHAP input ({n_features}) and {csv_path} ({len(feature_names)}). "
            "Falling back to feature indices."
        )
        return build_feature_names(n_features)
    return feature_names


def sanitize_output_name(name):
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(name))
    return cleaned.strip("_") or "class"


def sample_shap_inputs(X, valid_mask, shap_samples, shap_explain, seed=42):
    rng = np.random.RandomState(seed)
    n_bg = min(shap_samples, len(X))
    X_bg = X[rng.choice(len(X), n_bg, replace=False)]

    valid_idx = np.flatnonzero(valid_mask)
    if len(valid_idx) == 0:
        return X_bg, np.empty((0, X.shape[1]), dtype=X.dtype), np.array([], dtype=int)

    n_exp = min(shap_explain, len(valid_idx))
    exp_idx = rng.choice(valid_idx, n_exp, replace=False)
    return X_bg, X[exp_idx], exp_idx


def save_shap_outputs_by_diagnosis(sv_list, X_exp, y_exp, cancer_types, out_dir, feature_names, top_k=20):
    diagnosis_dir = out_dir / "by_diagnosis"
    diagnosis_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    n_classes = min(len(sv_list), len(cancer_types))
    for class_idx in range(n_classes):
        class_mask = y_exp == class_idx
        if not class_mask.any():
            continue

        class_name = cancer_types[class_idx]
        class_dir = diagnosis_dir / sanitize_output_name(class_name)
        class_dir.mkdir(parents=True, exist_ok=True)

        class_sv = np.asarray(sv_list[class_idx])[class_mask]
        class_x = X_exp[class_mask]
        profile_df = build_shap_spectrum_profile(class_sv, feature_names=feature_names)
        profile_df.insert(0, "diagnosis", class_name)
        profile_df.insert(1, "n_samples", int(class_mask.sum()))
        profile_df.to_csv(class_dir / "shap_spectrum_profile.csv", index=False)

        plot_class_shap_summary(
            class_sv,
            class_x,
            class_dir / "shap_summary.png",
            feature_names=feature_names,
            top_k=top_k,
            title=f"{class_name} - SHAP Summary",
        )

        importance_df = summarize_shap_feature_importance(
            class_sv,
            feature_names=feature_names,
            top_k=top_k,
        )
        importance_df.insert(0, "diagnosis", class_name)
        importance_df.insert(1, "n_samples", int(class_mask.sum()))
        importance_df.to_csv(class_dir / "feature_importance.csv", index=False)

        plot_shap_feature_importance_bar(
            importance_df,
            class_dir / "feature_importance.png",
            title=f"{class_name} - SHAP Feature Importance",
            color=GROUP_COLORS.get(class_name, "#c0392b"),
        )
        plot_shap_mean_magnitude_spectrum(
            class_sv,
            class_dir / "mean_abs_shap_spectrum.png",
            feature_names=feature_names,
            title=f"{class_name} - Mean |SHAP| Spectrum",
            color=GROUP_COLORS.get(class_name, "#c0392b"),
        )
        plot_shap_mean_signed_spectrum(
            class_sv,
            class_dir / "mean_shap_spectrum.png",
            feature_names=feature_names,
            title=f"{class_name} - Mean SHAP Spectrum",
        )

        np.save(class_dir / "shap_values.npy", class_sv)
        rows.append(importance_df)

    if rows:
        pd.concat(rows, ignore_index=True).to_csv(
            out_dir / "feature_importance_by_diagnosis.csv",
            index=False,
        )


def save_mean_spectra_by_diagnosis(X_val, y_val, y_pred, cancer_types, out_dir, feature_names):
    diagnosis_dir = out_dir / "by_diagnosis"
    diagnosis_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = list(feature_names)
    spectra_df = pd.DataFrame(np.asarray(X_val), columns=feature_cols)
    spectra_df["group"] = [cancer_types[idx] for idx in y_val]
    confusion_df = summarize_confusion_pairs(y_val, y_pred, cancer_types)

    spectra_by_class = []
    class_names = []
    peak_rows = []

    if not confusion_df.empty:
        confusion_df.to_csv(out_dir / "confusion_pairs.csv", index=False)

    for class_idx, class_name in enumerate(cancer_types):
        class_mask = y_val == class_idx
        if not class_mask.any():
            continue

        class_dir = diagnosis_dir / sanitize_output_name(class_name)
        class_dir.mkdir(parents=True, exist_ok=True)
        class_x = np.asarray(X_val[class_mask])

        profile_df = build_mean_spectrum_profile(class_x, feature_names=feature_names)
        profile_df.insert(0, "diagnosis", class_name)
        profile_df.insert(1, "n_samples", int(class_mask.sum()))
        profile_df.to_csv(class_dir / "mean_spectrum.csv", index=False)

        plot_mean_spectrum(
            class_x,
            class_dir / "mean_spectrum.png",
            feature_names=feature_names,
            title=f"{class_name} - Mean Spectrum",
            color=GROUP_COLORS.get(class_name, "#c0392b"),
        )

        peak_df = plot_peak_intensity_profile(
            class_x,
            class_dir / "peak_intensity_profile.png",
            feature_names=feature_names,
            title=f"{class_name} - Peak Intensity Profile",
            color=GROUP_COLORS.get(class_name, "#c0392b"),
            top_k=8,
        )
        peak_df.insert(0, "diagnosis", class_name)
        peak_df.insert(1, "n_samples", int(class_mask.sum()))
        peak_df.to_csv(class_dir / "peak_intensity_profile.csv", index=False)
        peak_rows.append(peak_df)

        reference_groups = [name for name in cancer_types if name != class_name and (y_val == cancer_types.index(name)).any()]
        if reference_groups:
            diff_df = plot_group_peak_difference(
                spectra_df,
                class_dir / "peak_difference_vs_rest.png",
                target_group=class_name,
                reference_groups=reference_groups,
                group_col="group",
                top_k=20,
                title=f"{class_name} vs Other Diagnoses Peak Difference",
                target_color=GROUP_COLORS.get(class_name, "#c0392b"),
            )
            diff_df.to_csv(class_dir / "peak_difference_vs_rest.csv", index=False)

        class_confusion_df = confusion_df[confusion_df["true_class"] == class_name].copy()
        if not class_confusion_df.empty:
            class_confusion_df.to_csv(class_dir / "confusion_classes.csv", index=False)
            plot_confusion_summary_bar(
                class_confusion_df,
                class_dir / "confusion_classes.png",
                title=f"{class_name} - Confused Classes",
                color=GROUP_COLORS.get(class_name, "#6c757d"),
            )

            for row in class_confusion_df.head(3).itertuples(index=False):
                confused_name = row.predicted_class
                confused_slug = sanitize_output_name(confused_name)
                confused_df = plot_group_peak_difference(
                    spectra_df,
                    class_dir / f"peak_difference_vs_confused_{confused_slug}.png",
                    target_group=class_name,
                    reference_groups=[confused_name],
                    group_col="group",
                    top_k=20,
                    title=f"{class_name} vs Confused {confused_name} Peak Difference",
                    target_color=GROUP_COLORS.get(class_name, "#c0392b"),
                    reference_color=GROUP_COLORS.get(confused_name, "#4c78a8"),
                )
                confused_df.insert(0, "confused_class", confused_name)
                confused_df.insert(1, "confusion_count", int(row.count))
                confused_df.insert(2, "confusion_rate", float(row.confusion_rate))
                confused_df.to_csv(
                    class_dir / f"peak_difference_vs_confused_{confused_slug}.csv",
                    index=False,
                )

        spectra_by_class.append(class_x)
        class_names.append(class_name)

    if spectra_by_class:
        overlay_df = plot_mean_spectra_overlay(
            spectra_by_class,
            class_names,
            out_dir / "mean_spectra_overlay.png",
            feature_names=feature_names,
            colors=[GROUP_COLORS.get(name, "#999999") for name in class_names],
            title="Stage 2 - Mean Spectra by Diagnosis",
        )
        overlay_df.to_csv(out_dir / "mean_spectra_overlay.csv", index=False)

    if peak_rows:
        peak_summary_df = pd.concat(peak_rows, ignore_index=True)
        peak_summary_df.to_csv(out_dir / "peak_intensity_by_diagnosis.csv", index=False)
        plot_peak_intensity_overview(
            peak_summary_df,
            out_dir / "peak_intensity_overview.png",
            title="Stage 2 - Top Peak Intensity by Diagnosis",
        )


def run_shap_s2(model, X_bg, X_exp, y_exp, cancer_types, device, out_dir, feature_names):
    logger.info("  SHAP Stage 2...")
    import torch.nn as nn

    class W(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
        def forward(self, x):
            return torch.softmax(self.base(x)["cancer_logits"], dim=-1)

    w = W(model).to(device)
    try:
        sv = compute_gradient_shap_values(w, X_bg, X_exp, device)
        sv_list = plot_multiclass_shap_summary(
            sv,
            X_exp,
            cancer_types,
            out_dir / "shap_summary.png",
            feature_names=feature_names,
            top_k=20,
            title="Stage 2 - SHAP per Cancer Type",
        )
        save_shap_outputs_by_diagnosis(
            sv_list,
            X_exp,
            y_exp,
            cancer_types,
            out_dir,
            feature_names,
            top_k=20,
        )
        np.save(
            out_dir / "shap_values_s2.npy",
            np.array(sv_list, dtype=object),
            allow_pickle=True,
        )
        logger.info(f"    Saved: {len(sv_list)} types")
    except Exception as e:
        logger.warning(f"    Failed: {e}")


# =============================================================================
# 6. t-SNE
# =============================================================================
def plot_tsne(emb, groups, bl, path):
    logger.info("  t-SNE...")
    try:
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(emb) - 1))
        co = tsne.fit_transform(emb)
    except Exception as exc:
        logger.warning(f"  t-SNE skipped: {exc}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for label, c, name in [(0, "#3498db", "Non-cancer"), (1, "#e74c3c", "Cancer")]:
        m = bl == label
        axes[0].scatter(co[m, 0], co[m, 1], c=c, s=15, alpha=0.5, label=f"{name} (n={m.sum()})")
    axes[0].set_title("t-SNE ??Binary"); axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.2)

    for g in sorted(set(groups)):
        m = groups == g
        axes[1].scatter(co[m, 0], co[m, 1], c=GROUP_COLORS.get(g, "#999"),
                        s=15, alpha=0.5, label=f"{g} (n={m.sum()})")
    axes[1].set_title("t-SNE ??All Groups (11)")
    axes[1].legend(fontsize=7, ncol=2); axes[1].grid(True, alpha=0.2)
    plt.tight_layout(); fig.savefig(path); plt.close()


# =============================================================================
# 7. Feature Selection & Grad-CAM
# =============================================================================
def run_feature_selection(X, y, out_dir, title, top_k=30):
    mi = mutual_info_classif(X, y, random_state=42)
    f_score, _ = f_classif(X, y)
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    rf_imp = rf.feature_importances_

    df = pd.DataFrame({
        "feature": [f"x_{i}" for i in range(X.shape[1])],
        "mi": mi,
        "f_score": f_score,
        "rf_importance": rf_imp,
    })

    for col in ["mi", "f_score", "rf_importance"]:
        df[f"rank_{col}"] = df[col].rank(ascending=False, method="min")
    df["rank_mean"] = df[["rank_mi", "rank_f_score", "rank_rf_importance"]].mean(axis=1)
    df = df.sort_values("rank_mean").reset_index(drop=True)
    df.to_csv(out_dir / "feature_scores.csv", index=False)

    top_df = df.head(min(top_k, len(df))).sort_values("rank_mean", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(top_df["feature"], top_df["rank_mean"], color="#5dade2")
    ax.set_title(f"{title} ??Top Features (ensemble rank)")
    ax.set_xlabel("Mean rank (lower is better)")
    plt.tight_layout(); fig.savefig(out_dir / "top_features.png"); plt.close()
    return df


def run_one_vs_rest_feature_selection(X, y_multiclass, names, out_dir, top_k=20):
    rows = []
    for idx, name in enumerate(names):
        y_bin = (y_multiclass == idx).astype(int)
        if y_bin.sum() < 5:
            continue
        class_dir = out_dir / f"ovr_{name}"
        class_dir.mkdir(parents=True, exist_ok=True)
        df = run_feature_selection(X, y_bin, class_dir, f"{name} vs Rest", top_k=top_k)
        best = df.iloc[0]
        rows.append({"class": name, "best_feature": best["feature"], "best_rank_mean": best["rank_mean"]})
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "ovr_feature_summary.csv", index=False)


def resolve_gradcam_target_layer(model):
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        return None
    if hasattr(encoder, "layer4"):
        return encoder.layer4
    if hasattr(encoder, "features") and len(encoder.features) >= 9:
        return encoder.features[8]
    return None


def run_gradcam_1d(
    model,
    X,
    out_dir,
    stage="binary",
    class_idx=None,
    class_name=None,
    max_samples=128,
):
    n_samples = min(max_samples, len(X))
    if n_samples == 0:
        return
    X = X[:n_samples]
    device = next(model.parameters()).device
    layer = resolve_gradcam_target_layer(model)
    if layer is None:
        logger.warning("  Grad-CAM skipped: no compatible encoder layer found")
        return

    activations, gradients = [], []

    def f_hook(_, __, output):
        activations.append(output)

    def b_hook(_, __, grad_output):
        gradients.append(grad_output[0])

    h1 = layer.register_forward_hook(f_hook)
    h2 = layer.register_full_backward_hook(b_hook)

    cams = []
    model.eval()
    for i in range(0, n_samples, 32):
        xb = torch.FloatTensor(X[i:i+32]).to(device)
        model.zero_grad(set_to_none=True)
        out = model(xb)
        if stage == "binary":
            score = out["binary_logit"].squeeze(-1)
            title = "Grad-CAM (Binary: Cancer vs Non-cancer)"
            out_name = "gradcam_binary.png"
        else:
            score = out["cancer_logits"][:, class_idx]
            resolved_name = class_name or f"class_{class_idx}"
            slug = sanitize_output_name(resolved_name)
            title = f"Grad-CAM ({resolved_name})"
            out_name = f"gradcam_{slug}.png"
        score.sum().backward()

        act = activations.pop()
        grad = gradients.pop()
        w = grad.mean(dim=2, keepdim=True)
        cam = torch.relu((w * act).sum(dim=1))
        cam = F.interpolate(cam.unsqueeze(1), size=X.shape[1], mode="linear", align_corners=False).squeeze(1)
        cams.append(cam.detach().cpu().numpy())

    h1.remove(); h2.remove()
    cam_mean = np.concatenate(cams, axis=0).mean(axis=0)
    if cam_mean.max() > 0:
        cam_mean = cam_mean / cam_mean.max()
    np.save(out_dir / out_name.replace(".png", ".npy"), cam_mean)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.arange(len(cam_mean)), cam_mean, color="#e74c3c", lw=2)
    ax.set_title(title)
    ax.set_xlabel("Feature index (x_i)")
    ax.set_ylabel("Normalized importance")
    ax.grid(True, alpha=0.3)
    plt.tight_layout(); fig.savefig(out_dir / out_name); plt.close()


# =============================================================================
# 8. Summary Table
# =============================================================================
def summary_table(data, summary, opt_t, out_dir):
    ct = summary["cancer_types"]
    bl, ctl = data["binary_labels"], data["cancer_type_labels"]

    valid = ~np.isnan(data["val_binary_prob"])
    val_m = binary_metrics(bl[valid], data["val_binary_prob"][valid], opt_t, "val_s1_")
    cm = ctl[valid] >= 0
    if cm.sum() > 0:
        val_m.update(multiclass_metrics(ctl[valid][cm], data["val_cancer_logits"][valid][cm], ct, "val_s2_"))

    train_m = {}
    if "train_binary_prob" in data and data["train_binary_prob"] is not None:
        train_m = binary_metrics(data["train_binary_true"], data["train_binary_prob"], opt_t, "train_s1_")
        tct = data.get("train_cancer_true")
        tcl = data.get("train_cancer_logits")
        if tct is not None:
            tcm = tct >= 0
            if tcm.sum() > 0:
                train_m.update(multiclass_metrics(tct[tcm], tcl[tcm], ct, "train_s2_"))

    rows = []
    for m in ["auc", "pr_auc", "accuracy", "sensitivity", "specificity", "ppv", "npv", "f1"]:
        rows.append({"stage": "S1", "metric": m, "train": train_m.get(f"train_s1_{m}", np.nan), "val": val_m.get(f"val_s1_{m}", np.nan)})
    for m in ["accuracy", "f1_macro", "auc_macro", "pr_auc_macro"]:
        rows.append({"stage": "S2", "metric": m, "train": train_m.get(f"train_s2_{m}", np.nan), "val": val_m.get(f"val_s2_{m}", np.nan)})
    for name in ct:
        for m in ["sens", "auc", "pr_auc"]:
            rows.append({"stage": f"S2-{name}", "metric": m,
                          "train": train_m.get(f"train_s2_{name}_{m}", np.nan),
                          "val": val_m.get(f"val_s2_{name}_{m}", np.nan)})

    df = pd.DataFrame(rows)
    df["gap"] = df["train"] - df["val"]
    df.to_csv(out_dir / "metrics_summary.csv", index=False)
    return df


# =============================================================================
# 9. Main
# =============================================================================
def _version_sort_key(path: Path):
    match = re.fullmatch(r"v(\d+)", path.name.lower())
    return int(match.group(1)) if match else -1


def resolve_train_dir(input_path: str | Path) -> Path:
    train_dir = Path(input_path)
    latest_file = train_dir / "latest.txt"
    if latest_file.exists():
        version_name = latest_file.read_text(encoding="utf-8").strip()
        candidate = train_dir / version_name
        if (candidate / "fold_predictions.npz").exists():
            logger.info(f"Resolved latest version: {candidate}")
            return candidate

    if (train_dir / "fold_predictions.npz").exists():
        return train_dir

    version_dirs = sorted(
        [p for p in train_dir.glob("v*") if p.is_dir() and (p / "fold_predictions.npz").exists()],
        key=_version_sort_key,
    )
    if version_dirs:
        logger.info(f"Resolved latest version: {version_dirs[-1]}")
        return version_dirs[-1]

    model_runs = []
    for child in train_dir.iterdir() if train_dir.exists() else []:
        if not child.is_dir():
            continue
        latest_file = child / "latest.txt"
        if latest_file.exists():
            version_name = latest_file.read_text(encoding="utf-8").strip()
            candidate = child / version_name
            if (candidate / "fold_predictions.npz").exists():
                model_runs.append(candidate)
        elif (child / "fold_predictions.npz").exists():
            model_runs.append(child)

    if len(model_runs) == 1:
        logger.info(f"Resolved single model run: {model_runs[0]}")
        return model_runs[0]
    if len(model_runs) > 1:
        raise FileNotFoundError(
            f"Multiple model runs found under {train_dir}. Please specify a model directory explicitly."
        )

    raise FileNotFoundError(f"No fold_predictions.npz found under {train_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="SERS ResNet18-1D Evaluation")
    p.add_argument("--input", "-i", default="models/results/_archive/resnet18_medoid_v1")
    p.add_argument("--processed-csv", default="results/processed_spectra.csv")
    p.add_argument("--device", default="auto")
    p.add_argument("--tsne-dim", type=int, choices=[2, 3], default=3)
    p.add_argument("--no-shap", action="store_true")
    p.add_argument("--no-feature-selection", action="store_true")
    p.add_argument("--no-gradcam", action="store_true")
    p.add_argument("--top-k-features", type=int, default=30)
    p.add_argument("--gradcam-samples", type=int, default=128)
    p.add_argument("--shap-samples", type=int, default=100)
    p.add_argument("--shap-explain", type=int, default=200)

    # ── External group inference ──
    p.add_argument(
        "--val-group",
        type=str,
        default=None,
        help="Pass an out-of-training group (e.g. SPAN) through the trained fold models "
             "and report classification results. Requires fold checkpoints.",
    )
    p.add_argument(
        "--aggregate", "-a",
        choices=["medoid", "mean", "none"],
        default="mean",
        help="Replicate aggregation for --val-group samples (default: mean).",
    )
    return p.parse_args()


def create_output_dirs(eval_dir: Path):
    s1_dir, s2_dir = eval_dir / "stage1", eval_dir / "stage2"
    emb_dir, met_dir = eval_dir / "embedding", eval_dir / "metrics"
    for d in [s1_dir, s2_dir, emb_dir, met_dir]:
        os.makedirs(d, exist_ok=True)
    return s1_dir, s2_dir, emb_dir, met_dir


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )


def run_stage1(vbt, vbp, vgrp, s1_dir, met_dir, tbp=None, tbt=None):
    logger.info("\n[Stage 1] Binary classification...")
    opt_t = plot_roc_s1(vbt, vbp, s1_dir / "roc_curve.png")
    logger.info(f"  Optimal threshold: {opt_t:.4f}")
    plot_pr_s1(vbt, vbp, s1_dir / "pr_curve.png")
    plot_cm_s1(vbt, vbp, opt_t, s1_dir / "confusion_matrix.png")
    plot_dist(vbt, vbp, opt_t, s1_dir / "prob_distribution.png")

    if tbp is not None:
        plot_train_val_roc(
            tbt,
            tbp,
            vbt,
            vbp,
            "Stage 1 ??Train vs Val (Overfitting Check)",
            s1_dir / "train_vs_val_roc.png",
        )

    pg = per_group_metrics(vbt, vbp, vgrp, opt_t)
    pg.to_csv(met_dir / "per_group_metrics.csv", index=False)
    logger.info(f"\n{pg.to_string(index=False)}")
    return opt_t


def run_stage2(vct, vcl, vX, tct, tcl, cancer_types, s2_dir, met_dir, feature_names):
    logger.info("\n[Stage 2] Cancer type (8-class)...")
    cm = vct >= 0
    if cm.sum() == 0:
        return cm

    plot_roc_s2(vct[cm], vcl[cm], cancer_types, s2_dir / "roc_curves_per_type.png")
    plot_pr_s2(vct[cm], vcl[cm], cancer_types, s2_dir / "pr_curves_per_type.png")
    plot_cm_s2(vct[cm], vcl[cm], cancer_types, s2_dir / "confusion_matrix.png")
    type_df = plot_bars_s2(vct[cm], vcl[cm], cancer_types, s2_dir / "per_type_metrics.png")
    type_df.to_csv(met_dir / "per_cancer_type_metrics.csv", index=False)
    logger.info(f"\n{type_df.to_string(index=False)}")
    y_prob = torch.softmax(torch.tensor(vcl[cm]), dim=-1).numpy()
    y_pred = y_prob.argmax(axis=1)
    save_mean_spectra_by_diagnosis(vX[cm], vct[cm], y_pred, cancer_types, s2_dir, feature_names)

    if tct is not None:
        tcm_mask = tct >= 0
        if tcm_mask.sum() > 0:
            plot_train_val_s2(
                tct[tcm_mask],
                tcl[tcm_mask],
                vct[cm],
                vcl[cm],
                cancer_types,
                s2_dir / "train_vs_val_roc.png",
            )
    return cm


def save_threshold_sweep(vbt, vbp, met_dir):
    thresh_rows = []
    for t in np.arange(0.3, 0.8, 0.05):
        m = binary_metrics(vbt, vbp, t)
        m["threshold"] = t
        thresh_rows.append(m)
    pd.DataFrame(thresh_rows).to_csv(met_dir / "threshold_sweep.csv", index=False)


def load_trained_model(train_dir, device, summary):
    model_name = str(summary.get("model_name", "resnet18")).lower()
    if model_name in {"xgboost", "logistic_regression", "random_forest"}:
        logger.info(f"  SHAP model loading skipped for non-torch model: {model_name}")
        return None

    ckpt_files = sorted((train_dir / "checkpoints").glob("fold_*.pt"))
    if not ckpt_files:
        return None

    ckpt = torch.load(ckpt_files[-1], map_location=device, weights_only=False)
    mc = ModelConfig(**ckpt["config"])
    ckpt_model_name = str(ckpt.get("model_name", model_name)).lower()
    model = build_model(ckpt_model_name, mc).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def run_shap_analysis(args, train_dir, device, X, valid_mask, ctl, s1_dir, s2_dir, cancer_types, summary):
    if args.no_shap:
        logger.info("\n[SHAP] Skipped (--no-shap)")
        return

    logger.info("\n[SHAP] Computing...")
    model = load_trained_model(train_dir, device, summary)
    if model is None:
        logger.warning("  No torch checkpoints available for SHAP")
        return

    X_bg, X_exp, exp_idx = sample_shap_inputs(
        X,
        valid_mask,
        args.shap_samples,
        args.shap_explain,
    )
    if len(exp_idx) == 0:
        logger.warning("  No validation samples available for SHAP")
        return

    feature_names = resolve_feature_names(args.processed_csv, X.shape[1])

    run_shap_s1(model, X_bg, X_exp, device, s1_dir, feature_names)

    cancer_mask = ctl[exp_idx] >= 0
    if cancer_mask.sum() <= 1:
        logger.warning("  Not enough cancer samples for Stage 2 SHAP")
        return

    run_shap_s2(
        model,
        X_bg,
        X_exp[cancer_mask],
        ctl[exp_idx][cancer_mask],
        cancer_types,
        device,
        s2_dir,
        feature_names,
    )


def _aggregate_group(df, feature_cols, method):
    """Aggregate replicates for val-group inference."""
    if method == "none":
        return df
    if method == "mean":
        agg = df.groupby(["group", "sample_id"])[feature_cols].mean().reset_index()
        logger.info(f"  Mean aggregation: {len(df)} spectra → {len(agg)} samples")
        return agg
    # medoid
    rows = []
    for (group, sid), sub in df.groupby(["group", "sample_id"]):
        if len(sub) == 1:
            rows.append(sub.iloc[0])
            continue
        spectra = sub[feature_cols].values
        medoid_idx = np.corrcoef(spectra).mean(axis=1).argmax()
        rows.append(sub.iloc[medoid_idx])
    result = pd.DataFrame(rows).reset_index(drop=True)
    logger.info(f"  Medoid aggregation: {len(df)} spectra → {len(result)} samples")
    return result


def run_val_group_inference(args):
    """Load fold checkpoints and run inference on an out-of-training group."""
    t0 = datetime.now()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-7s │ %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    group_name = args.val_group.upper()
    logger.info("=" * 64)
    logger.info(f"  Val-Group Inference: {group_name}")
    logger.info("=" * 64)

    # ── 1. Resolve training dir & load summary ──
    train_dir = resolve_train_dir(args.input)
    with open(train_dir / "training_summary.json") as f:
        summary = json.load(f)
    cancer_types = summary["cancer_types"]
    model_name = summary.get("model_name", "unknown")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Cancer types: {cancer_types}")
    logger.info(f"  Train dir: {train_dir}")

    # ── 2. Load processed spectra for the target group ──
    csv_path = Path(args.processed_csv)
    if not csv_path.is_absolute():
        csv_path = PROJECT_ROOT / csv_path
    df_all = pd.read_csv(csv_path)
    feature_cols = [c for c in df_all.columns if c.startswith("x_")]

    # Support group aliases (e.g., PAN → CPAN + YPAN)
    group_aliases = summary.get("group_aliases", {})
    raw_groups = [group_name]
    if group_name in group_aliases:
        raw_groups = group_aliases[group_name]
        logger.info(f"  Expanded alias {group_name} → {raw_groups}")

    df_group = df_all[df_all["group"].isin(raw_groups)].copy()
    if df_group.empty:
        logger.error(f"  No spectra found for group(s) {raw_groups} in {csv_path}")
        return 1

    logger.info(f"  Found {len(df_group)} spectra, "
                f"{df_group['sample_id'].nunique()} samples for {raw_groups}")

    # ── 3. Aggregate replicates ──
    df_agg = _aggregate_group(df_group, feature_cols, args.aggregate)
    X = df_agg[feature_cols].values.astype(np.float32)
    sample_ids = df_agg["sample_id"].values
    groups = df_agg["group"].values if "group" in df_agg.columns else np.array([group_name] * len(X))
    n_samples, n_features = X.shape
    logger.info(f"  Inference matrix: {n_samples} samples × {n_features} features")

    # ── 4. Load fold checkpoints ──
    ckpt_dir = train_dir / "checkpoints"
    if not ckpt_dir.exists():
        logger.error(f"  Checkpoint dir not found: {ckpt_dir}")
        return 1

    binary_models = sorted(ckpt_dir.glob("fold_*_binary.joblib"))
    stage2_models = sorted(ckpt_dir.glob("fold_*_stage2.joblib"))
    torch_models = sorted(ckpt_dir.glob("fold_*.pt"))

    is_classical = len(binary_models) > 0
    is_torch = len(torch_models) > 0 and not is_classical

    if not is_classical and not is_torch:
        logger.error("  No fold checkpoints found (need fold_*_binary.joblib or fold_*.pt)")
        return 1

    # ── 5. Run inference ──
    all_binary_probs = []
    all_stage2_logits = []

    if is_classical:
        logger.info(f"\n  Loading {len(binary_models)} classical fold models...")
        # StandardScaler: train.py wraps LR in make_pipeline(StandardScaler(), LR)
        for i, bp in enumerate(binary_models):
            bm = joblib.load(bp)
            if hasattr(bm, "predict_proba"):
                prob = bm.predict_proba(X)[:, 1]
            else:
                prob = bm.decision_function(X)
            all_binary_probs.append(prob)

            sp = ckpt_dir / f"fold_{i}_stage2.joblib"
            if sp.exists():
                sm = joblib.load(sp)
                if hasattr(sm, "predict_proba"):
                    logits = sm.predict_proba(X)
                else:
                    logits = sm.decision_function(X)
                all_stage2_logits.append(logits)
            logger.info(f"    Fold {i}: S1 mean_prob={prob.mean():.3f}")

    elif is_torch:
        device = resolve_device(args.device)
        logger.info(f"\n  Loading {len(torch_models)} torch fold models on {device}...")
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(device)

        for ckpt_path in torch_models:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            mc = ModelConfig(**ckpt["config"])
            ckpt_model_name = str(ckpt.get("model_name", model_name)).lower()
            model = build_model(ckpt_model_name, mc).to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            with torch.no_grad():
                out = model(X_tensor)
                bp = torch.sigmoid(out["binary_logit"]).cpu().numpy().squeeze()
                cl = out["cancer_logits"].cpu().numpy()
            all_binary_probs.append(bp)
            all_stage2_logits.append(cl)
            logger.info(f"    {ckpt_path.name}: S1 mean_prob={bp.mean():.3f}")

    # ── 6. Ensemble average ──
    binary_prob = np.mean(all_binary_probs, axis=0)
    binary_std = np.std(all_binary_probs, axis=0)

    if all_stage2_logits:
        stage2_logits = np.mean(all_stage2_logits, axis=0)
        stage2_prob = torch.softmax(torch.tensor(stage2_logits), dim=-1).numpy()
        stage2_pred = stage2_prob.argmax(axis=1)
        stage2_pred_names = [cancer_types[i] if i < len(cancer_types) else f"cls_{i}"
                            for i in stage2_pred]
    else:
        stage2_prob = None
        stage2_pred_names = ["N/A"] * n_samples

    # ── 7. Output ──
    logger.info(f"\n{'=' * 64}")
    logger.info(f"  Results: {group_name} ({n_samples} samples) through {model_name}")
    logger.info(f"{'=' * 64}")

    # Stage 1: Cancer probability
    logger.info(f"\n[Stage 1] Cancer Screening Probability")
    logger.info(f"  Mean cancer prob: {binary_prob.mean():.4f} ± {binary_prob.std():.4f}")
    for thresh in [0.3, 0.5, 0.7]:
        n_pos = (binary_prob > thresh).sum()
        logger.info(f"  Threshold {thresh:.1f}: {n_pos}/{n_samples} classified as cancer "
                    f"({n_pos/n_samples*100:.1f}%)")

    # Stage 2: Cancer type classification
    if stage2_prob is not None:
        logger.info(f"\n[Stage 2] Cancer Type Classification")
        pred_counts = pd.Series(stage2_pred_names).value_counts()
        for ct, count in pred_counts.items():
            logger.info(f"  {ct}: {count}/{n_samples} ({count/n_samples*100:.1f}%)")

        # Mean probability per cancer type
        logger.info(f"\n  Mean class probabilities:")
        for j, ct in enumerate(cancer_types):
            if j < stage2_prob.shape[1]:
                logger.info(f"    {ct}: {stage2_prob[:, j].mean():.4f} ± {stage2_prob[:, j].std():.4f}")

    # ── 8. Save per-sample results ──
    from src.sers.config import FIG_DIR, training_dir_to_figure_slug
    fig_slug = training_dir_to_figure_slug(train_dir)
    eval_dir = FIG_DIR / "training" / fig_slug / f"val_group_{group_name.lower()}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    result_rows = []
    for i in range(n_samples):
        row = {
            "group": groups[i] if i < len(groups) else group_name,
            "sample_id": sample_ids[i],
            "cancer_prob": binary_prob[i],
            "cancer_prob_std": binary_std[i],
            "predicted_type": stage2_pred_names[i],
        }
        if stage2_prob is not None:
            for j, ct in enumerate(cancer_types):
                if j < stage2_prob.shape[1]:
                    row[f"prob_{ct}"] = stage2_prob[i, j]
        result_rows.append(row)
    df_result = pd.DataFrame(result_rows)
    result_csv = eval_dir / "predictions.csv"
    df_result.to_csv(result_csv, index=False)
    logger.info(f"\n  Per-sample predictions saved to: {result_csv}")

    # ── 9. Visualizations ──
    # Cancer probability distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(binary_prob, bins=30, color="#E53935", alpha=0.7, edgecolor="black")
    ax.axvline(0.5, color="black", ls="--", lw=1.5, label="Threshold 0.5")
    ax.set(xlabel="Cancer Probability", ylabel="Count",
           title=f"{group_name} — Stage 1 Cancer Probability Distribution (n={n_samples})")
    ax.legend()
    fig.savefig(eval_dir / "cancer_prob_distribution.png")
    plt.close(fig)

    # Stage 2 prediction pie chart
    if stage2_prob is not None:
        fig, ax = plt.subplots(figsize=(7, 7))
        pred_counts = pd.Series(stage2_pred_names).value_counts()
        colors = [GROUP_COLORS.get(ct, "#999999") for ct in pred_counts.index]
        ax.pie(pred_counts.values, labels=[f"{ct}\n({c})" for ct, c in pred_counts.items()],
               colors=colors, autopct="%1.1f%%", startangle=90)
        ax.set_title(f"{group_name} — Predicted Cancer Types (n={n_samples})")
        fig.savefig(eval_dir / "predicted_types_pie.png")
        plt.close(fig)

        # Heatmap of per-sample class probabilities
        fig, ax = plt.subplots(figsize=(max(8, len(cancer_types)*1.2), max(6, n_samples*0.15)))
        im = ax.imshow(stage2_prob, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks(range(len(cancer_types)))
        ax.set_xticklabels(cancer_types, rotation=45, ha="right")
        ax.set_ylabel("Sample")
        ax.set_title(f"{group_name} — Stage 2 Class Probabilities")
        fig.colorbar(im, ax=ax, label="Probability")
        fig.savefig(eval_dir / "class_probability_heatmap.png")
        plt.close(fig)

    elapsed = datetime.now() - t0
    logger.info(f"\n{'=' * 64}")
    logger.info(f"  Val-group inference complete! ({elapsed})")
    logger.info(f"  Output: {eval_dir}/")
    logger.info(f"{'=' * 64}")
    return 0


def main():
    args = parse_args()

    # ── Val-group mode: run separate inference pipeline ──
    if args.val_group:
        return run_val_group_inference(args)

    t0 = datetime.now()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-7s │ %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler("test.log", mode="w", encoding="utf-8")],
    )

    logger.info("=" * 64)
    logger.info("  SERS ResNet18-1D Evaluation")
    logger.info("=" * 64)

    train_dir = resolve_train_dir(args.input)
    from src.sers.config import FIG_DIR, training_dir_to_figure_slug
    fig_slug = training_dir_to_figure_slug(train_dir)
    eval_dir = FIG_DIR / "training" / fig_slug
    s1_dir, s2_dir, emb_dir, met_dir = create_output_dirs(eval_dir)
    device = resolve_device(args.device)

    data, summary = load_outputs(train_dir)
    cancer_types = summary["cancer_types"]
    bl, ctl = data["binary_labels"], data["cancer_type_labels"]
    groups, X = data["groups"], data["X"]

    valid = ~np.isnan(data["val_binary_prob"])
    vbp, vbt = data["val_binary_prob"][valid], bl[valid]
    vcl, vct = data["val_cancer_logits"][valid], ctl[valid]
    vX = X[valid]
    vemb = data["val_embedding"][valid]
    vgrp = groups[valid]

    tbp = data.get("train_binary_prob")
    tbt = data.get("train_binary_true")
    tcl = data.get("train_cancer_logits")
    tct = data.get("train_cancer_true")

    feature_names = resolve_feature_names(args.processed_csv, X.shape[1])

    opt_t = run_stage1(vbt, vbp, vgrp, s1_dir, met_dir, tbp=tbp, tbt=tbt)
    cm = run_stage2(vct, vcl, vX, tct, tcl, cancer_types, s2_dir, met_dir, feature_names)

    # SHAP
    run_shap_analysis(args, train_dir, device, X, valid, ctl, s1_dir, s2_dir, cancer_types, summary)

    if not args.no_gradcam:
        model = load_trained_model(train_dir, device, summary)
        if model is None:
            logger.info("\n[Grad-CAM] Skipped (no torch model/checkpoint)")
        else:
            logger.info("\n[Grad-CAM] Computing saliency maps...")
            run_gradcam_1d(model, X[valid], s1_dir, stage="binary", max_samples=args.gradcam_samples)
            if cm.sum() > 0:
                for idx in sorted(set(vct[cm])):
                    class_name = cancer_types[int(idx)] if int(idx) < len(cancer_types) else f"class_{idx}"
                    run_gradcam_1d(
                        model,
                        X[valid][cm][vct[cm] == idx],
                        s2_dir,
                        stage="multiclass",
                        class_idx=int(idx),
                        class_name=class_name,
                        max_samples=args.gradcam_samples,
                    )
    else:
        logger.info("\n[Grad-CAM] Skipped (--no-gradcam)")
    elapsed = datetime.now() - t0
    logger.info(f"\n{'=' * 64}")
    logger.info(f"  Evaluation complete! ({elapsed})")
    logger.info(f"  Output: {eval_dir}/")
    logger.info(f"{'=' * 64}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

