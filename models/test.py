"""
SERS Cancer Detection — Evaluation & Visualization (ResNet18-1D)

Two-stage: Binary + Cancer Type (7: PRO, BRE, OVA, LUN, CRC, CPAN, SPAN)

Usage:
    python test.py
    python test.py -i results/training --no-shap

Output (results/training/evaluation/):
    metrics/   — summary, per-group, threshold sweep
    stage1/    — ROC, confusion, distribution, train vs val, SHAP
    stage2/    — per-type ROC, confusion, bar chart, train vs val, SHAP
    embedding/ — t-SNE
"""

from __future__ import annotations

import sys
import argparse
import logging
import os
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, auc,
    confusion_matrix,
)
from sklearn.manifold import TSNE

import warnings
warnings.filterwarnings("ignore")

from model import ModelConfig, SERSCancerDetector

logger = logging.getLogger(__name__)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 13, "axes.labelsize": 11,
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
               zorder=5, label=f"Optimal (J={j:.3f}, τ={opt_t:.3f})")
    ax.set(xlabel="1 − Specificity", ylabel="Sensitivity",
           title="Stage 1: Cancer vs Non-cancer (ResNet18-1D)")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right", fontsize=11); ax.grid(True, alpha=0.3)
    ax.text(0.55, 0.12, f"τ={opt_t:.3f}\nSens={tpr[oi]:.3f}\nSpec={1-fpr[oi]:.3f}",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9))
    plt.tight_layout(); fig.savefig(path); plt.close()
    return opt_t


def plot_cm_s1(yt, yp, t, path):
    ypr = (yp > t).astype(int)
    cm = confusion_matrix(yt, ypr, labels=[0, 1])
    cn = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    labels = ["Non-cancer", "Cancer"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=labels, yticklabels=labels, cbar=False, annot_kws={"size": 16})
    axes[0].set(xlabel="Predicted", ylabel="True", title=f"Counts (τ={t:.3f})")
    sns.heatmap(cn, annot=True, fmt=".2%", cmap="Blues", ax=axes[1],
                xticklabels=labels, yticklabels=labels, cbar=False, annot_kws={"size": 16}, vmin=0, vmax=1)
    axes[1].set(xlabel="Predicted", ylabel="True", title="Normalized")
    plt.suptitle("Stage 1 — Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout(); fig.savefig(path); plt.close()


def plot_dist(yt, yp, t, path):
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, 1, 60)
    ax.hist(yp[yt == 0], bins=bins, alpha=0.55, color="#3498db",
            label=f"Non-cancer (n={(yt==0).sum()})", density=True, edgecolor="white")
    ax.hist(yp[yt == 1], bins=bins, alpha=0.55, color="#e74c3c",
            label=f"Cancer (n={(yt==1).sum()})", density=True, edgecolor="white")
    ax.axvline(t, color="black", ls="--", lw=1.5, label=f"τ={t:.3f}")
    ax.set(xlabel="P(Cancer)", ylabel="Density", title="Stage 1 — Prediction Distribution")
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
    ax.set(xlabel="FPR", ylabel="TPR", title="Stage 2 — Per-Cancer ROC (OvR)")
    ax.legend(fontsize=9, loc="lower right"); ax.grid(True, alpha=0.3)
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
    plt.suptitle("Stage 2 — Cancer Type Confusion (7-class)", fontsize=14, fontweight="bold")
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
    ax.set(ylabel="Score", title="Stage 2 — Per-Type Metrics"); ax.legend()
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
    fig.suptitle("Stage 2 — Train vs Val per Cancer Type", fontsize=14, fontweight="bold")
    plt.tight_layout(); fig.savefig(path); plt.close()


# =============================================================================
# 5. SHAP
# =============================================================================
def run_shap_s1(model, X_bg, X_exp, device, out_dir):
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed"); return

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
        e = shap.GradientExplainer(w, torch.FloatTensor(X_bg).to(device))
        sv = e.shap_values(torch.FloatTensor(X_exp).to(device))
        if isinstance(sv, list): sv = sv[0]
        sv = np.array(sv)

        mean_abs = np.abs(sv).mean(axis=0)
        top_k = min(30, len(mean_abs))
        top_idx = np.argsort(mean_abs)[-top_k:][::-1]
        shap.summary_plot(sv[:, top_idx], X_exp[:, top_idx],
                          feature_names=[f"x_{i}" for i in top_idx],
                          show=False, max_display=top_k)
        plt.title("Stage 1 — SHAP (Top 30 Wavenumbers)")
        plt.tight_layout()
        plt.savefig(out_dir / "shap_summary.png", dpi=150); plt.close()
        np.save(out_dir / "shap_values_s1.npy", sv)
        logger.info(f"    Saved: {sv.shape}")
    except Exception as e:
        logger.warning(f"    Failed: {e}")


def run_shap_s2(model, X_bg, X_exp, cancer_types, device, out_dir):
    try:
        import shap
    except ImportError:
        return

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
        e = shap.GradientExplainer(w, torch.FloatTensor(X_bg).to(device))
        sv = e.shap_values(torch.FloatTensor(X_exp).to(device))
        if isinstance(sv, list):
            nc = len(sv)
        else:
            nc = sv.shape[-1]; sv = [sv[:, :, c] for c in range(nc)]

        nt = min(nc, len(cancer_types))
        cols = min(4, nt); rows_n = (nt + cols - 1) // cols
        fig, axes = plt.subplots(rows_n, cols, figsize=(5 * cols, 4.5 * rows_n))
        axes = np.array(axes).flatten() if nt > 1 else np.array([axes])

        for c in range(nt):
            plt.sca(axes[c])
            s = np.array(sv[c])
            ma = np.abs(s).mean(axis=0)
            tk = min(20, len(ma))
            ti = np.argsort(ma)[-tk:][::-1]
            shap.summary_plot(s[:, ti], X_exp[:, ti],
                              feature_names=[f"x_{i}" for i in ti],
                              show=False, max_display=tk, plot_size=None)
            axes[c].set_title(cancer_types[c])
        for j in range(nt, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle("Stage 2 — SHAP per Cancer Type", fontsize=14, fontweight="bold")
        plt.tight_layout()
        fig.savefig(out_dir / "shap_summary.png", dpi=150); plt.close()
        logger.info(f"    Saved: {nt} types")
    except Exception as e:
        logger.warning(f"    Failed: {e}")


# =============================================================================
# 6. t-SNE
# =============================================================================
def plot_tsne(emb, groups, bl, path):
    logger.info("  t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(emb) - 1))
    co = tsne.fit_transform(emb)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for label, c, name in [(0, "#3498db", "Non-cancer"), (1, "#e74c3c", "Cancer")]:
        m = bl == label
        axes[0].scatter(co[m, 0], co[m, 1], c=c, s=15, alpha=0.5, label=f"{name} (n={m.sum()})")
    axes[0].set_title("t-SNE — Binary"); axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.2)

    for g in sorted(set(groups)):
        m = groups == g
        axes[1].scatter(co[m, 0], co[m, 1], c=GROUP_COLORS.get(g, "#999"),
                        s=15, alpha=0.5, label=f"{g} (n={m.sum()})")
    axes[1].set_title("t-SNE — All Groups (11)")
    axes[1].legend(fontsize=7, ncol=2); axes[1].grid(True, alpha=0.2)
    plt.tight_layout(); fig.savefig(path); plt.close()


# =============================================================================
# 7. Summary Table
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
    for m in ["auc", "accuracy", "sensitivity", "specificity", "ppv", "npv", "f1"]:
        rows.append({"stage": "S1", "metric": m, "train": train_m.get(f"train_s1_{m}", np.nan), "val": val_m.get(f"val_s1_{m}", np.nan)})
    for m in ["accuracy", "f1_macro", "auc_macro"]:
        rows.append({"stage": "S2", "metric": m, "train": train_m.get(f"train_s2_{m}", np.nan), "val": val_m.get(f"val_s2_{m}", np.nan)})
    for name in ct:
        for m in ["sens", "auc"]:
            rows.append({"stage": f"S2-{name}", "metric": m,
                          "train": train_m.get(f"train_s2_{name}_{m}", np.nan),
                          "val": val_m.get(f"val_s2_{name}_{m}", np.nan)})

    df = pd.DataFrame(rows)
    df["gap"] = df["train"] - df["val"]
    df.to_csv(out_dir / "metrics_summary.csv", index=False)
    return df


# =============================================================================
# 8. Main
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="SERS ResNet18-1D Evaluation")
    p.add_argument("--input", "-i", default="results/training")
    p.add_argument("--device", default="auto")
    p.add_argument("--no-shap", action="store_true")
    p.add_argument("--shap-samples", type=int, default=100)
    p.add_argument("--shap-explain", type=int, default=200)
    return p.parse_args()


def main():
    args = parse_args()
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

    train_dir = Path(args.input)
    eval_dir = train_dir / "evaluation"
    s1_dir, s2_dir = eval_dir / "stage1", eval_dir / "stage2"
    emb_dir, met_dir = eval_dir / "embedding", eval_dir / "metrics"
    for d in [s1_dir, s2_dir, emb_dir, met_dir]:
        os.makedirs(d, exist_ok=True)

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = torch.device(args.device)

    data, summary = load_outputs(train_dir)
    cancer_types = summary["cancer_types"]
    bl, ctl = data["binary_labels"], data["cancer_type_labels"]
    groups, X = data["groups"], data["X"]

    valid = ~np.isnan(data["val_binary_prob"])
    vbp, vbt = data["val_binary_prob"][valid], bl[valid]
    vcl, vct = data["val_cancer_logits"][valid], ctl[valid]
    vemb = data["val_embedding"][valid]
    vgrp = groups[valid]

    tbp = data.get("train_binary_prob")
    tbt = data.get("train_binary_true")
    tcl = data.get("train_cancer_logits")
    tct = data.get("train_cancer_true")

    # ── Stage 1 ──
    logger.info("\n[Stage 1] Binary classification...")
    opt_t = plot_roc_s1(vbt, vbp, s1_dir / "roc_curve.png")
    logger.info(f"  Optimal threshold: {opt_t:.4f}")
    plot_cm_s1(vbt, vbp, opt_t, s1_dir / "confusion_matrix.png")
    plot_dist(vbt, vbp, opt_t, s1_dir / "prob_distribution.png")
    if tbp is not None:
        plot_train_val_roc(tbt, tbp, vbt, vbp,
                            "Stage 1 — Train vs Val (Overfitting Check)", s1_dir / "train_vs_val_roc.png")

    pg = per_group_metrics(vbt, vbp, vgrp, opt_t)
    pg.to_csv(met_dir / "per_group_metrics.csv", index=False)
    logger.info(f"\n{pg.to_string(index=False)}")

    # ── Stage 2 ──
    logger.info("\n[Stage 2] Cancer type (7-class)...")
    cm = vct >= 0
    if cm.sum() > 0:
        plot_roc_s2(vct[cm], vcl[cm], cancer_types, s2_dir / "roc_curves_per_type.png")
        plot_cm_s2(vct[cm], vcl[cm], cancer_types, s2_dir / "confusion_matrix.png")
        type_df = plot_bars_s2(vct[cm], vcl[cm], cancer_types, s2_dir / "per_type_metrics.png")
        type_df.to_csv(met_dir / "per_cancer_type_metrics.csv", index=False)
        logger.info(f"\n{type_df.to_string(index=False)}")

        if tct is not None:
            tcm_mask = tct >= 0
            if tcm_mask.sum() > 0:
                plot_train_val_s2(tct[tcm_mask], tcl[tcm_mask], vct[cm], vcl[cm],
                                   cancer_types, s2_dir / "train_vs_val_roc.png")

    # ── Embedding ──
    logger.info("\n[Embedding] t-SNE...")
    if vemb.shape[0] > 10:
        plot_tsne(vemb, vgrp, vbt, emb_dir / "tsne_embedding.png")

    # ── Summary ──
    logger.info("\n[Summary] Metrics table...")
    mdf = summary_table(data, summary, opt_t, met_dir)
    logger.info(f"\n{mdf.to_string(index=False)}")

    # Threshold sweep
    thresh_rows = []
    for t in np.arange(0.3, 0.8, 0.05):
        m = binary_metrics(vbt, vbp, t)
        m["threshold"] = t
        thresh_rows.append(m)
    pd.DataFrame(thresh_rows).to_csv(met_dir / "threshold_sweep.csv", index=False)

    # ── SHAP ──
    if not args.no_shap:
        logger.info("\n[SHAP] Computing...")
        ckpt_files = sorted((train_dir / "checkpoints").glob("fold_*.pt"))
        if ckpt_files:
            ckpt = torch.load(ckpt_files[-1], map_location=device, weights_only=False)
            mc = ModelConfig(**ckpt["config"])
            model = SERSCancerDetector(mc).to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            rng = np.random.RandomState(42)
            n_bg = min(args.shap_samples, len(X))
            X_bg = X[rng.choice(len(X), n_bg, replace=False)]
            n_exp = min(args.shap_explain, valid.sum())
            exp_idx = rng.choice(np.where(valid)[0], n_exp, replace=False)
            X_exp = X[exp_idx]

            run_shap_s1(model, X_bg, X_exp, device, s1_dir)

            cancer_exp = bl[exp_idx] == 1
            if cancer_exp.sum() > 10:
                run_shap_s2(model, X_bg, X_exp[cancer_exp], cancer_types, device, s2_dir)
        else:
            logger.warning("  No checkpoints found")
    else:
        logger.info("\n[SHAP] Skipped (--no-shap)")

    elapsed = datetime.now() - t0
    logger.info(f"\n{'=' * 64}")
    logger.info(f"  Evaluation complete! ({elapsed})")
    logger.info(f"  Output: {eval_dir}/")
    logger.info(f"{'=' * 64}")
    return 0


if __name__ == "__main__":
    sys.exit(main())