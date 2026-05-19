#!/usr/bin/env python3
"""Render figures from STK-V2 fixed-split artifacts.

Expected input files under ``--run-dir``:
  - fixed_test_predictions.npz
  - roc_curve_data.npz
  - single_vs_ensemble.csv
  - cancer_type_metrics.csv
  - confusion_matrix_counts.csv
  - confusion_matrix_normalized.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _default_fig_dir(run_dir: Path) -> Path:
    return Path("results") / "figures" / "training" / run_dir.name


def plot_stage1_roc(run_dir: Path, fig_dir: Path, dpi: int = 300) -> Path | None:
    roc_path = run_dir / "roc_curve_data.npz"
    if not roc_path.exists():
        return None
    d = np.load(roc_path)
    fpr = d["fpr"]
    tpr = d["tpr"]
    auc = float(np.ravel(d["auc"])[0]) if "auc" in d.files else np.nan

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.plot(fpr, tpr, color="#1f77b4", lw=2.2, label=f"AUC={auc:.4f}")
    ax.plot([0, 1], [0, 1], color="#999999", ls="--", lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Fixed Split Stage 1 ROC", fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = fig_dir / "fixed_stage1_roc.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_type_confusion(run_dir: Path, fig_dir: Path, dpi: int = 300) -> Path | None:
    counts_path = run_dir / "confusion_matrix_counts.csv"
    norm_path = run_dir / "confusion_matrix_normalized.csv"
    if not counts_path.exists() or not norm_path.exists():
        return None
    cm = pd.read_csv(counts_path, index_col=0)
    cm_norm = pd.read_csv(norm_path, index_col=0)
    labels = cm.index.astype(str).tolist()

    fig, ax = plt.subplots(figsize=(7.5, 6.6))
    im = ax.imshow(cm_norm.values, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    for (i, j), value in np.ndenumerate(cm.values):
        norm_value = float(cm_norm.values[i, j])
        ax.text(
            j, i, f"{int(value)}\n({norm_value:.2f})",
            ha="center", va="center",
            color="white" if norm_value > 0.5 else "black",
            fontsize=9,
            fontweight="bold" if i == j else "normal",
        )
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Fixed Split Stage 2 Confusion Matrix", fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized")
    fig.tight_layout()
    out = fig_dir / "fixed_type_confusion_matrix.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_cancer_type_metrics(run_dir: Path, fig_dir: Path, dpi: int = 300) -> Path | None:
    metrics_path = run_dir / "cancer_type_metrics.csv"
    if not metrics_path.exists():
        return None
    df = pd.read_csv(metrics_path)
    metrics = ["sensitivity", "specificity", "precision", "f1"]
    present = [m for m in metrics if m in df.columns]
    if not present:
        return None

    x = np.arange(len(df))
    width = 0.8 / len(present)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#4CAF50"]
    for i, metric in enumerate(present):
        offset = (i - (len(present) - 1) / 2) * width
        ax.bar(x + offset, df[metric].astype(float), width=width, label=metric, color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(df["cancer"].astype(str))
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Fixed Split Cancer-Type Metrics", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = fig_dir / "fixed_cancer_type_metrics.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_single_vs_ensemble(run_dir: Path, fig_dir: Path, dpi: int = 300) -> Path | None:
    csv_path = run_dir / "single_vs_ensemble.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if not {"model", "type", "auc", "f1_type"}.issubset(df.columns):
        return None

    color_map = {"single": "#607D8B", "ensemble": "#F57C00", "meta": "#C62828"}
    marker_map = {"single": "o", "ensemble": "s", "meta": "^"}
    fig, ax = plt.subplots(figsize=(8, 6))
    for typ, sub in df.groupby("type"):
        ax.scatter(
            sub["auc"], sub["f1_type"],
            s=80, marker=marker_map.get(typ, "o"),
            color=color_map.get(typ, "#333333"),
            edgecolor="white", linewidth=0.8, label=typ,
        )
        for _, row in sub.iterrows():
            ax.annotate(
                str(row["model"]), (row["auc"], row["f1_type"]),
                xytext=(4, 4), textcoords="offset points", fontsize=8,
            )
    ax.set_xlabel("Stage 1 AUROC")
    ax.set_ylabel("Stage 2 Macro F1")
    ax.set_title("Fixed Split Single Models vs Ensemble", fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out = fig_dir / "fixed_single_vs_ensemble.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def render_fixed_split_figures(run_dir: Path, fig_dir: Path | None = None, dpi: int = 300) -> list[Path]:
    run_dir = Path(run_dir)
    fig_dir = Path(fig_dir) if fig_dir is not None else _default_fig_dir(run_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    written = [
        plot_stage1_roc(run_dir, fig_dir, dpi=dpi),
        plot_type_confusion(run_dir, fig_dir, dpi=dpi),
        plot_cancer_type_metrics(run_dir, fig_dir, dpi=dpi),
        plot_single_vs_ensemble(run_dir, fig_dir, dpi=dpi),
    ]
    return [p for p in written if p is not None]


def main() -> int:
    parser = argparse.ArgumentParser(description="Render STK-V2 fixed-split figures.")
    parser.add_argument("--run-dir", type=Path, default=Path("results/training/stacking_v2_fixed"))
    parser.add_argument("--fig-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    written = render_fixed_split_figures(args.run_dir, args.fig_dir, args.dpi)
    if not written:
        print(f"No fixed-split figures rendered from {args.run_dir}")
        return 1
    for path in written:
        print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
