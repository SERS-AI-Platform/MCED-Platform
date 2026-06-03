#!/usr/bin/env python3
"""STK-V2 training diagnostic figures.

This script reads training artifacts from ``results/training/<run>`` and writes
figures to ``results/figures/training/<run>``. The training script should only
produce data artifacts; figure rendering belongs here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_meta_learner_comparison(results_df: pd.DataFrame, output_dir: Path, dpi: int = 300) -> Path:
    """Write a boxplot comparing meta-learners across nested-CV folds."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    meta_names = results_df["meta_learner"].drop_duplicates().tolist()
    colors = plt.cm.Set2(np.linspace(0, 1, len(meta_names)))

    ax = axes[0]
    data_auc = [results_df.loc[results_df["meta_learner"] == m, "s1_auc"].values for m in meta_names]
    bp = ax.boxplot(data_auc, tick_labels=meta_names, patch_artist=True, widths=0.6)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_ylabel("Stage 1 AUC")
    ax.set_title("Meta-Learner Comparison: Detection AUC", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for i, values in enumerate(data_auc):
        ax.text(i + 1, np.median(values) + 0.001, f"{np.median(values):.4f}", ha="center", fontsize=8)

    ax = axes[1]
    data_f1 = [results_df.loc[results_df["meta_learner"] == m, "s2_f1"].values for m in meta_names]
    bp = ax.boxplot(data_f1, tick_labels=meta_names, patch_artist=True, widths=0.6)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_ylabel("Stage 2 F1 Macro")
    ax.set_title("Meta-Learner Comparison: Cancer Type F1", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for i, values in enumerate(data_f1):
        ax.text(i + 1, np.median(values) + 0.005, f"{np.median(values):.4f}", ha="center", fontsize=8)

    plt.tight_layout()
    out = output_dir / "meta_learner_comparison.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_contribution_analysis(contributions: pd.DataFrame, output_dir: Path, dpi: int = 300) -> Path:
    """Write a bar chart of base-model permutation importance."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    names = contributions["base_model"].astype(str).tolist()
    drops = contributions["mean_auc_drop"].astype(float).to_numpy()
    stds = contributions["std_auc_drop"].astype(float).to_numpy()

    colors = ["#E91E63" if d > 0 else "#90CAF9" for d in drops]
    ax.barh(range(len(names)), drops, xerr=stds, color=colors, alpha=0.85,
            capsize=3, edgecolor="gray", linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Mean AUC Drop (higher = more important)")
    ax.set_title("Base Model Contribution (Permutation Importance)", fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.grid(axis="x", alpha=0.3)

    for i, (drop, std) in enumerate(zip(drops, stds)):
        ax.text(drop + std + 0.0005, i, f"{drop:.4f}", va="center", fontsize=8)

    plt.tight_layout()
    out = output_dir / "contribution_analysis.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Render STK-V2 nested-CV diagnostic figures.")
    parser.add_argument("--run-dir", type=Path, default=Path("results/training/stacking_v2"))
    parser.add_argument("--fig-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    run_dir = args.run_dir
    fig_dir = args.fig_dir or Path("results") / "figures" / "training" / run_dir.name
    fig_dir.mkdir(parents=True, exist_ok=True)

    nested_csv = run_dir / "nested_cv_results.csv"
    contrib_csv = run_dir / "base_model_contribution.csv"

    written: list[Path] = []
    if nested_csv.exists():
        written.append(plot_meta_learner_comparison(pd.read_csv(nested_csv), fig_dir, dpi=args.dpi))
    else:
        print(f"skip: {nested_csv} not found")

    if contrib_csv.exists():
        written.append(plot_contribution_analysis(pd.read_csv(contrib_csv), fig_dir, dpi=args.dpi))
    else:
        print(f"skip: {contrib_csv} not found")

    for path in written:
        print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
