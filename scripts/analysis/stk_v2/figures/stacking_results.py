#!/usr/bin/env python3
"""Generic stacking result summary plot."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_results(results: list[dict], output_dir: Path, data_name: str, dpi: int = 150) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    base = [r for r in results if r["type"] == "base"]
    ens = [r for r in results if r["type"] != "base"]
    ordered = base + ens

    names = [r["name"] for r in ordered]
    aucs = [r["auc"] for r in ordered]
    f1s = [r["f1_type"] for r in ordered]
    colors = ["#90CAF9"] * len(base) + ["#FFB74D", "#E91E63"]

    ax = axes[0]
    ax.barh(range(len(names)), aucs, color=colors, alpha=0.9)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("AUC")
    ax.set_title(f"Stage 1: Detection AUC ({data_name})", fontweight="bold")
    ax.set_xlim(0.9, 1.0)
    ax.grid(axis="x", alpha=0.3)
    for i, value in enumerate(aucs):
        ax.text(value + 0.001, i, f"{value:.4f}", va="center", fontsize=8)

    ax = axes[1]
    ax.barh(range(len(names)), f1s, color=colors, alpha=0.9)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("F1 Macro")
    ax.set_title(f"Stage 2: Cancer Type F1 ({data_name})", fontweight="bold")
    ax.set_xlim(0.6, 1.0)
    ax.axvline(0.9, color="red", linestyle="--", alpha=0.5, label="Target 0.9")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    for i, value in enumerate(f1s):
        ax.text(value + 0.005, i, f"{value:.4f}", va="center", fontsize=8)

    fig.tight_layout()
    out = output_dir / f"stacking_results_{data_name}.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out
