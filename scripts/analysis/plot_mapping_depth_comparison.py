#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib"]
# ///
# ─── How to run ───
# uv run plot_mapping_depth_comparison.py
# ──────────────────
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO / "results" / "mapping_depth_comparison_20260826_v1"
CONDITIONS = [("standard", "Standard"), ("trim_only", "Trim-only"), ("no_trim_raw", "No-trim raw")]
DEPTHS = (3, 12)
COLORS = {3: "#333333", 12: "#9467bd"}
RUNS = {
    ("standard", 3): REPO / "results" / "mapping_multiscale_resnet_20260826_standard_repro",
    ("standard", 12): REPO / "results" / "mapping_multiscale_resnet_20260826_deep12_standard",
    ("trim_only", 3): REPO / "results" / "mapping_trim_only_20260826_v1",
    ("trim_only", 12): REPO / "results" / "mapping_trim_only_20260826_deep12",
    ("no_trim_raw", 3): REPO / "results" / "mapping_no_trim_raw_20260826_v1",
    ("no_trim_raw", 12): REPO / "results" / "mapping_no_trim_raw_20260826_deep12",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    keys = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def configure_style() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 15, "figure.facecolor": "white", "axes.facecolor": "white"})


def finish_axes(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def label_bars(ax: plt.Axes, bars, values: np.ndarray) -> None:
    for bar, value in zip(bars, values):
        if np.isfinite(value):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008, f"{value:.3f}", ha="center", fontsize=9)


def oof_rows() -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for (condition, depth), run in RUNS.items():
        for row in read_rows(run / "oof_metrics.csv"):
            if row.get("aggregation") == "mean" and row.get("model") == "Multi-scale ResNet":
                output.append({"condition": condition, "depth": str(depth), **row})
    return output


def training_rows() -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for (condition, depth), run in RUNS.items():
        rows = read_rows(run / "training_history.csv")
        for task in ("binary", "three"):
            fold_rows = []
            for fold in sorted({row["fold"] for row in rows if row.get("task") == task}):
                selected = [row for row in rows if row.get("task") == task and row.get("fold") == fold]
                best = min(selected, key=lambda row: number(row, "validation_loss"))
                fold_rows.append(best)
            values = {key: np.array([number(row, key) for row in fold_rows]) for key in ("epoch", "train_loss", "validation_loss", "train_auc", "validation_auc")}
            output.append({"condition": condition, "depth": str(depth), "task": task, "folds": str(len(fold_rows)), **{f"{key}_mean": f"{np.nanmean(value):.10g}" for key, value in values.items()}, "train_val_auc_gap_mean": f"{np.nanmean(values['train_auc'] - values['validation_auc']):.10g}"})
    return output


def lookup(rows: list[dict[str, str]], condition: str, depth: int, task: str, key: str) -> float:
    for row in rows:
        if row.get("condition") == condition and row.get("depth") == str(depth) and row.get("task") == task:
            return number(row, key)
    return float("nan")


def save_figure(fig: plt.Figure, output: Path, name: str) -> None:
    fig.savefig(output / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_auc_bar(rows: list[dict[str, str]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    x = np.arange(len(CONDITIONS))
    for ax, task, metric, title, ymax in zip(axes, ("cancer_vs_non_cancer", "three_class"), ("roc_auc", "macro_roc_auc"), ("Cancer Screening", "Cancer Type ID (3-class)"), (0.85, 0.75)):
        for index, depth in enumerate(DEPTHS):
            values = np.array([next((number(row, metric) for row in rows if row.get("condition") == condition and row.get("depth") == str(depth) and row.get("task") == task), np.nan) for condition, _ in CONDITIONS])
            bars = ax.bar(x + (index - 0.5) * 0.28, values, 0.28, color=COLORS[depth], label=f"{depth} blocks")
            label_bars(ax, bars, values)
        ax.set_title(title)
        ax.set_xticks(x, [label for _, label in CONDITIONS])
        ax.set_ylabel("AUC")
        ax.set_ylim(0, ymax)
        finish_axes(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=2, frameon=False)
    fig.suptitle("ResNet depth comparison", fontsize=19, y=1.04)
    fig.text(0.02, -0.01, "Same preprocessing condition, cohort, patient-level 5-fold OOF, seed 20260826, 30 epochs, 100 MC iterations", fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 0.9))
    save_figure(fig, output, "figure_depth_auc_bar.png")


def make_gap_bar(rows: list[dict[str, str]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    x = np.arange(len(CONDITIONS))
    for ax, task, title in zip(axes, ("binary", "three"), ("Cancer Screening", "Cancer Type ID (3-class)")):
        for index, depth in enumerate(DEPTHS):
            values = np.array([lookup(rows, condition, depth, task, "train_val_auc_gap_mean") for condition, _ in CONDITIONS])
            bars = ax.bar(x + (index - 0.5) * 0.28, values, 0.28, color=COLORS[depth], label=f"{depth} blocks")
            label_bars(ax, bars, values)
        ax.set_title(title)
        ax.set_xticks(x, [label for _, label in CONDITIONS])
        ax.set_ylabel("Train AUC − validation AUC")
        ax.set_ylim(0, 0.25)
        finish_axes(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=2, frameon=False)
    fig.suptitle("Validation gap at best validation-loss epoch", fontsize=19, y=1.04)
    fig.text(0.02, -0.01, "Larger gap indicates stronger train/validation separation; fold means across the five outer folds", fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 0.9))
    save_figure(fig, output, "figure_depth_training_gap_bar.png")


def make_repeat_curve(output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), sharey=True)
    for ax, condition_label in zip(axes, CONDITIONS):
        condition = condition_label[0]
        for depth in DEPTHS:
            rows = [row for row in read_rows(RUNS[(condition, depth)] / "repeat_metrics_summary.csv") if row.get("task") == "cancer_vs_non_cancer" and row.get("model") == "Multi-scale ResNet" and row.get("aggregation") == "mean"]
            rows.sort(key=lambda row: number(row, "n"))
            ax.plot([number(row, "n") for row in rows], [number(row, "roc_auc_mean") for row in rows], color=COLORS[depth], marker="o" if depth == 3 else "s", linewidth=2, label=f"{depth} blocks")
        ax.set_title(condition_label[1])
        ax.set_xlabel("Number of repeats n")
        ax.set_xticks([1, 9, 25, 49, 81, 121])
        ax.set_xlim(0, 125)
        ax.set_ylim(0.35, 0.75)
        finish_axes(ax)
    axes[0].set_ylabel("Cancer Screening ROC-AUC")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
    fig.suptitle("Depth comparison across repeat counts", fontsize=19, y=1.07)
    fig.text(0.02, -0.01, "ResNet only • patient-level Monte-Carlo evaluation • 100 random subsets per n • mean aggregation", fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 0.91))
    save_figure(fig, output, "figure_depth_repeat_auc.png")


def write_report(output: Path, rows: list[dict[str, str]], charts: list[str]) -> None:
    trim3 = lookup(rows, "trim_only", 3, "cancer_vs_non_cancer", "roc_auc")
    trim12 = lookup(rows, "trim_only", 12, "cancer_vs_non_cancer", "roc_auc")
    lines = ["# Mapping ResNet depth comparison", "", "기존 3-block 결과를 보존한 상태에서 12-block 모델을 동일 조건으로 재학습한 결과입니다.", "", "## Figures", ""]
    lines.extend(f"![{chart}]({chart})\n" for chart in charts)
    lines.extend(["## 핵심 결과", "", f"- Trim-only Cancer Screening ROC-AUC: 3 blocks {trim3:.3f} → 12 blocks {trim12:.3f}.", "- 깊이 증가만으로 peak feature extraction이 회복되지 않았고, training gap을 함께 확인해야 합니다.", "- 기존 run을 삭제하지 않고 새 deep12 run과 이 요약 폴더를 분리해 재현성을 보존했습니다.", "", "## Source tables", "", "- `depth_oof_metrics.csv`: depth/condition별 OOF 지표", "- `depth_training_summary.csv`: best validation-loss epoch 기준 train/validation 요약"])
    (output / "VISUAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render ResNet depth comparison figures")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    configure_style()
    oof = oof_rows()
    training = training_rows()
    write_csv(output / "depth_oof_metrics.csv", oof)
    write_csv(output / "depth_training_summary.csv", training)
    make_auc_bar(oof, output)
    make_gap_bar(training, output)
    make_repeat_curve(output)
    write_report(output, oof, ["figure_depth_auc_bar.png", "figure_depth_training_gap_bar.png", "figure_depth_repeat_auc.png"])
    print(output)


if __name__ == "__main__":
    main()
