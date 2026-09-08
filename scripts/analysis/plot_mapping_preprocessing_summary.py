#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib"]
# ///
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY = REPO / "results" / "mapping_preprocessing_summary_20260826_v1"
CONDITIONS = [("standard", "Standard"), ("trim_only", "Trim-only"), ("no_trim_raw", "No-trim raw")]
MODELS = ["LR-reference", "Multi-scale ResNet", "LR+ResNet fixed blend"]
MODEL_COLORS = {MODELS[0]: "#333333", MODELS[1]: "#d62728", MODELS[2]: "#9467bd"}
MODEL_MARKERS = {MODELS[0]: "o", MODELS[1]: "s", MODELS[2]: "^"}
METRIC_COLORS = ["#333333", "#4c78a8", "#e45756", "#72b7b2"]
RUN_DIRS = {
    "standard": REPO / "results" / "mapping_multiscale_resnet_20260826_standard_repro",
    "trim_only": REPO / "results" / "mapping_trim_only_20260826_v1",
    "no_trim_raw": REPO / "results" / "mapping_no_trim_raw_20260826_v1",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def lookup(rows: list[dict[str, str]], keys: dict[str, str], metric: str) -> float:
    for row in rows:
        if all(row.get(key) == value for key, value in keys.items()):
            return number(row, metric)
    return float("nan")


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 11,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 1.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def finish_axes(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def label_bars(ax: plt.Axes, bars, values: np.ndarray, signed: bool = False) -> None:
    for bar, value in zip(bars, values):
        if not np.isfinite(value):
            continue
        height = bar.get_height()
        if signed and value < 0:
            y, va = height - 0.008, "top"
        else:
            y, va = height + 0.008, "bottom"
        ax.text(bar.get_x() + bar.get_width() / 2, y, f"{value:.3f}", ha="center", va=va, fontsize=9)


def save_figure(fig: plt.Figure, output: Path, name: str) -> str:
    path = output / name
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return name


def make_auc_bar(rows: list[dict[str, str]], output: Path) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=False)
    x = np.arange(len(CONDITIONS))
    width = 0.22
    specs = [("cancer_vs_non_cancer", "roc_auc", "Cancer Screening ROC-AUC", 0.85), ("three_class", "macro_roc_auc", "Cancer Type ID Macro ROC-AUC", 0.75)]
    for ax, (task, metric, title, ymax) in zip(axes, specs):
        for index, model in enumerate(MODELS):
            values = np.array([lookup(rows, {"condition": condition, "task": task, "model": model, "aggregation": "mean"}, metric) for condition, _ in CONDITIONS])
            bars = ax.bar(x + (index - 1) * width, values, width, color=MODEL_COLORS[model], label=model)
            label_bars(ax, bars, values)
        ax.set_title(title)
        ax.set_xticks(x, [label for _, label in CONDITIONS])
        ax.set_ylabel("AUC")
        ax.set_ylim(0, ymax)
        finish_axes(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=3, frameon=False)
    fig.suptitle("Performance by preprocessing condition", fontsize=19, y=1.04)
    fig.text(0.02, -0.01, "Patient-level 5-fold OOF • n=113 • 3-block Multi-scale ResNet • patient mean aggregation", fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 0.9))
    figure_name = "figure_1_preprocessing_performance_bar.png"
    for name in [figure_name, "preprocessing_ablation_auc.png"]:
        fig.savefig(output / name, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return figure_name


def make_resnet_metric_bar(rows: list[dict[str, str]], output: Path) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
    x = np.arange(len(CONDITIONS))
    specs = [
        ("cancer_vs_non_cancer", [("ROC-AUC", "roc_auc"), ("Balanced accuracy", "balanced_accuracy"), ("Sensitivity", "sensitivity"), ("Specificity", "specificity")], "Cancer Screening"),
        ("three_class", [("Macro ROC-AUC", "macro_roc_auc"), ("Balanced accuracy", "balanced_accuracy"), ("Macro F1", "macro_f1")], "Cancer Type ID (3-class)"),
    ]
    for ax, (task, metrics, title) in zip(axes, specs):
        width = 0.75 / len(metrics)
        for index, (label, metric) in enumerate(metrics):
            values = np.array([lookup(rows, {"condition": condition, "task": task, "model": "Multi-scale ResNet", "aggregation": "mean"}, metric) for condition, _ in CONDITIONS])
            bars = ax.bar(x + (index - (len(metrics) - 1) / 2) * width, values, width, color=METRIC_COLORS[index], label=label)
            label_bars(ax, bars, values)
        ax.set_title(title)
        ax.set_xticks(x, [label for _, label in CONDITIONS])
        ax.set_ylabel("Metric")
        ax.set_ylim(0, 0.85 if task == "cancer_vs_non_cancer" else 0.75)
        finish_axes(ax)
    for ax, metrics in zip(axes, [specs[0][1], specs[1][1]]):
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2 if len(metrics) > 3 else 3, frameon=False)
    fig.suptitle("Multi-scale ResNet metric profile", fontsize=19, y=1.15)
    fig.text(0.02, -0.01, "Same cohort, folds, seed, epochs, and 100-repeat evaluation across all preprocessing conditions", fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 0.86))
    return save_figure(fig, output, "figure_2_resnet_metrics_bar.png")


def make_delta_bar(rows: list[dict[str, str]], output: Path) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=False)
    x = np.arange(2)
    width = 0.22
    specs = [("cancer_vs_non_cancer", "roc_auc", "Cancer Screening Δ ROC-AUC", (-0.10, 0.25)), ("three_class", "macro_roc_auc", "Cancer Type ID Δ Macro ROC-AUC", (-0.10, 0.12))]
    for ax, (task, metric, title, limits) in zip(axes, specs):
        for index, model in enumerate(MODELS):
            values = np.array([lookup(rows, {"condition": condition, "task": task, "model": model, "metric": metric}, "delta_vs_standard") for condition, _ in CONDITIONS[1:]])
            bars = ax.bar(x + (index - 1) * width, values, width, color=MODEL_COLORS[model], label=model)
            label_bars(ax, bars, values, signed=True)
        ax.axhline(0, color="#333333", linewidth=1)
        ax.set_title(title)
        ax.set_xticks(x, [label for _, label in CONDITIONS[1:]])
        ax.set_ylabel("Δ versus Standard")
        ax.set_ylim(*limits)
        finish_axes(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=3, frameon=False)
    fig.suptitle("Effect of removing preprocessing operations", fontsize=19, y=1.04)
    fig.text(0.02, -0.01, "Δ = condition metric − Standard metric; positive values indicate improvement over the standard preprocessing bundle", fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 0.9))
    return save_figure(fig, output, "figure_3_delta_vs_standard_bar.png")


def make_minimum_bar(rows: list[dict[str, str]], output: Path) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    x = np.arange(len(CONDITIONS))
    width = 0.22
    for ax, task, title in zip(axes, ["cancer_vs_non_cancer", "three_class"], ["Cancer Screening", "Cancer Type ID (3-class)"]):
        for index, model in enumerate(MODELS):
            values = np.array([lookup(rows, {"condition": condition, "task": task, "model": model, "aggregation": "mean"}, "minimum_n") for condition, _ in CONDITIONS])
            bars = ax.bar(x + (index - 1) * width, values, width, color=MODEL_COLORS[model], label=model)
            label_bars(ax, bars, values)
        ax.set_title(title)
        ax.set_xticks(x, [label for _, label in CONDITIONS])
        ax.set_ylabel("Minimum repeats n*")
        ax.set_ylim(0, 121)
        ax.set_yticks([0, 25, 50, 75, 100, 121])
        finish_axes(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=3, frameon=False)
    fig.suptitle("Minimum repeat count under locked criteria", fontsize=19, y=1.04)
    fig.text(0.02, -0.01, "n* is relative to each condition's own 121-repeat reference; mean aggregation and 100 Monte-Carlo iterations", fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 0.9))
    return save_figure(fig, output, "figure_4_minimum_repeat_count_bar.png")


def make_repeat_line(summary: Path, output: Path, metric_file: str, metric: str, spread: str | None, ylabel: str, name: str, title: str, limits: tuple[float, float]) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), sharey=True)
    for ax, (condition, label) in zip(axes, CONDITIONS):
        rows = read_rows(RUN_DIRS[condition] / metric_file)
        for model in MODELS:
            selected = [row for row in rows if row.get("task") == "cancer_vs_non_cancer" and row.get("model") == model and row.get("aggregation") == "mean"]
            selected.sort(key=lambda row: number(row, "n"))
            ns = np.array([number(row, "n") for row in selected])
            values = np.array([number(row, metric) for row in selected])
            if len(ns) == 0:
                continue
            ax.plot(ns, values, color=MODEL_COLORS[model], marker=MODEL_MARKERS[model], linewidth=2.0, markersize=5, label=model)
            if spread:
                sd = np.array([number(row, spread) for row in selected])
                ax.fill_between(ns, values - sd, values + sd, color=MODEL_COLORS[model], alpha=0.10, linewidth=0)
        ax.set_title(label)
        ax.set_xlabel("Number of repeats n")
        ax.set_xticks([1, 9, 25, 49, 81, 121])
        ax.set_xlim(0, 125)
        ax.set_ylim(*limits)
        finish_axes(ax)
    axes[0].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False)
    fig.suptitle(title, fontsize=19, y=1.07)
    fig.text(0.02, -0.01, "Patient-level Monte-Carlo evaluation • 100 random subsets per n • mean aggregation", fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 0.91))
    return save_figure(fig, output, name)


def write_visual_report(output: Path, charts: list[str]) -> None:
    lines = [
        "# Mapping preprocessing visual summary",
        "",
        "이번 결과는 숫자 표가 아니라 아래 그래프를 중심으로 확인합니다.",
        "",
        "## Direct comparison",
        "",
        "![Preprocessing performance](figure_1_preprocessing_performance_bar.png)",
        "",
        "![ResNet metric profile](figure_2_resnet_metrics_bar.png)",
        "",
        "![Delta versus standard](figure_3_delta_vs_standard_bar.png)",
        "",
        "## Repeat-count analysis",
        "",
        "![Minimum repeat count](figure_4_minimum_repeat_count_bar.png)",
        "",
        "![Repeat-count ROC-AUC](figure_5_repeat_roc_auc.png)",
        "",
        "![Repeat-count balanced accuracy](figure_6_repeat_balanced_accuracy.png)",
        "",
        "![Prediction stability](figure_7_prediction_stability.png)",
        "",
        "## 핵심 시각적 결론",
        "",
        "- Standard 전처리에서는 Multi-scale ResNet의 Cancer Screening ROC-AUC가 0.442였고, Trim-only에서는 0.653으로 상승합니다.",
        "- Standard 대비 ResNet의 개선폭은 Trim-only에서 가장 크며, No-trim raw는 Trim-only보다 다시 낮아집니다.",
        "- 임상정보와 spectrum을 함께 쓴 LR/CNN은 이 전처리 ablation의 직접 비교군이 아니므로 historical_experiment_summary.csv에서 별도로 확인합니다.",
        "",
        "## Generated files",
        "",
    ]
    lines.extend(f"- `{chart}`" for chart in charts)
    (output / "VISUAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render mapping preprocessing comparison figures")
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()
    output = args.summary_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    configure_style()
    performance = read_rows(output / "same_condition_performance.csv")
    deltas = read_rows(output / "delta_vs_standard.csv")
    minimum = read_rows(output / "repeat_minimum_summary.csv")
    charts = [
        make_auc_bar(performance, output),
        make_resnet_metric_bar(performance, output),
        make_delta_bar(deltas, output),
        make_minimum_bar(minimum, output),
        make_repeat_line(output, output, "repeat_metrics_summary.csv", "roc_auc_mean", "roc_auc_sd", "ROC-AUC", "figure_5_repeat_roc_auc.png", "Cancer Screening ROC-AUC versus repeat count", (0.35, 0.80)),
        make_repeat_line(output, output, "repeat_metrics_summary.csv", "balanced_accuracy_mean", "balanced_accuracy_sd", "Balanced accuracy", "figure_6_repeat_balanced_accuracy.png", "Cancer Screening balanced accuracy versus repeat count", (0.30, 0.75)),
        make_repeat_line(output, output, "prediction_stability.csv", "probability_sd_mean", None, "Mean patient probability SD", "figure_7_prediction_stability.png", "Prediction stability versus repeat count", (0.0, 0.35)),
    ]
    write_visual_report(output, charts)
    print("\n".join(str(output / chart) for chart in charts))


if __name__ == "__main__":
    main()
