from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CONDITIONS = (
    ("raw_aligned", "Raw aligned"),
    ("baseline", "Baseline"),
    ("baseline_area", "Baseline + area"),
    ("baseline_dwt", "Baseline + DWT"),
    ("baseline_area_dwt", "Baseline + area + DWT"),
)
COLORS = ("#333333", "#4C78A8", "#54A24B", "#F58518", "#9467bd")


def _number(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def _find_row(rows: list[dict[str, str]], **criteria: str) -> dict[str, str]:
    return next((row for row in rows if all(row.get(key) == value for key, value in criteria.items())), {})


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 15,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _finish_axes(axis: plt.Axes) -> None:
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _label_bars(axis: plt.Axes, bars, values: np.ndarray) -> None:
    for bar, value in zip(bars, values):
        if np.isfinite(value):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{value:.3f}",
                ha="center",
                fontsize=9,
            )


def _save_figure(figure: plt.Figure, output: Path, name: str) -> None:
    figure.savefig(output / name, dpi=180, bbox_inches="tight")
    plt.close(figure)


def make_auc_bar(oof: list[dict[str, str]], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(15, 6))
    x = np.arange(len(CONDITIONS))
    for axis, task, metric, title, ymax in zip(
        axes,
        ("cancer_vs_non_cancer", "three_class"),
        ("roc_auc", "macro_roc_auc"),
        ("Cancer Screening", "Cancer Type ID (3-class)"),
        (0.8, 0.7),
    ):
        values = np.array(
            [_number(_find_row(oof, condition=condition, task=task, model="Multi-scale ResNet"), metric) for condition, _label in CONDITIONS]
        )
        bars = axis.bar(x, values, color=COLORS, width=0.62)
        _label_bars(axis, bars, values)
        axis.set_title(title)
        axis.set_xticks(x, [label for _condition, label in CONDITIONS], rotation=18, ha="right")
        axis.set_ylabel("AUC")
        axis.set_ylim(0, ymax)
        _finish_axes(axis)
    figure.suptitle("Raw aligned → baseline → area normalization → DWT", fontsize=19, y=1.03)
    figure.text(0.02, -0.01, "Multi-scale ResNet • 3 blocks • patient-level 5-fold OOF • seed 20260826 • 30 epochs", fontsize=10)
    figure.tight_layout()
    _save_figure(figure, output, "figure_preprocessing_auc_bar.png")


def make_metric_bar(oof: list[dict[str, str]], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(15, 6))
    x = np.arange(len(CONDITIONS))
    configs = (
        ("cancer_vs_non_cancer", ("balanced_accuracy", "sensitivity", "specificity"), "Cancer Screening"),
        ("three_class", ("balanced_accuracy", "macro_f1"), "Cancer Type ID (3-class)"),
    )
    metric_labels = {"balanced_accuracy": "Balanced accuracy", "sensitivity": "Sensitivity", "specificity": "Specificity", "macro_f1": "Macro F1"}
    for axis, (task, metrics, title) in zip(axes, configs):
        width = 0.72 / len(metrics)
        for index, metric in enumerate(metrics):
            values = np.array(
                [_number(_find_row(oof, condition=condition, task=task, model="Multi-scale ResNet"), metric) for condition, _label in CONDITIONS]
            )
            bars = axis.bar(x + (index - (len(metrics) - 1) / 2) * width, values, width, label=metric_labels[metric])
            _label_bars(axis, bars, values)
        axis.set_title(title)
        axis.set_xticks(x, [label for _condition, label in CONDITIONS], rotation=18, ha="right")
        axis.set_ylim(0, 1.08)
        axis.set_ylabel("Score")
        axis.legend(frameon=False, fontsize=9)
        _finish_axes(axis)
    figure.suptitle("ResNet metric profile by preprocessing condition", fontsize=19, y=1.03)
    figure.tight_layout()
    _save_figure(figure, output, "figure_preprocessing_metrics_bar.png")


def make_repeat_figures(repeat: list[dict[str, str]], stability: list[dict[str, str]], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    color_by_condition = {condition: COLORS[index] for index, (condition, _label) in enumerate(CONDITIONS)}
    for condition, label in CONDITIONS:
        repeat_rows = sorted([row for row in repeat if row.get("condition") == condition], key=lambda row: _number(row, "n"))
        stable_rows = sorted([row for row in stability if row.get("condition") == condition], key=lambda row: _number(row, "n"))
        axes[0].plot([_number(row, "n") for row in repeat_rows], [_number(row, "roc_auc_mean") for row in repeat_rows], marker="o", linewidth=2, color=color_by_condition[condition], label=label)
        axes[1].plot([_number(row, "n") for row in stable_rows], [_number(row, "probability_sd_mean") for row in stable_rows], marker="o", linewidth=2, color=color_by_condition[condition], label=label)
    for axis, ylabel, title in zip(axes, ("Cancer Screening ROC-AUC", "Mean probability SD"), ("Repeat count vs ROC-AUC", "Repeat count vs prediction stability")):
        axis.set_xlabel("Number of repeats n")
        axis.set_xticks([1, 9, 25, 49, 81, 121])
        axis.set_xlim(0, 125)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        _finish_axes(axis)
    axes[0].set_ylim(0.35, 0.75)
    axes[1].set_ylim(bottom=0)
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    figure.suptitle("Repeat-count analysis: ResNet mean aggregation", fontsize=19, y=1.03)
    figure.text(0.02, -0.01, "100 Monte-Carlo random subsets per n; lower probability SD indicates higher stability", fontsize=10)
    figure.tight_layout()
    _save_figure(figure, output, "figure_repeat_auc_stability.png")


def make_minimum_bar(minimum: list[dict[str, str]], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(15, 6))
    x = np.arange(len(CONDITIONS))
    for axis, task, title in zip(axes, ("cancer_vs_non_cancer", "three_class"), ("Cancer Screening", "Cancer Type ID (3-class)")):
        values = np.array(
            [_number(_find_row(minimum, condition=condition, task=task, model="Multi-scale ResNet"), "minimum_n") for condition, _label in CONDITIONS]
        )
        bars = axis.bar(x, values, color=COLORS, width=0.62)
        _label_bars(axis, bars, values)
        axis.set_title(title)
        axis.set_xticks(x, [label for _condition, label in CONDITIONS], rotation=18, ha="right")
        axis.set_xlabel("Preprocessing")
        axis.set_ylabel("Minimum repeat count n*")
        axis.set_ylim(0, 130)
        _finish_axes(axis)
    figure.suptitle("Minimum repeat count under the locked criteria", fontsize=19, y=1.03)
    figure.text(0.02, -0.01, "Relative to each condition's own 121-repeat reference; ResNet mean aggregation", fontsize=10)
    figure.tight_layout()
    _save_figure(figure, output, "figure_minimum_repeat_bar.png")
