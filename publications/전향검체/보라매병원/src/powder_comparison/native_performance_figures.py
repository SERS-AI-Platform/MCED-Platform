from __future__ import annotations

from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import RocCurveDisplay, confusion_matrix

from powder_comparison.models import PairedCrossfitResult
from powder_comparison.statistics import metric_values

BLUE: Final = "#2C7FB8"
PURPLE: Final = "#7A5195"
ORANGE: Final = "#D95F02"
GREY: Final = "#777777"
INK: Final = "#252525"


def _save(fig: plt.Figure, output: Path, name: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{name}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _draw_confusion_matrix(
    axis: plt.Axes,
    matrix: np.ndarray,
    labels: tuple[str, ...],
    title: str,
) -> None:
    axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set(xlabel="Predicted", ylabel="True", title=title)
    row_totals = matrix.sum(axis=1)
    midpoint = float(np.max(matrix)) / 2.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{matrix[row, column]}\n({matrix[row, column] / row_totals[row]:.1%})",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > midpoint else INK,
            )


def _metric_map(result: PairedCrossfitResult) -> dict[str, float]:
    return {
        row.name: row.value
        for row in metric_values(
            result.y_true,
            result.native_probabilities,
            result.task,
        )
    }


def _metric_bars(
    axis: plt.Axes,
    metrics: dict[str, float],
    names: tuple[str, ...],
    labels: tuple[str, ...],
    color: str,
    title: str,
) -> None:
    values = [metrics[name] for name in names]
    positions = np.arange(len(labels))
    bars = axis.bar(positions, values, color=color, alpha=0.85)
    axis.set_xticks(positions, labels, rotation=25, ha="right")
    axis.set_ylim(0, 1.05)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.2)
    axis.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3, fontsize=8)


def _screening_figure(result: PairedCrossfitResult, output: Path) -> None:
    predictions = result.native_probabilities.argmax(axis=1)
    matrix = confusion_matrix(result.y_true, predictions, labels=(0, 1))
    metrics = _metric_map(result)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    _draw_confusion_matrix(
        axes[0], matrix, ("Non-cancer", "Cancer"), "Powder-native confusion matrix"
    )
    RocCurveDisplay.from_predictions(
        result.y_true,
        result.native_probabilities[:, 1],
        name="Powder-native LR",
        ax=axes[1],
        curve_kwargs={"color": ORANGE},
    )
    axes[1].plot([0, 1], [0, 1], "--", color=GREY, lw=0.8)
    axes[1].set_title("Powder-native screening ROC")
    axes[1].legend(loc="lower right", fontsize=8)
    _metric_bars(
        axes[2],
        metrics,
        ("roc_auc", "balanced_accuracy", "sensitivity", "specificity"),
        ("ROC-AUC", "Balanced acc.", "Sensitivity", "Specificity"),
        ORANGE,
        "Powder-native nested OOF metrics",
    )
    fig.suptitle(
        f"Powder-native LR. Cancer vs Non-cancer: Control + Biopsy-negative vs Cancer (n={len(result.y_true)})",
        y=1.02,
    )
    _save(fig, output, "fig14_powder_native_screening_auc_confusion_matrix")


def _three_group_figure(result: PairedCrossfitResult, output: Path) -> None:
    labels = ("Control", "Biopsy-negative", "Cancer")
    predictions = result.native_probabilities.argmax(axis=1)
    matrix = confusion_matrix(result.y_true, predictions, labels=(0, 1, 2))
    metrics = _metric_map(result)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1, 1, 1.05]})
    _draw_confusion_matrix(axes[0], matrix, labels, "Powder-native 3-group confusion matrix")
    for class_index, (label, color) in enumerate(zip(labels, (BLUE, PURPLE, ORANGE), strict=True)):
        RocCurveDisplay.from_predictions(
            (result.y_true == class_index).astype(int),
            result.native_probabilities[:, class_index],
            name=label,
            ax=axes[1],
            curve_kwargs={"color": color},
        )
    axes[1].plot([0, 1], [0, 1], "--", color=GREY, lw=0.8)
    axes[1].set_title(f"One-vs-rest ROC; macro AUC={metrics['macro_ovr_roc_auc']:.3f}")
    axes[1].legend(loc="lower right", fontsize=8)
    _metric_bars(
        axes[2],
        metrics,
        ("macro_ovr_roc_auc", "balanced_accuracy", "macro_f1"),
        ("Macro OVR AUC", "Balanced acc.", "Macro F1"),
        GREY,
        "Powder-native 3-group nested OOF metrics",
    )
    fig.suptitle(
        f"Powder-native LR. 3-group: Control / Biopsy-negative / Cancer (n={len(result.y_true)})",
        y=1.02,
    )
    _save(fig, output, "fig15_powder_native_three_group_auc_confusion_matrix")


def write_native_performance_figures(
    screening: PairedCrossfitResult,
    three_group: PairedCrossfitResult,
    output: Path,
) -> None:
    _screening_figure(screening, output)
    _three_group_figure(three_group, output)
