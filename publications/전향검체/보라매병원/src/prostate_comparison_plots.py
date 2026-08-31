from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
from prostate_comparison_model import OofResult
from prostate_shift_alignment import PeakMatch
from sklearn.metrics import RocCurveDisplay

SCREENING_TITLE: Final = "Fig03. Cancer vs Non-cancer: Control + Biopsy-negative vs Cancer"
THREE_GROUP_TITLE: Final = "Fig06. 3-group: Control / Biopsy-negative / Cancer"


@dataclass(frozen=True, slots=True)
class ConfusionMatrixPlot:
    matrix: np.ndarray
    labels: tuple[str, ...]
    title: str


@dataclass(frozen=True, slots=True)
class AlignmentPlotData:
    grid: np.ndarray
    clean: np.ndarray
    prospective: np.ndarray
    aligned_clean: np.ndarray
    aligned_prospective: np.ndarray
    matches: tuple[tuple[PeakMatch, ...], tuple[PeakMatch, ...]]
    shifts: tuple[np.ndarray, np.ndarray]


def save_figure(fig: plt.Figure, output: Path, name: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{name}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_confusion_matrix(axis: plt.Axes, data: ConfusionMatrixPlot) -> None:
    axis.imshow(data.matrix, cmap="Blues")
    axis.set_xticks(range(len(data.labels)), data.labels, rotation=25, ha="right")
    axis.set_yticks(range(len(data.labels)), data.labels)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(data.title)
    for row in range(data.matrix.shape[0]):
        for column in range(data.matrix.shape[1]):
            axis.text(column, row, str(data.matrix[row, column]), ha="center", va="center")


def plot_screening_performance(result: OofResult, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    draw_confusion_matrix(
        axes[0],
        ConfusionMatrixPlot(
            result.confusion, ("Non-cancer", "Cancer"), "Screening confusion matrix"
        ),
    )
    RocCurveDisplay.from_predictions(
        result.y_true,
        result.probabilities[:, 1],
        name=f"ROC-AUC={result.metrics['roc_auc']:.3f}",
        ax=axes[1],
        curve_kwargs={"color": "#D95F02"},
    )
    axes[1].plot([0, 1], [0, 1], "--", color="#777777", lw=0.8)
    axes[1].set_title("Screening ROC")
    axes[1].legend(loc="lower right", fontsize=8)
    metric_names = ["roc_auc", "balanced_accuracy", "sensitivity", "specificity"]
    metric_labels = ["ROC-AUC", "Balanced acc.", "Sensitivity", "Specificity"]
    x = np.arange(len(metric_labels))
    axes[2].bar(x, [result.metrics[name] for name in metric_names], color="#D95F02", alpha=0.85)
    axes[2].set_xticks(x, metric_labels, rotation=25, ha="right")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title("Screening nested OOF metrics")
    axes[2].grid(axis="y", alpha=0.2)
    fig.suptitle(SCREENING_TITLE, y=1.02)
    save_figure(fig, output, "fig03_screening_binary_auc_confusion_matrix")


def plot_three_group_performance(result: OofResult, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1, 1, 1.05]})
    labels = ("Control", "Biopsy-negative", "Cancer")
    draw_confusion_matrix(
        axes[0], ConfusionMatrixPlot(result.confusion, labels, "3-group confusion matrix")
    )
    colors = ("#2C7FB8", "#7A5195", "#D95F02")
    for class_index, (label, color) in enumerate(zip(labels, colors, strict=True)):
        RocCurveDisplay.from_predictions(
            (result.y_true == class_index).astype(int),
            result.probabilities[:, class_index],
            name=label,
            ax=axes[1],
            curve_kwargs={"color": color},
        )
    axes[1].plot([0, 1], [0, 1], "--", color="#777777", lw=0.8)
    axes[1].set_title(f"One-vs-rest ROC; macro AUC={result.metrics['macro_ovr_roc_auc']:.3f}")
    axes[1].legend(loc="lower right", fontsize=8)
    metric_names = ["macro_ovr_roc_auc", "balanced_accuracy", "macro_f1"]
    metric_labels = ["Macro OVR AUC", "Balanced acc.", "Macro F1"]
    x = np.arange(len(metric_labels))
    axes[2].bar(x, [result.metrics[name] for name in metric_names], color="#4D4D4D", alpha=0.85)
    axes[2].set_xticks(x, metric_labels, rotation=25, ha="right")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title("3-group nested OOF metrics")
    axes[2].grid(axis="y", alpha=0.2)
    fig.suptitle(THREE_GROUP_TITLE, y=1.02)
    save_figure(fig, output, "fig06_three_group_auc_confusion_matrix")


def plot_peak_alignment(data: AlignmentPlotData, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    stages = (
        ("Before urea alignment", data.clean, data.prospective, data.matches[0]),
        ("After urea alignment", data.aligned_clean, data.aligned_prospective, data.matches[1]),
    )
    for axis, (title, clean, prospective, matches) in zip(axes[:2], stages, strict=True):
        axis.plot(data.grid, clean.mean(axis=0), color="#3B6FB6", lw=1.3, label="Clean PRO")
        axis.plot(
            data.grid, prospective.mean(axis=0), color="#D95F02", lw=1.3, label="Prospective cancer"
        )
        for match in matches:
            axis.axvspan(
                min(match.clean_peak, match.prospective_peak),
                max(match.clean_peak, match.prospective_peak),
                color="#777777",
                alpha=0.10,
            )
        axis.set_title(f"{title}\nmatched peaks={len(matches)}")
        axis.set_xlabel("Raman shift (cm$^{-1}$)")
        axis.set_ylabel("SNV intensity")
        axis.legend(frameon=False, fontsize=8)
        axis.grid(alpha=0.2)
    bins = np.linspace(-20, 20, 21)
    axes[2].hist(data.shifts[0], bins=bins, color="#3B6FB6", alpha=0.65, label="Clean PRO")
    axes[2].hist(data.shifts[1], bins=bins, color="#D95F02", alpha=0.65, label="Prospective")
    axes[2].axvline(0, color="#222222", lw=0.8)
    axes[2].set_title("Applied urea-anchor shifts")
    axes[2].set_xlabel("Shift (cm$^{-1}$)")
    axes[2].set_ylabel("Replicates")
    axes[2].legend(frameon=False, fontsize=8)
    axes[2].grid(axis="y", alpha=0.2)
    fig.suptitle("Fig07. Clean retrospective vs prospective prostate peak alignment", y=1.02)
    save_figure(fig, output, "fig07_legacy_vs_prospective_peak_alignment")
