from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from powder_comparison.runner import AnalysisResult
from powder_comparison.statistics import metric_values
from powder_comparison.threshold_analysis import ThresholdStrategy, threshold_strategies

BLUE = "#2C7FB8"
ORANGE = "#D95F02"
PURPLE = "#7A5195"
GREEN = "#1B9E77"
INK = "#252525"
GREY = "#8A8A8A"
GROUP_COLORS = {"Control": BLUE, "Biopsy-negative": PURPLE, "Cancer": ORANGE}


def _save(fig: plt.Figure, directory: Path, name: str) -> None:
    for extension in ("png", "pdf"):
        fig.savefig(directory / f"{name}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _short_name(strategy: ThresholdStrategy) -> str:
    modality = {
        "legacy_liquid": "Liquid",
        "powder_transfer": "Powder transfer",
        "powder_native": "Powder native",
    }[strategy.modality]
    method = "0.5" if strategy.method == "fixed_0.5" else "Youden"
    return f"{modality}\n{method}"


def _confusion_matrices(result: AnalysisResult, directory: Path) -> None:
    strategies = threshold_strategies(result)
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.2))
    for axis, strategy in zip(axes.flat, strategies, strict=True):
        matrix = confusion_matrix(result.screening.y_true, strategy.predictions, labels=(0, 1))
        axis.imshow(matrix, cmap="Blues", vmin=0, vmax=68)
        row_totals = matrix.sum(axis=1)
        for row in range(2):
            for column in range(2):
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]}\n({matrix[row, column] / row_totals[row]:.1%})",
                    ha="center",
                    va="center",
                    color="white" if matrix[row, column] > 34 else INK,
                    fontsize=11,
                )
        axis.set_xticks((0, 1), ("Non-cancer", "Cancer"))
        axis.set_yticks((0, 1), ("Non-cancer", "Cancer"))
        axis.set(xlabel="예측", ylabel="실제", title=_short_name(strategy))
    fig.suptitle("Cancer Screening LR: threshold별 confusion matrix (n=109)", fontsize=15)
    fig.tight_layout()
    _save(fig, directory, "fig11_lr_threshold_confusion_matrices")


def _threshold_diagnostics(result: AnalysisResult, directory: Path) -> None:
    strategies = threshold_strategies(result)
    truth = result.screening.y_true
    groups = result.spectra.clinical_groups
    labels = [_short_name(strategy).replace("\n", " ") for strategy in strategies]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    positions = np.arange(len(strategies))
    width = 0.24
    for offset, metric, color in (
        (-width, "sensitivity", ORANGE),
        (0.0, "specificity", BLUE),
        (width, "balanced_accuracy", GREEN),
    ):
        values = []
        for strategy in strategies:
            metrics = {
                row.name: row.value
                for row in metric_values(
                    truth,
                    strategy.probabilities,
                    "screening_binary",
                    strategy.predictions,
                )
            }
            values.append(metrics[metric])
        axes[0, 0].bar(positions + offset, values, width=width, label=metric, color=color)
    axes[0, 0].set_xticks(positions, labels, rotation=25, ha="right")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_title("Threshold별 분류 성능")
    axes[0, 0].set_ylabel("성능")
    axes[0, 0].legend(frameon=False, fontsize=8)

    bottoms = np.zeros(len(strategies))
    for group, color in (("Control", BLUE), ("Biopsy-negative", PURPLE)):
        counts = [
            int(np.sum((groups == group) & (truth == 0) & (strategy.predictions == 1)))
            for strategy in strategies
        ]
        axes[0, 1].bar(positions, counts, bottom=bottoms, label=group, color=color)
        bottoms += counts
    axes[0, 1].set_xticks(positions, labels, rotation=25, ha="right")
    axes[0, 1].set_title("False positive의 임상군 구성")
    axes[0, 1].set_ylabel("환자 수")
    axes[0, 1].legend(frameon=False)

    bands = ("<4", "4-<10", ">=10")
    band_positions = np.arange(len(bands))
    band_counts = [
        int(np.sum((groups == "Biopsy-negative") & (result.clinical.psa_bands == band)))
        for band in bands
    ]
    bar_width = 0.13
    for index, strategy in enumerate(strategies):
        rates = []
        for band in bands:
            selected = (groups == "Biopsy-negative") & (result.clinical.psa_bands == band)
            rates.append(float(np.mean(strategy.predictions[selected] == 1)))
        axes[1, 0].bar(
            band_positions + (index - 2.5) * bar_width,
            rates,
            width=bar_width,
            label=labels[index],
        )
    axes[1, 0].set_xticks(
        band_positions,
        [f"{band}\n(n={count})" for band, count in zip(bands, band_counts, strict=True)],
    )
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_title("Biopsy-negative의 PSA 구간별 Cancer 오분류율")
    axes[1, 0].set(xlabel="PSA (ng/mL)", ylabel="Cancer 예측 비율")
    axes[1, 0].legend(frameon=False, fontsize=7, ncol=2)

    optimized = result.screening.binary_thresholds
    assert optimized is not None
    folds = np.unique(result.screening.folds)
    liquid = [
        np.unique(optimized.legacy_thresholds[result.screening.folds == fold])[0] for fold in folds
    ]
    powder = [
        np.unique(optimized.native_thresholds[result.screening.folds == fold])[0] for fold in folds
    ]
    axes[1, 1].plot(folds, liquid, marker="o", color=BLUE, label="Liquid training")
    axes[1, 1].plot(folds, powder, marker="s", color=ORANGE, label="Powder training")
    axes[1, 1].axhline(0.5, color=GREY, linestyle="--", label="Fixed 0.5")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_xticks(folds)
    axes[1, 1].set(title="Outer fold별 Youden threshold", xlabel="Outer fold", ylabel="Threshold")
    axes[1, 1].legend(frameon=False)
    for axis in axes.flat:
        axis.grid(axis="y", color="#E5E5E5", linewidth=0.8)
    fig.suptitle("LR threshold 최적화의 성능·오분류 구성·안정성", fontsize=15)
    fig.tight_layout()
    _save(fig, directory, "fig12_lr_threshold_error_diagnostics")


def _patient_trace(result: AnalysisResult, directory: Path) -> None:
    strategies = threshold_strategies(result)
    groups = result.spectra.clinical_groups
    order = np.lexsort((result.spectra.sample_ids, groups))
    patient_ids = result.clinical.patient_ids[order]
    x = np.arange(len(order))
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True, sharey=True)
    for axis, fixed, optimized in zip(
        axes,
        strategies[::2],
        strategies[1::2],
        strict=True,
    ):
        for group, color in GROUP_COLORS.items():
            selected = groups[order] == group
            axis.scatter(
                x[selected],
                fixed.probabilities[order[selected], 1],
                s=26,
                color=color,
                alpha=0.8,
                label=group,
            )
        axis.axhline(0.5, color=GREY, linestyle="--", linewidth=1.2, label="Fixed 0.5")
        axis.scatter(
            x,
            optimized.thresholds[order],
            marker="_",
            s=90,
            color=INK,
            linewidth=1.3,
            label="Fold-specific Youden",
        )
        changed = fixed.predictions[order] != optimized.predictions[order]
        axis.scatter(
            x[changed],
            fixed.probabilities[order[changed], 1],
            s=75,
            facecolors="none",
            edgecolors=GREEN,
            linewidth=1.6,
            label="판정 변경",
        )
        axis.set_title(_short_name(fixed).split("\n")[0])
        axis.set_ylabel("Cancer 확률")
        axis.grid(axis="y", color="#E5E5E5", linewidth=0.8)
    axes[0].legend(frameon=False, ncol=6, fontsize=8)
    axes[-1].set_xticks(x, patient_ids, rotation=90, fontsize=4)
    axes[-1].set_xlabel("임상군·sample number 순으로 정렬한 patient ID")
    fig.suptitle("환자별 LR 확률과 fold-specific threshold 추적 (n=109)", fontsize=15)
    fig.tight_layout()
    _save(fig, directory, "fig13_lr_threshold_patient_trace")


def write_threshold_figures(result: AnalysisResult, directory: Path) -> None:
    _confusion_matrices(result, directory)
    _threshold_diagnostics(result, directory)
    _patient_trace(result, directory)
