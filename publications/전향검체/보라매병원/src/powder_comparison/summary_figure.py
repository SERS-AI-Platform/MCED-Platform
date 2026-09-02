from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.axes import Axes

from powder_comparison.evaluation import ComparisonBlock
from powder_comparison.statistics import MetricComparison

BLUE: Final = "#2C7FB8"
ORANGE: Final = "#D95F02"
PURPLE: Final = "#7A5195"
INK: Final = "#252525"
GREY: Final = "#8A8A8A"
LIGHT_GREY: Final = "#C9CED3"
SERIES_COLORS: Final = (BLUE, ORANGE, PURPLE)
SERIES_HATCHES: Final = ("", "//", "xx")


@dataclass(frozen=True, slots=True)
class BarComparison:
    title: str
    categories: tuple[str, ...]
    series_values: tuple[tuple[float, ...], ...]
    series_labels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SummaryFigureData:
    comparisons: tuple[ComparisonBlock, ComparisonBlock, ComparisonBlock, ComparisonBlock]
    n_subjects: int


def _metric(block: ComparisonBlock, name: str) -> MetricComparison:
    return next(metric for metric in block.metrics if metric.name == name)


def _comparison_panel(axis: Axes, comparison: BarComparison) -> None:
    positions = np.arange(len(comparison.categories))
    width = 0.72 / len(comparison.series_values)
    center = (len(comparison.series_values) - 1) / 2
    for index, (values, label) in enumerate(
        zip(comparison.series_values, comparison.series_labels, strict=True)
    ):
        bars = axis.bar(
            positions + (index - center) * width,
            values,
            width,
            color=SERIES_COLORS[index],
            edgecolor=INK,
            linewidth=0.6,
            hatch=SERIES_HATCHES[index],
            label=label,
        )
        axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    axis.set_xticks(positions, comparison.categories)
    axis.set_ylim(0.0, 1.08)
    axis.set_ylabel("Score")
    axis.set_title(comparison.title, loc="left", fontweight="bold")
    axis.grid(axis="y", color="#E5E5E5", linewidth=0.8)


def _screening_spec(
    transfer: ComparisonBlock, native: ComparisonBlock, *, overall: bool
) -> BarComparison:
    names = ("roc_auc", "sensitivity", "specificity", "balanced_accuracy")
    values = (
        tuple(_metric(transfer, name).legacy_value for name in names),
        tuple(_metric(transfer, name).powder_value for name in names),
    )
    if overall:
        values += (tuple(_metric(native, name).powder_value for name in names),)
    return BarComparison(
        title="A. Cancer Screening LR",
        categories=("AUROC", "민감도", "특이도", "균형정확도"),
        series_values=values,
        series_labels=("기존 액상", "Powder transfer", "Powder-native")
        if overall
        else ("기존 액상", "Powder transfer"),
    )


def _three_group_spec(
    transfer: ComparisonBlock, native: ComparisonBlock, *, overall: bool
) -> BarComparison:
    names = (
        "macro_ovr_roc_auc",
        "control_ovr_roc_auc",
        "biopsy_negative_ovr_roc_auc",
        "cancer_ovr_roc_auc",
    )
    values = (
        tuple(_metric(transfer, name).legacy_value for name in names),
        tuple(_metric(transfer, name).powder_value for name in names),
    )
    if overall:
        values += (tuple(_metric(native, name).powder_value for name in names),)
    return BarComparison(
        title="B. 3군 LR one-vs-rest AUROC",
        categories=("Macro", "Control", "Biopsy-negative", "Cancer"),
        series_values=values,
        series_labels=("기존 액상", "Powder transfer", "Powder-native")
        if overall
        else ("기존 액상", "Powder transfer"),
    )


def _save(fig: plt.Figure, directory: Path, names: tuple[str, ...]) -> None:
    for name in names:
        for extension in ("png", "pdf"):
            fig.savefig(directory / f"{name}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_transfer_figure(data: SummaryFigureData, directory: Path) -> None:
    screening_transfer, screening_native, three_transfer, three_native = data.comparisons
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), layout="constrained")
    _comparison_panel(axes[0], _screening_spec(screening_transfer, screening_native, overall=False))
    _comparison_panel(axes[1], _three_group_spec(three_transfer, three_native, overall=False))
    reclassification = screening_transfer.reclassification
    values = (
        reclassification.improved,
        reclassification.worsened,
        reclassification.stable_correct,
        reclassification.stable_incorrect,
    )
    bars = axes[2].bar(
        ("개선", "악화", "정분류 유지", "오분류 유지"),
        values,
        color=(BLUE, ORANGE, GREY, LIGHT_GREY),
        edgecolor=INK,
        linewidth=0.6,
    )
    axes[2].bar_label(bars, fmt="%d명", padding=3)
    axes[2].set_ylim(0, max(values) * 1.22)
    axes[2].set_ylabel("환자 수")
    axes[2].set_title("C. Cancer Screening 환자별 재분류", loc="left", fontweight="bold")
    axes[2].grid(axis="y", color="#E5E5E5", linewidth=0.8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.90), ncol=2)
    fig.suptitle(
        f"액상 학습 LR의 powder transfer 평가 (n={data.n_subjects})\n"
        "동일 nested 5-fold OOF 환자 | powder 재학습 없음",
        fontsize=16,
    )
    fig.supxlabel(
        f"핵심: Cancer Screening AUROC {_metric(screening_transfer, 'roc_auc').legacy_value:.3f}→"
        f"{_metric(screening_transfer, 'roc_auc').powder_value:.3f}; 개선 {reclassification.improved}명, "
        f"악화 {reclassification.worsened}명",
        fontsize=11,
    )
    _save(fig, directory, ("fig07a_powder_transfer_summary",))


def _write_overall_figure(data: SummaryFigureData, directory: Path) -> None:
    screening_transfer, screening_native, three_transfer, three_native = data.comparisons
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8), layout="constrained")
    _comparison_panel(axes[0], _screening_spec(screening_transfer, screening_native, overall=True))
    _comparison_panel(axes[1], _three_group_spec(three_transfer, three_native, overall=True))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.90), ncol=3)
    fig.suptitle(
        f"액상·powder transfer·powder-native LR 전체 비교 (n={data.n_subjects})\n"
        "동일 환자·nested 5-fold OOF | binary threshold 0.5 | 3군 argmax",
        fontsize=16,
    )
    fig.supxlabel(
        f"핵심: powder-native는 transfer보다 개선됐지만 Cancer Screening AUROC "
        f"{_metric(screening_native, 'roc_auc').powder_value:.3f}(<{_metric(screening_transfer, 'roc_auc').legacy_value:.3f})로 "
        "기존 액상 성능을 복구하지 못함",
        fontsize=11,
    )
    _save(
        fig,
        directory,
        ("fig07b_powder_overall_analysis_summary", "fig07_powder_comparison_summary"),
    )


def write_summary_figures(data: SummaryFigureData, directory: Path) -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    family = next(
        (name for name in ("NanumSquare", "Noto Sans KR") if name in available),
        "DejaVu Sans",
    )
    plt.rcParams["font.family"] = family
    _write_transfer_figure(data, directory)
    _write_overall_figure(data, directory)
