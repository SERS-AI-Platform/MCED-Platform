from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from sklearn.metrics import roc_curve

from powder_comparison.data import PairedSpectra
from powder_comparison.group_separation_figure import write_group_separation_figure
from powder_comparison.group_spectra_figure import write_group_spectra_figure
from powder_comparison.models import PairedCrossfitResult
from powder_comparison.native_performance_figures import write_native_performance_figures
from powder_comparison.preprocessing_figure import (
    PreprocessingFigureData,
    write_preprocessing_diagnostics_figure,
)
from powder_comparison.runner import AnalysisResult
from powder_comparison.signal_noise_figures_a import (
    write_peak_reproducibility_figure,
    write_signal_retention_figure,
)
from powder_comparison.signal_noise_figures_b import (
    write_ablation_figure,
    write_conclusion_figure,
)
from powder_comparison.signal_noise_lot_figure import write_lot_variance_figure
from powder_comparison.statistics import holm_adjust
from powder_comparison.summary_figure import SummaryFigureData, write_summary_figures
from powder_comparison.threshold_figures import write_threshold_figures

BLUE = "#2C7FB8"
ORANGE = "#D95F02"
PURPLE = "#7A5195"
INK = "#252525"
GREY = "#8A8A8A"
GROUP_COLORS = {"Control": BLUE, "Biopsy-negative": PURPLE, "Cancer": ORANGE}


def _configure_style() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    family = next(
        (name for name in ("NanumSquare", "Noto Sans KR") if name in available),
        "DejaVu Sans",
    )
    plt.rcParams.update(
        {
            "font.family": family,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": False,
        }
    )


def _save(fig: plt.Figure, directory: Path, name: str) -> None:
    for extension in ("png", "pdf"):
        fig.savefig(directory / f"{name}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _paired_probability_roc(result: AnalysisResult, directory: Path) -> None:
    truth = result.screening.y_true
    old = result.screening.legacy_probabilities[:, 1]
    new = result.screening.transfer_probabilities[:, 1]
    groups = result.spectra.clinical_groups
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for group in GROUP_COLORS:
        selected = groups == group
        axes[0].scatter(
            old[selected],
            new[selected],
            s=34,
            alpha=0.78,
            color=GROUP_COLORS[group],
            edgecolor="white",
            linewidth=0.5,
            label=f"{group} (n={int(np.sum(selected))})",
        )
    axes[0].plot([0, 1], [0, 1], color=INK, linestyle="--", linewidth=1)
    axes[0].axhline(0.5, color=GREY, linestyle=":", linewidth=1)
    axes[0].axvline(0.5, color=GREY, linestyle=":", linewidth=1)
    axes[0].set(
        xlabel="기존 액상 예측확률", ylabel="신규 powder 예측확률", xlim=(0, 1), ylim=(0, 1)
    )
    axes[0].set_title("동일 환자의 예측확률 비교")
    axes[0].legend(frameon=False, fontsize=8)
    for values, label, color, linestyle in (
        (old, "기존 액상", BLUE, "-"),
        (new, "신규 powder", ORANGE, "--"),
    ):
        false_positive, true_positive, _ = roc_curve(truth, values)
        axes[1].plot(
            false_positive, true_positive, label=label, color=color, linestyle=linestyle, lw=2
        )
    axes[1].plot([0, 1], [0, 1], color=GREY, linestyle=":", linewidth=1)
    axes[1].set(xlabel="1 - 특이도", ylabel="민감도", xlim=(0, 1), ylim=(0, 1))
    axes[1].set_title("동일 액상 학습 LR의 ROC")
    axes[1].legend(frameon=False)
    fig.suptitle(f"기존 액상과 신규 powder의 LR paired 성능 (n={len(truth)})", fontsize=14)
    fig.tight_layout()
    _save(fig, directory, "fig08_powder_paired_probability_roc")


def _order_diagnostics(result: AnalysisResult, directory: Path) -> None:
    order = result.spectra.powder_order
    groups = result.spectra.clinical_groups
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for group in GROUP_COLORS:
        selected = groups == group
        axes[0, 0].scatter(
            order[selected],
            result.screening.transfer_probabilities[selected, 1],
            color=GROUP_COLORS[group],
            s=28,
            alpha=0.75,
            label=group,
        )
        axes[0, 1].scatter(
            order[selected],
            result.order_pc1[selected],
            color=GROUP_COLORS[group],
            s=28,
            alpha=0.75,
        )
    axes[0, 0].axhline(0.5, color=GREY, linestyle="--", linewidth=1)
    axes[0, 0].set(
        title="측정순서별 LR 전이 예측확률", xlabel="재구성 측정순서", ylabel="암 예측확률"
    )
    axes[0, 0].legend(frameon=False)
    axes[0, 1].set(title="측정순서별 spectral PC1", xlabel="재구성 측정순서", ylabel="PC1 score")
    period = [row.period for row in result.tertiles]
    error = [row.error_rate for row in result.tertiles]
    correlation = [row.replicate_correlation_mean for row in result.tertiles]
    axes[1, 0].bar(period, error, color=ORANGE, edgecolor=INK, linewidth=0.6)
    axes[1, 0].set(
        title="BPRO 측정구간별 오분류율", xlabel="측정구간", ylabel="오분류율", ylim=(0, 1)
    )
    axes[1, 1].bar(period, correlation, color=BLUE, edgecolor=INK, linewidth=0.6)
    axes[1, 1].set(
        title="BPRO 측정구간별 반복 상관",
        xlabel="측정구간",
        ylabel="평균 replicate correlation",
        ylim=(0, 1),
    )
    for axis in axes.flat:
        axis.grid(axis="y", color="#E5E5E5", linewidth=0.8)
    fig.suptitle("신규 powder 측정순서 효과 진단 (BNOR 번호순 후 BPRO 번호순)", fontsize=14)
    fig.tight_layout()
    _save(fig, directory, "fig09a_powder_measurement_order_diagnostics")


def _order_forest(result: AnalysisResult, directory: Path) -> None:
    rows = [row for row in result.order_effects if row.scope == "BPRO_label_adjusted"]
    positions = np.arange(len(rows))
    adjusted = holm_adjust(np.array([row.result.p_value for row in rows]))
    colors = [ORANGE if p_value < 0.05 else BLUE for p_value in adjusted]
    fig, axis = plt.subplots(figsize=(9, 5.6))
    axis.scatter([row.result.rho for row in rows], positions, color=colors, s=55)
    axis.axvline(0.0, color=INK, linestyle="--", linewidth=1)
    axis.set_yticks(positions, [row.endpoint for row in rows])
    axis.invert_yaxis()
    axis.set_xlim(-1, 1)
    axis.set_xlabel("임상라벨 층화 Spearman ρ")
    axis.set_title("BPRO 내부 측정순서 효과와 층화 permutation 검정")
    for position, row, p_value in zip(positions, rows, adjusted, strict=True):
        axis.text(row.result.rho, position - 0.2, f"Holm p={p_value:.3f}", fontsize=8)
    axis.grid(axis="x", color="#E5E5E5", linewidth=0.8)
    fig.tight_layout()
    _save(fig, directory, "fig09b_powder_measurement_order_effect")


def _peak_importance(result: AnalysisResult, directory: Path) -> None:
    combined = result.peaks.legacy.global_importance + result.peaks.powder.global_importance
    selected = np.argsort(-combined)[:10][::-1]
    positions = np.arange(len(selected))
    fig, axis = plt.subplots(figsize=(9, 6.2))
    axis.barh(
        positions - 0.18,
        result.peaks.legacy.global_importance[selected],
        height=0.34,
        color=BLUE,
        label="기존 액상",
    )
    axis.barh(
        positions + 0.18,
        result.peaks.powder.global_importance[selected],
        height=0.34,
        color=ORANGE,
        label="신규 powder",
    )
    axis.set_yticks(positions, [result.peaks.legacy.names[index] for index in selected])
    axis.set_xlabel("모델 내 상대 중요도")
    axis.set_title("Cancer Screening LR의 중요 Raman shift 비교")
    axis.legend(frameon=False)
    axis.grid(axis="x", color="#E5E5E5", linewidth=0.8)
    fig.tight_layout()
    _save(fig, directory, "fig10_powder_important_peak_comparison")


def write_powder_native_performance_figures(
    screening: PairedCrossfitResult,
    three_group: PairedCrossfitResult,
    directory: Path,
) -> None:
    _configure_style()
    write_native_performance_figures(screening, three_group, directory)


def write_clinical_group_spectra_figure(spectra: PairedSpectra, directory: Path) -> None:
    _configure_style()
    write_group_spectra_figure(spectra, directory)


def write_clinical_group_separation_figure(spectra: PairedSpectra, directory: Path) -> None:
    _configure_style()
    write_group_separation_figure(spectra, directory)


def write_figures(result: AnalysisResult) -> Path:
    _configure_style()
    directory = result.sources.output_dir / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    write_summary_figures(
        SummaryFigureData(comparisons=result.comparisons, n_subjects=len(result.pairs)),
        directory,
    )
    _paired_probability_roc(result, directory)
    _order_diagnostics(result, directory)
    _order_forest(result, directory)
    _peak_importance(result, directory)
    write_preprocessing_diagnostics_figure(
        PreprocessingFigureData(
            grid=result.spectra.grid,
            legacy=result.spectra.legacy.stages,
            powder=result.spectra.powder.stages,
            diagnostics=result.preprocessing,
            clinical_groups=result.spectra.clinical_groups,
            patient_ids=result.clinical.patient_ids,
        ),
        directory,
    )
    write_group_spectra_figure(result.spectra, directory)
    write_group_separation_figure(result.spectra, directory)
    write_peak_reproducibility_figure(result.spectra, result.signal_noise, directory)
    write_signal_retention_figure(result.signal_noise, directory)
    write_lot_variance_figure(result.spectra.grid, result.signal_noise, directory)
    write_ablation_figure(result.signal_noise, directory)
    write_conclusion_figure(result.spectra.grid, result.signal_noise, directory)
    write_threshold_figures(result, directory)
    write_native_performance_figures(result.screening, result.three_group, directory)
    return directory
