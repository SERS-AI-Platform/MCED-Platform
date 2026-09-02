from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.axes import Axes

from powder_comparison.data import PreprocessingStageMatrices
from powder_comparison.preprocessing_diagnostics import PreprocessingDiagnostics

BLUE: Final = "#2C7FB8"
ORANGE: Final = "#D95F02"
PURPLE: Final = "#7A5195"
INK: Final = "#252525"
GROUP_COLORS: Final = {"Control": BLUE, "Biopsy-negative": PURPLE, "Cancer": ORANGE}


@dataclass(frozen=True, slots=True)
class PreprocessingFigureData:
    grid: np.ndarray
    legacy: PreprocessingStageMatrices
    powder: PreprocessingStageMatrices
    diagnostics: PreprocessingDiagnostics
    clinical_groups: np.ndarray
    patient_ids: np.ndarray


def _configure_font() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    family = next(
        (name for name in ("NanumSquare", "Noto Sans KR") if name in available),
        "DejaVu Sans",
    )
    plt.rcParams["font.family"] = family
    plt.rcParams["axes.unicode_minus"] = False


def _stage_arrays(stages: PreprocessingStageMatrices) -> tuple[np.ndarray, ...]:
    return (stages.raw, stages.smoothed, stages.baseline_corrected, stages.snv)


def _spectral_panel(
    axis: Axes,
    grid: np.ndarray,
    legacy: np.ndarray,
    powder: np.ndarray,
    *,
    title: str,
    ylabel: str,
) -> None:
    for values, label, color in (
        (legacy, "기존 액상", BLUE),
        (powder, "신규 powder", ORANGE),
    ):
        median = np.median(values, axis=0)
        q1, q3 = np.quantile(values, (0.25, 0.75), axis=0)
        axis.plot(grid, median, color=color, linewidth=1.7, label=label)
        axis.fill_between(grid, q1, q3, color=color, alpha=0.13)
    axis.set_title(title, loc="left", fontweight="bold")
    axis.set_xlabel("Raman shift (cm^-1)")
    axis.set_ylabel(ylabel)
    axis.grid(color="#E5E5E5", linewidth=0.7)


def _similarity_panel(axis: Axes, diagnostics: PreprocessingDiagnostics) -> None:
    positions = np.arange(len(diagnostics.rows))
    correlations = np.array([row.median_correlation for row in diagnostics.rows])
    lower = correlations - np.array([row.correlation_q1 for row in diagnostics.rows])
    upper = np.array([row.correlation_q3 for row in diagnostics.rows]) - correlations
    nrmse = np.array([row.median_nrmse for row in diagnostics.rows])
    bars = axis.twinx()
    bars.bar(positions, nrmse, width=0.55, color=ORANGE, alpha=0.25, label="Median NRMSE")
    axis.errorbar(
        positions,
        correlations,
        yerr=np.vstack((lower, upper)),
        color=BLUE,
        marker="o",
        capsize=4,
        linewidth=1.8,
        label="Median Pearson r (IQR)",
    )
    axis.set_xticks(positions, [row.stage for row in diagnostics.rows], rotation=15)
    axis.set_ylim(-1.05, 1.05)
    axis.set_ylabel("환자별 액상–powder Pearson r", color=BLUE)
    bars.set_ylabel("액상 RMS 기준 NRMSE", color=ORANGE)
    axis.set_title("E. 단계별 paired similarity", loc="left", fontweight="bold")
    handles, labels = axis.get_legend_handles_labels()
    bar_handles, bar_labels = bars.get_legend_handles_labels()
    axis.legend(handles + bar_handles, labels + bar_labels, frameon=False, fontsize=8)
    axis.grid(axis="y", color="#E5E5E5", linewidth=0.7)


def _prediction_panel(axis: Axes, data: PreprocessingFigureData) -> None:
    for group, color in GROUP_COLORS.items():
        selected = data.clinical_groups == group
        axis.scatter(
            data.diagnostics.snv_distance[selected],
            data.diagnostics.probability_shift[selected],
            color=color,
            s=34,
            alpha=0.75,
            edgecolor="white",
            linewidth=0.4,
            label=f"{group} (n={int(np.sum(selected))})",
        )
    annotation_count = min(5, len(data.patient_ids))
    selected_ids = np.argsort(-data.diagnostics.probability_shift)[:annotation_count]
    for index in selected_ids:
        axis.annotate(
            str(data.patient_ids[index]),
            (
                data.diagnostics.snv_distance[index],
                data.diagnostics.probability_shift[index],
            ),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )
    axis.set_xlabel("최종 SNV paired RMSE")
    axis.set_ylabel("|Powder transfer p - 액상 p|")
    axis.set_title(
        "F. 전처리 차이와 LR 예측 이동\n"
        f"Spearman ρ={data.diagnostics.distance_probability_rho:.3f}, "
        f"p={data.diagnostics.distance_probability_p_value:.3g}",
        loc="left",
        fontweight="bold",
    )
    axis.legend(frameon=False, fontsize=8)
    axis.grid(color="#E5E5E5", linewidth=0.7)


def _save(fig: plt.Figure, directory: Path) -> None:
    for extension in ("png", "pdf"):
        fig.savefig(
            directory / f"fig16_preprocessing_domain_shift_diagnostics.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def write_preprocessing_diagnostics_figure(data: PreprocessingFigureData, directory: Path) -> None:
    _configure_font()
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), layout="constrained")
    stage_titles = (
        "A. Raw",
        "B. Savitzky–Golay smoothing",
        "C. Rolling-minimum baseline correction",
        "D. SNV + model grid",
    )
    stage_ylabels = ("Intensity (a.u.)", "Intensity (a.u.)", "Intensity (a.u.)", "SNV")
    for axis, legacy, powder, title, ylabel in zip(
        axes.flat[:4],
        _stage_arrays(data.legacy),
        _stage_arrays(data.powder),
        stage_titles,
        stage_ylabels,
        strict=True,
    ):
        _spectral_panel(axis, data.grid, legacy, powder, title=title, ylabel=ylabel)
    for index, position in enumerate(data.diagnostics.peak_positions):
        axes.flat[3].axvspan(
            position - 9.0,
            position + 9.0,
            color=PURPLE,
            alpha=0.12,
            label="Legacy LR Top-5 peak ±9 cm^-1" if index == 0 else None,
        )
    axes.flat[0].legend(frameon=False)
    axes.flat[3].legend(frameon=False, fontsize=8)
    _similarity_panel(axes.flat[4], data.diagnostics)
    _prediction_panel(axes.flat[5], data)
    fig.suptitle(
        f"액상→powder 전처리 domain-shift 역추적 (paired n={len(data.patient_ids)})\n"
        "각 replicate 독립 전처리 후 5회 환자 평균; 선=코호트 중앙값, 음영=IQR",
        fontsize=16,
    )
    fig.supxlabel(
        f"Legacy LR Top-5 peak 구간의 최종 SNV shift enrichment: "
        f"{data.diagnostics.peak_shift_enrichment:.2f}× (구간 밖 대비)",
        fontsize=11,
    )
    _save(fig, directory)
