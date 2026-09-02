from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.axes import Axes

from .data import PairedSpectra

BLUE: Final = "#2C6E9B"
PURPLE: Final = "#72528F"
ORANGE: Final = "#A94712"
GREEN: Final = "#496D57"
INK: Final = "#202321"
GRID: Final = "#D8D5CD"
SURFACE: Final = "#FFFEFB"
POWDER_SOFT: Final = "#F6E5D7"
INVERT_CENTER: Final = 936.6
INVERT_WINDOW: Final = (930.0, 944.0)
DISPLAY_WINDOW: Final = (900.0, 970.0)
CLINICAL_GROUPS: Final = ("Control", "Biopsy-negative", "Cancer")
CLINICAL_COLORS: Final = {
    "Control": BLUE,
    "Biopsy-negative": PURPLE,
    "Cancer": ORANGE,
}
GRADE_GROUPS: Final = ("GG1-2", "GG3-5")
GRADE_COLORS: Final = {"GG1-2": GREEN, "GG3-5": ORANGE}


@dataclass(frozen=True, slots=True)
class SpectrumPanel:
    values: np.ndarray
    labels: np.ndarray
    groups: tuple[str, ...]
    colors: Mapping[str, str]
    title: str


def _plot_groups(
    axis: Axes,
    grid: np.ndarray,
    panel: SpectrumPanel,
) -> None:
    for group in panel.groups:
        selected = panel.labels == group
        group_values = panel.values[selected]
        q1, q3 = np.quantile(group_values, (0.25, 0.75), axis=0)
        axis.plot(
            grid,
            np.median(group_values, axis=0),
            color=panel.colors[group],
            linewidth=2.3,
            label=f"{group} (n={int(np.sum(selected))})",
        )
        axis.fill_between(grid, q1, q3, color=panel.colors[group], alpha=0.13)

    axis.axvspan(*INVERT_WINDOW, color=POWDER_SOFT, linewidth=0)
    axis.axvline(INVERT_CENTER, color=INK, linestyle=":", linewidth=1.2)
    axis.set_xlim(*DISPLAY_WINDOW)
    axis.grid(color=GRID, linewidth=0.8)
    axis.legend(frameon=False, fontsize=9, loc="best")
    axis.set_title(panel.title, loc="left", fontweight="bold")


def write_inverted_band_figure(
    spectra: PairedSpectra,
    grade_bands: np.ndarray,
    directory: Path,
) -> None:
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    font_family = next(
        (name for name in ("NanumSquare", "Noto Sans KR") if name in available_fonts),
        "DejaVu Sans",
    )
    plt.rcParams.update(
        {
            "font.family": font_family,
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
    mask = (spectra.grid >= DISPLAY_WINDOW[0]) & (spectra.grid <= DISPLAY_WINDOW[1])
    grid = spectra.grid[mask]
    legacy = spectra.legacy.spectra[:, mask]
    powder = spectra.powder.spectra[:, mask]
    cancer = spectra.clinical_groups == "Cancer"
    panels = (
        SpectrumPanel(
            legacy,
            spectra.clinical_groups,
            CLINICAL_GROUPS,
            CLINICAL_COLORS,
            "A. Liquid · 임상군",
        ),
        SpectrumPanel(
            powder,
            spectra.clinical_groups,
            CLINICAL_GROUPS,
            CLINICAL_COLORS,
            "B. Powder · 임상군",
        ),
        SpectrumPanel(
            legacy[cancer],
            grade_bands[cancer],
            GRADE_GROUPS,
            GRADE_COLORS,
            "C. Liquid · Cancer Grade Group",
        ),
        SpectrumPanel(
            powder[cancer],
            grade_bands[cancer],
            GRADE_GROUPS,
            GRADE_COLORS,
            "D. Powder · Cancer Grade Group",
        ),
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(14.2, 7.6),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    for axis, panel in zip(axes.flat, panels, strict=True):
        _plot_groups(axis, grid, panel)
    axes[1, 0].set_xlabel("Raman shift (cm$^{-1}$)")
    axes[1, 1].set_xlabel("Raman shift (cm$^{-1}$)")
    axes[0, 0].set_ylabel("SNV-normalized intensity (a.u.)")
    axes[1, 0].set_ylabel("SNV-normalized intensity (a.u.)")
    figure.suptitle(
        "936 cm$^{-1}$ 효과 방향 반전 구간의 임상군·Gleason 층화 스펙트럼\n"
        "선=환자 중앙값 · 음영=IQR · 노란 영역=930–944 cm$^{-1}$ "
        "· 점선=936.6 cm$^{-1}$",
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.005,
        "탐색적 관찰: Cancer–Control Hedges' g가 Liquid −0.359에서 Powder +0.073으로 반전",
        ha="center",
        color=INK,
        fontsize=10,
    )
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        figure.savefig(
            directory / f"fig26_inverted_band_group_spectra.{extension}",
            dpi=300,
            bbox_inches="tight",
            facecolor=SURFACE,
        )
    plt.close(figure)

    mobile_figure, mobile_axes = plt.subplots(
        4,
        1,
        figsize=(5.6, 15.2),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    for axis, panel in zip(mobile_axes, panels, strict=True):
        _plot_groups(axis, grid, panel)
        axis.set_xlabel("Raman shift (cm$^{-1}$)")
        axis.set_ylabel("SNV-normalized\nintensity (a.u.)")
        axis.title.set_fontsize(13)
        axis.tick_params(labelsize=10)
        axis.xaxis.label.set_size(12)
        axis.yaxis.label.set_size(12)
        for legend_text in axis.get_legend().get_texts():
            legend_text.set_fontsize(10)
    mobile_figure.suptitle(
        "936 cm$^{-1}$ 효과 방향 반전 구간\n"
        "선=환자 중앙값 · 음영=IQR · 강조 영역=930–944 cm$^{-1}$",
        fontsize=15,
        fontweight="bold",
    )
    mobile_figure.savefig(
        directory / "fig26_inverted_band_group_spectra_mobile.png",
        dpi=300,
        bbox_inches="tight",
        facecolor=SURFACE,
    )
    plt.close(mobile_figure)
