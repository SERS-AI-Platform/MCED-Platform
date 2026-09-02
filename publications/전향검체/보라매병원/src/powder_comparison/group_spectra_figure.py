from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .data import PairedSpectra

BLUE: Final = "#2C7FB8"
ORANGE: Final = "#D95F02"
PURPLE: Final = "#7A5195"
INK: Final = "#252525"
GRID: Final = "#E5E5E5"
GROUPS: Final = ("Control", "Biopsy-negative", "Cancer")
GROUP_COLORS: Final = {"Control": BLUE, "Biopsy-negative": PURPLE, "Cancer": ORANGE}


@dataclass(frozen=True, slots=True)
class SpectrumBand:
    label: str
    color: str
    linestyle: str
    median: np.ndarray
    q1: np.ndarray
    q3: np.ndarray


def _band(values: np.ndarray, label: str, color: str, linestyle: str = "-") -> SpectrumBand:
    q1, q3 = np.quantile(values, (0.25, 0.75), axis=0)
    return SpectrumBand(
        label=label,
        color=color,
        linestyle=linestyle,
        median=np.median(values, axis=0),
        q1=q1,
        q3=q3,
    )


def _group_bands(values: np.ndarray, groups: np.ndarray) -> tuple[SpectrumBand, ...]:
    return tuple(
        _band(
            values[groups == group],
            f"{group} (n={int(np.sum(groups == group))})",
            GROUP_COLORS[group],
        )
        for group in GROUPS
    )


def _modality_bands(spectra: PairedSpectra, group: str) -> tuple[SpectrumBand, ...]:
    selected = spectra.clinical_groups == group
    return (
        _band(spectra.legacy.spectra[selected], "기존 액상", BLUE),
        _band(spectra.powder.spectra[selected], "신규 powder", ORANGE, "--"),
    )


def _plot_bands(axis: Axes, grid: np.ndarray, bands: tuple[SpectrumBand, ...]) -> None:
    for band in bands:
        axis.plot(
            grid,
            band.median,
            color=band.color,
            linestyle=band.linestyle,
            linewidth=1.7,
            label=band.label,
        )
        axis.fill_between(grid, band.q1, band.q3, color=band.color, alpha=0.12)
    axis.grid(color=GRID, linewidth=0.7)
    axis.legend(frameon=False, fontsize=8, loc="best")


def _spectral_limits(bands: tuple[SpectrumBand, ...]) -> tuple[float, float]:
    lower = min(float(np.min(band.q1)) for band in bands)
    upper = max(float(np.max(band.q3)) for band in bands)
    padding = max((upper - lower) * 0.06, 0.1)
    return lower - padding, upper + padding


def _save(fig: Figure, directory: Path) -> None:
    for extension in ("png", "pdf"):
        fig.savefig(
            directory / f"fig17_clinical_group_spectra_comparison.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def write_group_spectra_figure(spectra: PairedSpectra, directory: Path) -> None:
    legacy_groups = _group_bands(spectra.legacy.spectra, spectra.clinical_groups)
    powder_groups = _group_bands(spectra.powder.spectra, spectra.clinical_groups)
    paired_difference = _group_bands(
        spectra.powder.spectra - spectra.legacy.spectra,
        spectra.clinical_groups,
    )
    modality_bands = tuple(_modality_bands(spectra, group) for group in GROUPS)
    y_limits = _spectral_limits(legacy_groups + powder_groups)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, layout="constrained")
    for axis, bands, title in (
        (axes[0, 0], legacy_groups, "A. 기존 액상: subgroup 간 비교"),
        (axes[0, 1], powder_groups, "B. 신규 powder: subgroup 간 비교"),
        (axes[0, 2], paired_difference, "C. Paired 변화: powder - 액상"),
    ):
        _plot_bands(axis, spectra.grid, bands)
        axis.set_title(title, loc="left", fontweight="bold")
    axes[0, 2].axhline(0.0, color=INK, linestyle=":", linewidth=1.0)
    axes[0, 2].set_ylabel("Paired Δ SNV")

    for axis, group, bands in zip(axes[1], GROUPS, modality_bands, strict=True):
        _plot_bands(axis, spectra.grid, bands)
        count = int(np.sum(spectra.clinical_groups == group))
        axis.set_title(f"{group}: 액상 vs powder (n={count})", loc="left", fontweight="bold")
        axis.set_xlabel("Raman shift (cm$^{-1}$)")

    for axis in (axes[0, 0], axes[0, 1], *axes[1]):
        axis.set_ylim(y_limits)
    axes[0, 0].set_ylabel("최종 SNV intensity")
    axes[1, 0].set_ylabel("최종 SNV intensity")
    fig.suptitle(
        f"임상 subgroup별 최종 SNV 스펙트럼 비교 (paired n={len(spectra.sample_ids)})\n"
        "각 replicate 독립 전처리 후 5회 환자 평균; 선=group 중앙값, 음영=IQR",
        fontsize=16,
    )
    fig.supxlabel(
        "A–C는 측정조건 내 subgroup 형태와 환자별 변화를, D–F는 동일 subgroup의 액상–powder 차이를 비교",
        fontsize=11,
    )
    _save(fig, directory)
