from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from .data import PairedSpectra
from .group_separation import (
    GROUPS,
    GroupSeparationDiagnostics,
    PairwiseSeparation,
    evaluate_group_separation,
)

BLUE: Final = "#2C7FB8"
ORANGE: Final = "#D95F02"
INK: Final = "#252525"
GREY: Final = "#8A8A8A"
GRID: Final = "#E5E5E5"


@dataclass(frozen=True, slots=True)
class HeatmapPanel:
    grid: np.ndarray
    rows: tuple[PairwiseSeparation, ...]
    norm: TwoSlopeNorm


def _effect_heatmap(
    axis: Axes,
    panel: HeatmapPanel,
) -> None:
    matrix = np.vstack([row.hedges_g for row in panel.rows])
    axis.imshow(
        matrix,
        aspect="auto",
        cmap="PuOr_r",
        norm=panel.norm,
        interpolation="nearest",
        extent=(
            float(panel.grid[0]),
            float(panel.grid[-1]),
            len(panel.rows) - 0.5,
            -0.5,
        ),
    )
    axis.set_yticks(np.arange(len(panel.rows)), [row.label for row in panel.rows])
    axis.set_xlabel("Raman shift (cm$^{-1}$)")
    axis.set_ylabel("Subgroup pair")


def _separation_bars(axis: Axes, diagnostics: GroupSeparationDiagnostics) -> None:
    positions = np.arange(len(diagnostics.legacy_pairs))
    height = 0.34
    legacy = np.array([row.separation_to_spread for row in diagnostics.legacy_pairs])
    powder = np.array([row.separation_to_spread for row in diagnostics.powder_pairs])
    legacy_bars = axis.barh(
        positions - height / 2,
        legacy,
        height,
        color=BLUE,
        label="기존 액상",
    )
    powder_bars = axis.barh(
        positions + height / 2,
        powder,
        height,
        color=ORANGE,
        hatch="//",
        label="신규 powder",
    )
    axis.bar_label(legacy_bars, fmt="%.2f", padding=3, fontsize=8)
    axis.bar_label(powder_bars, fmt="%.2f", padding=3, fontsize=8)
    axis.axvline(1.0, color=GREY, linestyle=":", linewidth=1.2, label="Ratio = 1")
    axis.set_yticks(positions, [row.label for row in diagnostics.legacy_pairs])
    axis.invert_yaxis()
    axis.set_xlabel("군간 centroid RMS / pooled 군내 RMS")
    axis.set_title("C. 군간 신호 대비 군내 개인차", loc="left", fontweight="bold")
    axis.legend(frameon=False, fontsize=8)
    axis.grid(axis="x", color=GRID, linewidth=0.7)


def _margin_boxplots(
    axis: Axes,
    spectra: PairedSpectra,
    diagnostics: GroupSeparationDiagnostics,
) -> None:
    positions = np.arange(len(GROUPS), dtype=float)
    for margins, offset, color, hatch in (
        (diagnostics.legacy_margins, -0.18, BLUE, ""),
        (diagnostics.powder_margins, 0.18, ORANGE, "//"),
    ):
        selected = tuple(
            margins.margin[spectra.clinical_groups == group] for group in GROUPS
        )
        boxplot = axis.boxplot(
            selected,
            positions=positions + offset,
            widths=0.30,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
        )
        for box in boxplot["boxes"]:
            box.set(facecolor=color, edgecolor=INK, alpha=0.55, hatch=hatch)
        for median in boxplot["medians"]:
            median.set(color=INK, linewidth=1.7)
        for key in ("whiskers", "caps"):
            for artist in boxplot[key]:
                artist.set(color=INK, linewidth=1.0)
        for position, values in zip(positions + offset, selected, strict=True):
            own_closer_rate = float(np.mean(values > 0.0))
            axis.text(
                position,
                0.98,
                f"{own_closer_rate:.0%}",
                transform=axis.get_xaxis_transform(),
                color=color,
                ha="center",
                va="top",
                fontsize=8,
                fontweight="bold",
            )
    axis.axhline(0.0, color=INK, linestyle=":", linewidth=1.2)
    axis.set_xticks(positions, GROUPS)
    axis.set_ylabel("Nearest-other RMS - own-centroid RMS")
    axis.set_title(
        "D. 환자별 leave-one-out centroid margin\n"
        "양수=자기 군 중심에 더 가까움; 상단 숫자=양수 비율",
        loc="left",
        fontweight="bold",
    )
    axis.legend(
        handles=(
            Patch(facecolor=BLUE, edgecolor=INK, alpha=0.55, label="기존 액상"),
            Patch(facecolor=ORANGE, edgecolor=INK, alpha=0.55, hatch="//", label="신규 powder"),
        ),
        frameon=False,
        fontsize=8,
    )
    axis.grid(axis="y", color=GRID, linewidth=0.7)


def _save(fig: Figure, directory: Path) -> None:
    for extension in ("png", "pdf"):
        fig.savefig(
            directory / f"fig18_clinical_group_separation_diagnostics.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def write_group_separation_figure(spectra: PairedSpectra, directory: Path) -> None:
    diagnostics = evaluate_group_separation(spectra)
    all_effects = np.concatenate(
        [row.hedges_g for row in diagnostics.legacy_pairs + diagnostics.powder_pairs]
    )
    effect_limit = max(float(np.max(np.abs(all_effects))), 0.1)
    norm = TwoSlopeNorm(vmin=-effect_limit, vcenter=0.0, vmax=effect_limit)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), layout="constrained")
    _effect_heatmap(axes[0, 0], HeatmapPanel(spectra.grid, diagnostics.legacy_pairs, norm))
    axes[0, 0].set_title("A. 기존 액상: Raman shift별 Hedges' g", loc="left", fontweight="bold")
    _effect_heatmap(axes[0, 1], HeatmapPanel(spectra.grid, diagnostics.powder_pairs, norm))
    axes[0, 1].set_title("B. 신규 powder: Raman shift별 Hedges' g", loc="left", fontweight="bold")
    image = axes[0, 1].images[0]
    fig.colorbar(image, ax=(axes[0, 0], axes[0, 1]), label="표준화 군간 차이 (Hedges' g)")
    _separation_bars(axes[1, 0], diagnostics)
    _margin_boxplots(axes[1, 1], spectra, diagnostics)
    fig.suptitle(
        f"액상·powder의 임상 subgroup spectral separation (paired n={len(spectra.sample_ids)})\n"
        "최종 SNV 환자평균; 동일 색상척도와 군내 변동 기준으로 비교",
        fontsize=16,
    )
    fig.supxlabel(
        "Hedges' g와 거리 지표는 설명용 기술통계이며 nested OOF 분류성능을 대체하지 않음",
        fontsize=10,
    )
    _save(fig, directory)
