from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Sequence

import numpy as np

SHADED_PEAK_COUNT: Final = 5
SHADED_HALF_WIDTH_CM: Final = 12.0
ConsensusRow = tuple[float, tuple[str, ...], tuple[float | None, ...]]


@dataclass(frozen=True, slots=True)
class ShadedPeak:
    rank: int
    center: float
    intensity_range: float
    present_groups: tuple[str, ...]
    group_centers: tuple[float | None, ...]
    group_intensities: tuple[float, ...]
    shaded_start: float
    shaded_end: float


def rank_shaded_peaks(
    group_matrices: Sequence[np.ndarray], grid: np.ndarray, consensus: Sequence[ConsensusRow]
) -> tuple[ShadedPeak, ...]:
    means = tuple(matrix.mean(axis=0) for matrix in group_matrices)
    scored: list[tuple[float, ConsensusRow, tuple[float, ...]]] = []
    for item in consensus:
        center, _, _ = item
        intensities = tuple(float(np.interp(center, grid, mean)) for mean in means)
        scored.append((max(intensities) - min(intensities), item, intensities))
    return tuple(
        ShadedPeak(
            rank=rank,
            center=center,
            intensity_range=score,
            present_groups=present_groups,
            group_centers=group_centers,
            group_intensities=intensities,
            shaded_start=round(center - SHADED_HALF_WIDTH_CM, 1),
            shaded_end=round(center + SHADED_HALF_WIDTH_CM, 1),
        )
        for rank, (score, (center, present_groups, group_centers), intensities) in enumerate(
            sorted(scored, key=lambda row: row[0], reverse=True)[:SHADED_PEAK_COUNT],
            start=1,
        )
    )
