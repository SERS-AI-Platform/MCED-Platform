from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from scipy.signal import find_peaks, peak_widths, savgol_filter

from .data import PreprocessingStageMatrices

SMOOTH_WINDOW: Final = 21
POLYORDER: Final = 3
SNR_THRESHOLD: Final = 3.0
RELATIVE_PROMINENCE_FLOOR: Final = 0.05
MINIMUM_DISTANCE_CM: Final = 18.0
MINIMUM_WIDTH_CM: Final = 6.0
MAXIMUM_WIDTH_CM: Final = 90.0
CONSENSUS_TOLERANCE_CM: Final = 12.0
MINIMUM_CONSENSUS_REPLICATES: Final = 4
STAGE_NAMES: Final = ("raw", "smoothed", "baseline_corrected", "snv")


class ReplicateStageShapeError(ValueError):
    """Raised when replicate spectra do not use subject × replicate × feature shape."""


@dataclass(frozen=True, slots=True)
class StageRepeatabilityRow:
    sample_id: int
    modality: str
    stage: str
    median_peak_count: float
    candidate_peak_count: int
    consensus_peak_count: int
    consensus_fraction: float
    median_noise_sigma: float


@dataclass(frozen=True, slots=True)
class PeakRepeatabilityDiagnostics:
    stage_rows: tuple[StageRepeatabilityRow, ...]
    legacy_presence: np.ndarray
    powder_presence: np.ndarray


@dataclass(frozen=True, slots=True)
class _PeakSet:
    centers: np.ndarray
    noise_sigma: float


def _smoothed(values: np.ndarray) -> np.ndarray:
    window = min(SMOOTH_WINDOW, len(values) if len(values) % 2 == 1 else len(values) - 1)
    if window <= POLYORDER:
        return values.copy()
    return savgol_filter(values, window_length=window, polyorder=POLYORDER)


def _robust_sigma(values: np.ndarray) -> float:
    centered = values - np.median(values)
    sigma = 1.4826 * float(np.median(np.abs(centered)))
    floor = max(float(np.ptp(values)) * 1e-6, np.finfo(float).eps)
    return max(sigma, floor)


def _detect_peaks(grid: np.ndarray, values: np.ndarray) -> _PeakSet:
    smooth = _smoothed(values)
    sigma = _robust_sigma(values - smooth)
    step = float(np.median(np.diff(grid)))
    distance = max(1, int(round(MINIMUM_DISTANCE_CM / step)))
    prominence = max(SNR_THRESHOLD * sigma, RELATIVE_PROMINENCE_FLOOR * float(np.ptp(values)))
    indices, _ = find_peaks(values, prominence=prominence, distance=distance)
    if len(indices) == 0:
        return _PeakSet(np.array([], dtype=float), sigma)
    widths = peak_widths(values, indices, rel_height=0.5)
    left = np.interp(widths[2], np.arange(len(grid)), grid)
    right = np.interp(widths[3], np.arange(len(grid)), grid)
    width_cm = right - left
    accepted = (width_cm >= MINIMUM_WIDTH_CM) & (width_cm <= MAXIMUM_WIDTH_CM)
    return _PeakSet(grid[indices[accepted]], sigma)


def _cluster_replicate_peaks(peak_sets: tuple[_PeakSet, ...]) -> tuple[tuple[float, int], ...]:
    clusters: list[list[tuple[float, int]]] = []
    entries = sorted(
        (float(center), replicate)
        for replicate, peak_set in enumerate(peak_sets)
        for center in peak_set.centers
    )
    for center, replicate in entries:
        candidates = [
            (abs(center - float(np.mean([entry[0] for entry in cluster]))), index)
            for index, cluster in enumerate(clusters)
            if replicate not in {entry[1] for entry in cluster}
        ]
        within = [candidate for candidate in candidates if candidate[0] <= CONSENSUS_TOLERANCE_CM]
        if not within:
            clusters.append([(center, replicate)])
            continue
        _, nearest = min(within)
        clusters[nearest].append((center, replicate))
    return tuple(
        (float(np.mean([entry[0] for entry in cluster])), len(cluster)) for cluster in clusters
    )


def _evaluate_subject(
    grid: np.ndarray,
    values: np.ndarray,
    *,
    sample_id: int,
    modality: str,
    stage: str,
) -> tuple[StageRepeatabilityRow, np.ndarray]:
    peak_sets = tuple(_detect_peaks(grid, replicate) for replicate in values)
    clusters = _cluster_replicate_peaks(peak_sets)
    consensus = tuple(center for center, count in clusters if count >= MINIMUM_CONSENSUS_REPLICATES)
    presence = np.zeros(len(grid), dtype=bool)
    for center in consensus:
        presence |= np.abs(grid - center) <= CONSENSUS_TOLERANCE_CM
    candidate_count = len(clusters)
    return (
        StageRepeatabilityRow(
            sample_id=sample_id,
            modality=modality,
            stage=stage,
            median_peak_count=float(np.median([len(peak_set.centers) for peak_set in peak_sets])),
            candidate_peak_count=candidate_count,
            consensus_peak_count=len(consensus),
            consensus_fraction=len(consensus) / candidate_count if candidate_count else 0.0,
            median_noise_sigma=float(np.median([peak_set.noise_sigma for peak_set in peak_sets])),
        ),
        presence,
    )


def _stage_values(stages: PreprocessingStageMatrices, stage: str) -> np.ndarray:
    match stage:
        case "raw":
            return stages.raw
        case "smoothed":
            return stages.smoothed
        case "baseline_corrected":
            return stages.baseline_corrected
        case "snv":
            return stages.snv
        case unreachable:
            raise ReplicateStageShapeError(f"Unknown preprocessing stage: {unreachable}")


def evaluate_peak_repeatability(
    grid: np.ndarray,
    legacy: PreprocessingStageMatrices,
    powder: PreprocessingStageMatrices,
    sample_ids: np.ndarray,
) -> PeakRepeatabilityDiagnostics:
    """Measure replicate-level apparent and 4-of-5 consensus peaks."""
    for stages in (legacy, powder):
        if stages.snv.ndim != 3:
            raise ReplicateStageShapeError(
                "Replicate stage matrices must be three-dimensional: subject × replicate × feature"
            )
    rows: list[StageRepeatabilityRow] = []
    presence_by_modality: list[np.ndarray] = []
    for modality, stages in (("legacy_liquid", legacy), ("powder", powder)):
        modality_presence = np.zeros((len(sample_ids), len(grid)), dtype=bool)
        for stage in STAGE_NAMES:
            stage_matrix = _stage_values(stages, stage)
            for index, sample_id in enumerate(sample_ids):
                row, presence = _evaluate_subject(
                    grid,
                    stage_matrix[index],
                    sample_id=int(sample_id),
                    modality=modality,
                    stage=stage,
                )
                rows.append(row)
                if stage == "snv":
                    modality_presence[index] = presence
        presence_by_modality.append(modality_presence)
    return PeakRepeatabilityDiagnostics(
        stage_rows=tuple(rows),
        legacy_presence=presence_by_modality[0],
        powder_presence=presence_by_modality[1],
    )
