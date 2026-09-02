from __future__ import annotations

from dataclasses import dataclass
from typing import Final, cast

import numpy as np
from scipy.stats import spearmanr

from powder_comparison.data import PreprocessingStageMatrices

STAGE_NAMES: Final = ("Raw", "Smoothed", "Baseline corrected", "SNV")


@dataclass(frozen=True, slots=True)
class StageShiftRow:
    stage: str
    median_correlation: float
    correlation_q1: float
    correlation_q3: float
    median_nrmse: float
    median_log2_rms_ratio: float


@dataclass(frozen=True, slots=True)
class PreprocessingDiagnostics:
    rows: tuple[StageShiftRow, ...]
    snv_distance: np.ndarray
    probability_shift: np.ndarray
    distance_probability_rho: float
    distance_probability_p_value: float
    peak_positions: tuple[float, ...]
    peak_shift_enrichment: float


def _stage_arrays(stages: PreprocessingStageMatrices) -> tuple[np.ndarray, ...]:
    return (stages.raw, stages.smoothed, stages.baseline_corrected, stages.snv)


def _row_correlation(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    centered_first = first - first.mean(axis=1, keepdims=True)
    centered_second = second - second.mean(axis=1, keepdims=True)
    numerator = np.sum(centered_first * centered_second, axis=1)
    denominator = np.linalg.norm(centered_first, axis=1) * np.linalg.norm(centered_second, axis=1)
    return np.divide(
        numerator,
        denominator,
        out=np.full(len(first), np.nan),
        where=denominator > 0,
    )


def _stage_row(name: str, legacy: np.ndarray, powder: np.ndarray) -> StageShiftRow:
    correlations = _row_correlation(legacy, powder)
    legacy_rms = np.sqrt(np.mean(np.square(legacy), axis=1))
    powder_rms = np.sqrt(np.mean(np.square(powder), axis=1))
    difference_rms = np.sqrt(np.mean(np.square(powder - legacy), axis=1))
    nrmse = np.divide(
        difference_rms,
        legacy_rms,
        out=np.full(len(legacy), np.nan),
        where=legacy_rms > 0,
    )
    rms_ratio = np.divide(
        powder_rms,
        legacy_rms,
        out=np.full(len(legacy), np.nan),
        where=legacy_rms > 0,
    )
    return StageShiftRow(
        stage=name,
        median_correlation=float(np.nanmedian(correlations)),
        correlation_q1=float(np.nanquantile(correlations, 0.25)),
        correlation_q3=float(np.nanquantile(correlations, 0.75)),
        median_nrmse=float(np.nanmedian(nrmse)),
        median_log2_rms_ratio=float(np.nanmedian(np.log2(rms_ratio))),
    )


def _peak_positions(
    peak_names: tuple[str, ...], peak_importance: np.ndarray, *, top_k: int
) -> tuple[float, ...]:
    selected = np.argsort(-peak_importance)[:top_k]
    return tuple(float(peak_names[index].split()[0]) for index in selected)


def _peak_shift_enrichment(
    grid: np.ndarray,
    legacy_snv: np.ndarray,
    powder_snv: np.ndarray,
    positions: tuple[float, ...],
) -> float:
    peak_mask = np.zeros(len(grid), dtype=bool)
    for position in positions:
        peak_mask |= np.abs(grid - position) <= 9.0
    median_difference = np.median(np.abs(powder_snv - legacy_snv), axis=0)
    peak_shift = float(np.mean(median_difference[peak_mask]))
    background_shift = float(np.mean(median_difference[~peak_mask]))
    return peak_shift / max(background_shift, np.finfo(float).eps)


def evaluate_preprocessing_shift(
    grid: np.ndarray,
    legacy: PreprocessingStageMatrices,
    powder: PreprocessingStageMatrices,
    *,
    peak_names: tuple[str, ...],
    peak_importance: np.ndarray,
    legacy_probability: np.ndarray,
    transfer_probability: np.ndarray,
) -> PreprocessingDiagnostics:
    rows = tuple(
        _stage_row(name, legacy_values, powder_values)
        for name, legacy_values, powder_values in zip(
            STAGE_NAMES, _stage_arrays(legacy), _stage_arrays(powder), strict=True
        )
    )
    snv_distance = np.sqrt(np.mean(np.square(powder.snv - legacy.snv), axis=1))
    probability_shift = np.abs(transfer_probability - legacy_probability)
    rho, p_value = cast(tuple[float, float], spearmanr(snv_distance, probability_shift))
    positions = _peak_positions(peak_names, peak_importance, top_k=5)
    return PreprocessingDiagnostics(
        rows=rows,
        snv_distance=snv_distance,
        probability_shift=probability_shift,
        distance_probability_rho=rho,
        distance_probability_p_value=p_value,
        peak_positions=positions,
        peak_shift_enrichment=_peak_shift_enrichment(grid, legacy.snv, powder.snv, positions),
    )
