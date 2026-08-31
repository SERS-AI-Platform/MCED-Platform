from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from boramae_repeat_average_core import correlation, read_xy, robust_qc
from mapping_repeat_average_core import MappingSubject
from mapping_repeat_average_peaks import (
    Peak,
    add_support,
    detect_peaks,
    high_frequency_noise,
    reproducible_peak_count,
)
from scipy.ndimage import minimum_filter1d
from scipy.signal import find_peaks, peak_prominences, savgol_filter

LEGACY_REFERENCE_WN = 1001.4
LEGACY_CALIBRATION_WINDOW = 10.0
LEGACY_CALIBRATION_SIGMA = 3.0
LEGACY_TRIM = (400.0, 2200.0)
LEGACY_SMOOTH_WINDOW = 11
LEGACY_SMOOTH_POLY = 3
LEGACY_BASELINE_WINDOW = 101


@dataclass(frozen=True, slots=True)
class PreprocessingComparison:
    subject: MappingSubject
    keep: np.ndarray
    raw_average: np.ndarray
    changed_after_previous_transform: np.ndarray
    legacy_replicates: np.ndarray
    legacy_average: np.ndarray
    calibration_shifts: np.ndarray
    raw_noise: float
    legacy_noise: float
    raw_peaks: tuple[Peak, ...]
    legacy_peaks: tuple[Peak, ...]
    raw_legacy_correlation: float
    transformed_shape_correlation: float


def _calibration_shift(x: np.ndarray, y: np.ndarray) -> float:
    mask = (x >= LEGACY_REFERENCE_WN - LEGACY_CALIBRATION_WINDOW) & (
        x <= LEGACY_REFERENCE_WN + LEGACY_CALIBRATION_WINDOW
    )
    x_window = x[mask]
    y_window = y[mask]
    if len(x_window) < 7:
        return 0.0
    smooth = savgol_filter(y_window, window_length=7, polyorder=2, mode="interp")
    indices, _ = find_peaks(smooth, distance=5)
    if len(indices) == 0:
        return 0.0
    prominences = peak_prominences(smooth, indices)[0]
    distances = x_window[indices] - LEGACY_REFERENCE_WN
    scores = prominences * np.exp(-(distances / LEGACY_CALIBRATION_SIGMA) ** 2)
    best = int(indices[int(np.argmax(scores))])
    if float(np.max(scores)) < 1.0:
        return 0.0
    if 1 <= best < len(x_window) - 1:
        y0, y1, y2 = smooth[best - 1], smooth[best], smooth[best + 1]
        denominator = 2.0 * (2.0 * y1 - y0 - y2)
        if abs(denominator) > 1e-10:
            offset = (y0 - y2) / denominator
            detected = x_window[best] + offset * (x_window[1] - x_window[0])
            return float(LEGACY_REFERENCE_WN - detected)
    return float(LEGACY_REFERENCE_WN - x_window[best])


def _legacy_rolling_baseline(y: np.ndarray) -> np.ndarray:
    return minimum_filter1d(y, size=LEGACY_BASELINE_WINDOW, mode="nearest")


def legacy_preprocess(
    x: np.ndarray, y: np.ndarray, grid: np.ndarray
) -> tuple[np.ndarray, float]:
    shift = _calibration_shift(x, y)
    calibrated_x = x + shift
    mask = (calibrated_x >= LEGACY_TRIM[0]) & (calibrated_x <= LEGACY_TRIM[1])
    trimmed_x = calibrated_x[mask]
    trimmed_y = y[mask]
    smoothed = savgol_filter(
        trimmed_y,
        window_length=LEGACY_SMOOTH_WINDOW,
        polyorder=LEGACY_SMOOTH_POLY,
        mode="interp",
    )
    corrected = smoothed - _legacy_rolling_baseline(smoothed)
    centered = corrected - corrected.mean()
    scale = float(corrected.std())
    normalized = centered / scale if scale > 1e-10 else centered
    return np.interp(grid, trimmed_x, normalized), shift


def _peak_summary(
    peaks: tuple[Peak, ...],
) -> tuple[int, float, float]:
    if not peaks:
        return 0, float("nan"), float("nan")
    snr = np.asarray([peak.snr for peak in peaks], dtype=float)
    return (
        reproducible_peak_count(peaks),
        float(np.median(snr)),
        float(np.max(snr)),
    )


def compare_subject(
    subject: MappingSubject, grid: np.ndarray
) -> PreprocessingComparison:
    keep, _ = robust_qc(subject.aligned_replicates)
    raw_selected = subject.aligned_replicates[keep]
    raw_average = raw_selected.mean(axis=0)
    changed_after_previous_transform, _ = legacy_preprocess(grid, raw_average, grid)
    legacy_rows: list[np.ndarray] = []
    shifts: list[float] = []
    for path, use in zip(subject.replicate_paths, keep, strict=True):
        if not use:
            continue
        x, y = read_xy(path)
        processed, shift = legacy_preprocess(x, y, grid)
        legacy_rows.append(processed)
        shifts.append(shift)
    legacy_replicates = np.vstack(legacy_rows)
    legacy_average = legacy_replicates.mean(axis=0)
    raw_noise = float(np.median([high_frequency_noise(row) for row in raw_selected]))
    legacy_noise = float(
        np.median([high_frequency_noise(row) for row in legacy_replicates])
    )
    raw_peaks = add_support(
        detect_peaks(raw_average, grid, high_frequency_noise(raw_average)),
        raw_selected,
        raw_noise,
        grid,
    )
    legacy_peaks = add_support(
        detect_peaks(legacy_average, grid, high_frequency_noise(legacy_average)),
        legacy_replicates,
        legacy_noise,
        grid,
    )
    return PreprocessingComparison(
        subject=subject,
        keep=keep,
        raw_average=raw_average,
        changed_after_previous_transform=changed_after_previous_transform,
        legacy_replicates=legacy_replicates,
        legacy_average=legacy_average,
        calibration_shifts=np.asarray(shifts, dtype=float),
        raw_noise=raw_noise,
        legacy_noise=legacy_noise,
        raw_peaks=raw_peaks,
        legacy_peaks=legacy_peaks,
        raw_legacy_correlation=correlation(raw_average, legacy_average),
        transformed_shape_correlation=correlation(
            changed_after_previous_transform, legacy_average
        ),
    )


def compare_subjects(
    subjects: list[MappingSubject], grid: np.ndarray
) -> list[PreprocessingComparison]:
    return [compare_subject(subject, grid) for subject in subjects]
