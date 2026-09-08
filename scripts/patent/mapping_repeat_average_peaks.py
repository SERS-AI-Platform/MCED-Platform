from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.ndimage import minimum_filter1d
from scipy.signal import find_peaks, savgol_filter


@dataclass(frozen=True, slots=True)
class Peak:
    index: int
    wavenumber: float
    prominence: float
    snr: float
    support_fraction: float = 0.0


def rolling_minimum_corrected(spectrum: np.ndarray, window: int = 101) -> np.ndarray:
    return spectrum - minimum_filter1d(spectrum, size=window, mode="nearest")


def high_frequency_noise(spectrum: np.ndarray) -> float:
    smooth = savgol_filter(spectrum, window_length=31, polyorder=2, mode="interp")
    residual = spectrum - smooth
    centered = residual - np.median(residual)
    return float(1.4826 * np.median(np.abs(centered)))


def detect_peaks(spectrum: np.ndarray, grid: np.ndarray, noise_scale: float) -> tuple[Peak, ...]:
    threshold = max(3.0 * noise_scale, np.finfo(float).eps)
    indices, properties = find_peaks(spectrum, distance=5, prominence=threshold)
    prominences = np.asarray(properties["prominences"], dtype=float)
    return tuple(
        Peak(
            index=int(index),
            wavenumber=float(grid[index]),
            prominence=float(prominence),
            snr=float(prominence / max(noise_scale, np.finfo(float).eps)),
        )
        for index, prominence in zip(indices, prominences, strict=True)
    )


def add_support(
    peaks: tuple[Peak, ...],
    spectra: np.ndarray,
    raw_noise: float,
    grid: np.ndarray,
    tolerance_points: int = 4,
) -> tuple[Peak, ...]:
    if not peaks:
        return peaks
    repeat_peaks = [detect_peaks(row, grid, raw_noise) for row in spectra]
    supported: list[Peak] = []
    for peak in peaks:
        support = sum(
            any(abs(repeat_peak.index - peak.index) <= tolerance_points for repeat_peak in row)
            for row in repeat_peaks
        ) / len(repeat_peaks)
        supported.append(replace(peak, support_fraction=float(support)))
    return tuple(supported)


def reproducible_peak_count(peaks: tuple[Peak, ...], minimum_support: float = 0.5) -> int:
    return sum(peak.support_fraction >= minimum_support for peak in peaks)
