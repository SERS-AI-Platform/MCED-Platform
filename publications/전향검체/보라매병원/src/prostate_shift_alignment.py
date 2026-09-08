from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

PEAK_MATCH_TOLERANCE = 15.0


@dataclass(frozen=True, slots=True)
class PeakMatch:
    stage: str
    clean_peak: float
    prospective_peak: float
    delta: float


@dataclass(frozen=True, slots=True)
class PeakSummary:
    stage: str
    clean_peaks: int
    prospective_peaks: int
    matched_peaks: int
    matched_within_5: int
    clean_unmatched: int
    prospective_unmatched: int
    median_signed_shift: float
    median_absolute_shift: float
    max_absolute_shift: float


def detect_mean_peaks(grid: np.ndarray, spectra: np.ndarray) -> np.ndarray:
    mean = spectra.mean(axis=0)
    prominence = max(0.12, float(np.ptp(mean)) * 0.08)
    indices, _ = find_peaks(mean, prominence=prominence, distance=8)
    return grid[indices]


def match_peaks(
    reference: np.ndarray, candidate: np.ndarray, tolerance: float = 15.0
) -> list[tuple[float, float]]:
    available = set(range(len(candidate)))
    matches: list[tuple[float, float]] = []
    for peak in reference:
        nearby = [index for index in available if abs(float(candidate[index] - peak)) <= tolerance]
        if nearby:
            best = min(nearby, key=lambda index: abs(float(candidate[index] - peak)))
            matches.append((float(peak), float(candidate[best])))
            available.remove(best)
    return matches


def summarize_peaks(
    stage: str,
    clean_peaks: np.ndarray,
    prospective_peaks: np.ndarray,
) -> tuple[PeakSummary, tuple[PeakMatch, ...]]:
    pairs = match_peaks(clean_peaks, prospective_peaks, PEAK_MATCH_TOLERANCE)
    matches = tuple(
        PeakMatch(stage, clean_peak, prospective_peak, prospective_peak - clean_peak)
        for clean_peak, prospective_peak in pairs
    )
    shifts = np.array([match.delta for match in matches], dtype=float)
    summary = PeakSummary(
        stage=stage,
        clean_peaks=len(clean_peaks),
        prospective_peaks=len(prospective_peaks),
        matched_peaks=len(matches),
        matched_within_5=int(np.sum(np.abs(shifts) <= 5.0)),
        clean_unmatched=len(clean_peaks) - len(matches),
        prospective_unmatched=len(prospective_peaks) - len(matches),
        median_signed_shift=float(np.median(shifts)) if len(shifts) else float("nan"),
        median_absolute_shift=float(np.median(np.abs(shifts))) if len(shifts) else float("nan"),
        max_absolute_shift=float(np.max(np.abs(shifts))) if len(shifts) else float("nan"),
    )
    return summary, matches
