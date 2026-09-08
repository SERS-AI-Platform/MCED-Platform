from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Final, Literal

import matplotlib.pyplot as plt
import numpy as np
from boramae_data import (
    COLORS,
    FIG_DIR,
    LABELS,
    SHORT_LABELS,
    TABLE_DIR,
    SubjectSpectrum,
    group_matrix,
)
from scipy.signal import find_peaks

PeakCategory = Literal["common", "differential"]

MIN_PROMINENCE: Final = 0.08
RELATIVE_PROMINENCE: Final = 0.10
MIN_DISTANCE_CM: Final = 18.0
CONSENSUS_TOLERANCE_CM: Final = 12.0


@dataclass(frozen=True, slots=True)
class PeakCriteria:
    group: str
    n_subjects: int
    prominence_threshold: float
    min_distance_cm: float
    consensus_tolerance_cm: float


@dataclass(frozen=True, slots=True)
class GroupPeak:
    group: str
    center: float
    intensity: float
    prominence: float


@dataclass(frozen=True, slots=True)
class PeakConsensus:
    center: float
    category: PeakCategory
    present_groups: tuple[str, ...]
    group_centers: tuple[float | None, ...]
    shaded_start: float
    shaded_end: float


def save_plot(name: str) -> None:
    for ext in ("png", "pdf"):
        plt.savefig(FIG_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close()


def peak_distance_points(grid: np.ndarray) -> int:
    step = float(np.median(np.diff(grid)))
    return max(1, int(round(MIN_DISTANCE_CM / step)))


def detect_group_peaks(
    subjects: list[SubjectSpectrum], grid: np.ndarray
) -> tuple[tuple[PeakCriteria, ...], tuple[GroupPeak, ...]]:
    matrices = {label: group_matrix(subjects, label) for label in LABELS}
    distance = peak_distance_points(grid)
    criteria: list[PeakCriteria] = []
    peaks: list[GroupPeak] = []
    for label in LABELS:
        mean = matrices[label].mean(axis=0)
        threshold = max(MIN_PROMINENCE, float(np.ptp(mean)) * RELATIVE_PROMINENCE)
        criteria.append(
            PeakCriteria(
                label, len(matrices[label]), threshold, MIN_DISTANCE_CM, CONSENSUS_TOLERANCE_CM
            )
        )
        indices, properties = find_peaks(mean, prominence=threshold, distance=distance)
        for index, prominence in zip(indices, properties["prominences"], strict=True):
            peaks.append(
                GroupPeak(
                    group=label,
                    center=round(float(grid[index]), 1),
                    intensity=float(mean[index]),
                    prominence=float(prominence),
                )
            )
    if not criteria:
        msg = "No Boramae clinical groups were available for peak detection."
        raise RuntimeError(msg)
    return tuple(criteria), tuple(peaks)


def cluster_peaks(peaks: tuple[GroupPeak, ...]) -> tuple[PeakConsensus, ...]:
    clusters: list[list[GroupPeak]] = []
    for peak in sorted(peaks, key=lambda item: item.center):
        if clusters:
            current_center = float(np.mean([item.center for item in clusters[-1]]))
            if abs(peak.center - current_center) <= CONSENSUS_TOLERANCE_CM:
                clusters[-1].append(peak)
                continue
        clusters.append([peak])
    consensus: list[PeakConsensus] = []
    for cluster in clusters:
        center = round(float(np.mean([item.center for item in cluster])), 1)
        group_centers: list[float | None] = []
        present_groups: list[str] = []
        for label in LABELS:
            candidates = [item for item in cluster if item.group == label]
            if candidates:
                chosen = min(candidates, key=lambda item: abs(item.center - center))
                group_centers.append(chosen.center)
                present_groups.append(label)
            else:
                group_centers.append(None)
        category: PeakCategory = "common" if len(present_groups) == len(LABELS) else "differential"
        consensus.append(
            PeakConsensus(
                center=center,
                category=category,
                present_groups=tuple(present_groups),
                group_centers=tuple(group_centers),
                shaded_start=round(center - CONSENSUS_TOLERANCE_CM, 1),
                shaded_end=round(center + CONSENSUS_TOLERANCE_CM, 1),
            )
        )
    return tuple(consensus)


def write_peak_tables(
    criteria: tuple[PeakCriteria, ...],
    peaks: tuple[GroupPeak, ...],
    consensus: tuple[PeakConsensus, ...],
) -> None:
    with (TABLE_DIR / "fig04_peak_detection_criteria.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["item", "value"])
        writer.writerow(["spectrum_level", "Group mean of subject-level preprocessed spectra"])
        writer.writerow(
            [
                "preprocessing",
                "trim -> Savitzky-Golay smooth -> baseline correction -> SNV -> model grid",
            ]
        )
        writer.writerow(["local_maximum", "scipy.signal.find_peaks on each group mean spectrum"])
        writer.writerow(
            [
                "prominence_threshold",
                f"max({MIN_PROMINENCE}, {RELATIVE_PROMINENCE} * group_mean_range)",
            ]
        )
        writer.writerow(["min_distance_cm_minus_1", MIN_DISTANCE_CM])
        writer.writerow(
            ["common_peak_definition", "All three groups have a detected peak within +/- tolerance"]
        )
        writer.writerow(["consensus_tolerance_cm_minus_1", CONSENSUS_TOLERANCE_CM])
        writer.writerow(["shaded_area_definition", "Differential peak clusters only"])
    with (TABLE_DIR / "fig04_group_peak_sets.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "clinical_group",
                "n_subjects",
                "peak_cm_minus_1",
                "mean_intensity",
                "prominence",
                "prominence_threshold",
            ]
        )
        thresholds = {item.group: item.prominence_threshold for item in criteria}
        subject_counts = {item.group: item.n_subjects for item in criteria}
        for peak in peaks:
            writer.writerow(
                [
                    peak.group,
                    subject_counts[peak.group],
                    peak.center,
                    peak.intensity,
                    peak.prominence,
                    thresholds[peak.group],
                ]
            )
    with (TABLE_DIR / "fig04_common_and_differential_peaks.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "consensus_peak_cm_minus_1",
                "category",
                "present_groups",
                "control_peak_cm_minus_1",
                "psa_bx_negative_peak_cm_minus_1",
                "prostate_peak_cm_minus_1",
                "shaded_start_cm_minus_1",
                "shaded_end_cm_minus_1",
            ]
        )
        for item in consensus:
            writer.writerow(
                [
                    item.center,
                    item.category,
                    ";".join(item.present_groups),
                    *["" if center is None else center for center in item.group_centers],
                    item.shaded_start if item.category == "differential" else "",
                    item.shaded_end if item.category == "differential" else "",
                ]
            )


def detect_group_peak_sets(
    subjects: list[SubjectSpectrum], grid: np.ndarray
) -> tuple[tuple[PeakCriteria, ...], tuple[GroupPeak, ...], tuple[PeakConsensus, ...]]:
    criteria, peaks = detect_group_peaks(subjects, grid)
    consensus = cluster_peaks(peaks)
    write_peak_tables(criteria, peaks, consensus)
    return criteria, peaks, consensus


def plot_group_peak_sets(
    subjects: list[SubjectSpectrum], grid: np.ndarray, consensus: tuple[PeakConsensus, ...]
) -> None:
    fig, axes = plt.subplots(
        2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]}
    )
    for item in consensus:
        if item.category == "differential":
            axes[0].axvspan(item.shaded_start, item.shaded_end, color="#E78A61", alpha=0.18, lw=0)
            axes[1].axvspan(item.shaded_start, item.shaded_end, color="#E78A61", alpha=0.18, lw=0)
        else:
            axes[0].axvline(item.center, color="#777777", lw=0.6, alpha=0.45)
    for label in LABELS:
        matrix = group_matrix(subjects, label)
        mean = matrix.mean(axis=0)
        sem = matrix.std(axis=0, ddof=1) / np.sqrt(len(matrix))
        axes[0].plot(grid, mean, color=COLORS[label], lw=1.5, label=f"{label} (n={len(matrix)})")
        axes[0].fill_between(
            grid, mean - 1.96 * sem, mean + 1.96 * sem, color=COLORS[label], alpha=0.10
        )
    y_min, y_max = axes[0].get_ylim()
    label_y = y_max - 0.08 * (y_max - y_min)
    for item in consensus:
        if item.category == "differential":
            axes[0].text(
                item.center, label_y, f"{item.center:.0f}", ha="center", va="top", fontsize=7
            )
    criteria_text = (
        "Peak criteria: group mean preprocessed spectra; local maxima; prominence >= max(0.08, 10% of group range); "
        "min distance 18 cm^-1\nCommon = all groups within +/-12 cm^-1; shaded = differential clusters only."
    )
    axes[0].text(
        0.01,
        0.04,
        criteria_text,
        transform=axes[0].transAxes,
        va="bottom",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82},
    )
    axes[0].set_title(
        "Fig04. Boramae group peak sets: common peaks and differential shaded regions"
    )
    axes[0].set_ylabel("SNV intensity")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    axes[0].grid(alpha=0.2)
    for group_index, label in enumerate(LABELS):
        for item in consensus:
            present = item.group_centers[group_index] is not None
            axes[1].scatter(
                item.center,
                group_index,
                s=48,
                color=COLORS[label] if present else "white",
                edgecolor=COLORS[label],
                linewidth=1.0,
            )
    axes[1].set_yticks(range(len(SHORT_LABELS)), SHORT_LABELS)
    axes[1].set_xlabel("Raman shift (cm$^{-1}$)")
    axes[1].set_title("Peak presence by group; orange bands mark differential peak clusters")
    axes[1].grid(axis="x", alpha=0.18)
    save_plot("fig04_group_peak_sets_common_differential")
