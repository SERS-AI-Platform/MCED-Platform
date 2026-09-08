from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

PeakCategory = Literal["common", "differential"]
MIN_PROMINENCE: Final = 0.08
RELATIVE_PROMINENCE: Final = 0.10
MIN_DISTANCE_CM: Final = 18.0
CONSENSUS_TOLERANCE_CM: Final = 12.0


@dataclass(frozen=True, slots=True)
class PeakObservation:
    group_index: int
    center: float
    intensity: float
    prominence: float


@dataclass(frozen=True, slots=True)
class DetectedPeaks:
    labels: tuple[str, ...]
    thresholds: tuple[float, ...]
    observations: tuple[PeakObservation, ...]


@dataclass(frozen=True, slots=True)
class PeakConsensus:
    center: float
    category: PeakCategory
    present_groups: tuple[str, ...]
    group_centers: tuple[float | None, ...]


def detect_peaks(
    labels: tuple[str, ...], matrices: tuple[np.ndarray, ...], grid: np.ndarray
) -> DetectedPeaks:
    distance = max(1, int(round(MIN_DISTANCE_CM / float(np.median(np.diff(grid))))))
    thresholds: list[float] = []
    observations: list[PeakObservation] = []
    for group_index, matrix in enumerate(matrices):
        mean = matrix.mean(axis=0)
        threshold = max(MIN_PROMINENCE, float(np.ptp(mean)) * RELATIVE_PROMINENCE)
        thresholds.append(threshold)
        indices, properties = find_peaks(mean, prominence=threshold, distance=distance)
        for index, prominence in zip(indices, properties["prominences"], strict=True):
            observations.append(
                PeakObservation(
                    group_index=group_index,
                    center=round(float(grid[index]), 1),
                    intensity=float(mean[index]),
                    prominence=float(prominence),
                )
            )
    return DetectedPeaks(labels, tuple(thresholds), tuple(observations))


def cluster_peaks(
    group_labels: tuple[str, ...], peaks: tuple[PeakObservation, ...]
) -> tuple[PeakConsensus, ...]:
    clusters: list[list[PeakObservation]] = []
    for peak in sorted(peaks, key=lambda item: item.center):
        prior_center = float(np.mean([item.center for item in clusters[-1]])) if clusters else None
        if prior_center is not None and abs(peak.center - prior_center) <= CONSENSUS_TOLERANCE_CM:
            clusters[-1].append(peak)
        else:
            clusters.append([peak])
    consensus: list[PeakConsensus] = []
    for cluster in clusters:
        center = round(float(np.mean([item.center for item in cluster])), 1)
        group_centers: list[float | None] = []
        present_groups: list[str] = []
        for group_index, label in enumerate(group_labels):
            candidates = [item for item in cluster if item.group_index == group_index]
            chosen = (
                min(candidates, key=lambda item: abs(item.center - center)) if candidates else None
            )
            group_centers.append(chosen.center if chosen is not None else None)
            if chosen is not None:
                present_groups.append(label)
        category: PeakCategory = (
            "common" if len(present_groups) == len(group_labels) else "differential"
        )
        consensus.append(
            PeakConsensus(center, category, tuple(present_groups), tuple(group_centers))
        )
    return tuple(consensus)


def write_peak_tables(
    detected: DetectedPeaks, consensus: tuple[PeakConsensus, ...], table_dir: Path
) -> None:
    with (table_dir / "fig04_screening_peak_detection_criteria.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["item", "value"])
        writer.writerow(["spectrum_level", "Group mean of subject-level preprocessed spectra"])
        writer.writerow(["prominence_threshold", "max(0.08, 0.10 * group_mean_range)"])
        writer.writerow(["min_distance_cm_minus_1", MIN_DISTANCE_CM])
        writer.writerow(["consensus_tolerance_cm_minus_1", CONSENSUS_TOLERANCE_CM])
        writer.writerow(["common_peak_definition", "Both YNOR and YPAN within tolerance"])
    with (table_dir / "fig04_screening_group_peak_sets.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["clinical_group", "peak_cm_minus_1", "mean_intensity", "prominence", "threshold"]
        )
        for peak in detected.observations:
            writer.writerow(
                [
                    detected.labels[peak.group_index],
                    peak.center,
                    peak.intensity,
                    peak.prominence,
                    detected.thresholds[peak.group_index],
                ]
            )
    with (table_dir / "fig04_screening_common_and_differential_peaks.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "consensus_peak_cm_minus_1",
                "category",
                "present_groups",
                "control_peak_cm_minus_1",
                "cancer_peak_cm_minus_1",
                "shaded_start_cm_minus_1",
                "shaded_end_cm_minus_1",
            ]
        )
        for item in consensus:
            differential = item.category == "differential"
            writer.writerow(
                [
                    item.center,
                    item.category,
                    ";".join(item.present_groups),
                    *["" if center is None else center for center in item.group_centers],
                    item.center - CONSENSUS_TOLERANCE_CM if differential else "",
                    item.center + CONSENSUS_TOLERANCE_CM if differential else "",
                ]
            )


def plot_peak_sets(
    matrices: tuple[np.ndarray, np.ndarray], grid: np.ndarray, figure_dir: Path, table_dir: Path
) -> tuple[int, int]:
    labels = ("Control (YNOR)", "Pancreatic cancer (YPAN)")
    colors = ("#2C7FB8", "#D95F02")
    detected = detect_peaks(labels, matrices, grid)
    consensus = cluster_peaks(labels, detected.observations)
    write_peak_tables(detected, consensus, table_dir)
    fig, axes = plt.subplots(
        2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]}
    )
    for item in consensus:
        if item.category == "differential":
            for axis in axes:
                axis.axvspan(
                    item.center - CONSENSUS_TOLERANCE_CM,
                    item.center + CONSENSUS_TOLERANCE_CM,
                    color="#E78A61",
                    alpha=0.18,
                    lw=0,
                )
        else:
            axes[0].axvline(item.center, color="#777777", lw=0.6, alpha=0.45)
    for matrix, label, color in zip(matrices, labels, colors, strict=True):
        mean = matrix.mean(axis=0)
        sem = matrix.std(axis=0, ddof=1) / np.sqrt(len(matrix))
        axes[0].plot(grid, mean, color=color, lw=1.5, label=f"{label} (n={len(matrix)})")
        axes[0].fill_between(grid, mean - 1.96 * sem, mean + 1.96 * sem, color=color, alpha=0.10)
    for group_index, (label, color) in enumerate(zip(labels, colors, strict=True)):
        for item in consensus:
            present = item.group_centers[group_index] is not None
            axes[1].scatter(
                item.center, group_index, s=48, color=color if present else "white", edgecolor=color
            )
    axes[0].set_title("Fig04. Severance screening peak sets: control vs pancreatic cancer")
    axes[0].set_ylabel("SNV intensity")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)
    axes[1].set_yticks(range(len(labels)), labels)
    axes[1].set_xlabel("Raman shift (cm$^{-1}$)")
    axes[1].set_title("Peak presence by group; orange bands mark differential peak clusters")
    axes[1].grid(axis="x", alpha=0.18)
    name = "fig04_screening_peak_sets_common_differential"
    for suffix in ("png", "pdf"):
        fig.savefig(figure_dir / f"{name}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    common = sum(item.category == "common" for item in consensus)
    differential = sum(item.category == "differential" for item in consensus)
    return common, differential
