from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from typing import Final, Literal

import matplotlib.pyplot as plt
import numpy as np
from boramae_data import COLORS, FIG_DIR, TABLE_DIR, SubjectSpectrum, group_matrix
from boramae_shaded_peaks import (
    SHADED_HALF_WIDTH_CM,
    SHADED_PEAK_COUNT,
    ShadedPeak,
    rank_shaded_peaks,
)
from scipy.signal import find_peaks

PeakCategory = Literal["common", "differential"]
MIN_PROMINENCE: Final = 0.08
RELATIVE_PROMINENCE: Final = 0.10
MIN_DISTANCE_CM: Final = 18.0
CONSENSUS_TOLERANCE_CM: Final = 12.0


@dataclass(frozen=True, slots=True)
class PeakGroup:
    label: str
    display: str
    color: str
    matrix: np.ndarray


@dataclass(frozen=True, slots=True)
class GroupPeak:
    group_index: int
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


@dataclass(frozen=True, slots=True)
class PeakModeResult:
    name: str
    common_count: int
    differential_count: int
    shaded_count: int


def slug(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", value.lower())).strip("_")


def peak_distance_points(grid: np.ndarray) -> int:
    step = float(np.median(np.diff(grid)))
    return max(1, int(round(MIN_DISTANCE_CM / step)))


def screening_groups(subjects: list[SubjectSpectrum]) -> tuple[PeakGroup, ...]:
    non_cancer = np.vstack(
        [group_matrix(subjects, "Control"), group_matrix(subjects, "Biopsy-negative")]
    )
    cancer = group_matrix(subjects, "Prostate cancer")
    return (
        PeakGroup("Non-cancer", "Non-cancer", "#2C7FB8", non_cancer),
        PeakGroup("Cancer", "Cancer", COLORS["Prostate cancer"], cancer),
    )


def three_groups(subjects: list[SubjectSpectrum]) -> tuple[PeakGroup, ...]:
    return (
        PeakGroup("Control", "Control", COLORS["Control"], group_matrix(subjects, "Control")),
        PeakGroup(
            "Biopsy-negative",
            "Biopsy-negative",
            COLORS["Biopsy-negative"],
            group_matrix(subjects, "Biopsy-negative"),
        ),
        PeakGroup(
            "Prostate cancer",
            "Cancer",
            COLORS["Prostate cancer"],
            group_matrix(subjects, "Prostate cancer"),
        ),
    )


def detect_group_peaks(
    groups: tuple[PeakGroup, ...], grid: np.ndarray
) -> tuple[dict[str, float], tuple[GroupPeak, ...]]:
    distance = peak_distance_points(grid)
    thresholds: dict[str, float] = {}
    peaks: list[GroupPeak] = []
    for group_index, group in enumerate(groups):
        mean = group.matrix.mean(axis=0)
        threshold = max(MIN_PROMINENCE, float(np.ptp(mean)) * RELATIVE_PROMINENCE)
        thresholds[group.label] = threshold
        indices, properties = find_peaks(mean, prominence=threshold, distance=distance)
        for index, prominence in zip(indices, properties["prominences"], strict=True):
            peaks.append(
                GroupPeak(
                    group_index, round(float(grid[index]), 1), float(mean[index]), float(prominence)
                )
            )
    return thresholds, tuple(peaks)


def cluster_peaks(
    groups: tuple[PeakGroup, ...], peaks: tuple[GroupPeak, ...]
) -> tuple[PeakConsensus, ...]:
    clusters: list[list[GroupPeak]] = []
    for peak in sorted(peaks, key=lambda item: item.center):
        if (
            clusters
            and abs(peak.center - float(np.mean([item.center for item in clusters[-1]])))
            <= CONSENSUS_TOLERANCE_CM
        ):
            clusters[-1].append(peak)
        else:
            clusters.append([peak])
    consensus: list[PeakConsensus] = []
    for cluster in clusters:
        center = round(float(np.mean([item.center for item in cluster])), 1)
        group_centers: list[float | None] = []
        present_groups: list[str] = []
        for group_index, group in enumerate(groups):
            candidates = [item for item in cluster if item.group_index == group_index]
            if candidates:
                chosen = min(candidates, key=lambda item: abs(item.center - center))
                group_centers.append(chosen.center)
                present_groups.append(group.label)
            else:
                group_centers.append(None)
        category: PeakCategory = "common" if len(present_groups) == len(groups) else "differential"
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
    stem: str,
    groups: tuple[PeakGroup, ...],
    thresholds: dict[str, float],
    peaks: tuple[GroupPeak, ...],
    consensus: tuple[PeakConsensus, ...],
    shaded_peaks: tuple[ShadedPeak, ...],
) -> None:
    with (TABLE_DIR / f"{stem}_peak_detection_criteria.csv").open(
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
            [
                "common_peak_definition",
                "All plotted groups have a detected peak within +/- tolerance",
            ]
        )
        writer.writerow(["consensus_tolerance_cm_minus_1", CONSENSUS_TOLERANCE_CM])
        writer.writerow(
            [
                "differential_peak_score",
                "Range of plotted group mean intensities at a detected peak cluster",
            ]
        )
        writer.writerow(
            [
                "shaded_area_definition",
                f"Top {SHADED_PEAK_COUNT} differential peak clusters, +/-{SHADED_HALF_WIDTH_CM} cm^-1",
            ]
        )
    with (TABLE_DIR / f"{stem}_group_peak_sets.csv").open(
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
        for peak in peaks:
            group = groups[peak.group_index]
            writer.writerow(
                [
                    group.label,
                    len(group.matrix),
                    peak.center,
                    peak.intensity,
                    peak.prominence,
                    thresholds[group.label],
                ]
            )
    with (TABLE_DIR / f"{stem}_common_and_differential_peaks.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "consensus_peak_cm_minus_1",
                "category",
                "present_groups",
                *[f"{slug(group.label)}_peak_cm_minus_1" for group in groups],
            ]
        )
        for item in consensus:
            writer.writerow(
                [
                    item.center,
                    item.category,
                    ";".join(item.present_groups),
                    *["" if center is None else center for center in item.group_centers],
                ]
            )
    with (TABLE_DIR / f"{stem}_differential_peak_regions.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "consensus_peak_cm_minus_1",
                "intensity_range",
                "present_groups",
                *[f"{slug(group.label)}_peak_cm_minus_1" for group in groups],
                *[f"{slug(group.label)}_intensity_au" for group in groups],
                "shaded_start_cm_minus_1",
                "shaded_end_cm_minus_1",
            ]
        )
        for item in shaded_peaks:
            writer.writerow(
                [
                    item.rank,
                    item.center,
                    item.intensity_range,
                    ";".join(item.present_groups),
                    *["" if center is None else center for center in item.group_centers],
                    *item.group_intensities,
                    item.shaded_start,
                    item.shaded_end,
                ]
            )


def plot_peak_mode(
    title: str,
    output_name: str,
    groups: tuple[PeakGroup, ...],
    grid: np.ndarray,
    consensus: tuple[PeakConsensus, ...],
    shaded_peaks: tuple[ShadedPeak, ...],
) -> None:
    fig, axis = plt.subplots(figsize=(13, 5.6))
    means = tuple(group.matrix.mean(axis=0) for group in groups)
    for item in sorted(shaded_peaks, key=lambda peak: peak.center):
        axis.axvspan(
            item.shaded_start,
            item.shaded_end,
            color="#E78A61",
            alpha=0.18,
            lw=0,
            label="Top differential peak region" if item.rank == 1 else None,
        )
    for item in consensus:
        if item.category == "common":
            axis.axvline(item.center, color="#777777", lw=0.6, alpha=0.35)
    for group in groups:
        mean = group.matrix.mean(axis=0)
        sem = group.matrix.std(axis=0, ddof=1) / np.sqrt(len(group.matrix))
        axis.plot(
            grid, mean, color=group.color, lw=1.5, label=f"{group.display} (n={len(group.matrix)})"
        )
        axis.fill_between(grid, mean - 1.96 * sem, mean + 1.96 * sem, color=group.color, alpha=0.10)
    for item in shaded_peaks:
        for group_index, center in enumerate(item.group_centers):
            if center is not None:
                y_value = float(np.interp(center, grid, means[group_index]))
                axis.scatter(
                    center,
                    y_value,
                    s=36,
                    color=groups[group_index].color,
                    edgecolor="#222222",
                    linewidth=0.8,
                    zorder=5,
                    label="Detected peak apex" if item.rank == 1 and group_index == 0 else None,
                )
    axis.set_title(title)
    axis.set_xlabel("Wavenumber shift (cm$^{-1}$)")
    axis.set_ylabel("Intensity (a.u.)")
    axis.legend(frameon=False, fontsize=8, loc="upper right")
    axis.grid(alpha=0.2)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{output_name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_peak_mode(
    stem: str, title: str, output_name: str, groups: tuple[PeakGroup, ...], grid: np.ndarray
) -> PeakModeResult:
    thresholds, peaks = detect_group_peaks(groups, grid)
    consensus = cluster_peaks(groups, peaks)
    consensus_rows = tuple(
        (item.center, item.present_groups, item.group_centers) for item in consensus
    )
    shaded_peaks = rank_shaded_peaks([group.matrix for group in groups], grid, consensus_rows)
    write_peak_tables(stem, groups, thresholds, peaks, consensus, shaded_peaks)
    plot_peak_mode(title, output_name, groups, grid, consensus, shaded_peaks)
    return PeakModeResult(
        stem,
        sum(1 for item in consensus if item.category == "common"),
        sum(1 for item in consensus if item.category == "differential"),
        len(shaded_peaks),
    )


def generate_peak_mode_outputs(
    subjects: list[SubjectSpectrum], grid: np.ndarray
) -> tuple[PeakModeResult, PeakModeResult]:
    for name in ("fig04_group_peak_sets_common_differential",):
        for ext in ("png", "pdf"):
            (FIG_DIR / f"{name}.{ext}").unlink(missing_ok=True)
    for name in (
        "fig04_peak_detection_criteria",
        "fig04_group_peak_sets",
        "fig04_common_and_differential_peaks",
        "fig04a_screening_feature_importance_top5",
        "fig04b_three_group_feature_importance_top5",
    ):
        (TABLE_DIR / f"{name}.csv").unlink(missing_ok=True)
    screening = build_peak_mode(
        "fig04a_screening",
        "Fig04a. Screening peak sets: Non-cancer vs Cancer",
        "fig04a_screening_peak_sets_common_differential",
        screening_groups(subjects),
        grid,
    )
    three_group = build_peak_mode(
        "fig04b_three_group",
        "Fig04b. 3-group peak sets: Control / Biopsy-negative / Cancer",
        "fig04b_three_group_peak_sets_common_differential",
        three_groups(subjects),
        grid,
    )
    return screening, three_group
