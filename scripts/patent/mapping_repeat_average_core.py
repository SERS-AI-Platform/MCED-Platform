from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openpyxl
from boramae_repeat_average_core import correlation, read_xy, robust_qc
from mapping_repeat_average_peaks import (
    Peak,
    add_support,
    detect_peaks,
    high_frequency_noise,
    rolling_minimum_corrected,
)

REPO = Path("/home/user/SERS-AI")
MAPPING_ROOT = REPO / "data/mapping"
CLINICAL_PATH = MAPPING_ROOT / "clinical_df.xlsx"
GRID_PATH = REPO / "artifacts/usersnet/v1.0.0/common_grid.npy"
GROUP_ORDER = ("Control", "Prostate disease control", "Prostate cancer")
GROUP_MAP = {
    "control": "Control",
    "prostate disease control": "Prostate disease control",
    "prostate": "Prostate cancer",
}
REPLICATE_PATTERN = re.compile(r"^(?:BNOR|BPRO) \d+_(?:ave)?(\d{4})\.CSV$")


@dataclass(frozen=True, slots=True)
class MappingSubject:
    ordinal: int
    label: str
    group: str
    folder: Path
    replicate_paths: tuple[Path, ...]
    average_path: Path
    native_axis: np.ndarray
    native_replicates: np.ndarray
    aligned_replicates: np.ndarray
    existing_average: np.ndarray


@dataclass(frozen=True, slots=True)
class SubjectResult:
    subject: MappingSubject
    keep: np.ndarray
    selected: np.ndarray
    patent_average: np.ndarray
    repeat_noise_floor: float
    expected_mean_noise: float
    single_hf_noise: float
    existing_hf_noise: float
    patent_hf_noise: float
    average_rmse: float
    average_normalized_rmse: float
    average_correlation: float
    method_difference_normalized_rmse: float
    method_difference_correlation: float
    baseline_aligned_normalized_rmse: float
    baseline_aligned_correlation: float
    raw_peak_count_median: float
    existing_peaks: tuple[Peak, ...]
    patent_peaks: tuple[Peak, ...]


def _repeat_number(path: Path) -> int:
    match = REPLICATE_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"unparseable repeat filename: {path.name}")
    return int(match.group(1))


def _replicate_paths(folder: Path, label: str) -> tuple[Path, ...]:
    prefix, number = label.split("_")
    candidates = [
        path
        for path in folder.glob(f"{prefix} {number}_*.CSV")
        if REPLICATE_PATTERN.fullmatch(path.name)
    ]
    ordered = tuple(sorted(candidates, key=_repeat_number))
    if len(ordered) != 121:
        raise ValueError(f"{label} has {len(ordered)} raw repeats; expected 121")
    return ordered


def load_subjects(grid: np.ndarray) -> tuple[list[MappingSubject], dict[str, float | int]]:
    workbook = openpyxl.load_workbook(CLINICAL_PATH, read_only=True, data_only=True)
    rows = list(workbook.active.iter_rows(values_only=True))
    columns = {str(value): index for index, value in enumerate(rows[0]) if value is not None}
    folders = {
        folder.name.replace(" ", "_"): folder
        for date_folder in MAPPING_ROOT.glob("2026*_mapping")
        for folder in date_folder.iterdir()
        if folder.is_dir()
    }
    subjects: list[MappingSubject] = []
    axis_min: list[float] = []
    axis_max: list[float] = []
    axis_steps: list[float] = []
    lengths: set[int] = set()
    signatures: set[tuple[float, float, float]] = set()
    for ordinal, row in enumerate(rows[1:], start=1):
        label = str(row[columns["solum_label"]])
        group = GROUP_MAP[str(row[columns["cohort_group"]])]
        folder = folders[label]
        paths = _replicate_paths(folder, label)
        average_path = folder / f"{label.replace('_', ' ')}_ave.CSV"
        native_axis: np.ndarray | None = None
        native: list[np.ndarray] = []
        aligned: list[np.ndarray] = []
        for path in paths:
            axis, intensity = read_xy(path)
            if native_axis is None:
                native_axis = axis
            native.append(intensity)
            aligned.append(np.interp(grid, axis, intensity))
            axis_min.append(float(axis[0]))
            axis_max.append(float(axis[-1]))
            step = float(np.median(np.diff(axis)))
            axis_steps.append(step)
            lengths.add(len(axis))
            signatures.add((round(float(axis[0]), 5), round(float(axis[-1]), 5), round(step, 8)))
        average_axis, average_intensity = read_xy(average_path)
        subjects.append(
            MappingSubject(
                ordinal=ordinal,
                label=label,
                group=group,
                folder=folder,
                replicate_paths=paths,
                average_path=average_path,
                native_axis=np.asarray(native_axis, dtype=float),
                native_replicates=np.vstack(native),
                aligned_replicates=np.vstack(aligned),
                existing_average=np.interp(grid, average_axis, average_intensity),
            )
        )
    raw_axis = {
        "unique_lengths": sorted(lengths),
        "min_cm-1": min(axis_min),
        "max_cm-1": max(axis_max),
        "step_cm-1": float(np.median(axis_steps)),
        "unique_axis_signature_count": len(signatures),
    }
    return subjects, raw_axis


def analyze_subject(subject: MappingSubject, grid: np.ndarray) -> SubjectResult:
    keep, _ = robust_qc(subject.aligned_replicates)
    selected = subject.aligned_replicates[keep]
    patent_average = selected.mean(axis=0)
    repeat_sd = selected.std(axis=0, ddof=1)
    repeat_noise = float(np.median(repeat_sd))
    average_native = np.loadtxt(subject.average_path, delimiter=",", dtype=np.float64)[:, 1]
    target = subject.native_replicates.mean(axis=0)
    rmse = float(np.sqrt(np.mean((average_native - target) ** 2)))
    normalized = rmse / max(float(np.ptp(target)), 1e-12)
    single_noise_levels = [high_frequency_noise(row) for row in selected]
    single_noise = float(np.median(single_noise_levels))
    existing_noise = high_frequency_noise(subject.existing_average)
    patent_noise = high_frequency_noise(patent_average)
    method_difference = float(np.sqrt(np.mean((patent_average - subject.existing_average) ** 2)))
    method_difference_normalized = method_difference / max(
        float(np.ptp(subject.existing_average)), 1e-12
    )
    existing_baseline_aligned = rolling_minimum_corrected(subject.existing_average)
    patent_baseline_aligned = rolling_minimum_corrected(patent_average)
    baseline_aligned_rmse = float(
        np.sqrt(np.mean((patent_baseline_aligned - existing_baseline_aligned) ** 2))
    ) / max(float(np.ptp(existing_baseline_aligned)), 1e-12)
    raw_peak_counts = [len(detect_peaks(row, grid, single_noise)) for row in selected]
    existing_peaks = add_support(
        detect_peaks(subject.existing_average, grid, existing_noise), selected, single_noise, grid
    )
    patent_peaks = add_support(
        detect_peaks(patent_average, grid, patent_noise), selected, single_noise, grid
    )
    return SubjectResult(
        subject=subject,
        keep=keep,
        selected=selected,
        patent_average=patent_average,
        repeat_noise_floor=repeat_noise,
        expected_mean_noise=repeat_noise / np.sqrt(len(selected)),
        single_hf_noise=single_noise,
        existing_hf_noise=existing_noise,
        patent_hf_noise=patent_noise,
        average_rmse=rmse,
        average_normalized_rmse=normalized,
        average_correlation=correlation(average_native, target),
        method_difference_normalized_rmse=method_difference_normalized,
        method_difference_correlation=correlation(patent_average, subject.existing_average),
        baseline_aligned_normalized_rmse=baseline_aligned_rmse,
        baseline_aligned_correlation=correlation(
            patent_baseline_aligned, existing_baseline_aligned
        ),
        raw_peak_count_median=float(np.median(raw_peak_counts)),
        existing_peaks=existing_peaks,
        patent_peaks=patent_peaks,
    )


def analyze_subjects(subjects: list[MappingSubject], grid: np.ndarray) -> list[SubjectResult]:
    return [analyze_subject(subject, grid) for subject in subjects]


def partial_average_rows(
    results: list[SubjectResult], seed: int = 20260825
) -> list[dict[str, float | int | str]]:
    rng = np.random.default_rng(seed)
    requested = (1, 2, 3, 5, 10, 20, 50, 100)
    rows: list[dict[str, float | int | str]] = []
    for result in results:
        n_max = len(result.selected)
        ns = tuple(n for n in requested if n <= n_max) + (n_max,)
        for n in dict.fromkeys(ns):
            subset_count = 1 if n == n_max else min(50, max(10, n_max))
            subset_means = []
            for _ in range(subset_count):
                indices = rng.choice(n_max, size=n, replace=False)
                subset_means.append(result.selected[indices].mean(axis=0))
            empirical = (
                float(np.median(np.vstack(subset_means).std(axis=0, ddof=1)))
                if subset_count > 1
                else float("nan")
            )
            rows.append(
                {
                    "subject_ordinal": result.subject.ordinal,
                    "group": result.subject.group,
                    "n": n,
                    "subset_count": subset_count,
                    "empirical_noise_scale_median": empirical,
                    "expected_noise_scale": result.repeat_noise_floor / np.sqrt(n),
                    "expected_variance_ratio_1_over_n": 1 / n,
                }
            )
    return rows
