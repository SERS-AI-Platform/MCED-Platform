from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from mapping_repeat_average_core import GROUP_ORDER, load_subjects, robust_qc
from scipy.signal import savgol_filter

from sers.signal import baseline_correction, snv

GROUP_TO_CLASS = {name: index for index, name in enumerate(GROUP_ORDER)}
CLASS_NAMES = list(GROUP_ORDER)


@dataclass(frozen=True, slots=True)
class PreparedSubject:
    ordinal: int
    group: str
    x: np.ndarray
    all_spectra: np.ndarray
    qc_keep: np.ndarray
    qc_corr: np.ndarray
    qc_rsd_pct: float


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def preprocess_subjects(grid: np.ndarray) -> tuple[list[PreparedSubject], dict[str, object]]:
    subjects, raw_axis = load_subjects(grid)
    prepared: list[PreparedSubject] = []
    for subject in subjects:
        x_native = np.asarray(subject.native_axis, dtype=float)
        trim_mask = (x_native >= float(grid[0])) & (x_native <= float(grid[-1]))
        x_trim = x_native[trim_mask]
        if len(x_trim) < 101:
            raise ValueError(f"subject {subject.ordinal} has too few trimmed points")
        baseline_rows: list[np.ndarray] = []
        snv_rows: list[np.ndarray] = []
        for y_native in np.asarray(subject.native_replicates, dtype=float):
            y_trim = y_native[trim_mask]
            y_smooth = savgol_filter(y_trim, window_length=11, polyorder=3, mode="interp")
            y_base = baseline_correction(y_smooth, window=101)
            y_snv = snv(y_base)
            baseline_rows.append(y_base)
            snv_rows.append(np.interp(grid, x_trim, y_snv))
        baseline_matrix = np.vstack(baseline_rows)
        all_spectra = np.vstack(snv_rows).astype(np.float32)
        finite = np.isfinite(baseline_matrix).all(axis=1) & np.isfinite(all_spectra).all(axis=1)
        center = np.nanmedian(baseline_matrix[finite], axis=0)
        center_centered = center - center.mean()
        center_norm = float(np.linalg.norm(center_centered))
        corr = np.full(len(baseline_matrix), np.nan, dtype=float)
        for index, row in enumerate(baseline_matrix):
            if not finite[index]:
                continue
            row_centered = row - row.mean()
            denominator = float(np.linalg.norm(row_centered) * center_norm)
            corr[index] = float(np.dot(row_centered, center_centered) / denominator) if denominator > 1e-12 else 0.0
        robust_keep, _ = robust_qc(baseline_matrix)
        keep = finite & robust_keep
        mean_baseline = np.nanmean(baseline_matrix, axis=0)
        rsd_pct = float(np.nanmean(np.nanstd(baseline_matrix, axis=0, ddof=1)) / max(abs(float(np.nanmax(mean_baseline))), 1e-12) * 100.0)
        prepared.append(
            PreparedSubject(
                ordinal=subject.ordinal,
                group=subject.group,
                x=grid.copy(),
                all_spectra=all_spectra[finite],
                qc_keep=keep,
                qc_corr=corr,
                qc_rsd_pct=rsd_pct,
            )
        )
    metadata: dict[str, object] = {
        "raw_axis": raw_axis,
        "subject_count": len(prepared),
        "group_counts": {group: sum(item.group == group for item in prepared) for group in GROUP_ORDER},
        "input_repeats": int(sum(len(item.qc_keep) for item in prepared)),
        "model_repeats": int(sum(len(item.all_spectra) for item in prepared)),
        "qc_passed_repeats": int(sum(int(item.qc_keep.sum()) for item in prepared)),
        "qc_passed_per_subject": {
            "min": int(min(item.qc_keep.sum() for item in prepared)),
            "median": float(np.median([item.qc_keep.sum() for item in prepared])),
            "max": int(max(item.qc_keep.sum() for item in prepared)),
        },
        "qc_rule": "finite spectrum and robust median/MAD outlier filter on baseline-corrected repeat set; QC is recorded as an inference audit and all finite repeats remain in the locked 121-repeat evaluation",
        "grid": {
            "min_cm-1": float(grid[0]),
            "max_cm-1": float(grid[-1]),
            "points": int(len(grid)),
            "step_cm-1": float(np.median(np.diff(grid))),
        },
        "preprocessing": {
            "trim": [400.0, 2200.0],
            "smooth": {"method": "Savitzky-Golay", "window": 11, "polyorder": 3},
            "baseline": {"method": "centered rolling minimum", "window": 101},
            "normalization": "SNV",
            "alignment": "linear interpolation onto fixed 402-2198 cm-1 grid",
        },
    }
    return prepared, metadata


def build_rows(subjects: Sequence[PreparedSubject]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = np.vstack([subject.all_spectra for subject in subjects]).astype(np.float32)
    patient_ids = np.concatenate([np.full(len(subject.all_spectra), subject.ordinal, dtype=int) for subject in subjects])
    y_three = np.concatenate([np.full(len(subject.all_spectra), GROUP_TO_CLASS[subject.group], dtype=int) for subject in subjects])
    y_binary = (y_three == GROUP_TO_CLASS["Prostate cancer"]).astype(int)
    repeat_ids = np.concatenate([np.arange(1, len(subject.all_spectra) + 1, dtype=int) for subject in subjects])
    return X, patient_ids, y_binary, y_three, repeat_ids


def patient_training_means(
    X: np.ndarray, y: np.ndarray, patient_ids: np.ndarray, row_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    patients = np.array(sorted(np.unique(patient_ids[row_indices])), dtype=int)
    means = np.vstack([X[row_indices][patient_ids[row_indices] == patient].mean(axis=0) for patient in patients])
    labels = np.asarray([y[row_indices][patient_ids[row_indices] == patient][0] for patient in patients], dtype=int)
    return means, labels, patients
