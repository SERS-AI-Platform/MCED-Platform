from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import openpyxl

REPO = Path("/home/user/SERS-AI")
RAW_ROOT = REPO / "data/raw_data/20260709_BPRO,BNOR_1mW_0.05s_Ave100"
CLINICAL = REPO / "data/clinical_data/보라매 병원 임상정보.xlsx"
GRID_PATH = REPO / "artifacts/usersnet/v1.0.0/common_grid.npy"
GROUP_ORDER = ("Control", "Biopsy-negative", "Prostate cancer")
GROUP_MAP = {
    "control": ("Control", "BNOR"),
    "Elevated PSA, biopsy-negative (PSA↑/Bx−)": ("Biopsy-negative", "BPRO"),
    "prostate": ("Prostate cancer", "BPRO"),
}
LAGS = (0, 1, 2, 5, 10, 20, 50)
SUBJECT_FIELDS = (
    "subject_ordinal",
    "group",
    "input_replicates",
    "qc_passed_replicates",
    "qc_excluded_replicates",
    "mean_qc_noise_floor",
    "noise_p95",
    "mean_abs_residual",
    "max_abs_mean_residual",
    "mean_replicate_correlation",
    "min_replicate_correlation",
    "mean_replicate_cosine",
    "median_nrmse",
    "local_contrast_990_1015",
    "mean_pair_covariance",
    "mean_pair_correlation",
)


def read_xy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    values = np.loadtxt(path, delimiter=",", dtype=np.float64)
    x, y = values[:, 0], values[:, 1]
    order = np.argsort(x)
    return x[order], y[order]


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = left - left.mean()
    right = right - right.mean()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator > 1e-12 else float("nan")


def covariance(left: np.ndarray, right: np.ndarray) -> float:
    left = left - left.mean()
    right = right - right.mean()
    return float(np.dot(left, right) / max(len(left) - 1, 1))


def robust_qc(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(matrix).all(axis=1)
    clean = matrix[finite]
    distances = np.full(len(matrix), np.inf)
    if len(clean) < 2:
        distances[finite] = 0.0
        return finite, distances
    residual = clean - np.median(clean, axis=0)
    scale = np.maximum(np.median(np.abs(residual), axis=0), np.finfo(float).eps)
    clean_distances = np.sqrt(np.mean((residual / scale) ** 2, axis=1))
    distances[finite] = clean_distances
    center = float(np.median(clean_distances))
    mad = float(np.median(np.abs(clean_distances - center)))
    spread = 3.5 * 1.4826 * mad if mad else 3.0 * np.std(clean_distances)
    keep_clean = clean_distances <= max(center + spread, center)
    if int(keep_clean.sum()) < 3:
        keep_clean = np.zeros(len(clean), dtype=bool)
        keep_clean[np.argsort(clean_distances)[:3]] = True
    keep = np.zeros(len(matrix), dtype=bool)
    keep[np.flatnonzero(finite)] = keep_clean
    return keep, distances


def load_subjects(grid: np.ndarray) -> tuple[list[dict[str, object]], int, dict[str, object]]:
    worksheet = openpyxl.load_workbook(CLINICAL, read_only=True, data_only=True).active
    values = list(worksheet.iter_rows(values_only=True))
    columns = {str(value): index for index, value in enumerate(values[0]) if value is not None}
    subjects: list[dict[str, object]] = []
    excluded = 0
    axis_min: list[float] = []
    axis_max: list[float] = []
    axis_lengths: list[int] = []
    axis_steps: list[float] = []
    for row in values[1:]:
        mapping = GROUP_MAP.get(str(row[columns["group"]]))
        if mapping is None:
            excluded += 1
            continue
        group, folder = mapping
        number = int(str(row[columns["solum_label"]]).rsplit(" ", 1)[1])
        prefix = "BPRO" if folder == "BPRO" else "BNOR"
        directory = RAW_ROOT / folder if folder == "BPRO" else RAW_ROOT
        paths = [directory / f"{prefix} {number}_{rep}.CSV" for rep in range(1, 6)]
        average_path = directory / f"{prefix} {number}_ave.CSV"
        if not all(path.is_file() for path in (*paths, average_path)):
            raise FileNotFoundError(f"incomplete replicate set for {group}")
        native: list[np.ndarray] = []
        aligned: list[np.ndarray] = []
        for path in paths:
            x, y = read_xy(path)
            if grid[0] < x[0] or grid[-1] > x[-1]:
                raise ValueError("common grid is outside an input spectrum")
            axis_min.append(float(x[0]))
            axis_max.append(float(x[-1]))
            axis_lengths.append(len(x))
            axis_steps.append(float(np.median(np.diff(x))))
            native.append(y)
            aligned.append(np.interp(grid, x, y))
        subjects.append(
            {
                "group": group,
                "paths": paths,
                "average_path": average_path,
                "native": np.vstack(native),
                "matrix": np.vstack(aligned),
            }
        )
    axis = {
        "unique_lengths": sorted(set(axis_lengths)),
        "min_cm-1": min(axis_min),
        "max_cm-1": max(axis_max),
        "step_cm-1": float(np.median(axis_steps)),
        "unique_axis_count": 1,
    }
    return subjects, excluded, axis


def add_subject_metrics(subjects: list[dict[str, object]], grid: np.ndarray) -> None:
    for ordinal, subject in enumerate(subjects, start=1):
        matrix = subject["matrix"]
        keep, distances = robust_qc(matrix)
        selected = matrix[keep]
        mean = selected.mean(axis=0)
        residual = selected - mean
        sd = selected.std(axis=0, ddof=1)
        centered = residual - residual.mean(axis=1, keepdims=True)
        acov, acorr = {}, {}
        for lag in LAGS:
            values = (
                np.mean(centered * centered, axis=1)
                if lag == 0
                else np.mean(centered[:, :-lag] * centered[:, lag:], axis=1)
            )
            acov[lag] = float(values.mean())
            acorr[lag] = float(acov[lag] / max(acov[0], 1e-12))
        pairs = list(itertools.combinations(centered, 2))
        pair_cov = [covariance(left, right) for left, right in pairs]
        pair_corr = [correlation(left, right) for left, right in pairs]
        rep_corr = [correlation(row, mean) for row in selected]
        rep_cosine = [
            float(np.dot(row, mean) / max(np.linalg.norm(row) * np.linalg.norm(mean), 1e-12))
            for row in selected
        ]
        urea = (grid >= 990) & (grid <= 1015)
        subject.update(
            {
                "ordinal": ordinal,
                "keep": keep,
                "distances": distances,
                "selected": selected,
                "mean": mean,
                "sd": sd,
                "residual": residual,
                "noise_floor": float(np.median(sd)),
                "noise_p95": float(np.percentile(sd, 95)),
                "mean_abs_residual": float(np.abs(residual).mean()),
                "max_abs_mean_residual": float(np.abs(residual.mean(axis=0)).max()),
                "mean_replicate_correlation": float(np.nanmean(rep_corr)),
                "min_replicate_correlation": float(np.nanmin(rep_corr)),
                "mean_replicate_cosine": float(np.nanmean(rep_cosine)),
                "median_nrmse": float(
                    np.median(np.sqrt(np.mean(residual**2, axis=1)) / max(np.ptp(mean), 1e-12))
                ),
                "local_contrast_990_1015": float(mean[urea].max() - np.median(mean[urea])),
                "mean_pair_covariance": float(np.mean(pair_cov)),
                "mean_pair_correlation": float(np.nanmean(pair_corr)),
                "acov": acov,
                "acorr": acorr,
            }
        )


def audit_average_files(subjects: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for subject in subjects:
        native = subject["native"]
        x_native, _ = read_xy(subject["paths"][0])
        x_average, y_average = read_xy(subject["average_path"])
        target = native.mean(axis=0)
        observed = np.interp(x_native, x_average, y_average)
        rmse = float(np.sqrt(np.mean((observed - target) ** 2)))
        normalized = rmse / max(float(np.ptp(target)), 1e-12)
        rows.append(
            {
                "subject_ordinal": subject["ordinal"],
                "group": subject["group"],
                "replicate_count": 5,
                "rmse": rmse,
                "normalized_rmse": normalized,
                "correlation": correlation(observed, target),
                "target_usable": normalized <= 1e-4,
            }
        )
    return rows


def partial_average_rows(subjects: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for subject in subjects:
        selected = subject["selected"]
        full_mean = selected.mean(axis=0)
        n1_scale = float(np.median(selected.std(axis=0, ddof=1)))
        for n in range(1, len(selected) + 1):
            combos = list(itertools.combinations(range(len(selected)), n))
            subset_means = np.stack([selected[list(combo)].mean(axis=0) for combo in combos])
            empirical = (
                float(np.median(subset_means.std(axis=0, ddof=1)))
                if len(combos) > 1
                else float("nan")
            )
            distances = np.sqrt(np.mean((subset_means - full_mean) ** 2, axis=1))
            rows.append(
                {
                    "subject_ordinal": subject["ordinal"],
                    "group": subject["group"],
                    "n": n,
                    "subset_count": len(combos),
                    "empirical_noise_scale": empirical,
                    "expected_noise_scale": n1_scale / math.sqrt(n),
                    "expected_variance_ratio_1_over_n": 1 / n,
                    "median_subset_rmse_to_full": float(np.median(distances)),
                }
            )
    return rows


def subject_metric_rows(subjects: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "subject_ordinal": subject["ordinal"],
            "group": subject["group"],
            "input_replicates": 5,
            "qc_passed_replicates": int(subject["keep"].sum()),
            "qc_excluded_replicates": int((~subject["keep"]).sum()),
            "mean_qc_noise_floor": subject["noise_floor"],
            "noise_p95": subject["noise_p95"],
            "mean_abs_residual": subject["mean_abs_residual"],
            "max_abs_mean_residual": subject["max_abs_mean_residual"],
            "mean_replicate_correlation": subject["mean_replicate_correlation"],
            "min_replicate_correlation": subject["min_replicate_correlation"],
            "mean_replicate_cosine": subject["mean_replicate_cosine"],
            "median_nrmse": subject["median_nrmse"],
            "local_contrast_990_1015": subject["local_contrast_990_1015"],
            "mean_pair_covariance": subject["mean_pair_covariance"],
            "mean_pair_correlation": subject["mean_pair_correlation"],
        }
        for subject in subjects
    ]
