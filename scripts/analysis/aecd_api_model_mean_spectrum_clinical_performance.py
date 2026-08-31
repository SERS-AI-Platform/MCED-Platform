#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import httpx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from psycopg2.extensions import connection as PgConnection
from scipy.signal import find_peaks, peak_widths
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(
    os.environ.get(
        "AECD_MEAN_SPECTRUM_OUTDIR",
        str(REPO_ROOT / "notebooks/aecd_api_model_mean_spectrum_outputs"),
    )
)
FIGURE_DIR = OUTPUT_DIR / "figures"
PUBLICATION_FIGURE_DIR = (
    REPO_ROOT / "publications/전향검체/보라매병원/figures/mean_spectrum"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
PUBLICATION_FIGURE_DIR.mkdir(parents=True, exist_ok=True)

SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

API_BASE_URL = os.environ.get("AECD_API_BASE_URL", "http://127.0.0.1:8000")
API_KEY = os.environ.get("AECD_API_KEY")
TARGET_COLUMN = "cohort_group"
POSITIVE_LABEL = "prostate"
NEGATIVE_LABELS = {"control", "prostate disease control"}
RANDOM_STATE = 20260825
SPECTRAL_RANGE_CM1 = (400.0, 2200.0)
CALIBRATION_STANDARD_MATERIAL = "Polystyrene (PS)"
INSTRUMENT_RESOLUTION_FWHM_CM1 = 2.0
QC_MAD_THRESHOLD = 3.5
FIXED_COMPARISON_THRESHOLD = 0.4
THRESHOLD_SELECTION_FOLDS = 5
SNR_PEAK_WLEN = 61
SIGNAL_SNR_CUTOFF = 3.0
SIGNAL_SNR_STRONG_CUTOFF = 5.0

PATENT_SOURCE = REPO_ROOT / "patent/특허_명세서_초안_SERS_반복측정_평균스펙트럼생성.docx"


@dataclass(frozen=True, slots=True)
class StandardMaterialCalibration:
    calibration_date: str
    instrument_name: str
    standard_material: str
    tolerance_cm1: float
    reference_peaks_cm1: tuple[float, ...]
    observed_peaks_cm1: tuple[float, ...]
    global_shift_cm1: float
    alignment_score: float | None
    replicate_shift_std_cm1: float | None


def select_optimal_threshold(
    y_true: np.ndarray,
    probability: np.ndarray,
) -> tuple[float, float]:
    """Select the probability threshold with maximum balanced accuracy."""
    labels = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probability, dtype=np.float64)
    if labels.ndim != 1 or probabilities.ndim != 1:
        raise ValueError("y_true and probability must be one-dimensional")
    if labels.shape != probabilities.shape:
        raise ValueError("y_true and probability must have the same length")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("y_true must contain both binary classes")
    if not np.isfinite(probabilities).all():
        raise ValueError("probability contains non-finite values")

    unique_probabilities = np.unique(probabilities)
    midpoints = (
        (unique_probabilities[:-1] + unique_probabilities[1:]) / 2.0
        if len(unique_probabilities) > 1
        else np.asarray([], dtype=np.float64)
    )
    candidates = np.unique(
        np.concatenate(
            [
                np.asarray([0.0, 1.0], dtype=np.float64),
                midpoints,
            ]
        )
    )
    scores = np.asarray(
        [
            balanced_accuracy_score(labels, probabilities >= threshold)
            for threshold in candidates
        ],
        dtype=np.float64,
    )
    best_score = float(np.max(scores))
    tied = candidates[np.isclose(scores, best_score, rtol=0.0, atol=1e-12)]
    selected = tied[int(np.argmin(np.abs(tied - 0.5)))]
    return float(selected), best_score


def _database_connection() -> PgConnection:
    password = os.environ.get("PGPASSWORD", "")
    if not password:
        raise RuntimeError("PGPASSWORD must be set to load standard calibration records.")
    return psycopg2.connect(
        host=os.environ.get("PGHOST", "localhost"),
        port=int(os.environ.get("PGPORT", "5432")),
        dbname=os.environ.get("PGDATABASE", "aecd_platform"),
        user=os.environ.get("PGUSER", "postgres"),
        password=password,
        connect_timeout=5,
        options="-c default_transaction_read_only=on -c statement_timeout=30000 -c lock_timeout=5000",
    )


def load_standard_material_calibrations() -> tuple[
    dict[tuple[str, str], StandardMaterialCalibration],
    dict[str, str | float | int | list[float]],
]:
    query = """
        SELECT
            calibration.calibration_date,
            instrument.instrument_name,
            calibration.standard_material,
            calibration.tolerance_cm1,
            calibration.reference_peaks_cm1,
            calibration.observed_peaks_cm1,
            calibration.global_shift_cm1,
            calibration.alignment_score,
            calibration.replicate_shift_std_cm1
        FROM measurement.calibrations AS calibration
        JOIN measurement.instruments AS instrument
          ON instrument.instrument_id = calibration.instrument_id
        WHERE calibration.standard_material = %s
          AND calibration.calibration_type = 'raman_shift'
          AND calibration.result = 'pass'
          AND calibration.global_shift_cm1 IS NOT NULL
        ORDER BY calibration.calibration_date, instrument.instrument_name
    """
    with closing(_database_connection()) as connection, connection.cursor() as cursor:
        cursor.execute("SELECT current_database()")
        database_name = str(cursor.fetchone()[0])
        if database_name != "aecd_platform":
            raise RuntimeError(f"Unexpected calibration database: {database_name}")
        cursor.execute(query, (CALIBRATION_STANDARD_MATERIAL,))
        rows = cursor.fetchall()

    calibrations: dict[tuple[str, str], StandardMaterialCalibration] = {}
    reference_catalogs: set[tuple[float, ...]] = set()
    for row in rows:
        calibration_date = pd.Timestamp(row[0]).date().isoformat()
        instrument_name = str(row[1])
        reference_peaks = tuple(float(value) for value in row[4])
        observed_peaks = tuple(float(value) for value in row[5])
        global_shift = float(row[6])
        if len(reference_peaks) == 0 or len(reference_peaks) != len(observed_peaks):
            raise RuntimeError(
                "Standard-material calibration has mismatched reference and observed peaks."
            )
        if not np.isfinite(global_shift):
            raise RuntimeError("Standard-material calibration has a non-finite global shift.")
        peak_errors = np.asarray(observed_peaks) - np.asarray(reference_peaks)
        if not np.allclose(peak_errors, global_shift, atol=0.05):
            raise RuntimeError(
                "Stored standard-material global shift disagrees with peak errors."
            )

        calibration = StandardMaterialCalibration(
            calibration_date=calibration_date,
            instrument_name=instrument_name,
            standard_material=str(row[2]),
            tolerance_cm1=float(row[3]),
            reference_peaks_cm1=reference_peaks,
            observed_peaks_cm1=observed_peaks,
            global_shift_cm1=global_shift,
            alignment_score=float(row[7]) if row[7] is not None else None,
            replicate_shift_std_cm1=float(row[8]) if row[8] is not None else None,
        )
        key = (calibration_date, instrument_name)
        if key in calibrations:
            raise RuntimeError(
                f"Ambiguous standard-material calibration for {calibration_date}/{instrument_name}."
            )
        calibrations[key] = calibration
        reference_catalogs.add(reference_peaks)

    if not calibrations:
        raise RuntimeError(
            f"No passing {CALIBRATION_STANDARD_MATERIAL} Raman-shift calibrations were found."
        )
    if len(reference_catalogs) != 1:
        raise RuntimeError("Standard-material reference peak catalog changes across calibration records.")

    reference_peaks = list(next(iter(reference_catalogs)))
    metadata: dict[str, str | float | int | list[float]] = {
        "database": "aecd_platform",
        "standard_material": CALIBRATION_STANDARD_MATERIAL,
        "calibration_type": "raman_shift",
        "calibration_records": len(calibrations),
        "reference_peaks_cm1": reference_peaks,
        "global_shift_definition": "observed_peak_minus_reference_peak",
        "applied_axis_correction_definition": "x_corrected = x_observed - global_shift_cm1",
        "join_key": "clinical measured_at date + instrument_name",
    }
    return calibrations, metadata


def apply_standard_material_axis_correction(
    source_grid: np.ndarray,
    global_shift_cm1: float,
) -> np.ndarray:
    return np.asarray(source_grid, dtype=np.float64) - float(global_shift_cm1)


def calibration_for_item(
    item: dict,
    calibrations: dict[tuple[str, str], StandardMaterialCalibration],
) -> StandardMaterialCalibration:
    calibration_date = pd.Timestamp(item["measured_at"]).date().isoformat()
    instrument_name = str(item["instrument_name"])
    key = (calibration_date, instrument_name)
    calibration = calibrations.get(key)
    if calibration is None:
        raise RuntimeError(
            f"No unique passing standard calibration for {calibration_date}/{instrument_name}."
        )
    return calibration


def load_api_spectra() -> tuple[list[dict], pd.DataFrame, dict]:
    headers = {"X-API-Key": API_KEY} if API_KEY else {}
    client_options = {
        "base_url": API_BASE_URL,
        "headers": headers,
        "timeout": httpx.Timeout(30.0, connect=5.0),
        "follow_redirects": True,
    }

    items: list[dict] = []
    offset = 0
    with httpx.Client(**client_options) as client:
        health_response = client.get("/health")
        health_response.raise_for_status()
        cohorts_response = client.get("/v1/cohorts")
        cohorts_response.raise_for_status()
        cohorts = pd.DataFrame(cohorts_response.json())

        while True:
            response = client.get(
                "/v1/spectra",
                params={"limit": 1000, "offset": offset},
            )
            response.raise_for_status()
            page = response.json()
            items.extend(page["items"])
            offset += len(page["items"])
            if not page["items"] or offset >= page["total"]:
                break

    if not items:
        raise RuntimeError("The AECD API returned no spectra.")

    return items, cohorts, {
        "health": health_response.json(),
        "spectrum_total_reported": int(page["total"]),
    }


def construct_common_grid(items: list[dict]) -> tuple[np.ndarray, float, int]:
    source_steps = np.asarray(
        [
            np.median(
                np.diff(np.asarray(item["wavenumber"], dtype=np.float64))
            )
            for item in items
        ],
        dtype=np.float64,
    )
    if not np.allclose(source_steps, np.median(source_steps), rtol=0, atol=1e-6):
        raise ValueError("Input spectra do not share a common sampling interval.")

    observed_step = float(np.median(source_steps))
    lower, upper = SPECTRAL_RANGE_CM1
    points = int(round((upper - lower) / observed_step)) + 1
    common_grid = np.linspace(lower, upper, points, dtype=np.float64)
    return common_grid, observed_step, points


def group_and_align_subjects(
    items: list[dict],
    common_grid: np.ndarray,
    calibrations: dict[tuple[str, str], StandardMaterialCalibration],
) -> tuple[list[str], list[str], list[np.ndarray], dict]:
    grouped: defaultdict[str, list[dict]] = defaultdict(list)
    for item in items:
        grouped[str(item["subject_key"])].append(item)

    subject_keys: list[str] = []
    labels: list[str] = []
    replicate_matrices: list[np.ndarray] = []
    skipped_subjects = 0
    replicate_counts: Counter[int] = Counter()
    calibration_shifts: list[float] = []
    calibration_rows: list[dict] = []

    for subject_key, rows in grouped.items():
        rows.sort(
            key=lambda row: (
                int(row.get("replicate_number", 0)),
                int(row.get("measurement_id", 0)),
            )
        )
        subject_labels = sorted(
            {str(row[TARGET_COLUMN]) for row in rows if row.get(TARGET_COLUMN)}
        )
        if len(subject_labels) != 1:
            skipped_subjects += 1
            continue

        aligned_replicates = []
        for row in rows:
            source_grid = np.asarray(row["wavenumber"], dtype=np.float64)
            values = np.asarray(row["intensities"], dtype=np.float64)
            if len(source_grid) != len(values):
                raise ValueError("Wavenumber and intensity lengths differ.")
            if not np.isfinite(values).all():
                raise ValueError("Input spectra contain non-finite intensities.")
            if np.any(np.diff(source_grid) <= 0):
                raise ValueError("Input wavenumber grids must be increasing.")

            calibration = calibration_for_item(row, calibrations)
            calibrated_grid = apply_standard_material_axis_correction(
                source_grid,
                calibration.global_shift_cm1,
            )
            if (
                calibrated_grid[0] > common_grid[0]
                or calibrated_grid[-1] < common_grid[-1]
            ):
                raise ValueError("A spectrum does not cover the common grid range.")
            aligned_replicates.append(
                np.interp(common_grid, calibrated_grid, values)
            )
            applied_correction = -calibration.global_shift_cm1
            calibration_shifts.append(float(applied_correction))
            calibration_rows.append(
                {
                    "measurement_id": int(row["measurement_id"]),
                    "subject_key": str(row["subject_key"]),
                    "measured_at": str(row["measured_at"]),
                    "instrument_name": calibration.instrument_name,
                    "standard_material": calibration.standard_material,
                    "calibration_date": calibration.calibration_date,
                    "global_shift_observed_minus_reference_cm1": calibration.global_shift_cm1,
                    "applied_axis_correction_cm1": applied_correction,
                    "calibration_tolerance_cm1": calibration.tolerance_cm1,
                    "alignment_score": calibration.alignment_score,
                    "replicate_shift_std_cm1": calibration.replicate_shift_std_cm1,
                }
            )

        matrix = np.vstack(aligned_replicates).astype(np.float64)
        subject_keys.append(subject_key)
        labels.append(subject_labels[0])
        replicate_matrices.append(matrix)
        replicate_counts[len(matrix)] += 1

    if not replicate_matrices:
        raise RuntimeError("No subject has a single unambiguous cohort label.")

    shift_array = np.asarray(calibration_shifts, dtype=np.float64)
    abs_shift_array = np.abs(shift_array)
    nonzero_shift_mask = abs_shift_array > 1e-12
    calibration_summary = {
        "standard_material": CALIBRATION_STANDARD_MATERIAL,
        "calibration_source": "measurement.calibrations",
        "join_key": "clinical measured_at date + instrument_name",
        "global_shift_definition": "observed_peak_minus_reference_peak",
        "applied_axis_correction_definition": "x_corrected = x_observed - global_shift_cm1",
        "total_aligned_spectra": int(len(shift_array)),
        "nonzero_shift_spectra": int(np.count_nonzero(nonzero_shift_mask)),
        "zero_shift_spectra": int(np.count_nonzero(~nonzero_shift_mask)),
        "unique_calibration_dates": int(
            len({row["calibration_date"] for row in calibration_rows})
        ),
        "mean_applied_correction_cm1": float(np.mean(shift_array)),
        "median_applied_correction_cm1": float(np.median(shift_array)),
        "p95_abs_applied_correction_cm1": float(np.percentile(abs_shift_array, 95)),
    }
    return subject_keys, labels, replicate_matrices, {
        "skipped_ambiguous_subjects": skipped_subjects,
        "replicate_count_distribution": {
            str(key): int(value) for key, value in sorted(replicate_counts.items())
        },
        "wavenumber_calibration": calibration_summary,
        "calibration_shift_values_cm1": calibration_shifts,
        "standard_material_calibration_rows": calibration_rows,
    }


def qc_repeats(
    replicates: np.ndarray,
    mad_threshold: float = QC_MAD_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, float]:
    replicates = np.asarray(replicates, dtype=np.float64)
    if replicates.ndim != 2 or replicates.shape[0] < 2:
        raise ValueError("At least two repeats are required for QC and noise.")
    if not np.isfinite(replicates).all():
        raise ValueError("QC input contains non-finite values.")

    median_spectrum = np.median(replicates, axis=0)
    residuals = replicates - median_spectrum
    mad_scale = np.median(np.abs(residuals), axis=0)
    safe_scale = np.where(
        mad_scale > np.finfo(float).eps,
        1.4826 * mad_scale,
        1.0,
    )
    distances = np.sqrt(np.mean((residuals / safe_scale) ** 2, axis=1))
    distance_median = float(np.median(distances))
    distance_mad = float(np.median(np.abs(distances - distance_median)))
    if distance_mad > np.finfo(float).eps:
        limit = distance_median + mad_threshold * 1.4826 * distance_mad
    else:
        limit = distance_median + mad_threshold * float(np.std(distances))

    keep = distances <= max(limit, distance_median)
    if int(np.count_nonzero(keep)) < 2:
        keep = np.zeros(len(replicates), dtype=bool)
        keep[np.argsort(distances)[:2]] = True
    return keep, distances, float(limit)


def build_mean_spectra(
    replicate_matrices: list[np.ndarray],
    common_grid: np.ndarray,
) -> tuple[np.ndarray, list[dict], np.ndarray]:
    means: list[np.ndarray] = []
    subject_rows: list[dict] = []
    noise_vectors: list[np.ndarray] = []
    for subject_index, replicates in enumerate(replicate_matrices):
        keep, distances, limit = qc_repeats(replicates)
        passed = replicates[keep]
        representative = passed.mean(axis=0)
        residuals = passed - representative
        noise_sigma = np.std(residuals, axis=0, ddof=1)
        background_mask = (common_grid >= 1800.0) & (common_grid <= 2200.0)
        subject_rows.append(
            {
                "subject_index_internal": int(subject_index),
                "repeats_total": int(len(replicates)),
                "repeats_passed": int(np.count_nonzero(keep)),
                "repeats_excluded": int(len(replicates) - np.count_nonzero(keep)),
                "qc_pass_rate": float(np.mean(keep)),
                "qc_distance_median": float(np.median(distances)),
                "qc_distance_max": float(np.max(distances)),
                "qc_distance_limit": float(limit),
                "noise_floor_global_median": float(np.median(noise_sigma)),
                "noise_floor_background_1800_2200": float(
                    np.median(noise_sigma[background_mask])
                ),
                "noise_sigma_mean": float(np.mean(noise_sigma)),
                "noise_sigma_p95": float(np.percentile(noise_sigma, 95)),
            }
        )
        means.append(representative)
        noise_vectors.append(noise_sigma)

    return (
        np.vstack(means),
        subject_rows,
        np.vstack(noise_vectors),
    )


def resolution_min_distance_points(common_grid: np.ndarray) -> int:
    grid_step = float(np.median(np.diff(common_grid)))
    return max(1, int(np.ceil(INSTRUMENT_RESOLUTION_FWHM_CM1 / grid_step)))


def build_snr_summary(
    common_grid: np.ndarray,
    means: np.ndarray,
    labels: list[str],
    noise_vectors: np.ndarray,
    qc_rows: list[dict],
    reference_peaks_cm1: tuple[float, ...],
) -> pd.DataFrame:
    labels_array = np.asarray(labels)
    rows: list[dict] = []
    min_distance_points = resolution_min_distance_points(common_grid)
    reference_peaks = np.asarray(reference_peaks_cm1, dtype=np.float64)
    grid_step = float(np.median(np.diff(common_grid)))
    for label in sorted(set(labels)):
        mask = labels_array == label
        group_spectrum = np.median(means[mask], axis=0)
        group_noise = np.median(noise_vectors[mask], axis=0)
        group_noise_floor = float(np.median(np.median(noise_vectors[mask], axis=1)))
        median_qc_count = float(
            np.median([qc_rows[index]["repeats_passed"] for index in np.flatnonzero(mask)])
        )
        mean_noise_vector = group_noise / np.sqrt(median_qc_count)
        prominence_min = SIGNAL_SNR_CUTOFF * np.maximum(
            mean_noise_vector,
            np.finfo(float).eps,
        )
        peaks, properties = find_peaks(
            group_spectrum,
            distance=min_distance_points,
            prominence=prominence_min,
            wlen=SNR_PEAK_WLEN,
        )
        order = np.argsort(properties["prominences"])[::-1]
        for rank, property_index in enumerate(order, start=1):
            peak_index = int(peaks[property_index])
            peak_intensity = float(group_spectrum[peak_index])
            prominence = float(properties["prominences"][property_index])
            local_noise = float(group_noise[peak_index])
            mean_noise = float(mean_noise_vector[peak_index])
            rows.append(
                {
                    "label": label,
                    "rank_by_prominence": int(rank),
                    "peak_wavenumber_cm1": float(common_grid[peak_index]),
                    "is_reference_peak": bool(
                        np.any(
                            np.abs(reference_peaks - common_grid[peak_index])
                            <= max(INSTRUMENT_RESOLUTION_FWHM_CM1, 2.0 * grid_step)
                        )
                    ),
                    "peak_intensity_raw": peak_intensity,
                    "peak_prominence_local_contrast": prominence,
                    "noise_sigma_at_peak": local_noise,
                    "noise_floor_global_median": group_noise_floor,
                    "median_qc_repeats": median_qc_count,
                    "snr_raw_over_local_noise": float(peak_intensity / local_noise),
                    "snr_raw_over_noise_floor": float(peak_intensity / group_noise_floor),
                    "snr_contrast_over_local_noise": float(prominence / local_noise),
                    "snr_contrast_over_noise_floor": float(
                        prominence / group_noise_floor
                    ),
                    "snr_contrast_over_mean_noise_independent": float(
                        prominence / mean_noise
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_band_snr_summary(
    common_grid: np.ndarray,
    means: np.ndarray,
    labels: list[str],
    noise_vectors: np.ndarray,
    qc_rows: list[dict],
) -> pd.DataFrame:
    labels_array = np.asarray(labels)
    rows: list[dict] = []
    for label in sorted(set(labels)):
        mask = labels_array == label
        signal = np.median(means[mask], axis=0)
        local_baseline = (
            pd.Series(signal)
            .rolling(window=SNR_PEAK_WLEN, center=True, min_periods=1)
            .median()
            .to_numpy()
        )
        local_contrast = np.maximum(signal - local_baseline, 0.0)
        noise = np.median(noise_vectors[mask], axis=0)
        median_qc_count = float(
            np.median(
                [
                    qc_rows[index]["repeats_passed"]
                    for index in np.flatnonzero(mask)
                ]
            )
        )
        snr = local_contrast / noise
        mean_snr = snr * np.sqrt(median_qc_count)
        for index, wavenumber in enumerate(common_grid):
            rows.append(
                {
                    "label": label,
                    "wavenumber_cm1": float(wavenumber),
                    "signal_raw_intensity": float(signal[index]),
                    "local_baseline": float(local_baseline[index]),
                    "local_signal_contrast": float(local_contrast[index]),
                    "noise_sigma_rep": float(noise[index]),
                    "noise_sigma_mean_independent": float(
                        noise[index] / np.sqrt(median_qc_count)
                    ),
                    "median_qc_repeats": median_qc_count,
                    "snr_contrast_local": float(snr[index]),
                    "snr_contrast_mean_independent": float(mean_snr[index]),
                    "signal_band_snr3": bool(snr[index] >= SIGNAL_SNR_CUTOFF),
                    "strong_signal_band_snr5": bool(
                        snr[index] >= SIGNAL_SNR_STRONG_CUTOFF
                    ),
                    "signal_band_mean_independent_snr3": bool(
                        mean_snr[index] >= SIGNAL_SNR_CUTOFF
                    ),
                    "strong_signal_band_mean_independent_snr5": bool(
                        mean_snr[index] >= SIGNAL_SNR_STRONG_CUTOFF
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_resolution_aware_peak_registry(
    common_grid: np.ndarray,
    band_frame: pd.DataFrame,
) -> pd.DataFrame:
    candidate_rows: list[dict] = []
    min_distance_points = resolution_min_distance_points(common_grid)
    for label in sorted(band_frame["label"].unique()):
        subset = band_frame[band_frame["label"] == label].sort_values(
            "wavenumber_cm1"
        )
        contrast = subset["local_signal_contrast"].to_numpy(dtype=np.float64)
        noise = subset["noise_sigma_mean_independent"].to_numpy(dtype=np.float64)
        peaks, properties = find_peaks(
            contrast,
            distance=min_distance_points,
            prominence=SIGNAL_SNR_CUTOFF
            * np.maximum(noise, np.finfo(float).eps),
            wlen=SNR_PEAK_WLEN,
        )
        for property_index, peak_index in enumerate(peaks):
            row = subset.iloc[int(peak_index)]
            mean_snr = float(row["snr_contrast_mean_independent"])
            if mean_snr < SIGNAL_SNR_CUTOFF:
                continue
            candidate_rows.append(
                {
                    "label": label,
                    "candidate_wavenumber_cm1": float(row["wavenumber_cm1"]),
                    "peak_prominence_local_contrast": float(
                        properties["prominences"][property_index]
                    ),
                    "snr_contrast_local": float(row["snr_contrast_local"]),
                    "snr_contrast_mean_independent": mean_snr,
                }
            )

    if not candidate_rows:
        return pd.DataFrame(
            columns=[
                "peak_id",
                "representative_wavenumber_cm1",
                "label_count",
                "labels_present",
                "common_candidate",
                "max_prominence_local_contrast",
                "max_snr_contrast_mean_independent",
                "min_snr_contrast_mean_independent",
                "match_tolerance_cm1",
                "candidate_centers_by_label",
                "candidate_mean_snr_by_label",
            ]
        )

    candidate_rows.sort(key=lambda row: row["candidate_wavenumber_cm1"])
    clusters: list[dict] = []
    for candidate in candidate_rows:
        if not clusters or abs(
            candidate["candidate_wavenumber_cm1"]
            - clusters[-1]["anchor_wavenumber_cm1"]
        ) > INSTRUMENT_RESOLUTION_FWHM_CM1:
            clusters.append(
                {
                    "anchor_wavenumber_cm1": candidate["candidate_wavenumber_cm1"],
                    "items": [candidate],
                }
            )
        else:
            clusters[-1]["items"].append(candidate)

    rows: list[dict] = []
    for peak_index, cluster in enumerate(clusters, start=1):
        items = cluster["items"]
        best = max(
            items,
            key=lambda item: (
                item["peak_prominence_local_contrast"],
                item["snr_contrast_mean_independent"],
            ),
        )
        centers_by_label = {
            item["label"]: item["candidate_wavenumber_cm1"] for item in items
        }
        snr_by_label = {
            item["label"]: item["snr_contrast_mean_independent"] for item in items
        }
        labels_present = sorted(centers_by_label)
        rows.append(
            {
                "peak_id": f"common_peak_{peak_index:02d}",
                "representative_wavenumber_cm1": float(
                    best["candidate_wavenumber_cm1"]
                ),
                "label_count": int(len(labels_present)),
                "labels_present": "|".join(labels_present),
                "common_candidate": bool(len(labels_present) >= 2),
                "max_prominence_local_contrast": float(
                    max(item["peak_prominence_local_contrast"] for item in items)
                ),
                "max_snr_contrast_mean_independent": float(
                    max(item["snr_contrast_mean_independent"] for item in items)
                ),
                "min_snr_contrast_mean_independent": float(
                    min(item["snr_contrast_mean_independent"] for item in items)
                ),
                "match_tolerance_cm1": INSTRUMENT_RESOLUTION_FWHM_CM1,
                "candidate_centers_by_label": json.dumps(
                    centers_by_label, ensure_ascii=False, sort_keys=True
                ),
                "candidate_mean_snr_by_label": json.dumps(
                    snr_by_label, ensure_ascii=False, sort_keys=True
                ),
            }
        )
    return pd.DataFrame(rows)


def make_direct_base_models():
    def lr():
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, C=1.0, random_state=RANDOM_STATE),
        )

    def ridge():
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, C=0.1, random_state=RANDOM_STATE),
        )

    def xgb():
        return XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbosity=0,
        )

    def rf():
        return RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )

    return [
        ("lr_raw_mean", lr),
        ("xgb_raw_mean", xgb),
        ("rf_raw_mean", rf),
        ("ridge_raw_mean", ridge),
    ]


def _make_meta_model() -> LogisticRegression:
    return LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.5,
        C=1.0,
        max_iter=5000,
        random_state=RANDOM_STATE,
    )


def _select_outer_train_threshold(
    meta_features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> tuple[float, float]:
    validation_probability = np.full(len(labels), np.nan, dtype=np.float64)
    inner = GroupKFold(n_splits=THRESHOLD_SELECTION_FOLDS)
    for train_index, validation_index in inner.split(
        meta_features,
        labels,
        groups,
    ):
        model = _make_meta_model()
        model.fit(meta_features[train_index], labels[train_index])
        validation_probability[validation_index] = model.predict_proba(
            meta_features[validation_index]
        )[:, 1]
    if np.isnan(validation_probability).any():
        raise RuntimeError("Threshold selection did not cover every outer-train subject.")
    return select_optimal_threshold(labels, validation_probability)


def run_nested_direct_stacking(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_outer: int = 5,
    n_inner: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    base_specs = make_direct_base_models()
    outer = GroupKFold(n_splits=n_outer)
    oof_final = np.full(len(y), np.nan, dtype=np.float64)
    oof_prediction = np.full(len(y), -1, dtype=int)
    oof_threshold = np.full(len(y), np.nan, dtype=np.float64)
    fold_rows: list[dict] = []

    for fold, (train_index, test_index) in enumerate(
        outer.split(X, y, groups),
        start=1,
    ):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        groups_train = groups[train_index]
        inner = GroupKFold(n_splits=n_inner)
        meta_train = np.zeros((len(train_index), len(base_specs)), dtype=np.float64)

        for model_index, (_name, factory) in enumerate(base_specs):
            oof_column = np.zeros(len(train_index), dtype=np.float64)
            for inner_train, inner_test in inner.split(
                X_train, y_train, groups_train
            ):
                model = factory()
                model.fit(X_train[inner_train], y_train[inner_train])
                oof_column[inner_test] = model.predict_proba(
                    X_train[inner_test]
                )[:, 1]
            meta_train[:, model_index] = oof_column

        threshold, threshold_selection_bacc = _select_outer_train_threshold(
            meta_train,
            y_train,
            groups_train,
        )
        meta = _make_meta_model()
        meta.fit(meta_train, y_train)

        meta_test = np.zeros((len(test_index), len(base_specs)), dtype=np.float64)
        for model_index, (_name, factory) in enumerate(base_specs):
            model = factory()
            model.fit(X_train, y_train)
            meta_test[:, model_index] = model.predict_proba(X_test)[:, 1]

        probability = meta.predict_proba(meta_test)[:, 1]
        oof_final[test_index] = probability
        prediction = (probability >= threshold).astype(int)
        oof_prediction[test_index] = prediction
        oof_threshold[test_index] = threshold
        fold_rows.append(
            {
                "fold": int(fold),
                "n_train_subjects": int(len(train_index)),
                "n_test_subjects": int(len(test_index)),
                "test_positive_subjects": int(y_test.sum()),
                "test_negative_subjects": int((1 - y_test).sum()),
                "auc": float(roc_auc_score(y_test, probability)),
                "balanced_accuracy": float(
                    balanced_accuracy_score(y_test, prediction)
                ),
                "threshold": float(threshold),
                "threshold_selection_balanced_accuracy": float(
                    threshold_selection_bacc
                ),
            }
        )
        print(
            f"  [outer {fold}] AUC={fold_rows[-1]['auc']:.4f} "
            f"balanced_acc={fold_rows[-1]['balanced_accuracy']:.4f}"
        )

    if np.isnan(oof_final).any():
        raise RuntimeError("Nested CV did not produce a prediction for every subject.")
    if np.any(oof_prediction < 0):
        raise RuntimeError("Nested CV did not produce a thresholded prediction for every subject.")
    if np.isnan(oof_threshold).any():
        raise RuntimeError("Nested CV did not record a threshold for every subject.")
    return oof_final, oof_prediction, oof_threshold, pd.DataFrame(fold_rows)


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def plot_roc(y: np.ndarray, probability: np.ndarray, path: Path) -> None:
    fpr, tpr, _ = roc_curve(y, probability)
    auc = roc_auc_score(y, probability)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(fpr, tpr, color="#7b2cbf", lw=2, label=f"OOF ROC (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#777777", lw=1)
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False positive rate", ylabel="True positive rate")
    ax.set_title("Direct QC-passed mean spectrum model")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(
    y: np.ndarray,
    prediction: np.ndarray,
    threshold_label: str,
    path: Path,
) -> None:
    matrix = confusion_matrix(y, prediction, labels=[0, 1])
    row_totals = matrix.sum(axis=1, keepdims=True)
    row_percentages = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_totals != 0,
    )
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    image = ax.imshow(matrix, cmap="Purples")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    for row in range(2):
        for column in range(2):
            ax.text(
                column,
                row,
                f"{int(matrix[row, column])}\n"
                f"({row_percentages[row, column]:.1%})",
                ha="center",
                va="center",
            )
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["negative", "positive"],
        yticklabels=["negative", "positive"],
        xlabel="Predicted label",
        ylabel="True label",
        title=f"OOF confusion matrix (count; row %) ({threshold_label})",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_spectra_and_noise(
    common_grid: np.ndarray,
    means: np.ndarray,
    labels: list[str],
    noise_vectors: np.ndarray,
    path: Path,
) -> None:
    labels_array = np.asarray(labels)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for label, color in [("control", "#457b9d"), ("prostate disease control", "#f4a261"), ("prostate", "#9d0208")]:
        mask = labels_array == label
        if not np.any(mask):
            continue
        median = np.median(means[mask], axis=0)
        lower = np.percentile(means[mask], 25, axis=0)
        upper = np.percentile(means[mask], 75, axis=0)
        axes[0].plot(common_grid, median, lw=1.5, color=color, label=label)
        axes[0].fill_between(common_grid, lower, upper, color=color, alpha=0.15)
    axes[0].set_ylabel("Relative intensity")
    axes[0].set_title("QC-passed mean representative spectra (median ± IQR)")
    axes[0].legend(frameon=False)

    noise_median = np.median(noise_vectors, axis=0)
    noise_lower = np.percentile(noise_vectors, 25, axis=0)
    noise_upper = np.percentile(noise_vectors, 75, axis=0)
    axes[1].plot(common_grid, noise_median, color="#6a4c93", lw=1.5)
    axes[1].fill_between(common_grid, noise_lower, noise_upper, color="#6a4c93", alpha=0.18)
    axes[1].set_xlabel("Wavenumber (cm$^{-1}$)")
    axes[1].set_ylabel("Repeat SD")
    axes[1].set_title("Residual-derived repeat-measurement noise (median ± IQR)")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_signal_noise_shading(
    common_grid: np.ndarray,
    means: np.ndarray,
    labels: list[str],
    noise_vectors: np.ndarray,
    qc_rows: list[dict],
    reference_peaks_cm1: tuple[float, ...],
    path: Path,
) -> None:
    labels_array = np.asarray(labels)
    colors = {
        "control": "#457b9d",
        "prostate disease control": "#f4a261",
        "prostate": "#9d0208",
    }
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for axis_index, (axis, averaging) in enumerate(
        [(axes[0], 1.0), (axes[1], "independent")]
    ):
        for label, color in colors.items():
            mask = labels_array == label
            if not np.any(mask):
                continue
            signal = np.median(means[mask], axis=0)
            noise = np.median(noise_vectors[mask], axis=0)
            if averaging == "independent":
                median_qc_count = float(
                    np.median(
                        [
                            qc_rows[index]["repeats_passed"]
                            for index in np.flatnonzero(mask)
                        ]
                    )
                )
                noise = noise / np.sqrt(median_qc_count)
            axis.plot(common_grid, signal, color=color, lw=1.4, label=label)
            axis.fill_between(
                common_grid,
                signal - noise,
                signal + noise,
                color=color,
                alpha=0.16,
            )
        for reference_peak in reference_peaks_cm1:
            axis.axvline(
                reference_peak,
                color="#555555",
                linestyle="--",
                linewidth=0.6,
                alpha=0.7,
            )
        axis.set_ylabel("API intensity")
        axis.legend(frameon=False, loc="upper right")
    axes[0].set_title("Representative spectrum ± repeat SD (original intensity units)")
    axes[1].set_title("Representative spectrum ± repeat SD / √N_QC (independent-repeat illustration)")
    axes[1].set_xlabel("Wavenumber (cm$^{-1}$)")
    fig.suptitle("Signal and noise shading; no signal scaling applied", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def shade_regions(axis, common_grid: np.ndarray, mask: np.ndarray, color: str, alpha: float) -> None:
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
    for start, end in zip(starts, ends):
        axis.axvspan(
            common_grid[start],
            common_grid[end],
            color=color,
            alpha=alpha,
            linewidth=0,
        )


def resolution_aware_peak_interval(
    common_grid: np.ndarray,
    band_subset: pd.DataFrame,
    center: float,
) -> tuple[float, float] | None:
    ordered = band_subset.sort_values("wavenumber_cm1")
    contrast = ordered["local_signal_contrast"].to_numpy(dtype=np.float64)
    noise = ordered["noise_sigma_mean_independent"].to_numpy(dtype=np.float64)
    wavenumbers = ordered["wavenumber_cm1"].to_numpy(dtype=np.float64)
    peaks, _ = find_peaks(
        contrast,
        distance=resolution_min_distance_points(common_grid),
        prominence=SIGNAL_SNR_CUTOFF * np.maximum(noise, np.finfo(float).eps),
        wlen=SNR_PEAK_WLEN,
    )
    if len(peaks) == 0:
        return None
    peak_index = int(min(peaks, key=lambda index: abs(wavenumbers[index] - center)))
    if abs(wavenumbers[peak_index] - center) > INSTRUMENT_RESOLUTION_FWHM_CM1:
        return None
    widths = peak_widths(contrast, [peak_index], rel_height=0.5)
    index_grid = np.arange(len(common_grid), dtype=np.float64)
    left = float(np.interp(widths[2][0], index_grid, common_grid))
    right = float(np.interp(widths[3][0], index_grid, common_grid))
    return left, right


def plot_signal_noise_band_validation(
    common_grid: np.ndarray,
    means: np.ndarray,
    labels: list[str],
    noise_vectors: np.ndarray,
    qc_rows: list[dict],
    band_frame: pd.DataFrame,
    peak_registry: pd.DataFrame,
    shift_values: np.ndarray,
    reference_peaks_cm1: tuple[float, ...],
    path: Path,
) -> None:
    labels_array = np.asarray(labels)
    colors = {
        "control": "#457b9d",
        "prostate disease control": "#f4a261",
        "prostate": "#9d0208",
    }
    selected_label = "prostate" if "prostate" in set(labels) else sorted(set(labels))[0]
    selected_mask = labels_array == selected_label
    selected_band = band_frame[band_frame["label"] == selected_label].sort_values(
        "wavenumber_cm1"
    )
    signal_mask = np.zeros(len(common_grid), dtype=bool)
    repeat_signal_mask = np.zeros(len(common_grid), dtype=bool)
    common_peaks = peak_registry[peak_registry["common_candidate"]]
    for _, peak_row in common_peaks.iterrows():
        labels_present = set(str(peak_row["labels_present"]).split("|"))
        if selected_label not in labels_present:
            continue
        interval = resolution_aware_peak_interval(
            common_grid,
            selected_band,
            float(peak_row["representative_wavenumber_cm1"]),
        )
        if interval is None:
            continue
        left, right = interval
        interval_mask = (common_grid >= left) & (common_grid <= right)
        signal_mask |= interval_mask
        repeat_signal_mask |= interval_mask & selected_band[
            "signal_band_snr3"
        ].to_numpy(dtype=bool)
    signal = np.median(means[selected_mask], axis=0)
    noise = np.median(noise_vectors[selected_mask], axis=0)
    fig, axes = plt.subplots(3, 1, figsize=(12, 11))

    shade_regions(axes[0], common_grid, signal_mask, "#2a9d8f", 0.12)
    shade_regions(axes[0], common_grid, ~signal_mask, "#adb5bd", 0.07)
    shade_regions(axes[0], common_grid, repeat_signal_mask, "#e76f51", 0.18)
    axes[0].fill_between(
        common_grid,
        signal - noise,
        signal + noise,
        color="#6c757d",
        alpha=0.25,
        label="± repeat SD",
    )
    axes[0].plot(
        common_grid,
        signal,
        color=colors[selected_label],
        linewidth=1.5,
        label=f"{selected_label} mean spectrum",
    )
    for reference_peak in reference_peaks_cm1:
        axes[0].axvline(
            reference_peak,
            color="#555555",
            linestyle="--",
            linewidth=0.6,
            alpha=0.7,
        )
    axes[0].set_title(
        f"{selected_label}: resolution-aware mean-SNR ≥ {SIGNAL_SNR_CUTOFF:.0f} peak bands vs noise"
    )
    axes[0].set_ylabel("API intensity")
    axes[0].legend(
        handles=[
            Patch(
                facecolor="#2a9d8f",
                alpha=0.25,
                label="mean-spectrum signal peak band",
            ),
            Patch(facecolor="#e76f51", alpha=0.25, label="repeat signal band"),
            Patch(facecolor="#adb5bd", alpha=0.25, label="noise-dominated band"),
            Patch(facecolor="#6c757d", alpha=0.25, label="± repeat SD"),
            Line2D(
                [0],
                [0],
                color="#555555",
                linestyle="--",
                linewidth=0.8,
                label=f"{CALIBRATION_STANDARD_MATERIAL} reference peaks",
            ),
        ],
        frameon=False,
        loc="upper right",
    )

    for label, color in colors.items():
        subset = band_frame[band_frame["label"] == label]
        if subset.empty:
            continue
        axes[1].plot(
            subset["wavenumber_cm1"],
            subset["snr_contrast_local"],
            color=color,
            linewidth=1.1,
            alpha=0.45,
            label=f"{label} repeat SNR",
        )
        axes[1].plot(
            subset["wavenumber_cm1"],
            subset["snr_contrast_mean_independent"],
            color=color,
            linewidth=1.2,
            linestyle="--",
            label=f"{label} mean SNR",
        )
    axes[1].axhline(SIGNAL_SNR_CUTOFF, color="#2a9d8f", linestyle="--", linewidth=1)
    axes[1].axhline(
        SIGNAL_SNR_STRONG_CUTOFF,
        color="#e76f51",
        linestyle=":",
        linewidth=1,
    )
    axes[1].set_title("Repeat residual SNR (solid) and independent-mean SNR (dashed)")
    axes[1].set_ylabel("SNR")
    axes[1].set_ylim(bottom=0)
    axes[1].legend(frameon=False, loc="upper right")

    axes[2].hist(shift_values, bins=41, color="#6a4c93", alpha=0.8)
    axes[2].axvline(0.0, color="#555555", linestyle="--", linewidth=0.8)
    axes[2].axvline(
        np.median(shift_values),
        color="#e76f51",
        linestyle=":",
        linewidth=1.2,
        label=f"median={np.median(shift_values):.2f} cm$^{{-1}}$",
    )
    axes[2].set_title(
        f"Applied {CALIBRATION_STANDARD_MATERIAL} axis correction distribution"
    )
    axes[2].set_xlabel("Applied correction (cm$^{-1}$)")
    axes[2].set_ylabel("Repeat spectra")
    axes[2].legend(frameon=False)
    fig.suptitle(
        "Signal/noise band validation after standard-material calibration; SNR cutoff is illustrative",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_resolution_aware_peak_zoom(
    common_grid: np.ndarray,
    means: np.ndarray,
    labels: list[str],
    noise_vectors: np.ndarray,
    band_frame: pd.DataFrame,
    peak_registry: pd.DataFrame,
    path: Path,
) -> None:
    colors = {
        "control": "#457b9d",
        "prostate disease control": "#f4a261",
        "prostate": "#9d0208",
    }
    common_peaks = peak_registry[peak_registry["common_candidate"]].sort_values(
        "representative_wavenumber_cm1"
    )
    if common_peaks.empty:
        raise RuntimeError("No resolution-aware common peak candidates were found.")
    common_peaks = common_peaks.head(9)
    ncols = 3
    nrows = int(np.ceil(len(common_peaks) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(15, 3.7 * nrows),
        squeeze=False,
    )
    labels_array = np.asarray(labels)
    plot_axes = axes.flat
    for axis, (_, peak_row) in zip(plot_axes, common_peaks.iterrows()):
        center = float(peak_row["representative_wavenumber_cm1"])
        axis.axvspan(
            center - 18.0,
            center + 18.0,
            color="#eef0f2",
            alpha=0.9,
            linewidth=0,
        )
        labels_present = set(str(peak_row["labels_present"]).split("|"))
        for label in colors:
            subset_mask = labels_array == label
            if not np.any(subset_mask):
                continue
            signal = np.median(means[subset_mask], axis=0)
            noise = np.median(noise_vectors[subset_mask], axis=0)
            axis.fill_between(
                common_grid,
                signal - noise,
                signal + noise,
                color=colors[label],
                alpha=0.10,
            )
            axis.plot(
                common_grid,
                signal,
                color=colors[label],
                linewidth=1.4,
                label=label,
            )
            if label not in labels_present:
                continue
            band_subset = band_frame[band_frame["label"] == label].sort_values(
                "wavenumber_cm1"
            )
            interval = resolution_aware_peak_interval(
                common_grid,
                band_subset,
                center,
            )
            if interval is None:
                continue
            left, right = interval
            axis.axvspan(
                left,
                right,
                color="#2a9d8f",
                alpha=0.28,
                linewidth=0,
                zorder=0.5,
            )
            interval_mask = (common_grid >= left) & (common_grid <= right)
            repeat_signal = band_subset["signal_band_snr3"].to_numpy(dtype=bool)
            if np.any(repeat_signal & interval_mask):
                axis.axvspan(
                    left,
                    right,
                    color="#e76f51",
                    alpha=0.30,
                    linewidth=0,
                    zorder=0.6,
                )
        axis.axvspan(
            center - INSTRUMENT_RESOLUTION_FWHM_CM1 / 2.0,
            center + INSTRUMENT_RESOLUTION_FWHM_CM1 / 2.0,
            color="#f4a261",
            alpha=0.20,
            linewidth=0,
        )
        axis.axvline(center, color="#333333", linestyle="--", linewidth=0.9)
        axis.set_xlim(center - 18.0, center + 18.0)
        axis.set_title(
            f"{peak_row['peak_id']}  {center:.1f} cm$^{{-1}}$\n"
            f"common in {int(peak_row['label_count'])} labels; "
            f"mean-SNR max={float(peak_row['max_snr_contrast_mean_independent']):.1f}"
        )
        axis.set_xlabel("Wavenumber (cm$^{-1}$)")
        axis.set_ylabel("API intensity")
        axis.grid(axis="y", alpha=0.18)

    for axis in plot_axes[len(common_peaks) :]:
        axis.set_visible(False)
    legend_handles = [
        Line2D([0], [0], color=colors["control"], lw=1.5, label="control"),
        Line2D(
            [0],
            [0],
            color=colors["prostate disease control"],
            lw=1.5,
            label="prostate disease control",
        ),
        Line2D([0], [0], color=colors["prostate"], lw=1.5, label="prostate"),
        Patch(
            facecolor="#2a9d8f",
            alpha=0.30,
            label="mean-spectrum SNR ≥ 3",
        ),
        Patch(facecolor="#e76f51", alpha=0.30, label="repeat SNR ≥ 3"),
        Patch(
            facecolor="#eef0f2",
            alpha=0.9,
            label="noise-dominated background",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.suptitle(
        "Resolution-aware common peak zoom after standard-material alignment\n"
        f"same peak tolerance = {INSTRUMENT_RESOLUTION_FWHM_CM1:.1f} cm$^{{-1}}$; "
        "green = candidate peak half-prominence width; shaded envelopes = repeat residual SD",
        y=1.08,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_resolution_aware_snr_zoom(
    common_grid: np.ndarray,
    band_frame: pd.DataFrame,
    peak_registry: pd.DataFrame,
    path: Path,
) -> None:
    colors = {
        "control": "#457b9d",
        "prostate disease control": "#f4a261",
        "prostate": "#9d0208",
    }
    common_peaks = peak_registry[peak_registry["common_candidate"]].sort_values(
        "representative_wavenumber_cm1"
    )
    if common_peaks.empty:
        raise RuntimeError("No resolution-aware common peak candidates were found.")
    common_peaks = common_peaks.head(9)
    ncols = 3
    nrows = int(np.ceil(len(common_peaks) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(15, 3.7 * nrows),
        squeeze=False,
    )
    plot_axes = axes.flat
    for axis, (_, peak_row) in zip(plot_axes, common_peaks.iterrows()):
        center = float(peak_row["representative_wavenumber_cm1"])
        labels_present = set(str(peak_row["labels_present"]).split("|"))
        axis.axvspan(
            center - 18.0,
            center + 18.0,
            color="#eef0f2",
            alpha=0.9,
            linewidth=0,
        )
        maximum = 0.0
        for label, color in colors.items():
            subset = band_frame[band_frame["label"] == label].sort_values(
                "wavenumber_cm1"
            )
            if subset.empty:
                continue
            window = subset[
                subset["wavenumber_cm1"].between(center - 18.0, center + 18.0)
            ]
            axis.plot(
                window["wavenumber_cm1"],
                window["snr_contrast_local"],
                color=color,
                linewidth=1.3,
            )
            axis.plot(
                window["wavenumber_cm1"],
                window["snr_contrast_mean_independent"],
                color=color,
                linewidth=1.3,
                linestyle="--",
            )
            maximum = max(
                maximum,
                float(window["snr_contrast_mean_independent"].max()),
            )
            if label not in labels_present:
                continue
            interval = resolution_aware_peak_interval(
                common_grid,
                subset,
                center,
            )
            if interval is None:
                continue
            left, right = interval
            axis.axvspan(
                left,
                right,
                color="#2a9d8f",
                alpha=0.28,
                linewidth=0,
                zorder=0.5,
            )
            interval_mask = subset["wavenumber_cm1"].between(left, right)
            if subset.loc[interval_mask, "signal_band_snr3"].any():
                axis.axvspan(
                    left,
                    right,
                    color="#e76f51",
                    alpha=0.30,
                    linewidth=0,
                    zorder=0.6,
                )
        axis.axhline(SIGNAL_SNR_CUTOFF, color="#2a9d8f", linestyle="--", linewidth=1)
        axis.axhline(
            SIGNAL_SNR_STRONG_CUTOFF,
            color="#e76f51",
            linestyle=":",
            linewidth=1,
        )
        axis.axvspan(
            center - INSTRUMENT_RESOLUTION_FWHM_CM1 / 2.0,
            center + INSTRUMENT_RESOLUTION_FWHM_CM1 / 2.0,
            color="#f4a261",
            alpha=0.20,
            linewidth=0,
        )
        axis.axvline(center, color="#333333", linestyle="--", linewidth=0.9)
        axis.set_xlim(center - 18.0, center + 18.0)
        axis.set_ylim(0, max(6.0, maximum * 1.10))
        axis.set_title(
            f"{peak_row['peak_id']}  {center:.1f} cm$^{{-1}}$\n"
            f"common in {int(peak_row['label_count'])} labels"
        )
        axis.set_xlabel("Wavenumber (cm$^{-1}$)")
        axis.set_ylabel("Peak-contrast SNR")
        axis.grid(axis="y", alpha=0.18)

    for axis in plot_axes[len(common_peaks) :]:
        axis.set_visible(False)
    legend_handles = [
        Line2D([0], [0], color="#333333", lw=1.5, label="repeat SNR, solid"),
        Line2D(
            [0],
            [0],
            color="#333333",
            lw=1.5,
            linestyle="--",
            label="mean-spectrum SNR, dashed",
        ),
        Patch(facecolor="#2a9d8f", alpha=0.30, label="mean-spectrum SNR ≥ 3"),
        Patch(facecolor="#e76f51", alpha=0.30, label="repeat SNR ≥ 3"),
        Patch(facecolor="#eef0f2", alpha=0.9, label="noise-dominated background"),
    ]
    fig.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.suptitle(
        "Resolution-aware signal/noise SNR zoom after standard-material alignment\n"
        "solid = repeat-level detectability; dashed = mean-spectrum precision",
        y=1.08,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_qc(qc_rows: list[dict], path: Path) -> None:
    total = sum(row["repeats_total"] for row in qc_rows)
    passed = sum(row["repeats_passed"] for row in qc_rows)
    excluded = total - passed
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    bars = ax.bar(["passed", "excluded"], [passed, excluded], color=["#2a9d8f", "#e76f51"])
    ax.bar_label(bars, fmt="%d")
    ax.set_ylabel("Number of repeat spectra")
    ax.set_title(f"Repeat QC: {passed / total:.1%} passed")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_snr_peaks(snr_frame: pd.DataFrame, path: Path) -> None:
    colors = {
        "control": "#457b9d",
        "prostate disease control": "#f4a261",
        "prostate": "#9d0208",
    }
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for label, color in colors.items():
        subset = snr_frame[snr_frame["label"] == label]
        if subset.empty:
            continue
        ax.scatter(
            subset["peak_wavenumber_cm1"],
            subset["snr_contrast_over_local_noise"],
            color=color,
            label=label,
            s=42,
        )
    ax.axhline(3.0, color="#777777", linestyle="--", linewidth=1)
    ax.set(
        xlabel="Candidate peak wavenumber (cm$^{-1}$)",
        ylabel="Peak-contrast SNR",
        title="Peak contrast relative to repeat-measurement noise",
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def copy_publication_figures() -> dict[str, str]:
    figure_names = {
        "fig01_mean_spectrum_clinical_roc.png": FIGURE_DIR / "clinical_roc.png",
        "fig02_mean_spectrum_confusion_matrix.png": FIGURE_DIR / "confusion_matrix.png",
        "fig03_mean_spectrum_noise.png": FIGURE_DIR / "representative_spectrum_and_noise.png",
        "fig04_mean_spectrum_qc.png": FIGURE_DIR / "repeat_qc_pass_rate.png",
        "fig05_mean_spectrum_snr.png": FIGURE_DIR / "snr_peak_summary.png",
        "fig06_mean_spectrum_signal_noise_shading.png": FIGURE_DIR / "spectrum_noise_shading.png",
        "fig07_signal_noise_band_validation.png": FIGURE_DIR / "signal_noise_band_validation.png",
        "fig08_resolution_aware_peak_zoom.png": FIGURE_DIR / "resolution_aware_peak_zoom.png",
        "fig09_resolution_aware_snr_zoom.png": FIGURE_DIR / "resolution_aware_snr_zoom.png",
    }
    copied: dict[str, str] = {}
    for destination_name, source in figure_names.items():
        destination = PUBLICATION_FIGURE_DIR / destination_name
        shutil.copy2(source, destination)
        copied[destination_name] = str(destination)
    return copied


def write_readme(
    metrics: dict,
    qc_summary: dict,
    metadata: dict,
) -> None:
    text = f"""# Average representative spectrum clinical evaluation

This folder is the corrected, separate result for:
`{PATENT_SOURCE.name}`

## Method order

1. Load passing Polystyrene (PS) Raman-shift calibration records from the read-only AECD database, matched by clinical measurement date and instrument.
2. Apply `x_corrected = x_observed - global_shift_cm1`, where the stored global shift is observed standard-material peak minus reference peak.
3. Interpolation onto the common {SPECTRAL_RANGE_CM1[0]:.0f}-{SPECTRAL_RANGE_CM1[1]:.0f} cm^-1 grid.
4. Within-subject repeat QC using robust spectral-shape distance (median/MAD cutoff {QC_MAD_THRESHOLD}).
5. Mean of QC-passed aligned repeats: the representative spectrum.
6. Residual noise: per-wavenumber sample SD around that representative spectrum.
7. Direct raw-spectrum stacking model on the representative spectrum.

Clinical urine spectra are not searched for a calibration anchor peak. The
calibration anchor is the PS standard-material record stored in the database.

After the mean spectrum was created, this run applied **no** DWT, baseline
correction, absolute intensity scaling, Savitzky-Golay smoothing, SNV,
derivatives, or peak-feature extraction.

## Clinical model

The direct model is a leakage-free subject-level nested GroupKFold stacking
model with four raw-mean-spectrum base estimators: scaled logistic regression,
XGBoost, random forest, and a stronger-L2 logistic regression. The meta learner
is elastic-net logistic regression. Model-internal scaling for the logistic
base estimators is not a spectrum-level preprocessing step.

## Results

- Subjects: {metadata["cohort"]["subjects_total"]} ({metadata["cohort"]["positive_subjects"]} positive, {metadata["cohort"]["negative_subjects"]} negative)
- Total repeats: {qc_summary["total_repeats"]}
- QC-passed repeats: {qc_summary["passed_repeats"]} ({qc_summary["repeat_pass_rate"]:.2%})
- QC-excluded repeats: {qc_summary["excluded_repeats"]}
- Overall OOF AUC: {metrics["overall_oof_auc"]:.4f}
- Overall OOF balanced accuracy: {metrics["overall_balanced_accuracy"]:.4f}
- Threshold policy: {metrics["threshold_policy"]}
- Outer-train threshold median: {metrics["threshold_median"]:.4f}

SNR summary

The SNR table uses the patent-compatible definitions `I_peak / sigma_rep` and
`I_peak / noise_floor`. It also reports local peak contrast instead of raw
intensity, because the spectra were intentionally not baseline-corrected.
The `sigma_rep / sqrt(N_QC)` version is reported separately as the independent
mean-spectrum precision estimate and is used for the resolution-aware candidate
rule `prominence >= 3 x sigma_rep / sqrt(N_QC)`.

The signal/noise shading example is saved as `spectrum_noise_shading.png`.

Signal/noise band validation uses local contrast divided by repeat residual
noise. With the illustrative SNR cutoff of {SIGNAL_SNR_CUTOFF:.0f}, the band
table and shift distribution are saved as `signal_noise_band_summary.csv` and
`calibration_shift_values.csv`. The figure shows both the conservative repeat
SNR and the independent-mean SNR assumption.

The resolution-aware common peak registry and raw/SNR zoom figures are saved
as `resolution_aware_peak_registry.csv`, `resolution_aware_peak_zoom.png`, and
`resolution_aware_snr_zoom.png`. These are descriptive validation outputs, not
model features.

- Highest candidate peak-contrast SNR over local repeat noise: {metadata["snr"]["headline"]["peak_wavenumber_cm1"]:.2f} cm^-1, {metadata["snr"]["headline"]["snr_contrast_over_local_noise"]:.2f}
- Same peak using global noise floor: {metadata["snr"]["headline"]["snr_contrast_over_noise_floor"]:.2f}

## Important interpretation limit

This is an internal subject-level evaluation of the available AECD cohort.
Because the positive and negative groups are from the same hospital/cohort
structure, the Cancer Screening AUC can be affected by hospital, collection,
or batch confounding. It must not be interpreted as cross-hospital
generalization without an independent external validation cohort.

See `run_metadata.json`, `qc_summary.csv`, `subject_qc_noise_summary.csv`,
`noise_summary.csv`, `snr_peak_summary.csv`, and
`resolution_aware_peak_registry.csv` for the complete processing history and
model input. The peak registry is descriptive only and is not used as a model
feature.
"""
    (OUTPUT_DIR / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    items, cohorts, api_metadata = load_api_spectra()
    standard_calibrations, standard_calibration_metadata = (
        load_standard_material_calibrations()
    )
    common_grid, observed_step, common_points = construct_common_grid(items)
    subject_keys, labels, replicate_matrices, alignment_metadata = (
        group_and_align_subjects(items, common_grid, standard_calibrations)
    )

    means, qc_rows, noise_vectors = build_mean_spectra(
        replicate_matrices,
        common_grid,
    )
    labels_array = np.asarray(labels)
    unknown_labels = sorted(set(labels_array) - {POSITIVE_LABEL} - NEGATIVE_LABELS)
    if unknown_labels:
        raise ValueError(f"Unexpected cohort labels: {unknown_labels}")
    y = (labels_array == POSITIVE_LABEL).astype(int)
    groups = np.arange(len(y), dtype=int)
    shift_values = np.asarray(
        alignment_metadata.pop("calibration_shift_values_cm1"),
        dtype=np.float64,
    )
    calibration_rows = alignment_metadata.pop(
        "standard_material_calibration_rows"
    )
    reference_peaks_cm1 = tuple(
        float(value) for value in standard_calibration_metadata["reference_peaks_cm1"]
    )
    band_frame = build_band_snr_summary(
        common_grid,
        means,
        labels,
        noise_vectors,
        qc_rows,
    )
    peak_registry = build_resolution_aware_peak_registry(common_grid, band_frame)
    if peak_registry.empty:
        raise RuntimeError("No resolution-aware peak candidates were available.")
    if len(shift_values) != len(items):
        raise RuntimeError("Calibration shift count does not match input spectra.")

    snr_frame = build_snr_summary(
        common_grid,
        means,
        labels,
        noise_vectors,
        qc_rows,
        reference_peaks_cm1,
    )
    if snr_frame.empty:
        raise RuntimeError("No candidate peaks were available for SNR calculation.")
    headline_row = snr_frame.loc[
        snr_frame["snr_contrast_over_local_noise"].idxmax()
    ]
    snr_metadata = {
        "primary_definitions": {
            "raw_local": "peak_intensity_raw / noise_sigma_at_peak",
            "raw_global": "peak_intensity_raw / noise_floor_global_median",
            "contrast_local": "peak_prominence_local_contrast / noise_sigma_at_peak",
            "contrast_global": "peak_prominence_local_contrast / noise_floor_global_median",
        },
        "independent_mean_assumption": "peak_prominence_local_contrast / (noise_sigma_at_peak / sqrt(N_QC))",
        "candidate_detection": {
            "prominence_rule": "peak prominence >= 3 x group-median repeat residual SD / sqrt(median QC-passed repeats) at each wavenumber",
            "minimum_distance_points": resolution_min_distance_points(common_grid),
            "effective_resolution_fwhm_cm1": INSTRUMENT_RESOLUTION_FWHM_CM1,
            "prominence_window_points": SNR_PEAK_WLEN,
            "purpose": "descriptive SNR assessment only; candidate peaks are not model features",
        },
        "headline": {
            "label": str(headline_row["label"]),
            "peak_wavenumber_cm1": float(headline_row["peak_wavenumber_cm1"]),
            "snr_raw_over_local_noise": float(headline_row["snr_raw_over_local_noise"]),
            "snr_raw_over_noise_floor": float(headline_row["snr_raw_over_noise_floor"]),
            "snr_contrast_over_local_noise": float(
                headline_row["snr_contrast_over_local_noise"]
            ),
            "snr_contrast_over_noise_floor": float(
                headline_row["snr_contrast_over_noise_floor"]
            ),
            "snr_contrast_over_mean_noise_independent": float(
                headline_row["snr_contrast_over_mean_noise_independent"]
            ),
        },
        "peak_rows": snr_frame.to_dict(orient="records"),
    }
    band_metadata = {
        "local_contrast_definition": "max(group-median spectrum - centered rolling median baseline, 0)",
        "noise_definition": "median subject-level repeat residual SD at each wavenumber",
        "illustrative_snr_cutoff": SIGNAL_SNR_CUTOFF,
        "strong_snr_cutoff": SIGNAL_SNR_STRONG_CUTOFF,
        "baseline_window_points": SNR_PEAK_WLEN,
        "signal_band_fraction_snr3": {
            str(label): float(frame["signal_band_snr3"].mean())
            for label, frame in band_frame.groupby("label")
        },
        "strong_signal_band_fraction_snr5": {
            str(label): float(frame["strong_signal_band_snr5"].mean())
            for label, frame in band_frame.groupby("label")
        },
        "signal_band_fraction_mean_independent_snr3": {
            str(label): float(frame["signal_band_mean_independent_snr3"].mean())
            for label, frame in band_frame.groupby("label")
        },
        "strong_signal_band_fraction_mean_independent_snr5": {
            str(label): float(
                frame["strong_signal_band_mean_independent_snr5"].mean()
            )
            for label, frame in band_frame.groupby("label")
        },
        "resolution_aware_peak_selection": {
            "effective_resolution_fwhm_cm1": INSTRUMENT_RESOLUTION_FWHM_CM1,
            "same_peak_match_tolerance_cm1": INSTRUMENT_RESOLUTION_FWHM_CM1,
            "minimum_separation_points": resolution_min_distance_points(common_grid),
            "common_candidate_count": int(
                peak_registry["common_candidate"].sum()
            ),
            "candidate_rule": "group-median local contrast peak with prominence >= 3 x repeat residual SD and mean-spectrum SNR >= 3; candidates within the effective resolution are grouped as one peak",
        },
    }

    probability, prediction, threshold_used, fold_metrics = run_nested_direct_stacking(
        means,
        y,
        groups,
    )
    threshold_values = fold_metrics["threshold"].to_numpy(dtype=np.float64)
    threshold_median = float(np.median(threshold_values))
    fixed_prediction = (probability >= FIXED_COMPARISON_THRESHOLD).astype(int)
    confusion = confusion_matrix(y, prediction, labels=[0, 1])
    tn, fp, fn, tp = confusion.ravel()
    fixed_confusion = confusion_matrix(y, fixed_prediction, labels=[0, 1])
    fixed_tn, fixed_fp, fixed_fn, fixed_tp = fixed_confusion.ravel()
    metrics = {
        "method": "direct_qc_passed_mean_spectrum_stacking",
        "threshold": threshold_median,
        "threshold_policy": "outer-train inner-meta OOF balanced-accuracy optimum",
        "threshold_mean": float(np.mean(threshold_values)),
        "threshold_median": threshold_median,
        "threshold_min": float(np.min(threshold_values)),
        "threshold_max": float(np.max(threshold_values)),
        "threshold_std": float(np.std(threshold_values)),
        "fixed_comparison_threshold": FIXED_COMPARISON_THRESHOLD,
        "fixed_comparison_balanced_accuracy": float(
            balanced_accuracy_score(y, fixed_prediction)
        ),
        "overall_oof_auc": float(roc_auc_score(y, probability)),
        "overall_balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "overall_f1": float(f1_score(y, prediction, zero_division=0)),
        "overall_precision": float(precision_score(y, prediction, zero_division=0)),
        "overall_recall": float(recall_score(y, prediction, zero_division=0)),
        "fold_auc_mean": float(fold_metrics["auc"].mean()),
        "fold_auc_std": float(fold_metrics["auc"].std()),
        "fold_balanced_accuracy_mean": float(
            fold_metrics["balanced_accuracy"].mean()
        ),
        "fold_balanced_accuracy_std": float(
            fold_metrics["balanced_accuracy"].std()
        ),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "fixed_comparison_true_negative": int(fixed_tn),
        "fixed_comparison_false_positive": int(fixed_fp),
        "fixed_comparison_false_negative": int(fixed_fn),
        "fixed_comparison_true_positive": int(fixed_tp),
    }
    total_repeats = int(sum(len(matrix) for matrix in replicate_matrices))
    passed_repeats = int(sum(row["repeats_passed"] for row in qc_rows))
    qc_summary = {
        "subjects_total": int(len(subject_keys)),
        "total_repeats": total_repeats,
        "passed_repeats": passed_repeats,
        "excluded_repeats": int(total_repeats - passed_repeats),
        "repeat_pass_rate": float(passed_repeats / total_repeats),
        "subjects_with_exclusion": int(
            sum(row["repeats_excluded"] > 0 for row in qc_rows)
        ),
        "qc_rule": "robust standardized RMS spectral distance from within-subject median; median/MAD cutoff",
        "qc_mad_threshold": QC_MAD_THRESHOLD,
        "minimum_passed_repeats": 2,
    }

    noise_summary = pd.DataFrame(
        {
            "wavenumber_cm1": common_grid,
            "noise_sigma_median": np.median(noise_vectors, axis=0),
            "noise_sigma_p25": np.percentile(noise_vectors, 25, axis=0),
            "noise_sigma_p75": np.percentile(noise_vectors, 75, axis=0),
            "subject_count": len(noise_vectors),
        }
    )
    mean_frame = pd.DataFrame(means, columns=[f"wn_{value:.4f}" for value in common_grid])
    mean_frame.insert(0, "label", labels)
    mean_frame.insert(0, "subject_index_internal", np.arange(len(means)))
    oof_frame = pd.DataFrame(
        {
            "subject_index_internal": np.arange(len(y)),
            "label": labels,
            "y_true": y,
            "oof_probability": probability,
            "prediction": prediction,
            "threshold_used": threshold_used,
            "prediction_fixed_0_4": fixed_prediction,
        }
    )
    qc_frame = pd.DataFrame(qc_rows)
    qc_frame.insert(1, "label", labels)

    save_csv(pd.DataFrame([metrics]), OUTPUT_DIR / "clinical_performance_summary.csv")
    save_csv(fold_metrics, OUTPUT_DIR / "clinical_fold_metrics.csv")
    save_csv(pd.DataFrame([qc_summary]), OUTPUT_DIR / "qc_summary.csv")
    save_csv(qc_frame, OUTPUT_DIR / "subject_qc_noise_summary.csv")
    save_csv(noise_summary, OUTPUT_DIR / "noise_summary.csv")
    save_csv(snr_frame, OUTPUT_DIR / "snr_peak_summary.csv")
    save_csv(band_frame, OUTPUT_DIR / "signal_noise_band_summary.csv")
    save_csv(peak_registry, OUTPUT_DIR / "resolution_aware_peak_registry.csv")
    save_csv(
        pd.DataFrame(calibration_rows),
        OUTPUT_DIR / "calibration_shift_values.csv",
    )
    save_csv(mean_frame, OUTPUT_DIR / "mean_representative_spectra.csv")
    save_csv(oof_frame, OUTPUT_DIR / "oof_predictions.csv")
    save_csv(
        pd.DataFrame([alignment_metadata["wavenumber_calibration"]]),
        OUTPUT_DIR / "wavenumber_calibration_summary.csv",
    )
    save_csv(cohorts, OUTPUT_DIR / "api_cohort_summary.csv")

    plot_roc(y, probability, FIGURE_DIR / "clinical_roc.png")
    plot_confusion(
        y,
        prediction,
        f"outer-train optimized; median threshold={threshold_median:.3f}",
        FIGURE_DIR / "confusion_matrix.png",
    )
    plot_spectra_and_noise(
        common_grid,
        means,
        labels,
        noise_vectors,
        FIGURE_DIR / "representative_spectrum_and_noise.png",
    )
    plot_qc(qc_rows, FIGURE_DIR / "repeat_qc_pass_rate.png")
    plot_snr_peaks(snr_frame, FIGURE_DIR / "snr_peak_summary.png")
    plot_signal_noise_shading(
        common_grid,
        means,
        labels,
        noise_vectors,
        qc_rows,
        reference_peaks_cm1,
        FIGURE_DIR / "spectrum_noise_shading.png",
    )
    plot_signal_noise_band_validation(
        common_grid,
        means,
        labels,
        noise_vectors,
        qc_rows,
        band_frame,
        peak_registry,
        shift_values,
        reference_peaks_cm1,
        FIGURE_DIR / "signal_noise_band_validation.png",
    )
    plot_resolution_aware_peak_zoom(
        common_grid,
        means,
        labels,
        noise_vectors,
        band_frame,
        peak_registry,
        FIGURE_DIR / "resolution_aware_peak_zoom.png",
    )
    plot_resolution_aware_snr_zoom(
        common_grid,
        band_frame,
        peak_registry,
        FIGURE_DIR / "resolution_aware_snr_zoom.png",
    )
    copied_figures = copy_publication_figures()

    metadata = {
        "run_timestamp_utc": pd.Timestamp.utcnow().isoformat(),
        "patent_source": str(PATENT_SOURCE),
        "data_source": {
            "type": "AECD REST API + AECD PostgreSQL",
            "base_url": API_BASE_URL,
            "api_health": api_metadata["health"],
            "spectrum_total_reported": api_metadata["spectrum_total_reported"],
            "calibration_source": "measurement.calibrations",
            "database": standard_calibration_metadata["database"],
            "standard_material": CALIBRATION_STANDARD_MATERIAL,
            "calibration_records": standard_calibration_metadata["calibration_records"],
        },
        "cohort": {
            "subjects_total": int(len(subject_keys)),
            "positive_label": POSITIVE_LABEL,
            "positive_subjects": int(y.sum()),
            "negative_subjects": int((1 - y).sum()),
            "labels": sorted(set(labels)),
        },
        "grid": {
            "range_cm1": list(SPECTRAL_RANGE_CM1),
            "observed_source_step_cm1": observed_step,
            "common_grid_points": common_points,
        },
        "patent_pipeline": {
            "reference_peak_alignment": {
                "enabled": True,
                "standard_material": CALIBRATION_STANDARD_MATERIAL,
                "source": "measurement.calibrations",
                "clinical_spectra_peak_detection": False,
            },
            "common_grid_interpolation": True,
            "repeat_qc_before_mean": True,
            "mean_of_qc_passed_repeats_is_model_input": True,
            "residual_noise_after_mean": True,
            "post_mean_spectral_preprocessing": [],
            "optional_post_mean_steps_not_applied": [
                "DWT denoising",
                "baseline correction",
                "absolute intensity scaling",
                "Savitzky-Golay smoothing",
                "SNV",
                "first/second derivatives",
                "peak feature extraction",
            ],
        },
        "qc": qc_summary,
        "standard_material_calibration": standard_calibration_metadata,
        "snr": snr_metadata,
        "signal_noise_band": band_metadata,
        "alignment": alignment_metadata,
        "model": {
            "name": "direct raw mean-spectrum stacking",
            "base_models": [
                "StandardScaler + LogisticRegression(C=1.0)",
                "XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05)",
                "RandomForestClassifier(n_estimators=400)",
                "StandardScaler + LogisticRegression(C=0.1)",
            ],
            "meta_model": "elastic-net LogisticRegression",
            "evaluation": "subject-level nested GroupKFold(outer=5, inner=5)",
            "decision_threshold": threshold_median,
            "decision_threshold_policy": metrics["threshold_policy"],
            "fixed_comparison_threshold": FIXED_COMPARISON_THRESHOLD,
            "random_state": RANDOM_STATE,
        },
        "clinical_metrics": metrics,
        "publication_figures": copied_figures,
        "interpretation_limit": "Cancer Screening AUC may be affected by hospital/cohort confounding and is not cross-hospital generalization.",
    }
    (OUTPUT_DIR / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_readme(metrics, qc_summary, metadata)

    print("=== Average representative spectrum clinical evaluation ===")
    print(f"[data] subjects={len(y)} repeats={total_repeats} X={means.shape}")
    print(
        f"[QC] passed={passed_repeats}/{total_repeats} "
        f"({qc_summary['repeat_pass_rate']:.2%})"
    )
    print(
        f"[clinical] overall OOF AUC={metrics['overall_oof_auc']:.4f} "
        f"balanced_accuracy={metrics['overall_balanced_accuracy']:.4f}"
    )
    print(
        f"[threshold] outer-train optimized median={threshold_median:.4f} "
        f"range={metrics['threshold_min']:.4f}-{metrics['threshold_max']:.4f}"
    )
    print(f"[output] {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
