from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import httpx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = REPO_ROOT / "notebooks"
SRC_DIR = REPO_ROOT / "src"
OUTPUT_DIR = Path(
    os.environ.get(
        "AECD_PATENT_DWT_OUTDIR",
        str(NOTEBOOK_DIR / "aecd_api_model_patent_dwt_outputs"),
    )
)
FIGURE_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

for module_dir in (NOTEBOOK_DIR, SRC_DIR):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

from build_stkv2_dataset import extract_peak_features
from stk_v2_preprocess import STK_TARGET_GRID, stk_v2_channels
from train_stkv2_stacking import (
    DECISION_THRESHOLD,
    build_views,
    run_nested_cv,
)

from sers.preprocessing import DEFAULT_REFERENCE_PEAK_WN, calibrate_spectrum

PATENT_SOURCE = REPO_ROOT / "patent/특허_명세서_초안_SERS_반복측정_DWT_노이즈제거.docx"
API_BASE_URL = os.environ.get("AECD_API_BASE_URL", "http://127.0.0.1:8000")
API_KEY = os.environ.get("AECD_API_KEY")
TARGET_COLUMN = "cohort_group"
RANDOM_STATE = 20260819
SPECTRAL_RANGE_CM1 = (400.0, 2200.0)
CALIBRATION_TARGET_WN = DEFAULT_REFERENCE_PEAK_WN
CALIBRATION_WINDOW_CM1 = 20.0

DWT_WAVELET = "db4"
DWT_MODE = "symmetric"
DWT_THRESHOLD_SCALE = 1.0
ASLS_LAMBDA = 1e5
ASLS_ASYMMETRY = 0.01


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
        health = client.get("/health")
        health.raise_for_status()
        cohorts_response = client.get("/v1/cohorts")
        cohorts_response.raise_for_status()
        cohorts = pd.DataFrame(cohorts_response.json())

        reference_response = client.get(
            "/v1/reference_peaks",
            params={"standard_material": "Polystyrene (PS)"},
        )
        reference_response.raise_for_status()
        reference_payload = reference_response.json()

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
        "health": health.json(),
        "reference_peak_catalog_items": int(
            len(reference_payload.get("items", []))
        ),
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
            calibrated_grid, values, shift = calibrate_spectrum(
                source_grid,
                values,
                target_wn=CALIBRATION_TARGET_WN,
                window=CALIBRATION_WINDOW_CM1,
            )
            if (
                calibrated_grid[0] > common_grid[0]
                or calibrated_grid[-1] < common_grid[-1]
            ):
                raise ValueError("A spectrum does not cover the common grid range.")
            aligned_replicates.append(np.interp(common_grid, calibrated_grid, values))
            calibration_shifts.append(float(shift))

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
        "target_peak_cm1": float(CALIBRATION_TARGET_WN),
        "search_window_cm1": float(CALIBRATION_WINDOW_CM1),
        "total_aligned_spectra": int(len(shift_array)),
        "nonzero_shift_spectra": int(np.count_nonzero(nonzero_shift_mask)),
        "zero_shift_spectra": int(np.count_nonzero(~nonzero_shift_mask)),
        "mean_shift_cm1": float(np.mean(shift_array)),
        "median_shift_cm1": float(np.median(shift_array)),
        "p95_abs_shift_cm1": float(np.percentile(abs_shift_array, 95)),
    }

    return subject_keys, labels, replicate_matrices, {
        "skipped_ambiguous_subjects": skipped_subjects,
        "replicate_count_distribution": {
            str(key): int(value) for key, value in sorted(replicate_counts.items())
        },
        "wavenumber_calibration": calibration_summary,
    }


def asls_baseline(
    values: np.ndarray,
    lam: float = ASLS_LAMBDA,
    asymmetry: float = ASLS_ASYMMETRY,
    iterations: int = 10,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    length = len(values)
    difference = sparse.diags(
        [1.0, -2.0, 1.0],
        [0, -1, -2],
        shape=(length, length - 2),
        dtype=float,
        format="csc",
    )
    penalty = lam * difference.dot(difference.T)
    weights = np.ones(length, dtype=np.float64)
    for _ in range(iterations):
        weight_matrix = sparse.spdiags(weights, 0, length, length).tocsc()
        baseline = spsolve(
            (weight_matrix + penalty).tocsc(),
            weights * values,
        )
        weights = asymmetry * (values > baseline) + (1 - asymmetry) * (
            values < baseline
        )
    return baseline


def estimate_dwt_thresholds(
    residuals: np.ndarray,
    wavelet: str,
    mode: str,
    level: int,
) -> tuple[dict[int, float], list[dict[str, float]], list[list[np.ndarray]]]:
    n_replicates = residuals.shape[0]
    if n_replicates < 2:
        raise ValueError("At least two repeats are required for residual noise estimation.")

    residual_coefficients = [
        pywt.wavedec(row, wavelet, mode=mode, level=level)
        for row in residuals
    ]
    thresholds: dict[int, float] = {}
    rows: list[dict[str, float]] = []

    for coefficient_index in range(1, len(residual_coefficients[0])):
        detail_stack = np.stack(
            [coefficients[coefficient_index] for coefficients in residual_coefficients],
            axis=0,
        )
        coefficient_mean = detail_stack.mean(axis=0)
        n_coefficients = detail_stack.shape[1]
        variance = np.sum(
            (detail_stack - coefficient_mean) ** 2,
            axis=0,
        ) / (n_replicates - 1)
        sigma_level = float(np.sqrt(np.mean(variance)))
        threshold = float(
            DWT_THRESHOLD_SCALE
            * sigma_level
            * np.sqrt(2.0 * np.log(max(n_coefficients, 2)))
        )
        level_number = level - coefficient_index + 1
        thresholds[coefficient_index] = threshold
        rows.append(
            {
                "wavelet_level": int(level_number),
                "coefficient_index": int(coefficient_index),
                "n_coefficients": int(n_coefficients),
                "sigma_level": sigma_level,
                "universal_threshold": threshold,
            }
        )

    return thresholds, rows, residual_coefficients


def dwt_soft_denoise(
    values: np.ndarray,
    thresholds: dict[int, float],
    wavelet: str,
    mode: str,
    level: int,
) -> np.ndarray:
    coefficients = pywt.wavedec(values, wavelet, mode=mode, level=level)
    denoised_coefficients = [coefficients[0]]
    for coefficient_index, detail in enumerate(coefficients[1:], start=1):
        denoised_coefficients.append(
            pywt.threshold(
                detail,
                thresholds[coefficient_index],
                mode="soft",
            )
        )
    reconstructed = pywt.waverec(
        denoised_coefficients,
        wavelet,
        mode=mode,
    )
    return np.asarray(reconstructed[: len(values)], dtype=np.float64)


def lag_one_correlation(residuals: np.ndarray) -> float:
    values = []
    for row in residuals:
        if np.std(row[:-1]) <= np.finfo(float).eps:
            continue
        values.append(float(np.corrcoef(row[:-1], row[1:])[0, 1]))
    return float(np.mean(values)) if values else float("nan")


def patent_preprocess_subject(
    replicates: np.ndarray,
    wavelet: str,
    mode: str,
    level: int,
) -> tuple[np.ndarray, dict[str, float], list[dict[str, float]], dict[str, np.ndarray]]:
    raw_mean = replicates.mean(axis=0)
    raw_residuals = replicates - raw_mean
    thresholds, threshold_rows, _ = estimate_dwt_thresholds(
        raw_residuals,
        wavelet=wavelet,
        mode=mode,
        level=level,
    )
    denoised_replicates = np.vstack(
        [
            dwt_soft_denoise(
                row,
                thresholds=thresholds,
                wavelet=wavelet,
                mode=mode,
                level=level,
            )
            for row in replicates
        ]
    )
    denoised_mean = denoised_replicates.mean(axis=0)

    centered_replicates = denoised_replicates - np.median(
        denoised_replicates,
        axis=1,
        keepdims=True,
    )
    scale_factors = np.sqrt(np.mean(centered_replicates**2, axis=1))
    safe_scale_factors = np.where(
        scale_factors <= np.finfo(float).eps,
        1.0,
        scale_factors,
    )
    relative_scaled_replicates = denoised_replicates / safe_scale_factors[:, None]
    relative_scaled_mean = relative_scaled_replicates.mean(axis=0)
    baseline = asls_baseline(relative_scaled_mean)
    scaled_mean = relative_scaled_mean - baseline

    denoised_residuals = denoised_replicates - denoised_mean
    raw_noise = float(np.sqrt(np.mean(raw_residuals**2)))
    dwt_noise = float(np.sqrt(np.mean(denoised_residuals**2)))
    quality = {
        "replicates": int(len(replicates)),
        "raw_noise_rms": raw_noise,
        "dwt_noise_rms": dwt_noise,
        "noise_reduction_pct": float(
            100.0 * (1.0 - dwt_noise / raw_noise)
            if raw_noise > np.finfo(float).eps
            else 0.0
        ),
        "raw_residual_lag1_corr": lag_one_correlation(raw_residuals),
        "dwt_residual_lag1_corr": lag_one_correlation(denoised_residuals),
        "scale_factor_rms": float(np.median(scale_factors)),
        "scale_factor_rms_p25": float(np.percentile(scale_factors, 25)),
        "scale_factor_rms_p75": float(np.percentile(scale_factors, 75)),
    }
    examples = {
        "raw_mean": raw_mean,
        "denoised_mean": denoised_mean,
        "relative_scaled_mean": relative_scaled_mean,
        "scaled_mean": scaled_mean,
        "raw_residual": raw_residuals[0],
        "dwt_residual": denoised_residuals[0],
    }
    return scaled_mean, quality, threshold_rows, examples


def build_stkv2_views(
    subject_spectra: np.ndarray,
    source_grid: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    channels = np.stack(
        [
            stk_v2_channels(
                source_grid,
                row,
                target_grid=STK_TARGET_GRID,
            )
            for row in subject_spectra
        ]
    )
    peak_features = np.vstack(
        [
            extract_peak_features(STK_TARGET_GRID, row)
            for row in channels[:, 0, :]
        ]
    )
    views = build_views(
        {
            "raw": channels[:, 0, :],
            "d1": channels[:, 1, :],
            "d2": channels[:, 2, :],
            "peak": peak_features,
        }
    )
    return views, channels


def evaluate_method(
    name: str,
    subject_spectra: np.ndarray,
    source_grid: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> dict:
    views, channels = build_stkv2_views(subject_spectra, source_grid)
    oof, fold_metrics = run_nested_cv(
        views=views,
        y=y,
        groups=groups,
        n_outer=5,
        n_inner=5,
        seed=0,
    )
    fold_metrics_array = np.asarray(fold_metrics, dtype=np.float64)
    predictions = (oof >= DECISION_THRESHOLD).astype(int)
    auc = float(roc_auc_score(y, oof))
    bacc = float(balanced_accuracy_score(y, predictions))
    cm = confusion_matrix(y, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "name": name,
        "views": views,
        "channels": channels,
        "oof": oof,
        "fold_metrics": fold_metrics_array,
        "confusion_matrix": cm,
        "metrics": {
            "method": name,
            "threshold": float(DECISION_THRESHOLD),
            "fold_auc_mean": float(fold_metrics_array[:, 0].mean()),
            "fold_auc_std": float(fold_metrics_array[:, 0].std()),
            "overall_oof_auc": auc,
            "fold_balanced_accuracy_mean": float(fold_metrics_array[:, 1].mean()),
            "fold_balanced_accuracy_std": float(fold_metrics_array[:, 1].std()),
            "overall_balanced_accuracy": bacc,
            "sensitivity": float(recall_score(y, predictions, zero_division=0)),
            "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
            "precision": float(precision_score(y, predictions, zero_division=0)),
            "f1": float(f1_score(y, predictions, zero_division=0)),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "n_subjects": int(len(y)),
            "n_cancer": int(y.sum()),
            "n_non_cancer": int((y == 0).sum()),
        },
    }


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def plot_roc(results: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    for result in results:
        fpr, tpr, _ = roc_curve(
            Y_GLOBAL,
            result["oof"],
        )
        auc = result["metrics"]["overall_oof_auc"]
        ax.plot(fpr, tpr, linewidth=2, label=f"{result['name']}: AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_title("Cancer Screening: subject-level nested OOF ROC")
    ax.set_xlabel("1 - Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.text(
        0.5,
        0.01,
        "Research-use only. Cancer Screening AUC may be hospital-confounded.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(result: dict, path: Path) -> None:
    cm = result["confusion_matrix"]
    normalized = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks([0, 1], ["Non-cancer", "Cancer"])
    ax.set_yticks([0, 1], ["Non-cancer", "Cancer"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Patent DWT preprocessing: OOF confusion")
    for row in range(2):
        for column in range(2):
            ax.text(
                column,
                row,
                f"{cm[row, column]}\n({normalized[row, column] * 100:.0f}%)",
                ha="center",
                va="center",
                color="white" if normalized[row, column] > 0.5 else "black",
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Row proportion")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_preprocessing(example: dict[str, np.ndarray], grid: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True)
    axes[0].plot(grid, example["raw_mean"], color="#555555", linewidth=1)
    axes[0].set_title(
        "Representative subject spectrum after reference-peak alignment (before DWT)"
    )
    axes[0].set_ylabel("Intensity")
    axes[0].grid(alpha=0.2)
    axes[1].plot(grid, example["scaled_mean"], color="#1f77b4", linewidth=1)
    axes[1].set_title("After residual-derived DWT, AsLS baseline, and relative RMS scaling")
    axes[1].set_xlabel("Raman shift (cm$^{-1}$)")
    axes[1].set_ylabel("Scaled intensity")
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_repeatability(quality_rows: list[dict[str, float]], path: Path) -> None:
    frame = pd.DataFrame(quality_rows)
    labels = ["Raw residual", "After DWT"]
    values = [frame["raw_noise_rms"], frame["dwt_noise_rms"]]
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    positions = np.arange(2)
    ax.boxplot(
        values,
        positions=positions,
        widths=0.45,
        patch_artist=True,
        boxprops={"facecolor": "#9ecae1"},
        medianprops={"color": "#d62728", "linewidth": 2},
    )
    ax.set_xticks(positions, labels)
    ax.set_ylabel("Replicate residual RMS")
    ax.set_title("Repeatability noise proxy before and after DWT")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_thresholds(threshold_frame: pd.DataFrame, path: Path) -> None:
    plot_frame = threshold_frame.sort_values("wavelet_level")
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(
        plot_frame["wavelet_level"],
        plot_frame["sigma_level_median"],
        "o-",
        label="Median residual sigma",
    )
    ax.plot(
        plot_frame["wavelet_level"],
        plot_frame["threshold_median"],
        "o-",
        label="Median universal threshold",
    )
    ax.set_xlabel("DWT level")
    ax.set_ylabel("Coefficient scale")
    ax.set_title("Residual-derived DWT noise parameters")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    global Y_GLOBAL

    items, cohort_summary, api_context = load_api_spectra()
    common_grid, observed_step, common_points = construct_common_grid(items)
    subject_keys, labels, replicate_matrices, grouping_context = (
        group_and_align_subjects(items, common_grid)
    )

    labels_array = np.asarray(labels, dtype=str)
    y = (labels_array == "prostate").astype(int)
    if len(np.unique(y)) != 2:
        raise ValueError("Cancer Screening target does not contain two classes.")
    groups = np.arange(len(y), dtype=int)
    Y_GLOBAL = y

    dwt_subjects = []
    quality_rows: list[dict[str, float]] = []
    threshold_rows: list[dict[str, float]] = []
    example: dict[str, np.ndarray] | None = None
    for subject_index, replicates in enumerate(replicate_matrices):
        dwt_mean, quality, subject_thresholds, subject_example = patent_preprocess_subject(
            replicates,
            wavelet=DWT_WAVELET,
            mode=DWT_MODE,
            level=pywt.dwt_max_level(
                len(common_grid),
                pywt.Wavelet(DWT_WAVELET).dec_len,
            ),
        )
        dwt_subjects.append(dwt_mean)
        quality_rows.append(quality)
        for row in subject_thresholds:
            threshold_rows.append(row | {"subject_index_internal": subject_index})
        if example is None:
            example = subject_example

    if example is None:
        raise RuntimeError("No preprocessing example was produced.")

    dwt_subject_matrix = np.vstack(dwt_subjects)
    raw_subject_matrix = np.vstack(
        [replicates.mean(axis=0) for replicates in replicate_matrices]
    )

    results = [
        evaluate_method(
            "Existing raw subject-mean",
            raw_subject_matrix,
            common_grid,
            y,
            groups,
        ),
        evaluate_method(
            "Patent DWT preprocessing",
            dwt_subject_matrix,
            common_grid,
            y,
            groups,
        ),
    ]

    threshold_frame = (
        pd.DataFrame(threshold_rows)
        .groupby(["wavelet_level", "coefficient_index", "n_coefficients"], as_index=False)
        .agg(
            sigma_level_median=("sigma_level", "median"),
            sigma_level_p25=("sigma_level", lambda values: np.percentile(values, 25)),
            sigma_level_p75=("sigma_level", lambda values: np.percentile(values, 75)),
            threshold_median=("universal_threshold", "median"),
            threshold_p25=("universal_threshold", lambda values: np.percentile(values, 25)),
            threshold_p75=("universal_threshold", lambda values: np.percentile(values, 75)),
            subjects=("subject_index_internal", "nunique"),
        )
    )
    quality_frame = pd.DataFrame(quality_rows)
    quality_summary = pd.DataFrame(
        [
            {
                "stage": "raw residual",
                "rms_median": float(quality_frame["raw_noise_rms"].median()),
                "rms_p25": float(quality_frame["raw_noise_rms"].quantile(0.25)),
                "rms_p75": float(quality_frame["raw_noise_rms"].quantile(0.75)),
                "lag1_corr_median": float(
                    quality_frame["raw_residual_lag1_corr"].median()
                ),
            },
            {
                "stage": "after DWT",
                "rms_median": float(quality_frame["dwt_noise_rms"].median()),
                "rms_p25": float(quality_frame["dwt_noise_rms"].quantile(0.25)),
                "rms_p75": float(quality_frame["dwt_noise_rms"].quantile(0.75)),
                "lag1_corr_median": float(
                    quality_frame["dwt_residual_lag1_corr"].median()
                ),
            },
        ]
    )

    performance_frame = pd.DataFrame([result["metrics"] for result in results])
    fold_frame = pd.DataFrame(
        [
            {
                "method": result["name"],
                "fold": fold_index,
                "fold_auc": float(values[0]),
                "fold_balanced_accuracy": float(values[1]),
            }
            for result in results
            for fold_index, values in enumerate(result["fold_metrics"], start=1)
        ]
    )
    save_csv(performance_frame, OUTPUT_DIR / "clinical_performance_summary.csv")
    save_csv(fold_frame, OUTPUT_DIR / "clinical_fold_metrics.csv")
    save_csv(threshold_frame, OUTPUT_DIR / "dwt_threshold_summary.csv")
    save_csv(quality_summary, OUTPUT_DIR / "repeatability_quality_summary.csv")
    save_csv(cohort_summary, OUTPUT_DIR / "api_cohort_summary.csv")
    save_csv(
        pd.DataFrame([grouping_context["wavenumber_calibration"]]),
        OUTPUT_DIR / "wavenumber_calibration_summary.csv",
    )

    plot_roc(results, FIGURE_DIR / "patent_dwt_clinical_roc_comparison.png")
    plot_confusion(
        results[1],
        FIGURE_DIR / "patent_dwt_clinical_confusion_matrix.png",
    )
    plot_preprocessing(
        example,
        common_grid,
        FIGURE_DIR / "patent_dwt_preprocessing_example.png",
    )
    plot_repeatability(
        quality_rows,
        FIGURE_DIR / "patent_dwt_repeatability_noise_reduction.png",
    )
    plot_thresholds(
        threshold_frame,
        FIGURE_DIR / "patent_dwt_wavelet_thresholds.png",
    )

    metadata = {
        "patent_source": str(PATENT_SOURCE),
        "api_base_url": API_BASE_URL,
        "api_health": api_context["health"],
        "observed_spectra": int(len(items)),
        "observed_subjects": int(len(subject_keys)),
        "cohort_labels": {
            label: int(count) for label, count in Counter(labels_array).items()
        },
        "target_definition": {
            "column": TARGET_COLUMN,
            "positive": "prostate",
            "negative": ["control", "prostate disease control"],
        },
        "common_grid": {
            "min_cm1": float(common_grid[0]),
            "max_cm1": float(common_grid[-1]),
            "points": int(common_points),
            "observed_source_step_cm1": observed_step,
            "common_grid_step_cm1": float(np.median(np.diff(common_grid))),
        },
        "wavenumber_shift": {
            "applied": True,
            "method": (
                "Per-spectrum intensity-derived reference-peak calibration using "
                "the repository calibrate_spectrum helper."
            ),
            **grouping_context["wavenumber_calibration"],
        },
        "dwt": {
            "wavelet": DWT_WAVELET,
            "mode": DWT_MODE,
            "level": int(
                pywt.dwt_max_level(
                    len(common_grid),
                    pywt.Wavelet(DWT_WAVELET).dec_len,
                )
            ),
            "threshold": "level-wise residual universal threshold",
            "threshold_scale": DWT_THRESHOLD_SCALE,
            "detail_thresholding": "soft",
            "approximation": "preserved",
        },
        "post_dwt_processing": {
            "baseline": "AsLS",
            "baseline_lambda": ASLS_LAMBDA,
            "baseline_asymmetry": ASLS_ASYMMETRY,
            "relative_scale": (
                "per-repeat median-centered RMS before repeat mean; AsLS baseline "
                "applied after the relative-scaled repeat mean"
            ),
        },
        "clinical_evaluation": {
            "model": "existing STK-V2 stacking",
            "outer_cv": "5-fold subject-level GroupKFold",
            "inner_cv": "5-fold subject-level GroupKFold",
            "replicate_leakage": "one row per subject; no replicate crosses a split",
            "decision_threshold": float(DECISION_THRESHOLD),
        },
        "grouping": grouping_context,
        "limitations": [
            "The API has no independent measured calibration field; the reference-peak shift is derived from each intensity trace using the repository urea anchor at 1001.4 cm^-1.",
            "Cancer Screening can be hospital-confounded; these results are not evidence of cross-hospital generalization.",
            "The current API cohort has repeated spectra per subject; deployment with fewer or no repeats needs a separate validation design.",
        ],
    }
    (OUTPUT_DIR / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=== Patent DWT clinical evaluation ===")
    print(json.dumps(metadata["cohort_labels"], ensure_ascii=False))
    print(performance_frame.to_string(index=False))
    print("output_dir=", OUTPUT_DIR)


if __name__ == "__main__":
    main()
