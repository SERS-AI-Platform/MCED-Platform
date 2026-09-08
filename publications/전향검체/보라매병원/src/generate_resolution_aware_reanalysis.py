from __future__ import annotations

import csv
import sys
from pathlib import Path
from runpy import run_path
from typing import Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = ["DejaVu Sans"]
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.signal import find_peaks
from scipy.sparse.linalg import spsolve
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

REPO: Final = Path(__file__).resolve().parents[4]
SOURCE_DIR: Final = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE_DIR))
sys.path.insert(0, str(REPO / "src"))

from boramae_data import (
    MODEL_GRID,
    collect_boramae_files,
    load_clinical_samples,
    raw_key,
)
from prostate_comparison_model import OofResult, build_task, nested_oof
from sers.io import read_spectrum

RAW_ROOT: Final = REPO / "data" / "raw_data" / "20260709_BPRO,BNOR_1mW_0.05s_Ave100"
OUT: Final = REPO / "publications" / "전향검체" / "보라매병원" / "resolution_aware_reanalysis"
FIG_DIR: Final = OUT / "figures"
TABLE_DIR: Final = OUT / "tables"
RANDOM_STATE: Final = 20260819
SPECTRAL_RANGE_CM1: Final = (400.0, 2200.0)
INSTRUMENT_RESOLUTION_FWHM_CM1: Final = 2.0
BASELINE_LAMBDA: Final = 1e5
BASELINE_ASYMMETRY: Final = 0.01
QC_MAD_THRESHOLD: Final = 3.5
BURDEN_REPEATS: Final = 5
BURDEN_AUC_TOLERANCE: Final = 0.02

LABELS: Final = ("Control", "Biopsy-negative", "Prostate cancer")
LABEL_DISPLAY: Final = {
    "Control": "Control",
    "Biopsy-negative": "Biopsy-negative",
    "Prostate cancer": "Cancer",
}
COLORS: Final = {
    "Control": "#2C7FB8",
    "Biopsy-negative": "#7A5195",
    "Prostate cancer": "#D95F02",
}


def _asls_baseline(
    values: np.ndarray,
    lam: float = BASELINE_LAMBDA,
    p: float = BASELINE_ASYMMETRY,
    niter: int = 10,
) -> np.ndarray:
    values_array = np.asarray(values, dtype=np.float64)
    length = len(values_array)
    if length < 3:
        return np.zeros_like(values_array)
    difference = sparse.diags(
        [1.0, -2.0, 1.0],
        [0, -1, -2],
        shape=(length, length - 2),
        dtype=float,
        format="csc",
    )
    penalty = (lam * difference.dot(difference.T)).tocsc()
    weights = np.ones(length, dtype=float)
    for _ in range(niter):
        weighted = sparse.spdiags(weights, 0, length, length).tocsc()
        baseline = spsolve((weighted + penalty).tocsc(), weights * values_array)
        weights = p * (values_array > baseline) + (1.0 - p) * (values_array < baseline)
    return np.asarray(baseline, dtype=np.float64)


def baseline_area_normalize(spectra: np.ndarray, grid: np.ndarray) -> np.ndarray:
    values = np.asarray(spectra, dtype=np.float64)
    matrix = values[None, :] if values.ndim == 1 else values
    processed = np.empty_like(matrix, dtype=np.float64)
    for index, row in enumerate(matrix):
        corrected = np.clip(row - _asls_baseline(row), 0.0, None)
        area = float(np.trapezoid(corrected, grid))
        processed[index] = corrected / area if area > 0.0 else corrected
    return processed[0] if values.ndim == 1 else processed


def qc_keep_mask(replicates: np.ndarray, mad_threshold: float = QC_MAD_THRESHOLD) -> np.ndarray:
    values = np.asarray(replicates, dtype=np.float64)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("replicates must be a non-empty 2D array")
    if len(values) < 2:
        return np.ones(len(values), dtype=bool)
    median = np.median(values, axis=0)
    residual = values - median
    scale = np.maximum(np.median(np.abs(residual), axis=0), np.finfo(float).eps)
    distance = np.sqrt(np.mean((residual / scale) ** 2, axis=1))
    distance_median = float(np.median(distance))
    distance_mad = float(np.median(np.abs(distance - distance_median)))
    if distance_mad == 0.0:
        limit = distance_median + 3.0 * float(np.std(distance))
    else:
        limit = distance_median + mad_threshold * 1.4826 * distance_mad
    keep = distance <= max(limit, distance_median)
    if int(keep.sum()) < 2:
        keep = np.zeros(len(values), dtype=bool)
        keep[np.argsort(distance)[:2]] = True
    return keep


def estimate_noise_sigma(values: np.ndarray) -> float:
    differences = np.diff(np.asarray(values, dtype=np.float64))
    if len(differences) == 0:
        return float(np.finfo(float).eps)
    sigma = float(np.median(np.abs(differences - np.median(differences))) / 0.6744897501960817)
    return max(sigma / np.sqrt(2.0), float(np.finfo(float).eps))


def load_subject_replicates(
    grid: np.ndarray,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, dict[str, list[Path]]]:
    files = collect_boramae_files()
    included = [
        sample
        for sample in load_clinical_samples()
        if not sample.excluded and sample.group in LABELS
    ]
    subject_keys: list[str] = []
    subject_labels: list[str] = []
    replicate_rows: list[np.ndarray] = []
    replicate_subjects: list[str] = []
    for sample in included:
        key = raw_key(sample, files)
        paths = sorted(files[key])
        subject_keys.append(sample.label)
        subject_labels.append(sample.group)
        for path in paths:
            x_values, y_values = read_spectrum(path)
            mask = (x_values >= float(grid[0])) & (x_values <= float(grid[-1]))
            if int(mask.sum()) < 2:
                raise RuntimeError(f"Spectrum does not cover the model grid: {path.name}")
            replicate_rows.append(np.interp(grid, x_values[mask], y_values[mask]))
            replicate_subjects.append(sample.label)
    return (
        subject_keys,
        np.asarray(subject_labels, dtype=str),
        np.vstack(replicate_rows),
        np.asarray(replicate_subjects, dtype=str),
        files,
    )


def build_subject_matrices(
    subject_keys: list[str],
    subject_labels: np.ndarray,
    raw_replicates: np.ndarray,
    replicate_subjects: np.ndarray,
    grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    qc_replicates: list[np.ndarray] = []
    qc_counts: list[int] = []
    raw_subject_means: list[np.ndarray] = []
    processed_subject_means: list[np.ndarray] = []
    kept_subject_keys: list[str] = []
    kept_labels: list[str] = []
    for key, label in zip(subject_keys, subject_labels, strict=True):
        values = raw_replicates[replicate_subjects == key]
        keep = qc_keep_mask(values)
        selected = values[keep]
        qc_replicates.append(selected)
        qc_counts.append(int(len(selected)))
        raw_subject_means.append(selected.mean(axis=0))
        processed_subject_means.append(baseline_area_normalize(selected.mean(axis=0), grid))
        kept_subject_keys.append(key)
        kept_labels.append(label)
    return (
        np.vstack(raw_subject_means),
        np.vstack(processed_subject_means),
        np.asarray(kept_labels, dtype=str),
        np.asarray(qc_counts, dtype=int),
        qc_replicates,
    )


def save_result(
    name: str,
    result: OofResult,
    class_names: tuple[str, ...],
) -> None:
    metrics = result.metrics
    intervals = result.intervals
    with (TABLE_DIR / f"{name}_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value", "ci95_low", "ci95_high", "n"])
        for metric, value in metrics.items():
            low, high = intervals[metric]
            writer.writerow([metric, value, low, high, len(result.y_true)])
    with (TABLE_DIR / f"{name}_confusion_matrix.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_label", *[f"pred_{label}" for label in class_names]])
        for label, row in zip(class_names, result.confusion, strict=True):
            writer.writerow([label, *row.tolist()])
    with (TABLE_DIR / f"{name}_oof_predictions.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case_id", "fold", "true_label", "pred_label", *[f"prob_{label}" for label in class_names]])
        for index, (true_value, pred_value, fold, probabilities) in enumerate(
            zip(result.y_true, result.y_pred, result.folds, result.probabilities, strict=True),
            start=1,
        ):
            writer.writerow([
                f"BORAMAE-{index:03d}",
                fold,
                class_names[int(true_value)],
                class_names[int(pred_value)],
                *probabilities,
            ])


def plot_preprocessing(
    grid: np.ndarray,
    raw_replicates: np.ndarray,
    raw_subject_mean: np.ndarray,
    processed_subject_mean: np.ndarray,
    output: Path,
) -> None:
    baseline = _asls_baseline(raw_subject_mean)
    corrected = np.clip(raw_subject_mean - baseline, 0.0, None)
    figure, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    for row in raw_replicates:
        axes[0, 0].plot(grid, row, color="#9ECAE1", alpha=0.35, linewidth=0.65)
    axes[0, 0].plot(grid, raw_subject_mean, color="#08519C", linewidth=1.4, label="QC-passed raw mean")
    axes[0, 0].set_title("Raw QC-passed replicates (no smoothing)", loc="left", fontweight="bold")
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 1].plot(grid, raw_subject_mean, color="#08519C", linewidth=1.1, label="raw mean")
    axes[0, 1].plot(grid, baseline, color="#CB181D", linewidth=1.1, label="AsLS baseline")
    axes[0, 1].set_title("AsLS baseline estimate", loc="left", fontweight="bold")
    axes[0, 1].legend(frameon=False, fontsize=8)
    axes[1, 0].plot(grid, corrected, color="#238B45", linewidth=1.1)
    axes[1, 0].set_title("Baseline-corrected positive signal", loc="left", fontweight="bold")
    axes[1, 1].plot(grid, processed_subject_mean, color="#54278F", linewidth=1.2)
    axes[1, 1].set_title("Area-normalized subject representation", loc="left", fontweight="bold")
    for axis in axes.flat:
        axis.set_ylabel("Intensity / normalized area")
        axis.grid(alpha=0.2)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    for axis in axes[1]:
        axis.set_xlabel("Raman shift (cm$^{-1}$)")
    figure.suptitle(
        "Boramae resolution-aware reanalysis: preprocessing audit\n"
        "No Savitzky-Golay or convolution smoothing; AsLS + area normalization",
        x=0.02,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(output, dpi=240, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def plot_group_spectra(
    grid: np.ndarray,
    processed: np.ndarray,
    labels: np.ndarray,
    output: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(14, 5.8))
    for label in LABELS:
        matrix = processed[labels == label]
        mean = matrix.mean(axis=0)
        lower = np.percentile(matrix, 25, axis=0)
        upper = np.percentile(matrix, 75, axis=0)
        axis.plot(grid, mean, color=COLORS[label], linewidth=1.5, label=f"{LABEL_DISPLAY[label]} (n={len(matrix)})")
        axis.fill_between(grid, lower, upper, color=COLORS[label], alpha=0.12, linewidth=0)
    axis.set_title("Processed subject spectra by clinical group", loc="left", fontweight="bold")
    axis.set_xlabel("Raman shift (cm$^{-1}$)")
    axis.set_ylabel("Area-normalized intensity")
    axis.legend(frameon=False, ncol=3)
    axis.grid(alpha=0.2)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.tight_layout()
    figure.savefig(output, dpi=240, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def plot_binary_performance(result: OofResult, output: Path) -> None:
    y_true = result.y_true
    y_pred = result.y_pred
    probabilities = result.probabilities
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw={"width_ratios": [1, 1.1, 1.15]})
    matrix = result.confusion
    axes[0].imshow(matrix, cmap="Blues")
    axes[0].set_xticks([0, 1], ["Non-cancer", "Cancer"], rotation=25, ha="right")
    axes[0].set_yticks([0, 1], ["Non-cancer", "Cancer"])
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_title("Screening confusion matrix")
    for row in range(2):
        for column in range(2):
            axes[0].text(column, row, str(matrix[row, column]), ha="center", va="center")
    score = probabilities[:, 1]
    fpr, tpr, _ = roc_curve(y_true, score)
    axes[1].plot(fpr, tpr, color=COLORS["Prostate cancer"], linewidth=2, label=f"AUC={result.metrics['roc_auc']:.3f}")
    axes[1].plot([0, 1], [0, 1], "--", color="0.5", linewidth=0.8)
    axes[1].set_xlabel("False positive rate")
    axes[1].set_ylabel("True positive rate")
    axes[1].set_title("Nested subject-level OOF ROC")
    axes[1].legend(frameon=False, loc="lower right")
    metric_names = ("roc_auc", "balanced_accuracy", "sensitivity", "specificity")
    metric_labels = ("ROC-AUC", "Balanced acc.", "Sensitivity", "Specificity")
    axes[2].bar(np.arange(4), [result.metrics[name] for name in metric_names], color=COLORS["Prostate cancer"])
    axes[2].set_xticks(np.arange(4), metric_labels, rotation=25, ha="right")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title("Screening metrics")
    axes[2].grid(axis="y", alpha=0.2)
    figure.suptitle("Boramae Cancer Screening: Control + Biopsy-negative vs Cancer", y=1.02, fontweight="bold")
    figure.tight_layout()
    figure.savefig(output, dpi=240, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def plot_three_group_performance(result: OofResult, output: Path) -> None:
    y_true = result.y_true
    probabilities = result.probabilities
    matrix = result.confusion
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1, 1.2, 1.1]})
    axes[0].imshow(matrix, cmap="Blues")
    axes[0].set_xticks(range(3), [LABEL_DISPLAY[label] for label in LABELS], rotation=25, ha="right")
    axes[0].set_yticks(range(3), [LABEL_DISPLAY[label] for label in LABELS])
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_title("3-group confusion matrix")
    for row in range(3):
        for column in range(3):
            axes[0].text(column, row, str(matrix[row, column]), ha="center", va="center")
    for class_index, label in enumerate(LABELS):
        fpr, tpr, _ = roc_curve((y_true == class_index).astype(int), probabilities[:, class_index])
        axes[1].plot(fpr, tpr, color=COLORS[label], linewidth=1.7, label=LABEL_DISPLAY[label])
    axes[1].plot([0, 1], [0, 1], "--", color="0.5", linewidth=0.8)
    axes[1].set_xlabel("False positive rate")
    axes[1].set_ylabel("True positive rate")
    axes[1].set_title(f"One-vs-rest ROC; macro AUC={result.metrics['macro_ovr_roc_auc']:.3f}")
    axes[1].legend(frameon=False, loc="lower right", fontsize=8)
    metric_names = ("macro_ovr_roc_auc", "balanced_accuracy", "macro_f1")
    metric_labels = ("Macro OVR AUC", "Balanced acc.", "Macro F1")
    axes[2].bar(np.arange(3), [result.metrics[name] for name in metric_names], color="#4D4D4D")
    axes[2].set_xticks(np.arange(3), metric_labels, rotation=25, ha="right")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title("3-group metrics")
    axes[2].grid(axis="y", alpha=0.2)
    figure.suptitle("Boramae 3-group classification", y=1.02, fontweight="bold")
    figure.tight_layout()
    figure.savefig(output, dpi=240, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def detect_resolution_aware_peaks(
    grid: np.ndarray,
    processed: np.ndarray,
    labels: np.ndarray,
) -> pd.DataFrame:
    grid_step = float(np.median(np.diff(grid)))
    min_distance_points = max(1, int(np.ceil(INSTRUMENT_RESOLUTION_FWHM_CM1 / grid_step)))
    rows: list[dict[str, float | int | str]] = []
    for label in LABELS:
        group_mean = processed[labels == label].mean(axis=0)
        noise_sigma = estimate_noise_sigma(group_mean)
        prominence_threshold = max(3.0 * noise_sigma, float(np.finfo(float).eps))
        indices, properties = find_peaks(
            group_mean,
            prominence=prominence_threshold,
            distance=min_distance_points,
        )
        for index, prominence in zip(indices, properties["prominences"], strict=True):
            rows.append({
                "clinical_group": label,
                "display_group": LABEL_DISPLAY[label],
                "peak_cm-1": float(grid[index]),
                "mean_intensity": float(group_mean[index]),
                "prominence": float(prominence),
                "noise_sigma": noise_sigma,
                "prominence_threshold": prominence_threshold,
                "instrument_resolution_fwhm_cm-1": INSTRUMENT_RESOLUTION_FWHM_CM1,
                "grid_step_cm-1": grid_step,
                "minimum_peak_distance_points": min_distance_points,
                "minimum_peak_distance_cm-1": min_distance_points * grid_step,
            })
    return pd.DataFrame(rows).sort_values(["clinical_group", "peak_cm-1"]).reset_index(drop=True)


def plot_resolution_aware_peaks(
    grid: np.ndarray,
    processed: np.ndarray,
    labels: np.ndarray,
    registry: pd.DataFrame,
    output: Path,
) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for axis, label in zip(axes, LABELS, strict=True):
        matrix = processed[labels == label]
        mean = matrix.mean(axis=0)
        axis.plot(grid, mean, color=COLORS[label], linewidth=1.1)
        peaks = registry.loc[registry["clinical_group"] == label]
        axis.scatter(peaks["peak_cm-1"], peaks["mean_intensity"], color="#111111", s=14, zorder=3)
        axis.set_ylabel("Area-normalized\nintensity")
        axis.set_title(
            f"{LABEL_DISPLAY[label]}: {len(peaks)} candidates; no peak bin merging",
            loc="left",
            fontweight="bold",
        )
        axis.grid(alpha=0.2)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[-1].set_xlabel("Raman shift (cm$^{-1}$)")
    figure.suptitle(
        "Resolution-aware peak candidates\n"
        f"unsmoothed processed spectra; FWHM={INSTRUMENT_RESOLUTION_FWHM_CM1:.1f} cm$^{{-1}}; "
        "min distance derived from grid/FWHM",
        x=0.02,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(output, dpi=240, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def write_manifest(
    grid: np.ndarray,
    labels: np.ndarray,
    raw_replicates: np.ndarray,
    qc_counts: np.ndarray,
    registry: pd.DataFrame,
) -> None:
    rows = [
        ("input_raw_root", str(RAW_ROOT)),
        ("clinical_label_source", "data/clinical_data/보라매 병원 임상정보.xlsx via existing documented parser"),
        ("included_subjects", int(len(labels))),
        ("control_subjects", int(np.sum(labels == "Control"))),
        ("biopsy_negative_subjects", int(np.sum(labels == "Biopsy-negative"))),
        ("cancer_subjects", int(np.sum(labels == "Prostate cancer"))),
        ("input_replicates", int(len(raw_replicates))),
        ("grid_min_cm-1", float(grid[0])),
        ("grid_max_cm-1", float(grid[-1])),
        ("grid_points", int(len(grid))),
        ("grid_step_cm-1", float(np.median(np.diff(grid)))),
        ("instrument_resolution_fwhm_cm-1", INSTRUMENT_RESOLUTION_FWHM_CM1),
        ("baseline", "AsLS"),
        ("baseline_lambda", BASELINE_LAMBDA),
        ("baseline_asymmetry_p", BASELINE_ASYMMETRY),
        ("normalization", "positive baseline-corrected area"),
        ("smoothing", "disabled"),
        ("qc_rule", f"replicate robust MAD distance; threshold={QC_MAD_THRESHOLD}"),
        ("qc_passed_min", int(np.min(qc_counts))),
        ("qc_passed_median", float(np.median(qc_counts))),
        ("qc_passed_max", int(np.max(qc_counts))),
        ("peak_rule", "scipy.signal.find_peaks on unsmoothed processed group means"),
        ("peak_prominence", "max(3 * robust difference-noise sigma, machine epsilon)"),
        ("peak_distance", "ceil(instrument FWHM / grid step); no neighboring-grid bin merge"),
        ("resolution_aware_peak_rows", int(len(registry))),
        ("identifier_policy", "internal patient/sample codes not exported"),
    ]
    pd.DataFrame(rows, columns=["item", "value"]).to_csv(
        TABLE_DIR / "analysis_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )


def write_report(
    labels: np.ndarray,
    raw_replicates: np.ndarray,
    qc_counts: np.ndarray,
    grid: np.ndarray,
    screening: object,
    three_group: object,
    registry: pd.DataFrame,
    burden: dict[str, object],
) -> None:
    report = OUT / "RESOLUTION_AWARE_REANALYSIS.md"
    with report.open("w", encoding="utf-8") as handle:
        handle.write("# 보라매병원 전향검체 resolution-aware 재분석\n\n")
        handle.write("## 분석 범위\n\n")
        handle.write(
            f"- 분석 대상: {len(labels)}명 — Control {int(np.sum(labels == 'Control'))}, "
            f"Biopsy-negative {int(np.sum(labels == 'Biopsy-negative'))}, "
            f"Prostate cancer {int(np.sum(labels == 'Prostate cancer'))}\n"
        )
        handle.write(f"- 입력 replicate: {_count_replicates(raw_replicates)}개; `_ave` 파일은 사용하지 않음\n")
        handle.write(
            f"- QC 통과 replicate 수: subject별 raw count를 공개하지 않고 aggregate만 기록 — "
            f"min/median/max = {int(np.min(qc_counts))}/{np.median(qc_counts):.0f}/{int(np.max(qc_counts))}\n"
        )
        handle.write(f"- 공통 grid: {grid[0]:.1f}-{grid[-1]:.1f} cm^-1, step {np.median(np.diff(grid)):.4f} cm^-1\n\n")
        handle.write("## Preprocessing 및 peak policy\n\n")
        handle.write("- Savitzky-Golay, convolution, moving-average smoothing: disabled\n")
        handle.write(f"- Baseline: AsLS (`lambda={BASELINE_LAMBDA:g}`, `p={BASELINE_ASYMMETRY:g}`)\n")
        handle.write("- Normalization: positive baseline-corrected signal의 area normalization\n")
        handle.write(
            f"- Working instrument resolution: FWHM {INSTRUMENT_RESOLUTION_FWHM_CM1:.1f} cm^-1; "
            "minimum peak distance는 grid step과 FWHM으로 계산\n"
        )
        handle.write("- Peak registry는 group별 후보를 원 grid 위치 그대로 기록하며 인접 bin을 임의 병합하지 않음\n\n")
        handle.write("## Subject-level model\n\n")
        handle.write(
            f"- Cancer Screening ROC-AUC: {screening.metrics['roc_auc']:.3f}; "
            f"balanced accuracy: {screening.metrics['balanced_accuracy']:.3f}\n"
        )
        handle.write(
            f"- 3-group macro OVR ROC-AUC: {three_group.metrics['macro_ovr_roc_auc']:.3f}; "
            f"balanced accuracy: {three_group.metrics['balanced_accuracy']:.3f}\n"
        )
        handle.write("- 두 성능 모두 subject-level nested 5-fold OOF의 exploratory 결과이며 외부검증 성능이 아님\n")
        handle.write("- Cancer Screening은 병원/검체 전처리 교란 가능성을 포함하므로 cross-hospital 일반화 근거로 해석하지 않음\n\n")
        handle.write("## 환자당 필요한 spectrum 수\n\n")
        handle.write(
            f"- 후보 k: {', '.join(str(v) for v in burden['tested_counts'])}; "
            f"random subset 반복: {BURDEN_REPEATS}회\n"
        )
        handle.write(f"- exploratory 권고: **{burden['recommended_spectra']}개 QC-passed spectra/patient**\n")
        handle.write(f"- 규칙: {burden['recommendation_rule']}\n")
        handle.write("- 이는 현재 cohort 내부의 안정화 분석이며 임상 운영의 고정 횟수로 확정하려면 독립 validation이 필요함\n\n")
        handle.write("## 산출물\n\n")
        handle.write("- Figures: `figures/A_preprocessing_no_smoothing.png`, `B_group_processed_spectra.png`, `C_screening_performance.png`, `D_three_group_performance.png`, `E_resolution_aware_peaks.png`, `D_spectra_availability.png`, `D_spectra_per_patient_performance.png`\n")
        handle.write("- Tables: `tables/analysis_manifest.csv`, model metrics/confusion/OOF CSV, `tables/peak_registry_resolution_aware.csv`; patient-burden outputs are `D_spectra_availability_summary.csv`, `D_spectra_per_patient_summary.csv`, and `D_spectra_per_patient_repeats.csv` in the reanalysis root\n")


def _count_replicates(raw_replicates: np.ndarray) -> int:
    return int(len(raw_replicates))


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    grid = np.asarray(np.load(MODEL_GRID), dtype=np.float64)
    subject_keys, subject_labels, raw_replicates, replicate_subjects, _files = load_subject_replicates(grid)
    raw_subject_means, processed_subject_means, labels, qc_counts, qc_replicates = build_subject_matrices(
        subject_keys,
        subject_labels,
        raw_replicates,
        replicate_subjects,
        grid,
    )
    if not np.array_equal(labels, subject_labels):
        raise RuntimeError("Subject labels changed during QC aggregation")
    if not np.isfinite(processed_subject_means).all():
        raise RuntimeError("Non-finite values found after AsLS + area normalization")

    screening_x, screening_y, _ = build_task(processed_subject_means, labels, "screening_binary")
    three_group_x, three_group_y, _ = build_task(processed_subject_means, labels, "three_group")
    screening = nested_oof(screening_x, screening_y)
    three_group = nested_oof(three_group_x, three_group_y)
    save_result("screening_binary", screening, ("Non-cancer", "Cancer"))
    save_result("three_group", three_group, tuple(LABEL_DISPLAY[label] for label in LABELS))

    registry = detect_resolution_aware_peaks(grid, processed_subject_means, labels)
    registry.to_csv(TABLE_DIR / "peak_registry_resolution_aware.csv", index=False, encoding="utf-8-sig")
    write_manifest(grid, labels, raw_replicates, qc_counts, registry)

    representative_index = int(np.argsort(qc_counts)[len(qc_counts) // 2])
    plot_preprocessing(
        grid,
        qc_replicates[representative_index],
        raw_subject_means[representative_index],
        processed_subject_means[representative_index],
        FIG_DIR / "A_preprocessing_no_smoothing.png",
    )
    plot_group_spectra(grid, processed_subject_means, labels, FIG_DIR / "B_group_processed_spectra.png")
    plot_binary_performance(screening, FIG_DIR / "C_screening_performance.png")
    plot_three_group_performance(three_group, FIG_DIR / "D_three_group_performance.png")
    plot_resolution_aware_peaks(
        grid,
        processed_subject_means,
        labels,
        registry,
        FIG_DIR / "E_resolution_aware_peaks.png",
    )

    helper = run_path(str(REPO / "notebooks" / "aecd_spectra_burden.py"))
    encoded_labels = {
        "Control": "control",
        "Biopsy-negative": "prostate disease control",
        "Prostate cancer": "prostate",
    }
    subject_label_by_key = dict(zip(subject_keys, labels, strict=True))
    metadata_rows = [
        {
            "subject_key": key,
            "cohort_group": encoded_labels[subject_label_by_key[key]],
        }
        for key in replicate_subjects
    ]
    burden = helper["run_spectra_burden_analysis"](
        aligned=raw_replicates,
        metadata=pd.DataFrame(metadata_rows),
        subject_keys=np.asarray(subject_keys),
        subject_labels=np.asarray([
            {"Control": "control", "Biopsy-negative": "prostate disease control", "Prostate cancer": "prostate"}[label]
            for label in labels
        ]),
        target_column="cohort_group",
        output_dir=OUT,
        random_state=RANDOM_STATE,
        qc_mad_threshold=QC_MAD_THRESHOLD,
        count_grid=(1, 2, 3, 4, 5),
        n_repeats=BURDEN_REPEATS,
        auc_tolerance=BURDEN_AUC_TOLERANCE,
        pca_components=10,
        logistic_c=0.01,
        label_order=("control", "prostate disease control", "prostate"),
        cancer_labels=("prostate",),
    )
    write_report(labels, raw_replicates, qc_counts, grid, screening, three_group, registry, burden)
    print({
        "output_dir": str(OUT),
        "subjects": int(len(labels)),
        "input_replicates": int(len(raw_replicates)),
        "qc_passed_min_median_max": (
            int(np.min(qc_counts)), float(np.median(qc_counts)), int(np.max(qc_counts))
        ),
        "screening_auc": screening.metrics["roc_auc"],
        "three_group_macro_auc": three_group.metrics["macro_ovr_roc_auc"],
        "peak_registry_rows": int(len(registry)),
        "recommended_spectra_per_patient": burden["recommended_spectra"],
    })


if __name__ == "__main__":
    main()
