from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = ["DejaVu Sans"]
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


RAW_CONTROL = "control"
RAW_PDC = "prostate disease control"
RAW_CANCER = "prostate"
CLASS_TO_INDEX = {
    RAW_CONTROL: 0,
    RAW_PDC: 1,
    RAW_CANCER: 2,
}
DEFAULT_COUNT_GRID = (1, 2, 3, 5, 10, 20, 40, 80, 120)
DEFAULT_REPEATS = 3
DEFAULT_AUC_TOLERANCE = 0.02
DEFAULT_PCA_COMPONENTS = 10
DEFAULT_LOGISTIC_C = 0.01


def _qc_keep_mask(replicates: np.ndarray, mad_threshold: float) -> np.ndarray:
    values = np.asarray(replicates, dtype=np.float64)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("replicates must be a non-empty 2D array")
    if len(values) < 2:
        return np.ones(len(values), dtype=bool)

    median = np.median(values, axis=0)
    residual = values - median
    scale = np.maximum(
        np.median(np.abs(residual), axis=0),
        np.finfo(float).eps,
    )
    distance = np.sqrt(np.mean((residual / scale) ** 2, axis=1))
    distance_median = float(np.median(distance))
    distance_mad = float(
        np.median(np.abs(distance - distance_median))
    )
    if distance_mad == 0.0:
        limit = distance_median + 3.0 * float(np.std(distance))
    else:
        limit = distance_median + mad_threshold * 1.4826 * distance_mad
    keep = distance <= max(limit, distance_median)
    if int(keep.sum()) < 2:
        keep = np.zeros(len(values), dtype=bool)
        keep[np.argsort(distance)[:2]] = True
    return keep


def _asls_baseline(
    values: np.ndarray,
    lam: float = 1e5,
    p: float = 0.01,
    niter: int = 10,
) -> np.ndarray:
    length = len(values)
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
        weighted = sparse.spdiags(
            weights,
            0,
            length,
            length,
        ).tocsc()
        baseline = spsolve(
            (weighted + penalty).tocsc(),
            weights * values,
        )
        weights = p * (values > baseline) + (1 - p) * (values < baseline)
    return baseline


def _preprocess(
    spectra: np.ndarray,
    lam: float = 1e5,
    p: float = 0.01,
) -> np.ndarray:
    values = np.asarray(spectra, dtype=np.float64)
    processed = np.empty_like(values, dtype=np.float64)
    for index, row in enumerate(values):
        corrected = row - _asls_baseline(row, lam=lam, p=p)
        corrected = np.clip(corrected, 0.0, None)
        area = np.trapezoid(corrected)
        processed[index] = corrected / area if area > 0 else corrected
    return processed


def _patient_replicates(
    aligned: np.ndarray,
    metadata: pd.DataFrame,
    subject_keys: np.ndarray,
    subject_labels: np.ndarray,
    target_column: str,
    mad_threshold: float,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    values = np.asarray(aligned, dtype=np.float64)
    subject_column = metadata["subject_key"].to_numpy()
    labels = np.asarray(subject_labels).astype(str)
    keys = np.asarray(subject_keys)
    if len(keys) != len(labels):
        raise ValueError("subject_keys and subject_labels must have the same length")

    patient_values: list[np.ndarray] = []
    patient_labels: list[str] = []
    qc_counts: list[int] = []
    for key, label in zip(keys, labels, strict=True):
        mask = subject_column == key
        if not np.any(mask):
            continue
        if target_column in metadata:
            observed_labels = metadata.loc[mask, target_column].dropna().astype(str).unique()
            if len(observed_labels) != 1 or observed_labels[0] != label:
                raise ValueError(
                    "Subject label differs between the subject-level matrix and metadata"
                )
        replicates = values[mask]
        keep = _qc_keep_mask(replicates, mad_threshold=mad_threshold)
        patient_values.append(replicates[keep])
        patient_labels.append(label)
        qc_counts.append(int(keep.sum()))

    if not patient_values:
        raise ValueError("No subject-level replicate groups were available")
    return (
        patient_values,
        np.asarray(patient_labels, dtype=str),
        np.asarray(qc_counts, dtype=int),
    )


def _fixed_cv_proba(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    n_classes: int,
    random_state: int,
    pca_components: int,
    logistic_c: float,
) -> np.ndarray:
    labels_array = np.asarray(labels, dtype=int)
    class_counts = np.bincount(labels_array, minlength=n_classes)
    nonzero_counts = class_counts[class_counts > 0]
    n_splits = min(5, int(nonzero_counts.min()))
    if n_splits < 2:
        raise ValueError("At least two subjects per class are required for CV")

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    out_of_fold = np.full(
        (len(features), n_classes),
        np.nan,
        dtype=np.float64,
    )
    for train_indices, test_indices in splitter.split(
        features,
        labels_array,
        groups=groups,
    ):
        effective_components = min(
            pca_components,
            len(train_indices) - 1,
            features.shape[1],
        )
        model = make_pipeline(
            StandardScaler(),
            PCA(
                n_components=effective_components,
                random_state=random_state,
            ),
            LogisticRegression(
                C=logistic_c,
                class_weight="balanced",
                max_iter=5000,
                random_state=random_state,
            ),
        )
        model.fit(features[train_indices], labels_array[train_indices])
        probabilities = model.predict_proba(features[test_indices])
        seen_classes = np.asarray(model[-1].classes_, dtype=int)
        full_probabilities = np.zeros(
            (len(test_indices), n_classes),
            dtype=np.float64,
        )
        for probability_index, class_index in enumerate(seen_classes):
            full_probabilities[:, class_index] = probabilities[:, probability_index]
        out_of_fold[test_indices] = full_probabilities
    return out_of_fold


def _binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    score = probabilities[:, 1]
    prediction = (score >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(labels, score)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
    }


def _multiclass_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    aucs = [
        roc_auc_score((labels == class_index).astype(int), probabilities[:, class_index])
        for class_index in range(probabilities.shape[1])
    ]
    return {
        "macro_auc": float(np.mean(aucs)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, probabilities.argmax(axis=1))
        ),
    }


def _candidate_counts(
    qc_counts: np.ndarray,
    count_grid: tuple[int, ...],
) -> list[int]:
    minimum_qc_count = int(np.min(qc_counts))
    counts = sorted({int(k) for k in count_grid if 1 <= int(k) <= minimum_qc_count})
    if minimum_qc_count not in counts:
        counts.append(minimum_qc_count)
    if not counts:
        raise ValueError("No requested spectrum count is available after QC")
    return counts


def _summary_table(repeat_results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for requested_count, group in repeat_results.groupby(
        "requested_spectra",
        sort=True,
    ):
        row: dict[str, float | int] = {
            "requested_spectra": int(requested_count),
            "repeats": int(len(group)),
        }
        for metric in (
            "screening_auc",
            "screening_balanced_accuracy",
            "three_class_macro_auc",
            "three_class_balanced_accuracy",
        ):
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_p025"] = float(np.percentile(values, 2.5))
            row[f"{metric}_p975"] = float(np.percentile(values, 97.5))
        rows.append(row)
    return pd.DataFrame(rows)


def _mark_saturation(
    summary: pd.DataFrame,
    tolerance: float,
) -> tuple[pd.DataFrame, int, str]:
    result = summary.copy()
    screening_max = float(result["screening_auc_mean"].max())
    three_class_max = float(result["three_class_macro_auc_mean"].max())
    result["within_auc_tolerance"] = (
        (result["screening_auc_mean"] >= screening_max - tolerance)
        & (result["three_class_macro_auc_mean"] >= three_class_max - tolerance)
    )
    eligible = result.loc[result["within_auc_tolerance"]]
    if len(eligible):
        recommended = int(eligible["requested_spectra"].min())
        rule = (
            "smallest k whose mean screening AUC and 3-class macro AUC are "
            f"within {tolerance:.2f} of their respective observed maxima"
        )
    else:
        gap = (
            (screening_max - result["screening_auc_mean"])
            + (three_class_max - result["three_class_macro_auc_mean"])
        )
        recommended = int(
            result.loc[gap.idxmin(), "requested_spectra"]
        )
        rule = (
            "no count met the tolerance on both curves; selected the smallest "
            "combined AUC gap as a fallback"
        )
    return result, recommended, rule


def _plot_availability(
    qc_counts: np.ndarray,
    output_path: Path,
) -> None:
    maximum = int(np.max(qc_counts))
    bins = np.arange(0.5, maximum + 1.5, 1.0)
    figure, axis = plt.subplots(figsize=(8.5, 5.0))
    axis.hist(
        qc_counts,
        bins=bins,
        color="#f0a202",
        edgecolor="#9a6700",
        alpha=0.65,
        label="QC-passed spectra per subject",
    )
    axis.set_title(
        "QC-passed spectra available per subject",
        loc="left",
        fontweight="bold",
    )
    axis.set_xlabel("Number of spectra")
    axis.set_ylabel("Number of subjects")
    axis.grid(axis="y", alpha=0.2)
    axis.set_xlim(max(0, int(np.min(qc_counts)) - 3), maximum + 2)
    axis.legend(frameon=False)
    axis.text(
        0.99,
        0.97,
        (
            f"n={len(qc_counts)} subjects\n"
            f"min / median / max = "
            f"{int(np.min(qc_counts))} / "
            f"{np.median(qc_counts):.0f} / "
            f"{int(np.max(qc_counts))}"
        ),
        transform=axis.transAxes,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def _plot_performance(
    summary: pd.DataFrame,
    recommended_count: int,
    n_repeats: int,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), sharex=True)
    metric_specs = (
        (
            axes[0],
            "AUC",
            (
                ("screening_auc", "Screening AUC", "#2c7fb8"),
                ("three_class_macro_auc", "3-class macro AUC", "#f0a202"),
            ),
        ),
        (
            axes[1],
            "Balanced accuracy",
            (
                (
                    "screening_balanced_accuracy",
                    "Screening balanced accuracy",
                    "#2c7fb8",
                ),
                (
                    "three_class_balanced_accuracy",
                    "3-class balanced accuracy",
                    "#f0a202",
                ),
            ),
        ),
    )
    x_values = summary["requested_spectra"].to_numpy(dtype=float)
    for axis, ylabel, metrics in metric_specs:
        for metric, label, color in metrics:
            mean = summary[f"{metric}_mean"].to_numpy(dtype=float)
            lower = summary[f"{metric}_p025"].to_numpy(dtype=float)
            upper = summary[f"{metric}_p975"].to_numpy(dtype=float)
            axis.plot(
                x_values,
                mean,
                marker="o",
                linewidth=2.0,
                color=color,
                label=label,
            )
            axis.fill_between(
                x_values,
                lower,
                upper,
                color=color,
                alpha=0.14,
                linewidth=0,
            )
        axis.axvline(
            recommended_count,
            color="0.35",
            linestyle="--",
            linewidth=1.1,
            label=(
                f"Exploratory recommendation: {recommended_count} spectra"
                if axis is axes[0]
                else None
            ),
        )
        axis.set_xscale("symlog", linthresh=1)
        axis.set_xticks(x_values)
        axis.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        axis.set_xlabel("QC-passed spectra averaged per subject (k)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].axhline(0.5, color="0.45", linestyle=":", linewidth=1.0, label="Binary chance")
    axes[0].axhline(1.0 / 3.0, color="0.65", linestyle=":", linewidth=1.0, label="3-class chance")
    axes[1].axhline(0.5, color="0.45", linestyle=":", linewidth=1.0)
    axes[1].axhline(1.0 / 3.0, color="0.65", linestyle=":", linewidth=1.0)
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    axes[1].legend(frameon=False, fontsize=8, loc="best")
    figure.suptitle(
        "Discrimination versus the number of spectra averaged per subject\n"
        f"Mean over {n_repeats} random QC-passed subsets; shaded area = 2.5th-97.5th percentile",
        x=0.02,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    figure.savefig(output_path, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def run_spectra_burden_analysis(
    *,
    aligned: np.ndarray,
    metadata: pd.DataFrame,
    subject_keys: np.ndarray,
    subject_labels: np.ndarray,
    target_column: str,
    output_dir: str | os.PathLike[str],
    random_state: int,
    qc_mad_threshold: float,
    count_grid: tuple[int, ...] = DEFAULT_COUNT_GRID,
    n_repeats: int = DEFAULT_REPEATS,
    auc_tolerance: float = DEFAULT_AUC_TOLERANCE,
    pca_components: int = DEFAULT_PCA_COMPONENTS,
    logistic_c: float = DEFAULT_LOGISTIC_C,
    label_order: tuple[str, ...] | None = None,
    cancer_labels: tuple[str, ...] | None = None,
) -> dict[str, object]:
    if n_repeats < 2:
        raise ValueError("n_repeats must be at least 2")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    patient_values, labels, qc_counts = _patient_replicates(
        aligned=aligned,
        metadata=metadata,
        subject_keys=subject_keys,
        subject_labels=subject_labels,
        target_column=target_column,
        mad_threshold=qc_mad_threshold,
    )
    if label_order is None:
        if set(labels).issubset(CLASS_TO_INDEX):
            label_order = tuple(CLASS_TO_INDEX)
        else:
            label_order = tuple(dict.fromkeys(labels.tolist()))
    if set(labels) != set(label_order):
        raise ValueError("label_order must contain each observed subject label exactly once")
    if len(label_order) < 2:
        raise ValueError("At least two subject labels are required")
    class_to_index = {label: index for index, label in enumerate(label_order)}
    if cancer_labels is None:
        cancer_labels = (label_order[-1],)
    if not set(cancer_labels).issubset(set(label_order)):
        raise ValueError("cancer_labels must be included in label_order")
    counts = _candidate_counts(qc_counts, count_grid)
    three_class_labels = np.asarray([class_to_index[label] for label in labels], dtype=int)
    screening_labels = np.asarray([label in cancer_labels for label in labels], dtype=int)
    groups = np.arange(len(labels), dtype=int)

    repeat_rows: list[dict[str, float | int]] = []
    for repeat in range(n_repeats):
        rng = np.random.default_rng(random_state + repeat)
        for requested_count in counts:
            subject_means = np.vstack([
                replicates[
                    rng.choice(
                        len(replicates),
                        size=requested_count,
                        replace=False,
                    )
                ].mean(axis=0)
                for replicates in patient_values
            ])
            processed = _preprocess(subject_means)
            screening_probabilities = _fixed_cv_proba(
                processed,
                screening_labels,
                groups,
                n_classes=2,
                random_state=random_state,
                pca_components=pca_components,
                logistic_c=logistic_c,
            )
            three_class_probabilities = _fixed_cv_proba(
                processed,
                three_class_labels,
                groups,
                n_classes=3,
                random_state=random_state,
                pca_components=pca_components,
                logistic_c=logistic_c,
            )
            screening_metrics = _binary_metrics(screening_labels, screening_probabilities)
            three_class_metrics = _multiclass_metrics(
                three_class_labels,
                three_class_probabilities,
            )
            repeat_rows.append({
                "repeat": repeat + 1,
                "requested_spectra": requested_count,
                "screening_auc": screening_metrics["auc"],
                "screening_balanced_accuracy": screening_metrics["balanced_accuracy"],
                "three_class_macro_auc": three_class_metrics["macro_auc"],
                "three_class_balanced_accuracy": three_class_metrics["balanced_accuracy"],
            })

    repeat_results = pd.DataFrame(repeat_rows)
    summary = _summary_table(repeat_results)
    summary, recommended_count, recommendation_rule = _mark_saturation(
        summary,
        tolerance=auc_tolerance,
    )
    availability_summary = pd.DataFrame([{
        "subjects": int(len(qc_counts)),
        "qc_passed_spectra_min": int(np.min(qc_counts)),
        "qc_passed_spectra_median": float(np.median(qc_counts)),
        "qc_passed_spectra_max": int(np.max(qc_counts)),
    }])

    summary.to_csv(
        output_path / "D_spectra_per_patient_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    repeat_results.to_csv(
        output_path / "D_spectra_per_patient_repeats.csv",
        index=False,
        encoding="utf-8-sig",
    )
    availability_summary.to_csv(
        output_path / "D_spectra_availability_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _plot_availability(
        qc_counts,
        output_path / "D_spectra_availability.png",
    )
    _plot_performance(
        summary,
        recommended_count,
        n_repeats,
        output_path / "D_spectra_per_patient_performance.png",
    )

    print({
        "subjects": int(len(labels)),
        "qc_passed_spectra_per_subject": {
            "min": int(np.min(qc_counts)),
            "median": float(np.median(qc_counts)),
            "max": int(np.max(qc_counts)),
        },
        "tested_counts": counts,
        "repeats": n_repeats,
        "exploratory_recommended_spectra": recommended_count,
        "recommendation_rule": recommendation_rule,
    })
    display_table = summary[
        [
            "requested_spectra",
            "repeats",
            "screening_auc_mean",
            "screening_auc_p025",
            "screening_auc_p975",
            "three_class_macro_auc_mean",
            "three_class_macro_auc_p025",
            "three_class_macro_auc_p975",
            "within_auc_tolerance",
        ]
    ].copy()
    print(display_table.to_string(index=False))
    return {
        "summary": summary,
        "repeat_results": repeat_results,
        "availability_summary": availability_summary,
        "recommended_spectra": recommended_count,
        "recommendation_rule": recommendation_rule,
        "tested_counts": counts,
        "output_dir": str(output_path.resolve()),
    }
