from __future__ import annotations

import csv
from dataclasses import dataclass

import numpy as np
from severance_data import TABLE_DIR
from severance_lr_model import (
    LrEvaluationConfig,
    RepeatedCvResult,
    SeveranceDataset,
    evaluate_repeated_cv,
    load_unique_severance_dataset,
)
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True, slots=True)
class BinaryMetrics:
    auroc: float
    pr_auc: float
    accuracy: float
    sensitivity: float
    specificity: float
    precision: float
    npv: float
    f1: float


def binary_metrics(labels: np.ndarray, probability: np.ndarray) -> BinaryMetrics:
    predicted = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return BinaryMetrics(
        auroc=float(roc_auc_score(labels, probability)),
        pr_auc=float(average_precision_score(labels, probability)),
        accuracy=float(accuracy_score(labels, predicted)),
        sensitivity=float(recall_score(labels, predicted)),
        specificity=float(tn / (tn + fp)),
        precision=float(precision_score(labels, predicted)),
        npv=float(tn / (tn + fn)),
        f1=float(f1_score(labels, predicted)),
    )


def metric_intervals(
    labels: np.ndarray, probability: np.ndarray, n_bootstrap: int = 5000
) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(20260711)
    draws: dict[str, list[float]] = {field: [] for field in BinaryMetrics.__dataclass_fields__}
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(labels), len(labels))
        if len(np.unique(labels[indices])) < 2:
            continue
        sampled = binary_metrics(labels[indices], probability[indices])
        for field in draws:
            draws[field].append(getattr(sampled, field))
    return {
        field: (
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        )
        for field, values in draws.items()
    }


def write_evaluation(dataset: SeveranceDataset, result: RepeatedCvResult) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    with (TABLE_DIR / "severance_lr_repeated_cv_auc.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["repeat", "pooled_oof_auroc"])
        for repeat, auc in enumerate(result.repeat_auc, start=1):
            writer.writerow([repeat, auc])
    with (TABLE_DIR / "severance_lr_cv_folds.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["repeat", "fold", "fold_auroc", "test_subjects"])
        for fold in result.folds:
            writer.writerow([fold.repeat, fold.fold, fold.auc, ";".join(fold.test_subjects)])
    with (TABLE_DIR / "severance_lr_oof_predictions.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "subject_id",
                "source_group",
                "true_label",
                "mean_probability",
                "probability_sd",
                "pred_label",
            ]
        )
        for index, subject_id in enumerate(dataset.subject_ids):
            probability = result.mean_probability[index]
            writer.writerow(
                [
                    subject_id,
                    dataset.source_groups[index],
                    dataset.labels[index],
                    probability,
                    result.probability_sd[index],
                    int(probability >= 0.5),
                ]
            )
    metrics = binary_metrics(dataset.labels, result.mean_probability)
    intervals = metric_intervals(dataset.labels, result.mean_probability)
    with (TABLE_DIR / "severance_lr_metrics_ci.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "estimate", "ci_lower", "ci_upper"])
        for field in BinaryMetrics.__dataclass_fields__:
            writer.writerow([field, getattr(metrics, field), *intervals[field]])
        writer.writerow(["repeated_cv_mean_auroc", result.repeat_auc.mean(), "", ""])
        writer.writerow(["repeated_cv_sd_auroc", result.repeat_auc.std(), "", ""])


def write_leaderboard(baseline: RepeatedCvResult, optimized: RepeatedCvResult) -> None:
    with (TABLE_DIR / "severance_lr_full_spectrum_baseline_auc.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["repeat", "pooled_oof_auroc"])
        for repeat, auc in enumerate(baseline.repeat_auc, start=1):
            writer.writerow([repeat, auc])
    with (TABLE_DIR / "severance_lr_optimization_leaderboard.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "model",
                "cohort",
                "evaluation",
                "n_features",
                "mean_auroc",
                "sd_auroc",
                "delta_vs_full_spectrum_lr",
            ]
        )
        baseline_mean = float(baseline.repeat_auc.mean())
        writer.writerow(
            [
                "full_spectrum_lr_c1",
                "59 unique subjects",
                "100x repeated stratified 5-fold pooled OOF",
                935,
                baseline_mean,
                baseline.repeat_auc.std(),
                0.0,
            ]
        )
        writer.writerow(
            [
                "bin32_lr_c3",
                "59 unique subjects",
                "100x repeated stratified 5-fold pooled OOF",
                29,
                optimized.repeat_auc.mean(),
                optimized.repeat_auc.std(),
                optimized.repeat_auc.mean() - baseline_mean,
            ]
        )


def generate_lr_outputs() -> tuple[SeveranceDataset, RepeatedCvResult]:
    dataset = load_unique_severance_dataset()
    baseline = evaluate_repeated_cv(dataset, LrEvaluationConfig(bin_width=1, regularization_c=1.0))
    result = evaluate_repeated_cv(dataset, LrEvaluationConfig())
    write_evaluation(dataset, result)
    write_leaderboard(baseline, result)
    return dataset, result
