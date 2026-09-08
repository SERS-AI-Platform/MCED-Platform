from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from powder_comparison.runner import AnalysisResult
from powder_comparison.statistics import metric_values
from powder_comparison.threshold_analysis import classification_status, threshold_strategies


def _writer(path: Path):
    handle = path.open("w", newline="", encoding="utf-8-sig")
    return handle, csv.writer(handle)


def _write_patient_trace(result: AnalysisResult, table_dir: Path) -> None:
    screening = result.screening
    optimized = screening.binary_thresholds
    assert optimized is not None
    legacy_fixed = screening.legacy_probabilities.argmax(axis=1)
    transfer_fixed = screening.transfer_probabilities.argmax(axis=1)
    native_fixed = screening.native_probabilities.argmax(axis=1)
    handle, writer = _writer(table_dir / "lr_screening_paired_predictions.csv")
    with handle:
        writer.writerow(
            [
                "patient_id",
                "sample_id",
                "fold",
                "clinical_group",
                "psa",
                "psa_band",
                "measurement_order",
                "legacy_probability",
                "legacy_youden_threshold",
                "legacy_fixed_prediction",
                "legacy_optimized_prediction",
                "legacy_threshold_status",
                "powder_transfer_probability",
                "powder_transfer_fixed_prediction",
                "powder_transfer_optimized_prediction",
                "powder_transfer_threshold_status",
                "powder_native_probability",
                "powder_native_youden_threshold",
                "powder_native_fixed_prediction",
                "powder_native_optimized_prediction",
                "powder_native_threshold_status",
            ]
        )
        for index, pair in enumerate(result.pairs):
            truth = int(screening.y_true[index])
            psa = result.clinical.psa[index]
            writer.writerow(
                [
                    result.clinical.patient_ids[index],
                    f"{pair.powder.prefix} {pair.sample_id}",
                    screening.folds[index],
                    pair.clinical_group,
                    "" if not np.isfinite(psa) else psa,
                    result.clinical.psa_bands[index],
                    pair.powder.order_index,
                    screening.legacy_probabilities[index, 1],
                    optimized.legacy_thresholds[index],
                    legacy_fixed[index],
                    optimized.legacy_predictions[index],
                    classification_status(
                        truth, legacy_fixed[index], optimized.legacy_predictions[index]
                    ),
                    screening.transfer_probabilities[index, 1],
                    transfer_fixed[index],
                    optimized.transfer_predictions[index],
                    classification_status(
                        truth, transfer_fixed[index], optimized.transfer_predictions[index]
                    ),
                    screening.native_probabilities[index, 1],
                    optimized.native_thresholds[index],
                    native_fixed[index],
                    optimized.native_predictions[index],
                    classification_status(
                        truth, native_fixed[index], optimized.native_predictions[index]
                    ),
                ]
            )


def _write_summary(result: AnalysisResult, table_dir: Path) -> None:
    handle, writer = _writer(table_dir / "lr_threshold_optimization.csv")
    with handle:
        writer.writerow(
            [
                "strategy",
                "modality",
                "threshold_method",
                "threshold_source",
                "threshold_mean",
                "threshold_min",
                "threshold_max",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "roc_auc",
                "sensitivity",
                "specificity",
                "brier_score",
            ]
        )
        for strategy in threshold_strategies(result):
            metrics = {
                row.name: row.value
                for row in metric_values(
                    result.screening.y_true,
                    strategy.probabilities,
                    "screening_binary",
                    strategy.predictions,
                )
            }
            writer.writerow(
                [
                    strategy.name,
                    strategy.modality,
                    strategy.method,
                    strategy.source,
                    float(np.mean(strategy.thresholds)),
                    float(np.min(strategy.thresholds)),
                    float(np.max(strategy.thresholds)),
                    metrics["accuracy"],
                    metrics["balanced_accuracy"],
                    metrics["macro_f1"],
                    metrics["roc_auc"],
                    metrics["sensitivity"],
                    metrics["specificity"],
                    metrics["brier_score"],
                ]
            )

    handle, writer = _writer(table_dir / "lr_threshold_metric_comparisons.csv")
    with handle:
        writer.writerow(
            ["comparison", "metric", "fixed_0.5", "optimized", "delta", "ci95_low", "ci95_high"]
        )
        for block in result.threshold_comparisons:
            for row in block.metrics:
                writer.writerow(
                    [
                        block.name,
                        row.name,
                        row.legacy_value,
                        row.powder_value,
                        row.delta,
                        row.ci_low,
                        row.ci_high,
                    ]
                )


def _write_fold_thresholds(result: AnalysisResult, table_dir: Path) -> None:
    optimized = result.screening.binary_thresholds
    assert optimized is not None
    handle, writer = _writer(table_dir / "lr_thresholds_by_fold.csv")
    with handle:
        writer.writerow(["fold", "liquid_training_youden", "powder_training_youden"])
        for fold in np.unique(result.screening.folds):
            selected = result.screening.folds == fold
            writer.writerow(
                [
                    fold,
                    np.unique(optimized.legacy_thresholds[selected])[0],
                    np.unique(optimized.native_thresholds[selected])[0],
                ]
            )


def _write_error_breakdowns(result: AnalysisResult, table_dir: Path) -> None:
    truth = result.screening.y_true
    groups = result.spectra.clinical_groups
    handle, writer = _writer(table_dir / "lr_false_positive_breakdown.csv")
    with handle:
        writer.writerow(
            [
                "strategy",
                "clinical_group",
                "group_n",
                "false_positive_n",
                "false_positive_rate",
                "share_of_all_false_positives",
            ]
        )
        for strategy in threshold_strategies(result):
            all_false_positive = (truth == 0) & (strategy.predictions == 1)
            total = int(np.sum(all_false_positive))
            for group in ("Control", "Biopsy-negative"):
                selected = groups == group
                count = int(np.sum(all_false_positive & selected))
                writer.writerow(
                    [
                        strategy.name,
                        group,
                        int(np.sum(selected)),
                        count,
                        count / int(np.sum(selected)),
                        0.0 if total == 0 else count / total,
                    ]
                )

    handle, writer = _writer(table_dir / "lr_biopsy_negative_psa_errors.csv")
    with handle:
        writer.writerow(
            [
                "strategy",
                "psa_band",
                "n",
                "predicted_cancer_n",
                "predicted_cancer_rate",
                "mean_cancer_probability",
            ]
        )
        for strategy in threshold_strategies(result):
            for band in ("<4", "4-<10", ">=10"):
                selected = (groups == "Biopsy-negative") & (result.clinical.psa_bands == band)
                count = int(np.sum(selected))
                writer.writerow(
                    [
                        strategy.name,
                        band,
                        count,
                        int(np.sum(strategy.predictions[selected] == 1)),
                        float(np.mean(strategy.predictions[selected] == 1)),
                        float(np.mean(strategy.probabilities[selected, 1])),
                    ]
                )


def write_threshold_tables(result: AnalysisResult, table_dir: Path) -> None:
    _write_patient_trace(result, table_dir)
    _write_summary(result, table_dir)
    _write_fold_thresholds(result, table_dir)
    _write_error_breakdowns(result, table_dir)
