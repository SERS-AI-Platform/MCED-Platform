from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from powder_comparison.group_separation_tables import write_group_separation_tables
from powder_comparison.order_tables import write_order_tables
from powder_comparison.peak_tables import write_peak_tables
from powder_comparison.preprocessing_tables import write_preprocessing_tables
from powder_comparison.runner import AnalysisResult
from powder_comparison.signal_noise_tables import write_signal_noise_tables
from powder_comparison.threshold_tables import write_threshold_tables


def _writer(path: Path):
    handle = path.open("w", newline="", encoding="utf-8-sig")
    return handle, csv.writer(handle)


def _write_manifest(result: AnalysisResult, table_dir: Path) -> None:
    handle, writer = _writer(table_dir / "cohort_manifest.csv")
    with handle:
        writer.writerow(
            [
                "patient_id",
                "sample_id",
                "clinical_group",
                "file_prefix",
                "powder_measurement_order",
                "legacy_replicates",
                "powder_replicates",
            ]
        )
        for index, pair in enumerate(result.pairs):
            writer.writerow(
                [
                    result.clinical.patient_ids[index],
                    f"{pair.powder.prefix} {pair.sample_id}",
                    pair.clinical_group,
                    pair.powder.prefix,
                    pair.powder.order_index,
                    len(pair.legacy.replicates),
                    len(pair.powder.replicates),
                ]
            )


def _write_lr_screening_predictions(result: AnalysisResult, table_dir: Path) -> None:
    handle, writer = _writer(table_dir / "lr_screening_paired_predictions.csv")
    with handle:
        writer.writerow(
            [
                "patient_id",
                "sample_id",
                "clinical_group",
                "measurement_order",
                "legacy_probability",
                "powder_transfer_probability",
                "legacy_prediction",
                "powder_transfer_prediction",
                "reclassification",
            ]
        )
        truth = result.screening.y_true
        legacy_prediction = result.screening.legacy_probabilities.argmax(axis=1)
        powder_prediction = result.screening.transfer_probabilities.argmax(axis=1)
        legacy_correct = legacy_prediction == truth
        powder_correct = powder_prediction == truth
        for index, pair in enumerate(result.pairs):
            if legacy_correct[index] and powder_correct[index]:
                status = "stable_correct"
            elif not legacy_correct[index] and powder_correct[index]:
                status = "improved"
            elif legacy_correct[index] and not powder_correct[index]:
                status = "worsened"
            else:
                status = "stable_incorrect"
            writer.writerow(
                [
                    result.clinical.patient_ids[index],
                    f"{pair.powder.prefix} {pair.sample_id}",
                    pair.clinical_group,
                    pair.powder.order_index,
                    result.screening.legacy_probabilities[index, 1],
                    result.screening.transfer_probabilities[index, 1],
                    legacy_prediction[index],
                    powder_prediction[index],
                    status,
                ]
            )


def _write_crossfit_predictions(result: AnalysisResult, table_dir: Path) -> None:
    for crossfit in (result.screening, result.three_group):
        handle, writer = _writer(table_dir / f"{crossfit.task}_paired_oof_predictions.csv")
        with handle:
            writer.writerow(
                [
                    "patient_id",
                    "fold",
                    "true_label",
                    "legacy_prediction",
                    "powder_transfer_prediction",
                    "powder_native_prediction",
                    *[f"legacy_prob_{name}" for name in crossfit.class_names],
                    *[f"powder_transfer_prob_{name}" for name in crossfit.class_names],
                    *[f"powder_native_prob_{name}" for name in crossfit.class_names],
                ]
            )
            for index in range(len(crossfit.y_true)):
                writer.writerow(
                    [
                        result.clinical.patient_ids[index],
                        crossfit.folds[index],
                        crossfit.class_names[crossfit.y_true[index]],
                        crossfit.class_names[crossfit.legacy_probabilities[index].argmax()],
                        crossfit.class_names[crossfit.transfer_probabilities[index].argmax()],
                        crossfit.class_names[crossfit.native_probabilities[index].argmax()],
                        *crossfit.legacy_probabilities[index],
                        *crossfit.transfer_probabilities[index],
                        *crossfit.native_probabilities[index],
                    ]
                )


def _write_comparisons(result: AnalysisResult, table_dir: Path) -> None:
    handle, writer = _writer(table_dir / "metric_comparisons.csv")
    with handle:
        writer.writerow(
            ["comparison", "metric", "legacy", "powder", "delta", "ci95_low", "ci95_high"]
        )
        for block in result.comparisons:
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
    handle, writer = _writer(table_dir / "agreement_and_reclassification.csv")
    with handle:
        writer.writerow(
            [
                "comparison",
                "exact_agreement",
                "cohen_kappa",
                "kappa_ci95_low",
                "kappa_ci95_high",
                "stable_correct",
                "improved",
                "worsened",
                "stable_incorrect",
                "net_improvement_rate",
            ]
        )
        for block in result.comparisons:
            writer.writerow(
                [
                    block.name,
                    block.agreement.exact_agreement,
                    block.agreement.kappa,
                    block.agreement.ci_low,
                    block.agreement.ci_high,
                    block.reclassification.stable_correct,
                    block.reclassification.improved,
                    block.reclassification.worsened,
                    block.reclassification.stable_incorrect,
                    block.reclassification.net_improvement_rate,
                ]
            )


def _write_confusion_matrices(result: AnalysisResult, table_dir: Path) -> None:
    for crossfit in (result.screening, result.three_group):
        path = table_dir / f"{crossfit.task}_lr_confusion_matrices.csv"
        handle, writer = _writer(path)
        with handle:
            writer.writerow(["modality", "true_label", *crossfit.class_names])
            optimized = crossfit.binary_thresholds
            modalities = (
                (
                    ("legacy_liquid_fixed_0.5", crossfit.legacy_probabilities.argmax(axis=1)),
                    ("legacy_liquid_optimized", optimized.legacy_predictions),
                    ("powder_transfer_fixed_0.5", crossfit.transfer_probabilities.argmax(axis=1)),
                    ("powder_transfer_optimized", optimized.transfer_predictions),
                    ("powder_native_fixed_0.5", crossfit.native_probabilities.argmax(axis=1)),
                    ("powder_native_optimized", optimized.native_predictions),
                )
                if optimized is not None
                else (
                    ("legacy_liquid", crossfit.legacy_probabilities.argmax(axis=1)),
                    ("powder_transfer", crossfit.transfer_probabilities.argmax(axis=1)),
                    ("powder_native", crossfit.native_probabilities.argmax(axis=1)),
                )
            )
            for modality, prediction in modalities:
                for class_index, class_name in enumerate(crossfit.class_names):
                    selected = crossfit.y_true == class_index
                    counts = [
                        int(np.sum(prediction[selected] == index))
                        for index in range(len(crossfit.class_names))
                    ]
                    writer.writerow([modality, class_name, *counts])


def write_tables(result: AnalysisResult) -> Path:
    table_dir = result.sources.output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    (table_dir / "production_paired_predictions.csv").unlink(missing_ok=True)
    _write_manifest(result, table_dir)
    write_threshold_tables(result, table_dir)
    _write_crossfit_predictions(result, table_dir)
    _write_comparisons(result, table_dir)
    _write_confusion_matrices(result, table_dir)
    write_order_tables(result, table_dir)
    write_peak_tables(result, table_dir)
    write_preprocessing_tables(result, table_dir)
    write_group_separation_tables(result, table_dir)
    write_signal_noise_tables(result.spectra.grid, result.signal_noise, table_dir)
    return table_dir
