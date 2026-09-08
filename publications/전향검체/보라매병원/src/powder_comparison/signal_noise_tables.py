from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .signal_noise_ablation import AblationResult
from .signal_noise_trace_tables import write_signal_noise_trace_tables
from .signal_noise_types import SignalNoiseDiagnostics


def _writer(path: Path):
    handle = path.open("w", newline="", encoding="utf-8-sig")
    return handle, csv.writer(handle)


def _write_repeatability(diagnostics: SignalNoiseDiagnostics, directory: Path) -> None:
    handle, writer = _writer(directory / "powder_peak_repeatability_by_patient_stage.csv")
    with handle:
        writer.writerow(
            [
                "sample_id",
                "modality",
                "preprocessing_stage",
                "median_peak_count_per_replicate",
                "candidate_peak_clusters",
                "consensus_peaks_4_of_5",
                "consensus_fraction",
                "median_residual_noise_sigma",
            ]
        )
        for row in diagnostics.repeatability.stage_rows:
            writer.writerow(
                [
                    row.sample_id,
                    row.modality,
                    row.stage,
                    row.median_peak_count,
                    row.candidate_peak_count,
                    row.consensus_peak_count,
                    row.consensus_fraction,
                    row.median_noise_sigma,
                ]
            )


def _write_landscape(
    grid: np.ndarray,
    diagnostics: SignalNoiseDiagnostics,
    directory: Path,
) -> None:
    handle, writer = _writer(directory / "powder_peak_signal_landscape.csv")
    landscape = diagnostics.landscape
    with handle:
        writer.writerow(
            [
                "raman_shift_cm_1",
                "legacy_consensus_subject_rate",
                "powder_consensus_subject_rate",
                "legacy_max_abs_hedges_g",
                "powder_max_abs_hedges_g",
                "peak_category",
            ]
        )
        for values in zip(
            grid,
            landscape.legacy_presence_rate,
            landscape.powder_presence_rate,
            landscape.legacy_max_abs_effect,
            landscape.powder_max_abs_effect,
            landscape.categories,
            strict=True,
        ):
            writer.writerow(values)


def _write_effect_retention(diagnostics: SignalNoiseDiagnostics, directory: Path) -> None:
    handle, writer = _writer(directory / "powder_subgroup_effect_retention_summary.csv")
    with handle:
        writer.writerow(
            [
                "group_pair",
                "relevant_features_abs_liquid_g_ge_0_10",
                "spearman_rho",
                "effect_direction_agreement",
                "median_absolute_effect_retention",
            ]
        )
        for row in diagnostics.effect_summaries:
            writer.writerow(
                [
                    row.group_pair,
                    row.relevant_feature_count,
                    row.spearman_rho,
                    row.sign_agreement,
                    row.median_absolute_retention,
                ]
            )
    handle, writer = _writer(directory / "powder_legacy_lr_peak_signal_retention.csv")
    with handle:
        writer.writerow(
            [
                "peak",
                "raman_shift_cm_1",
                "legacy_lr_relative_importance",
                "legacy_cancer_control_hedges_g",
                "powder_cancer_control_hedges_g",
                "absolute_effect_retention",
                "direction_agreement",
                "legacy_consensus_subject_rate",
                "powder_consensus_subject_rate",
                "peak_category",
            ]
        )
        for row in diagnostics.important_peaks:
            writer.writerow(
                [
                    row.peak_name,
                    row.shift_cm_1,
                    row.legacy_importance,
                    row.legacy_effect,
                    row.powder_effect,
                    row.absolute_effect_retention,
                    row.direction_agreement,
                    row.legacy_presence_rate,
                    row.powder_presence_rate,
                    row.category,
                ]
            )


def _write_lot_variance(
    grid: np.ndarray,
    diagnostics: SignalNoiseDiagnostics,
    directory: Path,
) -> None:
    variance = diagnostics.lot_variance
    handle, writer = _writer(directory / "powder_lot_variance_components.csv")
    with handle:
        writer.writerow(
            [
                "raman_shift_cm_1",
                "patient_variance",
                "lot_variance",
                "spot_residual_variance",
                "patient_fraction",
                "lot_fraction",
                "spot_residual_fraction",
                "technical_fraction",
            ]
        )
        for values in zip(
            grid,
            variance.patient,
            variance.lot,
            variance.residual,
            variance.patient_fraction,
            variance.lot_fraction,
            variance.residual_fraction,
            variance.lot_fraction + variance.residual_fraction,
            strict=True,
        ):
            writer.writerow(values)


def _write_ablation_rows(
    rows: tuple[AblationResult, ...], writer: csv.writer, confusion_writer: csv.writer
) -> None:
    for row in rows:
        writer.writerow(
            [
                row.task,
                row.feature_set,
                float(np.median(row.feature_counts)),
                row.metrics.roc_auc,
                row.metrics.balanced_accuracy,
                row.metrics.sensitivity,
                row.metrics.specificity,
            ]
        )
        for true_index, true_name in enumerate(row.class_names):
            for predicted_index, predicted_name in enumerate(row.class_names):
                confusion_writer.writerow(
                    [
                        row.task,
                        row.feature_set,
                        true_name,
                        predicted_name,
                        row.confusion[true_index, predicted_index],
                    ]
                )


def _write_ablation(diagnostics: SignalNoiseDiagnostics, directory: Path) -> None:
    metric_handle, metric_writer = _writer(directory / "powder_peak_ablation_metrics.csv")
    confusion_handle, confusion_writer = _writer(
        directory / "powder_peak_ablation_confusion_matrices.csv"
    )
    with metric_handle, confusion_handle:
        metric_writer.writerow(
            [
                "task",
                "feature_set",
                "median_features_across_folds",
                "oof_roc_auc",
                "oof_balanced_accuracy",
                "sensitivity",
                "specificity",
            ]
        )
        confusion_writer.writerow(["task", "feature_set", "true_label", "predicted_label", "count"])
        _write_ablation_rows(diagnostics.screening_ablation, metric_writer, confusion_writer)
        _write_ablation_rows(diagnostics.three_group_ablation, metric_writer, confusion_writer)


def write_signal_noise_tables(
    grid: np.ndarray,
    diagnostics: SignalNoiseDiagnostics,
    directory: Path,
) -> None:
    _write_repeatability(diagnostics, directory)
    _write_landscape(grid, diagnostics, directory)
    _write_effect_retention(diagnostics, directory)
    _write_lot_variance(grid, diagnostics, directory)
    _write_ablation(diagnostics, directory)
    write_signal_noise_trace_tables(diagnostics, directory)
