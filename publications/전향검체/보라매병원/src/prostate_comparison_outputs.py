from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from boramae_data import TABLE_DIR
from prostate_comparison_model import OofResult
from prostate_shift_alignment import PeakMatch, PeakSummary


@dataclass(frozen=True, slots=True)
class ResultOutput:
    task: str
    result: OofResult
    class_names: list[str]


@dataclass(frozen=True, slots=True)
class AlignmentOutput:
    summaries: tuple[PeakSummary, PeakSummary]
    matches: tuple[tuple[PeakMatch, ...], tuple[PeakMatch, ...]]
    shifts: tuple[np.ndarray, np.ndarray]


class ReportDriftError(RuntimeError):
    def __init__(self, missing: list[str]) -> None:
        super().__init__(f"Publication report is stale; missing generated values: {missing}")


def write_result(data: ResultOutput, table_dir: Path = TABLE_DIR) -> None:
    with (table_dir / f"{data.task}_metrics.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value", "ci95_low", "ci95_high", "n"])
        for name, value in data.result.metrics.items():
            low, high = data.result.intervals[name]
            writer.writerow([name, value, low, high, len(data.result.y_true)])
    with (table_dir / f"{data.task}_confusion_matrix.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_label", *[f"pred_{name}" for name in data.class_names]])
        for name, row in zip(data.class_names, data.result.confusion, strict=True):
            writer.writerow([name, *row.tolist()])
    with (table_dir / f"{data.task}_oof_predictions.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "case_id",
                "fold",
                "true_label",
                "pred_label",
                *[f"prob_{name}" for name in data.class_names],
            ]
        )
        prefix = data.task.upper().replace("_", "-")
        for index in range(len(data.result.y_true)):
            writer.writerow(
                [
                    f"{prefix}-{index + 1:03d}",
                    data.result.folds[index],
                    data.class_names[data.result.y_true[index]],
                    data.class_names[data.result.y_pred[index]],
                    *data.result.probabilities[index],
                ]
            )


def write_combined_metrics(tasks: list[str], table_dir: Path = TABLE_DIR) -> None:
    with (table_dir / "prostate_classification_metrics.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as output:
        writer = csv.writer(output)
        writer.writerow(["task", "metric", "value", "ci95_low", "ci95_high", "n"])
        for task in tasks:
            with (table_dir / f"{task}_metrics.csv").open(encoding="utf-8-sig") as source:
                for row in csv.DictReader(source):
                    writer.writerow(
                        [
                            task,
                            row["metric"],
                            row["value"],
                            row["ci95_low"],
                            row["ci95_high"],
                            row["n"],
                        ]
                    )


def write_alignment(data: AlignmentOutput, table_dir: Path = TABLE_DIR) -> None:
    with (table_dir / "prostate_peak_alignment_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "stage",
                "clean_peaks",
                "prospective_peaks",
                "matched_peaks",
                "matched_within_5cm-1",
                "clean_unmatched",
                "prospective_unmatched",
                "median_signed_shift_cm-1",
                "median_absolute_shift_cm-1",
                "max_absolute_shift_cm-1",
            ]
        )
        for row in data.summaries:
            writer.writerow(
                [
                    row.stage,
                    row.clean_peaks,
                    row.prospective_peaks,
                    row.matched_peaks,
                    row.matched_within_5,
                    row.clean_unmatched,
                    row.prospective_unmatched,
                    row.median_signed_shift,
                    row.median_absolute_shift,
                    row.max_absolute_shift,
                ]
            )
    with (table_dir / "prostate_peak_alignment_matches.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "stage",
                "clean_peak_cm-1",
                "prospective_peak_cm-1",
                "prospective_minus_clean_cm-1",
                "within_5cm-1",
            ]
        )
        for stage_matches in data.matches:
            for row in stage_matches:
                writer.writerow(
                    [
                        row.stage,
                        row.clean_peak,
                        row.prospective_peak,
                        row.delta,
                        abs(row.delta) <= 5.0,
                    ]
                )
    with (table_dir / "prostate_urea_calibration_shifts.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["cohort", "replicate_index", "applied_shift_cm-1", "alignment_applied"])
        for cohort, values in zip(("Clean PRO", "Prospective cancer"), data.shifts, strict=True):
            for index, value in enumerate(values, start=1):
                writer.writerow([cohort, index, value, value != 0.0])


def validate_report_snapshot(report: Path, table_dir: Path = TABLE_DIR) -> None:
    with (table_dir / "prostate_classification_metrics.csv").open(encoding="utf-8-sig") as handle:
        values = {
            (row["task"], row["metric"]): float(row["value"]) for row in csv.DictReader(handle)
        }
    expected = [
        f"| Screening: Control+Biopsy-negative vs Cancer | 109 | {values[('screening_binary', 'roc_auc')]:.3f}",
        f"| 3-group: Control/Biopsy-negative/Cancer | 109 | {values[('three_group', 'macro_ovr_roc_auc')]:.3f}",
    ]
    text = report.read_text(encoding="utf-8")
    missing = [item for item in expected if item not in text]
    if missing:
        raise ReportDriftError(missing)
