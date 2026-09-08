from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from powder_comparison.runner import AnalysisResult


def _writer(path: Path):
    handle = path.open("w", newline="", encoding="utf-8-sig")
    return handle, csv.writer(handle)


def write_peak_tables(result: AnalysisResult, table_dir: Path) -> None:
    handle, writer = _writer(table_dir / "lr_peak_importance_comparison.csv")
    with handle:
        writer.writerow(["peak", "legacy_relative_importance", "powder_relative_importance"])
        for name, legacy, powder in zip(
            result.peaks.legacy.names,
            result.peaks.legacy.global_importance,
            result.peaks.powder.global_importance,
            strict=True,
        ):
            writer.writerow([name, legacy, powder])
    handle, writer = _writer(table_dir / "reclassified_patient_peaks.csv")
    with handle:
        writer.writerow(
            [
                "patient_id",
                "sample_id",
                "reclassification",
                "legacy_top5_peaks",
                "powder_top5_peaks",
            ]
        )
        truth = result.screening.y_true
        legacy_prediction = result.screening.legacy_probabilities.argmax(axis=1)
        powder_prediction = result.screening.transfer_probabilities.argmax(axis=1)
        for subject_index in range(len(truth)):
            old_correct = legacy_prediction[subject_index] == truth[subject_index]
            new_correct = powder_prediction[subject_index] == truth[subject_index]
            if old_correct == new_correct:
                continue
            status = "improved" if new_correct else "worsened"
            old_order = np.argsort(-np.abs(result.peaks.legacy.local_contributions[subject_index]))[
                :5
            ]
            new_order = np.argsort(
                -np.abs(result.peaks.transfer.local_contributions[subject_index])
            )[:5]
            writer.writerow(
                [
                    result.clinical.patient_ids[subject_index],
                    f"{result.pairs[subject_index].powder.prefix} "
                    f"{result.pairs[subject_index].sample_id}",
                    status,
                    ";".join(result.peaks.legacy.names[index] for index in old_order),
                    ";".join(result.peaks.powder.names[index] for index in new_order),
                ]
            )
