from __future__ import annotations

import csv
from pathlib import Path

from powder_comparison.runner import AnalysisResult


def _writer(path: Path):
    handle = path.open("w", newline="", encoding="utf-8-sig")
    return handle, csv.writer(handle)


def write_preprocessing_tables(result: AnalysisResult, table_dir: Path) -> None:
    handle, writer = _writer(table_dir / "preprocessing_stage_shift.csv")
    with handle:
        writer.writerow(
            [
                "stage",
                "median_paired_pearson_r",
                "pearson_r_q1",
                "pearson_r_q3",
                "median_nrmse_vs_legacy_rms",
                "median_log2_powder_to_legacy_rms",
            ]
        )
        for row in result.preprocessing.rows:
            writer.writerow(
                [
                    row.stage,
                    row.median_correlation,
                    row.correlation_q1,
                    row.correlation_q3,
                    row.median_nrmse,
                    row.median_log2_rms_ratio,
                ]
            )
    handle, writer = _writer(table_dir / "preprocessing_patient_shift.csv")
    with handle:
        writer.writerow(
            [
                "patient_id",
                "sample_id",
                "clinical_group",
                "snv_paired_rmse",
                "absolute_lr_probability_shift",
            ]
        )
        for index, pair in enumerate(result.pairs):
            writer.writerow(
                [
                    result.clinical.patient_ids[index],
                    f"{pair.powder.prefix} {pair.sample_id}",
                    pair.clinical_group,
                    result.preprocessing.snv_distance[index],
                    result.preprocessing.probability_shift[index],
                ]
            )
