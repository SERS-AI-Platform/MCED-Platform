from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from powder_comparison.delong import correlated_auc_test
from powder_comparison.runner import AnalysisResult
from powder_comparison.statistics import holm_adjust


def _write_mcnemar(result: AnalysisResult, table_dir: Path) -> None:
    p_values = np.array([row.result.p_value for row in result.mcnemar])
    adjusted = holm_adjust(p_values)
    path = table_dir / "lr_screening_mcnemar.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["scope", "legacy_only_correct", "powder_only_correct", "exact_p", "holm_p"]
        )
        for row, holm_p in zip(result.mcnemar, adjusted, strict=True):
            writer.writerow(
                [
                    row.scope,
                    row.result.legacy_only,
                    row.result.powder_only,
                    row.result.p_value,
                    holm_p,
                ]
            )


def _write_delong(result: AnalysisResult, table_dir: Path) -> None:
    rows = (
        (
            "lr_screening_liquid_to_powder",
            correlated_auc_test(
                result.screening.y_true,
                result.screening.legacy_probabilities[:, 1],
                result.screening.transfer_probabilities[:, 1],
            ),
        ),
    )
    path = table_dir / "correlated_auc_delong.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["comparison", "legacy_auc", "powder_auc", "delta", "z_score", "p_value"])
        for name, row in rows:
            writer.writerow(
                [name, row.legacy_auc, row.powder_auc, row.delta, row.z_score, row.p_value]
            )


def _write_quality(result: AnalysisResult, table_dir: Path) -> None:
    path = table_dir / "modality_quality_summary.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "modality",
                "subjects",
                "replicates_per_subject_median",
                "raw_mean_intensity_median",
                "raw_max_intensity_median",
                "replicate_correlation_median",
            ]
        )
        for modality, spectra, replicate_counts in (
            (
                "legacy_liquid",
                result.spectra.legacy,
                [len(pair.legacy.replicates) for pair in result.pairs],
            ),
            (
                "powder",
                result.spectra.powder,
                [len(pair.powder.replicates) for pair in result.pairs],
            ),
        ):
            writer.writerow(
                [
                    modality,
                    len(result.pairs),
                    float(np.median(replicate_counts)),
                    float(np.median(spectra.raw_mean_intensity)),
                    float(np.median(spectra.raw_max_intensity)),
                    float(np.median(spectra.replicate_correlation)),
                ]
            )


def write_inference_tables(result: AnalysisResult) -> Path:
    table_dir = result.sources.output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    (table_dir / "production_mcnemar.csv").unlink(missing_ok=True)
    _write_mcnemar(result, table_dir)
    _write_delong(result, table_dir)
    _write_quality(result, table_dir)
    return table_dir
