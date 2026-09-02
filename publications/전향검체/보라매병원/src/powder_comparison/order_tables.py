from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from powder_comparison.runner import AnalysisResult
from powder_comparison.statistics import holm_adjust


def write_order_tables(result: AnalysisResult, table_dir: Path) -> None:
    primary = [row for row in result.order_effects if row.scope == "BPRO_label_adjusted"]
    secondary = [row for row in result.order_effects if row.scope != "BPRO_label_adjusted"]
    adjusted_by_row = {
        id(row): adjusted
        for family in (primary, secondary)
        for row, adjusted in zip(
            family,
            holm_adjust(np.array([item.result.p_value for item in family])),
            strict=True,
        )
    }
    effects_path = table_dir / "measurement_order_effects.csv"
    with effects_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "scope",
                "endpoint",
                "n",
                "stratified_spearman_rho",
                "permutation_p",
                "holm_p_within_analysis_family",
            ]
        )
        for row in result.order_effects:
            writer.writerow(
                [
                    row.scope,
                    row.endpoint,
                    row.result.n,
                    row.result.rho,
                    row.result.p_value,
                    adjusted_by_row[id(row)],
                ]
            )
    tertile_path = table_dir / "measurement_order_tertiles.csv"
    with tertile_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "period",
                "n",
                "probability_mean",
                "error_rate",
                "pc1_mean",
                "raw_intensity_mean",
                "replicate_correlation_mean",
            ]
        )
        for row in result.tertiles:
            writer.writerow(
                [
                    row.period,
                    row.n,
                    row.probability_mean,
                    row.error_rate,
                    row.pc1_mean,
                    row.raw_intensity_mean,
                    row.replicate_correlation_mean,
                ]
            )
