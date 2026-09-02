from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .group_separation import GroupSeparationDiagnostics, evaluate_group_separation
from .runner import AnalysisResult


def _writer(path: Path):
    handle = path.open("w", newline="", encoding="utf-8-sig")
    return handle, csv.writer(handle)


def _write_pair_summary(
    result: AnalysisResult,
    diagnostics: GroupSeparationDiagnostics,
    table_dir: Path,
) -> None:
    handle, writer = _writer(table_dir / "clinical_group_spectral_separation.csv")
    with handle:
        writer.writerow(
            [
                "modality",
                "group_a",
                "group_b",
                "n_a",
                "n_b",
                "centroid_rms",
                "pooled_within_rms",
                "separation_to_spread_ratio",
                "median_abs_hedges_g",
                "max_abs_hedges_g",
                "max_effect_raman_shift_cm_1",
            ]
        )
        for modality, rows in (
            ("legacy_liquid", diagnostics.legacy_pairs),
            ("powder", diagnostics.powder_pairs),
        ):
            for row in rows:
                absolute = np.abs(row.hedges_g)
                writer.writerow(
                    [
                        modality,
                        row.group_a,
                        row.group_b,
                        row.n_a,
                        row.n_b,
                        row.centroid_rms,
                        row.pooled_within_rms,
                        row.separation_to_spread,
                        float(np.median(absolute)),
                        float(np.max(absolute)),
                        result.spectra.grid[int(np.argmax(absolute))],
                    ]
                )


def _write_peak_effects(
    result: AnalysisResult,
    diagnostics: GroupSeparationDiagnostics,
    table_dir: Path,
) -> None:
    handle, writer = _writer(table_dir / "clinical_group_peak_effect_sizes.csv")
    with handle:
        writer.writerow(
            ["modality", "group_a", "group_b", "n_a", "n_b", "raman_shift_cm_1", "hedges_g"]
        )
        for modality, rows in (
            ("legacy_liquid", diagnostics.legacy_pairs),
            ("powder", diagnostics.powder_pairs),
        ):
            for row in rows:
                for shift, effect in zip(result.spectra.grid, row.hedges_g, strict=True):
                    writer.writerow(
                        [modality, row.group_a, row.group_b, row.n_a, row.n_b, shift, effect]
                    )


def _write_centroid_margins(
    result: AnalysisResult,
    diagnostics: GroupSeparationDiagnostics,
    table_dir: Path,
) -> None:
    handle, writer = _writer(table_dir / "clinical_group_centroid_margins.csv")
    with handle:
        writer.writerow(
            [
                "patient_id",
                "sample_id",
                "clinical_group",
                "modality",
                "own_centroid_rms",
                "nearest_other_centroid_rms",
                "centroid_margin",
                "nearest_other_group",
            ]
        )
        for modality, margins in (
            ("legacy_liquid", diagnostics.legacy_margins),
            ("powder", diagnostics.powder_margins),
        ):
            for index, pair in enumerate(result.pairs):
                writer.writerow(
                    [
                        result.clinical.patient_ids[index],
                        f"{pair.powder.prefix} {pair.sample_id}",
                        pair.clinical_group,
                        modality,
                        margins.own_distance[index],
                        margins.nearest_other_distance[index],
                        margins.margin[index],
                        margins.nearest_other_group[index],
                    ]
                )


def write_group_separation_tables(result: AnalysisResult, table_dir: Path) -> None:
    diagnostics = evaluate_group_separation(result.spectra)
    _write_pair_summary(result, diagnostics, table_dir)
    _write_peak_effects(result, diagnostics, table_dir)
    _write_centroid_margins(result, diagnostics, table_dir)
