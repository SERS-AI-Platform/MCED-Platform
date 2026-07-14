from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

SRC = (
    Path(__file__).resolve().parents[2] / "publications" / "전향검체" / "연세세브란스 병원" / "src"
)
sys.path.insert(0, str(SRC))

from severance_data import load_cohort_spectra  # noqa: E402
from severance_lr_model import bin_spectra  # noqa: E402
from severance_peaks import PeakObservation, cluster_peaks  # noqa: E402


def test_load_cohort_spectra_when_replicates_present_aggregates_by_subject(
    tmp_path: Path,
) -> None:
    # Given
    processed = tmp_path / "processed.csv"
    with processed.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "sample_id", "replicate", "x_400.0", "x_500.0"])
        writer.writerows(
            [
                ["YPAN", 7, 1, 1.0, 3.0],
                ["YPAN", 7, 2, 3.0, 5.0],
                ["YNOR", 2, 1, 0.0, 2.0],
                ["YNOR", 2, 2, 2.0, 4.0],
                ["CPAN", 4, 1, 5.0, 7.0],
                ["CPAN", 4, 2, 7.0, 9.0],
                ["CPAN", 5, 1, 100.0, 100.0],
            ]
        )
    manifest = tmp_path / "clean.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_group", "sample_id"])
        writer.writerow(["CPAN", 4])

    # When
    cohort = load_cohort_spectra(processed, manifest)

    # Then
    assert cohort.grid.tolist() == [400.0, 500.0]
    assert cohort.ypan_subject_ids == ("YPAN_7",)
    assert cohort.ynor_subject_ids == ("YNOR_2",)
    np.testing.assert_allclose(cohort.ypan, [[2.0, 4.0]])
    np.testing.assert_allclose(cohort.ynor, [[1.0, 3.0]])
    np.testing.assert_allclose(cohort.clean_cpan, [[6.0, 8.0]])


def test_cluster_peaks_when_all_groups_are_within_tolerance_marks_common() -> None:
    # Given
    peaks = (
        PeakObservation(group_index=0, center=1000.0, intensity=1.0, prominence=0.3),
        PeakObservation(group_index=1, center=1008.0, intensity=1.2, prominence=0.4),
        PeakObservation(group_index=1, center=1300.0, intensity=0.8, prominence=0.2),
    )

    # When
    consensus = cluster_peaks(("Control", "Cancer"), peaks)

    # Then
    assert [(item.center, item.category) for item in consensus] == [
        (1004.0, "common"),
        (1300.0, "differential"),
    ]


def test_load_cohort_spectra_when_second_measurement_exists_keeps_unique_subject(
    tmp_path: Path,
) -> None:
    # Given
    processed = tmp_path / "processed.csv"
    with processed.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "sample_id", "replicate", "x_400.0", "x_500.0"])
        writer.writerows(
            [
                ["YPAN", 7, 1, 1.0, 2.0],
                ["YNOR", 21, 1, 8.0, 9.0],
                ["YNOR", 48, 1, 3.0, 4.0],
                ["CPAN", 4, 1, 5.0, 6.0],
            ]
        )
    manifest = tmp_path / "clean.csv"
    with manifest.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_group", "sample_id"])
        writer.writerow(["CPAN", 4])

    # When
    cohort = load_cohort_spectra(processed, manifest)

    # Then
    assert cohort.ynor_subject_ids == ("YNOR_48",)
    np.testing.assert_allclose(cohort.ynor, [[3.0, 4.0]])


def test_bin_spectra_when_width_does_not_divide_grid_discards_trailing_points() -> None:
    # Given
    spectra = np.arange(20, dtype=float).reshape(2, 10)

    # When
    binned = bin_spectra(spectra, width=4)

    # Then
    np.testing.assert_allclose(binned, [[1.5, 5.5], [11.5, 15.5]])
