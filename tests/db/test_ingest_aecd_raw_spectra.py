from __future__ import annotations

from pathlib import Path

import pytest

from scripts.db.aecd_raw_spectra_ingest import (
    IngestError,
    SpectrumKey,
    discover_spectrum_files,
    read_spectrum,
    source_mapping_from_notes,
)


def _write_spectrum(path: Path, rows: tuple[tuple[float, float], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        "".join(f"{x},{y}\n" for x, y in rows),
        encoding="utf-8",
    )


def test_discovery_accepts_normal_and_ave_prefixed_point_files(tmp_path: Path) -> None:
    # Given
    normal = tmp_path / "20260810_BNOR_mapping" / "BNOR 1"
    special = tmp_path / "20260811_BNOR_mapping" / "BNOR 37"
    rows = ((50.0, 100.0), (52.0, 110.0))
    _write_spectrum(normal / "BNOR 1_0001.CSV", rows)
    _write_spectrum(normal / "BNOR 1_0002.CSV", rows)
    _write_spectrum(normal / "BNOR 1_ave.CSV", rows)
    _write_spectrum(special / "BNOR 37_ave0001.CSV", rows)
    _write_spectrum(special / "BNOR 37_ave0002.CSV", rows)
    _write_spectrum(special / "BNOR 37_ave.CSV", rows)

    # When
    discovery = discover_spectrum_files(tmp_path, expected_points_per_sample=2)

    # Then
    assert discovery.averages_excluded == 2
    assert tuple(item.key for item in discovery.files) == (
        SpectrumKey("20260810_BNOR_mapping", "BNOR_1", 1),
        SpectrumKey("20260810_BNOR_mapping", "BNOR_1", 2),
        SpectrumKey("20260811_BNOR_mapping", "BNOR_37", 1),
        SpectrumKey("20260811_BNOR_mapping", "BNOR_37", 2),
    )


def test_discovery_rejects_missing_point_number(tmp_path: Path) -> None:
    # Given
    sample = tmp_path / "20260810_BNOR_mapping" / "BNOR 1"
    rows = ((50.0, 100.0), (52.0, 110.0))
    _write_spectrum(sample / "BNOR 1_0001.CSV", rows)
    _write_spectrum(sample / "BNOR 1_0003.CSV", rows)

    # When / Then
    with pytest.raises(IngestError, match="point set mismatch"):
        _ = discover_spectrum_files(tmp_path, expected_points_per_sample=3)


def test_read_spectrum_rejects_non_increasing_wavenumber(tmp_path: Path) -> None:
    # Given
    path = tmp_path / "invalid.CSV"
    _write_spectrum(path, ((52.0, 100.0), (50.0, 110.0)))

    # When / Then
    with pytest.raises(IngestError, match="strictly increasing"):
        _ = read_spectrum(path)


def test_source_mapping_is_extracted_from_run_notes() -> None:
    # Given / When
    source_mapping = source_mapping_from_notes(
        "source_mapping=20260813_BPRO_mapping; strip_units=1"
    )

    # Then
    assert source_mapping == "20260813_BPRO_mapping"


def test_ingest_error_allows_traceback_assignment() -> None:
    # Given
    error = IngestError("mapping mismatch")

    # When
    error.__traceback__ = None

    # Then
    assert error.__traceback__ is None
