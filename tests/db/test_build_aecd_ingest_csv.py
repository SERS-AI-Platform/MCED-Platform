from __future__ import annotations

import csv
from pathlib import Path

import pytest
from openpyxl import Workbook

from scripts.db.build_aecd_ingest_csv import (
    DEFAULT_OUTPUT_DIR,
    BuildConfig,
    CsvTable,
    IngestCsvError,
    build_rows,
    resolve_output_path,
    write_rows,
)


def _column(table: CsvTable, name: str) -> tuple[str, ...]:
    position = table.headers.index(name)
    return tuple(row[position] for row in table.rows)


def _write_workbook(path: Path) -> None:
    workbook = Workbook()
    worksheet = workbook["Sheet"]
    worksheet.title = "table"
    worksheet.append(
        [
            "group",
            "solum_label",
            "patient_code",
            "sex",
            "birth_date",
            "smoking_history",
            "drinking_history",
            "sample_type",
            "sample_amount",
            "collection_date",
            "receipt_date",
            "primary_sample_timing_interpretation",
            "post_treatment_collection_date",
            "diagnosis_name",
            "psa",
            None,
        ]
    )
    worksheet.append(
        [
            "control",
            "BNOR_1",
            "P001",
            "남성",
            None,
            "never",
            "none",
            "urine",
            None,
            "2026-08-01",
            "2026-08-01",
            None,
            None,
            "control",
            "1.2",
            None,
        ]
    )
    worksheet.append(
        [
            "prostate",
            "BPRO 2",
            "P002",
            "남성",
            None,
            "former",
            "social",
            "urine",
            None,
            "2026-08-02",
            "2026-08-02",
            None,
            None,
            "prostate cancer",
            "8.4",
            None,
        ]
    )
    workbook.save(path)


def test_build_rows_matches_normalized_labels_and_preserves_source_values(
    tmp_path: Path,
) -> None:
    # Given
    workbook_path = tmp_path / "clinical.xlsx"
    mapping_root = tmp_path / "mapping"
    (mapping_root / "20260810_BNOR_mapping" / "BNOR 1").mkdir(parents=True)
    (mapping_root / "20260814_BPRO_mapping" / "BPRO_2").mkdir(parents=True)
    _write_workbook(workbook_path)
    config = BuildConfig(
        workbook_path=workbook_path,
        mapping_root=mapping_root,
        sheet_name="table",
        site_code="smcxd07",
    )

    # When
    table = build_rows(config)

    # Then
    assert _column(table, "solum_label") == ("BNOR_1", "BPRO 2")
    assert _column(table, "source_mapping") == (
        "20260810_BNOR_mapping",
        "20260814_BPRO_mapping",
    )
    assert _column(table, "site_code") == ("smcxd07", "smcxd07")


def test_write_rows_emits_utf8_bom_csv_after_validation(tmp_path: Path) -> None:
    # Given
    workbook_path = tmp_path / "clinical.xlsx"
    mapping_root = tmp_path / "mapping"
    output_path = tmp_path / "clinical_master.csv"
    (mapping_root / "20260810_BNOR_mapping" / "BNOR 1").mkdir(parents=True)
    (mapping_root / "20260814_BPRO_mapping" / "BPRO_2").mkdir(parents=True)
    _write_workbook(workbook_path)
    config = BuildConfig(
        workbook_path=workbook_path,
        mapping_root=mapping_root,
        sheet_name="table",
        site_code="smcxd07",
    )
    table = build_rows(config)

    # When
    write_rows(output_path, table)

    # Then
    assert output_path.read_bytes().startswith(b"\xef\xbb\xbf")
    with output_path.open(encoding="utf-8-sig", newline="") as source:
        records = list(csv.DictReader(source))
    assert len(records) == 2
    assert records[0]["patient_code"] == "P001"
    assert records[0]["diagnosis_name"] == "control"
    assert records[1]["psa"] == "8.4"
    assert "" not in records[0]


def test_build_rows_rejects_unmatched_mapping_folder(tmp_path: Path) -> None:
    # Given
    workbook_path = tmp_path / "clinical.xlsx"
    mapping_root = tmp_path / "mapping"
    (mapping_root / "20260810_BNOR_mapping" / "BNOR 999").mkdir(parents=True)
    _write_workbook(workbook_path)
    config = BuildConfig(
        workbook_path=workbook_path,
        mapping_root=mapping_root,
        sheet_name="table",
        site_code="smcxd07",
    )
    # When / Then
    with pytest.raises(IngestCsvError, match="mapping labels not found"):
        _ = build_rows(config)


def test_build_rows_rejects_empty_mapping_root(tmp_path: Path) -> None:
    # Given
    workbook_path = tmp_path / "clinical.xlsx"
    mapping_root = tmp_path / "mapping"
    mapping_root.mkdir()
    _write_workbook(workbook_path)
    config = BuildConfig(
        workbook_path=workbook_path,
        mapping_root=mapping_root,
        sheet_name="table",
        site_code="smcxd07",
    )

    # When / Then
    with pytest.raises(IngestCsvError, match="no sample folders found"):
        _ = build_rows(config)


def test_resolve_output_path_uses_site_code_when_output_is_omitted() -> None:
    # Given / When
    output_path = resolve_output_path(None, "smcxd07")

    # Then
    assert output_path == DEFAULT_OUTPUT_DIR / "smcxd07_clinical_master.csv"


def test_resolve_output_path_preserves_explicit_output(tmp_path: Path) -> None:
    # Given
    requested_path = tmp_path / "review.csv"

    # When
    output_path = resolve_output_path(requested_path, "smcxd07")

    # Then
    assert output_path == requested_path
