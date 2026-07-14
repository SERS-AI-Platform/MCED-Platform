from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "publications" / "전향검체" / "연세세브란스 병원"
TABLES = ROOT / "tables"


def read_rows(name: str) -> list[dict[str, str]]:
    with (TABLES / name).open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def test_group_inventory_separates_clinical_subject_and_acquisition_counts() -> None:
    rows = {row["source_group"]: row for row in read_rows("yonsei_ynor_ypan_group_inventory.csv")}

    assert rows["YNOR"]["clinical_records"] == "30"
    assert rows["YNOR"]["primary_spectrum_subjects"] == "29"
    assert rows["YNOR"]["processed_acquisitions"] == "30"
    assert rows["YNOR"]["primary_model_subjects"] == "29"
    assert rows["YPAN"]["clinical_records"] == "30"
    assert rows["YPAN"]["primary_model_subjects"] == "30"


def test_acquisition_inventory_has_primary_and_reacquired_sets() -> None:
    rows = read_rows("yonsei_ynor_ypan_acquisition_inventory.csv")

    assert len(rows) == 118
    assert sum(row["acquisition_type"] == "primary" for row in rows) == 59
    assert sum(row["acquisition_type"] == "reacquired" for row in rows) == 59
    assert sum(int(row["n_replicates"]) for row in rows) == 590


def test_clinical_inventory_is_pseudonymized_and_complete() -> None:
    rows = read_rows("yonsei_ynor_ypan_clinical_records.csv")

    assert len(rows) == 60
    assert sum(row["spectrum_link_status"] == "clinical_only" for row in rows) == 1
    assert "patient_id" not in rows[0]
    assert "source_file" not in rows[0]
    assert "past_history" not in rows[0]
    assert "diagnosis_date" not in rows[0]
    assert "sample_date" not in rows[0]
    assert "diagnosis_year" in rows[0]


def test_processed_inventory_marks_repeat_acquisition_excluded() -> None:
    rows = read_rows("yonsei_ynor_ypan_processed_inventory.csv")

    assert len(rows) == 60
    assert sum(row["analysis_included"] == "no" for row in rows) == 1
    excluded = next(row for row in rows if row["analysis_included"] == "no")
    assert excluded["acquisition_role"] == "repeat_acquisition"
    assert excluded["linked_primary_case_id"]


def test_summary_contains_clinical_composition_warning() -> None:
    text = (ROOT / "YNOR_YPAN_DATA_SUMMARY.md").read_text(encoding="utf-8")

    assert "건강한 자 10명" in text
    assert "췌장낭종 10명" in text
    assert "primary 분석은 59명" in text
    assert "cross-hospital 일반화" in text
