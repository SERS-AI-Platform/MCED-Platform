from __future__ import annotations

import sqlite3
from pathlib import Path

from openpyxl import Workbook

from sers.master_data.clinical_types import ClinicalFormat, ClinicalSource
from sers.master_data.schema import initialize_schema


def database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY)")
    initialize_schema(connection)
    return connection


def source(
    path: Path,
    site_code: str,
    source_format: ClinicalFormat,
) -> ClinicalSource:
    return ClinicalSource(
        path=path,
        site_code=site_code,
        site_name=f"Synthetic {site_code}",
        protocol_code="SYNTHETIC-CRF",
        source_group="SYNTHETIC",
        source_format=source_format,
        patient_id_fields=("SUBJID",),
        event_type="lab",
        sheet_name="Clinical" if source_format is ClinicalFormat.EXCEL else None,
    )


def write_csv(path: Path, rows: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "SUBJID,VISIT,LBORRES,mystery_field\n" + "\n".join(rows) + "\n",
        encoding="utf-8-sig",
    )


def write_excel(path: Path, source_key: str) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Clinical"
    sheet.append(("SUBJID", "VISIT", "AGE", "unknown_excel_field"))
    sheet.append((source_key, "V1", 44, "retained"))
    workbook.save(path)


def raw_blobs(raw_store: Path) -> set[Path]:
    return {
        path.relative_to(raw_store)
        for path in raw_store.rglob("*")
        if path.is_file()
    }
