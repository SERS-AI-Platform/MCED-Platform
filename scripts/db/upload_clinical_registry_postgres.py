#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "openpyxl>=3.1",
#     "psycopg2-binary>=2.9",
#     "typer>=0.12",
# ]
# ///

# ─── How to run ───
# 1. Install uv (if not installed):
#      curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Run directly (no venv, no pip install needed):
#      uv run upload_clinical_registry_postgres.py --input-file <XLSX> --dry-run
# 3. Or make executable and run the script without --dry-run for a database write.
# ──────────────────

"""Load the normalized patient workbook into a non-destructive PostgreSQL registry."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, time
from getpass import getpass
from pathlib import Path
from typing import Annotated, TypeAlias

import psycopg2
import typer
from openpyxl import load_workbook
from openpyxl.cell.rich_text import CellRichText
from openpyxl.worksheet.formula import ArrayFormula, DataTableFormula
from psycopg2.extras import Json, execute_values

JsonScalar: TypeAlias = str | int | float | bool | None
JsonRecord: TypeAlias = dict[str, JsonScalar]
ExcelValue: TypeAlias = str | int | float | bool | date | time | None | ArrayFormula | DataTableFormula | CellRichText
DbValue: TypeAlias = str | int | float | None | Json

DEFAULT_INPUT = Path("/mnt/c/Users/user/Downloads/전체환자_임상정보_정규화.xlsx")
DEFAULT_DATASET = "clinical_normalized_xlsx_v1"
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS public.clinical_patient_registry (
    id BIGSERIAL PRIMARY KEY,
    dataset_version TEXT NOT NULL,
    source_workbook TEXT NOT NULL,
    source_sha256 TEXT NOT NULL,
    source_row_number INTEGER NOT NULL,
    group_raw TEXT,
    solum_label TEXT NOT NULL,
    patient_code TEXT,
    cancer_type TEXT,
    sex TEXT,
    age DOUBLE PRECISION,
    birth_date TEXT,
    weight_kg DOUBLE PRECISION,
    height_cm DOUBLE PRECISION,
    bmi DOUBLE PRECISION,
    sample_type TEXT,
    collection_date TEXT,
    raw_record JSONB NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (dataset_version, solum_label)
);
CREATE INDEX IF NOT EXISTS idx_clinical_registry_group
    ON public.clinical_patient_registry (dataset_version, group_raw);
CREATE INDEX IF NOT EXISTS idx_clinical_registry_patient_code
    ON public.clinical_patient_registry (dataset_version, patient_code);
"""


@dataclass(frozen=True, slots=True)
class RegistryRow:
    row_number: int
    record: JsonRecord


@dataclass(frozen=True, slots=True)
class ScanSummary:
    rows: tuple[RegistryRow, ...]
    headers: tuple[str, ...]
    groups: tuple[tuple[str, int], ...]
    source_sha256: str


@dataclass(frozen=True, slots=True)
class ConnectionOptions:
    host: str
    port: int
    dbname: str
    user: str
    password: str


def scalar(value: ExcelValue) -> JsonScalar:
    """Convert an Excel cell to a JSON-safe scalar without changing text."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return str(value).strip() or None


def text_value(record: JsonRecord, key: str) -> str | None:
    value = record.get(key)
    return value if isinstance(value, str) else str(value) if value is not None else None


def numeric_value(record: JsonRecord, key: str) -> float | None:
    value = record.get(key)
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(str(value).strip())
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def unique_headers(values: tuple[ExcelValue, ...]) -> tuple[str, ...]:
    seen: dict[str, int] = {}
    headers: list[str] = []
    for position, value in enumerate(values, start=1):
        base = str(value).strip() if value is not None else ""
        base = base or f"column_{position}"
        count = seen.get(base, 0) + 1
        seen[base] = count
        headers.append(base if count == 1 else f"{base}__{count}")
    return tuple(headers)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scan_workbook(path: Path, sheet_name: str) -> ScanSummary:
    if not path.is_file():
        raise FileNotFoundError(path)
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        if sheet_name not in workbook.sheetnames:
            raise ValueError(f"Sheet not found: {sheet_name}; available={workbook.sheetnames}")
        worksheet = workbook[sheet_name]
        iterator = worksheet.iter_rows(values_only=True)
        headers = unique_headers(tuple(next(iterator)))
        rows = tuple(
            RegistryRow(row_number, dict(zip(headers, (scalar(value) for value in values))))
            for row_number, values in enumerate(iterator, start=2)
        )
    finally:
        workbook.close()
    groups = Counter(text_value(row.record, "group") or "<blank>" for row in rows)
    return ScanSummary(rows, headers, tuple(sorted(groups.items())), sha256_file(path))


def print_summary(path: Path, summary: ScanSummary, dataset_version: str) -> None:
    labels = [text_value(row.record, "solum_label") for row in summary.rows]
    nonnull_labels = [label for label in labels if label is not None]
    print({
        "input_file": str(path),
        "dataset_version": dataset_version,
        "sheet": "table",
        "rows": len(summary.rows),
        "columns": len(summary.headers),
        "groups": dict(summary.groups),
        "solum_label_nonnull": len(nonnull_labels),
        "solum_label_unique": len(set(nonnull_labels)),
        "source_sha256": summary.source_sha256,
    })


def connection_options(host: str | None, port: int | None, dbname: str | None, user: str | None) -> ConnectionOptions:
    password = os.environ.get("PGPASSWORD") or getpass("PostgreSQL password (not saved): ")
    if not password:
        raise RuntimeError("A PostgreSQL password is required for database writes")
    return ConnectionOptions(
        host or os.environ.get("PGHOST", "localhost"),
        port or int(os.environ.get("PGPORT", "5432")),
        dbname or os.environ.get("PGDATABASE", "sers_clinical"),
        user or os.environ.get("PGUSER", "postgres"),
        password,
    )


def write_database(
    summary: ScanSummary,
    path: Path,
    dataset_version: str,
    batch_size: int,
    host: str | None,
    port: int | None,
    dbname: str | None,
    user: str | None,
) -> None:
    insert_sql = """
    INSERT INTO public.clinical_patient_registry
        (dataset_version, source_workbook, source_sha256, source_row_number,
         group_raw, solum_label, patient_code, cancer_type, sex, age,
         birth_date, weight_kg, height_cm, bmi, sample_type, collection_date,
         raw_record)
    VALUES %s
    ON CONFLICT (dataset_version, solum_label) DO UPDATE SET
        source_workbook = EXCLUDED.source_workbook,
        source_sha256 = EXCLUDED.source_sha256,
        source_row_number = EXCLUDED.source_row_number,
        group_raw = EXCLUDED.group_raw,
        patient_code = EXCLUDED.patient_code,
        cancer_type = EXCLUDED.cancer_type,
        sex = EXCLUDED.sex,
        age = EXCLUDED.age,
        birth_date = EXCLUDED.birth_date,
        weight_kg = EXCLUDED.weight_kg,
        height_cm = EXCLUDED.height_cm,
        bmi = EXCLUDED.bmi,
        sample_type = EXCLUDED.sample_type,
        collection_date = EXCLUDED.collection_date,
        raw_record = EXCLUDED.raw_record,
        ingested_at = CURRENT_TIMESTAMP
    """
    options = connection_options(host, port, dbname, user)
    with psycopg2.connect(
        host=options.host, port=options.port, dbname=options.dbname,
        user=options.user, password=options.password,
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(SCHEMA_SQL)
            rows: list[tuple[DbValue, ...]] = []
            for item in summary.rows:
                record = item.record
                rows.append((
                    dataset_version, path.name, summary.source_sha256, item.row_number,
                    text_value(record, "group"), text_value(record, "solum_label"),
                    text_value(record, "patient_code"), text_value(record, "cancer_type"),
                    text_value(record, "sex"), numeric_value(record, "age"),
                    text_value(record, "birth_date"), numeric_value(record, "weight_kg"),
                    numeric_value(record, "height_cm"), numeric_value(record, "bmi"),
                    text_value(record, "sample_type"), text_value(record, "collection_date"),
                    Json(record, dumps=lambda value: json.dumps(value, ensure_ascii=False)),
                ))
                if len(rows) >= batch_size:
                    execute_values(cursor, insert_sql, rows, page_size=batch_size)
                    rows = []
            if rows:
                execute_values(cursor, insert_sql, rows, page_size=batch_size)
            cursor.execute(
                "SELECT COUNT(*), COUNT(DISTINCT solum_label) "
                "FROM public.clinical_patient_registry WHERE dataset_version = %s",
                (dataset_version,),
            )
            result = cursor.fetchone()
            if result is None:
                raise RuntimeError("Database validation query returned no row")
            total, unique_labels = result
    print({"db_rows": total, "db_unique_solum_labels": unique_labels, "dataset_version": dataset_version})
    if total != len(summary.rows) or unique_labels != len(summary.rows):
        raise RuntimeError(f"Database validation mismatch: expected {len(summary.rows)}, got {(total, unique_labels)}")


def main(
    input_file: Annotated[Path, typer.Option("--input-file")] = DEFAULT_INPUT,
    dataset_version: Annotated[str, typer.Option("--dataset-version")] = DEFAULT_DATASET,
    sheet_name: Annotated[str, typer.Option("--sheet-name")] = "table",
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = 200,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    host: str | None = None,
    port: int | None = None,
    dbname: str | None = None,
    user: str | None = None,
) -> None:
    """Scan or safely upsert a normalized clinical workbook."""
    summary = scan_workbook(input_file, sheet_name)
    print_summary(input_file, summary, dataset_version)
    if not dry_run:
        write_database(summary, input_file, dataset_version, batch_size, host, port, dbname, user)


if __name__ == "__main__":
    typer.run(main)
