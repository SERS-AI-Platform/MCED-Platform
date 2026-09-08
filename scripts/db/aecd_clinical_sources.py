from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

try:
    from .aecd_measurement_sources import sha256_file
except ImportError:
    from aecd_measurement_sources import sha256_file


@dataclass(frozen=True, slots=True)
class ClinicalRecord:
    patient_id: str
    source_file: str
    source_row_number: int
    source_sha256: str
    values: dict[str, str | None]


@dataclass(frozen=True, slots=True)
class ClinicalScan:
    records: tuple[ClinicalRecord, ...]
    source_files: tuple[str, ...]


def _normalize_row(row: dict[str, str | None]) -> dict[str, str | None]:
    return {str(key).strip(): (value.strip() if value and value.strip() else None) for key, value in row.items()}


def _clinical_paths(repo_root: Path) -> tuple[Path, ...]:
    root = repo_root / "data" / "clinical_data" / "standardized"
    canonical = root / "all_clinical_standardized.csv"
    if canonical.is_file():
        return (canonical,)
    return tuple(sorted(root.glob("*_clinical_standardized.csv")))


def scan_clinical(repo_root: Path) -> ClinicalScan:
    records_by_patient: dict[str, ClinicalRecord] = {}
    paths = _clinical_paths(repo_root)
    if not paths:
        raise FileNotFoundError("No standardized clinical CSV found")
    for path in paths:
        source_sha = sha256_file(path)
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames is None or "patient_id" not in reader.fieldnames:
                raise ValueError(f"Clinical CSV has no patient_id column: {path}")
            for row_number, row in enumerate(reader, start=2):
                values = _normalize_row(row)
                patient_id = values.get("patient_id")
                if patient_id is None:
                    continue
                records_by_patient[patient_id] = ClinicalRecord(
                    patient_id=patient_id,
                    source_file=str(path.resolve()),
                    source_row_number=row_number,
                    source_sha256=source_sha,
                    values=values,
                )
    return ClinicalScan(tuple(records_by_patient[key] for key in sorted(records_by_patient)), tuple(str(path.resolve()) for path in paths))


def text_value(record: ClinicalRecord, key: str) -> str | None:
    return record.values.get(key)


def numeric_value(record: ClinicalRecord, key: str) -> float | None:
    value = text_value(record, key)
    if value is None:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed
