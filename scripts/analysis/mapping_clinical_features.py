from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openpyxl

PRIMARY_FEATURES = (
    "psa_raw",
    "ua_ph",
    "ua_sg",
    "microscopy_wbc",
    "microscopy_rbc",
)
LEAKAGE_FIELDS = (
    "cohort_group",
    "cancer_type",
    "gleason_score",
    "grade_group",
    "overall_stage",
)


@dataclass(frozen=True, slots=True)
class ClinicalTable:
    feature_names: tuple[str, ...]
    values: np.ndarray
    groups: tuple[str, ...]
    metadata: dict[str, object]
    summary_rows: tuple[dict[str, object], ...]


def _parse_measurement(value: object) -> float:
    if value is None:
        return float("nan")
    text = str(value).strip().replace(",", "")
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        pass
    if text.startswith(">="):
        try:
            return float(text[2:])
        except ValueError:
            return float("nan")
    if text.startswith("<"):
        try:
            return float(text[1:]) / 2.0
        except ValueError:
            return float("nan")
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", text)
    if match is not None:
        return (float(match.group(1)) + float(match.group(2))) / 2.0
    return float("nan")


def _numeric_summary(values: np.ndarray) -> dict[str, float | int]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return {"n": 0, "missing": int(len(values)), "median": float("nan"), "q25": float("nan"), "q75": float("nan")}
    return {
        "n": int(len(finite)),
        "missing": int(len(values) - len(finite)),
        "median": float(np.median(finite)),
        "q25": float(np.quantile(finite, 0.25)),
        "q75": float(np.quantile(finite, 0.75)),
    }


def load_clinical_table(path: Path) -> ClinicalTable:
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows = list(workbook.active.iter_rows(values_only=True))
    headers = tuple(str(value) for value in rows[0] if value is not None)
    records = [dict(zip(headers, row)) for row in rows[1:]]
    groups = tuple(str(record["cohort_group"]) for record in records)
    values = np.asarray(
        [[_parse_measurement(record.get(name)) for name in PRIMARY_FEATURES] for record in records],
        dtype=np.float32,
    )
    summary_rows: list[dict[str, object]] = []
    for group in ("control", "prostate disease control", "prostate"):
        mask = np.asarray([item == group for item in groups], dtype=bool)
        for index, name in enumerate(PRIMARY_FEATURES):
            summary = _numeric_summary(values[mask, index])
            summary_rows.append({"group": group, "feature": name, **summary})
    constant_fields = []
    for name in headers:
        distinct = {str(record.get(name)) for record in records}
        if len(distinct) <= 1:
            constant_fields.append(name)
    metadata: dict[str, object] = {
        "row_count": len(records),
        "columns": list(headers),
        "primary_features": list(PRIMARY_FEATURES),
        "leakage_fields_excluded": list(LEAKAGE_FIELDS),
        "group_counts": {group: groups.count(group) for group in ("control", "prostate disease control", "prostate")},
        "constant_fields": constant_fields,
        "feature_missing_counts": {
            name: int(np.isnan(values[:, index]).sum()) for index, name in enumerate(PRIMARY_FEATURES)
        },
        "date_field_used": False,
        "site_field_used": False,
        "sex_field_used": False,
    }
    return ClinicalTable(tuple(PRIMARY_FEATURES), values, groups, metadata, tuple(summary_rows))


def clinical_feature_names() -> tuple[str, ...]:
    return PRIMARY_FEATURES
