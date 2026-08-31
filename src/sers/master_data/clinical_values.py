from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime, time, timedelta
from typing import Final, Literal, TypeAlias

from ._compat import assert_never
from .clinical_normalization import normalize_date, safe_float
from .clinical_types import ClinicalObservationValue

FieldKind = Literal["numeric", "date", "text"]
ExternalCell: TypeAlias = str | int | float | bool | date | datetime | time | timedelta | None

_CANONICAL_FIELDS: Final[dict[str, tuple[str, str | None, FieldKind]]] = {
    "AGE": ("age", "years", "numeric"),
    "age": ("age", "years", "numeric"),
    "SEX": ("sex", None, "text"),
    "sex": ("sex", None, "text"),
    "DMHEI": ("height_cm", "cm", "numeric"),
    "HEIGHT": ("height_cm", "cm", "numeric"),
    "height_cm": ("height_cm", "cm", "numeric"),
    "DMWEI": ("weight_kg", "kg", "numeric"),
    "WEIGHT": ("weight_kg", "kg", "numeric"),
    "weight_kg": ("weight_kg", "kg", "numeric"),
    "LBORRES": ("lab_result", None, "numeric"),
    "LBDTC": ("lab_date", None, "date"),
    "URDTC": ("urinalysis_date", None, "date"),
    "sample_date": ("sample_date", None, "date"),
    "diagnosis_date": ("diagnosis_date", None, "date"),
    "group": ("source_group", None, "text"),
    "patient_code": ("subject_key", None, "text"),
    "SUBJID": ("subject_key", None, "text"),
    "스크리닝번호": ("subject_key", None, "text"),
}


def column_names(values: Iterable[ExternalCell]) -> tuple[str, ...]:
    counts: dict[str, int] = {}
    names: list[str] = []
    for index, value in enumerate(values, start=1):
        base = f"column_{index}" if value is None or str(value) == "" else str(value)
        count = counts.get(base, 0) + 1
        counts[base] = count
        names.append(base if count == 1 else f"{base}#{count}")
    return tuple(names)


def raw_text(value: ExternalCell) -> str:
    match value:
        case datetime() | date() | time():
            return value.isoformat()
        case timedelta():
            return str(value)
        case bool():
            return "true" if value else "false"
        case int() | float() | str():
            return str(value)
        case None:
            return ""
        case unreachable:
            assert_never(unreachable)


def observation(field_name: str, raw_value: str) -> ClinicalObservationValue:
    metadata = _CANONICAL_FIELDS.get(field_name)
    if metadata is None:
        return ClinicalObservationValue(
            source_field_name=field_name,
            canonical_code=None,
            raw_value=raw_value,
            text_value=raw_value,
            numeric_value=None,
            date_value=None,
            unit=None,
            normalization_status="unmapped",
        )
    canonical_code, unit, kind = metadata
    match kind:
        case "numeric":
            numeric_parsed = safe_float(raw_value)
            numeric_value = (
                float(numeric_parsed)
                if isinstance(numeric_parsed, (int, float))
                else None
            )
            return ClinicalObservationValue(
                field_name,
                canonical_code,
                raw_value,
                raw_value if numeric_value is None else None,
                numeric_value,
                None,
                unit,
                "raw" if numeric_value is None else "normalized",
            )
        case "date":
            date_parsed = normalize_date(raw_value)
            date_value = (
                date_parsed if isinstance(date_parsed, str) and date_parsed else None
            )
            return ClinicalObservationValue(
                field_name,
                canonical_code,
                raw_value,
                raw_value if date_value is None else None,
                None,
                date_value,
                unit,
                "raw" if date_value is None else "normalized",
            )
        case "text":
            return ClinicalObservationValue(
                field_name,
                canonical_code,
                raw_value,
                raw_value,
                None,
                None,
                unit,
                "normalized",
            )
        case unreachable:
            assert_never(unreachable)
