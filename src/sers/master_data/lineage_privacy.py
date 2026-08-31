from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import NoReturn, TypeAlias

from .lineage_types import PredictionRunDraft

JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)


def assert_never(value: NoReturn) -> NoReturn:
    raise AssertionError


@dataclass(frozen=True, slots=True)
class SensitivePredictionMetadataError(ValueError):
    field: str

    def __str__(self) -> str:
        return f"prediction metadata field {self.field} contains a source identifier"


def _known_source_identifiers(
    connection: sqlite3.Connection,
) -> frozenset[str]:
    identifiers = {
        str(row[0])
        for row in connection.execute(
            """SELECT patient_id FROM subjects
               WHERE patient_id IS NOT NULL
               UNION
               SELECT solum_label FROM samples
               WHERE solum_label IS NOT NULL"""
        )
        if row[0] is not None and str(row[0])
    }
    table_names = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    if "spectrum_material_metadata" in table_names:
        identifiers.update(
            str(row[0])
            for row in connection.execute(
                """SELECT canonical_source_code
                   FROM spectrum_material_metadata"""
            )
            if row[0] is not None and str(row[0])
        )
    return frozenset(identifiers)


def _contains_identifier(
    value: JsonValue,
    identifiers: frozenset[str],
) -> bool:
    def contains_text(text: str) -> bool:
        return any(identifier in text for identifier in identifiers)

    match value:
        case str() as text:
            return contains_text(text)
        case bool() | None:
            return False
        case int() | float() as number:
            return contains_text(str(number))
        case list() as items:
            return any(_contains_identifier(item, identifiers) for item in items)
        case dict() as mapping:
            return any(
                contains_text(key) or _contains_identifier(item, identifiers)
                for key, item in mapping.items()
            )
        case unreachable:
            assert_never(unreachable)


def validate_prediction_privacy(
    connection: sqlite3.Connection,
    draft: PredictionRunDraft,
    output: JsonValue,
) -> None:
    identifiers = _known_source_identifiers(connection)
    metadata = (
        ("model_name", draft.model_name),
        ("model_version", draft.model_version),
        ("preprocessing_version", draft.preprocessing_version),
        ("feature_schema_version", draft.feature_schema_version),
        ("decision_policy_version", draft.decision_policy_version),
    )
    for field, value in metadata:
        if any(identifier in value for identifier in identifiers):
            raise SensitivePredictionMetadataError(field)
    if _contains_identifier(output, identifiers):
        raise SensitivePredictionMetadataError("output_json")
