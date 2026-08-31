from __future__ import annotations

import sqlite3
from typing import Final

CLINICAL_SCHEMA: Final = """
CREATE TABLE IF NOT EXISTS subject_source_aliases (
    subject_id TEXT NOT NULL REFERENCES subjects(id),
    source_asset_id TEXT NOT NULL REFERENCES source_assets(id),
    source_field_name TEXT NOT NULL CHECK(length(source_field_name) > 0),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(subject_id, source_asset_id)
);

CREATE TABLE IF NOT EXISTS clinical_source_metadata (
    source_asset_id TEXT PRIMARY KEY REFERENCES source_assets(id),
    protocol_code TEXT NOT NULL CHECK(length(protocol_code) > 0),
    source_group TEXT NOT NULL CHECK(length(source_group) > 0),
    canonical_alias TEXT,
    identity_status TEXT NOT NULL CHECK(
        identity_status IN ('canonical', 'alias_review')
    ),
    source_version TEXT NOT NULL CHECK(
        source_version IN ('current', 'historical')
    )
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_clinical_event_source_row
    ON clinical_events(source_asset_id, source_row_locator);
CREATE UNIQUE INDEX IF NOT EXISTS idx_clinical_observation_source_field
    ON clinical_observations(clinical_event_id, source_field_name);
"""

EVENT_COLUMNS: Final = {
    "source_asset_id": "TEXT REFERENCES source_assets(id)",
    "source_row_locator": "TEXT",
}

OBSERVATION_COLUMNS: Final = {
    "source_field_name": "TEXT",
    "canonical_code": "TEXT",
    "raw_value": "TEXT",
    "date_value": "TEXT",
    "normalization_status": (
        "TEXT CHECK(normalization_status IN ('normalized', 'raw', 'unmapped'))"
    ),
}


def _add_missing_columns(
    connection: sqlite3.Connection,
    table: str,
    columns: dict[str, str],
) -> None:
    existing = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
    for name, definition in columns.items():
        if name not in existing:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")


def initialize_clinical_schema(connection: sqlite3.Connection) -> None:
    _add_missing_columns(connection, "clinical_events", EVENT_COLUMNS)
    _add_missing_columns(connection, "clinical_observations", OBSERVATION_COLUMNS)
    connection.executescript(CLINICAL_SCHEMA)
