from __future__ import annotations

import sqlite3
from typing import Final

from .transaction import require_clean_connection

SPECTRUM_SCHEMA: Final = """
CREATE TABLE IF NOT EXISTS spectrum_inventory_records (
    source_asset_id TEXT PRIMARY KEY REFERENCES source_assets(id),
    source_kind TEXT NOT NULL,
    status TEXT NOT NULL CHECK(
        status IN ('ready', 'derived', 'excluded', 'quarantined')
    ),
    reason_code TEXT,
    acquisition_date TEXT NOT NULL,
    instrument_key TEXT NOT NULL,
    root_key TEXT NOT NULL,
    canonical_source_code TEXT,
    preparation TEXT,
    fasting_state TEXT NOT NULL DEFAULT 'unspecified' CHECK(
        fasting_state IN ('fasting', 'non_fasting', 'unspecified')
    ),
    specimen_timing TEXT NOT NULL DEFAULT 'unspecified' CHECK(
        specimen_timing IN ('post_operative', 'unspecified')
    ),
    lot_code TEXT,
    identity_status TEXT,
    alias_candidates TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS spectrum_material_metadata (
    analytical_material_id TEXT PRIMARY KEY REFERENCES analytical_materials(id),
    site_id TEXT NOT NULL REFERENCES sites(id),
    material_key TEXT NOT NULL,
    canonical_source_code TEXT NOT NULL,
    source_group TEXT NOT NULL,
    material_kind TEXT NOT NULL CHECK(
        material_kind IN ('biological', 'blank', 'reference', 'matrix_blank')
    ),
    preparation TEXT NOT NULL,
    fasting_state TEXT NOT NULL DEFAULT 'unspecified' CHECK(
        fasting_state IN ('fasting', 'non_fasting', 'unspecified')
    ),
    specimen_timing TEXT NOT NULL DEFAULT 'unspecified' CHECK(
        specimen_timing IN ('post_operative', 'unspecified')
    ),
    lot_code TEXT,
    identity_status TEXT NOT NULL CHECK(
        identity_status IN ('canonical', 'alias_review')
    ),
    alias_candidates TEXT NOT NULL DEFAULT '',
    UNIQUE(site_id, material_key)
);
"""

_SPECTRUM_COLUMNS: Final = {
    "fasting_state": (
        "TEXT NOT NULL DEFAULT 'unspecified' "
        "CHECK(fasting_state IN ('fasting', 'non_fasting', 'unspecified'))"
    ),
    "specimen_timing": (
        "TEXT NOT NULL DEFAULT 'unspecified' "
        "CHECK(specimen_timing IN ('post_operative', 'unspecified'))"
    ),
    "lot_code": "TEXT",
}


def _add_spectrum_columns(
    connection: sqlite3.Connection,
    table: str,
) -> None:
    existing = {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")}
    for name, definition in _SPECTRUM_COLUMNS.items():
        if name not in existing:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")


def initialize_spectrum_schema(connection: sqlite3.Connection) -> None:
    """Initialize spectrum tables only outside caller-owned transactions."""
    require_clean_connection(connection, "spectrum schema initialization")
    connection.executescript(SPECTRUM_SCHEMA)
    _add_spectrum_columns(connection, "spectrum_inventory_records")
    _add_spectrum_columns(connection, "spectrum_material_metadata")
