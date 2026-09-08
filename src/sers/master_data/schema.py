from __future__ import annotations

import sqlite3
from typing import Final

from ._clinical_schema import (
    CLINICAL_SCHEMA,
    EVENT_COLUMNS,
    OBSERVATION_COLUMNS,
    _add_missing_columns,
)
from ._core_schema import CORE_SCHEMA
from ._identifier_schema_migration import (
    AmbiguousSolumLabelError,
    migrate_identifiers_v7,
)
from ._lineage_schema import LINEAGE_SCHEMA
from ._manifest_schema import (
    ITEM_COLUMNS,
    MANIFEST_SCHEMA,
    PREDICTION_COLUMNS,
)
from ._schema_simplification import (
    AmbiguousSubjectIdentityError,
    simplify_legacy_schema,
)
from .transaction import require_clean_connection

MASTER_SCHEMA_VERSION: Final = 8


def _execute_script(connection: sqlite3.Connection, script: str) -> None:
    statement = ""
    for line in script.splitlines(keepends=True):
        statement += line
        if sqlite3.complete_statement(statement):
            connection.execute(statement)
            statement = ""
    if statement.strip():
        connection.execute(statement)


def initialize_schema(connection: sqlite3.Connection) -> None:
    """Create or simplify the master schema on a clean connection."""
    require_clean_connection(connection, "master-data schema initialization")
    foreign_keys_enabled = connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    if foreign_keys_enabled:
        connection.execute("PRAGMA foreign_keys = OFF")
    try:
        connection.execute("BEGIN IMMEDIATE")
        simplify_legacy_schema(connection, keep_operational_links=False)
        _execute_script(connection, CORE_SCHEMA)
        migrate_identifiers_v7(connection)
        _execute_script(connection, LINEAGE_SCHEMA)
        _add_missing_columns(connection, "dataset_manifest_items", ITEM_COLUMNS)
        _add_missing_columns(connection, "prediction_runs", PREDICTION_COLUMNS)
        _add_missing_columns(connection, "clinical_events", EVENT_COLUMNS)
        _add_missing_columns(
            connection,
            "clinical_observations",
            OBSERVATION_COLUMNS,
        )
        _execute_script(connection, MANIFEST_SCHEMA)
        _execute_script(connection, CLINICAL_SCHEMA)
        connection.execute(f"PRAGMA user_version = {MASTER_SCHEMA_VERSION}")
        connection.commit()
    except (
        sqlite3.Error,
        AmbiguousSolumLabelError,
        AmbiguousSubjectIdentityError,
    ):
        connection.rollback()
        raise
    finally:
        if foreign_keys_enabled:
            connection.execute("PRAGMA foreign_keys = ON")
