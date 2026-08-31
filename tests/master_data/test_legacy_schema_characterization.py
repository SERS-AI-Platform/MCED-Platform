from __future__ import annotations

import sqlite3
from pathlib import Path

from scripts.deployment.clinical_db_schema import migrate_schema

LEGACY_TABLES = {
    "audit_log",
    "predictions",
    "reports",
    "sessions",
    "spectra",
    "users",
}


def test_migration_preserves_legacy_tables_and_rows(tmp_path: Path) -> None:
    # Given: a version-3-shaped database with one legacy user row.
    database_path = tmp_path / "legacy.db"
    with sqlite3.connect(database_path) as connection:
        migrate_schema(connection)
        connection.execute(
            """INSERT INTO users (username, password_hash, role, display_name)
               VALUES ('legacy-user', 'synthetic-hash', 'clinician', 'Legacy User')"""
        )

    # When: the migration runs again.
    with sqlite3.connect(database_path) as connection:
        migrate_schema(connection)
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        legacy_user_count = connection.execute(
            "SELECT COUNT(*) FROM users WHERE username = 'legacy-user'"
        ).fetchone()[0]

    # Then: every legacy table and the existing row remain.
    assert LEGACY_TABLES <= tables
    assert legacy_user_count == 1


def test_migration_is_idempotent_for_legacy_schema(tmp_path: Path) -> None:
    # Given: a newly migrated clinical database.
    database_path = tmp_path / "clinical.db"
    with sqlite3.connect(database_path) as connection:
        migrate_schema(connection)
        first_schema = connection.execute(
            "SELECT name, sql FROM sqlite_master WHERE type IN ('table', 'index') ORDER BY name"
        ).fetchall()

    # When: the same migration is applied to stale state.
    with sqlite3.connect(database_path) as connection:
        migrate_schema(connection)
        second_schema = connection.execute(
            "SELECT name, sql FROM sqlite_master WHERE type IN ('table', 'index') ORDER BY name"
        ).fetchall()

    # Then: migration leaves the schema unchanged.
    assert second_schema == first_schema
