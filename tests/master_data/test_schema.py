from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from scripts.deployment.clinical_db_schema import (
    SchemaMigrationTransactionError,
    migrate_schema,
)

MASTER_TABLES = {
    "analytical_materials",
    "clinical_events",
    "clinical_observations",
    "dataset_manifest_items",
    "dataset_manifests",
    "ingest_batches",
    "match_candidates",
    "match_resolutions",
    "measurement_artifacts",
    "measurement_runs",
    "measurements",
    "operational_session_links",
    "operational_prediction_run_links",
    "prediction_input_measurements",
    "prediction_runs",
    "legacy_prediction_lineage_status",
    "qc_evaluations",
    "sample_label_evidence",
    "sample_labels",
    "samples",
    "sites",
    "source_assets",
    "subjects",
}


def test_clinical_migration_adds_master_tables_at_version_seven(tmp_path: Path) -> None:
    # Given: an empty clinical database.
    database_path = tmp_path / "clinical.db"

    # When: the clinical migration runs.
    with sqlite3.connect(database_path) as connection:
        migrate_schema(connection)
        table_names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        schema_version = connection.execute("PRAGMA user_version").fetchone()[0]

    # Then: all master tables exist and the schema advertises version seven.
    assert MASTER_TABLES <= table_names
    assert schema_version == 7


def test_migration_rejects_caller_owned_transaction_without_committing(
    tmp_path: Path,
) -> None:
    # Given: a migrated database and an uncommitted caller-owned insert.
    database_path = tmp_path / "clinical.db"
    with sqlite3.connect(database_path) as connection:
        migrate_schema(connection)
        schema_version = connection.execute("PRAGMA user_version").fetchone()[0]
        connection.execute(
            """INSERT INTO users (username, password_hash, role, display_name)
               VALUES ('caller-owned', 'hash', 'clinician', 'Caller Owned')"""
        )

        # When: migration is requested on the caller-owned transaction.
        with pytest.raises(SchemaMigrationTransactionError):
            migrate_schema(connection)
        connection.rollback()

        # Then: the caller can still roll back its insert and schema version is unchanged.
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM users WHERE username = 'caller-owned'"
            ).fetchone()[0]
            == 0
        )
        assert connection.execute("PRAGMA user_version").fetchone()[0] == schema_version


def test_failed_migration_rolls_back_all_schema_changes(tmp_path: Path) -> None:
    # Given: an incompatible pre-existing table that will fail the migration.
    database_path = tmp_path / "clinical.db"
    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY)")
        connection.commit()

        # When: migration fails after attempting additive schema work.
        with pytest.raises(sqlite3.OperationalError):
            migrate_schema(connection)

        # Then: no new table, column, or schema version survives the failure.
        table_names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        session_columns = [
            row[1] for row in connection.execute("PRAGMA table_info(sessions)")
        ]
        schema_version = connection.execute("PRAGMA user_version").fetchone()[0]

    assert table_names == {"sessions"}
    assert session_columns == ["id"]
    assert schema_version == 0


def test_migration_preserves_multi_statement_trigger_scripts(tmp_path: Path) -> None:
    # Given: a cleanly migrated database containing an append-only prediction run.
    database_path = tmp_path / "clinical.db"
    with sqlite3.connect(database_path) as connection:
        migrate_schema(connection)
        connection.execute(
            """INSERT INTO prediction_runs (id, model_name, model_version, status)
               VALUES ('run-1', 'screening', 'v1', 'completed')"""
        )

        # When: an update crosses the trigger boundary.
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "UPDATE prediction_runs SET model_version = 'v2' WHERE id = 'run-1'"
            )

        # Then: the complete trigger body ran and the immutable value remains.
        model_version = connection.execute(
            "SELECT model_version FROM prediction_runs WHERE id = 'run-1'"
        ).fetchone()[0]

    assert model_version == "v1"


def test_master_schema_enforces_domain_checks_and_foreign_keys(tmp_path: Path) -> None:
    # Given: a migrated connection with foreign-key enforcement enabled.
    database_path = tmp_path / "clinical.db"
    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        migrate_schema(connection)

        # When: malformed domain and orphan records are inserted.
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """INSERT INTO sites (id, code, name, status)
                   VALUES ('site-1', 'SITE', 'Synthetic Site', 'unknown')"""
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """INSERT INTO samples (id, subject_id, site_id, sample_type)
                   VALUES ('sample-1', 'missing-subject', 'missing-site', 'urine')"""
            )

        # Then: neither malformed record is stored.
        assert connection.execute("SELECT COUNT(*) FROM sites").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0] == 0


@pytest.mark.parametrize("invalid_hash", [" " * 64, "z" * 64])
def test_source_asset_hash_requires_hexadecimal_characters(
    tmp_path: Path,
    invalid_hash: str,
) -> None:
    # Given: a migrated database with one valid site.
    database_path = tmp_path / "clinical.db"
    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        migrate_schema(connection)
        connection.execute(
            """INSERT INTO sites (id, code, name)
               VALUES ('site-1', 'SITE', 'Synthetic Site')"""
        )

        # When: a 64-character non-hex hash crosses the SQLite boundary.
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """INSERT INTO source_assets (
                       id, site_id, uri, sha256, asset_kind
                   ) VALUES ('asset-1', 'site-1', 'synthetic://asset', ?, 'clinical')""",
                (invalid_hash,),
            )

        # Then: no malformed source asset is persisted.
        assert connection.execute("SELECT COUNT(*) FROM source_assets").fetchone()[0] == 0


def test_clinical_tables_expose_raw_row_and_field_lineage(tmp_path: Path) -> None:
    # Given: a freshly migrated v4 clinical database.
    database_path = tmp_path / "clinical.db"
    with sqlite3.connect(database_path) as connection:
        migrate_schema(connection)

        # When: clinical table columns are inspected.
        event_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(clinical_events)")
        }
        observation_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(clinical_observations)")
        }

    # Then: every event and observation can retain its raw source locator/value.
    assert {"source_asset_id", "source_row_locator"} <= event_columns
    assert {
        "source_field_name",
        "canonical_code",
        "raw_value",
        "date_value",
        "normalization_status",
    } <= observation_columns
