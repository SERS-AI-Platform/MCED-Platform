"""Idempotent SQLite schema and migrations for the clinical application."""

from __future__ import annotations

import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Final

from sers.master_data import (
    _clinical_schema,
    _core_schema,
    _lineage_schema,
    _manifest_schema,
)
from sers.master_data._identifier_schema_migration import migrate_identifiers_v7
from sers.master_data._schema_simplification import simplify_legacy_schema

from .clinical_db_lineage_migration import seed_legacy_prediction_pending_status
from .clinical_db_operational_schema import OPERATIONAL_LINEAGE_SCHEMA
from .clinical_qc import DEFAULT_MIN_VALID_COUNT, build_qc_summary

SCHEMA_VERSION: Final = 7
SAFE_ID_CHARACTER: Final = re.compile(r"[^A-Za-z0-9_-]")
MASTER_COLUMN_MIGRATIONS: Final = (
    ("dataset_manifest_items", _manifest_schema.ITEM_COLUMNS),
    ("qc_evaluations", _manifest_schema.QC_COLUMNS),
    ("prediction_runs", _manifest_schema.PREDICTION_COLUMNS),
    ("clinical_events", _clinical_schema.EVENT_COLUMNS),
    ("clinical_observations", _clinical_schema.OBSERVATION_COLUMNS),
)


@dataclass(frozen=True, slots=True)
class SchemaMigrationTransactionError(RuntimeError):
    transaction_owner: str = "caller"

    def __str__(self) -> str:
        return f"schema migration requires a clean connection; owned by {self.transaction_owner}"

SCHEMA: Final = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL, role TEXT NOT NULL CHECK(role IN ('clinician', 'admin')),
    display_name TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT 1
);
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY, patient_id TEXT NOT NULL, age REAL NOT NULL,
    sex TEXT NOT NULL CHECK(sex IN ('M', 'F')), bmi REAL,
    operating_mode TEXT NOT NULL CHECK(operating_mode IN ('screening', 'balanced', 'confirmatory')),
    model_variant TEXT, threshold REAL, status TEXT DEFAULT 'created'
        CHECK(status IN ('created', 'uploaded', 'qc_done', 'analyzed', 'reported')),
    created_by INTEGER REFERENCES users(id), created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP, test_id TEXT, test_date TEXT, test_sequence INTEGER,
    owner_user_id INTEGER REFERENCES users(id), retest_of_session_id TEXT REFERENCES sessions(id),
    qc_valid BOOLEAN
);
CREATE TABLE IF NOT EXISTS spectra (
    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL REFERENCES sessions(id),
    filename TEXT NOT NULL, qc_pass BOOLEAN, qc_flags TEXT,
    replicate_correlation REAL, fp_mean REAL, uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT UNIQUE NOT NULL REFERENCES sessions(id),
    cancer_detected BOOLEAN NOT NULL,
    decision_level TEXT CHECK(decision_level IN ('negative', 'moderate', 'positive')),
    decision_policy TEXT, screening_index REAL NOT NULL, majority_vote TEXT,
    cancer_type_prediction TEXT, cancer_type_confidence REAL,
    cancer_type_probabilities TEXT, per_replicate_results TEXT, result_json TEXT,
    predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS reports (
    id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id),
    generated_by INTEGER REFERENCES users(id), generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP, user_id INTEGER,
    session_id TEXT, action TEXT NOT NULL, detail TEXT
);
"""

SESSION_COLUMNS: Final = {
    "test_id": "TEXT",
    "test_date": "TEXT",
    "test_sequence": "INTEGER",
    "owner_user_id": "INTEGER REFERENCES users(id)",
    "retest_of_session_id": "TEXT REFERENCES sessions(id)",
    "qc_valid": "BOOLEAN",
}

PREDICTION_COLUMNS: Final = {
    "decision_level": "TEXT CHECK(decision_level IN ('negative', 'moderate', 'positive'))",
    "decision_policy": "TEXT",
}

INDEXES: Final = (
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_sessions_test_id_unique ON sessions(test_id)",
    "CREATE INDEX IF NOT EXISTS idx_sessions_owner_created ON sessions(owner_user_id, created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_sessions_patient_date ON sessions(patient_id, test_date)",
    "CREATE INDEX IF NOT EXISTS idx_sessions_test_date ON sessions(test_date)",
    "CREATE INDEX IF NOT EXISTS idx_spectra_session ON spectra(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_reports_session_created ON reports(session_id, generated_at DESC)",
)


def migrate_schema(connection: sqlite3.Connection) -> None:
    """Create or upgrade the database without replacing historical rows."""
    if connection.in_transaction:
        raise SchemaMigrationTransactionError

    foreign_keys_enabled = connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    migrate_user_roles = _requires_user_role_migration(connection)
    if foreign_keys_enabled:
        connection.execute("PRAGMA foreign_keys = OFF")
    try:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            simplify_legacy_schema(connection, keep_operational_links=True)
            _execute_script(connection, SCHEMA)
            _execute_script(connection, _core_schema.CORE_SCHEMA)
            migrate_identifiers_v7(connection)
            _execute_script(connection, _lineage_schema.LINEAGE_SCHEMA)
            _execute_script(connection, OPERATIONAL_LINEAGE_SCHEMA)
            for table, columns in MASTER_COLUMN_MIGRATIONS:
                _add_missing_columns(connection, table, columns)
            _execute_script(connection, _manifest_schema.MANIFEST_SCHEMA)
            _execute_script(connection, _clinical_schema.CLINICAL_SCHEMA)
            if migrate_user_roles:
                _migrate_user_roles(connection)
            _add_missing_columns(connection, "sessions", SESSION_COLUMNS)
            _add_missing_columns(connection, "predictions", PREDICTION_COLUMNS)
            connection.execute(
                """UPDATE predictions
                   SET decision_level = CASE
                           WHEN cancer_detected = 1 THEN 'positive' ELSE 'negative'
                       END,
                       decision_policy = 'fixed_three_replicate_threshold_v1'
                   WHERE decision_level IS NULL OR decision_policy IS NULL"""
            )
            _backfill_session_identity(connection)
            _backfill_qc_validity(connection)
            seed_legacy_prediction_pending_status(connection)
            for statement in INDEXES:
                connection.execute(statement)
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    finally:
        if foreign_keys_enabled:
            connection.execute("PRAGMA foreign_keys = ON")


def _execute_script(connection: sqlite3.Connection, script: str) -> None:
    statement = ""
    for line in script.splitlines(keepends=True):
        statement += line
        if sqlite3.complete_statement(statement):
            connection.execute(statement)
            statement = ""
    if statement.strip():
        connection.execute(statement)


def _add_missing_columns(
    connection: sqlite3.Connection,
    table: str,
    columns: dict[str, str],
) -> None:
    existing = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
    for name, definition in columns.items():
        if name not in existing:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")


def _requires_user_role_migration(connection: sqlite3.Connection) -> bool:
    row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'users'"
    ).fetchone()
    return row is not None and "'technician'" in row[0]


def _migrate_user_roles(connection: sqlite3.Connection) -> None:
    connection.execute("UPDATE users SET role = 'clinician' WHERE role = 'technician'")
    _execute_script(
        connection,
        """
        CREATE TABLE users_migrated (
            id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('clinician', 'admin')),
            display_name TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT 1
        );
        INSERT INTO users_migrated
            (id, username, password_hash, role, display_name, created_at, is_active)
        SELECT id, username, password_hash, role, display_name, created_at, is_active
        FROM users;
        DROP TABLE users;
        ALTER TABLE users_migrated RENAME TO users;
        """,
    )


def _backfill_session_identity(connection: sqlite3.Connection) -> None:
    rows = connection.execute(
        """SELECT id, patient_id, created_by, created_at, test_id, test_date,
                  test_sequence, owner_user_id
           FROM sessions
           ORDER BY created_at, id"""
    ).fetchall()
    used_test_ids = {row[4] for row in rows if row[4] is not None}
    next_sequence: defaultdict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        test_date = row[5] or _test_date_from_timestamp(row[3])
        group = (row[1], test_date)
        if row[6] is not None:
            next_sequence[group] = max(next_sequence[group], int(row[6]))

    for row in rows:
        session_id, patient_id, created_by = row[0], row[1], row[2]
        test_date = row[5] or _test_date_from_timestamp(row[3])
        group = (patient_id, test_date)
        sequence = row[6]
        if sequence is None:
            next_sequence[group] += 1
            sequence = next_sequence[group]
        test_id = row[4]
        if test_id is None:
            safe_patient_id = SAFE_ID_CHARACTER.sub("_", patient_id).strip("_-")
            safe_patient_id = safe_patient_id or f"LEGACY_{session_id[:8]}"
            candidate = f"{safe_patient_id}-{test_date.replace('-', '')}-{sequence:02d}"
            if candidate in used_test_ids:
                candidate = f"{candidate}-{session_id[:8]}"
            test_id = candidate
            used_test_ids.add(test_id)
        owner_user_id = row[7] if row[7] is not None else created_by
        connection.execute(
            """UPDATE sessions
               SET test_id = ?, test_date = ?, test_sequence = ?, owner_user_id = ?
               WHERE id = ?""",
            (test_id, test_date, sequence, owner_user_id, session_id),
        )


def _test_date_from_timestamp(raw_timestamp: str | None) -> str:
    if raw_timestamp:
        return raw_timestamp[:10]
    return datetime.now().date().isoformat()


def _backfill_qc_validity(connection: sqlite3.Connection) -> None:
    session_ids = connection.execute("SELECT id FROM sessions WHERE qc_valid IS NULL").fetchall()
    for (session_id,) in session_ids:
        spectra = [
            dict(row)
            for row in connection.execute(
                "SELECT qc_pass, qc_flags FROM spectra WHERE session_id = ?",
                (session_id,),
            ).fetchall()
        ]
        if spectra:
            summary = build_qc_summary(spectra, DEFAULT_MIN_VALID_COUNT)
            connection.execute(
                "UPDATE sessions SET qc_valid = ? WHERE id = ?",
                (summary["valid"], session_id),
            )
