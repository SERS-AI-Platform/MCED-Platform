from __future__ import annotations

import sqlite3
from pathlib import Path

from scripts.deployment import clinical_db_results
from scripts.deployment.clinical_db_schema import migrate_schema
from sers.master_data.dataset_manifest import build_dataset_manifest
from sers.master_data.lineage_types import (
    CreationProvenance,
    DatasetBuildRequest,
    DatasetMemberDraft,
    DatasetPolicy,
    QCEvaluationDraft,
)
from sers.master_data.operational_lineage import OperationalPredictionContext
from sers.master_data.qc_lineage import append_qc_evaluation
from sers.master_data.spectrum_schema import initialize_spectrum_schema
from tests.master_data.lineage_test_support import seed_lineage

LEGACY_V3_SCHEMA = """
CREATE TABLE users (
    id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL, role TEXT NOT NULL, display_name TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP, is_active INTEGER DEFAULT 1
);
CREATE TABLE sessions (
    id TEXT PRIMARY KEY, patient_id TEXT NOT NULL, age REAL NOT NULL, sex TEXT NOT NULL,
    bmi REAL, operating_mode TEXT NOT NULL, model_variant TEXT, threshold REAL,
    status TEXT DEFAULT 'created', created_by INTEGER, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    completed_at TEXT, test_id TEXT, test_date TEXT, test_sequence INTEGER,
    owner_user_id INTEGER, retest_of_session_id TEXT, qc_valid INTEGER
);
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT UNIQUE NOT NULL,
    cancer_detected INTEGER NOT NULL, screening_index REAL NOT NULL,
    majority_vote TEXT, cancer_type_prediction TEXT, cancer_type_confidence REAL,
    cancer_type_probabilities TEXT, per_replicate_results TEXT, result_json TEXT,
    predicted_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


def _create_v3_database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(LEGACY_V3_SCHEMA)
        connection.execute(
            """INSERT INTO users (
                   id, username, password_hash, role, display_name
               ) VALUES (1, 'legacy', 'hash', 'clinician', 'Legacy')"""
        )
        connection.executemany(
            """INSERT INTO sessions (
                   id, patient_id, age, sex, operating_mode, created_by,
                   test_id, test_date, test_sequence, owner_user_id
               ) VALUES (?, ?, 60, 'F', 'screening', 1, ?, '2026-01-01', 1, 1)""",
            (
                ("legacy-1", "patient-1", "LEGACY-1"),
                ("legacy-2", "patient-2", "LEGACY-2"),
            ),
        )
        connection.executemany(
            """INSERT INTO predictions (
                   session_id, cancer_detected, screening_index, result_json
               ) VALUES (?, ?, ?, ?)""",
            (
                ("legacy-1", 0, 0.1, '{"screening_index":0.1}'),
                ("legacy-2", 1, 0.9, '{"screening_index":0.9}'),
            ),
        )
        connection.execute("PRAGMA user_version = 3")


def _seed_verified_context(
    connection: sqlite3.Connection,
) -> OperationalPredictionContext:
    initialize_spectrum_schema(connection)
    seeded = seed_lineage(connection, "SITE-A", "MASTER-SUBJECT")
    append_qc_evaluation(
        connection,
        QCEvaluationDraft(
            seeded.measurement_id,
            "spectrum-qc-v1",
            "engine-v1",
            "pass",
            "{}",
        ),
    )
    manifest = build_dataset_manifest(
        connection,
        DatasetBuildRequest(
            name="legacy-operational-input",
            members=(
                DatasetMemberDraft(
                    seeded.measurement_id,
                    seeded.sample_label_id,
                    "inference",
                    0,
                ),
            ),
            policy=DatasetPolicy("qc-policy-v1", "spectrum-qc-v1", ("pass",)),
            preprocessing_version="baseline-v1",
            feature_schema_version="raman-grid-v1",
            decision_policy_version="threshold-v1",
            provenance=CreationProvenance("migration-test", "v3", "backfill"),
        ),
    )
    return OperationalPredictionContext(
        manifest.id,
        "screening-model",
        "model-v1",
        "baseline-v1",
        "raman-grid-v1",
        "threshold-v1",
        (seeded.measurement_id,),
    )


def test_v3_predictions_migrate_to_truthful_pending_and_backfill_one(
    tmp_path: Path,
) -> None:
    # Given: a true v3 database with two preserved legacy predictions.
    path = tmp_path / "legacy-v3.db"
    _create_v3_database(path)

    # When: v5 migration runs twice and one verified context is backfilled.
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        migrate_schema(connection)
        migrate_schema(connection)
        context = _seed_verified_context(connection)
        preserved = connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        pending = connection.execute(
            """SELECT COUNT(*) FROM legacy_prediction_lineage_status
               WHERE status = 'pending_context'"""
        ).fetchone()[0]
        run_links = connection.execute(
            "SELECT COUNT(*) FROM operational_prediction_run_links"
        ).fetchone()[0]
    clinical_db_results.backfill_prediction_lineage(path, "legacy-1", context)

    # Then: migration fabricated no lineage and verified backfill links only its target.
    assert (preserved, pending, run_links) == (2, 2, 0)
    with sqlite3.connect(path) as connection:
        statuses = tuple(
            row[0]
            for row in connection.execute(
                """SELECT status FROM legacy_prediction_lineage_status
                   ORDER BY session_id"""
            )
        )
        assert statuses == ("linked", "pending_context")
        assert connection.execute(
            "SELECT COUNT(*) FROM operational_prediction_run_links"
        ).fetchone()[0] == 1
