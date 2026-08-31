from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from scripts.deployment import clinical_db_results
from scripts.deployment.clinical_db_result_types import ClinicalPredictionResult
from scripts.deployment.clinical_db_schema import migrate_schema
from sers.master_data.dataset_manifest import build_dataset_manifest
from sers.master_data.lineage_types import (
    CreationProvenance,
    DatasetBuildRequest,
    DatasetManifestId,
    DatasetMemberDraft,
    DatasetPolicy,
    QCEvaluationDraft,
)
from sers.master_data.operational_lineage import (
    MissingLineageContextError,
    OperationalMeasurementMissingError,
    OperationalPredictionContext,
)
from sers.master_data.qc_lineage import append_qc_evaluation
from sers.master_data.spectrum_schema import initialize_spectrum_schema
from tests.master_data.lineage_test_support import seed_lineage


def _seed_operational_database(path: Path) -> tuple[str, str, OperationalPredictionContext]:
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        migrate_schema(connection)
        initialize_spectrum_schema(connection)
        connection.execute(
            """INSERT INTO users (username, password_hash, role, display_name)
               VALUES ('clinician', 'hash', 'clinician', 'Clinician')"""
        )
        connection.execute(
            """INSERT INTO sessions (
                   id, patient_id, age, sex, operating_mode, created_by, test_id
               ) VALUES ('session-1', 'legacy-patient', 60, 'F', 'screening', 1,
                         'TEST-1')"""
        )
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
                name="operational-input",
                members=(
                    DatasetMemberDraft(
                        seeded.measurement_id,
                        seeded.sample_label_id,
                        "inference",
                        0,
                    ),
                ),
                policy=DatasetPolicy(
                    "qc-policy-v1",
                    "spectrum-qc-v1",
                    ("pass",),
                ),
                preprocessing_version="baseline-v2",
                feature_schema_version="raman-grid-v3",
                decision_policy_version="threshold-v4",
                provenance=CreationProvenance(
                    "clinical-adapter",
                    "revision-1",
                    "operational prediction",
                ),
            ),
        )
    context = OperationalPredictionContext(
        manifest_id=manifest.id,
        model_name="screening-model",
        model_version="model-v5",
        preprocessing_version="baseline-v2",
        feature_schema_version="raman-grid-v3",
        decision_policy_version="threshold-v4",
        measurement_ids=(seeded.measurement_id,),
    )
    return "session-1", seeded.measurement_id, context


def _result(score: float = 0.91) -> ClinicalPredictionResult:
    return {
        "cancer_detected": True,
        "screening_index": score,
        "decision_level": "positive",
        "decision_policy": "threshold-v4",
    }


def test_operational_context_rejects_missing_version_at_boundary() -> None:
    # Given: an adapter request with a blank model version.
    # When/Then: construction fails before any database write can begin.
    with pytest.raises(MissingLineageContextError):
        OperationalPredictionContext(
            manifest_id=DatasetManifestId("manifest-id"),
            model_name="screening-model",
            model_version=" ",
            preprocessing_version="baseline-v2",
            feature_schema_version="raman-grid-v3",
            decision_policy_version="threshold-v4",
            measurement_ids=("measurement-id",),
        )


def test_operational_prediction_dual_write_is_atomic_and_append_only(
    tmp_path: Path,
) -> None:
    # Given: an operational session and an explicitly versioned master-data input.
    path = tmp_path / "clinical.db"
    session_id, measurement_id, context = _seed_operational_database(path)

    # When: the same operational inference is persisted twice.
    first = clinical_db_results.save_prediction_with_lineage(
        path, session_id, _result(), context
    )
    second = clinical_db_results.save_prediction_with_lineage(
        path, session_id, _result(), context
    )

    # Then: the projection remains singular while immutable runs and exact inputs append.
    with sqlite3.connect(path) as connection:
        assert first != second
        assert connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM prediction_runs").fetchone()[0] == 2
        assert tuple(
            row[0]
            for row in connection.execute(
                """SELECT measurement_id FROM prediction_input_measurements
                   ORDER BY prediction_run_id"""
            )
        ) == (measurement_id, measurement_id)
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM operational_session_links"
            ).fetchone()[0]
            == 1
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM operational_prediction_run_links"
            ).fetchone()[0]
            == 2
        )


def test_operational_prediction_failure_rolls_back_legacy_projection(
    tmp_path: Path,
) -> None:
    # Given: an explicit context that references a nonexistent measurement.
    path = tmp_path / "clinical.db"
    session_id, _, context = _seed_operational_database(path)
    invalid = OperationalPredictionContext(
        manifest_id=context.manifest_id,
        model_name=context.model_name,
        model_version=context.model_version,
        preprocessing_version=context.preprocessing_version,
        feature_schema_version=context.feature_schema_version,
        decision_policy_version=context.decision_policy_version,
        measurement_ids=("missing-measurement",),
    )

    # When: the atomic operational writer cannot resolve its exact input.
    with pytest.raises(OperationalMeasurementMissingError):
        clinical_db_results.save_prediction_with_lineage(
            path, session_id, _result(), invalid
        )

    # Then: neither side of the dual write was committed.
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM prediction_runs").fetchone()[0] == 0
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM operational_session_links"
            ).fetchone()[0]
            == 0
        )


def test_legacy_prediction_is_pending_then_backfill_is_idempotent(
    tmp_path: Path,
) -> None:
    # Given: a legacy caller that cannot supply truthful lineage context.
    path = tmp_path / "clinical.db"
    session_id, measurement_id, context = _seed_operational_database(path)
    clinical_db_results.save_prediction(path, session_id, _result())

    # When: an operator later supplies verified context twice.
    pending = clinical_db_results.list_pending_prediction_lineage(path)
    first = clinical_db_results.backfill_prediction_lineage(
        path, session_id, context
    )
    second = clinical_db_results.backfill_prediction_lineage(
        path, session_id, context
    )

    # Then: the legacy row was visible as pending and exactly one run was linked.
    assert len(pending) == 1
    assert pending[0].session_id == session_id
    assert first == second
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM prediction_runs").fetchone()[0] == 1
        assert connection.execute(
            "SELECT measurement_id FROM prediction_input_measurements"
        ).fetchone()[0] == measurement_id
        assert (
            connection.execute(
                """SELECT status FROM legacy_prediction_lineage_status
                   WHERE session_id = ?""",
                (session_id,),
            ).fetchone()[0]
            == "linked"
        )
