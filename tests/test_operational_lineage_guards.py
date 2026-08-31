from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from scripts.deployment import clinical_db_results
from sers.master_data.operational_lineage import (
    LegacyPredictionMissingError,
    LinkedPredictionContextMismatchError,
    OperationalLineageStatusMissingError,
    OperationalPredictionContext,
    OperationalPredictionRequest,
    OperationalSessionMismatchError,
    OperationalSessionMissingError,
    OperationalStatusTransitionError,
    append_operational_prediction,
)
from tests.test_clinical_operational_lineage import (
    _result,
    _seed_operational_database,
)


def _pending_request(
    path: Path,
    session_id: str,
    context: OperationalPredictionContext,
) -> OperationalPredictionRequest:
    with sqlite3.connect(path) as connection:
        row = connection.execute(
            """SELECT p.id, p.result_json, ls.result_sha256
               FROM predictions AS p
               JOIN legacy_prediction_lineage_status AS ls
                 ON ls.legacy_prediction_id = p.id
               WHERE p.session_id = ? AND ls.status = 'pending_context'""",
            (session_id,),
        ).fetchone()
    assert row is not None
    return OperationalPredictionRequest(session_id, row[0], row[2], row[1], context)


def _counts(path: Path) -> tuple[int, int, int]:
    with sqlite3.connect(path) as connection:
        return tuple(
            connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in (
                "prediction_runs",
                "prediction_input_measurements",
                "operational_prediction_run_links",
            )
        )


def test_append_rejects_missing_legacy_status_without_orphans(tmp_path: Path) -> None:
    # Given: a valid session/context but no legacy prediction or pending status.
    path = tmp_path / "missing-status.db"
    session_id, _, context = _seed_operational_database(path)
    request = OperationalPredictionRequest(
        session_id,
        999,
        "0" * 64,
        json.dumps(_result()),
        context,
    )

    # When: the public adapter is invoked and the caller catches the typed error.
    with sqlite3.connect(path) as database:
        database.execute("PRAGMA foreign_keys = ON")
        with pytest.raises(LegacyPredictionMissingError):
            append_operational_prediction(database, request)
        database.commit()

    # Then: no immutable or operational lineage row can be committed.
    assert _counts(path) == (0, 0, 0)

    clinical_db_results.save_prediction(path, session_id, _result())
    pending = _pending_request(path, session_id, context)
    with sqlite3.connect(path) as database:
        database.execute(
            "DELETE FROM legacy_prediction_lineage_status WHERE legacy_prediction_id = ?",
            (pending.legacy_prediction_id,),
        )
        with pytest.raises(OperationalLineageStatusMissingError):
            append_operational_prediction(database, pending)
        database.commit()
    assert _counts(path) == (0, 0, 0)


def test_append_rejects_missing_and_mismatched_sessions(tmp_path: Path) -> None:
    # Given: one pending legacy prediction tied to its real operational session.
    path = tmp_path / "session-guards.db"
    session_id, _, context = _seed_operational_database(path)
    clinical_db_results.save_prediction(path, session_id, _result())
    request = _pending_request(path, session_id, context)

    # When/Then: a nonexistent supplied session is rejected without writes.
    with sqlite3.connect(path) as database:
        database.execute("PRAGMA foreign_keys = ON")
        missing = OperationalPredictionRequest(
            "missing-session",
            request.legacy_prediction_id,
            request.result_sha256,
            request.result_json,
            context,
        )
        with pytest.raises(OperationalSessionMissingError):
            append_operational_prediction(database, missing)
        database.execute(
            """INSERT INTO sessions (
                   id, patient_id, age, sex, operating_mode, test_id
               ) VALUES ('other-session', 'other', 40, 'F', 'screening', 'OTHER')"""
        )
        mismatched = OperationalPredictionRequest(
            "other-session",
            request.legacy_prediction_id,
            request.result_sha256,
            request.result_json,
            context,
        )
        with pytest.raises(OperationalSessionMismatchError):
            append_operational_prediction(database, mismatched)
        database.commit()
    assert _counts(path) == (0, 0, 0)


def test_linked_context_conflict_is_rejected_before_append(tmp_path: Path) -> None:
    # Given: one completed linked write.
    path = tmp_path / "linked-conflict.db"
    session_id, _, context = _seed_operational_database(path)
    clinical_db_results.save_prediction_with_lineage(path, session_id, _result(), context)
    result_json = json.dumps(_result(), separators=(",", ":"), sort_keys=True)
    result_sha256 = hashlib.sha256(result_json.encode()).hexdigest()
    with sqlite3.connect(path) as connection:
        prediction_id = connection.execute("SELECT id FROM predictions").fetchone()[0]
    conflict = OperationalPredictionContext(
        manifest_id=context.manifest_id,
        model_name=context.model_name,
        model_version="conflicting-model",
        preprocessing_version=context.preprocessing_version,
        feature_schema_version=context.feature_schema_version,
        decision_policy_version=context.decision_policy_version,
        measurement_ids=context.measurement_ids,
    )

    # When: a linked legacy result is presented with conflicting context.
    with sqlite3.connect(path) as database:
        database.execute("PRAGMA foreign_keys = ON")
        with pytest.raises(LinkedPredictionContextMismatchError):
            append_operational_prediction(
                database,
                OperationalPredictionRequest(
                    session_id,
                    prediction_id,
                    result_sha256,
                    result_json,
                    conflict,
                ),
            )
        database.commit()

    # Then: the original run is the only immutable lineage.
    assert _counts(path) == (1, 1, 1)


def test_failed_status_transition_rolls_back_all_lineage_rows(tmp_path: Path) -> None:
    # Given: a valid pending write whose status transition is suppressed.
    path = tmp_path / "transition-rollback.db"
    session_id, _, context = _seed_operational_database(path)
    clinical_db_results.save_prediction(path, session_id, _result())
    request = _pending_request(path, session_id, context)
    with sqlite3.connect(path) as database:
        database.execute("PRAGMA foreign_keys = ON")
        database.execute(
            """CREATE TRIGGER suppress_lineage_transition
               BEFORE UPDATE ON legacy_prediction_lineage_status
               BEGIN SELECT RAISE(IGNORE); END"""
        )

        # When: the checked transition affects zero rows.
        with pytest.raises(OperationalStatusTransitionError):
            append_operational_prediction(database, request)
        database.commit()

    # Then: run, exact inputs, and operational run link all roll back.
    assert _counts(path) == (0, 0, 0)


def test_valid_pending_status_transitions_atomically(tmp_path: Path) -> None:
    # Given: a legacy projection with truthful pending context.
    path = tmp_path / "valid-transition.db"
    session_id, _, context = _seed_operational_database(path)
    clinical_db_results.save_prediction(path, session_id, _result())
    request = _pending_request(path, session_id, context)

    # When: the public adapter links verified context.
    with sqlite3.connect(path) as database:
        database.execute("PRAGMA foreign_keys = ON")
        append_operational_prediction(database, request)
        database.commit()

    # Then: every lineage row and the status transition commit together.
    assert _counts(path) == (1, 1, 1)
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT status FROM legacy_prediction_lineage_status"
        ).fetchone()[0] == "linked"
