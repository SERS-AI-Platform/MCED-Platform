"""Prediction and stable report metadata persistence."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from sers.master_data.lineage_types import PredictionRunId
from sers.master_data.operational_lineage import (
    OperationalPredictionContext,
    OperationalPredictionRequest,
    append_operational_prediction,
)

from .clinical_db_core import connection
from .clinical_db_records import (
    PredictionRecord,
    ReportRecord,
    prediction_record,
    report_record,
)
from .clinical_db_result_types import ClinicalPredictionResult


class SessionMissingError(RuntimeError):
    __slots__ = ("session_id",)

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"Session {session_id} does not exist")


@dataclass(frozen=True, slots=True)
class PendingPredictionLineage:
    id: str
    legacy_prediction_id: int
    session_id: str
    result_sha256: str
    created_at: str


def save_prediction(
    db_path: Path,
    session_id: str,
    result: ClinicalPredictionResult,
) -> None:
    result_json, result_sha256 = _canonical_result(result)
    with connection(db_path) as database:
        prediction_id = _upsert_prediction(
            database,
            session_id,
            result,
            result_json,
        )
        _mark_pending(
            database,
            prediction_id,
            session_id,
            result_sha256,
        )


def save_prediction_with_lineage(
    db_path: Path,
    session_id: str,
    result: ClinicalPredictionResult,
    context: OperationalPredictionContext,
) -> PredictionRunId:
    result_json, result_sha256 = _canonical_result(result)
    with connection(db_path) as database:
        prediction_id = _upsert_prediction(
            database,
            session_id,
            result,
            result_json,
        )
        _mark_pending(database, prediction_id, session_id, result_sha256)
        return append_operational_prediction(
            database,
            OperationalPredictionRequest(
                session_id,
                prediction_id,
                result_sha256,
                result_json,
                context,
            ),
        )


def backfill_prediction_lineage(
    db_path: Path,
    session_id: str,
    context: OperationalPredictionContext,
) -> PredictionRunId:
    with connection(db_path) as database:
        row = database.execute(
            """SELECT id, result_json FROM predictions
               WHERE session_id = ?""",
            (session_id,),
        ).fetchone()
        if row is None:
            raise SessionMissingError(session_id)
        result_json, result_sha256 = _canonical_result(
            json.loads(row["result_json"] or "{}")
        )
        prediction_id = int(row["id"])
        _mark_pending(database, prediction_id, session_id, result_sha256)
        return append_operational_prediction(
            database,
            OperationalPredictionRequest(
                session_id,
                prediction_id,
                result_sha256,
                result_json,
                context,
            ),
            idempotent=True,
        )


def list_pending_prediction_lineage(
    db_path: Path,
) -> tuple[PendingPredictionLineage, ...]:
    with connection(db_path) as database:
        rows = database.execute(
            """SELECT id, legacy_prediction_id, session_id, result_sha256,
                      created_at
               FROM legacy_prediction_lineage_status
               WHERE status = 'pending_context'
               ORDER BY created_at, id"""
        ).fetchall()
        return tuple(
            PendingPredictionLineage(
                id=str(row["id"]),
                legacy_prediction_id=int(row["legacy_prediction_id"]),
                session_id=str(row["session_id"]),
                result_sha256=str(row["result_sha256"]),
                created_at=str(row["created_at"]),
            )
            for row in rows
        )


def _canonical_result(result: ClinicalPredictionResult) -> tuple[str, str]:
    result_json = json.dumps(
        result,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return result_json, hashlib.sha256(result_json.encode()).hexdigest()


def _upsert_prediction(
    database: sqlite3.Connection,
    session_id: str,
    result: ClinicalPredictionResult,
    result_json: str,
) -> int:
    patient_decision = result.get("patient_decision", {})
    values = (
        result.get("cancer_detected", False),
        result.get("decision_level", patient_decision.get("decision_level")),
        result.get("decision_policy", patient_decision.get("decision_policy")),
        result.get("screening_index", result.get("cancer_signal_score", 0.0)),
        result.get("majority_vote"),
        result.get("cancer_type_prediction"),
        result.get("cancer_type_confidence"),
        json.dumps(result.get("cancer_type_probabilities", {})),
        json.dumps(result.get("per_replicate", []), ensure_ascii=False),
        result_json,
    )
    database.execute(
        """INSERT INTO predictions
           (session_id, cancer_detected, decision_level, decision_policy,
            screening_index, majority_vote,
            cancer_type_prediction, cancer_type_confidence,
            cancer_type_probabilities, per_replicate_results, result_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(session_id) DO UPDATE SET
               cancer_detected = excluded.cancer_detected,
               decision_level = excluded.decision_level,
               decision_policy = excluded.decision_policy,
               screening_index = excluded.screening_index,
               majority_vote = excluded.majority_vote,
               cancer_type_prediction = excluded.cancer_type_prediction,
               cancer_type_confidence = excluded.cancer_type_confidence,
               cancer_type_probabilities = excluded.cancer_type_probabilities,
               per_replicate_results = excluded.per_replicate_results,
               result_json = excluded.result_json,
               predicted_at = CURRENT_TIMESTAMP""",
        (session_id, *values),
    )
    row = database.execute(
        "SELECT id FROM predictions WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    if row is None:
        raise SessionMissingError(session_id)
    return int(row[0])


def _mark_pending(
    database: sqlite3.Connection,
    prediction_id: int,
    session_id: str,
    result_sha256: str,
) -> None:
    status_id = hashlib.sha256(
        f"{prediction_id}\0{result_sha256}".encode()
    ).hexdigest()
    database.execute(
        """INSERT OR IGNORE INTO legacy_prediction_lineage_status (
               id, legacy_prediction_id, session_id, result_sha256, status
           ) VALUES (?, ?, ?, ?, 'pending_context')""",
        (status_id, prediction_id, session_id, result_sha256),
    )


def get_prediction(db_path: Path, session_id: str) -> PredictionRecord | None:
    with connection(db_path) as database:
        row = database.execute(
            "SELECT * FROM predictions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row is None:
            return None
        return prediction_record(row)


def create_report(db_path: Path, session_id: str, generated_by: int) -> str:
    with connection(db_path) as database:
        existing = database.execute(
            """SELECT id FROM reports WHERE session_id = ?
               ORDER BY generated_at, id LIMIT 1""",
            (session_id,),
        ).fetchone()
        if existing:
            return str(existing[0])
        session = database.execute(
            "SELECT test_id FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if session is None:
            raise SessionMissingError(session_id)
        report_id = f"{session[0]}-R01"
        database.execute(
            """INSERT OR IGNORE INTO reports (id, session_id, generated_by)
               VALUES (?, ?, ?)""",
            (report_id, session_id, generated_by),
        )
        return report_id


def get_report(db_path: Path, report_id: str) -> ReportRecord | None:
    with connection(db_path) as database:
        row = database.execute("SELECT * FROM reports WHERE id = ?", (report_id,)).fetchone()
        return report_record(row) if row else None


def get_reports_for_session(db_path: Path, session_id: str) -> list[ReportRecord]:
    with connection(db_path) as database:
        rows = database.execute(
            """SELECT * FROM reports WHERE session_id = ?
               ORDER BY generated_at, id""",
            (session_id,),
        ).fetchall()
        return [report_record(row) for row in rows]
