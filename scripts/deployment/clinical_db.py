"""SQLite persistence facade for the SERS clinical application."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Mapping

from sers.master_data.lineage_privacy import JsonValue
from sers.master_data.operational_lineage import OperationalPredictionContext

from . import clinical_db_records as records
from . import clinical_db_results as result_store
from . import clinical_db_sessions as session_store
from .clinical_db_core import connection, migrate_legacy_database
from .clinical_db_result_types import ClinicalPredictionResult
from .clinical_db_schema import migrate_schema
from .clinical_db_sessions import (
    HistoryFilters,
    RetestBmiRequiredError,
    RetestNotAllowedError,
)

__all__ = [
    "HistoryFilters",
    "RetestBmiRequiredError",
    "RetestNotAllowedError",
]


DEFAULT_DB_PATH = Path(__file__).parent / "clinical_data.db"


def get_db_path() -> Path:
    configured_path = os.environ.get("SERS_CLINICAL_DB_PATH")
    if configured_path:
        return Path(configured_path)
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "clinical_data.db"
    return DEFAULT_DB_PATH


@contextmanager
def get_connection(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    with connection(db_path or get_db_path()) as database:
        yield database


def init_db(db_path: Path | None = None) -> None:
    target = db_path or get_db_path()
    if getattr(sys, "frozen", False):
        migrate_legacy_database(target, Path(__file__).parent / "clinical_data.db")
    target.parent.mkdir(parents=True, exist_ok=True)
    with connection(target) as database:
        migrate_schema(database)


def create_user(username: str, password_hash: str, role: str, display_name: str) -> int:
    with get_connection() as database:
        cursor = database.execute(
            """INSERT INTO users (username, password_hash, role, display_name)
               VALUES (?, ?, ?, ?)""",
            (username, password_hash, role, display_name),
        )
        user_id = cursor.lastrowid
        assert user_id is not None
        return user_id


def get_user_by_username(username: str) -> records.UserRecord | None:
    with get_connection() as database:
        row = database.execute(
            "SELECT * FROM users WHERE username = ? AND is_active = 1", (username,)
        ).fetchone()
        return records.user_record(row) if row else None


def get_user_by_id(user_id: int) -> records.UserRecord | None:
    with get_connection() as database:
        row = database.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return records.user_record(row) if row else None


def get_active_users() -> list[records.ActiveUserRecord]:
    with get_connection() as database:
        rows = database.execute(
            """SELECT id, username, display_name, role FROM users
               WHERE is_active = 1 ORDER BY display_name, username"""
        ).fetchall()
        return [records.active_user_record(row) for row in rows]


def update_user_password(user_id: int, password_hash: str) -> bool:
    with get_connection() as database:
        cursor = database.execute(
            "UPDATE users SET password_hash = ? WHERE id = ? AND is_active = 1",
            (password_hash, user_id),
        )
        return int(cursor.rowcount) == 1


def create_session(
    patient_id: str,
    age: float,
    sex: str,
    bmi: float | None,
    operating_mode: str,
    created_by: int,
    threshold: float | None = None,
) -> str:
    return str(session_store.create_session(
        get_db_path(),
        patient_id=patient_id,
        age=age,
        sex=sex,
        bmi=bmi,
        operating_mode=operating_mode,
        created_by=created_by,
        threshold=threshold,
    ))


def get_session(session_id: str) -> records.SessionRecord | None:
    return session_store.get_session(get_db_path(), session_id)


def get_session_for_user(
    session_id: str, user_id: int, is_admin: bool
) -> records.SessionRecord | None:
    return session_store.get_session_for_user(get_db_path(), session_id, user_id, is_admin)


def update_session_status(
    session_id: str,
    status: str,
    *,
    model_variant: str | None = None,
    qc_valid: bool | None = None,
) -> None:
    session_store.update_session_status(
        get_db_path(),
        session_id,
        status,
        model_variant=model_variant,
        qc_valid=qc_valid,
    )


def create_retest_session(
    source_session_id: str,
    performed_by: int,
    bmi_override: float | None = None,
) -> str:
    return str(session_store.create_retest_session(
        get_db_path(), source_session_id, performed_by, bmi_override
    ))


def get_session_history(
    user_id: int,
    is_admin: bool,
    filters: HistoryFilters | None = None,
) -> tuple[list[records.SessionHistoryRecord], int]:
    rows, count = session_store.get_session_history(get_db_path(), user_id, is_admin, filters)
    return list(rows), int(count)


def reset_session_measurements(session_id: str) -> None:
    """Clear only a not-yet-interpreted attempt kept for legacy callers."""
    with get_connection() as database:
        interpreted = database.execute(
            """SELECT EXISTS(SELECT 1 FROM predictions WHERE session_id = ?)
                      OR EXISTS(SELECT 1 FROM reports WHERE session_id = ?)""",
            (session_id, session_id),
        ).fetchone()[0]
        if interpreted:
            raise RetestNotAllowedError(session_id)
        database.execute("DELETE FROM spectra WHERE session_id = ?", (session_id,))
        database.execute(
            """UPDATE sessions SET status = 'created', model_variant = NULL,
                   completed_at = NULL, qc_valid = NULL WHERE id = ?""",
            (session_id,),
        )


def add_spectrum(session_id: str, filename: str) -> int:
    with get_connection() as database:
        cursor = database.execute(
            "INSERT INTO spectra (session_id, filename) VALUES (?, ?)",
            (session_id, filename),
        )
        spectrum_id = cursor.lastrowid
        assert spectrum_id is not None
        return spectrum_id


def update_spectrum_qc(
    spectrum_id: int,
    qc_pass: bool,
    qc_flags: list[str],
    correlation: float | None = None,
    fp_mean: float | None = None,
) -> None:
    with get_connection() as database:
        database.execute(
            """UPDATE spectra SET qc_pass = ?, qc_flags = ?,
                   replicate_correlation = ?, fp_mean = ? WHERE id = ?""",
            (
                qc_pass,
                json.dumps(qc_flags, ensure_ascii=False),
                correlation,
                fp_mean,
                spectrum_id,
            ),
        )


def get_spectra_for_session(session_id: str) -> list[records.SpectrumRecord]:
    with get_connection() as database:
        rows = database.execute(
            "SELECT * FROM spectra WHERE session_id = ? ORDER BY id", (session_id,)
        ).fetchall()
        return [records.spectrum_record(row) for row in rows]


def save_prediction(session_id: str, result: ClinicalPredictionResult) -> None:
    result_store.save_prediction(get_db_path(), session_id, result)


def save_prediction_with_lineage(
    session_id: str,
    result: ClinicalPredictionResult,
    context: OperationalPredictionContext,
) -> str:
    return str(
        result_store.save_prediction_with_lineage(get_db_path(), session_id, result, context)
    )


def backfill_prediction_lineage(
    session_id: str,
    context: OperationalPredictionContext,
) -> str:
    return str(result_store.backfill_prediction_lineage(get_db_path(), session_id, context))


def get_pending_prediction_lineage() -> tuple[result_store.PendingPredictionLineage, ...]:
    return tuple(result_store.list_pending_prediction_lineage(get_db_path()))


def get_prediction(session_id: str) -> records.PredictionRecord | None:
    return result_store.get_prediction(get_db_path(), session_id)


def create_report(session_id: str, generated_by: int) -> str:
    return str(result_store.create_report(get_db_path(), session_id, generated_by))


def get_report(report_id: str) -> records.ReportRecord | None:
    return result_store.get_report(get_db_path(), report_id)


def get_reports_for_session(session_id: str) -> list[records.ReportRecord]:
    return list(result_store.get_reports_for_session(get_db_path(), session_id))


def log_audit(
    action: str,
    user_id: int | None = None,
    session_id: str | None = None,
    detail: Mapping[str, JsonValue] | None = None,
) -> None:
    with get_connection() as database:
        database.execute(
            """INSERT INTO audit_log (user_id, session_id, action, detail)
               VALUES (?, ?, ?, ?)""",
            (
                user_id,
                session_id,
                action,
                json.dumps(detail, ensure_ascii=False) if detail else None,
            ),
        )


def get_audit_logs(limit: int = 100, offset: int = 0) -> list[records.AuditLogRecord]:
    with get_connection() as database:
        rows = database.execute(
            """SELECT a.*, u.display_name AS user_name
               FROM audit_log a LEFT JOIN users u ON a.user_id = u.id
               ORDER BY a.timestamp DESC, a.id DESC LIMIT ? OFFSET ?""",
            (limit, offset),
        ).fetchall()
        return [records.audit_log_record(row) for row in rows]


def ensure_default_users() -> None:
    with get_connection() as database:
        count = database.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        if count:
            return
        password_hash = hashlib.sha256("admin123".encode()).hexdigest()
        database.executemany(
            """INSERT INTO users (username, password_hash, role, display_name)
               VALUES (?, ?, ?, ?)""",
            (
                ("admin", password_hash, "admin", "관리자"),
                ("doctor1", password_hash, "clinician", "담당 의사"),
            ),
        )
