"""Session ownership, history, test identity, and retest persistence."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from .clinical_db_core import connection
from .clinical_db_records import (
    SessionHistoryRecord,
    SessionRecord,
    session_history_record,
    session_record,
)


@dataclass(frozen=True, slots=True)
class HistoryFilters:
    patient_id: str = ""
    test_date: str = ""
    qc_status: str = ""
    result_status: str = ""
    owner_user_id: int | None = None
    page: int = 1


@dataclass(frozen=True, slots=True)
class RetestNotAllowedError(RuntimeError):
    session_id: str

    def __str__(self) -> str:
        return f"Session {self.session_id} is not eligible for retest"


@dataclass(frozen=True, slots=True)
class RetestBmiRequiredError(RuntimeError):
    session_id: str

    def __str__(self) -> str:
        return f"Session {self.session_id} requires BMI before retest"


def create_session(
    db_path: Path,
    *,
    patient_id: str,
    age: float,
    sex: str,
    bmi: float | None,
    operating_mode: str,
    created_by: int,
    threshold: float | None = None,
    owner_user_id: int | None = None,
    retest_of_session_id: str | None = None,
) -> str:
    """Create a session and allocate its per-patient daily test sequence."""
    session_id = str(uuid.uuid4())
    test_date = date.today().isoformat()
    with connection(db_path) as database:
        database.execute("BEGIN IMMEDIATE")
        sequence = database.execute(
            """SELECT COALESCE(MAX(test_sequence), 0) + 1
               FROM sessions WHERE patient_id = ? AND test_date = ?""",
            (patient_id, test_date),
        ).fetchone()[0]
        test_id = f"{patient_id}-{test_date.replace('-', '')}-{sequence:02d}"
        database.execute(
            """INSERT INTO sessions
               (id, patient_id, age, sex, bmi, operating_mode, threshold,
                created_by, owner_user_id, retest_of_session_id,
                test_id, test_date, test_sequence)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                patient_id,
                age,
                sex,
                bmi,
                operating_mode,
                threshold,
                created_by,
                owner_user_id or created_by,
                retest_of_session_id,
                test_id,
                test_date,
                sequence,
            ),
        )
    return session_id


def get_session(db_path: Path, session_id: str) -> SessionRecord | None:
    with connection(db_path) as database:
        row = database.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return session_record(row) if row else None


def get_session_for_user(
    db_path: Path, session_id: str, user_id: int, is_admin: bool
) -> SessionRecord | None:
    with connection(db_path) as database:
        if is_admin:
            row = database.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        else:
            row = database.execute(
                """SELECT * FROM sessions
                   WHERE id = ? AND owner_user_id = ?""",
                (session_id, user_id),
            ).fetchone()
        return session_record(row) if row else None


def update_session_status(
    db_path: Path,
    session_id: str,
    status: str,
    *,
    model_variant: str | None = None,
    qc_valid: bool | None = None,
) -> None:
    assignments = ["status = ?"]
    values: list[str | bool | None] = [status]
    if model_variant is not None:
        assignments.append("model_variant = ?")
        values.append(model_variant)
    if qc_valid is not None:
        assignments.append("qc_valid = ?")
        values.append(qc_valid)
    if status in {"analyzed", "reported"}:
        assignments.append("completed_at = ?")
        values.append(datetime.now().isoformat())
    values.append(session_id)
    with connection(db_path) as database:
        database.execute(f"UPDATE sessions SET {', '.join(assignments)} WHERE id = ?", values)


def create_retest_session(
    db_path: Path,
    source_session_id: str,
    performed_by: int,
    bmi_override: float | None = None,
) -> str:
    source = get_session(db_path, source_session_id)
    if source is None or source["qc_valid"] != 0:
        raise RetestNotAllowedError(source_session_id)
    bmi = bmi_override if bmi_override is not None else source["bmi"]
    if bmi is None:
        raise RetestBmiRequiredError(source_session_id)
    return create_session(
        db_path,
        patient_id=source["patient_id"],
        age=source["age"],
        sex=source["sex"],
        bmi=bmi,
        operating_mode=source["operating_mode"],
        created_by=performed_by,
        threshold=source["threshold"],
        owner_user_id=source["owner_user_id"],
        retest_of_session_id=source_session_id,
    )


def get_session_history(
    db_path: Path,
    user_id: int,
    is_admin: bool,
    filters: HistoryFilters | None = None,
) -> tuple[list[SessionHistoryRecord], int]:
    query = filters or HistoryFilters()
    clauses: list[str] = []
    values: list[str | int] = []
    if not is_admin:
        clauses.append("s.owner_user_id = ?")
        values.append(user_id)
    elif query.owner_user_id is not None:
        clauses.append("s.owner_user_id = ?")
        values.append(query.owner_user_id)
    if query.patient_id:
        clauses.append("s.patient_id LIKE ?")
        values.append(f"%{query.patient_id}%")
    if query.test_date:
        clauses.append("s.test_date = ?")
        values.append(query.test_date)
    if query.qc_status:
        qc_value = {"passed": 1, "failed": 0}.get(query.qc_status)
        if qc_value is None:
            clauses.append("s.qc_valid IS NULL")
        else:
            clauses.append("s.qc_valid = ?")
            values.append(qc_value)
    if query.result_status:
        result_clause = {
            "positive": "p.decision_level = 'positive'",
            "moderate": "p.decision_level = 'moderate'",
            "negative": "p.decision_level = 'negative'",
            "pending": "p.id IS NULL",
        }.get(query.result_status)
        if result_clause:
            clauses.append(result_clause)
    where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with connection(db_path) as database:
        total = database.execute(
            f"""SELECT COUNT(*) FROM sessions s
                LEFT JOIN predictions p ON p.session_id = s.id
                {where_clause}""",
            values,
        ).fetchone()[0]
        rows = database.execute(
            f"""SELECT s.*, u.username AS owner_username,
                       u.display_name AS owner_display_name,
                       p.cancer_detected, p.decision_level, p.decision_policy,
                       p.screening_index,
                       (SELECT r.id FROM reports r WHERE r.session_id = s.id
                        ORDER BY r.generated_at, r.id LIMIT 1) AS report_id
                FROM sessions s
                LEFT JOIN users u ON u.id = s.owner_user_id
                LEFT JOIN predictions p ON p.session_id = s.id
                {where_clause}
                ORDER BY s.created_at DESC, s.id DESC
                LIMIT 50 OFFSET ?""",
            [*values, (max(query.page, 1) - 1) * 50],
        ).fetchall()
    return [session_history_record(row) for row in rows], int(total)
