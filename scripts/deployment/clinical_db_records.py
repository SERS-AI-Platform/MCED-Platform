from __future__ import annotations

import json
import sqlite3
from typing import TypedDict

from sers.master_data.lineage_privacy import JsonValue

from .clinical_db_result_types import ClinicalPredictionResult


class UserRecord(TypedDict):
    id: int
    username: str
    password_hash: str
    role: str
    display_name: str
    created_at: str
    is_active: int


class ActiveUserRecord(TypedDict):
    id: int
    username: str
    display_name: str
    role: str


class SessionRecord(TypedDict):
    id: str
    patient_id: str
    age: float
    sex: str
    bmi: float | None
    operating_mode: str
    model_variant: str | None
    threshold: float | None
    status: str
    created_by: int | None
    created_at: str
    completed_at: str | None
    test_id: str | None
    test_date: str | None
    test_sequence: int | None
    owner_user_id: int | None
    retest_of_session_id: str | None
    qc_valid: int | None


class SessionHistoryRecord(SessionRecord):
    owner_username: str | None
    owner_display_name: str | None
    cancer_detected: int | None
    decision_level: str | None
    decision_policy: str | None
    screening_index: float | None
    report_id: str | None


class SpectrumRecord(TypedDict):
    id: int
    session_id: str
    filename: str
    qc_pass: int | None
    qc_flags: str | None
    replicate_correlation: float | None
    fp_mean: float | None
    uploaded_at: str


class PredictionRecord(TypedDict):
    id: int
    session_id: str
    cancer_detected: int
    decision_level: str | None
    decision_policy: str | None
    screening_index: float
    majority_vote: str | None
    cancer_type_prediction: str | None
    cancer_type_confidence: float | None
    cancer_type_probabilities: dict[str, float]
    per_replicate_results: list[JsonValue]
    result_json: ClinicalPredictionResult
    predicted_at: str


class ReportRecord(TypedDict):
    id: str
    session_id: str
    generated_by: int | None
    generated_at: str


class AuditLogRecord(TypedDict):
    id: int
    timestamp: str
    user_id: int | None
    session_id: str | None
    action: str
    detail: str | None
    user_name: str | None


def user_record(row: sqlite3.Row) -> UserRecord:
    return {
        "id": row["id"],
        "username": row["username"],
        "password_hash": row["password_hash"],
        "role": row["role"],
        "display_name": row["display_name"],
        "created_at": row["created_at"],
        "is_active": row["is_active"],
    }


def active_user_record(row: sqlite3.Row) -> ActiveUserRecord:
    return {
        "id": row["id"],
        "username": row["username"],
        "display_name": row["display_name"],
        "role": row["role"],
    }


def session_record(row: sqlite3.Row) -> SessionRecord:
    return {
        "id": row["id"],
        "patient_id": row["patient_id"],
        "age": row["age"],
        "sex": row["sex"],
        "bmi": row["bmi"],
        "operating_mode": row["operating_mode"],
        "model_variant": row["model_variant"],
        "threshold": row["threshold"],
        "status": row["status"],
        "created_by": row["created_by"],
        "created_at": row["created_at"],
        "completed_at": row["completed_at"],
        "test_id": row["test_id"],
        "test_date": row["test_date"],
        "test_sequence": row["test_sequence"],
        "owner_user_id": row["owner_user_id"],
        "retest_of_session_id": row["retest_of_session_id"],
        "qc_valid": row["qc_valid"],
    }


def session_history_record(row: sqlite3.Row) -> SessionHistoryRecord:
    record: SessionHistoryRecord = {
        **session_record(row),
        "owner_username": row["owner_username"],
        "owner_display_name": row["owner_display_name"],
        "cancer_detected": row["cancer_detected"],
        "decision_level": row["decision_level"],
        "decision_policy": row["decision_policy"],
        "screening_index": row["screening_index"],
        "report_id": row["report_id"],
    }
    return record


def spectrum_record(row: sqlite3.Row) -> SpectrumRecord:
    return {
        "id": row["id"],
        "session_id": row["session_id"],
        "filename": row["filename"],
        "qc_pass": row["qc_pass"],
        "qc_flags": row["qc_flags"],
        "replicate_correlation": row["replicate_correlation"],
        "fp_mean": row["fp_mean"],
        "uploaded_at": row["uploaded_at"],
    }


def prediction_record(row: sqlite3.Row) -> PredictionRecord:
    probabilities: dict[str, float] = json.loads(
        row["cancer_type_probabilities"] or "{}"
    )
    replicate_results: list[JsonValue] = json.loads(
        row["per_replicate_results"] or "[]"
    )
    result: ClinicalPredictionResult = json.loads(row["result_json"] or "{}")
    return {
        "id": row["id"],
        "session_id": row["session_id"],
        "cancer_detected": row["cancer_detected"],
        "decision_level": row["decision_level"],
        "decision_policy": row["decision_policy"],
        "screening_index": row["screening_index"],
        "majority_vote": row["majority_vote"],
        "cancer_type_prediction": row["cancer_type_prediction"],
        "cancer_type_confidence": row["cancer_type_confidence"],
        "cancer_type_probabilities": probabilities,
        "per_replicate_results": replicate_results,
        "result_json": result,
        "predicted_at": row["predicted_at"],
    }


def report_record(row: sqlite3.Row) -> ReportRecord:
    return {
        "id": row["id"],
        "session_id": row["session_id"],
        "generated_by": row["generated_by"],
        "generated_at": row["generated_at"],
    }


def audit_log_record(row: sqlite3.Row) -> AuditLogRecord:
    return {
        "id": row["id"],
        "timestamp": row["timestamp"],
        "user_id": row["user_id"],
        "session_id": row["session_id"],
        "action": row["action"],
        "detail": row["detail"],
        "user_name": row["user_name"],
    }
