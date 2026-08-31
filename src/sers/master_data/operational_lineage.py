from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from uuid import uuid4

from .lineage_types import DatasetManifestId, PredictionRunDraft, PredictionRunId
from .operational_lineage_validation import (
    LegacyPredictionMissingError,
    LinkedPredictionContextMismatchError,
    OperationalLineageStatusMissingError,
    OperationalMeasurementMissingError,
    OperationalRelation,
    OperationalRunLinkConflictError,
    OperationalSampleLinkMismatchError,
    OperationalSessionMismatchError,
    OperationalSessionMissingError,
    OperationalStatusTransitionError,
    verify_operational_request,
)
from .prediction_lineage import append_prediction_run

__all__ = [
    "LegacyPredictionMissingError",
    "LinkedPredictionContextMismatchError",
    "MissingLineageContextError",
    "OperationalLineageStatusMissingError",
    "OperationalMeasurementMissingError",
    "OperationalPredictionContext",
    "OperationalPredictionRequest",
    "OperationalRunLinkConflictError",
    "OperationalSampleLinkMismatchError",
    "OperationalSessionMismatchError",
    "OperationalSessionMissingError",
    "OperationalStatusTransitionError",
    "append_operational_prediction",
]


class MissingLineageContextError(ValueError):
    __slots__ = ("field",)

    def __init__(self, field: str) -> None:
        self.field = field
        super().__init__(f"operational prediction requires explicit {field}")


@dataclass(frozen=True, slots=True)
class OperationalPredictionContext:
    manifest_id: DatasetManifestId
    model_name: str
    model_version: str
    preprocessing_version: str
    feature_schema_version: str
    decision_policy_version: str
    measurement_ids: tuple[str, ...]
    relation: OperationalRelation = "source"

    def __post_init__(self) -> None:
        required = (
            ("manifest_id", self.manifest_id),
            ("model_name", self.model_name),
            ("model_version", self.model_version),
            ("preprocessing_version", self.preprocessing_version),
            ("feature_schema_version", self.feature_schema_version),
            ("decision_policy_version", self.decision_policy_version),
        )
        for field, value in required:
            if not value.strip():
                raise MissingLineageContextError(field)
        if not self.measurement_ids or any(
            not measurement_id.strip() for measurement_id in self.measurement_ids
        ):
            raise MissingLineageContextError("measurement_ids")


@dataclass(frozen=True, slots=True)
class OperationalPredictionRequest:
    session_id: str
    legacy_prediction_id: int
    result_sha256: str
    result_json: str
    context: OperationalPredictionContext


def append_operational_prediction(
    connection: sqlite3.Connection,
    request: OperationalPredictionRequest,
    *,
    idempotent: bool = False,
) -> PredictionRunId:
    state = verify_operational_request(connection, request)
    if idempotent and state.status == "linked":
        linked_run = state.linked_prediction_run_id
        assert linked_run is not None
        return linked_run

    savepoint = f"operational_{uuid4().hex}"
    connection.execute(f"SAVEPOINT {savepoint}")
    try:
        prediction_run_id = append_prediction_run(
            connection,
            PredictionRunDraft(
                manifest_id=request.context.manifest_id,
                model_name=request.context.model_name,
                model_version=request.context.model_version,
                preprocessing_version=request.context.preprocessing_version,
                feature_schema_version=request.context.feature_schema_version,
                decision_policy_version=request.context.decision_policy_version,
                measurement_ids=request.context.measurement_ids,
                status="completed",
                output_json=request.result_json,
            ),
        )
        for sample_id in state.sample_ids:
            link_key = (
                f"{request.session_id}\0{sample_id}\0{request.context.relation}".encode()
            )
            connection.execute(
                """INSERT OR IGNORE INTO operational_session_links (
                       id, session_id, sample_id, relation
                   ) VALUES (?, ?, ?, ?)""",
                (
                    hashlib.sha256(link_key).hexdigest(),
                    request.session_id,
                    sample_id,
                    request.context.relation,
                ),
            )
        connection.execute(
            """INSERT INTO operational_prediction_run_links (
                   id, legacy_prediction_id, session_id, result_sha256,
                   prediction_run_id
               ) VALUES (?, ?, ?, ?, ?)""",
            (
                hashlib.sha256(
                    f"{request.legacy_prediction_id}\0{prediction_run_id}".encode()
                ).hexdigest(),
                request.legacy_prediction_id,
                request.session_id,
                request.result_sha256,
                prediction_run_id,
            ),
        )
        if state.status == "pending_context":
            transition = connection.execute(
                """UPDATE legacy_prediction_lineage_status
                   SET status = 'linked',
                       linked_prediction_run_id = ?,
                       linked_at = CURRENT_TIMESTAMP
                   WHERE legacy_prediction_id = ? AND result_sha256 = ?
                         AND status = 'pending_context'
                         AND linked_prediction_run_id IS NULL""",
                (
                    prediction_run_id,
                    request.legacy_prediction_id,
                    request.result_sha256,
                ),
            )
            if transition.rowcount != 1:
                raise OperationalStatusTransitionError(request.legacy_prediction_id)
    except (sqlite3.Error, OperationalStatusTransitionError):
        connection.execute(f"ROLLBACK TO {savepoint}")
        connection.execute(f"RELEASE {savepoint}")
        raise
    connection.execute(f"RELEASE {savepoint}")
    return prediction_run_id
