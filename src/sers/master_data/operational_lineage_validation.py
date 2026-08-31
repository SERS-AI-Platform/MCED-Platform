from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Literal, Protocol

from .lineage_types import DatasetManifestId, PredictionRunId

OperationalRelation = Literal["source", "retest", "derived"]
OperationalStatus = Literal["pending_context", "linked"]


class OperationalContext(Protocol):
    @property
    def manifest_id(self) -> DatasetManifestId: ...

    @property
    def model_name(self) -> str: ...

    @property
    def model_version(self) -> str: ...

    @property
    def preprocessing_version(self) -> str: ...

    @property
    def feature_schema_version(self) -> str: ...

    @property
    def decision_policy_version(self) -> str: ...

    @property
    def measurement_ids(self) -> tuple[str, ...]: ...

    @property
    def relation(self) -> OperationalRelation: ...


class OperationalRequest(Protocol):
    @property
    def session_id(self) -> str: ...

    @property
    def legacy_prediction_id(self) -> int: ...

    @property
    def result_sha256(self) -> str: ...

    @property
    def context(self) -> OperationalContext: ...


class LegacyPredictionMissingError(ValueError):
    __slots__ = ("legacy_prediction_id",)

    def __init__(self, legacy_prediction_id: int) -> None:
        self.legacy_prediction_id = legacy_prediction_id
        super().__init__(f"legacy prediction does not exist: {legacy_prediction_id}")


class OperationalLineageStatusMissingError(ValueError):
    __slots__ = ("legacy_prediction_id",)

    def __init__(self, legacy_prediction_id: int) -> None:
        self.legacy_prediction_id = legacy_prediction_id
        super().__init__(f"legacy prediction has no lineage status: {legacy_prediction_id}")


class OperationalSessionMissingError(ValueError):
    __slots__ = ("session_id",)

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"operational session does not exist: {session_id}")


class OperationalSessionMismatchError(ValueError):
    __slots__ = ("session_id",)

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"operational session does not match legacy context: {session_id}")


class OperationalSampleLinkMismatchError(ValueError):
    __slots__ = ("session_id",)

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"operational sample links conflict for session {session_id}")


class OperationalRunLinkConflictError(ValueError):
    __slots__ = ("legacy_prediction_id",)

    def __init__(self, legacy_prediction_id: int) -> None:
        self.legacy_prediction_id = legacy_prediction_id
        super().__init__(f"operational run links conflict for prediction {legacy_prediction_id}")


class LinkedPredictionContextMismatchError(ValueError):
    __slots__ = ("session_id",)

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"linked prediction context differs for session {session_id}")


class InvalidOperationalStatusError(ValueError):
    __slots__ = ("status",)

    def __init__(self, status: str) -> None:
        self.status = status
        super().__init__(f"invalid operational lineage status: {status}")


class OperationalMeasurementMissingError(ValueError):
    __slots__ = ("measurement_id",)

    def __init__(self, measurement_id: str) -> None:
        self.measurement_id = measurement_id
        super().__init__(f"operational measurement does not exist: {measurement_id}")


class OperationalStatusTransitionError(RuntimeError):
    __slots__ = ("legacy_prediction_id",)

    def __init__(self, legacy_prediction_id: int) -> None:
        self.legacy_prediction_id = legacy_prediction_id
        super().__init__(
            f"operational status transition failed for prediction {legacy_prediction_id}"
        )


@dataclass(frozen=True, slots=True)
class VerifiedOperationalState:
    status: OperationalStatus
    linked_prediction_run_id: PredictionRunId | None
    sample_ids: tuple[str, ...]


def verify_operational_request(
    connection: sqlite3.Connection,
    request: OperationalRequest,
) -> VerifiedOperationalState:
    prediction = connection.execute(
        "SELECT session_id FROM predictions WHERE id = ?",
        (request.legacy_prediction_id,),
    ).fetchone()
    if prediction is None:
        raise LegacyPredictionMissingError(request.legacy_prediction_id)
    status_row = connection.execute(
        """SELECT status, linked_prediction_run_id, session_id
           FROM legacy_prediction_lineage_status
           WHERE legacy_prediction_id = ? AND result_sha256 = ?""",
        (request.legacy_prediction_id, request.result_sha256),
    ).fetchone()
    if status_row is None:
        raise OperationalLineageStatusMissingError(request.legacy_prediction_id)
    session = connection.execute(
        "SELECT 1 FROM sessions WHERE id = ?",
        (request.session_id,),
    ).fetchone()
    if session is None:
        raise OperationalSessionMissingError(request.session_id)
    if request.session_id not in {str(prediction[0]), str(status_row[2])} or (
        str(prediction[0]) != str(status_row[2])
    ):
        raise OperationalSessionMismatchError(request.session_id)

    sample_ids = _resolve_sample_ids(connection, request.context.measurement_ids)
    linked_samples = frozenset(
        str(row[0])
        for row in connection.execute(
            """SELECT sample_id FROM operational_session_links
               WHERE session_id = ? AND relation = ?""",
            (request.session_id, request.context.relation),
        )
    )
    if linked_samples and linked_samples != frozenset(sample_ids):
        raise OperationalSampleLinkMismatchError(request.session_id)

    raw_status = str(status_row[0])
    if raw_status == "pending_context":
        status: OperationalStatus = "pending_context"
    elif raw_status == "linked":
        status = "linked"
    else:
        raise InvalidOperationalStatusError(raw_status)
    linked_run = (
        None
        if status_row[1] is None
        else PredictionRunId(str(status_row[1]))
    )
    existing_links = tuple(
        (str(row[0]), PredictionRunId(str(row[1])))
        for row in connection.execute(
            """SELECT session_id, prediction_run_id
               FROM operational_prediction_run_links
               WHERE legacy_prediction_id = ? AND result_sha256 = ?""",
            (request.legacy_prediction_id, request.result_sha256),
        )
    )
    if status == "pending_context" and existing_links:
        raise OperationalRunLinkConflictError(request.legacy_prediction_id)
    if status == "linked":
        if not linked_samples:
            raise OperationalSampleLinkMismatchError(request.session_id)
        if linked_run is None or not _context_matches(
            connection, linked_run, request.context
        ):
            raise LinkedPredictionContextMismatchError(request.session_id)
        if linked_run not in {run_id for _, run_id in existing_links}:
            raise OperationalRunLinkConflictError(request.legacy_prediction_id)
        if any(
            session_id != request.session_id
            or not _context_matches(connection, run_id, request.context)
            for session_id, run_id in existing_links
        ):
            raise OperationalRunLinkConflictError(request.legacy_prediction_id)
    return VerifiedOperationalState(status, linked_run, sample_ids)


def _resolve_sample_ids(
    connection: sqlite3.Connection,
    measurement_ids: tuple[str, ...],
) -> tuple[str, ...]:
    sample_ids: set[str] = set()
    for measurement_id in measurement_ids:
        row = connection.execute(
            """SELECT am.sample_id
               FROM measurements AS m
               JOIN analytical_materials AS am
                 ON am.id = m.analytical_material_id
               WHERE m.id = ?""",
            (measurement_id,),
        ).fetchone()
        if row is None:
            raise OperationalMeasurementMissingError(measurement_id)
        sample_ids.add(str(row[0]))
    return tuple(sorted(sample_ids))


def _context_matches(
    connection: sqlite3.Connection,
    prediction_run_id: PredictionRunId,
    context: OperationalContext,
) -> bool:
    row = connection.execute(
        """SELECT dataset_manifest_id, model_name, model_version,
                  preprocessing_version, feature_schema_version,
                  decision_policy_version
           FROM prediction_runs WHERE id = ?""",
        (prediction_run_id,),
    ).fetchone()
    if row is None:
        return False
    stored_measurements = tuple(
        str(item[0])
        for item in connection.execute(
            """SELECT measurement_id FROM prediction_input_measurements
               WHERE prediction_run_id = ? ORDER BY measurement_id""",
            (prediction_run_id,),
        )
    )
    expected = (
        context.manifest_id,
        context.model_name,
        context.model_version,
        context.preprocessing_version,
        context.feature_schema_version,
        context.decision_policy_version,
    )
    return tuple(row) == expected and stored_measurements == tuple(
        sorted(context.measurement_ids)
    )
