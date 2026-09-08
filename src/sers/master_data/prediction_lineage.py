from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from uuid import uuid4

from .lineage_privacy import (
    JsonValue,
    SensitivePredictionMetadataError,
    validate_prediction_privacy,
)
from .lineage_types import PredictionRunDraft, PredictionRunId

__all__ = [
    "SensitivePredictionMetadataError",
    "append_prediction_run",
]


@dataclass(frozen=True, slots=True)
class DuplicatePredictionInputError(ValueError):
    measurement_id: str

    def __str__(self) -> str:
        return f"prediction input appears more than once: {self.measurement_id}"


@dataclass(frozen=True, slots=True)
class PredictionInputMismatchError(ValueError):
    measurement_id: str

    def __str__(self) -> str:
        return f"prediction input is not included in its manifest: {self.measurement_id}"


@dataclass(frozen=True, slots=True)
class InvalidPredictionOutputError(ValueError):
    reason: str

    def __str__(self) -> str:
        return f"prediction output_json is invalid: {self.reason}"


def append_prediction_run(
    connection: sqlite3.Connection,
    draft: PredictionRunDraft,
) -> PredictionRunId:
    measurement_ids = tuple(sorted(draft.measurement_ids))
    for index, measurement_id in enumerate(measurement_ids[1:], start=1):
        if measurement_id == measurement_ids[index - 1]:
            raise DuplicatePredictionInputError(measurement_id)
    manifest_inputs = {
        row[0]
        for row in connection.execute(
            """SELECT measurement_id FROM dataset_manifest_items
               WHERE dataset_manifest_id = ?""",
            (draft.manifest_id,),
        )
    }
    for measurement_id in measurement_ids:
        if measurement_id not in manifest_inputs:
            raise PredictionInputMismatchError(measurement_id)
    try:
        output_value: JsonValue = (
            None if draft.output_json is None else json.loads(draft.output_json)
        )
        output_json = (
            None
            if draft.output_json is None
            else json.dumps(
                output_value,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    except json.JSONDecodeError as error:
        raise InvalidPredictionOutputError(str(error)) from error
    validate_prediction_privacy(connection, draft, output_value)
    input_json = json.dumps(measurement_ids, separators=(",", ":"))
    input_set_sha256 = hashlib.sha256(input_json.encode()).hexdigest()
    prediction_id = PredictionRunId(str(uuid4()))
    savepoint = f"prediction_{uuid4().hex}"
    connection.execute(f"SAVEPOINT {savepoint}")
    try:
        connection.execute(
            """INSERT INTO prediction_runs (
                   id, model_name, model_version, status, output_json,
                   dataset_manifest_id, preprocessing_version,
                   feature_schema_version, decision_policy_version,
                   input_set_sha256
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                prediction_id,
                draft.model_name,
                draft.model_version,
                draft.status,
                output_json,
                draft.manifest_id,
                draft.preprocessing_version,
                draft.feature_schema_version,
                draft.decision_policy_version,
                input_set_sha256,
            ),
        )
        connection.executemany(
            """INSERT INTO prediction_input_measurements (
                   prediction_run_id, measurement_id
               ) VALUES (?, ?)""",
            ((prediction_id, measurement_id) for measurement_id in measurement_ids),
        )
    except sqlite3.Error:
        connection.execute(f"ROLLBACK TO {savepoint}")
        connection.execute(f"RELEASE {savepoint}")
        raise
    connection.execute(f"RELEASE {savepoint}")
    return prediction_id
