from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from uuid import uuid4

from .lineage_types import QCEvaluationDraft, QCEvaluationId


@dataclass(frozen=True, slots=True)
class InvalidMetricsJsonError(ValueError):
    reason: str

    def __str__(self) -> str:
        return f"QC metrics_json is invalid: {self.reason}"


def append_qc_evaluation(
    connection: sqlite3.Connection,
    draft: QCEvaluationDraft,
) -> QCEvaluationId:
    try:
        metrics_json = json.dumps(
            json.loads(draft.metrics_json),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except json.JSONDecodeError as error:
        raise InvalidMetricsJsonError(str(error)) from error
    evaluation_id = QCEvaluationId(str(uuid4()))
    connection.execute(
        """INSERT INTO qc_evaluations (
               id, measurement_id, rule_version, evaluation_version,
               outcome, metrics_json
           ) VALUES (?, ?, ?, ?, ?, ?)""",
        (
            evaluation_id,
            draft.measurement_id,
            draft.rule_version,
            draft.evaluation_version,
            draft.outcome,
            metrics_json,
        ),
    )
    return evaluation_id
