from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NewType

DatasetManifestId = NewType("DatasetManifestId", str)
QCEvaluationId = NewType("QCEvaluationId", str)
PredictionRunId = NewType("PredictionRunId", str)

QCOutcome = Literal["pass", "fail", "review"]
PredictionStatus = Literal["completed", "failed"]


@dataclass(frozen=True, slots=True)
class QCEvaluationDraft:
    measurement_id: str
    rule_version: str
    evaluation_version: str
    outcome: QCOutcome
    metrics_json: str


@dataclass(frozen=True, slots=True)
class DatasetPolicy:
    version: str
    qc_rule_version: str
    accepted_qc_outcomes: tuple[QCOutcome, ...]


@dataclass(frozen=True, slots=True)
class CreationProvenance:
    created_by: str
    source_revision: str
    purpose: str


@dataclass(frozen=True, slots=True)
class DatasetMemberDraft:
    measurement_id: str
    sample_label_id: str
    split_name: str
    fold_index: int


@dataclass(frozen=True, slots=True)
class DatasetBuildRequest:
    name: str
    members: tuple[DatasetMemberDraft, ...]
    policy: DatasetPolicy
    preprocessing_version: str
    feature_schema_version: str
    decision_policy_version: str
    provenance: CreationProvenance


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    id: DatasetManifestId
    content_sha256: str
    included_count: int
    excluded_count: int


@dataclass(frozen=True, slots=True)
class PredictionRunDraft:
    manifest_id: DatasetManifestId
    model_name: str
    model_version: str
    preprocessing_version: str
    feature_schema_version: str
    decision_policy_version: str
    measurement_ids: tuple[str, ...]
    status: PredictionStatus
    output_json: str | None
