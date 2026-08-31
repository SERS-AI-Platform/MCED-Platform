from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Literal

from .lineage_types import DatasetMemberDraft, DatasetPolicy, QCOutcome

ExclusionReason = Literal[
    "measurement_invalid",
    "qc_missing",
    "qc_outcome_rejected",
]


@dataclass(frozen=True, slots=True)
class ManifestInputNotFoundError(LookupError):
    measurement_id: str

    def __str__(self) -> str:
        return f"manifest input {self.measurement_id} is incomplete"


@dataclass(frozen=True, slots=True)
class ManifestMemberMismatchError(ValueError):
    measurement_id: str
    sample_label_id: str

    def __str__(self) -> str:
        return f"label does not belong to measurement {self.measurement_id}"


@dataclass(frozen=True, slots=True)
class ResolvedMember:
    draft: DatasetMemberDraft
    artifact_sha256: str
    label_definition_version: str
    label_evidence_ids: tuple[str, ...]
    measurement_status: str
    qc_evaluation_id: str | None
    qc_rule_version: str | None
    qc_evaluation_version: str | None
    qc_outcome: QCOutcome | None
    exclusion_reason: ExclusionReason | None


def resolve_member(
    connection: sqlite3.Connection,
    draft: DatasetMemberDraft,
    policy: DatasetPolicy,
) -> ResolvedMember:
    row = connection.execute(
        """SELECT m.status, am.sample_id, sa.sha256, sl.sample_id,
                  sl.label_definition_version, sl.status
           FROM measurements AS m
           JOIN analytical_materials AS am ON am.id = m.analytical_material_id
           JOIN measurement_artifacts AS ma
             ON ma.measurement_id = m.id AND ma.artifact_role = 'raw'
           JOIN source_assets AS sa ON sa.id = ma.source_asset_id
           JOIN sample_labels AS sl ON sl.id = ?
           WHERE m.id = ?""",
        (draft.sample_label_id, draft.measurement_id),
    ).fetchone()
    if row is None:
        raise ManifestInputNotFoundError(draft.measurement_id)
    if row[1] != row[3] or row[5] != "confirmed":
        raise ManifestMemberMismatchError(
            draft.measurement_id,
            draft.sample_label_id,
        )
    qc_row = connection.execute(
        """SELECT id, rule_version, evaluation_version, outcome
           FROM qc_evaluations
           WHERE measurement_id = ? AND rule_version = ?
           ORDER BY evaluated_at DESC, rowid DESC
           LIMIT 1""",
        (draft.measurement_id, policy.qc_rule_version),
    ).fetchone()
    if row[0] == "invalid":
        reason: ExclusionReason | None = "measurement_invalid"
    elif qc_row is None:
        reason = "qc_missing"
    elif qc_row[3] not in policy.accepted_qc_outcomes:
        reason = "qc_outcome_rejected"
    else:
        reason = None
    evidence_ids = tuple(
        evidence_row[0]
        for evidence_row in connection.execute(
            """SELECT id FROM sample_label_evidence
               WHERE sample_label_id = ? ORDER BY id""",
            (draft.sample_label_id,),
        )
    )
    return ResolvedMember(
        draft=draft,
        artifact_sha256=row[2],
        label_definition_version=row[4],
        label_evidence_ids=evidence_ids,
        measurement_status=row[0],
        qc_evaluation_id=None if qc_row is None else qc_row[0],
        qc_rule_version=None if qc_row is None else qc_row[1],
        qc_evaluation_version=None if qc_row is None else qc_row[2],
        qc_outcome=None if qc_row is None else qc_row[3],
        exclusion_reason=reason,
    )
