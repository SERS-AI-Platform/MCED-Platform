from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from .lineage_types import DatasetBuildRequest
from .manifest_queries import ResolvedMember


@dataclass(frozen=True, slots=True)
class CanonicalManifest:
    content_sha256: str
    inclusion_policy_json: str
    exclusion_policy_json: str
    provenance_json: str


def canonical_manifest(
    request: DatasetBuildRequest,
    members: tuple[ResolvedMember, ...],
) -> CanonicalManifest:
    inclusion_policy = {
        "accepted_qc_outcomes": sorted(request.policy.accepted_qc_outcomes),
        "artifact_role": "raw",
        "label_status": "confirmed",
        "measurement_status": "acquired",
    }
    exclusion_policy = {
        "invalid_measurement": "exclude",
        "missing_qc": "exclude",
        "rejected_qc": "exclude",
    }
    provenance = {
        "created_by": request.provenance.created_by,
        "purpose": request.provenance.purpose,
        "source_revision": request.provenance.source_revision,
    }
    payload = {
        "decision_policy_version": request.decision_policy_version,
        "exclusion_policy": exclusion_policy,
        "feature_schema_version": request.feature_schema_version,
        "inclusion_policy": inclusion_policy,
        "members": [
            {
                "artifact_sha256": member.artifact_sha256,
                "exclusion_reason": member.exclusion_reason,
                "fold_index": member.draft.fold_index,
                "label_definition_version": member.label_definition_version,
                "label_evidence_ids": list(member.label_evidence_ids),
                "measurement_id": member.draft.measurement_id,
                "measurement_status": member.measurement_status,
                "qc_evaluation_id": member.qc_evaluation_id,
                "qc_evaluation_version": member.qc_evaluation_version,
                "qc_outcome": member.qc_outcome,
                "qc_rule_version": member.qc_rule_version,
                "sample_label_id": member.draft.sample_label_id,
                "split_name": member.draft.split_name,
            }
            for member in members
        ],
        "name": request.name,
        "preprocessing_version": request.preprocessing_version,
        "provenance": provenance,
        "qc_policy_version": request.policy.version,
        "qc_rule_version": request.policy.qc_rule_version,
    }
    canonical_json = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return CanonicalManifest(
        content_sha256=hashlib.sha256(canonical_json.encode()).hexdigest(),
        inclusion_policy_json=json.dumps(
            inclusion_policy,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        exclusion_policy_json=json.dumps(
            exclusion_policy,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        provenance_json=json.dumps(
            provenance,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
    )
