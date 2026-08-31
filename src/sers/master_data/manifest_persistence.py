from __future__ import annotations

import sqlite3
from uuid import uuid4

from .lineage_types import (
    DatasetBuildRequest,
    DatasetManifest,
    DatasetManifestId,
)
from .manifest_canonical import CanonicalManifest
from .manifest_queries import ResolvedMember
from .matching_persistence import stable_id


def _persist_included(
    connection: sqlite3.Connection,
    manifest_id: DatasetManifestId,
    member: ResolvedMember,
    ordinal: int,
) -> None:
    assert member.qc_evaluation_id is not None
    item_id = stable_id(
        "dataset-manifest-item",
        f"{manifest_id}:{member.draft.measurement_id}",
    )
    connection.execute(
        """INSERT INTO dataset_manifest_items (
               id, dataset_manifest_id, measurement_id, sample_label_id,
               qc_evaluation_id, ordinal, artifact_sha256,
               label_definition_version, qc_rule_version,
               qc_evaluation_version, split_name, fold_index
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            item_id,
            manifest_id,
            member.draft.measurement_id,
            member.draft.sample_label_id,
            member.qc_evaluation_id,
            ordinal,
            member.artifact_sha256,
            member.label_definition_version,
            member.qc_rule_version,
            member.qc_evaluation_version,
            member.draft.split_name,
            member.draft.fold_index,
        ),
    )
    connection.executemany(
        """INSERT INTO dataset_manifest_label_evidence (
               dataset_manifest_item_id, sample_label_evidence_id
           ) VALUES (?, ?)""",
        ((item_id, evidence_id) for evidence_id in member.label_evidence_ids),
    )


def _persist_excluded(
    connection: sqlite3.Connection,
    manifest_id: DatasetManifestId,
    member: ResolvedMember,
) -> None:
    assert member.exclusion_reason is not None
    connection.execute(
        """INSERT INTO dataset_manifest_exclusions (
               id, dataset_manifest_id, measurement_id, sample_label_id,
               artifact_sha256, qc_evaluation_id, reason
           ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            stable_id(
                "dataset-manifest-exclusion",
                f"{manifest_id}:{member.draft.measurement_id}",
            ),
            manifest_id,
            member.draft.measurement_id,
            member.draft.sample_label_id,
            member.artifact_sha256,
            member.qc_evaluation_id,
            member.exclusion_reason,
        ),
    )


def persist_manifest(
    connection: sqlite3.Connection,
    request: DatasetBuildRequest,
    members: tuple[ResolvedMember, ...],
    canonical: CanonicalManifest,
) -> DatasetManifest:
    manifest_id = DatasetManifestId(
        stable_id("dataset-manifest", canonical.content_sha256)
    )
    included = tuple(member for member in members if member.exclusion_reason is None)
    excluded = tuple(member for member in members if member.exclusion_reason is not None)
    existing = connection.execute(
        "SELECT id FROM dataset_manifests WHERE id = ?",
        (manifest_id,),
    ).fetchone()
    if existing is not None:
        return DatasetManifest(
            manifest_id,
            canonical.content_sha256,
            len(included),
            len(excluded),
        )
    savepoint = f"manifest_{uuid4().hex}"
    connection.execute(f"SAVEPOINT {savepoint}")
    try:
        connection.execute(
            """INSERT INTO dataset_manifests (
                   id, name, version, content_sha256, status
               ) VALUES (?, ?, ?, ?, 'draft')""",
            (
                manifest_id,
                request.name,
                canonical.content_sha256[:16],
                canonical.content_sha256,
            ),
        )
        connection.execute(
            """INSERT INTO dataset_manifest_specs (
                   dataset_manifest_id, preprocessing_version,
                   feature_schema_version, decision_policy_version,
                   qc_policy_version, qc_rule_version,
                   inclusion_policy_json, exclusion_policy_json, provenance_json
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                manifest_id,
                request.preprocessing_version,
                request.feature_schema_version,
                request.decision_policy_version,
                request.policy.version,
                request.policy.qc_rule_version,
                canonical.inclusion_policy_json,
                canonical.exclusion_policy_json,
                canonical.provenance_json,
            ),
        )
        for ordinal, member in enumerate(included):
            _persist_included(connection, manifest_id, member, ordinal)
        for member in excluded:
            _persist_excluded(connection, manifest_id, member)
        connection.execute(
            "UPDATE dataset_manifests SET status = 'frozen' WHERE id = ?",
            (manifest_id,),
        )
    except sqlite3.Error:
        connection.execute(f"ROLLBACK TO {savepoint}")
        connection.execute(f"RELEASE {savepoint}")
        raise
    connection.execute(f"RELEASE {savepoint}")
    return DatasetManifest(
        manifest_id,
        canonical.content_sha256,
        len(included),
        len(excluded),
    )
