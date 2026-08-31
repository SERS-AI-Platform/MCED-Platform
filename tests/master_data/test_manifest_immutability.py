from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from sers.master_data.dataset_export import export_dataset_manifest
from sers.master_data.dataset_manifest import build_dataset_manifest
from sers.master_data.lineage_types import (
    CreationProvenance,
    DatasetBuildRequest,
    DatasetManifest,
    DatasetMemberDraft,
    DatasetPolicy,
    QCEvaluationDraft,
)
from sers.master_data.qc_lineage import append_qc_evaluation
from tests.master_data.lineage_test_support import seed_lineage
from tests.master_data.matching_test_support import database

MUTATION_SQL = (
    "UPDATE dataset_manifests SET name = name",
    "DELETE FROM dataset_manifests",
    "UPDATE dataset_manifest_specs SET preprocessing_version = preprocessing_version",
    "DELETE FROM dataset_manifest_specs",
    "UPDATE dataset_manifest_items SET ordinal = ordinal",
    "DELETE FROM dataset_manifest_items",
    "UPDATE dataset_manifest_exclusions SET reason = reason",
    "DELETE FROM dataset_manifest_exclusions",
    """UPDATE dataset_manifest_label_evidence
       SET sample_label_evidence_id = sample_label_evidence_id""",
    "DELETE FROM dataset_manifest_label_evidence",
)


def _build_mixed_manifest(connection: sqlite3.Connection) -> DatasetManifest:
    accepted = seed_lineage(connection, "SITE-A", "PATIENT-A")
    rejected = seed_lineage(connection, "SITE-A", "PATIENT-B")
    for seeded, outcome in ((accepted, "pass"), (rejected, "fail")):
        append_qc_evaluation(
            connection,
            QCEvaluationDraft(
                seeded.measurement_id,
                "qc-v1",
                "engine-v1",
                outcome,
                "{}",
            ),
        )
    return build_dataset_manifest(
        connection,
        DatasetBuildRequest(
            name="immutable",
            members=(
                DatasetMemberDraft(
                    accepted.measurement_id,
                    accepted.sample_label_id,
                    "train",
                    0,
                ),
                DatasetMemberDraft(
                    rejected.measurement_id,
                    rejected.sample_label_id,
                    "test",
                    1,
                ),
            ),
            policy=DatasetPolicy("policy-v1", "qc-v1", ("pass",)),
            preprocessing_version="prep-v1",
            feature_schema_version="grid-v1",
            decision_policy_version="decision-v1",
            provenance=CreationProvenance("pipeline", "revision-1", "training"),
        ),
    )


FROZEN_INSERT_SQL = (
    """INSERT INTO dataset_manifest_specs
       SELECT dataset_manifest_id, preprocessing_version, feature_schema_version,
              decision_policy_version, qc_policy_version, qc_rule_version,
              inclusion_policy_json, exclusion_policy_json, provenance_json
       FROM dataset_manifest_specs LIMIT 1""",
    """INSERT INTO dataset_manifest_items (
           id, dataset_manifest_id, measurement_id, sample_label_id,
           qc_evaluation_id, ordinal
       )
       SELECT 'late-item', dataset_manifest_id, measurement_id, sample_label_id,
              qc_evaluation_id, ordinal + 100
       FROM dataset_manifest_items LIMIT 1""",
    """INSERT INTO dataset_manifest_exclusions (
           id, dataset_manifest_id, measurement_id, sample_label_id,
           artifact_sha256, qc_evaluation_id, reason
       )
       SELECT 'late-exclusion', dataset_manifest_id, measurement_id,
              sample_label_id, artifact_sha256, qc_evaluation_id, reason
       FROM dataset_manifest_exclusions LIMIT 1""",
    """INSERT INTO dataset_manifest_label_evidence (
           dataset_manifest_item_id, sample_label_evidence_id
       )
       SELECT dataset_manifest_item_id, 'late-evidence'
       FROM dataset_manifest_label_evidence LIMIT 1""",
)


@pytest.mark.parametrize("insert_sql", FROZEN_INSERT_SQL)
def test_frozen_manifest_rejects_direct_child_insert_without_export_change(
    tmp_path: Path,
    insert_sql: str,
) -> None:
    # Given: a canonically hashed frozen manifest and its exported rows.
    with database(tmp_path / "insert.db") as connection:
        manifest = _build_mixed_manifest(connection)
        before_path = tmp_path / "before.csv"
        export_dataset_manifest(connection, manifest.id, before_path)

        # When: SQL attempts to append lineage beneath that frozen manifest.
        with pytest.raises(sqlite3.IntegrityError, match="frozen"):
            connection.execute(insert_sql)
        after_path = tmp_path / "after.csv"
        result = export_dataset_manifest(connection, manifest.id, after_path)

        # Then: both the canonical hash and exported bytes remain invariant.
        assert result.content_sha256 == manifest.content_sha256
        assert after_path.read_bytes() == before_path.read_bytes()


@pytest.mark.parametrize("mutation_sql", MUTATION_SQL)
def test_frozen_manifest_lineage_rejects_direct_sql_mutation(
    tmp_path: Path,
    mutation_sql: str,
) -> None:
    # Given: a frozen manifest with included, excluded, and evidence lineage.
    with database(tmp_path / "immutable.db") as connection:
        _build_mixed_manifest(connection)

        # When/Then: direct SQL cannot update or delete any frozen lineage row.
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(mutation_sql)


def test_unreferenced_frozen_manifest_header_cannot_be_deleted(tmp_path: Path) -> None:
    # Given: an unreferenced frozen manifest header.
    with database(tmp_path / "unreferenced.db") as connection:
        connection.execute(
            """INSERT INTO dataset_manifests (
                   id, name, version, content_sha256, status
               ) VALUES ('frozen', 'frozen', 'v1', ?, 'frozen')""",
            ("a" * 64,),
        )

        # When/Then: frozen lifecycle forbids deletion even without child rows.
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "DELETE FROM dataset_manifests WHERE id = 'frozen'"
            )
