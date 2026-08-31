from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from sers.master_data.dataset_manifest import (
    ManifestMemberMismatchError,
    build_dataset_manifest,
)
from sers.master_data.lineage_types import (
    CreationProvenance,
    DatasetBuildRequest,
    DatasetMemberDraft,
    DatasetPolicy,
    QCEvaluationDraft,
)
from sers.master_data.qc_lineage import append_qc_evaluation
from tests.master_data.lineage_test_support import seed_lineage
from tests.master_data.matching_test_support import database


def _request(
    members: tuple[DatasetMemberDraft, ...],
    policy: DatasetPolicy,
    *,
    preprocessing_version: str = "baseline-v1",
) -> DatasetBuildRequest:
    return DatasetBuildRequest(
        name="screening-training",
        members=members,
        policy=policy,
        preprocessing_version=preprocessing_version,
        feature_schema_version="raman-grid-v1",
        decision_policy_version="screening-threshold-v1",
        provenance=CreationProvenance(
            created_by="pipeline",
            source_revision="revision-1",
            purpose="model-training",
        ),
    )


def _policy() -> DatasetPolicy:
    return DatasetPolicy(
        version="qc-policy-v1",
        qc_rule_version="spectrum-qc-v1",
        accepted_qc_outcomes=("pass",),
    )


def test_manifest_hash_is_deterministic_across_member_order(tmp_path: Path) -> None:
    # Given: two measurements with accepted append-only QC evaluations.
    with database(tmp_path / "deterministic.db") as connection:
        first = seed_lineage(connection, "SITE-A", "PATIENT-A")
        second = seed_lineage(connection, "SITE-A", "PATIENT-B")
        for seeded in (first, second):
            append_qc_evaluation(
                connection,
                QCEvaluationDraft(
                    measurement_id=seeded.measurement_id,
                    rule_version="spectrum-qc-v1",
                    evaluation_version="engine-v1",
                    outcome="pass",
                    metrics_json='{"snr":12.5}',
                ),
            )
        members = (
            DatasetMemberDraft(first.measurement_id, first.sample_label_id, "train", 0),
            DatasetMemberDraft(second.measurement_id, second.sample_label_id, "test", 1),
        )

        # When: identical content is built in opposite caller order.
        forward = build_dataset_manifest(connection, _request(members, _policy()))
        reverse = build_dataset_manifest(
            connection,
            _request(tuple(reversed(members)), _policy()),
        )

        # Then: identity, hash, and persisted membership are identical.
        assert reverse == forward
        assert forward.included_count == 2
        assert forward.excluded_count == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM dataset_manifests"
        ).fetchone()[0] == 1
        persisted = tuple(
            tuple(row)
            for row in connection.execute(
                """SELECT measurement_id, ordinal
                   FROM dataset_manifest_items ORDER BY ordinal"""
            )
        )
        assert persisted == tuple(
            (measurement_id, ordinal)
            for ordinal, measurement_id in enumerate(
                sorted((first.measurement_id, second.measurement_id))
            )
        )


def test_manifest_binds_exact_artifact_label_evidence_and_qc(tmp_path: Path) -> None:
    # Given: one labeled measurement and its exact accepted QC evaluation.
    with database(tmp_path / "lineage.db") as connection:
        seeded = seed_lineage(connection, "SITE-A", "PATIENT-A")
        qc_id = append_qc_evaluation(
            connection,
            QCEvaluationDraft(
                measurement_id=seeded.measurement_id,
                rule_version="spectrum-qc-v1",
                evaluation_version="engine-v3",
                outcome="pass",
                metrics_json='{"snr":15}',
            ),
        )

        # When: a frozen manifest is built.
        manifest = build_dataset_manifest(
            connection,
            _request(
                (
                    DatasetMemberDraft(
                        seeded.measurement_id,
                        seeded.sample_label_id,
                        "validation",
                        2,
                    ),
                ),
                _policy(),
            ),
        )
        row = connection.execute(
            """SELECT dmi.measurement_id, dmi.artifact_sha256,
                      dmi.sample_label_id, dmi.label_definition_version,
                      dmi.qc_evaluation_id, dmi.qc_rule_version,
                      dmi.qc_evaluation_version, dmi.split_name, dmi.fold_index,
                      dmle.sample_label_evidence_id
               FROM dataset_manifest_items AS dmi
               JOIN dataset_manifest_label_evidence AS dmle
                 ON dmle.dataset_manifest_item_id = dmi.id
               WHERE dmi.dataset_manifest_id = ?""",
            (manifest.id,),
        ).fetchone()

        # Then: every immutable input and its split/fold are queryable.
        assert tuple(row) == (
            seeded.measurement_id,
            seeded.artifact_sha256,
            seeded.sample_label_id,
            "pathology-v1",
            qc_id,
            "spectrum-qc-v1",
            "engine-v3",
            "validation",
            2,
            seeded.evidence_id,
        )


def test_invalid_qc_inputs_are_excluded_and_change_identity(tmp_path: Path) -> None:
    # Given: accepted, QC-failed, and measurement-invalid inputs.
    with database(tmp_path / "qc-policy.db") as connection:
        accepted = seed_lineage(connection, "SITE-A", "PATIENT-A")
        failed = seed_lineage(connection, "SITE-A", "PATIENT-B")
        invalid = seed_lineage(
            connection,
            "SITE-A",
            "PATIENT-C",
            measurement_status="invalid",
        )
        for seeded, outcome in ((accepted, "pass"), (failed, "fail"), (invalid, "pass")):
            append_qc_evaluation(
                connection,
                QCEvaluationDraft(
                    measurement_id=seeded.measurement_id,
                    rule_version="spectrum-qc-v1",
                    evaluation_version="engine-v1",
                    outcome=outcome,
                    metrics_json="{}",
                ),
            )
        members = tuple(
            DatasetMemberDraft(item.measurement_id, item.sample_label_id, "train", 0)
            for item in (accepted, failed, invalid)
        )

        # When: the explicit QC policy builds a manifest, then a version changes.
        initial = build_dataset_manifest(connection, _request(members, _policy()))
        changed = build_dataset_manifest(
            connection,
            _request(members, _policy(), preprocessing_version="baseline-v2"),
        )
        exclusions = tuple(
            row[0]
            for row in connection.execute(
                """SELECT reason FROM dataset_manifest_exclusions
                   WHERE dataset_manifest_id = ? ORDER BY reason""",
                (initial.id,),
            )
        )

        # Then: only valid accepted input is included and version drift creates a new ID.
        assert (initial.included_count, initial.excluded_count) == (1, 2)
        assert exclusions == ("measurement_invalid", "qc_outcome_rejected")
        assert changed.id != initial.id
        assert changed.content_sha256 != initial.content_sha256


def test_manifest_validation_rolls_back_without_partial_rows(tmp_path: Path) -> None:
    # Given: a label belonging to another measurement's sample.
    with database(tmp_path / "rollback.db") as connection:
        first = seed_lineage(connection, "SITE-A", "PATIENT-A")
        second = seed_lineage(connection, "SITE-B", "PATIENT-A")
        append_qc_evaluation(
            connection,
            QCEvaluationDraft(
                measurement_id=first.measurement_id,
                rule_version="spectrum-qc-v1",
                evaluation_version="engine-v1",
                outcome="pass",
                metrics_json="{}",
            ),
        )

        # When: mismatched cross-site lineage crosses the manifest boundary.
        with pytest.raises(ManifestMemberMismatchError):
            build_dataset_manifest(
                connection,
                _request(
                    (
                        DatasetMemberDraft(
                            first.measurement_id,
                            second.sample_label_id,
                            "train",
                            0,
                        ),
                    ),
                    _policy(),
                ),
            )

        # Then: no partial manifest records remain.
        assert connection.execute(
            "SELECT COUNT(*) FROM dataset_manifests"
        ).fetchone()[0] == 0


def test_manifest_sql_failure_rolls_back_header_and_members(tmp_path: Path) -> None:
    # Given: a valid member and a database fault after manifest header insertion.
    with database(tmp_path / "atomicity.db") as connection:
        seeded = seed_lineage(connection, "SITE-A", "PATIENT-A")
        append_qc_evaluation(
            connection,
            QCEvaluationDraft(
                seeded.measurement_id,
                "spectrum-qc-v1",
                "engine-v1",
                "pass",
                "{}",
            ),
        )
        connection.execute(
            """CREATE TRIGGER fail_manifest_spec
               BEFORE INSERT ON dataset_manifest_specs
               BEGIN SELECT RAISE(ABORT, 'injected failure'); END"""
        )

        # When: persistence fails inside its savepoint.
        with pytest.raises(sqlite3.IntegrityError):
            build_dataset_manifest(
                connection,
                _request(
                    (
                        DatasetMemberDraft(
                            seeded.measurement_id,
                            seeded.sample_label_id,
                            "train",
                            0,
                        ),
                    ),
                    _policy(),
                ),
            )

        # Then: neither header nor membership leaks from the failed build.
        assert connection.execute(
            "SELECT COUNT(*) FROM dataset_manifests"
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM dataset_manifest_items"
        ).fetchone()[0] == 0
