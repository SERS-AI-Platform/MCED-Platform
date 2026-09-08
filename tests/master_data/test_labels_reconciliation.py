from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

from sers.master_data.labels import (
    GroupCodeLabelRuleError,
    LabelMapping,
    LabelRule,
    derive_sample_labels,
)
from sers.master_data.matching import run_deterministic_matching
from sers.master_data.reconciliation import reconciliation_counts
from tests.master_data.matching_test_support import database, seed_identity, seed_material


def test_labels_are_versioned_idempotent_and_evidence_linked(tmp_path: Path) -> None:
    # Given: one exact match with a pathology observation.
    with database(tmp_path / "labels.db") as connection:
        identity = seed_identity(connection, "SITE-A", "PRO 001")
        seed_material(connection, identity.site_id, "PRO 001")
        run_deterministic_matching(connection, "exact-v1")
        rule = LabelRule(
            task_name="prostate_screening",
            label_definition_version="pathology-v1",
            observation_code="pathology_result",
            label_source="pathology",
            mappings=(LabelMapping("Cancer", "Cancer"),),
        )

        # When: labels are derived twice from the same explicit rule.
        first = derive_sample_labels(connection, rule)
        second = derive_sample_labels(connection, rule)
        label_row = connection.execute(
            """SELECT task_name, label_definition_version, label_value, label_source
               FROM sample_labels"""
        ).fetchone()
        evidence_row = connection.execute(
            """SELECT clinical_observation_id
               FROM sample_label_evidence"""
        ).fetchone()

        # Then: one append-only versioned label retains clinical fact evidence.
        assert first == second == 1
        assert tuple(label_row) == (
            "prostate_screening",
            "pathology-v1",
            "Cancer",
            "pathology",
        )
        assert evidence_row[0] == identity.observation_id


def test_group_code_alone_cannot_define_ground_truth(tmp_path: Path) -> None:
    # Given: a rule that tries to use source-group metadata as clinical truth.
    with database(tmp_path / "group.db") as connection:
        rule = LabelRule(
            task_name="cancer_screening",
            label_definition_version="invalid-v1",
            observation_code="source_group",
            label_source="registry_group",
            mappings=(LabelMapping("PRO", "Cancer"),),
        )

        # When/Then: the label boundary rejects the rule.
        with pytest.raises(GroupCodeLabelRuleError):
            derive_sample_labels(connection, rule)


def test_reconciliation_contains_aggregate_counts_only(tmp_path: Path) -> None:
    # Given: matched and unmatched rows containing a sensitive pseudonymous key.
    sensitive_key = "PSEUDONYMOUS-PATIENT-777"
    with database(tmp_path / "reconciliation.db") as connection:
        identity = seed_identity(connection, "SITE-A", sensitive_key)
        seed_material(connection, identity.site_id, sensitive_key)
        seed_material(connection, identity.site_id, "NO-MATCH")
        run_deterministic_matching(connection, "exact-v1")

        # When: a reconciliation summary is created.
        summary = reconciliation_counts(connection, "exact-v1")
        serialized = repr(asdict(summary))

        # Then: only aggregate state counts leave the boundary.
        assert summary.matched == 1
        assert summary.unmatched_spectrum == 1
        assert sensitive_key not in serialized
        assert set(asdict(summary)) == {
            "matched",
            "unmatched_clinical",
            "unmatched_spectrum",
            "ambiguous",
            "duplicate",
        }
