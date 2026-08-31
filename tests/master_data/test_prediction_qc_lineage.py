from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from sers.master_data.dataset_manifest import build_dataset_manifest
from sers.master_data.lineage_types import (
    CreationProvenance,
    DatasetBuildRequest,
    DatasetManifest,
    DatasetMemberDraft,
    DatasetPolicy,
    PredictionRunDraft,
    QCEvaluationDraft,
)
from sers.master_data.prediction_lineage import append_prediction_run
from sers.master_data.qc_lineage import append_qc_evaluation
from tests.master_data.lineage_test_support import SeededLineage, seed_lineage
from tests.master_data.matching_test_support import database


def _manifest(
    connection: sqlite3.Connection,
    seeded: SeededLineage,
) -> DatasetManifest:
    append_qc_evaluation(
        connection,
        QCEvaluationDraft(
            measurement_id=seeded.measurement_id,
            rule_version="spectrum-qc-v1",
            evaluation_version="engine-v1",
            outcome="pass",
            metrics_json="{}",
        ),
    )
    return build_dataset_manifest(
        connection,
        DatasetBuildRequest(
            name="prediction-input",
            members=(
                DatasetMemberDraft(
                    seeded.measurement_id,
                    seeded.sample_label_id,
                    "inference",
                    0,
                ),
            ),
            policy=DatasetPolicy(
                version="qc-policy-v1",
                qc_rule_version="spectrum-qc-v1",
                accepted_qc_outcomes=("pass",),
            ),
            preprocessing_version="baseline-v1",
            feature_schema_version="raman-grid-v1",
            decision_policy_version="threshold-v1",
            provenance=CreationProvenance("pipeline", "revision-1", "prediction"),
        ),
    )


def test_qc_reruns_are_append_only_by_rule_version(tmp_path: Path) -> None:
    # Given: one physical measurement.
    with database(tmp_path / "qc.db") as connection:
        seeded = seed_lineage(connection, "SITE-A", "PATIENT-A")

        # When: it is evaluated repeatedly under current and changed QC rules.
        first = append_qc_evaluation(
            connection,
            QCEvaluationDraft(
                seeded.measurement_id,
                "spectrum-qc-v1",
                "engine-v1",
                "pass",
                '{"snr":12}',
            ),
        )
        second = append_qc_evaluation(
            connection,
            QCEvaluationDraft(
                seeded.measurement_id,
                "spectrum-qc-v1",
                "engine-v2",
                "review",
                '{"snr":9}',
            ),
        )
        third = append_qc_evaluation(
            connection,
            QCEvaluationDraft(
                seeded.measurement_id,
                "spectrum-qc-v2",
                "engine-v2",
                "fail",
                '{"snr":9}',
            ),
        )

        # Then: all immutable evaluations remain independently addressable.
        assert len({first, second, third}) == 3
        assert tuple(
            tuple(row)
            for row in connection.execute(
                """SELECT rule_version, evaluation_version, outcome
                   FROM qc_evaluations ORDER BY rowid"""
            )
        ) == (
            ("spectrum-qc-v1", "engine-v1", "pass"),
            ("spectrum-qc-v1", "engine-v2", "review"),
            ("spectrum-qc-v2", "engine-v2", "fail"),
        )


def test_prediction_reruns_append_exact_manifest_and_inputs(tmp_path: Path) -> None:
    # Given: one immutable dataset manifest.
    with database(tmp_path / "prediction.db") as connection:
        seeded = seed_lineage(connection, "SITE-A", "PATIENT-A")
        manifest = _manifest(connection, seeded)
        draft = PredictionRunDraft(
            manifest_id=manifest.id,
            model_name="screening-model",
            model_version="model-v4",
            preprocessing_version="baseline-v1",
            feature_schema_version="raman-grid-v1",
            decision_policy_version="threshold-v1",
            measurement_ids=(seeded.measurement_id,),
            status="completed",
            output_json='{"score":0.91}',
        )

        # When: the same model execution is recorded twice.
        first = append_prediction_run(connection, draft)
        second = append_prediction_run(connection, draft)
        rows = tuple(
            connection.execute(
                """SELECT id, dataset_manifest_id, model_version,
                          preprocessing_version, feature_schema_version,
                          decision_policy_version, input_set_sha256
                   FROM prediction_runs ORDER BY rowid"""
            )
        )

        # Then: reruns append, retain exact versions, and share deterministic input hash.
        assert first != second
        assert len(rows) == 2
        assert rows[0][1:6] == (
            manifest.id,
            "model-v4",
            "baseline-v1",
            "raman-grid-v1",
            "threshold-v1",
        )
        assert rows[0][6] == rows[1][6]
        assert tuple(
            tuple(row)
            for row in connection.execute(
                """SELECT prediction_run_id, measurement_id
                   FROM prediction_input_measurements ORDER BY prediction_run_id"""
            )
        ) == tuple(
            sorted(
                (
                    (first, seeded.measurement_id),
                    (second, seeded.measurement_id),
                )
            )
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "UPDATE prediction_runs SET model_version = 'overwritten' WHERE id = ?",
                (first,),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "UPDATE qc_evaluations SET outcome = 'fail'"
            )


def test_lineage_surfaces_do_not_copy_site_subject_keys(tmp_path: Path) -> None:
    # Given: identical sensitive pseudonymous keys scoped to two sites.
    sensitive_key = "PSEUDONYMOUS-PATIENT-777"
    with database(tmp_path / "privacy.db") as connection:
        first = seed_lineage(connection, "SITE-A", sensitive_key)
        second = seed_lineage(connection, "SITE-B", sensitive_key)

        # When: manifest and prediction lineage are persisted for one site.
        manifest = _manifest(connection, first)
        append_prediction_run(
            connection,
            PredictionRunDraft(
                manifest.id,
                "screening-model",
                "model-v1",
                "baseline-v1",
                "raman-grid-v1",
                "threshold-v1",
                (first.measurement_id,),
                "completed",
                '{"class":"Cancer"}',
            ),
        )
        lineage_values = "\n".join(
            str(value)
            for table in (
                "dataset_manifests",
                "dataset_manifest_specs",
                "dataset_manifest_items",
                "dataset_manifest_exclusions",
                "prediction_runs",
                "prediction_input_measurements",
            )
            for row in connection.execute(f"SELECT * FROM {table}")
            for value in row
        )

        # Then: no subject/source code is copied into model lineage tables.
        assert second.measurement_id != first.measurement_id
        assert sensitive_key not in lineage_values
