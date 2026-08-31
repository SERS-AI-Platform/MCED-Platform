from __future__ import annotations

from pathlib import Path

import pytest

from sers.master_data.dataset_manifest import build_dataset_manifest
from sers.master_data.lineage_types import (
    CreationProvenance,
    DatasetBuildRequest,
    DatasetMemberDraft,
    DatasetPolicy,
    PredictionRunDraft,
    QCEvaluationDraft,
)
from sers.master_data.prediction_lineage import (
    SensitivePredictionMetadataError,
    append_prediction_run,
)
from sers.master_data.qc_lineage import append_qc_evaluation
from tests.master_data.lineage_test_support import seed_lineage
from tests.master_data.matching_test_support import database


def test_prediction_rejects_known_subject_key_without_echoing_it(tmp_path: Path) -> None:
    # Given: a frozen input whose site-scoped subject key is known to the database.
    sensitive_key = "SITE-SCOPED-PSEUDONYM-777"
    with database(tmp_path / "subject-key.db") as connection:
        seeded = seed_lineage(connection, "SITE-A", sensitive_key)
        append_qc_evaluation(
            connection,
            QCEvaluationDraft(
                seeded.measurement_id,
                "qc-v1",
                "engine-v1",
                "pass",
                "{}",
            ),
        )
        manifest = build_dataset_manifest(
            connection,
            DatasetBuildRequest(
                "privacy",
                (
                    DatasetMemberDraft(
                        seeded.measurement_id,
                        seeded.sample_label_id,
                        "inference",
                        0,
                    ),
                ),
                DatasetPolicy("policy-v1", "qc-v1", ("pass",)),
                "prep-v1",
                "grid-v1",
                "decision-v1",
                CreationProvenance("pipeline", "revision-1", "prediction"),
            ),
        )

        # When: the exact subject key is supplied as prediction metadata.
        with pytest.raises(SensitivePredictionMetadataError) as captured:
            append_prediction_run(
                connection,
                PredictionRunDraft(
                    manifest.id,
                    sensitive_key,
                    "model-v1",
                    "prep-v1",
                    "grid-v1",
                    "decision-v1",
                    (seeded.measurement_id,),
                    "completed",
                    '{"score":0.9}',
                ),
            )

        # Then: persistence is rejected without reflecting the sensitive value.
        assert sensitive_key not in str(captured.value)
        assert connection.execute(
            "SELECT COUNT(*) FROM prediction_runs"
        ).fetchone()[0] == 0


def test_prediction_recursively_rejects_known_sample_code_in_json(
    tmp_path: Path,
) -> None:
    # Given: a sample code present in the site-scoped sample table.
    sample_code = "PRO 001"
    with database(tmp_path / "sample-code.db") as connection:
        seeded = seed_lineage(connection, "SITE-A", sample_code)
        append_qc_evaluation(
            connection,
            QCEvaluationDraft(
                seeded.measurement_id,
                "qc-v1",
                "engine-v1",
                "pass",
                "{}",
            ),
        )
        manifest = build_dataset_manifest(
            connection,
            DatasetBuildRequest(
                "privacy",
                (
                    DatasetMemberDraft(
                        seeded.measurement_id,
                        seeded.sample_label_id,
                        "inference",
                        0,
                    ),
                ),
                DatasetPolicy("policy-v1", "qc-v1", ("pass",)),
                "prep-v1",
                "grid-v1",
                "decision-v1",
                CreationProvenance("pipeline", "revision-1", "prediction"),
            ),
        )

        # When/Then: a deeply nested exact code is rejected.
        with pytest.raises(SensitivePredictionMetadataError):
            append_prediction_run(
                connection,
                PredictionRunDraft(
                    manifest.id,
                    "screening-model",
                    "model-v1",
                    "prep-v1",
                    "grid-v1",
                    "decision-v1",
                    (seeded.measurement_id,),
                    "completed",
                    '{"details":{"inputs":["harmless","PRO 001"]}}',
                ),
            )


def test_prediction_allows_harmless_model_metadata_and_json(tmp_path: Path) -> None:
    # Given: a valid frozen input and ordinary model metadata.
    with database(tmp_path / "harmless.db") as connection:
        seeded = seed_lineage(connection, "SITE-A", "PRO 001")
        append_qc_evaluation(
            connection,
            QCEvaluationDraft(
                seeded.measurement_id,
                "qc-v1",
                "engine-v1",
                "pass",
                "{}",
            ),
        )
        manifest = build_dataset_manifest(
            connection,
            DatasetBuildRequest(
                "privacy",
                (
                    DatasetMemberDraft(
                        seeded.measurement_id,
                        seeded.sample_label_id,
                        "inference",
                        0,
                    ),
                ),
                DatasetPolicy("policy-v1", "qc-v1", ("pass",)),
                "prep-v1",
                "grid-v1",
                "decision-v1",
                CreationProvenance("pipeline", "revision-1", "prediction"),
            ),
        )

        # When: non-identifier metadata and nested output are appended.
        prediction_id = append_prediction_run(
            connection,
            PredictionRunDraft(
                manifest.id,
                "screening-model",
                "model-v1",
                "prep-v1",
                "grid-v1",
                "decision-v1",
                (seeded.measurement_id,),
                "completed",
                '{"details":{"class":"Cancer","scores":[0.9,0.1]}}',
            ),
        )

        # Then: ordinary model names are not rejected by a broad heuristic.
        assert prediction_id
