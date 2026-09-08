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
from sers.master_data.prediction_lineage import (
    SensitivePredictionMetadataError,
    append_prediction_run,
)
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
            seeded.measurement_id,
            "qc-v1",
            "engine-v1",
            "pass",
            "{}",
        ),
    )
    return build_dataset_manifest(
        connection,
        DatasetBuildRequest(
            "privacy-substrings",
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


@pytest.mark.parametrize(
    "output_json",
    (
        '{"details":{"trace-PRO 001-suffix":"harmless"}}',
        '{"details":{"inputs":["harmless","prefix-PRO 001-suffix"]}}',
    ),
)
def test_prediction_rejects_identifier_substrings_in_nested_keys_and_lists(
    tmp_path: Path,
    output_json: str,
) -> None:
    # Given: a database-known source code and nested JSON containing that code.
    with database(tmp_path / "nested.db") as connection:
        seeded = seed_lineage(connection, "SITE-A", "PRO 001")
        manifest = _manifest(connection, seeded)

        # When/Then: key and sequence traversal reject embedded identifiers.
        with pytest.raises(SensitivePredictionMetadataError) as captured:
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
                    output_json,
                ),
            )
        assert "PRO 001" not in str(captured.value)
        assert output_json not in str(captured.value)


def test_prediction_rejects_identifier_embedded_in_metadata_text(tmp_path: Path) -> None:
    # Given: a known site-scoped subject key.
    sensitive_key = "SITE-PSEUDONYM-777"
    with database(tmp_path / "metadata.db") as connection:
        seeded = seed_lineage(connection, "SITE-A", sensitive_key)
        manifest = _manifest(connection, seeded)

        # When/Then: a prefixed and suffixed metadata string is rejected.
        with pytest.raises(SensitivePredictionMetadataError) as captured:
            append_prediction_run(
                connection,
                PredictionRunDraft(
                    manifest.id,
                    f"model-prefix-{sensitive_key}-suffix",
                    "model-v1",
                    "prep-v1",
                    "grid-v1",
                    "decision-v1",
                    (seeded.measurement_id,),
                    "completed",
                    '{"score":0.9}',
                ),
            )
        assert sensitive_key not in str(captured.value)


def test_prediction_allows_harmless_near_match_without_normalization(
    tmp_path: Path,
) -> None:
    # Given: one exact DB-known source code.
    with database(tmp_path / "near-match.db") as connection:
        seeded = seed_lineage(connection, "SITE-A", "PRO 001")
        manifest = _manifest(connection, seeded)

        # When: similar but non-containing source-like text is supplied.
        prediction_id = append_prediction_run(
            connection,
            PredictionRunDraft(
                manifest.id,
                "screening-model-PRO 002",
                "model-v1",
                "prep-v1",
                "grid-v1",
                "decision-v1",
                (seeded.measurement_id,),
                "completed",
                '{"details":{"source_like":"pro 001"}}',
            ),
        )

        # Then: no case-folding, normalization, or source-code rewrite is applied.
        assert prediction_id
