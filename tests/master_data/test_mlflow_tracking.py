from __future__ import annotations

from pathlib import Path

from sers.master_data.dataset_export import DatasetExportResult
from sers.mlflow_tracking import tracking_payload


def test_tracking_payload_excludes_paths_and_source_identifiers(tmp_path: Path) -> None:
    # Given: an export result whose local filename contains a pseudonymous source key.
    source_key = "PSEUDONYM-SECRET-001"
    result = DatasetExportResult(
        tmp_path / f"{source_key}.csv",
        "manifest-id",
        "a" * 64,
        4,
        1,
        "preprocess-v1",
        "grid-v1",
        "decision-v1",
        "qc-policy-v1",
    )

    # When: the optional MLflow payload is constructed.
    payload = tracking_payload(result)

    # Then: only manifest/version metadata and aggregate counts are loggable.
    serialized = repr(payload)
    assert source_key not in serialized
    assert str(tmp_path) not in serialized
    assert {name for name, _ in payload.params} == {
        "preprocessing_version",
        "feature_schema_version",
        "decision_policy_version",
        "qc_policy_version",
    }
    assert {name for name, _ in payload.metrics} == {
        "included_rows",
        "excluded_rows",
    }
