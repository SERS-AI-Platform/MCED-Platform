from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

from sers.master_data.dataset_export import DatasetExportResult


@dataclass(frozen=True, slots=True)
class MLflowUnavailableError(RuntimeError):
    def __str__(self) -> str:
        return "MLflow is unavailable; install the mlops optional dependency"


@dataclass(frozen=True, slots=True)
class SafeTrackingPayload:
    params: tuple[tuple[str, str], ...]
    metrics: tuple[tuple[str, float], ...]
    tags: tuple[tuple[str, str], ...]


def mlflow_available() -> bool:
    return importlib.util.find_spec("mlflow") is not None


def tracking_payload(result: DatasetExportResult) -> SafeTrackingPayload:
    return SafeTrackingPayload(
        params=(
            ("preprocessing_version", result.preprocessing_version),
            ("feature_schema_version", result.feature_schema_version),
            ("decision_policy_version", result.decision_policy_version),
            ("qc_policy_version", result.qc_policy_version),
        ),
        metrics=(
            ("included_rows", float(result.row_count)),
            ("excluded_rows", float(result.excluded_count)),
        ),
        tags=(
            ("manifest_id", result.manifest_id),
            ("manifest_sha256", result.content_sha256),
            ("purpose", "dataset-export"),
        ),
    )


def log_dataset_export(tracking_db: Path, result: DatasetExportResult) -> None:
    try:
        import mlflow
    except ModuleNotFoundError as error:
        raise MLflowUnavailableError from error
    payload = tracking_payload(result)
    tracking_db.parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{tracking_db.resolve()}")
    mlflow.set_experiment("sers-dataset-export")
    with mlflow.start_run():
        mlflow.log_params(dict(payload.params))
        mlflow.log_metrics(dict(payload.metrics))
        mlflow.set_tags(dict(payload.tags))
        mlflow.log_artifact(str(result.output_path))
