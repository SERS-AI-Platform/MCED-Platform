from __future__ import annotations

import hashlib
import math
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypeAlias

import mlflow
from mlflow.tracking import MlflowClient

from scripts.db.aecd_experiment_registry_core import allowed_artifact, slug
from scripts.db.aecd_model_artifact_backfill_core import (
    HistoricalModelBundle,
    flatten_mlflow_params,
    model_bundle_hash,
    parameter_changes,
)

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


def _relative(path: Path, repo_root: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def _registry_name(bundle: HistoricalModelBundle, repo_root: Path) -> str:
    relative = _relative(bundle.source_dir, repo_root)
    prefix = "aecd-production-" if relative.startswith("models/legacy/artifacts/") else "aecd-legacy-"
    return f"{prefix}{slug(bundle.model_display_name)[:100]}"


def _metrics(value: JsonValue, prefix: str = "historical") -> dict[str, float]:
    if isinstance(value, dict):
        result: dict[str, float] = {}
        for key, child in value.items():
            result.update(_metrics(child, f"{prefix}_{slug(key)}"))
        return result
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return {}
    number = float(value)
    return {prefix[:250]: number} if math.isfinite(number) else {}


def _params(bundle: HistoricalModelBundle) -> tuple[tuple[str, str], ...]:
    values = dict(flatten_mlflow_params(bundle.metadata))
    values.update(
        {
            "source_model_name": bundle.source_model_name[:250],
            "model_display_name": bundle.model_display_name[:250],
            "source_model_version": bundle.source_version[:250],
            "experiment_label": bundle.experiment_label[:250],
            "metadata_status": "available" if bundle.metadata_path else "not_found",
            "model_file_count": str(len(bundle.model_files)),
        }
    )
    return tuple(sorted(values.items()))


def _artifact_destination(path: Path, bundle: HistoricalModelBundle, root: str) -> str:
    try:
        relative = path.relative_to(bundle.source_dir).parent.as_posix()
    except ValueError:
        relative = "metadata"
    return f"{root}/{relative}" if relative != "." else root


def _evidence_paths(bundle: HistoricalModelBundle) -> tuple[Path, ...]:
    model_files = set(bundle.model_files)
    paths = {
        path
        for path in bundle.source_dir.rglob("*")
        if path.is_file()
        and path not in model_files
        and not path.name.casefold().endswith("_meta.json")
        and allowed_artifact(path)
    }
    if bundle.metadata_path and bundle.metadata_path.is_file():
        paths.add(bundle.metadata_path)
    return tuple(sorted(paths))


def _lineage(bundle: HistoricalModelBundle, repo_root: Path, bundle_hash: str) -> str:
    payload = f"{_relative(bundle.source_dir, repo_root)}\0{bundle_hash}".encode("utf-8")
    return f"historical_model:{hashlib.sha256(payload).hexdigest()}"


def _git_sha(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _ensure_registered_models(client: MlflowClient, names: Sequence[str]) -> None:
    registered = {model.name for model in client.search_registered_models()}
    for name in sorted(set(names)):
        if name not in registered:
            client.create_registered_model(
                name,
                description="Historical AECD checkpoint bundle; registered for lineage, not serving.",
                tags={"artifact_type": "legacy_checkpoint_bundle", "loadable_mlflow_model": "false"},
            )


def _create_run(
    bundle: HistoricalModelBundle,
    repo_root: Path,
    experiment_id: str,
    registry_name: str,
    previous: Mapping[str, str],
    git_sha: str,
) -> tuple[str, int]:
    params = _params(bundle)
    bundle_hash = model_bundle_hash(bundle)
    changes = parameter_changes(previous, dict(params)) if previous else {}
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=f"historical_{slug(bundle.model_display_name)}_{bundle.source_version}_{bundle_hash[:8]}",
    ) as active:
        run_id = active.info.run_id
        mlflow.log_params(dict(params))
        metrics = _metrics(bundle.metadata.get("metrics")) if bundle.metadata else {}
        if metrics:
            mlflow.log_metrics(metrics)
        mlflow.set_tags(
            {
                "lineage_key": _lineage(bundle, repo_root, bundle_hash),
                "source_catalog": "historical_model_artifact_backfill",
                "source_phase": "pre_change_unverified",
                "evidence_status": "historical_checkpoint_with_metadata" if bundle.metadata else "historical_checkpoint_only",
                "source_dir": _relative(bundle.source_dir, repo_root),
                "source_model_version": bundle.source_version,
                "model_bundle_sha256": bundle_hash,
                "registry_model_name": registry_name,
                "artifact_type": "legacy_checkpoint_bundle",
                "loadable_mlflow_model": "false",
                "reconstructed_at_git_sha": git_sha,
                "data_policy": "model_artifact_only_no_prediction_arrays",
                "cancer_screening_disclaimer": "hospital_measurement_confounding_possible",
            }
        )
        for path in bundle.model_files:
            mlflow.log_artifact(str(path), artifact_path=_artifact_destination(path, bundle, "model_bundle"))
        for path in _evidence_paths(bundle):
            mlflow.log_artifact(str(path), artifact_path=_artifact_destination(path, bundle, "evidence"))
        mlflow.log_dict(
            {
                "source_dir": _relative(bundle.source_dir, repo_root),
                "source_model_version": bundle.source_version,
                "model_bundle_sha256": bundle_hash,
                "parameter_change_count": len(changes),
                "parameter_changes": changes,
                "comparison_scope": "chronological historical bundles within registered model family",
            },
            "lineage/parameter_changes.json",
        )
    return run_id, len(changes)


def _backfill_one(
    bundle: HistoricalModelBundle,
    repo_root: Path,
    client: MlflowClient,
    experiment_id: str,
    existing_by_lineage: dict[str, str],
    previous: Mapping[str, str],
    git_sha: str,
) -> dict[str, str]:
    registry_name = _registry_name(bundle, repo_root)
    bundle_hash = model_bundle_hash(bundle)
    lineage = _lineage(bundle, repo_root, bundle_hash)
    run_id = existing_by_lineage.get(lineage)
    if run_id is None:
        run_id, change_count = _create_run(bundle, repo_root, experiment_id, registry_name, previous, git_sha)
        existing_by_lineage[lineage] = run_id
    else:
        change_count = 0
    run = client.get_run(run_id)
    model_version = run.data.tags.get("registry_model_version")
    if not model_version:
        registered = client.create_model_version(
            name=registry_name,
            source=f"runs:/{run_id}/model_bundle",
            run_id=run_id,
            description="Retrospective checkpoint bundle. Raw legacy format; not an MLflow serving model.",
            tags={
                "source_model_version": bundle.source_version,
                "model_bundle_sha256": bundle_hash,
                "artifact_type": "legacy_checkpoint_bundle",
                "loadable_mlflow_model": "false",
            },
        )
        model_version = str(registered.version)
        client.set_tag(run_id, "registry_model_version", model_version)
    return {
        "run_id": run_id,
        "registry_model_name": registry_name,
        "registry_model_version": model_version,
        "source_dir": _relative(bundle.source_dir, repo_root),
        "source_model_version": bundle.source_version,
        "model_display_name": bundle.model_display_name,
        "model_file_count": str(len(bundle.model_files)),
        "metadata_status": "available" if bundle.metadata_path else "not_found",
        "parameter_change_count": str(change_count),
        "model_bundle_sha256": bundle_hash,
    }


def backfill_model_bundles(
    bundles: Sequence[HistoricalModelBundle],
    repo_root: Path,
    tracking_db: Path,
) -> tuple[list[dict[str, str]], str, str]:
    """Log historical checkpoint bundles and create explicitly non-serving registry versions."""
    mlflow.set_tracking_uri(f"sqlite:///{tracking_db.resolve()}")
    experiment_name = "aecd-model-artifact-backfill"
    mlflow.set_experiment(experiment_name)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment was not created: {experiment_name}")
    _ensure_registered_models(client, [_registry_name(bundle, repo_root) for bundle in bundles])
    existing_runs = client.search_runs([experiment.experiment_id], max_results=5000)
    existing_by_lineage = {
        run.data.tags["lineage_key"]: run.info.run_id
        for run in existing_runs
        if run.data.tags.get("lineage_key")
    }
    previous_by_model: dict[str, dict[str, str]] = {}
    rows: list[dict[str, str]] = []
    git_sha = _git_sha(repo_root)
    ordered = sorted(bundles, key=lambda bundle: (bundle.timestamp or "9999", str(bundle.source_dir)))
    for bundle in ordered:
        registry_name = _registry_name(bundle, repo_root)
        row = _backfill_one(
            bundle,
            repo_root,
            client,
            experiment.experiment_id,
            existing_by_lineage,
            previous_by_model.get(registry_name, {}),
            git_sha,
        )
        previous_by_model[registry_name] = dict(_params(bundle))
        rows.append(row)
    return rows, git_sha, experiment.experiment_id
