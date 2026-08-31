#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "matplotlib>=3.7",
#     "mlflow==3.15.2",
#     "typing_extensions>=4.0",
# ]
# ///

# ─── How to run ───
# 1. Install uv (if not installed):
#      curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Run directly (no venv, no pip install needed):
#      uv run aecd_model_lineage_timeline.py
# 3. Or make executable and run:
#      chmod +x aecd_model_lineage_timeline.py && ./aecd_model_lineage_timeline.py
# ──────────────────

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import mlflow
from mlflow.entities import Experiment, Run
from mlflow.tracking import MlflowClient

from scripts.db.aecd_model_lineage_timeline_core import (
    ParameterChangeRow,
    TimelineRow,
    TimelineRun,
    build_timeline,
    parameter_change_rows,
)
from scripts.db.aecd_model_lineage_timeline_outputs import write_timeline_outputs
from scripts.db.aecd_model_lineage_timeline_plots import (
    save_parameter_change_figure,
    save_performance_timeline,
    save_version_timeline,
)

TIMELINE_VERSION = "v1"
TIMELINE_KEY = "aecd-model-lineage-timeline-v1"


@dataclass(frozen=True, slots=True)
class TimelineBundle:
    source_experiment_id: str
    report_dir: Path
    rows: tuple[TimelineRow, ...]
    changes: tuple[ParameterChangeRow, ...]


def _source_runs(client: MlflowClient, experiment_id: str) -> tuple[TimelineRun, ...]:
    runs = client.search_runs([experiment_id], max_results=5000, order_by=["attributes.start_time ASC"])
    return tuple(
        TimelineRun(
            run_id=run.info.run_id,
            registry_model_name=run.data.tags.get("registry_model_name", "unknown"),
            registry_model_version=run.data.tags.get("registry_model_version", ""),
            source_dir=run.data.tags.get("source_dir", ""),
            source_model_version=run.data.params.get("source_model_version", ""),
            model_display_name=run.data.params.get("model_display_name", "unknown"),
            params=dict(run.data.params),
            metrics=dict(run.data.metrics),
        )
        for run in runs
    )


def _source_experiment(client: MlflowClient, name: str) -> Experiment:
    experiment = client.get_experiment_by_name(name)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment was not found: {name}")
    return experiment


def _timeline_run(client: MlflowClient, experiment_id: str) -> Run | None:
    runs = client.search_runs([experiment_id], max_results=5000, order_by=["attributes.start_time DESC"])
    return next((run for run in runs if run.data.tags.get("timeline_key") == TIMELINE_KEY), None)


def _log_timeline_contents(bundle: TimelineBundle, run_id: str, client: MlflowClient) -> None:
    observed = sum(row.source_time_status == "observed" for row in bundle.rows)
    mlflow.log_params(
        {
            "source_experiment_id": bundle.source_experiment_id,
            "timeline_version": TIMELINE_VERSION,
            "timeline_order": "source_timestamp_with_missing_last_per_family",
            "timeline_timestamp_assumption": "timezone_naive_source_timestamps_interpreted_as_UTC",
            "timeline_chart_contract": "version_progression_parameter_changes_performance_evidence",
        }
    )
    mlflow.set_tags(
        {
            "timeline_key": TIMELINE_KEY,
            "artifact_type": "model_lineage_timeline",
            "source_catalog": "historical_model_artifact_backfill",
            "source_phase": "pre_change_unverified",
            "evidence_status": "lineage_visualization",
            "data_policy": "model_lineage_only_no_patient_ids_or_raw_uris",
            "cancer_screening_disclaimer": "hospital_measurement_confounding_possible",
        }
    )
    summary = {
        "timeline_bundle_count": float(len(bundle.rows)),
        "timeline_family_count": float(len({row.registry_model_name for row in bundle.rows})),
        "timeline_dated_bundle_count": float(observed),
        "timeline_parameter_count": float(len(bundle.changes)),
    }
    for key, value in summary.items():
        if not client.get_metric_history(run_id, key):
            client.log_metric(run_id, key, value, step=0)
    for path in sorted(bundle.report_dir.iterdir()):
        if path.is_file() and path.suffix.casefold() in {".csv", ".md", ".png"}:
            mlflow.log_artifact(str(path), artifact_path="timeline")


def _upsert_timeline_run(client: MlflowClient, experiment: Experiment, bundle: TimelineBundle) -> str:
    existing = _timeline_run(client, experiment.experiment_id)
    if existing is None:
        with mlflow.start_run(
            experiment_id=experiment.experiment_id,
            run_name="historical_model_lineage_timeline_v1",
        ) as active:
            _log_timeline_contents(bundle, active.info.run_id, client)
            return active.info.run_id
    with mlflow.start_run(run_id=existing.info.run_id) as active:
        _log_timeline_contents(bundle, active.info.run_id, client)
        return active.info.run_id


def _log_source_lineage(client: MlflowClient, row: TimelineRow, timeline_run_id: str) -> int:
    for key, value in {
        "timeline_run_id": timeline_run_id,
        "timeline_version": TIMELINE_VERSION,
        "timeline_source_timestamp": row.source_timestamp or "missing",
        "timeline_source_time_status": row.source_time_status,
        "timeline_history_order": str(row.history_order),
        "timeline_hyperparameter_change_count": str(row.hyperparameter_change_count),
    }.items():
        client.set_tag(row.run_id, key, value)
    if row.source_timestamp_epoch is None:
        return 0
    timestamp_ms = int(row.source_timestamp_epoch * 1000)
    metric_values: list[tuple[str, float]] = [
        ("lineage_source_timestamp_epoch", row.source_timestamp_epoch),
        ("lineage_history_order", float(row.history_order)),
        ("lineage_hyperparameter_change_count", float(row.hyperparameter_change_count)),
    ]
    if row.cancer_screening_auc is not None:
        metric_values.append(("lineage_cancer_screening_auc", row.cancer_screening_auc))
    if row.cancer_type_id_auc is not None:
        metric_values.append(("lineage_cancer_type_id_auc", row.cancer_type_id_auc))
    logged = 0
    for key, value in metric_values:
        history = client.get_metric_history(row.run_id, key)
        if any(point.step == row.history_order for point in history):
            continue
        client.log_metric(row.run_id, key, value, timestamp=timestamp_ms, step=row.history_order)
        logged += 1
    return logged


def _build_bundle(
    client: MlflowClient,
    source_experiment_id: str,
    report_dir: Path,
) -> TimelineBundle:
    runs = _source_runs(client, source_experiment_id)
    rows = build_timeline(runs)
    changes = parameter_change_rows(rows)
    return TimelineBundle(
        source_experiment_id=source_experiment_id,
        report_dir=report_dir,
        rows=rows,
        changes=changes,
    )


def _run(args: argparse.Namespace) -> dict[str, int | str]:
    mlflow.set_tracking_uri(f"sqlite:///{args.mlflow_db.resolve()}")
    client = MlflowClient()
    source = _source_experiment(client, args.source_experiment)
    bundle = _build_bundle(client, source.experiment_id, args.report_dir.resolve())
    write_timeline_outputs(bundle.rows, bundle.changes, bundle.report_dir)
    save_version_timeline(bundle.rows, bundle.report_dir / "fig01_model_version_timeline.png")
    save_parameter_change_figure(bundle.rows, bundle.changes, bundle.report_dir / "fig02_parameter_change_matrix.png")
    save_performance_timeline(bundle.rows, bundle.report_dir / "fig03_performance_timeline.png")
    timeline_name = args.timeline_experiment
    mlflow.set_experiment(timeline_name)
    timeline_experiment = _source_experiment(client, timeline_name)
    timeline_run_id = _upsert_timeline_run(client, timeline_experiment, bundle)
    logged_metrics = sum(_log_source_lineage(client, row, timeline_run_id) for row in bundle.rows)
    return {
        "source_experiment": args.source_experiment,
        "source_experiment_id": source.experiment_id,
        "timeline_experiment": timeline_name,
        "timeline_experiment_id": timeline_experiment.experiment_id,
        "timeline_run_id": timeline_run_id,
        "bundles": len(bundle.rows),
        "families": len({row.registry_model_name for row in bundle.rows}),
        "dated_bundles": sum(row.source_time_status == "observed" for row in bundle.rows),
        "parameter_change_rows": len(bundle.changes),
        "lineage_metric_points_logged": logged_metrics,
        "report_dir": str(bundle.report_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a source-time MLflow model lineage timeline")
    parser.add_argument("--mlflow-db", type=Path, default=REPO / "mlflow.db")
    parser.add_argument("--source-experiment", default="aecd-model-artifact-backfill")
    parser.add_argument("--timeline-experiment", default="aecd-model-lineage-timeline")
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=REPO / "results/mlflow_model_lineage_timeline_20260827_v1",
    )
    args = parser.parse_args()
    print(json.dumps(_run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
