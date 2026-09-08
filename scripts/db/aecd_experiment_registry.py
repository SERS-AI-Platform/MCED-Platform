#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mlflow>=3.1,<4",
# ]
# ///

# ─── How to run ───
# Register aggregate experiment history in local MLflow:
#     uv run scripts/db/aecd_experiment_registry.py
# Use --repo-root and --mlflow-db to select another checkout or tracking store.
# ──────────────────

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.db.aecd_experiment_registry_core import (  # noqa: E402
    RegistryRun,
    allowed_artifact,
)
from scripts.db.aecd_experiment_registry_history import (  # noqa: E402
    build_historical_summary_runs,
    build_legacy_log_runs,
)
from scripts.db.aecd_experiment_registry_sources import (  # noqa: E402
    build_aligned_runs,
    build_recent_runs,
)


def _log_runs(runs: Sequence[RegistryRun], tracking_db: Path) -> tuple[str, int, int]:
    import mlflow
    from mlflow.tracking import MlflowClient

    tracking_db.parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{tracking_db.resolve()}")
    experiment_name = "aecd-experiment-registry"
    mlflow.set_experiment(experiment_name)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment was not created: {experiment_name}")
    existing = client.search_runs([experiment.experiment_id], max_results=5000)
    existing_keys = {run.data.tags.get("lineage_key") for run in existing}
    created = 0
    skipped = 0
    for spec in runs:
        if spec.lineage_key in existing_keys:
            skipped += 1
            continue
        with mlflow.start_run(run_name=spec.run_name):
            mlflow.log_params(dict(spec.params))
            if spec.metrics:
                mlflow.log_metrics(dict(spec.metrics))
            mlflow.set_tags(
                {
                    "lineage_key": spec.lineage_key,
                    "source_phase": spec.source_phase,
                    "evidence_status": spec.evidence_status,
                    "purpose": "historical-experiment-registry",
                    "data_policy": "aggregate_only_no_patient_ids_or_raw_uris",
                    "cancer_screening_disclaimer": "hospital_measurement_confounding_possible",
                    **dict(spec.tags),
                }
            )
            artifact_dir = f"evidence/{spec.run_name[:80]}"
            for artifact in spec.artifact_paths:
                if artifact.is_file() and allowed_artifact(artifact):
                    mlflow.log_artifact(str(artifact), artifact_path=artifact_dir)
        existing_keys.add(spec.lineage_key)
        created += 1
    return experiment.experiment_id, created, skipped


def build_all_runs(repo_root: Path) -> list[RegistryRun]:
    runs = build_recent_runs(repo_root)
    runs.extend(build_aligned_runs(repo_root))
    runs.extend(
        build_historical_summary_runs(
            repo_root / "results/mapping_preprocessing_summary_20260826_v1/historical_experiment_summary.csv"
        )
    )
    runs.extend(build_legacy_log_runs(repo_root / "logs/experiment_runs.jsonl"))
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description="Register AECD experiment history in local MLflow")
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument("--mlflow-db", type=Path, default=REPO / "mlflow.db")
    args = parser.parse_args()
    runs = build_all_runs(args.repo_root.resolve())
    experiment_id, created, skipped = _log_runs(runs, args.mlflow_db.resolve())
    print(
        json.dumps(
            {
                "experiment": "aecd-experiment-registry",
                "experiment_id": experiment_id,
                "candidate_runs": len(runs),
                "created_runs": created,
                "existing_runs": skipped,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
