#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mlflow>=3.1,<4",
# ]
# ///

# ─── How to run ───
# Backfill historical model bundles into MLflow:
#     uv run scripts/db/aecd_model_artifact_backfill.py
# Inventory only:
#     uv run scripts/db/aecd_model_artifact_backfill.py --dry-run
# ──────────────────

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.db.aecd_model_artifact_backfill_core import (  # noqa: E402
    discover_model_bundles,
)
from scripts.db.aecd_model_artifact_backfill_mlflow import backfill_model_bundles  # noqa: E402
from scripts.db.aecd_model_artifact_backfill_outputs import write_backfill_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill historical model artifacts into MLflow")
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument("--mlflow-db", type=Path, default=REPO / "mlflow.db")
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=REPO / "results/mlflow_historical_model_backfill_20260827_v1",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    roots = (
        repo_root / "results/legacy/hidden_2026-05-19",
        repo_root / "models/legacy/artifacts",
    )
    bundles = discover_model_bundles(roots)
    if args.dry_run:
        print(json.dumps({"bundles": len(bundles), "model_files": sum(len(bundle.model_files) for bundle in bundles)}, indent=2))
        return
    rows, git_sha, experiment_id = backfill_model_bundles(
        bundles,
        repo_root,
        args.mlflow_db.resolve(),
    )
    write_backfill_report(rows, args.report_dir.resolve(), git_sha)
    print(
        json.dumps(
            {
                "experiment": "aecd-model-artifact-backfill",
                "experiment_id": experiment_id,
                "bundles": len(bundles),
                "model_files": sum(len(bundle.model_files) for bundle in bundles),
                "report_dir": str(args.report_dir.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
