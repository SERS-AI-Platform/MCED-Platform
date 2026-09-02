#!/usr/bin/env python
"""Import logs/experiment_registry.json into aecd_platform experiment.runs.

The registry is the git-tracked append-only SSOT (see CLAUDE.md "실험 히스토리
기록"); this loads it into the database so experiment history is queryable and
joinable against clinical/measurement data. The JSON stays authoritative — this
import is idempotent and can be re-run after the registry is appended to.

IMPORTANT: `result_summary` is free-text prose (e.g. "Det AUC 0.793, Id F1
0.356"). It is stored verbatim in experiment.runs.result_summary and is NEVER
parsed into experiment.run_metrics — deriving structured metrics from prose
would be inventing numbers. run_metrics is populated only by runs that produce
real structured metrics.

Usage:
    python scripts/db/experiment_tracking/import_registry.py --dry-run
    python scripts/db/experiment_tracking/import_registry.py
"""

from __future__ import annotations

import argparse
import json
from contextlib import closing
from pathlib import Path
from typing import Any

import psycopg2
from psycopg2.extras import execute_values

from sers.aecd_api.repository import DatabaseSettings

REGISTRY_PATH = Path(__file__).resolve().parents[3] / "logs" / "experiment_registry.json"

INSERT_SQL = """
INSERT INTO experiment.runs (
    run_name, phase, run_date, hypothesis, variable, baseline,
    cancer_types, non_cancer_groups, aggregation, models, tags,
    n_samples, result_summary, artifacts_dir, status
) VALUES %s
ON CONFLICT (run_name) DO UPDATE SET
    phase = EXCLUDED.phase,
    run_date = EXCLUDED.run_date,
    hypothesis = EXCLUDED.hypothesis,
    variable = EXCLUDED.variable,
    baseline = EXCLUDED.baseline,
    cancer_types = EXCLUDED.cancer_types,
    non_cancer_groups = EXCLUDED.non_cancer_groups,
    aggregation = EXCLUDED.aggregation,
    models = EXCLUDED.models,
    tags = EXCLUDED.tags,
    n_samples = EXCLUDED.n_samples,
    result_summary = EXCLUDED.result_summary,
    artifacts_dir = EXCLUDED.artifacts_dir
"""


def _row(entry: dict[str, Any]) -> tuple[Any, ...]:
    return (
        entry["name"],
        entry.get("phase"),
        entry.get("date"),
        entry.get("hypothesis"),
        entry.get("variable"),
        entry.get("baseline"),
        entry.get("cancer_types") or None,
        entry.get("non_cancer_groups") or None,
        entry.get("aggregation"),
        entry.get("models") or None,
        entry.get("tags") or None,
        entry.get("n_samples"),
        entry.get("result_summary"),
        entry.get("artifacts_dir"),
        "complete",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    parser.add_argument("--dry-run", action="store_true", help="print what would be written")
    args = parser.parse_args()

    entries = json.loads(args.registry.read_text())["experiments"]
    rows = [_row(entry) for entry in entries]
    print(f"registry: {args.registry}")
    print(f"entries : {len(rows)}")

    if args.dry_run:
        for row in rows[:3]:
            print(f"  sample: name={row[0]!r} phase={row[1]!r} date={row[2]!r} n_samples={row[11]!r}")
        print(f"  ... ({len(rows)} total). --dry-run: nothing written.")
        return

    settings = DatabaseSettings.from_environment()
    if settings.database != "aecd_platform":
        raise SystemExit(f"refusing to write to database {settings.database!r}; expected aecd_platform")

    with closing(
        psycopg2.connect(
            host=settings.host,
            port=settings.port,
            dbname=settings.database,
            user=settings.user,
            password=settings.password,
            connect_timeout=5,
        )
    ) as connection, connection.cursor() as cursor:
        execute_values(cursor, INSERT_SQL, rows)
        connection.commit()
        cursor.execute("SELECT COUNT(*) FROM experiment.runs")
        total = cursor.fetchone()[0]
    print(f"upserted: {len(rows)}  |  experiment.runs now holds {total} rows")


if __name__ == "__main__":
    main()
