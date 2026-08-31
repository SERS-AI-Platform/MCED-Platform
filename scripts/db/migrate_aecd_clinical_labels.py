#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "psycopg2-binary>=2.9,<3",
#     "pydantic>=2,<3",
#     "typer>=0.12,<1",
#     "typing-extensions>=4.12,<5",
# ]
# ///

# ─── How to run ───
# 1. Install uv (if not installed):
#      curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Preview: uv run scripts/db/migrate_aecd_clinical_labels.py --mode check-db
# 3. Write only after approval: add --mode write --confirm-version <version>
# ──────────────────

# pyright: reportImplicitRelativeImport=false, reportUnnecessaryComparison=false

from __future__ import annotations

import json
import os
from contextlib import closing
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated, Final

import psycopg2
import typer
from psycopg2.extensions import connection as PgConnection
from pydantic import TypeAdapter
from typing_extensions import assert_never

try:
    from scripts.db.aecd_label_contract import (
        ClinicalSource,
        LabelRule,
        MigrationError,
        MigrationPlan,
        load_rules,
        load_sources,
        resolve_labels,
    )
except ModuleNotFoundError:
    from aecd_label_contract import (
        ClinicalSource,
        LabelRule,
        MigrationError,
        MigrationPlan,
        load_rules,
        load_sources,
        resolve_labels,
    )

DEFAULT_INPUT: Final = Path("data/processed/aecd_platform_ingest/smcxd07_clinical_master.csv")
DEFAULT_MAPPING: Final = Path("config/aecd_label_mapping_v1.csv")
DATABASE_ROWS: Final = TypeAdapter(list[tuple[int, str]])


@dataclass(frozen=True, slots=True)
class DatabaseConfig:
    host: str
    port: int
    database: str
    user: str
    password: str


class Mode(str, Enum):
    CHECK_DB = "check-db"
    WRITE = "write"
    ROLLBACK = "rollback"


def database_settings() -> DatabaseConfig:
    password = os.environ.get("PGPASSWORD", "")
    if not password:
        raise MigrationError("Set PGPASSWORD in the environment")
    return DatabaseConfig(
        host=os.environ.get("PGHOST", "localhost"),
        port=int(os.environ.get("PGPORT", "5432")),
        database=os.environ.get("PGDATABASE", "aecd_platform"),
        user=os.environ.get("PGUSER", "postgres"),
        password=password,
    )


def connect(read_only: bool) -> PgConnection:
    config = database_settings()
    options = "-c statement_timeout=30000 -c lock_timeout=5000"
    if read_only:
        options += " -c default_transaction_read_only=on"
    return psycopg2.connect(
        host=config.host,
        port=config.port,
        dbname=config.database,
        user=config.user,
        password=config.password,
        connect_timeout=10,
        options=options,
    )


def build_plan(
    connection: PgConnection,
    sources: tuple[ClinicalSource, ...],
    rules: tuple[LabelRule, ...],
) -> MigrationPlan:
    with connection.cursor() as cursor:
        cursor.execute(
            """SELECT diagnosis.diagnosis_id, sample.solum_label
               FROM clinical.diagnoses AS diagnosis
               JOIN master.samples AS sample USING (subject_id)"""
        )
        database_rows = tuple(DATABASE_ROWS.validate_python(cursor.fetchall()))
    return resolve_labels(
        database_rows,
        sources,
        rules,
    )


def apply_plan(connection: PgConnection, plan: MigrationPlan) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            """ALTER TABLE clinical.diagnoses
               ADD COLUMN IF NOT EXISTS source_cancer_file_raw text,
               ADD COLUMN IF NOT EXISTS study_cancer_type text,
               ADD COLUMN IF NOT EXISTS diagnosed_cancer_type text,
               ADD COLUMN IF NOT EXISTS case_status text
                   CONSTRAINT diagnoses_case_status_check
                   CHECK (case_status IS NULL OR case_status IN
                       ('cancer_case', 'healthy_control', 'disease_control', 'excluded')),
               ADD COLUMN IF NOT EXISTS label_mapping_version text"""
        )
        cursor.executemany(
            """UPDATE clinical.diagnoses AS diagnosis SET
               source_cancer_file_raw = %s,
               study_cancer_type = %s,
               diagnosed_cancer_type = CASE
                   WHEN diagnosis.label_mapping_version IS NULL
                   THEN diagnosis.cancer_type
                   ELSE diagnosis.diagnosed_cancer_type
               END,
               case_status = %s,
               label_mapping_version = %s,
               cancer_type = %s
               WHERE diagnosis.diagnosis_id = %s""",
            [
                (
                    row.source_cancer_file_raw,
                    row.study_cancer_type,
                    row.case_status.value,
                    plan.mapping_version,
                    row.study_cancer_type,
                    row.diagnosis_id,
                )
                for row in plan.rows
            ],
        )
    connection.commit()


def rollback(connection: PgConnection, mapping_version: str) -> int:
    with connection.cursor() as cursor:
        cursor.execute(
            """UPDATE clinical.diagnoses
               SET cancer_type = diagnosed_cancer_type,
                   source_cancer_file_raw = NULL,
                   study_cancer_type = NULL,
                   diagnosed_cancer_type = NULL,
                   case_status = NULL,
                   label_mapping_version = NULL
               WHERE label_mapping_version = %s""",
            (mapping_version,),
        )
        changed = cursor.rowcount
    connection.commit()
    return changed


def emit(plan: MigrationPlan, action: str) -> None:
    counts: dict[str, int] = {}
    for row in plan.rows:
        counts[row.case_status.value] = counts.get(row.case_status.value, 0) + 1
    typer.echo(
        json.dumps(
            {
                "action": action,
                "mapping_version": plan.mapping_version,
                "input_rows": plan.input_rows,
                "database_rows": plan.database_rows,
                "planned_updates": len(plan.rows),
                "case_status_counts": counts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main(
    mode: Annotated[Mode, typer.Option()] = Mode.CHECK_DB,
    input_file: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = DEFAULT_INPUT,
    mapping_file: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = DEFAULT_MAPPING,
    confirm_version: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Preview, apply, or roll back the versioned AECD label contract."""
    rules = load_rules(mapping_file)
    version = rules[0].mapping_version
    match mode:
        case Mode.CHECK_DB:
            sources = load_sources(input_file)
            with closing(connect(read_only=True)) as connection:
                plan = build_plan(connection, sources, rules)
            emit(plan, mode.value)
        case Mode.WRITE:
            if confirm_version != version:
                raise typer.BadParameter(f"Set --confirm-version {version}")
            sources = load_sources(input_file)
            with closing(connect(read_only=False)) as connection:
                plan = build_plan(connection, sources, rules)
                apply_plan(connection, plan)
            emit(plan, mode.value)
        case Mode.ROLLBACK:
            if confirm_version != version:
                raise typer.BadParameter(f"Set --confirm-version {version}")
            with closing(connect(read_only=False)) as connection:
                changed = rollback(connection, version)
            typer.echo(json.dumps({"action": "rollback", "rows": changed}))
        case unreachable:
            assert_never(unreachable)


if __name__ == "__main__":
    typer.run(main)
