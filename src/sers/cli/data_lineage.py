from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import click

from sers.master_data.dataset_export import (
    DatasetExportExistsError,
    DatasetManifestNotFoundError,
    export_dataset_manifest,
)
from sers.master_data.labels import (
    GroupCodeLabelRuleError,
    LabelMapping,
    LabelRule,
    derive_sample_labels,
)
from sers.master_data.matching import run_deterministic_matching
from sers.master_data.matching_types import MatchingCounts
from sers.master_data.reconciliation import reconciliation_counts

from .data_db import open_master_database

_PATH = click.Path(path_type=Path)


def _summary(counts: MatchingCounts) -> str:
    return (
        f"matched={counts.matched},unmatched_clinical={counts.unmatched_clinical},"
        f"unmatched_spectrum={counts.unmatched_spectrum},"
        f"ambiguous={counts.ambiguous},duplicate={counts.duplicate}"
    )


def _mappings(values: tuple[str, ...]) -> tuple[LabelMapping, ...]:
    parsed: list[LabelMapping] = []
    for value in values:
        observed, separator, label = value.partition("=")
        if separator == "" or observed == "" or label == "":
            raise click.BadParameter("mapping must use non-empty OBSERVED=LABEL")
        parsed.append(LabelMapping(observed, label))
    return tuple(parsed)


@click.command("match")
@click.option("--db", type=_PATH, required=True)
@click.option("--rule-version", default="exact-v1", show_default=True)
def match_command(db: Path, rule_version: str) -> None:
    try:
        with closing(open_master_database(db)) as connection, connection:
            counts = run_deterministic_matching(connection, rule_version)
    except sqlite3.Error as error:
        raise click.ClickException("matching failed (database error)") from error
    click.echo(_summary(counts))


@click.command("build-labels")
@click.option("--db", type=_PATH, required=True)
@click.option("--task-name", required=True)
@click.option("--definition-version", required=True)
@click.option("--observation-code", required=True)
@click.option("--label-source", required=True)
@click.option("--mapping", "mapping_values", multiple=True, required=True)
def build_labels_command(
    db: Path,
    task_name: str,
    definition_version: str,
    observation_code: str,
    label_source: str,
    mapping_values: tuple[str, ...],
) -> None:
    rule = LabelRule(
        task_name,
        definition_version,
        observation_code,
        label_source,
        _mappings(mapping_values),
    )
    try:
        with closing(open_master_database(db)) as connection, connection:
            count = derive_sample_labels(connection, rule)
    except GroupCodeLabelRuleError as error:
        raise click.ClickException("label rule rejected") from error
    except sqlite3.Error as error:
        raise click.ClickException("label derivation failed (database error)") from error
    click.echo(f"labels={count}")


@click.command("reconcile")
@click.option("--db", type=_PATH, required=True)
@click.option("--rule-version", default="exact-v1", show_default=True)
def reconcile_command(db: Path, rule_version: str) -> None:
    try:
        with closing(open_master_database(db)) as connection, connection:
            counts = reconciliation_counts(connection, rule_version)
    except sqlite3.Error as error:
        raise click.ClickException("reconciliation failed (database error)") from error
    click.echo(_summary(counts))


@click.command("export-dataset")
@click.option("--db", type=_PATH, required=True)
@click.option("--manifest-id", required=True)
@click.option("--output", type=_PATH, required=True)
@click.option("--overwrite", is_flag=True)
@click.option("--mlflow-db", type=_PATH, default=None)
def export_dataset_command(
    db: Path,
    manifest_id: str,
    output: Path,
    overwrite: bool,
    mlflow_db: Path | None,
) -> None:
    if mlflow_db is not None:
        from sers.mlflow_tracking import mlflow_available

        if not mlflow_available():
            raise click.ClickException(
                "MLflow is unavailable; install the mlops optional dependency"
            )
    try:
        with closing(open_master_database(db)) as connection, connection:
            result = export_dataset_manifest(
                connection,
                manifest_id,
                output,
                overwrite=overwrite,
            )
        if mlflow_db is not None:
            from sers.mlflow_tracking import (
                MLflowUnavailableError,
                log_dataset_export,
            )

            try:
                log_dataset_export(mlflow_db, result)
            except MLflowUnavailableError as error:
                raise click.ClickException(str(error)) from None
    except (DatasetExportExistsError, DatasetManifestNotFoundError) as error:
        raise click.ClickException(str(error)) from None
    except (OSError, sqlite3.Error) as error:
        raise click.ClickException("dataset export failed") from error
    click.echo(
        f"rows={result.row_count},excluded={result.excluded_count},"
        f"manifest_sha256={result.content_sha256}"
    )
