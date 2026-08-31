from __future__ import annotations

import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path

import click

from sers.master_data.clinical_ingestion import (
    ClinicalIngestionRequest,
    EmptyClinicalSourceError,
    ingest_clinical_registry,
)
from sers.master_data.clinical_inventory import inventory_clinical_sources
from sers.master_data.clinical_parser import (
    ConflictingClinicalIdentityAliasError,
    MalformedClinicalSourceError,
    MissingClinicalHeaderError,
    MissingClinicalSheetError,
    MissingSubjectKeyError,
    load_clinical_rows,
)
from sers.master_data.spectrum_persistence import SourceAssetChangedError

from .data_db import open_master_database

_PATH = click.Path(path_type=Path)
_EXISTING_DIR = click.Path(exists=True, file_okay=False, path_type=Path)


def _safe_error(action: str, error: BaseException) -> click.ClickException:
    return click.ClickException(f"{action} failed ({type(error).__name__})")


@click.command("ingest-clinical-registry")
@click.option("--clinical-root", type=_EXISTING_DIR, required=True)
@click.option("--db", type=_PATH, default=None)
@click.option("--raw-store", type=_PATH, default=None)
@click.option("--validate-only", is_flag=True)
def ingest_clinical_registry_command(
    clinical_root: Path,
    db: Path | None,
    raw_store: Path | None,
    validate_only: bool,
) -> None:
    """Validate or ingest the source-file-specific built-in clinical registry."""
    inventory = inventory_clinical_sources(clinical_root)
    issue_types: Counter[str] = Counter(
        issue.reason_code for issue in inventory.issues
    )
    row_count = 0
    for source in inventory.sources:
        try:
            rows = load_clinical_rows(source)
            if not rows:
                raise EmptyClinicalSourceError(source.path)
            row_count += len(rows)
        except (
            ConflictingClinicalIdentityAliasError,
            EmptyClinicalSourceError,
            MalformedClinicalSourceError,
            MissingClinicalHeaderError,
            MissingClinicalSheetError,
            MissingSubjectKeyError,
            OSError,
        ) as error:
            issue_types[type(error).__name__] += 1
    sheet_count = sum(
        len(source.sheet_contracts) or 1 for source in inventory.sources
    )
    click.echo(
        f"sources={len(inventory.sources)},sheets={sheet_count},"
        f"rows={row_count},issues={sum(issue_types.values())}"
    )
    for reason, count in sorted(issue_types.items()):
        click.echo(f"{reason}={count}")
    if issue_types:
        raise click.ClickException("clinical registry validation failed")
    if not inventory.sources:
        raise click.ClickException("clinical registry contains no sources")
    if validate_only:
        return
    if db is None or raw_store is None:
        raise click.UsageError("--db and --raw-store are required for ingestion")
    try:
        with closing(open_master_database(db)) as connection, connection:
            counts = ingest_clinical_registry(
                connection,
                tuple(
                    ClinicalIngestionRequest(source, raw_store)
                    for source in inventory.sources
                ),
            )
    except (
        ConflictingClinicalIdentityAliasError,
        EmptyClinicalSourceError,
        MalformedClinicalSourceError,
        MissingClinicalHeaderError,
        MissingClinicalSheetError,
        MissingSubjectKeyError,
        SourceAssetChangedError,
        OSError,
        sqlite3.Error,
    ) as error:
        raise _safe_error("clinical registry ingestion", error) from None
    click.echo(
        f"source_assets={counts.source_assets},subjects={counts.subjects},"
        f"events={counts.events},observations={counts.observations}"
    )
