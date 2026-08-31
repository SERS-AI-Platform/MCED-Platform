from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import click

from sers.master_data.legacy_crosswalk import (
    ConflictingCrosswalkLabelError,
    CrosswalkResolutionError,
    InvalidCrosswalkError,
    ingest_legacy_crosswalk,
)
from sers.master_data.spectrum_persistence import SourceAssetChangedError

from .data_db import open_master_database

_PATH = click.Path(path_type=Path)
_EXISTING_FILE = click.Path(exists=True, dir_okay=False, path_type=Path)


@click.command("ingest-legacy-crosswalk")
@click.option("--db", type=_PATH, required=True)
@click.option("--source", type=_EXISTING_FILE, required=True)
def ingest_legacy_crosswalk_command(db: Path, source: Path) -> None:
    try:
        with closing(open_master_database(db)) as connection, connection:
            counts = ingest_legacy_crosswalk(connection, source)
    except (
        ConflictingCrosswalkLabelError,
        CrosswalkResolutionError,
        InvalidCrosswalkError,
        SourceAssetChangedError,
        OSError,
        sqlite3.Error,
    ) as error:
        raise click.ClickException(
            f"legacy crosswalk ingestion failed ({type(error).__name__})"
        ) from None
    click.echo(
        f"rows={counts.rows},mapped={counts.mapped},created={counts.created},"
        f"existing={counts.existing}"
    )
