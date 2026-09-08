from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import click

from sers.master_data.inventory import inventory_spectra
from sers.master_data.repository import upsert_site
from sers.master_data.spectrum_ingestion import (
    SpectrumIngestionRequest,
    ingest_spectra,
)
from sers.master_data.spectrum_persistence import SourceAssetChangedError
from sers.master_data.spectrum_types import InventoryRoot, SourceKind
from sers.master_data.types import SiteDraft

from .data_db import open_master_database

_PATH = click.Path(path_type=Path)
_EXISTING_DIR = click.Path(exists=True, file_okay=False, path_type=Path)
_SOURCE_KIND_VALUES = (
    "thermo",
    "medical",
    "remeasurement",
    "boramae_liquid",
    "boramae_powder",
    "powder_reproducibility",
)


def _safe_error(action: str, error: BaseException) -> click.ClickException:
    return click.ClickException(f"{action} failed ({type(error).__name__})")


@click.command("ingest-spectra")
@click.option("--db", type=_PATH, required=True)
@click.option("--root", type=_EXISTING_DIR, required=True)
@click.option("--site-code", required=True)
@click.option("--site-name", default=None)
@click.option(
    "--source-kind",
    type=click.Choice(_SOURCE_KIND_VALUES),
    default="remeasurement",
    show_default=True,
)
@click.option("--instrument-key", default="Thermo", show_default=True)
@click.option("--preparation", default="liquid", show_default=True)
@click.option("--raw-store", type=_PATH, required=True)
def ingest_spectra_command(
    db: Path,
    root: Path,
    site_code: str,
    site_name: str | None,
    source_kind: str,
    instrument_key: str,
    preparation: str,
    raw_store: Path,
) -> None:
    report = inventory_spectra(
        (
            InventoryRoot(
                root,
                SourceKind(source_kind),
                instrument_key,
                preparation,
            ),
        )
    )
    try:
        with closing(open_master_database(db)) as connection, connection:
            site = upsert_site(
                connection,
                SiteDraft(code=site_code, name=site_name or site_code),
            )
            connection.commit()
            counts = ingest_spectra(
                connection,
                SpectrumIngestionRequest(report, site.id, raw_store),
            )
    except (SourceAssetChangedError, OSError, sqlite3.Error) as error:
        raise _safe_error("spectrum ingestion", error) from None
    click.echo(
        f"source_assets={counts.source_assets},measurements={counts.measurements},"
        f"derived={counts.derived_artifacts},quarantined={counts.quarantined}"
    )
