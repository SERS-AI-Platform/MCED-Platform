from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Final

import click

from sers.config import load_config
from sers.master_data.clinical_ingestion import (
    ClinicalIngestionRequest,
    EmptyClinicalSourceError,
    ingest_clinical_source,
)
from sers.master_data.clinical_inventory import inventory_clinical_sources
from sers.master_data.clinical_parser import (
    ConflictingClinicalIdentityAliasError,
    MalformedClinicalSourceError,
    MissingClinicalHeaderError,
    MissingClinicalSheetError,
    MissingSubjectKeyError,
)
from sers.master_data.clinical_types import (
    ClinicalEventType,
    ClinicalFormat,
    ClinicalIdentityStatus,
    ClinicalSource,
    ClinicalSourceVersion,
)
from sers.master_data.inventory import (
    configured_inventory_roots,
    inventory_spectra,
)
from sers.master_data.spectrum_persistence import SourceAssetChangedError
from sers.master_data.spectrum_types import InventoryRoot, SourceKind

from ._run import PROJECT_ROOT
from .data_clinical_registry import (
    ingest_clinical_registry_command as ingest_clinical_registry_command,
)
from .data_db import open_master_database
from .data_spectrum_ingest import ingest_spectra_command as ingest_spectra_command

_PATH = click.Path(path_type=Path)
_EXISTING_FILE = click.Path(exists=True, dir_okay=False, path_type=Path)
_EXISTING_DIR = click.Path(exists=True, file_okay=False, path_type=Path)
_SOURCE_KIND_VALUES = (
    "thermo",
    "medical",
    "remeasurement",
    "boramae_liquid",
    "boramae_powder",
    "powder_reproducibility",
)
_CLINICAL_FORMATS: Final = {
    ".csv": ClinicalFormat("csv"),
    ".xlsx": ClinicalFormat("excel"),
    ".xlsm": ClinicalFormat("excel"),
}


def _clinical_format(path: Path) -> ClinicalFormat:
    try:
        return _CLINICAL_FORMATS[path.suffix.casefold()]
    except KeyError as error:
        raise click.BadParameter("source must be CSV, XLSX, or XLSM") from error


def _safe_error(action: str, error: BaseException) -> click.ClickException:
    return click.ClickException(f"{action} failed ({type(error).__name__})")


@click.command("inventory")
@click.option("--db", type=_PATH, default=Path("data/sers_master.db"), show_default=True)
@click.option("--repo-root", type=_EXISTING_DIR, default=PROJECT_ROOT, show_default=True)
@click.option("--config", "config_path", type=_EXISTING_FILE, default=None)
@click.option("--clinical-root", type=_EXISTING_DIR, default=None)
@click.option("--spectrum-root", type=_EXISTING_DIR, multiple=True)
@click.option(
    "--source-kind",
    type=click.Choice(_SOURCE_KIND_VALUES),
    default="remeasurement",
    show_default=True,
)
def inventory_command(
    db: Path,
    repo_root: Path,
    config_path: Path | None,
    clinical_root: Path | None,
    spectrum_root: tuple[Path, ...],
    source_kind: str,
) -> None:
    clinical = inventory_clinical_sources(
        clinical_root or repo_root / "data" / "clinical_data"
    )
    roots = (
        tuple(
            InventoryRoot(
                path=root,
                source_kind=SourceKind(source_kind),
                instrument_key="Thermo",
                preparation="liquid",
            )
            for root in spectrum_root
        )
        if spectrum_root
        else configured_inventory_roots(repo_root, load_config(config_path))
    )
    spectra = inventory_spectra(roots)
    with closing(open_master_database(db)) as connection, connection:
        click.echo(
            f"clinical[{clinical.redacted_summary()}] "
            f"spectra[{spectra.redacted_summary()}]"
        )


@click.command("ingest-clinical")
@click.option("--db", type=_PATH, required=True)
@click.option("--source", type=_EXISTING_FILE, required=True)
@click.option("--site-code", required=True)
@click.option("--site-name", default=None)
@click.option("--protocol-code", required=True)
@click.option("--source-group", required=True)
@click.option(
    "--patient-id-field",
    "--subject-key-field",
    "patient_id_field",
    multiple=True,
    required=True,
)
@click.option(
    "--event-type",
    type=click.Choice(["collection", "diagnosis", "procedure", "lab", "other"]),
    default="other",
    show_default=True,
)
@click.option("--sheet-name", default=None)
@click.option("--header-row", type=click.IntRange(min=1), default=1, show_default=True)
@click.option("--canonical-alias", default=None)
@click.option("--historical", is_flag=True)
@click.option("--raw-store", type=_PATH, required=True)
def ingest_clinical_command(
    db: Path,
    source: Path,
    site_code: str,
    site_name: str | None,
    protocol_code: str,
    source_group: str,
    patient_id_field: tuple[str, ...],
    event_type: ClinicalEventType,
    sheet_name: str | None,
    header_row: int,
    canonical_alias: str | None,
    historical: bool,
    raw_store: Path,
) -> None:
    contract = ClinicalSource(
        path=source,
        site_code=site_code,
        site_name=site_name or site_code,
        protocol_code=protocol_code,
        source_group=source_group,
        source_format=_clinical_format(source),
        patient_id_fields=patient_id_field,
        event_type=event_type,
        sheet_name=sheet_name,
        header_row=header_row,
        canonical_alias=canonical_alias,
        identity_status=(
            ClinicalIdentityStatus("alias_review")
            if canonical_alias
            else ClinicalIdentityStatus("canonical")
        ),
        source_version=(
            ClinicalSourceVersion("historical")
            if historical
            else ClinicalSourceVersion("current")
        ),
    )
    try:
        with closing(open_master_database(db)) as connection, connection:
            counts = ingest_clinical_source(
                connection,
                ClinicalIngestionRequest(contract, raw_store),
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
        raise _safe_error("clinical ingestion", error) from None
    click.echo(
        f"source_assets={counts.source_assets},subjects={counts.subjects},"
        f"events={counts.events},observations={counts.observations}"
    )
