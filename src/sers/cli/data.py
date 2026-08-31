"""sers data — Data management commands."""

from collections import Counter
from pathlib import Path

import click

from ._run import run_script
from .data_ingest import (
    ingest_clinical_command,
    ingest_clinical_registry_command,
    ingest_spectra_command,
    inventory_command,
)
from .data_legacy_crosswalk import ingest_legacy_crosswalk_command
from .data_lineage import (
    build_labels_command,
    export_dataset_command,
    match_command,
    reconcile_command,
)


@click.group(invoke_without_command=True)
@click.pass_context
def data(ctx):
    """Manage governed clinical, spectrum, and model-lineage data.

    \b
    Subcommands:
        sers data inventory         Inventory configured raw sources
        sers data ingest-clinical   Ingest one immutable clinical source
        sers data ingest-clinical-registry
                                    Validate or ingest the built-in registry
        sers data ingest-spectra    Ingest immutable spectrum sources
        sers data match             Match clinical and spectrum identity
        sers data build-labels      Derive versioned evidence-linked labels
        sers data reconcile         Report aggregate matching states
        sers data export-dataset    Export one frozen dataset manifest
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@data.command()
@click.option(
    "--output-dir",
    "-o",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
    help="Output directory [data/clinical_data/standardized].",
)
@click.option("--dry-run", is_flag=True, default=False,
              help="List files without writing.")
def standardize(output_dir: Path | None, dry_run: bool) -> None:
    """Run the legacy ungoverned clinical standardization predecessor.

    \b
    Examples:
        sers data standardize
        sers data standardize --dry-run
        sers data standardize -o ./output
    """
    click.echo(
        "DEPRECATED: this legacy output is not eligible for governed lineage; "
        "use 'sers data ingest-clinical-registry'.",
        err=True,
    )
    args: list[str] = []
    if output_dir:
        args += ["--output-dir", str(output_dir)]
    if dry_run:
        args.append("--dry-run")
    run_script("scripts/pipeline/standardize_clinical_data.py", args)


@data.command()
@click.option("--input", "-i", "input_dir", default=None,
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="Input data directory.")
def validate(input_dir: Path | None) -> None:
    """Validate spectrum data files.

    \b
    Examples:
        sers data validate
        sers data validate -i ./data/raw_data
    """
    from ..config import DATA_ROOT
    from ..io import find_spectra, parse_filename, read_spectrum

    data_dir = input_dir or DATA_ROOT
    files = find_spectra(data_dir)
    click.echo(f"Found {len(files)} spectrum files")

    errors: Counter[str] = Counter()
    for path in files:
        try:
            parse_filename(path)
            read_spectrum(path)
        except (OSError, ValueError) as error:
            errors[type(error).__name__] += 1

    if errors:
        click.echo(f"issues={sum(errors.values())}")
        for reason, count in sorted(errors.items()):
            click.echo(f"{reason}={count}")
        raise click.ClickException("spectrum validation failed")
    click.echo("All files valid!")


@data.command()
@click.option("--target", "-t",
              type=click.Choice(["postgres", "supabase"]),
              default="supabase", show_default=True,
              help="Upload target.")
@click.option("--host", default=None, help="DB host (postgres only).")
@click.option("--port", type=int, default=None, help="DB port (postgres only).")
@click.option("--dbname", default=None, help="DB name (postgres only).")
@click.option("--user", default=None, help="DB user (postgres only).")
@click.option("--password", default=None, help="DB password (postgres only).")
@click.option("--drop", is_flag=True, default=False,
              help="Drop and recreate tables (postgres only).")
def upload(target, host, port, dbname, user, password, drop):
    """Upload SERS data to database.

    \b
    Examples:
        sers data upload                          # supabase (default)
        sers data upload --target postgres --drop
        sers data upload -t postgres --host localhost --port 5432
    """
    if target == "supabase":
        run_script("scripts/db/upload_to_supabase.py")
    else:
        args = []
        pg_map = {
            "--host": host,
            "--port": port,
            "--dbname": dbname,
            "--user": user,
            "--password": password,
        }
        for flag, val in pg_map.items():
            if val is not None:
                args += [flag, str(val)]
        if drop:
            args.append("--drop")
        run_script("scripts/db/upload_to_postgres.py", args)


data.add_command(inventory_command)
data.add_command(ingest_clinical_command)
data.add_command(ingest_clinical_registry_command)
data.add_command(ingest_legacy_crosswalk_command)
data.add_command(ingest_spectra_command)
data.add_command(match_command)
data.add_command(build_labels_command)
data.add_command(reconcile_command)
data.add_command(export_dataset_command)
