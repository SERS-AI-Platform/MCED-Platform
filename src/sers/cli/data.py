"""sers data — Data management commands."""

import click

from ._run import run_script


@click.group(invoke_without_command=True)
@click.pass_context
def data(ctx):
    """Data management: standardize, validate, upload.

    \b
    Subcommands:
        sers data standardize   Standardize clinical data
        sers data validate      Validate spectrum files
        sers data upload        Upload to database
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@data.command()
@click.option("--output-dir", "-o", default=None,
              help="Output directory [data/clinical_data/standardized].")
@click.option("--dry-run", is_flag=True, default=False,
              help="List files without writing.")
def standardize(output_dir, dry_run):
    """Standardize clinical data into unified format.

    \b
    Examples:
        sers data standardize
        sers data standardize --dry-run
        sers data standardize -o ./output
    """
    args = []
    if output_dir:
        args += ["--output-dir", output_dir]
    if dry_run:
        args.append("--dry-run")
    run_script("scripts/pipeline/standardize_clinical_data.py", args)


@data.command()
@click.option("--input", "-i", "input_dir", default=None,
              help="Input data directory.")
def validate(input_dir):
    """Validate spectrum data files.

    \b
    Examples:
        sers data validate
        sers data validate -i ./data/raw_data
    """
    from ..config import load_config, DATA_DIR
    from ..io import find_spectra, parse_filename, read_spectrum

    data_dir = input_dir or str(DATA_DIR)
    files = find_spectra(data_dir)
    click.echo(f"Found {len(files)} spectrum files")

    errors = []
    for path in files:
        try:
            parse_filename(path)
            read_spectrum(path)
        except Exception as e:
            errors.append((path.name, str(e)))

    if errors:
        click.echo(f"\n{len(errors)} files with issues:")
        for name, err in errors[:10]:
            click.echo(f"  {name}: {err}")
        if len(errors) > 10:
            click.echo(f"  ... and {len(errors) - 10} more")
        raise SystemExit(1)
    else:
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
