"""sers preprocess — Spectrum preprocessing and QC pipeline."""

import click

from ._run import run_script


@click.command()
@click.option("--source", "-s", type=click.Choice(["thermo", "medical"]),
              default="thermo", show_default=True,
              help="Instrument data source.")
@click.option("--config", "-c", default="config/config.yaml", show_default=True,
              help="Path to config.yaml.")
@click.option("--normalization", "-n",
              type=click.Choice(["snv", "minmax", "l2", "area", "none"]),
              default=None, help="Override normalization method.")
@click.option("--no-trim", is_flag=True, default=False,
              help="Skip trimming to fingerprint region.")
@click.option("--output-dir", "-o", default=None,
              help="Override output directory.")
@click.option("--skip-qc", is_flag=True, default=False,
              help="Skip QC and use all preprocessed spectra.")
@click.option("--qc-policy", type=click.Choice(["v2", "strict", "none"]),
              default="v2", show_default=True, help="QC policy.")
@click.option("--raw-only", is_flag=True, default=False,
              help="Only run raw preprocessing (main.py), skip QC pipeline.")
def preprocess(source, config, normalization, no_trim, output_dir,
               skip_qc, qc_policy, raw_only):
    """Preprocess spectra: load → parse → QC → normalize → save.

    \b
    Examples:
        sers preprocess
        sers preprocess --source medical
        sers preprocess --normalization snv --qc-policy strict
        sers preprocess --raw-only --source medical
    """
    if raw_only:
        # Run main.py only (raw preprocessing)
        args = ["--data-source", source]
        run_script("main.py", args)
    else:
        # Run full QC + preprocessing pipeline
        args = ["--config", config, "--qc-policy", qc_policy]
        if normalization:
            args += ["--normalization", normalization]
        if no_trim:
            args.append("--no-trim")
        if output_dir:
            args += ["--output-dir", output_dir]
        if skip_qc:
            args.append("--skip-qc")
        run_script("scripts/pipeline/run_qc_preprocess.py", args)
