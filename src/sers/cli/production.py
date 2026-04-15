"""sers production — Build production model artifacts."""

import click

from ._run import run_script


@click.command()
@click.option("--stacking", is_flag=True, default=False,
              help="Build stacking production model instead of default.")
@click.option("--output-dir", "-o", default=None,
              help="Output directory [artifacts/usersnet/current].")
@click.option("--fit-pds", is_flag=True, default=False,
              help="Also fit PDS calibration artifact.")
@click.option("--grid", default=None,
              help="Common grid .npy path (for --fit-pds).")
@click.option("--pds-out", default=None,
              help="PDS output .npz path (for --fit-pds).")
def production(stacking, output_dir, fit_pds, grid, pds_out):
    """Build production model artifacts (trains on ALL data).

    \b
    Examples:
        sers production                               # default model
        sers production --stacking                     # stacking model
        sers production -o artifacts/usersnet/current
        sers production --fit-pds --grid artifacts/usersnet/current/common_grid.npy
    """
    if stacking:
        run_script("models/build_production_stacking.py")
    else:
        args = []
        if output_dir:
            args += ["--output-dir", output_dir]
        run_script("models/build_production_model.py", args)

    if fit_pds:
        pds_args = []
        if grid:
            pds_args += ["--grid", grid]
        if pds_out:
            pds_args += ["--out", pds_out]
        run_script("scripts/deployment/fit_pds_artifact.py", pds_args)
