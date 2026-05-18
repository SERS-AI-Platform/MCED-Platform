"""sers qc — QC validation phases."""

import click

from ._run import run_script

# Map phase IDs to script paths
_PHASES = {
    "01": ("01_literature_standards.py", "Literature standard validation"),
    "02": ("02_statistical_optimization.py", "Statistical optimization"),
    "03": ("03_clinical_impact.py", "Clinical impact analysis"),
    "04": ("04_generate_report.py", "Report generation"),
    "05": ("05_cross_instrument_variance.py", "Cross-instrument variance"),
    "06": ("06_preprocessing_validation.py", "Preprocessing validation"),
    "06b": ("06b_calibration_diagnosis.py", "Calibration diagnostics"),
    "06c": ("06c_batch_leakage_check.py", "Batch leakage detection"),
    "06d": ("06d_two_stage_qc_test.py", "Two-stage QC testing"),
    "07": ("07_threshold_derivation.py", "Threshold optimization"),
    "08": ("08_hospital_confound.py", "Hospital confounding analysis"),
    "09": ("09_hospital_stratified.py", "Hospital stratification"),
    "10": ("10_within_hospital_cancer_type.py", "Within-hospital cancer type"),
}


@click.group(invoke_without_command=True)
@click.pass_context
def qc(ctx):
    """QC validation pipeline.

    \b
    Subcommands:
        sers qc phase <ID>    Run a specific QC phase
        sers qc list          List available phases
        sers qc all           Run all phases sequentially
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@qc.command("list")
def qc_list():
    """List all available QC validation phases."""
    click.echo("Available QC phases:\n")
    for phase_id, (script, desc) in _PHASES.items():
        click.echo(f"  {phase_id:>4s}  {desc}")
    click.echo(f"\nRun with: sers qc phase <ID>")


@qc.command("phase")
@click.argument("phase_id")
def qc_phase(phase_id):
    """Run a specific QC validation phase.

    \b
    Examples:
        sers qc phase 01
        sers qc phase 06b
        sers qc phase 10
    """
    if phase_id not in _PHASES:
        available = ", ".join(_PHASES.keys())
        raise click.BadParameter(
            f"Unknown phase '{phase_id}'. Available: {available}")
    script, desc = _PHASES[phase_id]
    click.echo(f"Running QC phase {phase_id}: {desc}")
    run_script(f"scripts/qc_validation/{script}")


@qc.command("all")
@click.option("--stop-on-error", is_flag=True, default=False,
              help="Stop at first failure.")
def qc_all(stop_on_error):
    """Run all QC phases sequentially (01 → 10).

    \b
    Examples:
        sers qc all
        sers qc all --stop-on-error
    """
    import subprocess
    import sys
    from pathlib import Path
    from ._run import PROJECT_ROOT

    for phase_id, (script, desc) in _PHASES.items():
        click.echo(f"\n{'='*60}")
        click.echo(f"Phase {phase_id}: {desc}")
        click.echo(f"{'='*60}")
        full_path = PROJECT_ROOT / "scripts" / "qc_validation" / script
        if not full_path.exists():
            click.echo(f"  ⚠ Script not found, skipping: {script}")
            continue
        result = subprocess.run(
            [sys.executable, str(full_path)],
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode != 0:
            click.echo(f"  ✗ Phase {phase_id} failed (exit {result.returncode})")
            if stop_on_error:
                sys.exit(result.returncode)
        else:
            click.echo(f"  ✓ Phase {phase_id} complete")

    click.echo(f"\n{'='*60}")
    click.echo("All QC phases finished.")
