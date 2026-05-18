"""Helper to run existing scripts as subprocesses.

This avoids import side-effects and keeps the CLI thin wrappers
around existing scripts that already work standalone.
"""

import subprocess
import sys
from pathlib import Path

# Project root (SERS-AI/)
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def run_script(script_path: str, args: list[str] | None = None) -> None:
    """Run a Python script as a subprocess, forwarding exit code.

    Parameters
    ----------
    script_path : str
        Path relative to PROJECT_ROOT (e.g. "models/train.py").
    args : list[str], optional
        CLI arguments to pass to the script.
    """
    full_path = PROJECT_ROOT / script_path
    if not full_path.exists():
        click.echo(f"Error: script not found: {full_path}", err=True)
        sys.exit(1)

    cmd = [sys.executable, str(full_path)] + (args or [])
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    sys.exit(result.returncode)


import click  # noqa: E402
