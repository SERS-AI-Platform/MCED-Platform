"""sers serve — Launch web application."""

import click

from ._run import run_script


@click.command()
@click.option("--clinical", is_flag=True, default=False,
              help="Launch IEC 62366 clinical interface instead of basic.")
@click.option("--host", default=None, help="Host address [0.0.0.0].")
@click.option("--port", "-p", type=int, default=None,
              help="Port [8000 basic / 8080 clinical].")
@click.option("--reload", is_flag=True, default=False,
              help="Auto-reload on code changes (clinical only).")
def serve(clinical, host, port, reload):
    """Launch SERS web application.

    \b
    Examples:
        sers serve                          # basic webapp on :8000
        sers serve --clinical               # clinical webapp on :8080
        sers serve --clinical -p 9000 --reload
        sers serve --host 127.0.0.1 -p 5000
    """
    if clinical:
        args = []
        if host:
            args += ["--host", host]
        if port is not None:
            args += ["--port", str(port)]
        if reload:
            args.append("--reload")
        run_script("scripts/deployment/sers_clinical_webapp.py", args)
    else:
        args = []
        if host:
            args += ["--host", host]
        if port is not None:
            args += ["--port", str(port)]
        run_script("scripts/deployment/sers_webapp.py", args)
