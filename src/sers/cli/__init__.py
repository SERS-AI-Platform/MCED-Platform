"""Command-line entry point for SERS workflows."""

import click

from .. import __version__
from .analyze import analyze
from .data import data
from .predict import predict
from .preprocess import preprocess
from .production import production
from .qc import qc
from .serve import serve
from .test import test as test_command
from .train import benchmark, train


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="sers")
def cli():
    """Run MCED-Platform SERS data, model, and deployment workflows."""


cli.add_command(preprocess)
cli.add_command(train)
cli.add_command(benchmark)
cli.add_command(test_command, "test")
cli.add_command(predict)
cli.add_command(serve)
cli.add_command(production)
cli.add_command(qc)
cli.add_command(data)
cli.add_command(analyze)


def main() -> None:
    """Console-script compatible wrapper."""
    cli()
