"""Unified CLI for SERS analysis pipeline.

Usage:
    sers preprocess          Preprocess spectra (QC + normalization)
    sers train <model>       Train a model (resnet18, lr, xgboost, stacking, ...)
    sers benchmark           Benchmark multiple models
    sers test                Evaluate a trained model
    sers predict             Run inference on spectrum files
    sers serve               Launch web application
    sers qc                  Run QC validation phases
    sers data                Data management (standardize, validate, upload)
    sers production          Build production model artifacts
"""

import click

from .. import __version__


HELP_TEXT = """
SERS multi-cancer screening pipeline CLI.

\b
Workflow:
  preprocess → train → test → predict

\b
Commands:
  preprocess   Preprocess spectra (load → QC → normalize → save)
  train        Train a model (resnet18, lr, xgboost, stacking, ...)
  benchmark    Benchmark multiple models side-by-side
  test         Evaluate trained model (ROC, SHAP, t-SNE, ...)
  predict      Run inference on spectrum CSV files
  serve        Launch web application (basic / clinical)
  qc           Run QC validation phases (01–10)
  data         Data management (standardize, validate, upload)
  production   Build production model artifacts
  analyze      Analysis & visualization (t-SNE, explore, ...)

\b
Quick start:
  sers preprocess                          # default QC + SNV
  sers train resnet18 --epochs 200         # train ResNet18
  sers train stacking --dry-run            # stacking ensemble
  sers test -i results/training            # evaluate
  sers predict sample.csv --age 55         # inference
  sers serve                               # webapp on :8000

\b
Docs: docs/CLI.md
"""


@click.group(
    help=HELP_TEXT,
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.version_option(__version__, prog_name="sers")
def cli():
    pass


from .preprocess import preprocess
from .train import train, benchmark
from .test import test
from .predict import predict
from .serve import serve
from .qc import qc
from .data import data
from .production import production
from .analyze import analyze
from .compare import compare

cli.add_command(preprocess)
cli.add_command(train)
cli.add_command(benchmark)
cli.add_command(test)
cli.add_command(predict)
cli.add_command(serve)
cli.add_command(qc)
cli.add_command(data)
cli.add_command(production)
cli.add_command(analyze)
cli.add_command(compare)
