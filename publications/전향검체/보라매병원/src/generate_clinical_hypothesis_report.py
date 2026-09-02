#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "numpy>=2.0",
#     "polars>=1.30",
#     "scipy>=1.15",
#     "typer>=0.16",
# ]
# ///

"""Generate the prespecified three-group clinical hypothesis analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
import typer
from clinical_hypothesis_models import AncovaData, analyze_ancova
from clinical_hypothesis_narrative import write_report
from clinical_hypothesis_outputs import (
    AssumptionContext,
    assumption_frame,
    contrast_frame,
    primary_frame,
    secondary_frame,
    sensitivity_frame,
)
from clinical_hypothesis_statistics import GROUPS, MetricAnalysis, MetricData, analyze_metric


@dataclass(frozen=True, slots=True)
class MetricSpec:
    name: str
    column: str
    scale: str
    log10_transform: bool


METRICS: Final = (
    MetricSpec("Age", "age", "original", False),
    MetricSpec("BMI", "bmi", "original", False),
    MetricSpec("PSA", "psa_ng_ml", "log10", True),
)


def load_data(path: Path) -> pl.DataFrame:
    """Parse the reviewed subject-level covariate table."""
    return pl.read_csv(
        path,
        schema_overrides={
            "subject_id": pl.String,
            "group": pl.String,
            "age": pl.Float64,
            "bmi": pl.Float64,
            "psa_ng_ml": pl.Float64,
        },
        null_values=[""],
    )


def metric_data(data: pl.DataFrame, spec: MetricSpec) -> MetricData:
    """Build fixed-order complete-case arrays for one outcome."""
    samples: list[np.ndarray[tuple[int], np.dtype[np.float64]]] = []
    for group in GROUPS:
        values = (
            data.filter(pl.col("group") == group)[spec.column]
            .drop_nulls()
            .to_numpy()
            .astype(np.float64)
        )
        samples.append(np.log10(values) if spec.log10_transform else values)
    return MetricData(spec.name, spec.scale, tuple(samples))


def ancova_data(data: pl.DataFrame) -> AncovaData:
    """Build the complete-case data for the prespecified PSA ANCOVA."""
    complete = data.drop_nulls(["age", "bmi", "psa_ng_ml"])
    group_codes = (
        complete["group"]
        .replace_strict({"Control": 0.0, "Biopsy-negative": 1.0, "Cancer": 2.0})
        .to_numpy()
        .astype(np.float64)
    )
    return AncovaData(
        np.log10(complete["psa_ng_ml"].to_numpy().astype(np.float64)),
        group_codes,
        complete["age"].to_numpy().astype(np.float64),
        complete["bmi"].to_numpy().astype(np.float64),
    )


def overlap(data: pl.DataFrame, column: str, digits: int) -> str:
    """Return the covariate interval shared by all groups."""
    ranges = (
        data.group_by("group")
        .agg(pl.col(column).drop_nulls().min().alias("minimum"))
        .join(
            data.group_by("group").agg(
                pl.col(column).drop_nulls().max().alias("maximum")
            ),
            on="group",
        )
    )
    lower = float(ranges["minimum"].max())
    upper = float(ranges["maximum"].min())
    return f"{lower:.{digits}f}–{upper:.{digits}f}"


def assumption_context(data: pl.DataFrame) -> AssumptionContext:
    """Summarize transform, missingness, and covariate-overlap conditions."""
    positive_psa = int(data.select((pl.col("psa_ng_ml") > 0).sum()).item())
    missing_bmi = int(data.select(pl.col("bmi").is_null().sum()).item())
    return AssumptionContext(
        total_n=data.height,
        positive_psa_n=positive_psa,
        missing_bmi_n=missing_bmi,
        age_overlap=f"{overlap(data, 'age', 0)} years",
        bmi_overlap=f"{overlap(data, 'bmi', 2)} kg/m²",
    )


def main(
    input_path: Path = typer.Option(..., "--input", exists=True, readable=True),
    output_dir: Path = typer.Option(..., "--output-dir"),
) -> None:
    """Write hypothesis tests, diagnostics, and a technical Markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_data(input_path)
    analyses: tuple[MetricAnalysis, ...] = tuple(
        analyze_metric(metric_data(data, spec)) for spec in METRICS
    )
    age, bmi, psa = analyses
    ancova = analyze_ancova(ancova_data(data))
    context = assumption_context(data)

    primary_frame(ancova).write_csv(
        output_dir / "clinical_hypothesis_primary_ancova.csv"
    )
    contrast_frame(ancova).write_csv(
        output_dir / "clinical_hypothesis_pairwise_contrasts.csv"
    )
    secondary_frame(age, bmi).write_csv(
        output_dir / "clinical_hypothesis_secondary_tests.csv"
    )
    sensitivity_frame(psa).write_csv(
        output_dir / "clinical_hypothesis_sensitivity_tests.csv"
    )
    assumption_frame(ancova, context).write_csv(
        output_dir / "clinical_hypothesis_assumption_checks.csv"
    )
    write_report(
        output_dir / "clinical_hypothesis_report.md",
        ancova,
        (age, bmi),
        psa,
        context,
    )


if __name__ == "__main__":
    typer.run(main)
