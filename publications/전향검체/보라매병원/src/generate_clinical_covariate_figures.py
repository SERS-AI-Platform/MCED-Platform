#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "matplotlib>=3.10",
#     "numpy>=2.0",
#     "polars>=1.30",
#     "typer>=0.16",
# ]
# ///

# ─── How to run ───
# 1. Install uv (if not installed):
#      curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Run directly (no venv, no pip install needed):
#      uv run generate_clinical_covariate_figures.py --input <CSV> \
#        --output-dir <DIR> --statistics-dir <DIR>
# 3. Or make executable and run:
#      chmod +x generate_clinical_covariate_figures.py && ./generate_clinical_covariate_figures.py
# ──────────────────

from __future__ import annotations

from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import polars as pl
import typer
from clinical_covariate_figure_statistics import (
    FigureStatistics,
    annotation_frame,
    load_figure_statistics,
    significance_symbol,
)
from clinical_covariate_plotting import (
    INK,
    MetricSpec,
    SignificanceComparison,
    add_significance_brackets,
    configure_style,
    draw_distribution,
    save_figure,
)

FIGURE_STEMS: Final = (
    "fig24_psa_distribution_by_group",
    "fig25_clinical_covariate_distributions",
)


def load_clinical_data(path: Path) -> pl.DataFrame:
    """Parse the reviewed clinical covariate table."""
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


def write_summary(data: pl.DataFrame, output_dir: Path) -> None:
    """Write group counts and descriptive statistics used by the figures."""
    summary = data.group_by("group", maintain_order=True).agg(
        pl.col("age").count().alias("n_age"),
        pl.col("bmi").count().alias("n_bmi"),
        pl.col("psa_ng_ml").count().alias("n_psa"),
        pl.col("age").mean().alias("age_mean"),
        pl.col("age").std().alias("age_sd"),
        pl.col("age").median().alias("age_median"),
        pl.col("bmi").mean().alias("bmi_mean"),
        pl.col("bmi").std().alias("bmi_sd"),
        pl.col("bmi").median().alias("bmi_median"),
        pl.col("psa_ng_ml").mean().alias("psa_mean"),
        pl.col("psa_ng_ml").std().alias("psa_sd"),
        pl.col("psa_ng_ml").median().alias("psa_median"),
    )
    summary.write_csv(output_dir / "clinical_group_covariate_summary.csv", float_precision=4)


def build_psa_figure(
    data: pl.DataFrame,
    output_dir: Path,
    statistics: FigureStatistics,
) -> None:
    """Build the focused PSA group-distribution figure."""
    figure, axis = plt.subplots(figsize=(9.0, 7.0))
    draw_distribution(
        axis,
        data,
        MetricSpec("psa_ng_ml", "", "PSA (ng/mL)", True),
    )
    add_significance_brackets(
        axis,
        (
            SignificanceComparison(
                1,
                2,
                significance_symbol(statistics.biopsy_control_p),
            ),
            SignificanceComparison(
                2,
                3,
                significance_symbol(statistics.biopsy_cancer_p),
            ),
            SignificanceComparison(
                1,
                3,
                significance_symbol(statistics.cancer_control_p),
            ),
        ),
        float(data["psa_ng_ml"].max()),
        log_scale=True,
    )
    figure.suptitle(
        "PSA distribution by study group",
        x=0.08,
        y=0.99,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color=INK,
    )
    figure.text(
        0.08,
        0.935,
        "Age- and BMI-adjusted HC3 ANCOVA planned contrasts; Holm correction.",
        color="#59616B",
        fontsize=9.5,
    )
    figure.text(
        0.08,
        0.015,
        "* p<0.05, ** p<0.01, *** p<0.001, **** p<0.0001. "
        "Boxes show IQR; points are subjects; y-axis is logarithmic.",
        color="#6B7280",
        fontsize=8,
    )
    figure.tight_layout(rect=(0.04, 0.075, 0.98, 0.91))
    save_figure(figure, output_dir, FIGURE_STEMS[0])


def build_covariate_figure(
    data: pl.DataFrame,
    output_dir: Path,
    statistics: FigureStatistics,
) -> None:
    """Build the age, BMI, and PSA small-multiple figure."""
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 6.8))
    specs = (
        MetricSpec("age", "A  Age", "Age (years)"),
        MetricSpec("bmi", "B  BMI", "BMI (kg/m²)"),
        MetricSpec("psa_ng_ml", "C  PSA", "PSA (ng/mL)", True),
    )
    for axis, spec in zip(axes, specs, strict=True):
        draw_distribution(axis, data, spec)

    add_significance_brackets(
        axes[0],
        (
            SignificanceComparison(
                1,
                2,
                significance_symbol(statistics.age_control_biopsy_p),
            ),
            SignificanceComparison(
                2,
                3,
                significance_symbol(statistics.age_biopsy_cancer_p),
            ),
        ),
        float(data["age"].max()),
        log_scale=False,
    )
    axes[1].text(
        0.5,
        1.015,
        "Welch ANOVA: ns",
        transform=axes[1].transAxes,
        ha="center",
        va="bottom",
        color=INK,
        fontsize=8.5,
    )
    add_significance_brackets(
        axes[2],
        (
            SignificanceComparison(
                1,
                2,
                significance_symbol(statistics.biopsy_control_p),
            ),
            SignificanceComparison(
                2,
                3,
                significance_symbol(statistics.biopsy_cancer_p),
            ),
            SignificanceComparison(
                1,
                3,
                significance_symbol(statistics.cancer_control_p),
            ),
        ),
        float(data["psa_ng_ml"].max()),
        log_scale=True,
    )
    figure.suptitle(
        "Clinical covariate distributions by study group",
        x=0.055,
        y=0.99,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color=INK,
    )
    figure.text(
        0.055,
        0.925,
        "Significant pairwise comparisons only: Age, Games–Howell; "
        "PSA, adjusted HC3 ANCOVA contrasts.",
        color="#59616B",
        fontsize=9.5,
    )
    figure.text(
        0.055,
        0.015,
        "* p<0.05, ** p<0.01, *** p<0.001, **** p<0.0001. "
        "BMI omnibus Holm p=0.219 (ns); BMI is missing for 2 Cancer subjects.",
        color="#6B7280",
        fontsize=8,
    )
    figure.tight_layout(rect=(0.025, 0.09, 0.995, 0.90), w_pad=2.2)
    save_figure(figure, output_dir, FIGURE_STEMS[1])


def main(
    input_path: Path = typer.Option(..., "--input", exists=True, readable=True),
    output_dir: Path = typer.Option(..., "--output-dir"),
    statistics_dir: Path = typer.Option(
        ...,
        "--statistics-dir",
        exists=True,
        file_okay=False,
        readable=True,
    ),
) -> None:
    """Generate Boramae clinical covariate figures and their summary table."""
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    data = load_clinical_data(input_path)
    statistics = load_figure_statistics(statistics_dir)
    write_summary(data, output_dir)
    annotation_frame(statistics).write_csv(
        output_dir / "clinical_covariate_figure_annotations.csv"
    )
    build_psa_figure(data, output_dir, statistics)
    build_covariate_figure(data, output_dir, statistics)


if __name__ == "__main__":
    typer.run(main)
