#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#   "markdown>=3.7",
#   "matplotlib>=3.7",
#   "numpy>=1.24",
#   "openpyxl>=3.1",
#   "scikit-learn==1.7.2",
#   "scipy>=1.10",
# ]
# ///
# How to run:
#   uv run publications/전향검체/보라매병원/src/generate_powder_comparison.py
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
SOURCE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(SOURCE_DIR))

from powder_comparison.figures import write_figures
from powder_comparison.inference_tables import write_inference_tables
from powder_comparison.report import write_report
from powder_comparison.runner import default_sources, run_analysis
from powder_comparison.tables import write_tables
from three_cohort_spectra import write_three_cohort_figure


def main() -> None:
    result = run_analysis(default_sources(REPO))
    table_dir = write_tables(result)
    write_inference_tables(result)
    figure_dir = write_figures(result)
    write_three_cohort_figure(REPO)
    markdown_path, html_path = write_report(result)
    print(f"Decision: {result.decision}")
    print(f"Tables: {table_dir}")
    print(f"Figures: {figure_dir}")
    print(f"Markdown: {markdown_path}")
    print(f"HTML: {html_path}")


if __name__ == "__main__":
    main()
