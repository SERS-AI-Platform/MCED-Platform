# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "matplotlib", "openpyxl"]
# ///
# ─── How to run ───
# uv run scripts/patent/apply_mapping_preprocessing_comparison.py

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from mapping_preprocessing_compare_core import compare_subjects
from mapping_preprocessing_compare_outputs import write_tables
from mapping_preprocessing_compare_plots import write_plots
from mapping_preprocessing_compare_report import metadata, write_report
from mapping_repeat_average_core import GRID_PATH, GROUP_ORDER, load_subjects

OUT = Path("/home/user/SERS-AI/results/mapping_preprocessing_comparison_20260825_v1")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    grid = np.asarray(np.load(GRID_PATH), dtype=float)
    subjects, raw_axis = load_subjects(grid)
    results = compare_subjects(subjects, grid)
    tables = write_tables(OUT, grid, results, GROUP_ORDER)
    metadata_value = metadata(results, grid, raw_axis, GROUP_ORDER)
    write_report(OUT, results, tables, metadata_value, GROUP_ORDER)
    write_plots(OUT, grid, results, tables, GROUP_ORDER)
    print(
        json.dumps(
            {
                "output_dir": str(OUT),
                "included_subjects": len(results),
                "raw_repeats": metadata_value["raw_repeats"],
                "qc_passed_repeats": metadata_value["qc_passed_repeats"],
                "changed_previous_shape_correlation_median": next(
                    row["median"]
                    for row in tables.summary_rows
                    if row["group"] == "All included"
                    and row["metric"] == "changed_previous_shape_correlation"
                ),
                "changed_reproducible_peak_median": next(
                    row["median"]
                    for row in tables.summary_rows
                    if row["group"] == "All included"
                    and row["metric"] == "changed_reproducible_peak_count"
                ),
                "previous_reproducible_peak_median": next(
                    row["median"]
                    for row in tables.summary_rows
                    if row["group"] == "All included"
                    and row["metric"] == "previous_reproducible_peak_count"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
