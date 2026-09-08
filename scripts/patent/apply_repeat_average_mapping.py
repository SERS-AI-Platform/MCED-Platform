# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "matplotlib", "openpyxl"]
# ///
# ─── How to run ───
# uv run scripts/patent/apply_repeat_average_mapping.py

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from mapping_repeat_average_core import (
    GRID_PATH,
    MAPPING_ROOT,
    analyze_subjects,
    load_subjects,
    partial_average_rows,
)
from mapping_repeat_average_outputs import write_tables
from mapping_repeat_average_plots import write_plots
from mapping_repeat_average_report import (
    build_metadata,
    reference_audit,
    write_metadata,
    write_report,
)

OUT = Path("/home/user/SERS-AI/results/mapping_repeat_average_patent_20260825_v1")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    grid = np.asarray(np.load(GRID_PATH), dtype=float)
    subjects, raw_axis = load_subjects(grid)
    results = analyze_subjects(subjects, grid)
    partial_rows = partial_average_rows(results)
    tables = write_tables(OUT, grid, results, partial_rows)
    references = reference_audit(MAPPING_ROOT / "Thermo Reference")
    metadata = build_metadata(results, grid, raw_axis, references)
    write_metadata(OUT, metadata)
    write_report(OUT, results, grid, metadata, tables)
    write_plots(OUT, grid, results, tables)
    print(
        json.dumps(
            {
                "output_dir": str(OUT),
                "included_subjects": len(results),
                "input_repeats": metadata["input_repeats"],
                "qc_passed_repeats": metadata["qc_passed_repeats"],
                "qc_passed_per_subject": metadata["qc_passed_per_subject"],
                "instrument_average_audit_pass": sum(
                    r.average_normalized_rmse <= 1e-4 for r in results
                ),
                "instrument_average_audit_total": len(results),
                "median_existing_noise_reduction_pct": next(
                    row["median"]
                    for row in tables.summary_rows
                    if row["group"] == "All included"
                    and row["metric"] == "existing_noise_reduction_vs_single_pct"
                ),
                "median_patent_noise_reduction_pct": next(
                    row["median"]
                    for row in tables.summary_rows
                    if row["group"] == "All included"
                    and row["metric"] == "patent_noise_reduction_vs_single_pct"
                ),
                "median_patent_extra_reduction_vs_existing_pct": next(
                    row["median"]
                    for row in tables.summary_rows
                    if row["group"] == "All included"
                    and row["metric"] == "patent_extra_reduction_vs_existing_pct"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
