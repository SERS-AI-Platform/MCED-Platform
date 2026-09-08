from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from boramae_repeat_average_core import (
    GRID_PATH,
    add_subject_metrics,
    audit_average_files,
    load_subjects,
    partial_average_rows,
)
from boramae_repeat_average_plots import write_plots
from boramae_repeat_average_report import build_metadata, write_metadata, write_report
from boramae_repeat_average_tables import write_tables

OUT = Path("/home/user/SERS-AI/results/boramae_repeat_average_patent_20260825_v1")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    grid = np.asarray(np.load(GRID_PATH), dtype=float)
    subjects, excluded_rows, raw_axis = load_subjects(grid)
    add_subject_metrics(subjects, grid)
    audits = audit_average_files(subjects)
    partial_rows = partial_average_rows(subjects)
    tables = write_tables(OUT, grid, subjects, audits, partial_rows)
    metadata = build_metadata(subjects, excluded_rows, grid, raw_axis, audits)
    write_metadata(OUT, metadata)
    write_report(OUT, subjects, excluded_rows, grid, metadata, audits)
    write_plots(OUT, grid, subjects, tables["partial_summary"], tables["covariance_rows"])
    audit_rmse = [r["normalized_rmse"] for r in audits]
    print(
        json.dumps(
            {
                "output_dir": str(OUT),
                "included_subjects": len(subjects),
                "excluded_clinical_rows": excluded_rows,
                "group_counts": metadata["group_counts"],
                "input_replicates": len(subjects) * 5,
                "qc_counts": metadata["qc"]["counts"],
                "average_audit_pass": sum(bool(r["target_usable"]) for r in audits),
                "average_audit_total": len(audits),
                "average_audit_normalized_rmse_median": float(np.median(audit_rmse)),
                "mean_replicate_correlation_median": metadata["aggregate"][
                    "mean_replicate_correlation_median"
                ],
                "noise_floor_median": metadata["aggregate"]["noise_floor_median"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
