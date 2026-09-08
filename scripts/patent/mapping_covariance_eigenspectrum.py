#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib", "openpyxl"]
# ///

# ─── How to run ───
# 1. Install uv (if not installed):
#      curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Run directly (no venv, no pip install needed):
#      uv run scripts/patent/mapping_covariance_eigenspectrum.py
# 3. Or make executable and run:
#      chmod +x scripts/patent/mapping_covariance_eigenspectrum.py && ./scripts/patent/mapping_covariance_eigenspectrum.py
# ──────────────────

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from mapping_covariance_eigenspectrum_core import compute_experiment
from mapping_covariance_eigenspectrum_outputs import write_metadata, write_report, write_tables
from mapping_repeat_average_core import load_subjects

REPO = Path("/home/user/SERS-AI")
OUT = REPO / "results" / "mapping_covariance_eigenspectrum_20260826_v1"


def main() -> None:
    from mapping_covariance_eigenspectrum_plots import write_plots

    OUT.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(402.0, 2198.0, 933, dtype=float)
    subjects, raw_axis = load_subjects(grid)
    result = compute_experiment(grid, subjects)
    write_tables(OUT, result)
    write_plots(OUT, result)
    write_metadata(OUT, result, raw_axis)
    write_report(OUT, result, raw_axis)
    print(json.dumps({"output_dir": str(OUT), "subjects": len(subjects), "repeats": int(result.repeat_counts.sum())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
