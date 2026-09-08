#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#   "matplotlib>=3.7",
#   "numpy>=1.24",
#   "openpyxl>=3.1",
#   "pandas>=2.0",
#   "pyyaml>=6.0",
#   "scikit-learn>=1.3",
#   "scipy>=1.10",
# ]
# ///
# How to run:
#   uv run publications/전향검체/보라매병원/src/generate_three_cohort_spectra.py
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from three_cohort_spectra import write_three_cohort_figure


def main() -> None:
    output = write_three_cohort_figure(REPO)
    print(f"Figure: {output}")


if __name__ == "__main__":
    main()
