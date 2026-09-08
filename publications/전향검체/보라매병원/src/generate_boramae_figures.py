# /// script
# requires-python = ">=3.10"
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
#   uv run publications/전향검체/보라매병원/src/generate_boramae_figures.py
from __future__ import annotations

import numpy as np
from boramae_data import (
    MODEL_GRID,
    build_boramae_subjects,
    build_clean_pro_subjects,
    ensure_dirs,
    load_clinical_samples,
)
from boramae_peak_modes import generate_peak_mode_outputs
from boramae_plots import (
    plot_clean_pro_comparison,
    plot_preprocessing,
    plot_prostate_grade_difference,
    write_summary,
)


def main() -> None:
    ensure_dirs()
    grid = np.load(MODEL_GRID)
    samples = load_clinical_samples()
    subjects = build_boramae_subjects(samples, grid)
    clean_pro = build_clean_pro_subjects(grid)
    plot_preprocessing(subjects, grid)
    plot_clean_pro_comparison(subjects, clean_pro, grid)
    screening_peaks, three_group_peaks = generate_peak_mode_outputs(subjects, grid)
    plot_prostate_grade_difference(subjects, grid)
    write_summary(samples, subjects, screening_peaks, three_group_peaks)
    print("Wrote Boramae prospective outputs to publications/전향검체/보라매병원")


if __name__ == "__main__":
    main()
