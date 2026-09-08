# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "matplotlib>=3.7",
#   "numpy>=1.24",
#   "pandas>=2.0",
#   "pyyaml>=6.0",
#   "scikit-learn>=1.3",
#   "scipy>=1.10",
# ]
# ///
# ─── How to run ───
# uv run publications/전향검체/연세세브란스\ 병원/src/generate_severance_figures.py
from __future__ import annotations

from severance_data import (
    CLEAN_MANIFEST,
    CLINICAL_MAPPING,
    FIG_DIR,
    OPTIMIZED_PREDICTIONS,
    PROCESSED_CSV,
    TABLE_DIR,
    ensure_output_dirs,
    load_cohort_spectra,
    load_preprocessing_traces,
)
from severance_lr_outputs import generate_lr_outputs
from severance_peaks import plot_peak_sets
from severance_plots import (
    plot_clean_pan_comparison,
    plot_preprocessing,
    plot_screening_performance,
    plot_stage_difference,
)


def main() -> None:
    ensure_output_dirs()
    _, lr_result = generate_lr_outputs()
    cohort = load_cohort_spectra(PROCESSED_CSV, CLEAN_MANIFEST)
    traces = load_preprocessing_traces(cohort.grid)
    plot_preprocessing(traces, FIG_DIR)
    plot_clean_pan_comparison(cohort, FIG_DIR)
    plot_screening_performance(OPTIMIZED_PREDICTIONS, FIG_DIR)
    common, differential = plot_peak_sets(
        (cohort.ynor, cohort.ypan), cohort.grid, FIG_DIR, TABLE_DIR
    )
    plot_stage_difference(cohort, CLINICAL_MAPPING, FIG_DIR, TABLE_DIR)
    print(
        "Wrote Severance matched figure suite: "
        f"YPAN={len(cohort.ypan)}, YNOR={len(cohort.ynor)}, clean CPAN={len(cohort.clean_cpan)}, "
        f"common peaks={common}, differential peaks={differential}, "
        f"repeated CV AUROC={lr_result.repeat_auc.mean():.3f}"
    )


if __name__ == "__main__":
    main()
