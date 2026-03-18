"""Public QC API for SERS spectroscopy."""

from .qc import (
    calculate_intensity_gate,
    calculate_replicate_qc,
    calculate_variance_convergence,
    detect_outliers,
    filter_by_correlation,
    filter_by_intensity_gate,
    find_medoid,
    identify_qc_failures,
    run_qc_pipeline,
    select_medoid_spectra,
    summarize_qc_by_group,
)

__all__ = [
    "calculate_intensity_gate",
    "filter_by_intensity_gate",
    "calculate_replicate_qc",
    "identify_qc_failures",
    "summarize_qc_by_group",
    "calculate_variance_convergence",
    "filter_by_correlation",
    "find_medoid",
    "select_medoid_spectra",
    "detect_outliers",
    "run_qc_pipeline",
]
