"""
SERS Quality Control Package.

QC Pipeline:
  Level 0: Intensity Gate       → "Did SERS enhancement work?"
  Level 1: Replicate RSD        → "Is intensity reproducible?"
  Level 1: Replicate Correlation → "Is spectral shape consistent?"

Public API
----------
Intensity Gate:
    calculate_intensity_gate    - Per-spectrum enhancement check
    filter_by_intensity_gate    - Remove failed spectra

Replicate QC:
    calculate_replicate_qc      - RSD + Correlation per sample

Failure identification:
    identify_qc_failures        - Flag samples below thresholds
    summarize_qc_by_group       - Group-level QC summary

Analysis:
    calculate_variance_convergence - Optimal replicate count

Filtering:
    filter_by_correlation       - Remove inconsistent replicates

Aggregation:
    find_medoid                 - Most representative spectrum
    select_medoid_spectra       - Medoid per sample

Detection:
    detect_outliers             - Z-score / IQR outlier detection

Pipeline:
    run_qc_pipeline             - Full QC orchestration

Helpers:
    interpolate_to_grid         - Interpolation to common grid
    group_spectra_by_sample     - Group by (group, sample_id)

Usage
-----
    from sers.config import load_config
    from sers.qc import run_qc_pipeline

    config = load_config()
    gate_df, qc_stats, failures, summary = run_qc_pipeline(
        spectra, common_grid, qc_config=config.qc
    )
"""

from .qc import (
    # Intensity Gate
    calculate_intensity_gate,
    filter_by_intensity_gate,
    # Core QC
    calculate_replicate_qc,
    # Failure identification
    identify_qc_failures,
    summarize_qc_by_group,
    # Analysis
    calculate_variance_convergence,
    # Filtering
    filter_by_correlation,
    # Aggregation
    find_medoid,
    select_medoid_spectra,
    # Detection
    detect_outliers,
    # Pipeline
    run_qc_pipeline,
    # Helpers
    interpolate_to_grid,
    group_spectra_by_sample,
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
    "interpolate_to_grid",
    "group_spectra_by_sample",
]