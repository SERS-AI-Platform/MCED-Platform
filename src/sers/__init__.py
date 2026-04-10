"""
SERS Spectroscopy Analysis Package.

A comprehensive package for SERS (Surface-Enhanced Raman Spectroscopy) data analysis,
including preprocessing, quality control, and machine learning pipelines.
"""

__version__ = "0.1.0"
__author__ = "SOLUM Healthcare"

# Configuration
from .config import Config, load_config

# I/O operations
from .io import SpectrumID, find_spectra, make_common_grid, parse_filename, read_spectrum

# Signal processing
from .signal import baseline_correction, resample, smooth, snv

# Quality control
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

# Preprocessing pipeline
from .preprocessing import (
    calculate_group_variance,
    calculate_replicate_variance,
    identify_problematic_samples,
    preprocess_spectra,
    save_processed_spectra,
)

# Validation
from .validation import (
    ValidationIssue,
    ValidationResult,
    validate_csv_structure,
    validate_spectrum,
    validate_spectra_batch,
    validate_processed_spectrum,
    detect_duplicate_samples,
)

# Data analysis and exploration
from .analysis import (
    analyze_dataset_structure,
    check_data_completeness,
    generate_dataset_report,
    get_group_statistics,
)
from .visualization import (
    build_mean_spectrum_profile,
    build_shap_spectrum_profile,
    compute_gradient_shap_values,
    plot_binary_shap_summary,
    plot_class_shap_summary,
    plot_cancer_peak_difference,
    plot_confusion_summary_bar,
    plot_group_peak_difference,
    plot_mean_spectrum,
    plot_mean_spectra_overlay,
    plot_multiclass_shap_summary,
    plot_peak_intensity_overview,
    plot_peak_intensity_profile,
    plot_shap_feature_importance_bar,
    plot_shap_mean_magnitude_spectrum,
    plot_shap_mean_signed_spectrum,
    summarize_confusion_pairs,
    summarize_spectrum_peaks,
    summarize_shap_feature_importance,
)


__all__ = [
    # Config
    "load_config",
    "Config",
    # I/O
    "read_spectrum",
    "parse_filename",
    "find_spectra",
    "make_common_grid",
    "SpectrumID",
    # Signal processing
    "smooth",
    "baseline_correction",
    "snv",
    "resample",
    # QC
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
    # Preprocessing
    "preprocess_spectra",
    "calculate_replicate_variance",
    "calculate_group_variance",
    "identify_problematic_samples",
    "save_processed_spectra",
    # Visualization
    "plot_cancer_peak_difference",
    "plot_confusion_summary_bar",
    "plot_group_peak_difference",
    "build_mean_spectrum_profile",
    "build_shap_spectrum_profile",
    "compute_gradient_shap_values",
    "plot_binary_shap_summary",
    "plot_class_shap_summary",
    "plot_mean_spectrum",
    "plot_mean_spectra_overlay",
    "plot_multiclass_shap_summary",
    "plot_peak_intensity_overview",
    "plot_peak_intensity_profile",
    "plot_shap_feature_importance_bar",
    "plot_shap_mean_magnitude_spectrum",
    "plot_shap_mean_signed_spectrum",
    "summarize_confusion_pairs",
    "summarize_spectrum_peaks",
    "summarize_shap_feature_importance",
    # Validation
    "ValidationIssue",
    "ValidationResult",
    "validate_csv_structure",
    "validate_spectrum",
    "validate_spectra_batch",
    "validate_processed_spectrum",
    "detect_duplicate_samples",
    # Analysis
    "analyze_dataset_structure",
    "check_data_completeness",
    "generate_dataset_report",
    "get_group_statistics",
]
