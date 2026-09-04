"""
SERS Visualization Package.

Submodules:
    spectra      — Raw/preprocessed/mean spectrum plots
    distribution — Sample/spectra distribution and QC variance
    analysis     — Peak-difference, confusion-pair analysis
    fonts        — Korean font setup for matplotlib
    shap         — SHAP-based model explanations
"""

# --- spectra ---
# --- analysis ---
from .analysis import (
    plot_cancer_peak_difference,
    plot_confusion_summary_bar,
    plot_group_peak_difference,
    plot_peak_intensity_overview,
    plot_peak_intensity_profile,
    summarize_confusion_pairs,
    summarize_spectrum_peaks,
)

# --- fonts ---
from .fonts import available_korean_font, use_korean_font

# --- distribution ---
from .distribution import (
    plot_replicate_variance_by_group,
    plot_sample_distribution_pie,
    plot_spectra_count_bar,
    plot_variance_heatmap,
)

# --- shap ---
from .shap import (
    build_shap_spectrum_profile,
    compute_gradient_shap_values,
    plot_binary_shap_summary,
    plot_class_shap_summary,
    plot_multiclass_shap_summary,
    plot_shap_feature_importance_bar,
    plot_shap_mean_magnitude_spectrum,
    plot_shap_mean_signed_spectrum,
    summarize_shap_feature_importance,
)
from .spectra import (
    build_mean_spectrum_profile,
    plot_mean_spectra_overlay,
    plot_mean_spectrum,
    visualize_preprocessed_spectra_by_replicate,
    visualize_raw_spectra,
    visualize_raw_spectra_by_sample,
)

__all__ = [
    # spectra
    "build_mean_spectrum_profile",
    "plot_mean_spectrum",
    "plot_mean_spectra_overlay",
    "visualize_preprocessed_spectra_by_replicate",
    "visualize_raw_spectra",
    "visualize_raw_spectra_by_sample",
    # distribution
    "plot_replicate_variance_by_group",
    "plot_sample_distribution_pie",
    "plot_spectra_count_bar",
    "plot_variance_heatmap",
    # fonts
    "available_korean_font",
    "use_korean_font",
    # analysis
    "plot_cancer_peak_difference",
    "plot_confusion_summary_bar",
    "plot_group_peak_difference",
    "plot_peak_intensity_overview",
    "plot_peak_intensity_profile",
    "summarize_confusion_pairs",
    "summarize_spectrum_peaks",
    # shap
    "build_shap_spectrum_profile",
    "compute_gradient_shap_values",
    "plot_binary_shap_summary",
    "plot_class_shap_summary",
    "plot_multiclass_shap_summary",
    "plot_shap_feature_importance_bar",
    "plot_shap_mean_magnitude_spectrum",
    "plot_shap_mean_signed_spectrum",
    "summarize_shap_feature_importance",
]
