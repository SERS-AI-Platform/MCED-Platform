"""
SERS Spectroscopy Analysis Package

A comprehensive package for SERS (Surface-Enhanced Raman Spectroscopy) data analysis,
including preprocessing, quality control, and machine learning pipelines.
"""

__version__ = "0.1.0"
__author__ = "SOLUM Healthcare"

# Configuration
from .config import load_config, Config

# I/O operations
from .io import (
    read_spectrum,
    parse_filename,
    find_spectra,
    make_common_grid,
    SpectrumID
)

# Signal processing
from .signal import (
    smooth,
    baseline_correction,
    snv
)

# Quality control
#from .qc import ()

# Preprocessing pipeline
from .preprocessing import (
    preprocess_spectra,
    calculate_replicate_variance,
    calculate_group_variance,
    identify_problematic_samples,
    save_processed_spectra
)

# Data analysis and exploration
from .analysis import (
    analyze_dataset_structure,
    check_data_completeness,
    generate_dataset_report,
    get_group_statistics
)

# Visualization
from .visualization import (
    visualize_raw_spectra,
    visualize_preprocessed_spectra_by_replicate,
    visualize_raw_spectra_by_sample,

    plot_sample_distribution_pie,
    plot_spectra_count_bar,
    # heatmap
    plot_variance_heatmap 
)

__all__ = [
    # Config
    'load_config',
    'Config',
    
    # I/O
    'read_spectrum',
    'parse_filename',
    'find_spectra',
    'make_common_grid',
    'SpectrumID',
    
    # Signal processing
    'smooth',
    'baseline_correction',
    'snv',
    'normalize',
    'resample',
    
    # QC
    'filter_by_snr',
    'filter_by_correlation',
    'find_medoid',
    'calculate_snr',
    
    # Preprocessing
    'preprocess_spectra',
    'calculate_replicate_variance',
    'calculate_group_variance',
    'identify_problematic_samples',
    'save_processed_spectra',
    
    # Analysis
    'analyze_dataset_structure',
    'check_data_completeness',
    'generate_dataset_report',
    'get_group_statistics',
    
    # Visualization
    'visualize_preprocessing',
    'plot_replicate_variance_by_group',
    'plot_variance_heatmap',
    'plot_cv_boxplot',
    'plot_group_variance_summary',
]