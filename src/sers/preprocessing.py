"""
Preprocessing functions and statistics for SERS spectroscopy data.

This module contains:
- Preprocessing pipeline functions
- Replicate variance calculation
- Group-level variance analysis
- Quality control metrics
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import variation

from .signal import smooth, baseline_correction, snv, resample

import logging

logger = logging.getLogger(__name__)


class PreprocessingError(Exception):
    """Exception raised during preprocessing."""
    pass


def preprocess_spectra(
    raw_spectra: Dict,
    grid: np.ndarray,
    config
) -> Tuple[Dict, pd.DataFrame]:
    """
    Preprocess raw spectra with smoothing, baseline correction, and normalization.
    
    Parameters
    ----------
    raw_spectra : dict
        Dict with keys (group, sample_id, replicate) -> (x, y)
    grid : np.ndarray
        Common wavenumber grid for resampling
    config : Config
        Configuration object with preprocessing parameters
        
    Returns
    -------
    processed : dict
        Dict with keys (group, sample_id, replicate) -> y_processed
    stats_df : pd.DataFrame
        Statistics for each spectrum (raw and processed)
    """
    processed = {}
    stats = []

    for key, (x, y) in raw_spectra.items():
        group, sid, rep = key

        try:
            y_raw = y.copy()

            # Apply preprocessing steps
            if config.preprocessing.do_smooth:
                y = smooth(
                    y,
                    config.preprocessing.smooth_window,
                    config.preprocessing.smooth_poly
                )
            
            y = baseline_correction(y, config.preprocessing.baseline_window)

            if config.preprocessing.use_snv:
                y = snv(y)

            # Resample to common grid
            y_grid = resample(x, y, grid)

            processed[key] = y_grid

            # Collect statistics
            stats.append({
                "group": group,
                "sample_id": sid,
                "replicate": rep,
                "n_points": len(y_grid),
                "raw_mean": float(y_raw.mean()),
                "raw_std": float(y_raw.std()),
                "raw_min": float(y_raw.min()),
                "raw_max": float(y_raw.max()),
                "proc_mean": float(y_grid.mean()),
                "proc_std": float(y_grid.std()),
                "proc_min": float(y_grid.min()),
                "proc_max": float(y_grid.max())
            })
        
        except Exception as e:
            logger.error(f"Failed to preprocess {key}: {e}")
            raise PreprocessingError(f"Preprocessing failed for {key}: {e}")
    
    return processed, pd.DataFrame(stats)


def calculate_replicate_variance(
    processed_spectra: Dict,
    grid: np.ndarray,
    peak_region: Tuple[float, float] = (600, 1800)
) -> pd.DataFrame:
    """
    Calculate variance statistics across replicates for each sample.
    
    This function computes:
    - Coefficient of Variation (CV) across the spectrum
    - Pairwise correlations between replicates
    - Signal-to-noise metrics
    
    Parameters
    ----------
    processed_spectra : dict
        Dict with keys (group, sample_id, replicate) -> y_processed
    grid : np.ndarray
        Common wavenumber grid
    peak_region : tuple of float
        (min, max) wavenumber range for focused CV calculation
        
    Returns
    -------
    pd.DataFrame
        Variance statistics for each sample with columns:
        - group, sample_id, n_replicates
        - mean_cv, median_cv, max_cv, cv_at_peak_regions
        - mean_pairwise_correlation, min_pairwise_correlation
        - replicate_snr
    """
    # Group by (group, sample_id)
    samples = {}
    for key, y_proc in processed_spectra.items():
        group, sample_id, replicate = key
        sample_key = (group, sample_id)
        
        if sample_key not in samples:
            samples[sample_key] = []
        samples[sample_key].append(y_proc)
    
    stats = []
    
    for (group, sample_id), spectra_list in samples.items():
        if len(spectra_list) < 2:
            logger.warning(
                f"Sample {group}-{sample_id} has <2 replicates, skipping variance calculation"
            )
            continue
        
        # Stack replicates: shape (n_replicates, n_wavenumbers)
        spectra_matrix = np.vstack(spectra_list)
        
        # Calculate statistics
        mean_spectrum = spectra_matrix.mean(axis=0)
        std_spectrum = spectra_matrix.std(axis=0, ddof=1)
        
        # Coefficient of Variation (CV) at each wavenumber
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            cv_spectrum = np.where(
                mean_spectrum != 0,
                (std_spectrum / np.abs(mean_spectrum)) * 100,
                0
            )
        
        # Mean pairwise correlation between replicates
        n_reps = len(spectra_list)
        correlations = []
        for i in range(n_reps):
            for j in range(i+1, n_reps):
                corr = np.corrcoef(spectra_list[i], spectra_list[j])[0, 1]
                correlations.append(corr)
        
        mean_correlation = np.mean(correlations)
        min_correlation = np.min(correlations)
        
        # CV in peak region
        peak_mask = (grid >= peak_region[0]) & (grid <= peak_region[1])
        cv_in_peak = cv_spectrum[peak_mask].mean()
        
        # Replicate SNR: signal (mean peak) / noise (std across replicates)
        replicate_snr = mean_spectrum.max() / std_spectrum.mean() if std_spectrum.mean() > 0 else 0
        
        stats.append({
            'group': group,
            'sample_id': sample_id,
            'n_replicates': len(spectra_list),
            'mean_cv': cv_spectrum.mean(),
            'median_cv': np.median(cv_spectrum),
            'max_cv': cv_spectrum.max(),
            'cv_at_peak_regions': cv_in_peak,
            'mean_pairwise_correlation': mean_correlation,
            'min_pairwise_correlation': min_correlation,
            'replicate_snr': replicate_snr
        })
    
    return pd.DataFrame(stats)


def calculate_group_variance(
    medoid_spectra: Dict,
    grid: np.ndarray
) -> pd.DataFrame:
    """
    Calculate inter-sample variance within each group.
    
    This should be called AFTER medoid selection to analyze
    biological variability (between samples) rather than
    technical variability (between replicates).
    
    Parameters
    ----------
    medoid_spectra : dict
        Dict with keys (group, sample_id) -> y_medoid
    grid : np.ndarray
        Common wavenumber grid
        
    Returns
    -------
    pd.DataFrame
        Group-level variance statistics with columns:
        - group, n_samples
        - mean_intersample_cv, max_intersample_cv
        - mean_intersample_correlation, std_intersample_correlation
        - group_snr
    """
    # Group medoid spectra by group
    groups = {}
    for key, y_medoid in medoid_spectra.items():
        group, sample_id = key
        
        if group not in groups:
            groups[group] = []
        groups[group].append(y_medoid)
    
    stats = []
    
    for group, spectra_list in groups.items():
        if len(spectra_list) < 2:
            logger.warning(
                f"Group {group} has <2 samples, skipping group variance calculation"
            )
            continue
        
        spectra_matrix = np.vstack(spectra_list)
        
        # Calculate group-level statistics
        group_mean = spectra_matrix.mean(axis=0)
        group_std = spectra_matrix.std(axis=0, ddof=1)
        
        # Between-sample CV
        with np.errstate(divide='ignore', invalid='ignore'):
            group_cv = np.where(
                group_mean != 0,
                (group_std / np.abs(group_mean)) * 100,
                0
            )
        
        # Pairwise correlations between samples
        n_samples = len(spectra_list)
        correlations = []
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                corr = np.corrcoef(spectra_list[i], spectra_list[j])[0, 1]
                correlations.append(corr)
        
        # Group SNR
        group_snr = group_mean.max() / group_std.mean() if group_std.mean() > 0 else 0
        
        stats.append({
            'group': group,
            'n_samples': n_samples,
            'mean_intersample_cv': group_cv.mean(),
            'max_intersample_cv': group_cv.max(),
            'mean_intersample_correlation': np.mean(correlations),
            'std_intersample_correlation': np.std(correlations),
            'group_snr': group_snr
        })
    
    return pd.DataFrame(stats)


def identify_problematic_samples(
    variance_df: pd.DataFrame,
    cv_threshold: float = 15.0,
    correlation_threshold: float = 0.90
) -> pd.DataFrame:
    """
    Identify samples with poor replicate reproducibility.
    
    Parameters
    ----------
    variance_df : pd.DataFrame
        Output from calculate_replicate_variance()
    cv_threshold : float
        Maximum acceptable mean CV (%)
    correlation_threshold : float
        Minimum acceptable mean pairwise correlation
        
    Returns
    -------
    pd.DataFrame
        Problematic samples that fail QC criteria
    """
    problematic = variance_df[
        (variance_df['mean_cv'] > cv_threshold) |
        (variance_df['mean_pairwise_correlation'] < correlation_threshold)
    ].copy()
    
    # Add failure reason
    def get_failure_reason(row):
        reasons = []
        if row['mean_cv'] > cv_threshold:
            reasons.append(f"High CV ({row['mean_cv']:.2f}%)")
        if row['mean_pairwise_correlation'] < correlation_threshold:
            reasons.append(f"Low correlation ({row['mean_pairwise_correlation']:.3f})")
        return "; ".join(reasons)
    
    problematic['failure_reason'] = problematic.apply(get_failure_reason, axis=1)
    
    return problematic


def save_processed_spectra(
    processed_spectra: Dict,
    grid: np.ndarray,
    output_path: Path
) -> None:
    """
    Save processed spectra to CSV file.
    
    Parameters
    ----------
    processed_spectra : dict
        Dict with keys (group, sample_id, replicate) -> y_processed
    grid : np.ndarray
        Common wavenumber grid
    output_path : Path
        Path to save CSV file
    """
    data = []
    for key, y_proc in processed_spectra.items():
        group, sid, rep = key
        row = {
            "group": group,
            "sample_id": sid,
            "replicate": rep
        }
        # Add intensity values
        for i, x_val in enumerate(grid):
            row[f"x_{x_val:.2f}"] = y_proc[i]
        data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    logger.info(f"Processed spectra saved to {output_path}")