"""
Quality control functions for SERS spectroscopy.

This module contains QC functions for:
- Signal-to-noise ratio (SNR) calculation
- Replicate correlation analysis
- Medoid selection for representative spectra
- Outlier detection
"""

from typing import Dict, List, Tuple
import numpy as np
from scipy.spatial.distance import pdist, squareform

import logging

logger = logging.getLogger(__name__)


def calculate_snr(
    y: np.ndarray,
    signal_region: Tuple[int, int] = None,
    noise_region: Tuple[int, int] = None
) -> float:
    """
    Calculate Signal-to-Noise Ratio (SNR).
    
    SNR = max(signal) / std(noise)
    
    If regions not specified, uses:
    - Signal: maximum value in spectrum
    - Noise: std of entire spectrum
    
    Parameters
    ----------
    y : np.ndarray
        Spectrum intensity
    signal_region : tuple of int, optional
        (start, end) indices for signal region
    noise_region : tuple of int, optional
        (start, end) indices for noise region
        
    Returns
    -------
    float
        Signal-to-noise ratio
    """
    if signal_region is not None:
        signal = y[signal_region[0]:signal_region[1]].max()
    else:
        signal = y.max()
    
    if noise_region is not None:
        noise = y[noise_region[0]:noise_region[1]].std()
    else:
        noise = y.std()
    
    if noise < 1e-10:
        logger.warning("Noise is near zero, SNR undefined")
        return np.inf
    
    return signal / noise


def filter_by_snr(
    spectra_dict: Dict,
    min_snr: float = 10.0
) -> Dict:
    """
    Filter spectra by minimum SNR threshold.
    
    Parameters
    ----------
    spectra_dict : dict
        Dictionary with keys (group, sample_id, replicate) -> y_values
    min_snr : float
        Minimum acceptable SNR
        
    Returns
    -------
    dict
        Filtered dictionary with only spectra passing SNR threshold
    """
    filtered = {}
    rejected_count = 0
    
    for key, y in spectra_dict.items():
        snr = calculate_snr(y)
        
        if snr >= min_snr:
            filtered[key] = y
        else:
            group, sample_id, replicate = key
            logger.warning(
                f"Rejected {group}-{sample_id}-{replicate}: SNR={snr:.2f} < {min_snr}"
            )
            rejected_count += 1
    
    logger.info(
        f"SNR filter: kept {len(filtered)}/{len(spectra_dict)} spectra "
        f"({rejected_count} rejected)"
    )
    
    return filtered


def filter_by_correlation(
    spectra_dict: Dict,
    min_correlation: float = 0.90,
    by_sample: bool = True
) -> Dict:
    """
    Filter replicates by correlation with their median.
    
    For each sample, computes median spectrum and removes replicates
    with correlation below threshold.
    
    Parameters
    ----------
    spectra_dict : dict
        Dictionary with keys (group, sample_id, replicate) -> y_values
    min_correlation : float
        Minimum acceptable correlation
    by_sample : bool
        If True, filter within each sample's replicates
        If False, filter globally
        
    Returns
    -------
    dict
        Filtered dictionary
    """
    if not by_sample:
        # Global filtering not implemented yet
        logger.warning("Global correlation filtering not implemented, using by_sample=True")
    
    # Group by (group, sample_id)
    samples = {}
    for key, y in spectra_dict.items():
        group, sample_id, replicate = key
        sample_key = (group, sample_id)
        
        if sample_key not in samples:
            samples[sample_key] = []
        samples[sample_key].append((key, y))
    
    filtered = {}
    rejected_count = 0
    
    for sample_key, replicate_list in samples.items():
        if len(replicate_list) < 2:
            # Keep single replicates
            filtered[replicate_list[0][0]] = replicate_list[0][1]
            continue
        
        # Compute median spectrum
        spectra_matrix = np.vstack([y for _, y in replicate_list])
        median_spectrum = np.median(spectra_matrix, axis=0)
        
        # Check correlation with median
        for key, y in replicate_list:
            corr = np.corrcoef(y, median_spectrum)[0, 1]
            
            if corr >= min_correlation:
                filtered[key] = y
            else:
                group, sample_id, replicate = key
                logger.warning(
                    f"Rejected {group}-{sample_id}-{replicate}: "
                    f"correlation={corr:.3f} < {min_correlation}"
                )
                rejected_count += 1
    
    logger.info(
        f"Correlation filter: kept {len(filtered)}/{len(spectra_dict)} spectra "
        f"({rejected_count} rejected)"
    )
    
    return filtered


def find_medoid(
    spectra_list: List[np.ndarray],
    metric: str = "correlation"
) -> int:
    """
    Find medoid (most representative) spectrum from a list.
    
    The medoid is the spectrum with minimum total distance to all others.
    
    Parameters
    ----------
    spectra_list : list of np.ndarray
        List of spectra (all same length)
    metric : str
        Distance metric:
        - "correlation": 1 - correlation
        - "euclidean": Euclidean distance
        - "cosine": Cosine distance
        
    Returns
    -------
    int
        Index of medoid spectrum
    """
    if len(spectra_list) == 1:
        return 0
    
    # Stack spectra into matrix
    spectra_matrix = np.vstack(spectra_list)
    
    # Compute pairwise distances
    if metric == "correlation":
        # Compute correlation distances
        n = len(spectra_list)
        dist_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                corr = np.corrcoef(spectra_list[i], spectra_list[j])[0, 1]
                dist = 1 - corr
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
    else:
        # Use scipy's pdist
        distances = pdist(spectra_matrix, metric=metric)
        dist_matrix = squareform(distances)
    
    # Find spectrum with minimum total distance
    total_distances = dist_matrix.sum(axis=1)
    medoid_idx = np.argmin(total_distances)
    
    return medoid_idx


def select_medoid_spectra(
    spectra_dict: Dict
) -> Dict:
    """
    Select medoid spectrum for each sample.
    
    Parameters
    ----------
    spectra_dict : dict
        Dictionary with keys (group, sample_id, replicate) -> y_values
        
    Returns
    -------
    dict
        Dictionary with keys (group, sample_id) -> y_medoid
    """
    # Group by (group, sample_id)
    samples = {}
    for key, y in spectra_dict.items():
        group, sample_id, replicate = key
        sample_key = (group, sample_id)
        
        if sample_key not in samples:
            samples[sample_key] = []
        samples[sample_key].append((key, y))
    
    medoid_spectra = {}
    
    for sample_key, replicate_list in samples.items():
        if len(replicate_list) == 1:
            # Only one replicate, use it
            medoid_spectra[sample_key] = replicate_list[0][1]
        else:
            # Find medoid
            spectra_list = [y for _, y in replicate_list]
            medoid_idx = find_medoid(spectra_list, metric="correlation")
            medoid_spectra[sample_key] = spectra_list[medoid_idx]
            
            # Log which replicate was selected
            selected_key = replicate_list[medoid_idx][0]
            logger.debug(f"Medoid for {sample_key}: replicate {selected_key[2]}")
    
    logger.info(f"Selected {len(medoid_spectra)} medoid spectra")
    
    return medoid_spectra


def detect_outliers(
    spectra_dict: Dict,
    method: str = "zscore",
    threshold: float = 3.0
) -> List[Tuple]:
    """
    Detect outlier spectra.
    
    Parameters
    ----------
    spectra_dict : dict
        Dictionary with keys (group, sample_id, replicate) -> y_values
    method : str
        Outlier detection method:
        - "zscore": Z-score on mean intensity
        - "iqr": Interquartile range
    threshold : float
        Threshold for outlier detection
        
    Returns
    -------
    list of tuple
        List of keys for outlier spectra
    """
    # Calculate mean intensity for each spectrum
    mean_intensities = {}
    for key, y in spectra_dict.items():
        mean_intensities[key] = y.mean()
    
    values = np.array(list(mean_intensities.values()))
    
    if method == "zscore":
        # Z-score method
        mean = values.mean()
        std = values.std()
        
        outliers = []
        for key, value in mean_intensities.items():
            z_score = abs(value - mean) / std if std > 0 else 0
            if z_score > threshold:
                outliers.append(key)
                logger.warning(f"Outlier detected: {key}, z-score={z_score:.2f}")
    
    elif method == "iqr":
        # IQR method
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
        outliers = []
        for key, value in mean_intensities.items():
            if value < lower_bound or value > upper_bound:
                outliers.append(key)
                logger.warning(f"Outlier detected: {key}, value={value:.2e}")
    
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")
    
    logger.info(f"Detected {len(outliers)} outliers")
    
    return outliers