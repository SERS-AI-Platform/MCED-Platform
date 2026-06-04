"""
Preprocessing functions and statistics for SERS spectroscopy data.

This module contains:
- Post-QC preprocessing pipeline (trim → smooth → baseline → normalize)
- Multiple normalization methods (SNV, Min-Max, L2, Area)
- Replicate variance calculation
- Group-level variance analysis
- Quality control metrics

Preprocessing Philosophy for SERS Urine Metabolomics:
------------------------------------------------------
After QC filters out unreliable spectra, preprocessing transforms raw
intensity data into a format suitable for ML models. The key steps are:

1. Trim: Remove non-fingerprint regions (substrate features, noise)
2. Smooth: Reduce high-frequency noise while preserving peak shapes
3. Baseline correction: Remove fluorescence background
4. Normalize: Remove measurement artifacts (SERS hot-spot variation)

Normalization Choice Matters:
- SNV: Removes scale + offset, preserves peak ratios (recommended for SERS)
- Min-Max: Maps to [0,1], destroys absolute intensity info
- L2 (Vector): Unit vector, preserves ratios like SNV
- Area: Normalizes by total signal, preserves relative contributions
- None: Keeps raw intensities (use when absolute intensity is informative)

Reference:
    Previous discussion: SNV recommended over Min-Max for SERS cancer screening
    because Min-Max destroys inter-sample intensity differences that may carry
    diagnostic information (e.g., metabolite concentration differences).
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

from sers.logging_config import setup_logging
from sers.validation import validate_processed_spectrum

setup_logging()

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================
FINGERPRINT_REGION = (400, 2200)  # Default trim region (cm⁻¹)


class PreprocessingError(Exception):
    """Exception raised during preprocessing."""
    pass


# =============================================================================
# 0. Wavenumber Calibration
# =============================================================================
# Reference peak: 1001.4 cm⁻¹ (urea symmetric C-N stretch)
# Urine SERS spectra always contain urea, making this a reliable calibration anchor.
# Instrument drift causes the observed peak position to shift by a few cm⁻¹.
# Per-spectrum calibration shifts the x-axis so the reference peak aligns exactly.

DEFAULT_REFERENCE_PEAK_WN = 1001.4
DEFAULT_CALIBRATION_WINDOW = 10.0
DEFAULT_CALIBRATION_SIGMA = 3.0   # cm⁻¹, gaussian distance penalty
DEFAULT_CALIBRATION_MIN_SCORE = 1.0  # min score to accept; else skip (shift=0)


def find_reference_peak(
    x: np.ndarray,
    y: np.ndarray,
    target_wn: float = DEFAULT_REFERENCE_PEAK_WN,
    window: float = DEFAULT_CALIBRATION_WINDOW,
) -> Optional[float]:
    """
    Find the position of the reference peak nearest to target_wn.

    Uses light Savitzky-Golay smoothing + peak detection in a local window,
    with parabolic interpolation for sub-pixel accuracy.

    Parameters
    ----------
    x : np.ndarray
        Wavenumber values (cm⁻¹)
    y : np.ndarray
        Intensity values
    target_wn : float
        Expected reference peak position (cm⁻¹)
    window : float
        ± search range around target_wn (cm⁻¹)

    Returns
    -------
    float or None
        Detected peak position in cm⁻¹, or None if not found.
    """
    mask = (x >= target_wn - window) & (x <= target_wn + window)
    x_win = x[mask]
    y_win = y[mask]
    if len(x_win) < 5:
        return None

    if len(y_win) >= 7:
        y_smooth = savgol_filter(y_win, window_length=7, polyorder=2)
    else:
        y_smooth = y_win

    peaks, _ = find_peaks(y_smooth, distance=5)
    if len(peaks) == 0:
        # Fail-safe: no peak found → skip calibration
        return None

    # Distance-weighted scoring: score = prominence × gaussian(dist_to_target)
    # This avoids picking strong adjacent metabolite peaks (e.g., 1007.4, 995.9)
    # when they outshine the genuine but weaker urea peak.
    from scipy.signal import peak_prominences
    proms, _, _ = peak_prominences(y_smooth, peaks)
    sigma = DEFAULT_CALIBRATION_SIGMA
    dists = x_win[peaks] - target_wn
    weights = np.exp(-(dists / sigma) ** 2)
    scores = proms * weights

    best_arg = int(np.argmax(scores))
    best_score = float(scores[best_arg])

    # Fail-safe: if no candidate has a meaningful score, skip calibration
    if best_score < DEFAULT_CALIBRATION_MIN_SCORE:
        return None

    best = int(peaks[best_arg])
    # Sub-pixel refinement: parabolic interpolation
    if 1 <= best < len(x_win) - 1:
        y0, y1, y2 = y_smooth[best - 1], y_smooth[best], y_smooth[best + 1]
        denom = 2 * (2 * y1 - y0 - y2)
        if abs(denom) > 1e-10:
            offset = (y0 - y2) / denom
            return x_win[best] + offset * (x_win[1] - x_win[0])
    return float(x_win[best])


def calibrate_spectrum(
    x: np.ndarray,
    y: np.ndarray,
    target_wn: float = DEFAULT_REFERENCE_PEAK_WN,
    window: float = DEFAULT_CALIBRATION_WINDOW,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Shift x-axis so that the reference peak aligns to target_wn.

    Parameters
    ----------
    x : np.ndarray
        Wavenumber values
    y : np.ndarray
        Intensity values
    target_wn : float
        Target reference peak position
    window : float
        Search window around target

    Returns
    -------
    x_shifted : np.ndarray
        Calibrated wavenumber values
    y : np.ndarray
        Unchanged intensity values
    shift : float
        Applied shift in cm⁻¹ (positive = shifted right)
    """
    detected = find_reference_peak(x, y, target_wn, window)
    if detected is None:
        return x, y, 0.0
    shift = target_wn - detected
    return x + shift, y, shift


def calibrate_spectra_batch(
    raw_spectra: Dict[Tuple, Tuple[np.ndarray, np.ndarray]],
    target_wn: float = DEFAULT_REFERENCE_PEAK_WN,
    window: float = DEFAULT_CALIBRATION_WINDOW,
) -> Tuple[Dict[Tuple, Tuple[np.ndarray, np.ndarray]], pd.DataFrame]:
    """
    Apply wavenumber calibration to all spectra.

    Parameters
    ----------
    raw_spectra : dict
        (group, sample_id, replicate) -> (x, y)
    target_wn : float
        Reference peak target position
    window : float
        Search window

    Returns
    -------
    calibrated : dict
        Same structure as input with calibrated x-arrays
    shift_df : pd.DataFrame
        Calibration statistics per spectrum
    """
    calibrated = {}
    shift_records = []

    for key, (x, y) in raw_spectra.items():
        x_cal, y_cal, shift = calibrate_spectrum(x, y, target_wn, window)
        calibrated[key] = (x_cal, y_cal)
        group, sid, rep = key
        shift_records.append({
            "group": group,
            "sample_id": sid,
            "replicate": rep,
            "shift_cm1": shift,
        })

    shift_df = pd.DataFrame(shift_records)
    return calibrated, shift_df


# =============================================================================
# 1. Spectral Trimming
# =============================================================================
def trim_spectrum(
    x: np.ndarray,
    y: np.ndarray,
    region: Tuple[float, float] = FINGERPRINT_REGION
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Trim spectrum to specified wavenumber region.

    Removes non-fingerprint regions that contain substrate features
    (paper C-H/O-H stretches in 2200+ cm⁻¹) and low-wavenumber noise
    (0-400 cm⁻¹) which are not metabolite signals.

    Parameters
    ----------
    x : np.ndarray
        Wavenumber values (cm⁻¹)
    y : np.ndarray
        Intensity values
    region : tuple of float
        (min_wavenumber, max_wavenumber) to keep

    Returns
    -------
    x_trimmed, y_trimmed : tuple of np.ndarray

    Notes
    -----
    Must be applied BEFORE smoothing/baseline to avoid edge artifacts
    from signal processing at region boundaries.
    """
    mask = (x >= region[0]) & (x <= region[1])

    if not mask.any():
        raise PreprocessingError(
            f"No data points in region {region}. "
            f"Data range: {x.min():.1f}-{x.max():.1f} cm⁻¹"
        )

    n_removed = len(x) - mask.sum()
    if n_removed > 0:
        logger.debug(
            f"Trimmed {n_removed} points outside {region[0]}-{region[1]} cm⁻¹"
        )

    return x[mask], y[mask]


def trim_to_grid(
    grid: np.ndarray,
    region: Tuple[float, float] = FINGERPRINT_REGION
) -> np.ndarray:
    """
    Trim common grid to fingerprint region.

    Parameters
    ----------
    grid : np.ndarray
        Full common wavenumber grid
    region : tuple of float
        (min, max) wavenumber region

    Returns
    -------
    np.ndarray
        Trimmed grid
    """
    mask = (grid >= region[0]) & (grid <= region[1])
    return grid[mask]


# =============================================================================
# 2. Smoothing
# =============================================================================
def smooth(
    y: np.ndarray,
    window_length: int = 11,
    polyorder: int = 3
) -> np.ndarray:
    """
    Savitzky-Golay smoothing filter.

    Reduces high-frequency noise while preserving peak shapes and positions.

    Parameters
    ----------
    y : np.ndarray
        Input intensity values
    window_length : int
        Filter window size (must be odd, >= polyorder + 2)
    polyorder : int
        Polynomial order for fitting

    Returns
    -------
    np.ndarray
        Smoothed intensity values
    """
    if window_length % 2 == 0:
        window_length += 1
        logger.warning(f"Window length must be odd, adjusted to {window_length}")

    if window_length <= polyorder:
        raise PreprocessingError(
            f"window_length ({window_length}) must be > polyorder ({polyorder})"
        )

    return savgol_filter(y, window_length=window_length, polyorder=polyorder, mode="interp")


# =============================================================================
# 3. Baseline Correction
# =============================================================================
def baseline_correction(
    y: np.ndarray,
    window: int = 101
) -> np.ndarray:
    """
    Rolling minimum baseline correction.

    Estimates baseline as the rolling minimum of the spectrum,
    then subtracts it. This removes the broad fluorescence background
    common in SERS measurements.

    Parameters
    ----------
    y : np.ndarray
        Input intensity values
    window : int
        Rolling window size for minimum calculation

    Returns
    -------
    np.ndarray
        Baseline-corrected intensity values (all >= 0)
    """
    baseline = pd.Series(y).rolling(
        window, center=True, min_periods=1
    ).min().to_numpy()

    return y - baseline


# =============================================================================
# 4. Normalization Methods
# =============================================================================
def snv(y: np.ndarray) -> np.ndarray:
    """
    Standard Normal Variate (SNV) normalization.

    SNV(y) = (y - mean(y)) / std(y)

    Removes multiplicative scatter effects (SERS hot-spot variation).
    Preserves peak ratios and spectral shape.
    Most commonly used normalization for SERS spectroscopy.

    Parameters
    ----------
    y : np.ndarray
        Input intensity values

    Returns
    -------
    np.ndarray
        Normalized intensity values (zero mean, unit variance)

    Notes
    -----
    Recommended for SERS cancer screening because:
    - SERS hot-spots cause same sample to vary 2-10x in overall intensity
    - This variation is measurement artifact, not diagnostic information
    - SNV removes it while preserving metabolite peak ratios
    """
    mean = y.mean()
    std = y.std()

    if std > 1e-10:
        return (y - mean) / std
    else:
        logger.warning("Near-zero std in SNV, returning mean-centered only")
        return y - mean


def minmax_scale(y: np.ndarray) -> np.ndarray:
    """
    Min-Max scaling to [0, 1] range.

    MinMax(y) = (y - min(y)) / (max(y) - min(y))

    Maps all values to [0, 1]. Preserves spectral shape only.

    Parameters
    ----------
    y : np.ndarray
        Input intensity values

    Returns
    -------
    np.ndarray
        Scaled intensity values in [0, 1]

    Warnings
    --------
    Min-Max destroys absolute intensity differences between samples.
    If cancer patients have higher metabolite concentrations (= higher peaks),
    Min-Max makes cancer and normal spectra look identical if they have
    the same shape. Use with caution for diagnostic applications.

    May be useful when:
    - Comparing spectral shapes across very different equipment
    - Input to models that require [0,1] range (some neural networks)
    - Equipment with highly variable baseline levels
    """
    y_min = y.min()
    y_max = y.max()
    range_val = y_max - y_min

    if range_val > 1e-10:
        return (y - y_min) / range_val
    else:
        logger.warning("Near-zero range in min-max scaling, returning zeros")
        return np.zeros_like(y)


def vector_normalize(y: np.ndarray) -> np.ndarray:
    """
    Vector (L2) normalization.

    L2(y) = y / ||y||₂

    Normalizes spectrum to unit length. Preserves peak ratios
    and angular relationships between spectra.

    Parameters
    ----------
    y : np.ndarray
        Input intensity values

    Returns
    -------
    np.ndarray
        L2-normalized intensity values (unit vector)

    Notes
    -----
    Similar to SNV in preserving peak ratios, but:
    - Does not center to zero mean
    - Better geometric interpretation (cosine similarity = dot product)
    - Useful for spectral matching and library search
    """
    norm = np.linalg.norm(y)

    if norm > 1e-10:
        return y / norm
    else:
        logger.warning("Near-zero L2 norm, returning original")
        return y.copy()


def area_normalize(y: np.ndarray) -> np.ndarray:
    """
    Area (total signal) normalization.

    Area(y) = y / Σ|y|

    Normalizes by total integrated signal. Each point represents
    its fractional contribution to the total spectrum.

    Parameters
    ----------
    y : np.ndarray
        Input intensity values

    Returns
    -------
    np.ndarray
        Area-normalized intensity values (sum of abs = 1)

    Notes
    -----
    Preserves relative contributions of different spectral features.
    Useful when comparing metabolite composition ratios.
    Uses absolute values to handle baseline-corrected spectra
    that may have negative values near zero.
    """
    total = np.sum(np.abs(y))

    if total > 1e-10:
        return y / total
    else:
        logger.warning("Near-zero total area, returning original")
        return y.copy()


def normalize_spectrum(
    y: np.ndarray,
    method: str = "snv"
) -> np.ndarray:
    """
    Apply normalization by method name.

    Parameters
    ----------
    y : np.ndarray
        Input intensity values
    method : str
        One of: "snv", "minmax", "l2", "area", "none"

    Returns
    -------
    np.ndarray
        Normalized intensity values

    Raises
    ------
    ValueError
        If method is not recognized
    """
    methods = {
        "snv": snv,
        "minmax": minmax_scale,
        "l2": vector_normalize,
        "area": area_normalize,
        "none": lambda y: y.copy(),
    }

    if method not in methods:
        raise ValueError(
            f"Unknown normalization method: '{method}'. "
            f"Choose from: {list(methods.keys())}"
        )

    return methods[method](y)


# =============================================================================
# 5. Resampling
# =============================================================================
def resample(
    x: np.ndarray,
    y: np.ndarray,
    grid: np.ndarray
) -> np.ndarray:
    """
    Interpolate spectrum to common wavenumber grid.

    Parameters
    ----------
    x : np.ndarray
        Original wavenumber array
    y : np.ndarray
        Original intensity array
    grid : np.ndarray
        Target wavenumber grid

    Returns
    -------
    np.ndarray
        Resampled intensity values
    """
    return np.interp(grid, x, y)


# =============================================================================
# 6. Full Preprocessing Pipeline (Post-QC)
# =============================================================================
def preprocess_single_spectrum(
    x: np.ndarray,
    y: np.ndarray,
    grid: np.ndarray,
    do_trim: bool = True,
    trim_region: Tuple[float, float] = FINGERPRINT_REGION,
    do_smooth: bool = True,
    smooth_window: int = 11,
    smooth_poly: int = 3,
    do_baseline: bool = True,
    baseline_window: int = 101,
    normalization: str = "snv",
) -> np.ndarray:
    """
    Apply full preprocessing pipeline to a single spectrum.

    Pipeline order:
        ① Trim to fingerprint region (400-2200 cm⁻¹)
        ② Savitzky-Golay smoothing
        ③ Rolling minimum baseline correction
        ④ Normalization (SNV / MinMax / L2 / Area / None)
        ⑤ Resample to common grid

    Parameters
    ----------
    x : np.ndarray
        Wavenumber values
    y : np.ndarray
        Intensity values
    grid : np.ndarray
        Target common wavenumber grid (should be trimmed to same region)
    do_trim : bool
        Whether to trim to fingerprint region
    trim_region : tuple of float
        (min, max) wavenumber for trimming
    do_smooth : bool
        Whether to apply Savitzky-Golay smoothing
    smooth_window : int
        Smoothing window size
    smooth_poly : int
        Smoothing polynomial order
    do_baseline : bool
        Whether to apply baseline correction
    baseline_window : int
        Baseline rolling window size
    normalization : str
        Normalization method: "snv", "minmax", "l2", "area", "none"

    Returns
    -------
    np.ndarray
        Preprocessed intensity on common grid
    """
    y_proc = y.copy()

    # ① Trim
    if do_trim:
        x, y_proc = trim_spectrum(x, y_proc, region=trim_region)

    # ② Smooth
    if do_smooth:
        y_proc = smooth(y_proc, window_length=smooth_window, polyorder=smooth_poly)

    # ③ Baseline correction
    if do_baseline:
        y_proc = baseline_correction(y_proc, window=baseline_window)

    # ④ Normalize
    y_proc = normalize_spectrum(y_proc, method=normalization)

    # ⑤ Resample to common grid
    y_grid = resample(x, y_proc, grid)

    return y_grid


def preprocess_spectra(
    raw_spectra: Dict[Tuple, Tuple[np.ndarray, np.ndarray]],
    grid: np.ndarray,
    config,
    qc_passed_keys: Optional[set] = None,
) -> Tuple[Dict, pd.DataFrame]:
    """
    Preprocess raw spectra with full pipeline.

    If qc_passed_keys is provided, only those spectra are processed.
    This connects the QC pipeline output to preprocessing.

    Parameters
    ----------
    raw_spectra : dict
        Dict with keys (group, sample_id, replicate) -> (x, y)
    grid : np.ndarray
        Common wavenumber grid for resampling.
        If config.preprocessing.do_trim is True, this grid will be
        automatically trimmed to the fingerprint region.
    config : Config
        Configuration object with preprocessing parameters.
        Expected attributes:
            config.preprocessing.do_trim (bool)
            config.preprocessing.trim_region (list/tuple, e.g. [400, 2200])
            config.preprocessing.do_smooth (bool)
            config.preprocessing.smooth_window (int)
            config.preprocessing.smooth_poly (int)
            config.preprocessing.do_baseline (bool)
            config.preprocessing.baseline_window (int)
            config.preprocessing.normalization (str)
            config.preprocessing.use_snv (bool, legacy)
    qc_passed_keys : set, optional
        Set of (group, sample_id, replicate) keys that passed QC.
        If None, all spectra are processed.

    Returns
    -------
    processed : dict
        Dict with keys (group, sample_id, replicate) -> y_processed
    stats_df : pd.DataFrame
        Statistics for each spectrum (raw and processed)
    """
    prep = config.preprocessing

    # Resolve normalization method
    # Support both new 'normalization' field and legacy 'use_snv'
    normalization = getattr(prep, 'normalization', None)
    if normalization is None:
        # Legacy: fall back to use_snv flag
        normalization = "snv" if getattr(prep, 'use_snv', True) else "none"

    # Trim settings
    do_trim = getattr(prep, 'do_trim', True)
    trim_region = tuple(getattr(prep, 'trim_region', list(FINGERPRINT_REGION)))

    # Baseline settings
    do_baseline = getattr(prep, 'do_baseline', True)
    baseline_window = getattr(prep, 'baseline_window', 101)

    # Prepare trimmed grid
    if do_trim:
        proc_grid = trim_to_grid(grid, region=trim_region)
    else:
        proc_grid = grid

    logger.info("Preprocessing pipeline:")
    logger.info(f"  ① Trim: {do_trim} → region {trim_region} cm⁻¹")
    logger.info(f"  ② Smooth: {prep.do_smooth} (window={prep.smooth_window}, poly={prep.smooth_poly})")
    logger.info(f"  ③ Baseline: {do_baseline} (window={baseline_window})")
    logger.info(f"  ④ Normalization: {normalization}")
    logger.info(f"  ⑤ Resample: {len(proc_grid)} grid points")

    # Filter to QC-passed keys
    if qc_passed_keys is not None:
        keys_to_process = [k for k in raw_spectra if k in qc_passed_keys]
        n_skipped = len(raw_spectra) - len(keys_to_process)
        logger.info(f"  QC filter: processing {len(keys_to_process)}/{len(raw_spectra)} "
                     f"spectra ({n_skipped} failed QC)")
    else:
        keys_to_process = list(raw_spectra.keys())
        logger.info(f"  No QC filter: processing all {len(keys_to_process)} spectra")

    processed = {}
    stats = []

    for key in keys_to_process:
        x, y = raw_spectra[key]
        group, sid, rep = key

        try:
            y_raw = y.copy()

            # Apply full pipeline
            y_grid = preprocess_single_spectrum(
                x=x,
                y=y,
                grid=proc_grid,
                do_trim=do_trim,
                trim_region=trim_region,
                do_smooth=prep.do_smooth,
                smooth_window=prep.smooth_window,
                smooth_poly=prep.smooth_poly,
                do_baseline=do_baseline,
                baseline_window=baseline_window,
                normalization=normalization,
            )

            processed[key] = y_grid

            # Post-preprocessing validation
            vr = validate_processed_spectrum(y_grid, proc_grid, label=str(key))
            if not vr.is_valid:
                logger.warning(f"Post-preprocessing validation failed for {key}: {vr.summary()}")

            # Collect statistics
            stats.append({
                "group": group,
                "sample_id": sid,
                "replicate": rep,
                "n_points_raw": len(y_raw),
                "n_points_proc": len(y_grid),
                "raw_mean": float(y_raw.mean()),
                "raw_std": float(y_raw.std()),
                "raw_min": float(y_raw.min()),
                "raw_max": float(y_raw.max()),
                "proc_mean": float(y_grid.mean()),
                "proc_std": float(y_grid.std()),
                "proc_min": float(y_grid.min()),
                "proc_max": float(y_grid.max()),
                "normalization": normalization,
                "trimmed": do_trim,
            })

        except Exception as e:
            logger.error(f"Failed to preprocess {key}: {e}")
            raise PreprocessingError(f"Preprocessing failed for {key}: {e}")

    logger.info(f"✓ Preprocessed {len(processed)} spectra")

    return processed, pd.DataFrame(stats), proc_grid


# =============================================================================
# 7. Replicate Variance Analysis
# =============================================================================
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
        Variance statistics for each sample
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
                f"Sample {group}-{sample_id} has <2 replicates, skipping variance"
            )
            continue

        # Stack: shape (n_replicates, n_wavenumbers)
        spectra_matrix = np.vstack(spectra_list)

        mean_spectrum = spectra_matrix.mean(axis=0)
        std_spectrum = spectra_matrix.std(axis=0, ddof=1)

        # CV at each wavenumber (avoid division by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            cv_spectrum = np.where(
                np.abs(mean_spectrum) > 1e-10,
                (std_spectrum / np.abs(mean_spectrum)) * 100,
                0
            )

        # Pairwise correlations
        n_reps = len(spectra_list)
        correlations = []
        for i in range(n_reps):
            for j in range(i + 1, n_reps):
                corr = np.corrcoef(spectra_list[i], spectra_list[j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)

        mean_correlation = np.mean(correlations) if correlations else np.nan
        min_correlation = np.min(correlations) if correlations else np.nan

        # CV in peak region
        peak_mask = (grid >= peak_region[0]) & (grid <= peak_region[1])
        cv_in_peak = cv_spectrum[peak_mask].mean() if peak_mask.any() else cv_spectrum.mean()

        # Replicate SNR
        replicate_snr = (
            mean_spectrum.max() / std_spectrum.mean()
            if std_spectrum.mean() > 0 else 0
        )

        stats.append({
            'group': group,
            'sample_id': sample_id,
            'n_replicates': len(spectra_list),
            'mean_cv': float(cv_spectrum.mean()),
            'median_cv': float(np.median(cv_spectrum)),
            'max_cv': float(cv_spectrum.max()),
            'cv_at_peak_regions': float(cv_in_peak),
            'mean_pairwise_correlation': float(mean_correlation),
            'min_pairwise_correlation': float(min_correlation),
            'replicate_snr': float(replicate_snr),
        })

    return pd.DataFrame(stats)


# =============================================================================
# 8. Group-level Variance Analysis
# =============================================================================
def calculate_group_variance(
    medoid_spectra: Dict,
    grid: np.ndarray
) -> pd.DataFrame:
    """
    Calculate inter-sample variance within each group.

    Call AFTER medoid selection to analyze biological variability
    (between samples) rather than technical variability (between replicates).

    Parameters
    ----------
    medoid_spectra : dict
        Dict with keys (group, sample_id) -> y_medoid
    grid : np.ndarray
        Common wavenumber grid

    Returns
    -------
    pd.DataFrame
        Group-level variance statistics
    """
    groups = {}
    for key, y_medoid in medoid_spectra.items():
        group, sample_id = key
        if group not in groups:
            groups[group] = []
        groups[group].append(y_medoid)

    stats = []
    for group, spectra_list in groups.items():
        if len(spectra_list) < 2:
            logger.warning(f"Group {group} has <2 samples, skipping")
            continue

        spectra_matrix = np.vstack(spectra_list)
        group_mean = spectra_matrix.mean(axis=0)
        group_std = spectra_matrix.std(axis=0, ddof=1)

        with np.errstate(divide='ignore', invalid='ignore'):
            group_cv = np.where(
                np.abs(group_mean) > 1e-10,
                (group_std / np.abs(group_mean)) * 100,
                0
            )

        n_samples = len(spectra_list)
        correlations = []
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                corr = np.corrcoef(spectra_list[i], spectra_list[j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)

        group_snr = (
            group_mean.max() / group_std.mean()
            if group_std.mean() > 0 else 0
        )

        stats.append({
            'group': group,
            'n_samples': n_samples,
            'mean_intersample_cv': float(group_cv.mean()),
            'max_intersample_cv': float(group_cv.max()),
            'mean_intersample_correlation': float(np.mean(correlations)) if correlations else np.nan,
            'std_intersample_correlation': float(np.std(correlations)) if correlations else np.nan,
            'group_snr': float(group_snr),
        })

    return pd.DataFrame(stats)


# =============================================================================
# 9. Problematic Sample Identification
# =============================================================================
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

    def get_failure_reason(row):
        reasons = []
        if row['mean_cv'] > cv_threshold:
            reasons.append(f"High CV ({row['mean_cv']:.2f}%)")
        if row['mean_pairwise_correlation'] < correlation_threshold:
            reasons.append(f"Low correlation ({row['mean_pairwise_correlation']:.3f})")
        return "; ".join(reasons)

    if len(problematic) > 0:
        problematic['failure_reason'] = problematic.apply(get_failure_reason, axis=1)

    return problematic


# =============================================================================
# 10. Save Processed Data
# =============================================================================
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
            "replicate": rep,
        }
        for i, x_val in enumerate(grid):
            row[f"x_{x_val:.2f}"] = y_proc[i]
        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    logger.info(f"Processed spectra saved to {output_path}")
