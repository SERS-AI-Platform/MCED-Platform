"""
Preprocessing functions and statistics for SERS spectroscopy data.

This module contains:
- Post-QC preprocessing pipeline (trim → smooth → baseline → normalize)
- Pluggable denoising methods (Savitzky-Golay, median, Gaussian, wavelet)
- Pluggable baseline correction methods (rolling min, ALS/arPLS/airPLS, etc.)
- Multiple normalization methods (SNV, robust SNV, Min-Max, L2, PQN, MSC, etc.)
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
from scipy import sparse
from scipy.ndimage import gaussian_filter1d, grey_opening
from scipy.signal import find_peaks, medfilt, savgol_filter
from scipy.sparse.linalg import spsolve

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


def median_smooth(y: np.ndarray, window_length: int = 5) -> np.ndarray:
    """Median-filter denoising for spike-like noise."""
    if window_length % 2 == 0:
        window_length += 1
    if window_length < 3:
        return y.copy()
    if len(y) < window_length:
        return y.copy()
    return medfilt(y, kernel_size=window_length)


def gaussian_smooth(y: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Gaussian smoothing for broad high-frequency noise suppression."""
    if sigma <= 0:
        return y.copy()
    return gaussian_filter1d(y, sigma=sigma, mode="nearest")


def moving_average_smooth(y: np.ndarray, window_length: int = 5) -> np.ndarray:
    """Centered moving-average smoothing."""
    if window_length < 2 or len(y) < window_length:
        return y.copy()
    kernel = np.ones(int(window_length), dtype=float) / float(window_length)
    return np.convolve(y, kernel, mode="same")


def _next_power_of_two(n: int) -> int:
    return 1 if n <= 1 else 2 ** int(np.ceil(np.log2(n)))


def _haar_denoise_padded(
    y: np.ndarray,
    threshold_scale: float = 1.0,
    level: Optional[int] = None,
) -> np.ndarray:
    """
    Dependency-free Haar wavelet soft-threshold denoising.

    This keeps wavelet denoising available without requiring PyWavelets. It is
    intentionally conservative and best treated as an experiment module.
    """
    n = len(y)
    if n < 4:
        return y.copy()

    padded_n = _next_power_of_two(n)
    padded = np.pad(y.astype(float), (0, padded_n - n), mode="edge")
    coeff = padded.copy()
    detail_slices = []
    length = padded_n
    max_level = int(np.log2(padded_n))
    if level is None:
        level = max_level
    level = max(1, min(level, max_level))

    for _ in range(level):
        half = length // 2
        avg = (coeff[:length:2] + coeff[1:length:2]) / np.sqrt(2.0)
        detail = (coeff[:length:2] - coeff[1:length:2]) / np.sqrt(2.0)
        coeff[:half] = avg
        coeff[half:length] = detail
        detail_slices.append(slice(half, length))
        length = half

    finest = coeff[detail_slices[0]]
    sigma = np.median(np.abs(finest - np.median(finest))) / 0.6745
    threshold = threshold_scale * sigma * np.sqrt(2.0 * np.log(padded_n))
    if threshold > 0:
        for slc in detail_slices:
            detail = coeff[slc]
            coeff[slc] = np.sign(detail) * np.maximum(np.abs(detail) - threshold, 0.0)

    length = padded_n // (2 ** level)
    for _ in range(level):
        half = length
        length *= 2
        avg = coeff[:half].copy()
        detail = coeff[half:length].copy()
        coeff[:length:2] = (avg + detail) / np.sqrt(2.0)
        coeff[1:length:2] = (avg - detail) / np.sqrt(2.0)

    return coeff[:n]


def wavelet_denoise(
    y: np.ndarray,
    threshold_scale: float = 1.0,
    level: Optional[int] = None,
    wavelet: str = "haar",
) -> np.ndarray:
    """Wavelet denoising. Currently supports dependency-free Haar wavelets."""
    if wavelet.lower() != "haar":
        raise PreprocessingError("Only dependency-free Haar wavelet denoising is available")
    return _haar_denoise_padded(y, threshold_scale=threshold_scale, level=level)


def whitaker_hayes_despike(
    y: np.ndarray,
    z_threshold: float = 6.0,
    window: int = 5,
    force_endpoints: bool = True,
) -> np.ndarray:
    """Remove cosmic-ray spikes via the modified Z-score of the first difference.

    Whitaker DA, Hayes K. A simple algorithm for despiking Raman spectra.
    Chemom Intell Lab Syst. 2018;179:82-4.

    Audited 2026-09-02 against the authors' own deposited R implementation
    (Mendeley Data ``sxjgbgg95y`` v1), not just the design-guide summary. The
    published article body itself was not readable from here, so anything the
    deposit does not state is flagged below rather than guessed.

    The procedure:

        grad(i)  = y(i) - y(i-1)
        z_mod(i) = 0.6745 * (grad(i) - median(grad)) / MAD(grad)
        |z_mod(i)| > threshold  ->  replace y(i) with a local mean

    Unlike :func:`sers.qc.qc.detect_cosmic_ray`, which inspects only the global
    maximum and drops the whole spectrum, this repairs each flagged point in
    place and keeps the spectrum.

    Parameters
    ----------
    y : np.ndarray
        Intensity values.
    z_threshold : float, default 6.0
        Modified Z-score cutoff. The authors' deposit hard-codes
        ``threshold = 6`` and advises starting high and decreasing.
    window : int, default 5
        Half-width of the neighbourhood averaged to replace a flagged point
        (the deposit's ``ma = 5``). Flagged neighbours are excluded from that
        average, which is what makes the method robust to broad spikes.
    force_endpoints : bool, default True
        Match the deposit's ``z[1] = z[n] = 1``: the first and last samples are
        unconditionally treated as spikes and replaced. Set False to leave the
        endpoints untouched (a deviation from the reference implementation).

    Returns
    -------
    np.ndarray
        Despiked copy of ``y``. Returned unchanged when the spectrum is too
        short or when MAD is zero (no dispersion to score against).
    """
    y_out = np.asarray(y, dtype=float).copy()
    if y_out.size < 3 or window < 1:
        return y_out

    grad = np.diff(y_out)
    mad = float(np.median(np.abs(grad - np.median(grad))))
    if mad <= 0:
        return y_out

    # The deposit uses R's mad(), whose default constant is 1.4826; 1/1.4826 =
    # 0.67449, so this is the same score as the NIST 0.6745 form.
    z_mod = np.zeros_like(y_out)
    z_mod[1:] = 0.6745 * (grad - np.median(grad)) / mad
    flagged = np.abs(z_mod) > z_threshold
    if force_endpoints:
        flagged[0] = flagged[-1] = True
    if not flagged.any():
        return y_out

    clean = ~flagged
    for idx in np.flatnonzero(flagged):
        lo = max(0, idx - window)
        hi = min(y_out.size, idx + window + 1)
        neighbours = y_out[lo:hi][clean[lo:hi]]
        if neighbours.size:
            y_out[idx] = float(neighbours.mean())
    return y_out


def despike_spectrum(
    y: np.ndarray,
    method: str = "whitaker_hayes",
    z_threshold: float = 6.0,
    window: int = 5,
    force_endpoints: bool = True,
) -> np.ndarray:
    """Remove cosmic-ray spikes by method name."""
    method = method.lower().replace("-", "_")
    aliases = {
        "wh": "whitaker_hayes",
        "whitaker": "whitaker_hayes",
        "modified_zscore": "whitaker_hayes",
    }
    method = aliases.get(method, method)

    if method == "none":
        return np.asarray(y, dtype=float).copy()
    if method == "whitaker_hayes":
        return whitaker_hayes_despike(
            y, z_threshold=z_threshold, window=window, force_endpoints=force_endpoints
        )
    raise ValueError(
        f"Unknown despiking method: '{method}'. "
        "Choose from: whitaker_hayes, none"
    )


def smooth_spectrum(
    y: np.ndarray,
    method: str = "savgol",
    window_length: int = 11,
    polyorder: int = 3,
    median_window: int = 5,
    gaussian_sigma: float = 1.0,
    moving_window: int = 5,
    wavelet_threshold: float = 1.0,
    wavelet_level: Optional[int] = None,
    wavelet: str = "haar",
) -> np.ndarray:
    """Apply smoothing/denoising by method name."""
    method = method.lower().replace("-", "_")
    aliases = {
        "sg": "savgol",
        "savitzky_golay": "savgol",
        "savitzky": "savgol",
        "median_filter": "median",
        "gauss": "gaussian",
        "moving": "moving_average",
        "ma": "moving_average",
        "wavelet": "wavelet_haar",
        "haar": "wavelet_haar",
    }
    method = aliases.get(method, method)

    if method == "savgol":
        return smooth(y, window_length=window_length, polyorder=polyorder)
    if method == "median":
        return median_smooth(y, window_length=median_window)
    if method == "gaussian":
        return gaussian_smooth(y, sigma=gaussian_sigma)
    if method == "moving_average":
        return moving_average_smooth(y, window_length=moving_window)
    if method == "wavelet_haar":
        return wavelet_denoise(
            y,
            threshold_scale=wavelet_threshold,
            level=wavelet_level,
            wavelet=wavelet,
        )
    if method == "none":
        return y.copy()

    raise ValueError(
        f"Unknown smoothing method: '{method}'. "
        "Choose from: savgol, median, gaussian, moving_average, wavelet_haar, none"
    )


# =============================================================================
# 3. Baseline Correction
# =============================================================================
def rolling_minimum_baseline(
    y: np.ndarray,
    window: int = 101
) -> np.ndarray:
    """
    Estimate baseline with a centered rolling minimum.

    This is the current production default. Because the estimated baseline is
    taken directly from local minima of the signal, subtracting it produces
    non-negative corrected intensities up to numerical precision.
    """
    return pd.Series(y).rolling(
        window, center=True, min_periods=1
    ).min().to_numpy()


def asymmetric_least_squares_baseline(
    y: np.ndarray,
    lam: float = 1e6,
    p: float = 0.01,
    niter: int = 10,
) -> np.ndarray:
    """
    Estimate baseline using asymmetric least squares.

    ALS is a common Raman/SERS baseline method for separating broad background
    from narrow peaks. Higher ``lam`` makes the baseline smoother; lower ``p``
    penalizes points above the baseline less strongly, keeping peaks from
    pulling the baseline upward.
    """
    if lam <= 0:
        raise PreprocessingError("ALS lam must be > 0")
    if not 0 < p < 1:
        raise PreprocessingError("ALS p must be in (0, 1)")
    if niter < 1:
        raise PreprocessingError("ALS niter must be >= 1")

    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 3:
        return np.full_like(y, y.min() if n else 0.0, dtype=float)

    dmat = sparse.diags(
        [1.0, -2.0, 1.0],
        [0, -1, -2],
        shape=(n, n - 2),
        dtype=float,
    )
    weights = np.ones(n)
    for _ in range(niter):
        wmat = sparse.spdiags(weights, 0, n, n)
        zmat = (wmat + lam * dmat.dot(dmat.T)).tocsc()
        baseline = spsolve(zmat, weights * y)
        weights = p * (y > baseline) + (1 - p) * (y <= baseline)
    return np.asarray(baseline)


def airpls_baseline(
    y: np.ndarray,
    lam: float = 1e5,
    niter: int = 15,
    tol: float = 1e-3,
) -> np.ndarray:
    """
    Adaptive iteratively reweighted penalized least-squares baseline.

    airPLS is frequently used for Raman/SERS fluorescence-background removal.
    It downweights positive residuals as likely peaks and iteratively fits the
    smooth lower background.
    """
    if lam <= 0:
        raise PreprocessingError("airPLS lam must be > 0")
    if niter < 1:
        raise PreprocessingError("airPLS niter must be >= 1")

    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 3:
        return np.full_like(y, y.min() if n else 0.0, dtype=float)

    dmat = sparse.diags(
        [1.0, -2.0, 1.0],
        [0, -1, -2],
        shape=(n, n - 2),
        dtype=float,
    )
    smoothness = lam * dmat.dot(dmat.T)
    weights = np.ones(n)
    total_abs = np.sum(np.abs(y)) + 1e-12
    baseline = y.copy()
    for iteration in range(1, niter + 1):
        wmat = sparse.spdiags(weights, 0, n, n)
        baseline = spsolve((wmat + smoothness).tocsc(), weights * y)
        residual = y - baseline
        negative = residual < 0
        negative_sum = np.abs(residual[negative].sum())
        if negative_sum < tol * total_abs:
            break
        weights[residual >= 0] = 0.0
        weights[negative] = np.exp(
            np.clip(iteration * np.abs(residual[negative]) / negative_sum, 0, 50)
        )
        edge_weight = np.max(weights[negative]) if np.any(negative) else 1.0
        weights[0] = edge_weight
        weights[-1] = edge_weight
    return np.asarray(baseline)


def arpls_baseline(
    y: np.ndarray,
    lam: float = 1e5,
    ratio: float = 1e-6,
    niter: int = 50,
) -> np.ndarray:
    """
    Asymmetrically reweighted penalized least-squares baseline.

    arPLS updates weights with a logistic function of residuals and is often
    less sensitive than ALS to the asymmetry parameter.
    """
    if lam <= 0:
        raise PreprocessingError("arPLS lam must be > 0")
    if niter < 1:
        raise PreprocessingError("arPLS niter must be >= 1")

    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 3:
        return np.full_like(y, y.min() if n else 0.0, dtype=float)

    dmat = sparse.diags(
        [1.0, -2.0, 1.0],
        [0, -1, -2],
        shape=(n, n - 2),
        dtype=float,
    )
    smoothness = lam * dmat.dot(dmat.T)
    weights = np.ones(n)
    baseline = y.copy()
    for _ in range(niter):
        wmat = sparse.spdiags(weights, 0, n, n)
        baseline = spsolve((wmat + smoothness).tocsc(), weights * y)
        residual = y - baseline
        negative = residual[residual < 0]
        if len(negative) == 0:
            break
        mean_neg = np.mean(negative)
        std_neg = np.std(negative)
        if std_neg < 1e-12:
            break
        exponent = np.clip(2.0 * (residual - (2.0 * std_neg - mean_neg)) / std_neg, -50, 50)
        new_weights = 1.0 / (1.0 + np.exp(exponent))
        rel_change = np.linalg.norm(new_weights - weights) / (np.linalg.norm(weights) + 1e-12)
        weights = new_weights
        if rel_change < ratio:
            break
    return np.asarray(baseline)


def moving_quantile_baseline(
    y: np.ndarray,
    window: int = 101,
    quantile: float = 0.1,
) -> np.ndarray:
    """Estimate a local lower-envelope baseline with a rolling quantile."""
    if not 0 <= quantile <= 1:
        raise PreprocessingError("Moving quantile must be in [0, 1]")
    return pd.Series(y).rolling(
        window, center=True, min_periods=1
    ).quantile(quantile).to_numpy()


def morphological_tophat_baseline(
    y: np.ndarray,
    window: int = 101,
) -> np.ndarray:
    """
    Estimate baseline by grayscale morphological opening.

    Subtracting this baseline corresponds to a white top-hat transform, a common
    morphology-based spectroscopy background correction.
    """
    if window < 1:
        raise PreprocessingError("Top-hat baseline window must be >= 1")
    return grey_opening(np.asarray(y, dtype=float), size=int(window), mode="nearest")


def polynomial_baseline(
    y: np.ndarray,
    order: int = 3,
    quantile: float = 0.2,
    n_segments: int = 64,
) -> np.ndarray:
    """
    Estimate baseline with a low-order polynomial fit to lower-envelope points.

    The spectrum is split into segments, lower quantile points are selected,
    and a polynomial is fit to those likely-background points. This avoids
    fitting directly through strong SERS peaks.
    """
    if order < 0:
        raise PreprocessingError("Polynomial baseline order must be >= 0")
    if not 0 < quantile <= 1:
        raise PreprocessingError("Polynomial baseline quantile must be in (0, 1]")

    y = np.asarray(y, dtype=float)
    n = len(y)
    if n == 0:
        return y.copy()
    if n <= order + 1:
        return np.full_like(y, y.min())

    x = np.linspace(-1.0, 1.0, n)
    segment_edges = np.linspace(0, n, min(n_segments, n) + 1, dtype=int)
    keep = np.zeros(n, dtype=bool)
    for start, end in zip(segment_edges[:-1], segment_edges[1:]):
        if end <= start:
            continue
        segment = y[start:end]
        threshold = np.quantile(segment, quantile)
        keep[start:end] = segment <= threshold

    if keep.sum() <= order:
        keep[np.argsort(y)[: order + 1]] = True

    coeff = np.polyfit(x[keep], y[keep], deg=order)
    return np.polyval(coeff, x)


def rubberband_baseline(
    y: np.ndarray,
    x: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Estimate baseline with a lower convex-hull rubberband.

    This is a classic spectroscopy baseline model: stretch a rubber band under
    the spectrum and subtract the lower envelope. It is deterministic and fast,
    but can underfit curved fluorescence backgrounds.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n == 0:
        return y.copy()
    if x is None:
        x = np.arange(n, dtype=float)
    else:
        x = np.asarray(x, dtype=float)
    if len(x) != n:
        raise PreprocessingError("Rubberband baseline requires x and y to match")

    order = np.argsort(x)
    xs = x[order]
    ys = y[order]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for point in zip(xs, ys):
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    hull_x = np.array([p[0] for p in lower])
    hull_y = np.array([p[1] for p in lower])
    baseline_sorted = np.interp(xs, hull_x, hull_y)
    baseline = np.empty_like(baseline_sorted)
    baseline[order] = baseline_sorted
    return baseline


def estimate_baseline(
    y: np.ndarray,
    method: str = "rolling_min",
    window: int = 101,
    x: Optional[np.ndarray] = None,
    als_lam: float = 1e6,
    als_p: float = 0.01,
    als_niter: int = 10,
    arpls_lam: float = 1e5,
    arpls_ratio: float = 1e-6,
    arpls_niter: int = 50,
    airpls_lam: float = 1e5,
    airpls_niter: int = 15,
    airpls_tol: float = 1e-3,
    poly_order: int = 3,
    poly_quantile: float = 0.2,
    moving_quantile: float = 0.1,
) -> np.ndarray:
    """
    Estimate baseline by method name.

    Methods
    -------
    rolling_min
        Current production default; centered rolling minimum.
    als
        Asymmetric least squares baseline.
    airpls
        Adaptive iteratively reweighted penalized least-squares baseline.
    arpls
        Asymmetrically reweighted penalized least-squares baseline.
    polynomial
        Lower-envelope polynomial baseline.
    rubberband
        Lower convex-hull baseline.
    moving_quantile
        Rolling lower quantile baseline.
    tophat
        Morphological opening baseline.
    none
        Zero baseline.
    """
    method = method.lower().replace("-", "_")
    aliases = {
        "rolling": "rolling_min",
        "rolling_minimum": "rolling_min",
        "rm": "rolling_min",
        "asls": "als",
        "poly": "polynomial",
        "quantile": "moving_quantile",
        "rolling_quantile": "moving_quantile",
        "morphological": "tophat",
        "morphology": "tophat",
        "white_tophat": "tophat",
    }
    method = aliases.get(method, method)

    if method == "rolling_min":
        return rolling_minimum_baseline(y, window=window)
    if method == "als":
        return asymmetric_least_squares_baseline(
            y, lam=als_lam, p=als_p, niter=als_niter
        )
    if method == "airpls":
        return airpls_baseline(
            y, lam=airpls_lam, niter=airpls_niter, tol=airpls_tol
        )
    if method == "arpls":
        return arpls_baseline(
            y, lam=arpls_lam, ratio=arpls_ratio, niter=arpls_niter
        )
    if method == "polynomial":
        return polynomial_baseline(y, order=poly_order, quantile=poly_quantile)
    if method == "rubberband":
        return rubberband_baseline(y, x=x)
    if method == "moving_quantile":
        return moving_quantile_baseline(y, window=window, quantile=moving_quantile)
    if method == "tophat":
        return morphological_tophat_baseline(y, window=window)
    if method == "none":
        return np.zeros_like(y, dtype=float)

    raise ValueError(
        f"Unknown baseline method: '{method}'. "
        "Choose from: rolling_min, als, airpls, arpls, polynomial, "
        "rubberband, moving_quantile, tophat, none"
    )


def baseline_correction(
    y: np.ndarray,
    window: int = 101,
    method: str = "rolling_min",
    x: Optional[np.ndarray] = None,
    als_lam: float = 1e6,
    als_p: float = 0.01,
    als_niter: int = 10,
    arpls_lam: float = 1e5,
    arpls_ratio: float = 1e-6,
    arpls_niter: int = 50,
    airpls_lam: float = 1e5,
    airpls_niter: int = 15,
    airpls_tol: float = 1e-3,
    poly_order: int = 3,
    poly_quantile: float = 0.2,
    moving_quantile: float = 0.1,
    clip_negative: bool = False,
) -> np.ndarray:
    """
    Baseline correction dispatcher.

    The default ``method="rolling_min"`` preserves the original production
    behavior. Other SERS/Raman baseline methods can be selected without
    changing the rest of the preprocessing pipeline.

    Parameters
    ----------
    y : np.ndarray
        Input intensity values
    window : int
        Rolling window size for rolling-minimum baseline
    method : str
        Baseline method: rolling_min, als, airpls, arpls, polynomial,
        rubberband, moving_quantile, tophat, none
    x : np.ndarray, optional
        Wavenumber values; used by rubberband baseline
    clip_negative : bool
        If True, clip corrected intensities below zero. Kept false by default
        because ALS/polynomial residuals can legitimately cross zero.

    Returns
    -------
    np.ndarray
        Baseline-corrected intensity values
    """
    baseline = estimate_baseline(
        y,
        method=method,
        window=window,
        x=x,
        als_lam=als_lam,
        als_p=als_p,
        als_niter=als_niter,
        arpls_lam=arpls_lam,
        arpls_ratio=arpls_ratio,
        arpls_niter=arpls_niter,
        airpls_lam=airpls_lam,
        airpls_niter=airpls_niter,
        airpls_tol=airpls_tol,
        poly_order=poly_order,
        poly_quantile=poly_quantile,
        moving_quantile=moving_quantile,
    )
    corrected = np.asarray(y, dtype=float) - baseline
    if clip_negative:
        corrected = np.maximum(corrected, 0.0)

    return corrected


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


def max_normalize(y: np.ndarray) -> np.ndarray:
    """Normalize by maximum absolute intensity."""
    denom = np.max(np.abs(y))
    if denom > 1e-10:
        return y / denom
    logger.warning("Near-zero max intensity, returning original")
    return y.copy()


def peak_normalize(
    y: np.ndarray,
    x: Optional[np.ndarray] = None,
    peak_wn: Optional[float] = None,
    window: float = 10.0,
) -> np.ndarray:
    """
    Normalize by a selected peak height.

    If ``peak_wn`` and ``x`` are supplied, the maximum absolute intensity inside
    the local window is used. Otherwise the global maximum absolute intensity is
    used.
    """
    if peak_wn is not None:
        if x is None:
            raise PreprocessingError("peak normalization with peak_wn requires x")
        mask = (x >= peak_wn - window) & (x <= peak_wn + window)
        if not mask.any():
            raise PreprocessingError("No points inside peak normalization window")
        denom = np.max(np.abs(y[mask]))
    else:
        denom = np.max(np.abs(y))
    if denom > 1e-10:
        return y / denom
    logger.warning("Near-zero peak intensity, returning original")
    return y.copy()


def mean_center(y: np.ndarray) -> np.ndarray:
    """Subtract the per-spectrum mean without scaling."""
    return y - np.mean(y)


def pareto_normalize(y: np.ndarray) -> np.ndarray:
    """Mean-center and divide by sqrt(std), a softer scaling than SNV."""
    std = np.std(y)
    centered = y - np.mean(y)
    if std > 1e-10:
        return centered / np.sqrt(std)
    logger.warning("Near-zero std in Pareto normalization, returning centered")
    return centered


def robust_snv(y: np.ndarray) -> np.ndarray:
    """Robust SNV using median and MAD instead of mean and standard deviation."""
    median = np.median(y)
    mad = np.median(np.abs(y - median))
    scale = 1.4826 * mad
    if scale > 1e-10:
        return (y - median) / scale
    logger.warning("Near-zero MAD in robust SNV, returning median-centered")
    return y - median


def pqn_normalize(y: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Probabilistic quotient normalization against a reference spectrum.

    PQN is a sample-wise dilution correction. It requires a stable reference,
    usually the median spectrum of a training cohort.
    """
    if reference is None:
        raise PreprocessingError("PQN normalization requires a reference spectrum")
    reference = np.asarray(reference, dtype=float)
    if len(reference) != len(y):
        raise PreprocessingError("PQN reference length must match spectrum length")
    mask = np.abs(reference) > 1e-10
    if not mask.any():
        raise PreprocessingError("PQN reference is near zero everywhere")
    quotients = y[mask] / reference[mask]
    factor = np.median(quotients[np.isfinite(quotients)])
    if abs(factor) > 1e-10:
        return y / factor
    logger.warning("Near-zero PQN dilution factor, returning original")
    return y.copy()


def msc_normalize(y: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Multiplicative scatter correction against a reference spectrum.

    Fits y = intercept + slope * reference and returns (y - intercept) / slope.
    """
    if reference is None:
        raise PreprocessingError("MSC normalization requires a reference spectrum")
    reference = np.asarray(reference, dtype=float)
    if len(reference) != len(y):
        raise PreprocessingError("MSC reference length must match spectrum length")
    design = np.column_stack([np.ones_like(reference), reference])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    if abs(slope) <= 1e-10:
        raise PreprocessingError("MSC slope is near zero")
    return (y - intercept) / slope


def emsc_normalize(
    y: np.ndarray,
    reference: np.ndarray,
    x: Optional[np.ndarray] = None,
    order: int = 2,
) -> np.ndarray:
    """
    Extended multiplicative scatter correction.

    EMSC extends MSC with polynomial baseline terms. The corrected spectrum is
    the measured spectrum after removing additive polynomial/background terms,
    divided by the fitted reference coefficient.
    """
    if reference is None:
        raise PreprocessingError("EMSC normalization requires a reference spectrum")
    reference = np.asarray(reference, dtype=float)
    if len(reference) != len(y):
        raise PreprocessingError("EMSC reference length must match spectrum length")
    if order < 0:
        raise PreprocessingError("EMSC polynomial order must be >= 0")
    if x is None:
        x_scaled = np.linspace(-1.0, 1.0, len(y))
    else:
        x = np.asarray(x, dtype=float)
        if len(x) != len(y):
            raise PreprocessingError("EMSC x length must match spectrum length")
        span = x.max() - x.min()
        x_scaled = np.zeros_like(x) if span <= 1e-12 else 2.0 * (x - x.min()) / span - 1.0

    poly_terms = [x_scaled ** power for power in range(order + 1)]
    design = np.column_stack([reference] + poly_terms)
    coeff = np.linalg.lstsq(design, y, rcond=None)[0]
    ref_coeff = coeff[0]
    if abs(ref_coeff) <= 1e-10:
        raise PreprocessingError("EMSC reference coefficient is near zero")
    additive = design[:, 1:] @ coeff[1:]
    return (y - additive) / ref_coeff


def normalize_spectrum(
    y: np.ndarray,
    method: str = "snv",
    x: Optional[np.ndarray] = None,
    reference: Optional[np.ndarray] = None,
    peak_wn: Optional[float] = None,
    peak_window: float = 10.0,
    emsc_order: int = 2,
) -> np.ndarray:
    """
    Apply normalization by method name.

    Parameters
    ----------
    y : np.ndarray
        Input intensity values
    method : str
        One of: "snv", "robust_snv", "minmax", "l2", "area", "max",
        "peak", "mean_center", "pareto", "pqn", "msc", "emsc", "none"

    Returns
    -------
    np.ndarray
        Normalized intensity values

    Raises
    ------
    ValueError
        If method is not recognized
    """
    method = method.lower().replace("-", "_")
    aliases = {
        "standard_normal_variate": "snv",
        "rsnv": "robust_snv",
        "min_max": "minmax",
        "vector": "l2",
        "vector_normalize": "l2",
        "total_area": "area",
        "tic": "area",
        "maximum": "max",
        "peak_area": "peak",
        "center": "mean_center",
        "mean": "mean_center",
    }
    method = aliases.get(method, method)

    if method == "snv":
        return snv(y)
    if method == "robust_snv":
        return robust_snv(y)
    if method == "minmax":
        return minmax_scale(y)
    if method == "l2":
        return vector_normalize(y)
    if method == "area":
        return area_normalize(y)
    if method == "max":
        return max_normalize(y)
    if method == "peak":
        return peak_normalize(y, x=x, peak_wn=peak_wn, window=peak_window)
    if method == "mean_center":
        return mean_center(y)
    if method == "pareto":
        return pareto_normalize(y)
    if method == "pqn":
        return pqn_normalize(y, reference=reference)
    if method == "msc":
        return msc_normalize(y, reference=reference)
    if method == "emsc":
        return emsc_normalize(y, reference=reference, x=x, order=emsc_order)
    if method == "none":
        return y.copy()

    raise ValueError(
        f"Unknown normalization method: '{method}'. "
        "Choose from: snv, robust_snv, minmax, l2, area, max, peak, "
        "mean_center, pareto, pqn, msc, emsc, none"
    )


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
    do_despike: bool = False,
    despike_method: str = "whitaker_hayes",
    despike_z_threshold: float = 7.0,
    despike_window: int = 5,
    do_trim: bool = True,
    trim_region: Tuple[float, float] = FINGERPRINT_REGION,
    do_smooth: bool = True,
    smoothing_method: str = "savgol",
    smooth_window: int = 11,
    smooth_poly: int = 3,
    median_window: int = 5,
    gaussian_sigma: float = 1.0,
    moving_window: int = 5,
    wavelet_threshold: float = 1.0,
    wavelet_level: Optional[int] = None,
    do_baseline: bool = True,
    baseline_window: int = 101,
    baseline_method: str = "rolling_min",
    baseline_als_lam: float = 1e6,
    baseline_als_p: float = 0.01,
    baseline_als_niter: int = 10,
    baseline_arpls_lam: float = 1e5,
    baseline_arpls_ratio: float = 1e-6,
    baseline_arpls_niter: int = 50,
    baseline_airpls_lam: float = 1e5,
    baseline_airpls_niter: int = 15,
    baseline_airpls_tol: float = 1e-3,
    baseline_poly_order: int = 3,
    baseline_poly_quantile: float = 0.2,
    baseline_moving_quantile: float = 0.1,
    baseline_clip_negative: bool = False,
    normalization: str = "snv",
    normalization_reference: Optional[np.ndarray] = None,
    normalization_peak_wn: Optional[float] = None,
    normalization_peak_window: float = 10.0,
    normalization_emsc_order: int = 2,
    baseline_before_smooth: bool = False,
) -> np.ndarray:
    """
    Apply full preprocessing pipeline to a single spectrum.

    Pipeline order:
        ① Trim to fingerprint region (400-2200 cm⁻¹)
        ② Savitzky-Golay smoothing
        ③ Baseline correction
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
    smoothing_method : str
        Smoothing method: savgol, median, gaussian, moving_average,
        wavelet_haar, none
    smooth_window : int
        Smoothing window size
    smooth_poly : int
        Smoothing polynomial order
    do_baseline : bool
        Whether to apply baseline correction
    baseline_window : int
        Baseline rolling window size
    baseline_method : str
        Baseline method: rolling_min, als, polynomial, rubberband, none
    normalization : str
        Normalization method: snv, robust_snv, minmax, l2, area, max, peak,
        mean_center, pareto, pqn, msc, emsc, none
    baseline_before_smooth : bool
        If True, run baseline correction before smoothing (design guide §2
        says the smoothing/baseline order is contested and both should be
        tried). Default False keeps the production order smooth → baseline.

    Returns
    -------
    np.ndarray
        Preprocessed intensity on common grid
    """
    y_proc = y.copy()

    # ⓪ Despike (opt-in) — 설계 가이드 ②단계. trim보다 앞서야 절단 경계 밖의
    #    spike도 제거된다.
    if do_despike:
        y_proc = despike_spectrum(
            y_proc,
            method=despike_method,
            z_threshold=despike_z_threshold,
            window=despike_window,
        )

    # ① Trim
    if do_trim:
        x, y_proc = trim_spectrum(x, y_proc, region=trim_region)

    # ② Smooth / ③ Baseline — order switchable (design guide §2, order is contested)
    def _smooth(v: np.ndarray) -> np.ndarray:
        return smooth_spectrum(
            v,
            method=smoothing_method,
            window_length=smooth_window,
            polyorder=smooth_poly,
            median_window=median_window,
            gaussian_sigma=gaussian_sigma,
            moving_window=moving_window,
            wavelet_threshold=wavelet_threshold,
            wavelet_level=wavelet_level,
        )

    def _baseline(v: np.ndarray) -> np.ndarray:
        return baseline_correction(
            v,
            window=baseline_window,
            method=baseline_method,
            x=x,
            als_lam=baseline_als_lam,
            als_p=baseline_als_p,
            als_niter=baseline_als_niter,
            arpls_lam=baseline_arpls_lam,
            arpls_ratio=baseline_arpls_ratio,
            arpls_niter=baseline_arpls_niter,
            airpls_lam=baseline_airpls_lam,
            airpls_niter=baseline_airpls_niter,
            airpls_tol=baseline_airpls_tol,
            poly_order=baseline_poly_order,
            poly_quantile=baseline_poly_quantile,
            moving_quantile=baseline_moving_quantile,
            clip_negative=baseline_clip_negative,
        )

    stages = [("baseline", _baseline), ("smooth", _smooth)] if baseline_before_smooth \
        else [("smooth", _smooth), ("baseline", _baseline)]
    for name, fn in stages:
        if (name == "smooth" and do_smooth) or (name == "baseline" and do_baseline):
            y_proc = fn(y_proc)

    # ④ Normalize
    y_proc = normalize_spectrum(
        y_proc,
        method=normalization,
        x=x,
        reference=normalization_reference,
        peak_wn=normalization_peak_wn,
        peak_window=normalization_peak_window,
        emsc_order=normalization_emsc_order,
    )

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
            config.preprocessing.baseline_method (str)
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
    do_despike = getattr(prep, 'do_despike', False)
    despike_method = getattr(prep, 'despike_method', 'whitaker_hayes')
    despike_z_threshold = getattr(prep, 'despike_z_threshold', 7.0)
    despike_window = getattr(prep, 'despike_window', 5)
    do_trim = getattr(prep, 'do_trim', True)
    trim_region = tuple(getattr(prep, 'trim_region', list(FINGERPRINT_REGION)))

    # Baseline settings
    smoothing_method = getattr(prep, 'smoothing_method', 'savgol')
    median_window = getattr(prep, 'median_window', 5)
    gaussian_sigma = getattr(prep, 'gaussian_sigma', 1.0)
    moving_window = getattr(prep, 'moving_window', 5)
    wavelet_threshold = getattr(prep, 'wavelet_threshold', 1.0)
    wavelet_level = getattr(prep, 'wavelet_level', None)
    do_baseline = getattr(prep, 'do_baseline', True)
    baseline_window = getattr(prep, 'baseline_window', 101)
    baseline_method = getattr(prep, 'baseline_method', 'rolling_min')
    baseline_als_lam = getattr(prep, 'baseline_als_lam', 1e6)
    baseline_als_p = getattr(prep, 'baseline_als_p', 0.01)
    baseline_als_niter = getattr(prep, 'baseline_als_niter', 10)
    baseline_arpls_lam = getattr(prep, 'baseline_arpls_lam', 1e5)
    baseline_arpls_ratio = getattr(prep, 'baseline_arpls_ratio', 1e-6)
    baseline_arpls_niter = getattr(prep, 'baseline_arpls_niter', 50)
    baseline_airpls_lam = getattr(prep, 'baseline_airpls_lam', 1e5)
    baseline_airpls_niter = getattr(prep, 'baseline_airpls_niter', 15)
    baseline_airpls_tol = getattr(prep, 'baseline_airpls_tol', 1e-3)
    baseline_poly_order = getattr(prep, 'baseline_poly_order', 3)
    baseline_poly_quantile = getattr(prep, 'baseline_poly_quantile', 0.2)
    baseline_moving_quantile = getattr(prep, 'baseline_moving_quantile', 0.1)
    baseline_clip_negative = getattr(prep, 'baseline_clip_negative', False)
    normalization_peak_wn = getattr(prep, 'normalization_peak_wn', None)
    normalization_peak_window = getattr(prep, 'normalization_peak_window', 10.0)
    normalization_emsc_order = getattr(prep, 'normalization_emsc_order', 2)

    # Prepare trimmed grid
    if do_trim:
        proc_grid = trim_to_grid(grid, region=trim_region)
    else:
        proc_grid = grid

    logger.info("Preprocessing pipeline:")
    logger.info(
        "  \u24ea Despike: %s%s",
        do_despike,
        f" (method={despike_method}, z>{despike_z_threshold})" if do_despike else "",
    )
    logger.info(f"  ① Trim: {do_trim} → region {trim_region} cm⁻¹")
    logger.info(
        f"  ② Smooth: {prep.do_smooth} "
        f"(method={smoothing_method}, window={prep.smooth_window}, poly={prep.smooth_poly})"
    )
    logger.info(
        f"  ③ Baseline: {do_baseline} "
        f"(method={baseline_method}, window={baseline_window})"
    )
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
                do_despike=do_despike,
                despike_method=despike_method,
                despike_z_threshold=despike_z_threshold,
                despike_window=despike_window,
                do_trim=do_trim,
                trim_region=trim_region,
                do_smooth=prep.do_smooth,
                smoothing_method=smoothing_method,
                baseline_before_smooth=getattr(prep, "baseline_before_smooth", False),
                smooth_window=prep.smooth_window,
                smooth_poly=prep.smooth_poly,
                median_window=median_window,
                gaussian_sigma=gaussian_sigma,
                moving_window=moving_window,
                wavelet_threshold=wavelet_threshold,
                wavelet_level=wavelet_level,
                do_baseline=do_baseline,
                baseline_window=baseline_window,
                baseline_method=baseline_method,
                baseline_als_lam=baseline_als_lam,
                baseline_als_p=baseline_als_p,
                baseline_als_niter=baseline_als_niter,
                baseline_arpls_lam=baseline_arpls_lam,
                baseline_arpls_ratio=baseline_arpls_ratio,
                baseline_arpls_niter=baseline_arpls_niter,
                baseline_airpls_lam=baseline_airpls_lam,
                baseline_airpls_niter=baseline_airpls_niter,
                baseline_airpls_tol=baseline_airpls_tol,
                baseline_poly_order=baseline_poly_order,
                baseline_poly_quantile=baseline_poly_quantile,
                baseline_moving_quantile=baseline_moving_quantile,
                baseline_clip_negative=baseline_clip_negative,
                normalization=normalization,
                normalization_peak_wn=normalization_peak_wn,
                normalization_peak_window=normalization_peak_window,
                normalization_emsc_order=normalization_emsc_order,
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
                "smoothing_method": smoothing_method,
                "baseline_method": baseline_method,
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
