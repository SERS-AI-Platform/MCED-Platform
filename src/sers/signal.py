"""
Signal processing functions for SERS spectroscopy.

This module contains standard preprocessing functions:
- Smoothing (Savitzky-Golay filter)
- Baseline correction (polynomial fitting, asymmetric least squares)
- Normalization (SNV, Min-Max, L2)
- Resampling to common grid
"""

import logging

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)

def smooth(
    y: np.ndarray,
    window: int = 11,
    poly: int = 3
) -> np.ndarray:
    """
    Savitzky-Golay smoothing filter.

    Parameters
    ----------
    y : np.ndarray
        Input spectrum intensity
    window : int
        Window length (must be odd, >= poly + 2)
    poly : int
        Polynomial order for local fitting

    Returns
    -------
    np.ndarray
        Smoothed spectrum

    Notes
    -----
    If spectrum is shorter than window, returns original unchanged.
    Window is automatically adjusted to be odd if even.
    """
    if len(y) < window:
        return y.copy()

    # Ensure window is odd
    if window % 2 == 0:
        window += 1

    return savgol_filter(y, window_length=window, polyorder=poly, mode="interp")


def baseline_correction(y: np.ndarray, window: int = 101) -> np.ndarray:
    """
    Rolling minimum baseline subtraction.

    Parameters
    ----------
    y : np.ndarray
        Input spectrum intensity
    window : int
        Rolling window size for baseline estimation

    Returns
    -------
    np.ndarray
        Baseline-corrected spectrum

    Notes
    -----
    Uses centered rolling minimum as baseline estimate.
    Simple but effective for SERS spectra with broad backgrounds.
    """
    baseline = pd.Series(y).rolling(window, center=True, min_periods=1).min().to_numpy()
    return y - baseline


def snv(y: np.ndarray) -> np.ndarray:
    """
    Standard Normal Variate (SNV) normalization.

    Centers spectrum to zero mean and unit variance.
    Corrects for multiplicative scatter effects.

    Parameters
    ----------
    y : np.ndarray
        Input spectrum intensity

    Returns
    -------
    np.ndarray
        Normalized spectrum

    Notes
    -----
    If std is zero (constant signal), returns mean-centered only.
    """
    mean = y.mean()
    std = y.std()

    if std > 0:
        return (y - mean) / std
    return y - mean


def resample(x: np.ndarray, y: np.ndarray, new_x: np.ndarray) -> np.ndarray:
    """
    Interpolate spectrum onto new x-grid.

    Parameters
    ----------
    x : np.ndarray
        Original Raman shift values
    y : np.ndarray
        Original intensity values
    new_x : np.ndarray
        Target Raman shift grid

    Returns
    -------
    np.ndarray
        Interpolated intensity values at new_x points
    """
    f = interp1d(x, y, kind='linear', bounds_error=False, fill_value='extrapolate')
    return f(new_x)


__all__ = [
    "smooth",
    "baseline_correction",
    "snv",
    "resample",
]
