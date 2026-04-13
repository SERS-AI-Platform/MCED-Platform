"""Test helper functions for generating synthetic spectra."""

import numpy as np


def make_spectrum(grid, peaks=None, noise_level=0.01, baseline_slope=0.0, seed=42):
    """Generate a synthetic SERS-like spectrum with Gaussian peaks.

    Parameters
    ----------
    grid : np.ndarray
        Wavenumber values.
    peaks : list of (center, height, width), optional
        Gaussian peak parameters. Defaults to 3 typical SERS peaks.
    noise_level : float
        Gaussian noise std as fraction of max peak height.
    baseline_slope : float
        Linear baseline slope per wavenumber unit.
    seed : int
        Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)

    if peaks is None:
        peaks = [
            (620, 5000, 20),   # phenylalanine-like
            (1003, 8000, 15),  # uric acid-like
            (1600, 3000, 25),  # amide-like
        ]

    y = np.zeros_like(grid, dtype=float)
    max_height = max(h for _, h, _ in peaks)

    for center, height, width in peaks:
        y += height * np.exp(-0.5 * ((grid - center) / width) ** 2)

    # Add linear baseline (fluorescence)
    y += baseline_slope * (grid - grid[0])

    # Add noise
    y += rng.normal(0, noise_level * max_height, size=len(grid))

    return y
