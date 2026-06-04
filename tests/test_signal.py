"""Tests for sers.signal — the thin signal-processing wrapper module."""

import numpy as np

from sers.signal import baseline_correction, resample, smooth, snv


class TestSmooth:
    def test_output_length(self):
        y = np.random.default_rng(0).random(200)
        assert len(smooth(y)) == 200

    def test_short_input_returns_copy(self):
        y = np.array([1.0, 2.0, 3.0])
        result = smooth(y, window=11)
        np.testing.assert_array_equal(result, y)
        assert result is not y  # Should be a copy

    def test_even_window_adjusted(self):
        y = np.ones(100)
        result = smooth(y, window=10, poly=3)
        assert len(result) == 100

    def test_reduces_noise(self):
        rng = np.random.default_rng(42)
        signal = np.sin(np.linspace(0, 4 * np.pi, 500))
        noisy = signal + rng.normal(0, 0.3, 500)
        smoothed = smooth(noisy, window=21, poly=3)
        assert np.std(smoothed - signal) < np.std(noisy - signal)


class TestBaselineCorrection:
    def test_flat_baseline_removed(self):
        y = np.full(100, 50.0)
        result = baseline_correction(y)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_peak_preserved(self):
        y = np.full(200, 10.0)
        y[100] = 100.0
        result = baseline_correction(y, window=21)
        assert result[100] > result.mean()


class TestSNV:
    def test_zero_mean(self):
        y = np.random.default_rng(0).normal(100, 20, 500)
        result = snv(y)
        np.testing.assert_allclose(result.mean(), 0.0, atol=1e-10)

    def test_unit_std(self):
        y = np.random.default_rng(0).normal(100, 20, 500)
        result = snv(y)
        np.testing.assert_allclose(result.std(), 1.0, atol=1e-10)

    def test_constant_returns_mean_centered(self):
        y = np.full(100, 42.0)
        result = snv(y)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)


class TestResample:
    def test_interpolation(self):
        x = np.array([0.0, 10.0])
        y = np.array([0.0, 10.0])
        new_x = np.array([5.0])
        result = resample(x, y, new_x)
        np.testing.assert_allclose(result[0], 5.0, atol=1e-10)

    def test_extrapolation(self):
        x = np.array([0.0, 10.0])
        y = np.array([0.0, 10.0])
        new_x = np.array([15.0])
        result = resample(x, y, new_x)
        # interp1d with fill_value='extrapolate'
        assert result[0] > 10.0

    def test_same_grid(self):
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        result = resample(x, y, x)
        np.testing.assert_allclose(result, y, atol=1e-10)
