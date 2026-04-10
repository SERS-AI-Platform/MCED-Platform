"""Tests for sers.preprocessing — smoothing, baseline, normalization, pipeline."""

import numpy as np
import pandas as pd
import pytest

from sers.preprocessing import (
    FINGERPRINT_REGION,
    PreprocessingError,
    area_normalize,
    baseline_correction,
    minmax_scale,
    normalize_spectrum,
    preprocess_single_spectrum,
    resample,
    smooth,
    snv,
    trim_spectrum,
    trim_to_grid,
    vector_normalize,
    calculate_replicate_variance,
    identify_problematic_samples,
)


# =============================================================================
# Trimming
# =============================================================================
class TestTrimSpectrum:
    def test_trims_to_region(self):
        x = np.linspace(100, 3000, 1000)
        y = np.ones_like(x)
        x_t, y_t = trim_spectrum(x, y, region=(400, 2200))
        assert x_t.min() >= 400
        assert x_t.max() <= 2200
        assert len(x_t) < len(x)

    def test_no_data_in_region_raises(self):
        x = np.linspace(100, 300, 100)
        y = np.ones_like(x)
        with pytest.raises(PreprocessingError, match="No data points in region"):
            trim_spectrum(x, y, region=(400, 2200))

    def test_all_data_in_region(self):
        x = np.linspace(500, 2000, 100)
        y = np.ones_like(x)
        x_t, y_t = trim_spectrum(x, y, region=(400, 2200))
        assert len(x_t) == len(x)

    def test_preserves_values(self):
        x = np.array([300, 500, 1000, 1500, 2500])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        x_t, y_t = trim_spectrum(x, y, region=(400, 2200))
        np.testing.assert_array_equal(x_t, [500, 1000, 1500])
        np.testing.assert_array_equal(y_t, [2.0, 3.0, 4.0])


class TestTrimToGrid:
    def test_trims_grid(self):
        grid = np.linspace(100, 3000, 1000)
        trimmed = trim_to_grid(grid, region=(400, 2200))
        assert trimmed.min() >= 400
        assert trimmed.max() <= 2200

    def test_empty_result(self):
        grid = np.linspace(100, 300, 50)
        trimmed = trim_to_grid(grid, region=(400, 2200))
        assert len(trimmed) == 0


# =============================================================================
# Smoothing
# =============================================================================
class TestSmooth:
    def test_output_same_length(self):
        y = np.random.default_rng(0).normal(0, 1, 200)
        result = smooth(y, window_length=11, polyorder=3)
        assert len(result) == len(y)

    def test_reduces_noise(self):
        rng = np.random.default_rng(42)
        signal = np.sin(np.linspace(0, 4 * np.pi, 500))
        noisy = signal + rng.normal(0, 0.3, 500)
        smoothed = smooth(noisy, window_length=21, polyorder=3)
        # Smoothed should be closer to original signal
        assert np.std(smoothed - signal) < np.std(noisy - signal)

    def test_even_window_adjusted(self):
        y = np.ones(100)
        # Should not raise, even window adjusted internally
        result = smooth(y, window_length=10, polyorder=3)
        assert len(result) == 100

    def test_window_leq_polyorder_raises(self):
        y = np.ones(100)
        with pytest.raises(PreprocessingError, match="window_length.*must be > polyorder"):
            smooth(y, window_length=3, polyorder=5)

    def test_constant_input_unchanged(self):
        y = np.full(100, 5.0)
        result = smooth(y, window_length=11, polyorder=3)
        np.testing.assert_allclose(result, 5.0, atol=1e-10)


# =============================================================================
# Baseline Correction
# =============================================================================
class TestBaselineCorrection:
    def test_output_same_length(self):
        y = np.random.default_rng(0).random(200)
        result = baseline_correction(y, window=21)
        assert len(result) == len(y)

    def test_removes_constant_baseline(self):
        # Peaks on top of a constant offset
        y = np.full(200, 100.0)
        y[50] = 200.0  # peak
        result = baseline_correction(y, window=21)
        # The constant baseline should be removed; peak should remain
        assert result[50] > result.mean()

    def test_result_non_negative(self):
        """Rolling min baseline should produce values >= 0."""
        rng = np.random.default_rng(0)
        y = np.abs(rng.normal(100, 10, 300))
        result = baseline_correction(y, window=51)
        # Result should be >= 0 (or very close)
        assert result.min() >= -1e-10

    def test_flat_input_returns_zeros(self):
        y = np.full(100, 42.0)
        result = baseline_correction(y, window=21)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)


# =============================================================================
# SNV Normalization
# =============================================================================
class TestSNV:
    def test_zero_mean_unit_variance(self):
        rng = np.random.default_rng(0)
        y = rng.normal(100, 30, 500)
        result = snv(y)
        np.testing.assert_allclose(result.mean(), 0.0, atol=1e-10)
        np.testing.assert_allclose(result.std(), 1.0, atol=1e-10)

    def test_constant_input_zero_mean(self):
        y = np.full(100, 42.0)
        result = snv(y)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_near_zero_std_fallback(self):
        y = np.full(100, 1e-15)
        result = snv(y)
        # Should return mean-centered (all zeros essentially)
        np.testing.assert_allclose(result.mean(), 0.0, atol=1e-10)

    def test_preserves_shape(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = snv(y)
        # Relative ordering preserved
        assert np.all(np.diff(result) > 0)


# =============================================================================
# Min-Max Scaling
# =============================================================================
class TestMinMaxScale:
    def test_output_range_zero_one(self):
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        result = minmax_scale(y)
        np.testing.assert_allclose(result.min(), 0.0)
        np.testing.assert_allclose(result.max(), 1.0)

    def test_constant_returns_zeros(self):
        y = np.full(100, 5.0)
        result = minmax_scale(y)
        np.testing.assert_allclose(result, 0.0)

    def test_preserves_ordering(self):
        y = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        result = minmax_scale(y)
        np.testing.assert_array_equal(np.argsort(result), np.argsort(y))


# =============================================================================
# Vector (L2) Normalization
# =============================================================================
class TestVectorNormalize:
    def test_unit_norm(self):
        y = np.array([3.0, 4.0])
        result = vector_normalize(y)
        np.testing.assert_allclose(np.linalg.norm(result), 1.0)

    def test_zero_vector_returned_as_copy(self):
        y = np.zeros(100)
        result = vector_normalize(y)
        np.testing.assert_array_equal(result, y)
        assert result is not y  # should be a copy

    def test_preserves_direction(self):
        y = np.array([1.0, 2.0, 3.0])
        result = vector_normalize(y)
        # Ratios should be preserved
        np.testing.assert_allclose(result[1] / result[0], 2.0, atol=1e-10)
        np.testing.assert_allclose(result[2] / result[0], 3.0, atol=1e-10)


# =============================================================================
# Area Normalization
# =============================================================================
class TestAreaNormalize:
    def test_sum_abs_equals_one(self):
        y = np.array([1.0, -2.0, 3.0, -4.0, 5.0])
        result = area_normalize(y)
        np.testing.assert_allclose(np.sum(np.abs(result)), 1.0)

    def test_all_positive(self):
        y = np.array([10.0, 20.0, 30.0])
        result = area_normalize(y)
        np.testing.assert_allclose(np.sum(result), 1.0)

    def test_zero_vector_returned_as_copy(self):
        y = np.zeros(100)
        result = area_normalize(y)
        np.testing.assert_array_equal(result, y)
        assert result is not y


# =============================================================================
# normalize_spectrum dispatcher
# =============================================================================
class TestNormalizeSpectrum:
    @pytest.mark.parametrize("method", ["snv", "minmax", "l2", "area", "none"])
    def test_valid_methods(self, method):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = normalize_spectrum(y, method=method)
        assert len(result) == len(y)

    def test_none_returns_copy(self):
        y = np.array([1.0, 2.0, 3.0])
        result = normalize_spectrum(y, method="none")
        np.testing.assert_array_equal(result, y)
        assert result is not y

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown normalization method"):
            normalize_spectrum(np.array([1.0]), method="invalid")


# =============================================================================
# Resampling
# =============================================================================
class TestResample:
    def test_interpolates_to_grid(self):
        x = np.array([400.0, 800.0, 1200.0, 1600.0, 2000.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        grid = np.linspace(400, 2000, 10)
        result = resample(x, y, grid)
        assert len(result) == 10
        # At known points, should match exactly
        np.testing.assert_allclose(result[0], 1.0, atol=1e-10)
        np.testing.assert_allclose(result[-1], 5.0, atol=1e-10)

    def test_linear_interpolation(self):
        x = np.array([0.0, 10.0])
        y = np.array([0.0, 10.0])
        grid = np.array([5.0])
        result = resample(x, y, grid)
        np.testing.assert_allclose(result[0], 5.0, atol=1e-10)


# =============================================================================
# preprocess_single_spectrum (full pipeline)
# =============================================================================
class TestPreprocessSingleSpectrum:
    def test_full_pipeline(self, full_wavenumber_grid, wavenumber_grid):
        """Full pipeline: trim + smooth + baseline + snv + resample."""
        from tests.helpers import make_spectrum

        x = full_wavenumber_grid
        y = make_spectrum(x, baseline_slope=1.0)
        grid = wavenumber_grid

        result = preprocess_single_spectrum(
            x, y, grid,
            do_trim=True,
            trim_region=FINGERPRINT_REGION,
            do_smooth=True,
            smooth_window=11,
            smooth_poly=3,
            do_baseline=True,
            baseline_window=101,
            normalization="snv",
        )
        assert len(result) == len(grid)
        # SNV-normalized → mean should be near 0
        np.testing.assert_allclose(result.mean(), 0.0, atol=0.1)

    def test_no_processing(self, wavenumber_grid):
        """Pipeline with all steps disabled."""
        from tests.helpers import make_spectrum

        x = wavenumber_grid
        y = make_spectrum(x)
        grid = wavenumber_grid

        result = preprocess_single_spectrum(
            x, y, grid,
            do_trim=False,
            do_smooth=False,
            do_baseline=False,
            normalization="none",
        )
        assert len(result) == len(grid)
        # With same grid and no processing, result should match input
        np.testing.assert_allclose(result, y, atol=1e-10)


# =============================================================================
# Replicate variance
# =============================================================================
class TestCalculateReplicateVariance:
    def test_basic(self, small_grid):
        processed = {}
        rng = np.random.default_rng(42)
        for rep in range(1, 4):
            y = np.sin(small_grid / 300) + rng.normal(0, 0.01, len(small_grid))
            processed[("CRC", "001", rep)] = y

        df = calculate_replicate_variance(processed, small_grid)
        assert len(df) == 1
        assert df.iloc[0]["group"] == "CRC"
        assert df.iloc[0]["n_replicates"] == 3
        assert df.iloc[0]["mean_pairwise_correlation"] > 0.99

    def test_single_replicate_skipped(self, small_grid):
        processed = {("CRC", "001", 1): np.ones(len(small_grid))}
        df = calculate_replicate_variance(processed, small_grid)
        assert len(df) == 0


# =============================================================================
# Problematic sample identification
# =============================================================================
class TestIdentifyProblematicSamples:
    def test_flags_high_cv(self):
        df = pd.DataFrame([
            {"group": "CRC", "sample_id": "001", "mean_cv": 20.0, "mean_pairwise_correlation": 0.99},
            {"group": "CRC", "sample_id": "002", "mean_cv": 5.0, "mean_pairwise_correlation": 0.99},
        ])
        problems = identify_problematic_samples(df, cv_threshold=15.0)
        assert len(problems) == 1
        assert problems.iloc[0]["sample_id"] == "001"

    def test_flags_low_correlation(self):
        df = pd.DataFrame([
            {"group": "CRC", "sample_id": "001", "mean_cv": 5.0, "mean_pairwise_correlation": 0.80},
        ])
        problems = identify_problematic_samples(df, correlation_threshold=0.90)
        assert len(problems) == 1

    def test_no_problems(self):
        df = pd.DataFrame([
            {"group": "CRC", "sample_id": "001", "mean_cv": 5.0, "mean_pairwise_correlation": 0.99},
        ])
        problems = identify_problematic_samples(df)
        assert len(problems) == 0

    def test_empty_input(self):
        df = pd.DataFrame(columns=["group", "sample_id", "mean_cv", "mean_pairwise_correlation"])
        problems = identify_problematic_samples(df)
        assert len(problems) == 0


# =============================================================================
# Edge cases: NaN, empty, special values
# =============================================================================
class TestEdgeCases:
    def test_snv_with_nan(self):
        y = np.array([1.0, np.nan, 3.0])
        result = snv(y)
        # Result should contain NaN propagation
        assert np.isnan(result).any()

    def test_smooth_short_array(self):
        """Array shorter than window should still work (preprocessing module)."""
        y = np.array([1.0, 2.0, 3.0])
        # Window > array length: should raise because window_length > polyorder check
        # but window is adjusted. In preprocessing.smooth, it adjusts even windows
        # and requires window > polyorder. With window=11, poly=3, len=3:
        # scipy will raise because window > array length.
        # The signal.py version handles this gracefully.
        with pytest.raises(Exception):
            smooth(y, window_length=11, polyorder=3)

    def test_baseline_correction_single_value(self):
        y = np.array([42.0])
        result = baseline_correction(y, window=5)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_snv_single_value(self):
        y = np.array([5.0])
        result = snv(y)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)
