"""Tests for sers.preprocessing — smoothing, baseline, normalization, pipeline."""

import numpy as np
import pandas as pd
import pytest

from sers.preprocessing import (
    FINGERPRINT_REGION,
    PreprocessingError,
    airpls_baseline,
    area_normalize,
    arpls_baseline,
    asymmetric_least_squares_baseline,
    baseline_correction,
    calculate_replicate_variance,
    despike_spectrum,
    emsc_normalize,
    estimate_baseline,
    gaussian_smooth,
    identify_problematic_samples,
    max_normalize,
    mean_center,
    median_smooth,
    minmax_scale,
    morphological_tophat_baseline,
    moving_average_smooth,
    moving_quantile_baseline,
    msc_normalize,
    normalize_spectrum,
    pareto_normalize,
    peak_normalize,
    polynomial_baseline,
    pqn_normalize,
    preprocess_single_spectrum,
    resample,
    robust_snv,
    rolling_minimum_baseline,
    rubberband_baseline,
    smooth,
    smooth_spectrum,
    snv,
    trim_spectrum,
    trim_to_grid,
    vector_normalize,
    wavelet_denoise,
    whitaker_hayes_despike,
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

    @pytest.mark.parametrize(
        "method",
        ["savgol", "median", "gaussian", "moving_average", "wavelet_haar", "none"],
    )
    def test_smoothing_dispatch_methods(self, method):
        rng = np.random.default_rng(0)
        y = np.sin(np.linspace(0, 4 * np.pi, 128)) + rng.normal(0, 0.1, 128)
        result = smooth_spectrum(y, method=method, window_length=11, polyorder=3)
        assert len(result) == len(y)
        assert np.isfinite(result).all()

    def test_median_smooth_suppresses_spike(self):
        y = np.ones(21)
        y[10] = 100
        result = median_smooth(y, window_length=5)
        assert result[10] < 10

    def test_gaussian_smooth_finite(self):
        y = np.random.default_rng(0).normal(0, 1, 100)
        result = gaussian_smooth(y, sigma=1.5)
        assert np.isfinite(result).all()

    def test_moving_average_smooth_finite(self):
        y = np.random.default_rng(0).normal(0, 1, 100)
        result = moving_average_smooth(y, window_length=5)
        assert np.isfinite(result).all()

    def test_wavelet_denoise_finite(self):
        y = np.random.default_rng(0).normal(0, 1, 100)
        result = wavelet_denoise(y)
        assert len(result) == len(y)
        assert np.isfinite(result).all()

    def test_unknown_smoothing_method_raises(self):
        with pytest.raises(ValueError, match="Unknown smoothing method"):
            smooth_spectrum(np.ones(20), method="invalid")


# =============================================================================
# Despiking (Whitaker-Hayes) — 설계 가이드 ②단계
# =============================================================================
class TestDespike:
    @staticmethod
    def _spiked():
        y = 50 + 10 * np.sin(np.linspace(0, 4 * np.pi, 200))
        y[80] += 500
        return y

    def test_replaces_spike_with_local_mean(self):
        y = self._spiked()
        result = whitaker_hayes_despike(y)
        assert result[80] < 100
        assert abs(result[80] - np.mean([y[78], y[79], y[81], y[82]])) < 5

    def test_clean_spectrum_interior_untouched(self):
        """저자 구현은 끝점을 무조건 교체하므로 내부만 불변이어야 한다."""
        y = 50 + 10 * np.sin(np.linspace(0, 4 * np.pi, 200))
        assert np.allclose(whitaker_hayes_despike(y)[1:-1], y[1:-1])

    def test_force_endpoints_matches_reference_implementation(self):
        """deposit의 z[1] = z[n] = 1 — 양 끝점은 항상 교체된다."""
        y = 50 + 10 * np.sin(np.linspace(0, 4 * np.pi, 200))
        forced = whitaker_hayes_despike(y, force_endpoints=True)
        assert forced[0] != y[0]
        assert forced[-1] != y[-1]

    def test_force_endpoints_disabled_leaves_clean_spectrum_untouched(self):
        y = 50 + 10 * np.sin(np.linspace(0, 4 * np.pi, 200))
        assert np.allclose(whitaker_hayes_despike(y, force_endpoints=False), y)

    def test_default_threshold_matches_authors_deposit(self):
        """Mendeley deposit sxjgbgg95y의 threshold = 6."""
        import inspect

        assert inspect.signature(whitaker_hayes_despike).parameters["z_threshold"].default == 6.0

    def test_preserves_length_and_finiteness(self):
        result = whitaker_hayes_despike(self._spiked())
        assert len(result) == 200
        assert np.isfinite(result).all()

    def test_does_not_mutate_input(self):
        y = self._spiked()
        original = y.copy()
        whitaker_hayes_despike(y)
        assert np.array_equal(y, original)

    def test_higher_threshold_keeps_spike(self):
        y = self._spiked()
        assert whitaker_hayes_despike(y, z_threshold=1e6)[80] == pytest.approx(y[80])

    def test_constant_spectrum_returns_copy(self):
        """MAD가 0이면 점수를 낼 수 없으므로 끝점 강제 이전에 그대로 반환한다."""
        y = np.ones(50)
        assert np.allclose(whitaker_hayes_despike(y), y)

    def test_short_spectrum_returns_copy(self):
        y = np.array([1.0, 2.0])
        assert np.allclose(whitaker_hayes_despike(y), y)

    @pytest.mark.parametrize("method", ["whitaker_hayes", "wh", "whitaker", "none"])
    def test_dispatch_methods(self, method):
        result = despike_spectrum(self._spiked(), method=method)
        assert len(result) == 200
        assert np.isfinite(result).all()

    def test_none_is_passthrough(self):
        y = self._spiked()
        assert np.allclose(despike_spectrum(y, method="none"), y)

    def test_unknown_despike_method_raises(self):
        with pytest.raises(ValueError, match="Unknown despiking method"):
            despike_spectrum(np.ones(20), method="invalid")


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

    def test_rolling_minimum_baseline_matches_default(self):
        y = np.array([5.0, 4.0, 10.0, 4.0, 5.0])
        baseline = rolling_minimum_baseline(y, window=3)
        result = baseline_correction(y, window=3)
        np.testing.assert_allclose(result, y - baseline)

    @pytest.mark.parametrize(
        "method",
        [
            "rolling_min",
            "als",
            "airpls",
            "arpls",
            "polynomial",
            "rubberband",
            "moving_quantile",
            "tophat",
            "none",
        ],
    )
    def test_baseline_dispatch_methods(self, method):
        x = np.linspace(400, 2200, 200)
        y = 0.002 * (x - 400) + np.exp(-0.5 * ((x - 1000) / 20) ** 2) * 10
        baseline = estimate_baseline(y, method=method, x=x, window=21)
        corrected = baseline_correction(y, method=method, x=x, window=21)
        assert len(baseline) == len(y)
        assert len(corrected) == len(y)
        assert np.isfinite(baseline).all()
        assert np.isfinite(corrected).all()

    def test_als_baseline_preserves_peak_residual(self):
        x = np.linspace(400, 2200, 300)
        broad_baseline = 20 + 0.002 * (x - 400) + 0.000001 * (x - 1300) ** 2
        peak = 30 * np.exp(-0.5 * ((x - 1000) / 16) ** 2)
        y = broad_baseline + peak
        corrected = baseline_correction(
            y, method="als", als_lam=1e5, als_p=0.01, als_niter=8
        )
        assert corrected[np.argmax(peak)] > np.median(corrected) + 10

    def test_polynomial_baseline_is_finite(self):
        y = np.linspace(0, 1, 100) + np.sin(np.linspace(0, 6, 100)) * 0.1
        result = polynomial_baseline(y, order=2)
        assert len(result) == len(y)
        assert np.isfinite(result).all()

    def test_rubberband_baseline_is_finite(self):
        x = np.linspace(400, 2200, 100)
        y = np.sin(np.linspace(0, 6, 100)) + np.linspace(0, 1, 100)
        result = rubberband_baseline(y, x=x)
        assert len(result) == len(y)
        assert np.isfinite(result).all()

    def test_asymmetric_least_squares_baseline_is_finite(self):
        y = np.linspace(0, 1, 100) + np.exp(-0.5 * ((np.arange(100) - 50) / 5) ** 2)
        result = asymmetric_least_squares_baseline(y, lam=1e4, p=0.01, niter=5)
        assert len(result) == len(y)
        assert np.isfinite(result).all()

    def test_airpls_baseline_is_finite(self):
        y = np.linspace(0, 1, 100) + np.exp(-0.5 * ((np.arange(100) - 50) / 5) ** 2)
        result = airpls_baseline(y, lam=1e4, niter=5)
        assert len(result) == len(y)
        assert np.isfinite(result).all()

    def test_arpls_baseline_is_finite(self):
        y = np.linspace(0, 1, 100) + np.exp(-0.5 * ((np.arange(100) - 50) / 5) ** 2)
        result = arpls_baseline(y, lam=1e4, niter=5)
        assert len(result) == len(y)
        assert np.isfinite(result).all()

    def test_moving_quantile_baseline_is_finite(self):
        y = np.linspace(0, 1, 100) + np.random.default_rng(0).normal(0, 0.01, 100)
        result = moving_quantile_baseline(y, window=11, quantile=0.2)
        assert len(result) == len(y)
        assert np.isfinite(result).all()

    def test_morphological_tophat_baseline_is_finite(self):
        y = np.linspace(0, 1, 100) + np.exp(-0.5 * ((np.arange(100) - 50) / 5) ** 2)
        result = morphological_tophat_baseline(y, window=11)
        assert len(result) == len(y)
        assert np.isfinite(result).all()

    def test_unknown_baseline_method_raises(self):
        with pytest.raises(ValueError, match="Unknown baseline method"):
            baseline_correction(np.array([1.0, 2.0, 3.0]), method="invalid")


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
# Additional Normalization Methods
# =============================================================================
class TestAdditionalNormalization:
    def test_robust_snv_median_centered(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
        result = robust_snv(y)
        np.testing.assert_allclose(np.median(result), 0.0, atol=1e-10)

    def test_max_normalize_range(self):
        y = np.array([-2.0, 1.0, 4.0])
        result = max_normalize(y)
        np.testing.assert_allclose(np.max(np.abs(result)), 1.0)

    def test_peak_normalize_global(self):
        y = np.array([1.0, 2.0, 4.0])
        result = peak_normalize(y)
        np.testing.assert_allclose(result[-1], 1.0)

    def test_peak_normalize_local(self):
        x = np.array([990.0, 1000.0, 1010.0, 1200.0])
        y = np.array([1.0, 5.0, 3.0, 20.0])
        result = peak_normalize(y, x=x, peak_wn=1000.0, window=12.0)
        np.testing.assert_allclose(result[1], 1.0)

    def test_mean_center(self):
        y = np.array([1.0, 2.0, 3.0])
        result = mean_center(y)
        np.testing.assert_allclose(result.mean(), 0.0)

    def test_pareto_normalize_mean_centered(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        result = pareto_normalize(y)
        np.testing.assert_allclose(result.mean(), 0.0)

    def test_pqn_normalize_reference(self):
        reference = np.array([1.0, 2.0, 4.0])
        y = reference * 2.0
        result = pqn_normalize(y, reference)
        np.testing.assert_allclose(result, reference)

    def test_msc_normalize_reference(self):
        reference = np.linspace(1.0, 10.0, 20)
        y = 5.0 + 2.0 * reference
        result = msc_normalize(y, reference)
        np.testing.assert_allclose(result, reference, atol=1e-10)

    def test_emsc_normalize_reference(self):
        x = np.linspace(400, 2200, 50)
        reference = np.sin(np.linspace(0, 4, 50)) + 2
        baseline = 5 + 0.01 * (x - x.mean())
        y = 3.0 * reference + baseline
        result = emsc_normalize(y, reference=reference, x=x, order=1)
        np.testing.assert_allclose(result, reference, atol=1e-8)

    @pytest.mark.parametrize("method", ["pqn", "msc", "emsc"])
    def test_reference_methods_require_reference(self, method):
        with pytest.raises(PreprocessingError, match="requires a reference"):
            normalize_spectrum(np.array([1.0, 2.0, 3.0]), method=method)


# =============================================================================
# normalize_spectrum dispatcher
# =============================================================================
class TestNormalizeSpectrum:
    @pytest.mark.parametrize(
        "method",
        [
            "snv",
            "robust_snv",
            "minmax",
            "l2",
            "area",
            "max",
            "peak",
            "mean_center",
            "pareto",
            "none",
        ],
    )
    def test_valid_methods(self, method):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = normalize_spectrum(y, method=method)
        assert len(result) == len(y)

    @pytest.mark.parametrize("method", ["pqn", "msc", "emsc"])
    def test_reference_methods_with_reference(self, method):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        reference = y.copy()
        result = normalize_spectrum(y, method=method, reference=reference)
        assert len(result) == len(y)
        assert np.isfinite(result).all()

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

    def test_als_baseline_pipeline(self, full_wavenumber_grid, wavenumber_grid):
        """Full pipeline can swap baseline module to ALS."""
        from tests.helpers import make_spectrum

        x = full_wavenumber_grid
        y = make_spectrum(x, baseline_slope=1.0)
        result = preprocess_single_spectrum(
            x,
            y,
            wavenumber_grid,
            do_trim=True,
            do_smooth=True,
            do_baseline=True,
            baseline_method="als",
            baseline_als_lam=1e5,
            baseline_als_p=0.01,
            baseline_als_niter=5,
            normalization="snv",
        )
        assert len(result) == len(wavenumber_grid)
        assert np.isfinite(result).all()


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


def test_baseline_before_smooth_changes_order_only():
    """Order switch must leave the default path untouched and give a close but
    different result when enabled (design guide §2: both orders should be tried)."""
    x = np.linspace(400, 2200, 1801)
    rng = np.random.default_rng(0)
    y = 5 + 0.001 * (x - 400) + 3 * np.exp(-((x - 1000) / 8) ** 2) + rng.normal(0, 0.05, len(x))
    grid = np.linspace(402, 2198, 935)
    default = preprocess_single_spectrum(x, y, grid, smooth_window=5, smooth_poly=3)
    explicit = preprocess_single_spectrum(x, y, grid, smooth_window=5, smooth_poly=3,
                                          baseline_before_smooth=False)
    swapped = preprocess_single_spectrum(x, y, grid, smooth_window=5, smooth_poly=3,
                                         baseline_before_smooth=True)
    np.testing.assert_allclose(default, explicit)
    assert not np.allclose(default, swapped)
    assert np.corrcoef(default, swapped)[0, 1] > 0.98
