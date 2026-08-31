from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[2] / "publications" / "전향검체" / "보라매병원" / "src"
sys.path.insert(0, str(SRC))

from powder_comparison.data import PreprocessingStageMatrices
from powder_comparison.models import run_paired_crossfit
from powder_comparison.signal_noise_ablation import build_feature_masks, run_peak_ablation
from powder_comparison.signal_noise_diagnostics import summarize_effect_retention
from powder_comparison.signal_noise_lot import load_lot_replicates, variance_components
from powder_comparison.signal_noise_peak import evaluate_peak_repeatability


def _gaussian(grid: np.ndarray, center: float, width: float = 5.0) -> np.ndarray:
    return np.exp(-0.5 * np.square((grid - center) / width))


def _stages(values: np.ndarray) -> PreprocessingStageMatrices:
    return PreprocessingStageMatrices(
        raw=values + 20.0,
        smoothed=values + 10.0,
        baseline_corrected=values,
        snv=values,
    )


def test_peak_repeatability_separates_stable_signal_from_extra_unstable_peaks() -> None:
    # Given: both modalities share a stable peak, while powder has replicate-specific peaks.
    grid = np.linspace(400.0, 800.0, 201)
    rng = np.random.default_rng(17)
    legacy = np.empty((6, 5, len(grid)), dtype=float)
    powder = np.empty_like(legacy)
    for subject in range(6):
        for replicate in range(5):
            shared = 4.0 * _gaussian(grid, 520.0)
            legacy[subject, replicate] = shared + rng.normal(0.0, 0.03, len(grid))
            extra_center = 610.0 + 24.0 * replicate
            powder[subject, replicate] = (
                shared
                + 2.0 * _gaussian(grid, extra_center, width=3.0)
                + rng.normal(0.0, 0.08, len(grid))
            )

    # When: peak detection is performed at replicate and consensus levels.
    result = evaluate_peak_repeatability(
        grid,
        _stages(legacy),
        _stages(powder),
        np.arange(1, 7),
    )

    # Then: powder exposes more peaks but a smaller fraction survives the 4-of-5 rule.
    legacy_rows = [
        row for row in result.stage_rows if row.modality == "legacy_liquid" and row.stage == "snv"
    ]
    powder_rows = [
        row for row in result.stage_rows if row.modality == "powder" and row.stage == "snv"
    ]
    assert np.mean([row.median_peak_count for row in powder_rows]) > np.mean(
        [row.median_peak_count for row in legacy_rows]
    )
    assert np.mean([row.consensus_fraction for row in powder_rows]) < np.mean(
        [row.consensus_fraction for row in legacy_rows]
    )
    shared_index = int(np.argmin(np.abs(grid - 520.0)))
    assert np.all(result.legacy_presence[:, shared_index])
    assert np.all(result.powder_presence[:, shared_index])


def test_variance_components_recover_patient_lot_and_residual_sources() -> None:
    # Given: a balanced crossed design with known patient, lot, and residual variation.
    rng = np.random.default_rng(23)
    patient = np.arange(5, dtype=float)[:, None, None, None] * 4.0
    lot = np.arange(5, dtype=float)[None, :, None, None] * 2.0
    residual = rng.normal(0.0, 0.5, (5, 5, 6, 3))
    values = patient + lot + residual

    # When: method-of-moments crossed variance components are estimated.
    result = variance_components(values)

    # Then: patient variation dominates lot, which dominates spot residual variation.
    assert np.all(result.patient > result.lot)
    assert np.all(result.lot > result.residual)
    assert np.allclose(
        result.patient_fraction + result.lot_fraction + result.residual_fraction, 1.0
    )


def test_feature_masks_distinguish_common_and_powder_only_consensus_regions() -> None:
    # Given: training subjects with one shared stable band and one powder-only stable band.
    legacy_presence = np.zeros((10, 12), dtype=bool)
    powder_presence = np.zeros_like(legacy_presence)
    legacy_presence[:8, 2:4] = True
    powder_presence[:9, 2:4] = True
    powder_presence[:7, 8:10] = True

    # When: label-free feature masks are learned from the training subjects only.
    masks = build_feature_masks(legacy_presence, powder_presence, minimum_subject_rate=0.30)

    # Then: shared and powder-only regions remain disjoint and exclusion is complementary.
    assert np.array_equal(np.flatnonzero(masks.common_stable), np.array([2, 3]))
    assert np.array_equal(np.flatnonzero(masks.powder_only), np.array([8, 9]))
    assert not np.any(masks.common_stable & masks.powder_only)
    assert np.array_equal(masks.powder_only_excluded, ~masks.powder_only)


def test_variance_components_reject_unbalanced_input() -> None:
    # Given: a tensor without the required patient, lot, replicate, and feature axes.
    values = np.zeros((5, 5, 3), dtype=float)

    # When/Then: the boundary rejects the invalid design explicitly.
    with pytest.raises(ValueError, match="four-dimensional"):
        variance_components(values)


def test_actual_powder_lot_inventory_is_balanced() -> None:
    # Given: the reviewed five-lot powder reproducibility experiment.
    repo = Path(__file__).resolve().parents[2]
    grid = np.load(repo / "artifacts" / "usersnet" / "v1.0.0" / "common_grid.npy")

    # When: raw replicate files are parsed without using _ave files.
    result = load_lot_replicates(
        repo / "data" / "20260715_Powder_Reproducibility test",
        grid,
    )

    # Then: five patients, five lots, and five independent replicates are retained.
    assert result.spectra.shape == (5, 5, 5, len(grid))
    assert np.array_equal(result.patient_ids, np.array([89, 90, 91, 92, 93]))
    assert np.array_equal(result.lot_ids, np.array([1, 2, 3, 4, 5]))


def test_peak_ablation_uses_shared_outer_folds_and_training_only_masks() -> None:
    # Given: a binary cohort with two common stable features and a powder-only region.
    rng = np.random.default_rng(31)
    labels = np.array(["Control"] * 15 + ["Cancer"] * 15)
    spectra = rng.normal(0.0, 0.3, (30, 12))
    spectra[labels == "Cancer", 2:4] += 1.5
    legacy_presence = np.zeros((30, 12), dtype=bool)
    powder_presence = np.zeros_like(legacy_presence)
    legacy_presence[:, 2:4] = True
    powder_presence[:, 2:4] = True
    powder_presence[:24, 8:10] = True

    # When: each ablation is evaluated with masks recomputed inside each outer fold.
    results = run_peak_ablation(
        spectra,
        labels,
        legacy_presence,
        powder_presence,
        task="screening_binary",
        outer_splits=3,
        inner_splits=2,
        c_values=(0.1, 1.0),
    )

    # Then: all feature sets share held-out folds and produce normalized probabilities.
    assert {row.feature_set for row in results} == {
        "full_spectrum",
        "common_stable",
        "powder_only",
        "powder_only_excluded",
    }
    reference_folds = results[0].folds
    assert all(np.array_equal(row.folds, reference_folds) for row in results)
    assert all(np.allclose(row.probabilities.sum(axis=1), 1.0) for row in results)
    common = next(row for row in results if row.feature_set == "common_stable")
    assert np.all(common.feature_counts == 2)
    assert common.metrics.roc_auc > 0.95


def test_full_spectrum_ablation_reproduces_powder_native_lr() -> None:
    # Given: a small cohort evaluated with the publication model settings.
    rng = np.random.default_rng(37)
    labels = np.array(["Control"] * 15 + ["Cancer"] * 15)
    spectra = rng.normal(0.0, 1.0, (30, 24))
    spectra[labels == "Cancer", :4] += 0.7
    presence = np.ones_like(spectra, dtype=bool)
    settings = {
        "task": "screening_binary",
        "outer_splits": 3,
        "inner_splits": 2,
        "seed": 42,
        "c_values": (0.001, 0.1, 10.0),
    }

    # When: the ordinary powder-native analysis and full-spectrum ablation are run.
    reference = run_paired_crossfit(spectra, spectra, labels, **settings)
    ablations = run_peak_ablation(spectra, labels, presence, presence, **settings)
    full = next(row for row in ablations if row.feature_set == "full_spectrum")

    # Then: the full-spectrum arm is the same estimator, not a nearby re-analysis.
    assert np.array_equal(full.folds, reference.folds)
    assert np.allclose(full.probabilities, reference.native_probabilities)


def test_effect_retention_reports_direction_and_rank_preservation() -> None:
    # Given: powder retains the ordering of liquid effects but reverses one relevant direction.
    legacy = np.array([0.8, 0.5, -0.4, 0.02])
    powder = np.array([0.6, 0.3, 0.2, -0.01])

    # When: only liquid effects above the relevance floor are summarized.
    result = summarize_effect_retention(legacy, powder, minimum_abs_legacy_effect=0.10)

    # Then: two of three directions agree and the effect attenuation is explicit.
    assert result.relevant_feature_count == 3
    assert result.sign_agreement == pytest.approx(2 / 3)
    assert result.median_absolute_retention == pytest.approx(0.60)
    assert -1.0 <= result.spearman_rho <= 1.0
