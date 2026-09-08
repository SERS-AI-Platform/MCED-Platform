from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PATENT_SCRIPTS = Path(__file__).parents[2] / "scripts" / "patent"
sys.path.insert(0, str(PATENT_SCRIPTS))

from mapping_covariance_eigenspectrum_core import (  # noqa: E402
    corrected_between_covariance,
    effective_rank,
    ordered_eigenspectrum,
)


def test_corrected_between_covariance_removes_mean_noise_contribution() -> None:
    raw_mean_covariance = np.diag(np.array([7.0, 4.0]))
    within_covariance = np.diag(np.array([2.0, 1.0]))

    corrected = corrected_between_covariance(
        raw_mean_covariance,
        within_covariance,
        average_inverse_repeats=0.5,
    )

    np.testing.assert_allclose(np.diag(corrected), np.array([6.0, 3.5]))


def test_ordered_eigenspectrum_is_descending_and_normalized() -> None:
    eigenvalues, eigenvectors, explained, cumulative = ordered_eigenspectrum(
        np.diag(np.array([1.0, 4.0, 2.0])),
    )

    np.testing.assert_allclose(eigenvalues, np.array([4.0, 2.0, 1.0]))
    np.testing.assert_allclose(explained.sum(), 1.0)
    np.testing.assert_allclose(cumulative[-1], 1.0)
    np.testing.assert_allclose(eigenvectors[:, 0], np.array([0.0, 1.0, 0.0]))


def test_effective_rank_is_one_for_single_mode_covariance() -> None:
    assert effective_rank(np.array([10.0, 0.0, 0.0])) == 1.0
