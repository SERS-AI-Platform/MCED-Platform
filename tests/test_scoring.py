"""Tests for user-facing scoring transforms."""

import pytest

from sers.scoring import (
    SSI_CUTOFF,
    SSI_MAX,
    compute_cti_scores,
    format_ssi,
    probability_to_cti,
    probability_to_ssi,
    ssi_risk_level,
    ssi_to_probability,
)


def test_probability_threshold_maps_to_ssi_cutoff():
    assert probability_to_ssi(0.6, threshold=0.6) == pytest.approx(SSI_CUTOFF)


def test_probability_to_ssi_clips_probabilities_to_supported_range():
    assert probability_to_ssi(-0.2, threshold=0.6) == 0.0
    assert probability_to_ssi(1.2, threshold=0.6) == SSI_MAX


def test_ssi_transform_is_invertible_around_threshold():
    threshold = 0.6

    for probability in [0.0, 0.3, 0.6, 0.8, 1.0]:
        ssi = probability_to_ssi(probability, threshold=threshold)

        assert ssi_to_probability(ssi, threshold=threshold) == pytest.approx(probability)


def test_invalid_threshold_falls_back_to_linear_scale():
    assert probability_to_ssi(0.25, threshold=0.0) == pytest.approx(2.5)
    assert ssi_to_probability(2.5, threshold=1.0) == pytest.approx(0.25)


def test_probability_to_cti_clips_and_formats_to_one_decimal():
    assert probability_to_cti(-1.0) == 0.0
    assert probability_to_cti(0.756) == 7.6
    assert probability_to_cti(2.0) == 10.0


def test_compute_cti_scores_converts_all_probabilities():
    assert compute_cti_scores({"CRC": 0.91, "LUN": 0.02}) == {"CRC": 9.1, "LUN": 0.2}


def test_ssi_risk_level_returns_expected_bands():
    assert ssi_risk_level(1.9)[0] == "LOW"
    assert ssi_risk_level(4.0)[0] == "MODERATE"
    assert ssi_risk_level(9.9)[0] == "VERY_HIGH"


def test_format_ssi_uses_one_decimal_place():
    assert format_ssi(4) == "4.0"
    assert format_ssi(4.04) == "4.0"
    assert format_ssi(4.06) == "4.1"
