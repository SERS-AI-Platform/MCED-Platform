from pathlib import Path

import numpy as np
import pytest

from scripts.deployment import sers_predict
from scripts.deployment.clinical_decision import final_decision, ssi_risk_info
from scripts.deployment.clinical_i18n import STRINGS
from scripts.deployment.sers_predict import ProductionPredictor


@pytest.mark.parametrize("language", ["ko", "en"])
def test_three_risk_band_interpretations_exist_when_loading_clinical_copy(
    language: str,
) -> None:
    # Given: the localized copy used by the clinical result surfaces.
    strings = STRINGS[language]

    # When: the three SSI risk-band interpretation entries are selected.
    interpretations = [
        strings.get("risk_low_interpretation"),
        strings.get("risk_moderate_interpretation"),
        strings.get("risk_high_interpretation"),
    ]

    # Then: every band has its own non-empty explanation.
    assert all(isinstance(text, str) and text.strip() for text in interpretations)
    assert len(set(interpretations)) == 3


@pytest.mark.parametrize(
    ("ssi", "expected_level", "expected_rate", "expected_n"),
    [
        (0.99, "LOW", 1.8, 388),
        (1.0, "MODERATE", 46.0, 50),
        (3.99, "MODERATE", 46.0, 50),
        (4.0, "MODERATE", 46.0, 50),
        (4.0001, "HIGH", 98.2, 1190),
    ],
)
def test_ssi_band_metadata_uses_inclusive_action_boundaries(
    ssi: float,
    expected_level: str,
    expected_rate: float,
    expected_n: int,
) -> None:
    # Given: a patient-level mean SSI at or around the signal-level boundaries.
    # When: the SSI reference metadata is selected.
    signal_level = ssi_risk_info(ssi)

    # Then: it identifies the validation band used by the SSI action policy.
    assert signal_level["level"] == expected_level
    assert signal_level["observed_cancer_rate"] == expected_rate
    assert signal_level["n"] == expected_n
    assert "display_state" not in signal_level
    assert "decision_key" not in signal_level


@pytest.mark.parametrize(
    ("ssi_score", "expected_decision"),
    [
        (0.9999, "negative"),
        (1.0, "moderate"),
        (4.0, "moderate"),
        (4.0001, "positive"),
    ],
)
def test_final_decision_uses_mean_ssi_three_band_policy(
    ssi_score: float,
    expected_decision: str,
) -> None:
    # Given: a valid test produced by the mean-SSI three-band policy.
    prediction = {
        "cancer_detected": expected_decision == "positive",
        "ssi_score": ssi_score,
        "decision_policy": "mean_ssi_three_band_v1",
    }

    # When: the final display decision is selected.
    decision = final_decision(prediction, qc_valid=True)

    # Then: the exact 1.0 and 4.0 inclusivity rules select the action.
    assert decision == expected_decision


def test_final_decision_preserves_legacy_binary_result() -> None:
    prediction = {
        "cancer_detected": True,
        "ssi_score": 0.5,
        "decision_policy": "fixed_three_replicate_threshold_v1",
    }

    assert final_decision(prediction, qc_valid=True) == "positive"


class _SequentialBinaryModel:
    def __init__(self, probabilities: list[float]) -> None:
        self._probabilities = iter(probabilities)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        probability = next(self._probabilities)
        return np.array([[1.0 - probability, probability]])


class _FixedTypeModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return np.array([[0.8, 0.2]])


def _predict_with_replicate_probabilities(
    monkeypatch: pytest.MonkeyPatch,
    probabilities: list[float],
) -> dict:
    monkeypatch.setattr(
        sers_predict,
        "read_spectrum",
        lambda filepath: (np.arange(5, dtype=float), np.arange(5, dtype=float)),
    )
    predictor = ProductionPredictor.__new__(ProductionPredictor)
    predictor.prep = {"trim_region": [0.0, 4.0]}
    predictor.preprocess = lambda x, y, instrument: np.arange(5, dtype=float)
    predictor._qc_replicate_set = lambda spectra: [
        {
            **spectrum,
            "qc_pass": True,
            "qc_flags": [],
            "replicate_correlation": 0.99,
        }
        for spectrum in spectra
    ]
    predictor.threshold = 0.60
    predictor.s1_sers = _SequentialBinaryModel(probabilities)
    predictor.s2_sers = _FixedTypeModel()
    predictor.cancer_types = ["PRO", "LUN"]
    predictor.pds = None

    return predictor.predict_patient(
        [Path(f"replicate_{index}.csv") for index in range(len(probabilities))]
    )


@pytest.mark.parametrize(
    ("probabilities", "expected_level", "expected_count"),
    [
        ([0.10, 0.10, 0.10, 0.10, 0.10], "negative", 0),
        ([0.61, 0.61, 0.61, 0.10, 0.10], "moderate", 3),
        ([0.61, 0.61, 0.61, 0.10], "moderate", 3),
        ([1.00, 1.00, 0.59, 0.59, 0.59], "positive", 2),
        ([0.61, 0.61, 0.60], "positive", 2),
    ],
)
def test_patient_decision_uses_mean_ssi_action_policy(
    monkeypatch: pytest.MonkeyPatch,
    probabilities: list[float],
    expected_level: str,
    expected_count: int,
) -> None:
    # Given: QC-valid replicate probabilities around the SSI action boundaries.
    result = _predict_with_replicate_probabilities(monkeypatch, probabilities)

    # When: the patient-level decision is read from the production prediction output.
    patient_decision = result["patient_decision"]

    # Then: mean SSI selects the action; the cancer-signal count remains reference data.
    assert patient_decision["decision_level"] == expected_level
    assert patient_decision["cancer_detected"] is (expected_level == "positive")
    assert patient_decision["cancer_signal_spectra_count"] == expected_count
    assert patient_decision["qc_valid_spectra_count"] == len(probabilities)
    assert "majority_vote" not in patient_decision
    assert patient_decision["method"] == "mean_ssi_three_band"
    assert patient_decision["decision_policy"] == "mean_ssi_three_band_v1"
    assert (result["cancer_type_prediction"] is not None) is (expected_level == "positive")


@pytest.mark.parametrize("language", ["ko", "en"])
def test_clinical_copy_separates_final_decisions_and_ssi_signal_levels(
    language: str,
) -> None:
    # Given: the localized result-surface copy.
    strings = STRINGS[language]

    # When: the final decisions and SSI reference levels are selected.
    decisions = [
        strings["negative_decision"],
        strings["moderate_decision"],
        strings["positive_decision"],
    ]
    signal_levels = [
        strings["risk_low"],
        strings["risk_moderate"],
        strings["risk_high"],
    ]

    # Then: both vocabularies are complete and internally distinct.
    assert all(decision.strip() for decision in decisions)
    assert len(set(decisions)) == 3
    assert all(level.strip() for level in signal_levels)
    assert len(set(signal_levels)) == 3
    assert strings["ssi_calculation_text"]
    assert strings["model_display_name"] == "uSERS-Net Ver1"
    assert strings["replicate_above_internal_threshold"]
    assert strings["replicate_below_internal_threshold"]


@pytest.mark.parametrize("language", ["ko", "en"])
def test_result_page_centers_mean_ssi_validation_reference(language: str) -> None:
    # Given: the localized SSI reference copy and live result-page template.
    strings = STRINGS[language]
    template = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "deployment"
        / "templates"
        / "results.html"
    ).read_text(encoding="utf-8")

    # When: the mean-SSI reference structure is inspected.
    reference_keys = [
        "mean_ssi_reference_title",
        "risk_observed_cancer_rate",
        "risk_band_sample_count",
        "risk_band_note",
    ]

    # Then: the page shows validation context without repeating decision criteria.
    assert all(strings.get(key, "").strip() for key in reference_keys)
    assert 'class="mean-ssi-reference"' in template
    assert 'class="decision-explanation"' not in template
    assert all(f"s.{key}" in template for key in reference_keys)


def test_result_page_exposes_heading_and_collapsible_state_semantics() -> None:
    # Given: the result-page template and shared interaction script.
    project_root = Path(__file__).resolve().parents[1]
    template = (
        project_root / "scripts" / "deployment" / "templates" / "results.html"
    ).read_text(encoding="utf-8")
    script = (
        project_root / "scripts" / "deployment" / "static" / "js" / "clinical.js"
    ).read_text(encoding="utf-8")

    # When: the page heading and collapsible controls are inspected.
    # Then: the page has one top-level heading and both panels expose state.
    assert template.count("<h1") == 1
    assert template.count("<h2") >= 3
    assert "<h4" not in template
    assert template.count('aria-expanded="false"') == 2
    assert 'aria-controls="qcDetail"' in template
    assert 'aria-controls="replicateDetail"' in template
    assert 'id="replicateDetail"' in template
    assert "setAttribute('aria-expanded'" in script
    assert template.count('class="qc-table result-detail-table"') == 2
    assert template.count('data-label="{{ s.file }}"') >= 2
