import pytest

from scripts.deployment.clinical_i18n import STRINGS


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
