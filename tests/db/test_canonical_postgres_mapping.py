from scripts.db.canonical_postgres_mapping import (
    ClinicalIdentity,
    RawInput,
    decide_mapping,
    normalize_label,
    run_metadata,
)


def test_normalize_label_preserves_canonical_h_d_spacing() -> None:
    assert normalize_label("H.D.", "01") == "H. D._1"


def test_run_metadata_marks_path_values_as_inferred() -> None:
    result = run_metadata("raw", "20260407_Bladder_1mW_0.05s_Ave 100")
    assert result.acquisition_date == "2026-04-07"
    assert result.laser_power_mw == 1.0
    assert result.integration_time_s == 0.05
    assert result.average_count == 100
    assert result.metadata_status == "inferred_from_path"


def test_unmatched_spectrum_is_kept_for_review() -> None:
    raw = RawInput("dataset", 1, "raw", "BLC 1_1.CSV", "batch", "replicate", None, "BLC", "1", 1, "a" * 64)
    result = decide_mapping(raw, {})
    assert result.mapping_status == "unmatched_spectrum"
    assert result.review_note == "no_clinical_registry_match"


def test_control_spectrum_is_not_mapped_to_a_patient() -> None:
    raw = RawInput("dataset", 2, "raw", "PS 1_1.CSV", "batch", "calibration_control", "PS", None, "1", 1, "b" * 64)
    identity = ClinicalIdentity(1, "BLC_1", "subject", "sample", "event")
    result = decide_mapping(raw, {identity.solum_label: identity})
    assert result.material_role == "qc"
    assert result.qc_role == "PS"
    assert result.identity is None
