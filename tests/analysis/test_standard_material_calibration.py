from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts/analysis/aecd_api_model_mean_spectrum_clinical_performance.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "aecd_api_model_mean_spectrum_clinical_performance",
        SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_standard_material_correction_maps_observed_peak_to_reference() -> None:
    module = _load_script()
    observed = np.asarray([621.4248, 1001.3248], dtype=np.float64)

    corrected = module.apply_standard_material_axis_correction(observed, 0.1248)

    np.testing.assert_allclose(corrected, [621.3, 1001.2], atol=1e-6)


def test_group_alignment_uses_date_instrument_standard_calibration() -> None:
    module = _load_script()
    calibration = module.StandardMaterialCalibration(
        calibration_date="2026-08-10",
        instrument_name="SERS-01",
        standard_material="Polystyrene (PS)",
        tolerance_cm1=2.0,
        reference_peaks_cm1=(621.3,),
        observed_peaks_cm1=(621.5,),
        global_shift_cm1=0.2,
        alignment_score=0.99,
        replicate_shift_std_cm1=0.01,
    )
    item = {
        "subject_key": "subject:1",
        "cohort_group": "prostate",
        "measurement_id": 1,
        "replicate_number": 1,
        "measured_at": "2026-08-10T00:00:00Z",
        "instrument_name": "SERS-01",
        "wavenumber": [400.2, 401.2, 402.2],
        "intensities": [10.0, 20.0, 30.0],
    }
    subject_keys, labels, matrices, metadata = module.group_and_align_subjects(
        [item],
        np.asarray([400.0, 401.0, 402.0]),
        {("2026-08-10", "SERS-01"): calibration},
    )
    assert subject_keys == ["subject:1"]
    assert labels == ["prostate"]
    np.testing.assert_allclose(matrices[0], [[10.0, 20.0, 30.0]], atol=1e-6)
    assert metadata["calibration_shift_values_cm1"] == [-0.2]


def test_optimal_threshold_maximizes_balanced_accuracy() -> None:
    module = _load_script()
    y_true = np.asarray([0, 0, 0, 1, 1, 1], dtype=int)
    probability = np.asarray([0.10, 0.20, 0.45, 0.55, 0.70, 0.80])

    threshold, score = module.select_optimal_threshold(y_true, probability)

    assert threshold == 0.50
    assert score == 1.0
