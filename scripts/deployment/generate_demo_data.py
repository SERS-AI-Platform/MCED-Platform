#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0"]
# ///
# ─── How to run ───
# uv run scripts/deployment/generate_demo_data.py
"""Generate deterministic demo spectra for the clinical deployment UI."""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.deployment.demo_data_catalog import QC_PATIENT, USABILITY_PATIENTS, UsabilityPatient

OUT_DIR = Path(__file__).resolve().parent / "demo_data"
NORMAL_DIR = OUT_DIR / "normal"
BAD_DIR = OUT_DIR / "bad"
SEED = 20260511
RAW_DIR = PROJECT_ROOT / "data" / "raw_data"

PATIENTS = [
    {"patient_id": "DUMMY-001", "age": 54, "sex": "F", "bmi": 22.8, "notes": "normal demo patient"},
    {"patient_id": "DUMMY-002", "age": 61, "sex": "M", "bmi": 25.1, "notes": "normal demo patient"},
    {"patient_id": "DUMMY-003", "age": 47, "sex": "F", "bmi": 20.9, "notes": "normal demo patient"},
    {"patient_id": "DUMMY-004", "age": 68, "sex": "M", "bmi": 27.4, "notes": "normal demo patient"},
    {"patient_id": "DUMMY-005", "age": 59, "sex": "F", "bmi": 24.2, "notes": "normal demo patient"},
]


def gaussian(x: np.ndarray, center: float, width: float, amplitude: float) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)


def base_spectrum(x: np.ndarray, patient_index: int) -> np.ndarray:
    """Create a Raman-like synthetic spectrum with patient-level variation."""
    offset = (patient_index - 2) * 3.0
    baseline = 520 + 0.02 * (x - 400) + 28 * np.sin(x / 210)
    peaks = (
        gaussian(x, 620 + offset, 18, 165)
        + gaussian(x, 785 - offset * 0.4, 24, 260)
        + gaussian(x, 1004 + offset * 0.2, 20, 310)
        + gaussian(x, 1265 - offset * 0.3, 34, 210)
        + gaussian(x, 1445 + offset * 0.5, 30, 285)
        + gaussian(x, 1660 - offset * 0.2, 38, 175)
    )
    return baseline + peaks


def write_spectrum(path: Path, x: np.ndarray, y: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for wn, intensity in zip(x, y):
            writer.writerow([f"{wn:.6f}", f"{intensity:.6f}"])


def generate_normal_files() -> None:
    rng = np.random.default_rng(SEED)
    x = np.linspace(400, 2200, 935)

    for patient_index, patient in enumerate(PATIENTS):
        template = base_spectrum(x, patient_index)
        for replicate in range(1, 6):
            scale = 1.0 + rng.normal(0, 0.012)
            drift = rng.normal(0, 2.0) * np.sin(x / 170)
            noise = rng.normal(0, 4.5, size=x.shape)
            y = np.clip(template * scale + drift + noise, 1.0, None)
            filename = f"{patient['patient_id']}_{replicate}.csv"
            write_spectrum(NORMAL_DIR / filename, x, y)


def generate_bad_files() -> None:
    rng = np.random.default_rng(SEED + 99)
    x = np.linspace(400, 2200, 935)
    reference = base_spectrum(x, 0)

    write_spectrum(BAD_DIR / "LOWINT-001_1.csv", x, reference * 0.003)
    write_spectrum(BAD_DIR / "FLAT-001_1.csv", x, np.full_like(x, 500.0))
    write_spectrum(BAD_DIR / "NOISY-001_1.csv", x, rng.normal(700, 260, size=x.shape))

    short_x = np.linspace(400, 2200, 80)
    short_y = base_spectrum(short_x, 1) + rng.normal(0, 5.0, size=short_x.shape)
    write_spectrum(BAD_DIR / "SHORT-001_1.csv", short_x, short_y)

    range_x = np.linspace(2400, 3200, 935)
    range_y = base_spectrum(np.linspace(400, 2200, 935), 2) + rng.normal(0, 5.0, size=x.shape)
    write_spectrum(BAD_DIR / "RANGE-001_1.csv", range_x, range_y)


def write_patient_ids() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "patient_ids.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "age", "sex", "bmi", "notes"])
        writer.writeheader()
        writer.writerows(PATIENTS)


def _copy_patient_set(patient: UsabilityPatient, destination: Path, raw_dir: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    source_dir = raw_dir / patient.source_folder
    for replicate in range(1, 6):
        source = source_dir / f"{patient.source_prefix} {patient.source_sample_id}_{replicate}.CSV"
        shutil.copyfile(source, destination / f"{patient.patient_id}_{replicate}.csv")


def _write_qc_fail_set(patient: UsabilityPatient, destination: Path, raw_dir: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    source_dir = raw_dir / patient.source_folder
    for replicate in range(1, 6):
        source = source_dir / f"{patient.source_prefix} {patient.source_sample_id}_{replicate}.CSV"
        target = destination / f"{patient.patient_id}_{replicate}.csv"
        if replicate <= 2:
            shutil.copyfile(source, target)
            continue
        spectrum = np.loadtxt(source, delimiter=",")
        intensities = spectrum[:, 1].copy()
        plateau_start = max(0, min(len(intensities) - 5, int(np.argmax(intensities)) - 2))
        intensities[plateau_start : plateau_start + 5] = float(np.max(intensities))
        write_spectrum(target, spectrum[:, 0], intensities)


def build_usability_data(output_dir: Path, raw_dir: Path) -> None:
    usability_dir = output_dir / "usability"
    rows: list[dict[str, str]] = []
    for patient in USABILITY_PATIENTS:
        relative_path = (
            Path("usability") / "heldout_pass" / f"{patient.patient_id}_{patient.source_group}"
        )
        _copy_patient_set(patient, output_dir / relative_path, raw_dir)
        rows.append(_mapping_row(patient, "heldout_pass", relative_path, "5"))

    initial_relative = (
        Path("usability") / "qc_remeasurement" / QC_PATIENT.patient_id / "initial_fail"
    )
    remeasure_relative = (
        Path("usability") / "qc_remeasurement" / QC_PATIENT.patient_id / "remeasure_pass"
    )
    _write_qc_fail_set(QC_PATIENT, output_dir / initial_relative, raw_dir)
    _copy_patient_set(QC_PATIENT, output_dir / remeasure_relative, raw_dir)
    rows.append(_mapping_row(QC_PATIENT, "qc_initial_fail", initial_relative, "2"))
    rows.append(_mapping_row(QC_PATIENT, "qc_remeasure_pass", remeasure_relative, "5"))

    mapping_path = usability_dir / "patient_mapping.csv"
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with mapping_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mapping_row(
    patient: UsabilityPatient,
    scenario: str,
    relative_path: Path,
    expected_qc_passed: str,
) -> dict[str, str]:
    has_valid_result = int(expected_qc_passed) >= 3
    return {
        "patient_id": patient.patient_id,
        "age": str(patient.age),
        "sex": patient.sex,
        "bmi": str(patient.bmi),
        "scenario": scenario,
        "relative_path": relative_path.as_posix(),
        "source_split": "heldout_test_60_20_20_seed42",
        "source_group": patient.source_group,
        "source_sample_id": str(patient.source_sample_id),
        "expected_qc_passed": expected_qc_passed,
        "expected_ssi": f"{patient.expected_ssi:.2f}" if has_valid_result else "",
        "expected_ssi_band": patient.expected_ssi_band if has_valid_result else "",
        "expected_cancer_detected": (
            str(bool(patient.expected_cancer_type)).lower() if has_valid_result else ""
        ),
        "expected_cancer_type": patient.expected_cancer_type if has_valid_result else "",
        "verified_model": "usersnet/v1.0.0",
    }


def write_readme() -> None:
    text = """# SERS Clinical Demo Data

Synthetic upload data for UI, build, and QC testing. These files are not clinical
examples and must not be used for validation.

- `patient_ids.csv`: five demo patient IDs with age, sex, and BMI values.
- `normal/`: five replicate CSV spectra per demo patient.
- `bad/`: intentionally poor-quality spectra for QC and warning checks.
- `usability/heldout_pass/`: de-identified heldout-test sets covering all seven
  cancer types plus low and medium SSI examples.
- `usability/qc_remeasurement/`: one patient-level QC Fail set followed by a
  five-file remeasurement Pass set.
- `usability/patient_mapping.csv`: patient-entry values, expected QC/SSI/type,
  source split, and relative upload directory for every usability scenario.

Each spectrum CSV has no header and uses two numeric columns:

1. Raman shift / wavenumber
2. intensity

For QC testing, upload four normal files for one patient plus one file from
`bad/`, especially `LOWINT-001_1.csv` or `NOISY-001_1.csv`.

For the summative usability workflow, use only the patient-level directories
under `usability/` and enter the matching age, sex, and BMI from
`patient_mapping.csv`. The original hospital identifier and the remaining
clinical table are intentionally excluded. These retrospective fixtures are
for UI/QC demonstration only, not clinical validation. Cancer-vs-control
outputs may be hospital-confounded and must not be presented as cross-hospital
generalization evidence.
"""
    (OUT_DIR / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    generate_normal_files()
    generate_bad_files()
    write_patient_ids()
    build_usability_data(OUT_DIR, RAW_DIR)
    write_readme()
    print(f"Generated demo data in {OUT_DIR}")


if __name__ == "__main__":
    main()
