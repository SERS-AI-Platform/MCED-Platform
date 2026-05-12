#!/usr/bin/env python3
"""Generate deterministic demo spectra for the clinical deployment UI."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


OUT_DIR = Path(__file__).resolve().parent / "demo_data"
NORMAL_DIR = OUT_DIR / "normal"
BAD_DIR = OUT_DIR / "bad"
SEED = 20260511

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
    with path.open("w", newline="") as f:
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
    with (OUT_DIR / "patient_ids.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "age", "sex", "bmi", "notes"])
        writer.writeheader()
        writer.writerows(PATIENTS)


def write_readme() -> None:
    text = """# SERS Clinical Demo Data

Synthetic upload data for UI, build, and QC testing. These files are not clinical
examples and must not be used for validation.

- `patient_ids.csv`: five demo patient IDs with age, sex, and BMI values.
- `normal/`: five replicate CSV spectra per demo patient.
- `bad/`: intentionally poor-quality spectra for QC and warning checks.

Each spectrum CSV has no header and uses two numeric columns:

1. Raman shift / wavenumber
2. intensity

For QC testing, upload four normal files for one patient plus one file from
`bad/`, especially `LOWINT-001_1.csv` or `NOISY-001_1.csv`.
"""
    (OUT_DIR / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    generate_normal_files()
    generate_bad_files()
    write_patient_ids()
    write_readme()
    print(f"Generated demo data in {OUT_DIR}")


if __name__ == "__main__":
    main()
