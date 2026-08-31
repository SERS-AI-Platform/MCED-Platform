from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from scripts.deployment.generate_demo_data import build_usability_data
from src.sers.io import read_spectrum
from src.sers.qc.qc import detect_saturation

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_build_usability_data_creates_mapped_heldout_sets(tmp_path: Path) -> None:
    # Given
    output_dir = tmp_path / "demo_data"
    raw_dir = PROJECT_ROOT / "data" / "raw_data"

    # When
    build_usability_data(output_dir, raw_dir)

    # Then
    mapping_path = output_dir / "usability" / "patient_mapping.csv"
    with mapping_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))

    pass_rows = [row for row in rows if row["scenario"] == "heldout_pass"]
    assert len(pass_rows) == 9
    assert {row["source_group"] for row in pass_rows} >= {
        "PRO",
        "LUN",
        "CRC",
        "PAN",
        "OVA",
        "BRE",
        "BLC",
    }
    assert {row["expected_ssi_band"] for row in pass_rows} == {
        "low",
        "medium",
        "high",
    }
    for row in pass_rows:
        files = list((output_dir / row["relative_path"]).glob("*.csv"))
        assert len(files) == 5


def test_build_usability_data_creates_qc_remeasurement_pair(tmp_path: Path) -> None:
    # Given
    output_dir = tmp_path / "demo_data"
    raw_dir = PROJECT_ROOT / "data" / "raw_data"

    # When
    build_usability_data(output_dir, raw_dir)

    # Then
    qc_dir = output_dir / "usability" / "qc_remeasurement" / "UT-QC-001"
    initial_files = sorted((qc_dir / "initial_fail").glob("*.csv"))
    assert len(initial_files) == 5
    assert len(list((qc_dir / "remeasure_pass").glob("*.csv"))) == 5
    for path in initial_files[2:]:
        spectrum = np.loadtxt(path, delimiter=",", encoding="utf-8-sig")
        assert detect_saturation(spectrum[:, 1])
    source_x, _ = read_spectrum(raw_dir / "4. Lung cancer (300개)" / "LUN 261_3.CSV")
    generated_x, _ = read_spectrum(initial_files[2])
    assert len(generated_x) == len(source_x)

    mapping_path = output_dir / "usability" / "patient_mapping.csv"
    with mapping_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    initial_row = next(row for row in rows if row["scenario"] == "qc_initial_fail")
    assert initial_row["expected_qc_passed"] == "2"
    assert initial_row["expected_ssi"] == ""
    assert initial_row["expected_cancer_detected"] == ""
