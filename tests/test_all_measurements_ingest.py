from __future__ import annotations

from pathlib import Path

import pytest

from scripts.db.aecd_measurement_sources import (
    ClassificationError,
    SourceSpec,
    classify_measurement_file,
)
from scripts.db.upload_all_measurements_postgres import (
    measurement_source_path,
)


def test_medical_background_and_average_share_measurement_key() -> None:
    root = Path("/data/medical/1. Prostate cancer (100개)")
    spec = SourceSpec("medical", "Medical Raman", "liquid", "medical")

    raw = classify_measurement_file(root / "PRO 7_1.txt", root, spec)
    background = classify_measurement_file(
        root / "Background" / "PRO 7_1.txt", root, spec
    )
    average = classify_measurement_file(
        root / "Background" / "PRO 7_ave.txt", root, spec
    )

    assert raw.artifact_role == "raw"
    assert background.artifact_role == "background"
    assert average.artifact_role == "average"
    assert raw.measurement_key == background.measurement_key
    assert average.measurement_key.endswith(":average")
    assert average.is_averaged is True


def test_equipment_filename_keeps_sample_and_replicate() -> None:
    root = Path("/data/equipment/handheld")
    spec = SourceSpec("equipment_test", "Handheld", "liquid", "equipment_test")

    record = classify_measurement_file(
        root / "1. NOR" / "NOR 100_1(1)_Sample_26_02_09_14_47_44_219.csv",
        root,
        spec,
    )

    assert record.group_code == "NOR"
    assert record.sample_id == "100"
    assert record.replicate == 1
    assert record.artifact_role == "equipment_test"


def test_non_spectrum_machine_metadata_is_rejected() -> None:
    root = Path("/data/equipment/medical_raw")
    spec = SourceSpec("equipment_test", "Medical Raman", "liquid", "equipment_test")

    with pytest.raises(ClassificationError):
        classify_measurement_file(root / "1. NOR" / "MultiData.txt", root, spec)


def test_onedrive_condition_suffix_is_preserved_as_variant() -> None:
    root = Path("/data/OneDrive_2026-07-21 (2)")
    spec = SourceSpec("thermo", "Thermo", "liquid", "thermo_onedrive")

    record = classify_measurement_file(
        root / "10-3. Y-Pancreatic cancer (27개)" / "YPAN 1 NF_3.CSV",
        root,
        spec,
    )

    assert record.group_code == "YPAN"
    assert record.sample_id == "1"
    assert record.replicate == 3
    assert record.variant == "non_fasting"


def test_source_path_keeps_measurements_from_different_roots_distinct() -> None:
    first_root = Path("/data/thermo/liquid")
    second_root = Path("/data/thermo/powder")
    first_spec = SourceSpec("thermo", "Thermo", "liquid", "thermo_liquid")
    second_spec = SourceSpec("thermo", "Thermo", "powder", "boramae_powder")

    first = classify_measurement_file(first_root / "1. NOR" / "NOR 1_1.txt", first_root, first_spec)
    second = classify_measurement_file(second_root / "1. NOR" / "NOR 1_1.txt", second_root, second_spec)

    assert measurement_source_path(first) != measurement_source_path(second)
