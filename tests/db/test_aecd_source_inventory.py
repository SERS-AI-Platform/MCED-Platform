from __future__ import annotations

from pathlib import Path

from scripts.db.aecd_measurement_scan import discover_source_specs
from scripts.db.aecd_measurement_sources import SourceSpec, classify_measurement_file
from scripts.db.aecd_source_inventory import screening_eligibility


def test_mapping_clinical_record_carries_post_change_lineage(tmp_path: Path) -> None:
    root = tmp_path / "mapping"
    spec = SourceSpec(
        "mapping",
        "Thermo",
        "liquid",
        "mapping_clinical",
        root,
        "post_change_verified",
        "Methoxyamine hydrochloride 98%",
        "measurement.runs reagent_lot observation for 2026-08-10..14",
    )

    record = classify_measurement_file(
        root / "20260810_BNOR_mapping" / "BNOR 1" / "BNOR 1_0001.CSV",
        root,
        spec,
    )

    assert record.source_kind == "mapping_clinical"
    assert record.group_code == "BNOR"
    assert record.replicate == 1
    assert record.reagent_phase == "post_change_verified"
    assert record.reagent_name == "Methoxyamine hydrochloride 98%"


def test_mapping_reference_file_is_calibration_control(tmp_path: Path) -> None:
    root = tmp_path / "mapping" / "Thermo Reference"
    spec = SourceSpec(
        "mapping_reference",
        "Thermo",
        "liquid",
        "thermo_reference",
        root,
        "post_change_verified",
        "Methoxyamine hydrochloride 98%",
        "same post-change reagent evidence as mapping source",
    )

    record = classify_measurement_file(
        root / "20260805_20260804Cali_PS_1.CSV",
        root,
        spec,
    )

    assert record.artifact_role == "calibration_control"
    assert record.control_type == "PS"
    assert record.replicate == 1
    assert record.group_code is None


def test_mapping_ave_prefixed_point_is_a_replicate_not_an_average() -> None:
    root = Path("/data/mapping")
    spec = SourceSpec("mapping", "Thermo", "liquid", "mapping_clinical", root)

    record = classify_measurement_file(
        root / "20260811_BNOR_mapping" / "BNOR 14" / "BNOR 14_ave0001.CSV",
        root,
        spec,
    )

    assert record.is_averaged is False
    assert record.replicate == 1


def test_powder_reproducibility_condition_stays_in_measurement_key() -> None:
    root = Path("/data/powder")
    spec = SourceSpec("thermo", "Thermo", "powder", "powder_reproducibility", root)

    sigma_one = classify_measurement_file(
        root / "Sigma 1_BPRO 89_1.CSV",
        root,
        spec,
    )
    sigma_two = classify_measurement_file(
        root / "Sigma 2_BPRO 89_1.CSV",
        root,
        spec,
    )

    assert sigma_one.measurement_key != sigma_two.measurement_key


def test_discovered_specs_include_mapping_and_historical_phase(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "data" / "mapping").mkdir(parents=True)
    (repo / "data" / "raw_data").mkdir(parents=True)

    specs = discover_source_specs(repo)
    mapping = next(spec for spec in specs if spec.source_kind == "mapping_clinical")
    thermo = next(spec for spec in specs if spec.source_kind == "thermo")

    assert mapping.reagent_phase == "post_change_verified"
    assert thermo.reagent_phase == "pre_change_unverified"


def test_inventory_policy_excludes_derived_and_reference_artifacts(tmp_path: Path) -> None:
    clinical_root = tmp_path / "mapping"
    clinical_spec = SourceSpec(
        "mapping",
        "Thermo",
        "liquid",
        "mapping_clinical",
        clinical_root,
        "post_change_verified",
        "Methoxyamine hydrochloride 98%",
        "database observation",
    )
    clinical = classify_measurement_file(
        clinical_root / "20260810_BNOR_mapping" / "BNOR 1" / "BNOR 1_0001.CSV",
        clinical_root,
        clinical_spec,
    )
    average = classify_measurement_file(
        clinical_root / "20260810_BNOR_mapping" / "BNOR 1" / "BNOR 1_ave.CSV",
        clinical_root,
        clinical_spec,
    )

    assert screening_eligibility(clinical) == "include_in_candidate_pool"
    assert screening_eligibility(average) == "exclude_derived_average"
