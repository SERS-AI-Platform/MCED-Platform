import ast

from scripts.db.bridge_aecd_to_canonical_postgres import (
    AecdRow,
    build_plans,
    print_summary,
)
from scripts.db.canonical_postgres_mapping import ClinicalInput


def test_aecd_rows_use_exact_solum_label_mapping_and_preserve_roles() -> None:
    clinical = (
        ClinicalInput(
            "clinical_normalized_xlsx_v1",
            1,
            "registry.xlsx",
            "a" * 64,
            2,
            "BLC_1",
            "urine",
            "2026-01-01",
            {"patient_code": "P1"},
        ),
    )
    rows = (
        AecdRow(
            10,
            "aecd_v2",
            "medical",
            "medical",
            "Medical Raman",
            "liquid",
            "/root",
            "batch",
            "BLC 1_1.CSV",
            "raw",
            "raw",
            False,
            "medical:batch:BLC:1:1",
            "BLC",
            "1",
            1,
            None,
            "2026-01-01",
            "b" * 64,
            {},
        ),
        AecdRow(
            11,
            "aecd_v2",
            "medical",
            "medical",
            "Medical Raman",
            "liquid",
            "/root",
            "batch",
            "BLC 1_ave.CSV",
            "average",
            "average",
            True,
            "medical:batch:BLC:1:average",
            "BLC",
            "1",
            None,
            None,
            "2026-01-01",
            "c" * 64,
            {},
        ),
        AecdRow(
            12,
            "aecd_v2",
            "equipment_test",
            "equipment_nanoscope",
            "NanoScope",
            "liquid",
            "/root",
            "batch",
            "PS 1_1.CSV",
            "calibration_control",
            "control",
            False,
            "equipment:batch:PS:1:1",
            None,
            "1",
            1,
            "PS",
            "2026-01-01",
            "d" * 64,
            {},
        ),
    )
    _, plans = build_plans(clinical, rows)

    assert plans[0].mapping_status == "matched"
    assert plans[0].measurement is not None
    assert plans[1].measurement_id == plans[0].measurement_id
    assert plans[1].artifact is not None
    assert plans[1].artifact[3] == "processed"
    assert plans[2].qc_artifact is not None
    assert plans[2].mapping_status == "not_applicable"


def test_patient_summary_excludes_average_rows(capsys) -> None:
    clinical = (
        ClinicalInput(
            "clinical_normalized_xlsx_v1",
            1,
            "registry.xlsx",
            "a" * 64,
            2,
            "BLC_1",
            "urine",
            "2026-01-01",
            {},
        ),
    )
    rows = (
        AecdRow(
            10,
            "aecd_v2",
            "medical",
            "medical",
            "Medical Raman",
            "liquid",
            "/root",
            "batch",
            "BLC 1_1.CSV",
            "raw",
            "raw",
            False,
            "medical:batch:BLC:1:1",
            "BLC",
            "1",
            1,
            None,
            "2026-01-01",
            "b" * 64,
            {},
        ),
        AecdRow(
            11,
            "aecd_v2",
            "medical",
            "medical",
            "Medical Raman",
            "liquid",
            "/root",
            "batch",
            "BLC 1_ave.CSV",
            "average",
            "average",
            True,
            "medical:batch:BLC:1:average",
            "BLC",
            "1",
            None,
            None,
            "2026-01-01",
            "c" * 64,
            {},
        ),
    )
    _, plans = build_plans(clinical, rows)
    print_summary(clinical, rows, plans, True)
    result = ast.literal_eval(capsys.readouterr().out.strip())

    assert result["patient_measurements"] == 1
    assert result["matched_patient_measurements"] == 1
    assert result["unmatched_patient_measurements"] == 0
    assert result["average_mapping_status"] == {"matched": 1}
