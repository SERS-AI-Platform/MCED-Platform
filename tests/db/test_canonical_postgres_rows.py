from scripts.db.canonical_postgres_mapping import (
    ClinicalInput,
    RawInput,
    decide_mapping,
    run_metadata,
)
from scripts.db.canonical_postgres_rows import build_clinical_plan, build_raw_plan


def test_clinical_plan_uses_solum_label_as_sample_identity() -> None:
    record = ClinicalInput("clinical", 1, "registry.xlsx", "a" * 64, 2, "BLC_1", "urine", "2026-01-01", {"group": "BLC"})
    plan = build_clinical_plan(record, "SOLUM_RESEARCH")
    assert plan.identity.solum_label == "BLC_1"
    assert plan.sample[3] == "BLC_1"
    assert plan.subject[2] is None


def test_raw_plan_links_confirmed_measurement_to_sample_and_event() -> None:
    raw = RawInput("raw", 1, "raw", "BLC 1_1.CSV", "batch", "replicate", None, "BLC", "1", 1, "b" * 64)
    record = ClinicalInput("clinical", 1, "registry.xlsx", "a" * 64, 2, "BLC_1", "urine", None, {})
    clinical = build_clinical_plan(record, "SOLUM_RESEARCH")
    identity = clinical.identity
    decision = decide_mapping(raw, {identity.solum_label: identity})
    plan = build_raw_plan(raw, decision, run_metadata("raw", "batch"), "SOLUM_RESEARCH")
    assert plan.measurement is not None
    assert plan.material is not None
    assert plan.material[1] == identity.sample_id
    assert plan.match_candidate is not None
    assert plan.match_candidate[6] == "matched"
