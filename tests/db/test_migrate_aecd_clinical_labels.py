from __future__ import annotations

import subprocess
from pathlib import Path

from scripts.db import aecd_label_contract as contract


def test_label_migration_cli_exposes_safe_modes() -> None:
    # Given
    script = Path("scripts/db/migrate_aecd_clinical_labels.py")

    # When
    completed = subprocess.run(
        ["uv", "run", str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    # Then
    assert completed.returncode == 0
    assert "check-db" in completed.stdout
    assert "write" in completed.stdout
    assert "rollback" in completed.stdout


def test_resolve_labels_uses_versioned_rule_for_an_arbitrary_cohort() -> None:
    # Given
    source = contract.ClinicalSource(
        solum_label="RENAL_1",
        source_cancer_file="renal-source",
        cohort_group="renal surveillance",
        source_cancer_type="renal",
    )
    rule = contract.LabelRule(
        mapping_version="test-v9",
        source_cancer_file="renal-source",
        cohort_group="renal surveillance",
        source_cancer_type="renal",
        study_cancer_type="renal",
        case_status=contract.CaseStatus.DISEASE_CONTROL,
    )

    # When
    plan = contract.resolve_labels(((91, "RENAL 1"),), (source,), (rule,))

    # Then
    assert plan.mapping_version == "test-v9"
    assert plan.rows[0].case_status is contract.CaseStatus.DISEASE_CONTROL
