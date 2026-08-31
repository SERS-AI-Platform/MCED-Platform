from __future__ import annotations

import csv
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PUBLICATION = REPO / "publications" / "전향검체" / "보라매병원"
SCRIPT = PUBLICATION / "src" / "generate_clinical_hypothesis_report.py"
DATA = PUBLICATION / "tables" / "clinical_group_covariates.csv"


def test_generates_prespecified_clinical_hypothesis_report(tmp_path: Path) -> None:
    # Given: the reviewed three-group covariate table.
    command = [
        "uv",
        "run",
        str(SCRIPT),
        "--input",
        str(DATA),
        "--output-dir",
        str(tmp_path),
    ]

    # When: the prespecified statistical workflow is run.
    subprocess.run(command, cwd=REPO, check=True)

    # Then: primary, secondary, sensitivity, assumption, and narrative outputs are written.
    expected = (
        "clinical_hypothesis_primary_ancova.csv",
        "clinical_hypothesis_pairwise_contrasts.csv",
        "clinical_hypothesis_secondary_tests.csv",
        "clinical_hypothesis_sensitivity_tests.csv",
        "clinical_hypothesis_assumption_checks.csv",
        "clinical_hypothesis_report.md",
    )
    assert all((tmp_path / name).stat().st_size > 100 for name in expected)

    with (tmp_path / expected[0]).open(encoding="utf-8", newline="") as handle:
        primary = list(csv.DictReader(handle))
    with (tmp_path / expected[1]).open(encoding="utf-8", newline="") as handle:
        pairwise = list(csv.DictReader(handle))
    with (tmp_path / expected[4]).open(encoding="utf-8", newline="") as handle:
        assumptions = list(csv.DictReader(handle))

    assert primary[0]["hypothesis_id"] == "H1"
    assert primary[0]["method"] == "log10-PSA ANCOVA with HC3 covariance"
    assert primary[0]["complete_case_n"] == "110"
    assert len(pairwise) == 3
    assert all(row["multiplicity"] == "Holm across three planned contrasts" for row in pairwise)
    assert {row["comparison"] for row in pairwise} == {
        "Biopsy-negative vs Control",
        "Cancer vs Control",
        "Biopsy-negative vs Cancer",
    }
    assert {row["check"] for row in assumptions} >= {
        "common_slopes_joint",
        "positive_psa",
        "age_overlap",
        "bmi_overlap",
    }
