"""Unit-level checks for the Boramae clinical covariate scripts on synthetic data.

The reviewed covariate table (``publications/.../tables/clinical_group_covariates.csv``)
is not tracked in git, so the data-regression tests in
``test_clinical_hypothesis_report.py`` / ``test_clinical_covariate_figures.py`` only run
locally (``slow``). These tests feed both scripts a deterministic synthetic table and
assert the *structure* of what they produce - files, columns, hypothesis ids, methods -
never the statistics themselves, which are meaningless on synthetic data.
"""

from __future__ import annotations

import csv
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "publications" / "전향검체" / "보라매병원" / "src"
HYPOTHESIS_SCRIPT = SRC / "generate_clinical_hypothesis_report.py"
FIGURE_SCRIPT = SRC / "generate_clinical_covariate_figures.py"

GROUPS = ("Control", "Biopsy-negative", "Cancer")
PER_GROUP = 20
# (age mean, bmi mean, psa median) - only the group ordering matters for the scripts.
GROUP_PROFILE = {
    "Control": (62.0, 24.0, 1.2),
    "Biopsy-negative": (66.0, 24.5, 8.0),
    "Cancer": (69.0, 24.2, 15.0),
}
# Blank cells at fixed positions so complete-case counts are known by construction.
MISSING_BMI = {"Control": (3,), "Biopsy-negative": (1, 7, 12, 18), "Cancer": (9,)}
MISSING_PSA = {"Control": (), "Biopsy-negative": (), "Cancer": (4, 15)}

pytestmark = pytest.mark.skipif(shutil.which("uv") is None, reason="uv is not installed")


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=REPO, check=True, capture_output=True, text=True)


@pytest.fixture(scope="module")
def synthetic_table(tmp_path_factory: pytest.TempPathFactory) -> Path:
    rng = np.random.default_rng(20260909)
    path = tmp_path_factory.mktemp("covariates") / "clinical_group_covariates.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["subject_id", "group", "age", "bmi", "psa_ng_ml"])
        subject = 1
        for group in GROUPS:
            age_mean, bmi_mean, psa_median = GROUP_PROFILE[group]
            for index in range(PER_GROUP):
                age = int(round(rng.normal(age_mean, 7.0)))
                bmi = "" if index in MISSING_BMI[group] else f"{rng.normal(bmi_mean, 3.0):.1f}"
                psa = (
                    ""
                    if index in MISSING_PSA[group]
                    else f"{rng.lognormal(np.log(psa_median), 0.5):.2f}"
                )
                writer.writerow([f"S{subject:03d}", group, age, bmi, psa])
                subject += 1
    return path


@pytest.fixture(scope="module")
def hypothesis_outputs(synthetic_table: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("hypothesis")
    _run(["uv", "run", str(HYPOTHESIS_SCRIPT), "--input", str(synthetic_table), "--output-dir", str(out)])
    return out


@pytest.fixture(scope="module")
def figure_outputs(
    synthetic_table: Path, hypothesis_outputs: Path, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    out = tmp_path_factory.mktemp("figures")
    _run(
        [
            "uv",
            "run",
            str(FIGURE_SCRIPT),
            "--input",
            str(synthetic_table),
            "--output-dir",
            str(out),
            "--statistics-dir",
            str(hypothesis_outputs),
        ]
    )
    return out


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_hypothesis_report_writes_every_prespecified_output(hypothesis_outputs: Path) -> None:
    expected = (
        "clinical_hypothesis_primary_ancova.csv",
        "clinical_hypothesis_pairwise_contrasts.csv",
        "clinical_hypothesis_secondary_tests.csv",
        "clinical_hypothesis_sensitivity_tests.csv",
        "clinical_hypothesis_assumption_checks.csv",
        "clinical_hypothesis_report.md",
    )
    assert all((hypothesis_outputs / name).stat().st_size > 100 for name in expected)


def test_hypothesis_report_primary_uses_complete_cases(hypothesis_outputs: Path) -> None:
    primary = _read(hypothesis_outputs / "clinical_hypothesis_primary_ancova.csv")
    complete = sum(
        PER_GROUP - len(set(MISSING_BMI[group]) | set(MISSING_PSA[group])) for group in GROUPS
    )

    assert [row["hypothesis_id"] for row in primary] == ["H1"]
    assert primary[0]["method"] == "log10-PSA ANCOVA with HC3 covariance"
    assert int(primary[0]["complete_case_n"]) == complete
    assert primary[0]["decision"] in {"reject_H0", "fail_to_reject_H0"}


def test_hypothesis_report_plans_three_holm_contrasts(hypothesis_outputs: Path) -> None:
    pairwise = _read(hypothesis_outputs / "clinical_hypothesis_pairwise_contrasts.csv")

    assert {row["comparison"] for row in pairwise} == {
        "Biopsy-negative vs Control",
        "Cancer vs Control",
        "Biopsy-negative vs Cancer",
    }
    assert {row["hypothesis_id"] for row in pairwise} == {"H1_pairwise"}


def test_hypothesis_report_covers_secondary_metrics(hypothesis_outputs: Path) -> None:
    secondary = _read(hypothesis_outputs / "clinical_hypothesis_secondary_tests.csv")

    assert {(row["hypothesis_id"], row["metric"]) for row in secondary} == {
        ("H2a", "Age"),
        ("H2b", "BMI"),
    }
    assert {row["level"] for row in secondary} == {"omnibus", "pairwise"}


def test_figures_are_written_at_publication_size(figure_outputs: Path) -> None:
    pytest.importorskip("PIL")
    from PIL import Image

    for stem in ("fig24_psa_distribution_by_group", "fig25_clinical_covariate_distributions"):
        assert (figure_outputs / f"{stem}.pdf").stat().st_size > 10_000
        with Image.open(figure_outputs / f"{stem}.png") as image:
            assert image.width >= 2_000
            assert image.height >= 1_200


def test_summary_counts_follow_the_missing_cells(figure_outputs: Path) -> None:
    summary = {row["group"]: row for row in _read(figure_outputs / "clinical_group_covariate_summary.csv")}

    assert list(summary) == list(GROUPS)
    for group in GROUPS:
        assert int(summary[group]["n_age"]) == PER_GROUP
        assert int(summary[group]["n_bmi"]) == PER_GROUP - len(MISSING_BMI[group])
        assert int(summary[group]["n_psa"]) == PER_GROUP - len(MISSING_PSA[group])


def test_annotation_audit_names_method_per_panel(figure_outputs: Path) -> None:
    keyed = {
        (row["panel"], row["role"]): row
        for row in _read(figure_outputs / "clinical_covariate_figure_annotations.csv")
    }

    assert keyed[("PSA", "primary_omnibus")]["method"] == "HC3 ANCOVA Wald F"
    assert keyed[("PSA", "rank_sensitivity")]["method"] == "Kruskal-Wallis"
    assert keyed[("Age", "secondary_omnibus")]["method"] == "Welch ANOVA"
    assert {panel for panel, _ in keyed} == {"PSA", "Age", "BMI"}
    for row in keyed.values():
        assert row["gate_status"]
        assert 0.0 <= float(row["p_value"]) <= 1.0
