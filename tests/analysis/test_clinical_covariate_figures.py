from __future__ import annotations

import csv
import subprocess
from pathlib import Path

import pytest
from PIL import Image

# Real-data regression test: reviewed covariate table and hypothesis statistics are not tracked in git.
# Run locally with `pytest -m slow`; CI runs `-m "not slow"`.
pytestmark = pytest.mark.slow

REPO = Path(__file__).resolve().parents[2]
PUBLICATION = REPO / "publications" / "전향검체" / "보라매병원"
SCRIPT = PUBLICATION / "src" / "generate_clinical_covariate_figures.py"
DATA = PUBLICATION / "tables" / "clinical_group_covariates.csv"
STATISTICS = PUBLICATION / "tables"


def test_generates_publication_ready_covariate_figures(tmp_path: Path) -> None:
    # Given: the reviewed three-group clinical covariate table.
    command = [
        "uv",
        "run",
        str(SCRIPT),
        "--input",
        str(DATA),
        "--output-dir",
        str(tmp_path),
        "--statistics-dir",
        str(STATISTICS),
    ]

    # When: the standalone figure generator is run.
    subprocess.run(command, cwd=REPO, check=True)

    # Then: both PNG and vector PDF outputs are publication sized.
    expected_stems = (
        "fig24_psa_distribution_by_group",
        "fig25_clinical_covariate_distributions",
    )
    for stem in expected_stems:
        assert (tmp_path / f"{stem}.pdf").stat().st_size > 10_000
        with Image.open(tmp_path / f"{stem}.png") as image:
            assert image.width >= 2_000
            assert image.height >= 1_200


def test_writes_summary_with_complete_group_counts(tmp_path: Path) -> None:
    # Given: the reviewed table containing 21, 43, and 48 subjects.
    command = [
        "uv",
        "run",
        str(SCRIPT),
        "--input",
        str(DATA),
        "--output-dir",
        str(tmp_path),
        "--statistics-dir",
        str(STATISTICS),
    ]

    # When: the figure generator writes its analytical summary.
    subprocess.run(command, cwd=REPO, check=True)
    summary = (tmp_path / "clinical_group_covariate_summary.csv").read_text(encoding="utf-8")

    # Then: all group counts and the two missing BMI values are explicit.
    assert "Control,21,21,21" in summary
    assert "Biopsy-negative,43,43,43" in summary
    assert "Cancer,48,46,48" in summary


def test_writes_figure_statistical_annotation_audit(tmp_path: Path) -> None:
    # Given: reviewed covariates and the prespecified hypothesis-test outputs.
    command = [
        "uv",
        "run",
        str(SCRIPT),
        "--input",
        str(DATA),
        "--output-dir",
        str(tmp_path),
        "--statistics-dir",
        str(STATISTICS),
    ]

    # When: the annotated figures are generated.
    subprocess.run(command, cwd=REPO, check=True)
    with (tmp_path / "clinical_covariate_figure_annotations.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        annotations = list(csv.DictReader(handle))

    # Then: each visible inferential role is traceable to a named method and p-value.
    keyed = {(row["panel"], row["role"]): row for row in annotations}
    assert keyed[("PSA", "primary_omnibus")]["method"] == "HC3 ANCOVA Wald F"
    assert float(keyed[("PSA", "primary_omnibus")]["p_value"]) < 1e-18
    assert keyed[("Age", "secondary_omnibus")]["method"] == "Welch ANOVA"
    assert float(keyed[("Age", "secondary_omnibus")]["p_adjusted"]) < 0.01
    assert keyed[("BMI", "secondary_omnibus")]["gate_status"] == "closed"
    assert keyed[("PSA", "rank_sensitivity")]["method"] == "Kruskal-Wallis"
    assert keyed[("PSA", "pairwise_biopsy_negative_vs_control")]["symbol"] == "****"
    assert keyed[("PSA", "pairwise_cancer_vs_control")]["symbol"] == "****"
    assert keyed[("PSA", "pairwise_biopsy_negative_vs_cancer")]["symbol"] == "**"
    assert keyed[("Age", "pairwise_control_vs_biopsy_negative")]["symbol"] == "**"
    assert keyed[("Age", "pairwise_biopsy_negative_vs_cancer")]["symbol"] == "*"
