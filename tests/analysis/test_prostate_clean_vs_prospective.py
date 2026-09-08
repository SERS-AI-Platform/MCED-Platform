from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import numpy as np

PUBLICATION = (
    Path(__file__).resolve().parents[2]
    / "publications"
    / "전향검체"
    / "보라매병원"
    / "clean_vs_prospective"
)
SRC = PUBLICATION.parent / "src"
sys.path.insert(0, str(SRC))

from prostate_comparison_model import build_task  # noqa: E402


def test_strict_binary_excludes_biopsy_negative() -> None:
    x = np.arange(18, dtype=float).reshape(6, 3)
    labels = np.array(
        [
            "Control",
            "Biopsy-negative",
            "Prostate cancer",
            "Control",
            "Biopsy-negative",
            "Prostate cancer",
        ]
    )

    task_x, task_y, indices = build_task(x, labels, "strict_binary")

    assert task_x.shape == (4, 3)
    assert task_y.tolist() == [0, 1, 0, 1]
    assert indices.tolist() == [0, 2, 3, 5]


def test_requested_outputs_exist() -> None:
    expected = [
        PUBLICATION / "figures" / "fig07_legacy_vs_prospective_peak_alignment.png",
        PUBLICATION / "tables" / "strict_binary_metrics.csv",
        PUBLICATION / "tables" / "aligned_screening_binary_metrics.csv",
        PUBLICATION / "tables" / "aligned_three_group_metrics.csv",
        PUBLICATION / "tables" / "prostate_peak_alignment_summary.csv",
        PUBLICATION / "tables" / "prostate_urea_calibration_shifts.csv",
    ]

    assert not [path for path in expected if not path.exists()]


def test_combined_metrics_has_eight_unique_tasks() -> None:
    path = PUBLICATION / "tables" / "prostate_classification_metrics.csv"
    with path.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    keys = [(row["task"], row["metric"]) for row in rows]

    assert len(rows) == 50
    assert len(keys) == len(set(keys))
    assert len({row["task"] for row in rows}) == 8


def test_calibration_shift_counts_match_replicates() -> None:
    path = PUBLICATION / "tables" / "prostate_urea_calibration_shifts.csv"
    with path.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    assert sum(row["cohort"] == "Clean PRO" for row in rows) == 455
    assert sum(row["cohort"] == "Prospective cancer" for row in rows) == 205


def test_publication_outputs_are_anonymized_and_bom_encoded() -> None:
    pattern = re.compile(r"\b(?:BPRO|BNOR|PRO)\s+\d+\b")
    paths = [
        PUBLICATION / "SUMMARY.md",
        PUBLICATION / "PROSTATE_COMPARISON.md",
        *sorted((PUBLICATION / "tables").glob("*.csv")),
    ]

    for path in paths:
        assert pattern.search(path.read_text(encoding="utf-8-sig")) is None, path
    assert all(
        path.read_bytes().startswith(b"\xef\xbb\xbf") for path in paths if path.suffix == ".csv"
    )


def test_report_contains_alignment_and_hospital_confound_warning() -> None:
    report = (PUBLICATION / "PROSTATE_COMPARISON.md").read_text(encoding="utf-8")

    assert "평균 전처리 스펙트럼 Pearson r: 0.826" in report
    assert "Cohort-source ROC-AUC: 0.999; urea alignment 후 0.998" in report
    assert "cross-hospital 일반화 근거로 사용하면 안 된다" in report
