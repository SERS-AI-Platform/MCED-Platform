from __future__ import annotations

import csv
import re
import sys
import warnings
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[2] / "publications" / "전향검체" / "보라매병원" / "src"
sys.path.insert(0, str(SRC))

import boramae_data  # noqa: E402
import prostate_comparison_plots  # noqa: E402
from prostate_comparison_model import OofResult, build_task  # noqa: E402
from prostate_comparison_plots import (  # noqa: E402
    plot_screening_performance,
    plot_three_group_performance,
)
from prostate_shift_alignment import match_peaks  # noqa: E402


def test_build_task_when_screening_binary_includes_biopsy_negative() -> None:
    # Given
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

    # When
    task_x, task_y, task_subjects = build_task(x, labels, "screening_binary")

    # Then
    assert task_x.shape == (6, 3)
    assert task_y.tolist() == [0, 0, 1, 0, 0, 1]
    assert task_subjects.tolist() == [0, 1, 2, 3, 4, 5]


def test_build_task_when_three_group_keeps_all_boramae_groups() -> None:
    # Given
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

    # When
    task_x, task_y, task_subjects = build_task(x, labels, "three_group")

    # Then
    assert task_x.shape == (6, 3)
    assert task_y.tolist() == [0, 1, 2, 0, 1, 2]
    assert task_subjects.tolist() == [0, 1, 2, 3, 4, 5]


def test_analysis_script_declares_transitive_project_dependencies() -> None:
    script = (SRC / "generate_prostate_clean_vs_prospective.py").read_text(encoding="utf-8")

    for dependency in (
        '"click>=8.1"',
        '"matplotlib>=3.7"',
        '"numpy>=1.24"',
        '"openpyxl>=3.1"',
        '"pandas>=2.0"',
        '"pyyaml>=6.0"',
        '"scikit-learn>=1.3"',
        '"scipy>=1.10"',
        '"tqdm>=4.65"',
    ):
        assert dependency in script


def test_boramae_data_uses_canonical_sers_package() -> None:
    assert boramae_data.read_spectrum.__module__ == "sers.io"
    assert boramae_data.baseline_correction.__module__ == "sers.signal"


def test_parse_group_when_biopsy_is_negative_uses_publication_label() -> None:
    # Given
    raw_group = "Elevated PSA, biopsy-negative (PSA↑/Bx−)"

    # When
    label = boramae_data.parse_group(raw_group)

    # Then
    assert label == "Biopsy-negative"


def test_publication_figure_titles_use_exact_biopsy_negative_label() -> None:
    assert prostate_comparison_plots.SCREENING_TITLE == (
        "Fig03. Screening: Control + Biopsy-negative vs prostate cancer"
    )
    assert prostate_comparison_plots.THREE_GROUP_TITLE == (
        "Fig06. 3-group: Control / Biopsy-negative / prostate cancer"
    )


def test_plot_performance_when_using_current_sklearn(tmp_path: Path) -> None:
    three_y = np.array([0, 1, 2, 0, 1, 2])
    three = OofResult(
        y_true=three_y,
        y_pred=three_y,
        probabilities=np.eye(3)[three_y],
        folds=np.ones(6, dtype=int),
        metrics={"balanced_accuracy": 1.0, "macro_f1": 1.0, "macro_ovr_roc_auc": 1.0},
        intervals={},
        confusion=np.eye(3, dtype=int) * 2,
    )
    binary_y = np.array([0, 1, 0, 1, 0, 1])
    binary = OofResult(
        y_true=binary_y,
        y_pred=binary_y,
        probabilities=np.column_stack([1 - binary_y, binary_y]),
        folds=np.ones(6, dtype=int),
        metrics={
            "balanced_accuracy": 1.0,
            "macro_f1": 1.0,
            "roc_auc": 1.0,
            "sensitivity": 1.0,
            "specificity": 1.0,
        },
        intervals={},
        confusion=np.eye(2, dtype=int) * 3,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        plot_screening_performance(binary, tmp_path)
        plot_three_group_performance(three, tmp_path)

    assert (tmp_path / "fig03_screening_binary_auc_confusion_matrix.png").exists()
    assert (tmp_path / "fig06_three_group_auc_confusion_matrix.png").exists()


def test_match_peaks_when_candidates_share_nearest_reference() -> None:
    reference = np.array([800.0, 1000.0, 1200.0])
    candidate = np.array([798.0, 1004.0, 1010.0, 1207.0])

    matches = match_peaks(reference, candidate, tolerance=12.0)

    assert matches == [(800.0, 798.0), (1000.0, 1004.0), (1200.0, 1207.0)]


def test_requested_comparison_outputs_exist() -> None:
    publication = SRC.parent
    expected = [
        publication / "figures" / "fig03_screening_binary_auc_confusion_matrix.png",
        publication / "figures" / "fig04a_screening_peak_sets_common_differential.png",
        publication / "figures" / "fig04b_three_group_peak_sets_common_differential.png",
        publication / "figures" / "fig06_three_group_auc_confusion_matrix.png",
        publication / "tables" / "screening_binary_metrics.csv",
        publication / "tables" / "three_group_metrics.csv",
        publication / "tables" / "fig04a_screening_common_and_differential_peaks.csv",
        publication / "tables" / "fig04b_three_group_common_and_differential_peaks.csv",
        publication / "tables" / "fig04a_screening_differential_peak_regions.csv",
        publication / "tables" / "fig04b_three_group_differential_peak_regions.csv",
    ]

    assert not [path for path in expected if not path.exists()]
    obsolete = [
        publication / "figures" / "fig04_group_peak_sets_common_differential.png",
        publication / "figures" / "fig07_legacy_vs_prospective_peak_alignment.png",
        publication / "tables" / "strict_binary_metrics.csv",
        publication / "tables" / "aligned_screening_binary_metrics.csv",
        publication / "tables" / "aligned_three_group_metrics.csv",
        publication / "tables" / "prostate_peak_alignment_summary.csv",
        publication / "tables" / "prostate_urea_calibration_shifts.csv",
        publication / "tables" / "fig04a_screening_feature_importance_top5.csv",
        publication / "tables" / "fig04b_three_group_feature_importance_top5.csv",
    ]
    assert not [path for path in obsolete if path.exists()]


def test_publication_outputs_do_not_expose_source_sample_codes() -> None:
    publication = SRC.parent
    pattern = re.compile(r"\b(?:BPRO|BNOR|PRO)\s+\d+\b")
    paths = [
        publication / "SUMMARY.md",
        publication / "PROSTATE_COMPARISON.md",
        *sorted((publication / "tables").glob("*.csv")),
    ]

    for path in paths:
        assert pattern.search(path.read_text(encoding="utf-8-sig")) is None, path


def test_publication_outputs_use_biopsy_negative_label() -> None:
    publication = SRC.parent
    obsolete_labels = ("PSA/Bx-", "PSA-Bx-", "Elevated PSA/Bx-")
    paths = [
        publication / "SUMMARY.md",
        publication / "PROSTATE_COMPARISON.md",
        publication / "FURTHER_STUDY_PLAN.md",
        *sorted((publication / "tables").glob("*.csv")),
    ]

    for path in paths:
        contents = path.read_text(encoding="utf-8-sig")
        assert not [label for label in obsolete_labels if label in contents], path


def test_combined_metrics_contains_one_row_per_task_metric() -> None:
    path = SRC.parent / "tables" / "prostate_classification_metrics.csv"
    with path.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    keys = [(row["task"], row["metric"]) for row in rows]

    assert len(keys) == len(set(keys))
    assert {row["task"] for row in rows} == {"screening_binary", "three_group"}


def test_peak_tables_match_screening_and_three_group_modes() -> None:
    table_dir = SRC.parent / "tables"
    with (table_dir / "fig04a_screening_common_and_differential_peaks.csv").open(
        encoding="utf-8-sig"
    ) as handle:
        screening = list(csv.DictReader(handle))
    with (table_dir / "fig04b_three_group_common_and_differential_peaks.csv").open(
        encoding="utf-8-sig"
    ) as handle:
        three_group = list(csv.DictReader(handle))
    with (table_dir / "fig04a_screening_differential_peak_regions.csv").open(
        encoding="utf-8-sig"
    ) as handle:
        screening_regions = list(csv.DictReader(handle))
    with (table_dir / "fig04b_three_group_differential_peak_regions.csv").open(
        encoding="utf-8-sig"
    ) as handle:
        three_group_regions = list(csv.DictReader(handle))

    assert {row["present_groups"] for row in screening} == {"Non-cancer;Cancer"}
    assert sum(row["category"] == "common" for row in screening) == 11
    assert sum(row["category"] == "differential" for row in screening) == 0
    assert sum(row["category"] == "common" for row in three_group) == 11
    assert sum(row["category"] == "differential" for row in three_group) == 1
    assert [row["rank"] for row in screening_regions] == ["1", "2", "3", "4", "5"]
    assert [row["rank"] for row in three_group_regions] == ["1", "2", "3", "4", "5"]
    assert all(float(row["intensity_range"]) > 0 for row in screening_regions)
    assert all(float(row["intensity_range"]) > 0 for row in three_group_regions)
