"""Unit checks for the master_clinical risk experiment helpers."""

import numpy as np
import pandas as pd
from scripts.analysis.master_clinical_risk_experiment import (
    build_cohorts,
    compute_binary_metrics,
    normalize_group,
    usable_marker_columns,
)


def test_normalize_group_aliases():
    assert normalize_group("CPAN") == "PAN"
    assert normalize_group("PAN") == "PAN"
    assert normalize_group("COL") == "CRC"
    assert normalize_group("CRC") == "CRC"
    assert normalize_group("BLC") == "BLA"
    assert normalize_group("H.D.") == "H.D."


def test_build_cohorts_keeps_review_flags_as_sensitivity_strata():
    df = pd.DataFrame(
        {
            "cohort_flag_definite_exclude": [False, True, False, False],
            "cohort_flag_review_periop": [False, False, True, False],
            "cohort_flag_review_undefined": [False, False, False, True],
        }
    )

    cohorts = build_cohorts(df)

    assert len(cohorts["full_1628"]) == 4
    assert len(cohorts["review_included"]) == 4
    assert len(cohorts["clean"]) == 3
    assert len(cohorts["strict_screening"]) == 1


def test_usable_marker_columns_require_both_classes():
    df = pd.DataFrame(
        {
            "good_marker": [1.0, 2.0, 3.0, 4.0],
            "positive_only_marker": [1.0, 2.0, np.nan, np.nan],
        }
    )
    y = np.array([1, 1, 0, 0])

    usable, reason = usable_marker_columns(
        df,
        y,
        ("good_marker", "positive_only_marker", "missing_marker"),
        min_per_class=2,
    )

    assert usable == ("good_marker",)
    assert "positive_only_marker:insufficient_nonmissing" in reason
    assert "missing_marker:missing_column" in reason


def test_compute_binary_metrics_reports_fixed_sensitivity_specificity():
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.05, 0.20, 0.30, 0.70, 0.80, 0.95])

    metrics = compute_binary_metrics(y, scores)

    assert metrics["auroc"] == 1.0
    assert metrics["sensitivity"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["specificity_at_sensitivity_95"] == 1.0
