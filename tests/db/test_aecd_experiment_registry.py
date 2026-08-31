from __future__ import annotations

import csv
import math
from pathlib import Path

from scripts.db.aecd_experiment_registry import build_historical_summary_runs
from scripts.db.aecd_experiment_registry_core import (
    allowed_artifact,
    finite_metric,
    select_recent_rows,
)


def test_select_recent_rows_keeps_only_evidence_bearing_runs() -> None:
    rows = [
        {"directory": "complete", "status": "주 분석 결과"},
        {"directory": "reference", "status": "참고 결과"},
        {"directory": "smoke", "status": "증거 제외"},
        {"directory": "intermediate", "status": "완료 metric 없음"},
        {"directory": "summary", "status": "파생 산출물"},
    ]

    selected = select_recent_rows(rows)

    assert [row["directory"] for row in selected] == ["complete", "reference"]


def test_build_historical_summary_runs_groups_tasks_without_raw_paths(tmp_path: Path) -> None:
    path = tmp_path / "historical.csv"
    rows = [
        {
            "experiment": "standard preprocessing, 3 blocks, historical run",
            "preprocessing": "trim + SG",
            "model": "Multi-scale ResNet",
            "task": "cancer_vs_non_cancer",
            "roc_auc": "0.45",
            "macro_roc_auc": "",
            "balanced_accuracy": "0.47",
            "macro_f1": "",
            "comparison_status": "historical; seed/run differs",
            "notes": "same cohort",
        },
        {
            "experiment": "standard preprocessing, 3 blocks, historical run",
            "preprocessing": "trim + SG",
            "model": "Multi-scale ResNet",
            "task": "three_class",
            "roc_auc": "",
            "macro_roc_auc": "0.48",
            "balanced_accuracy": "0.35",
            "macro_f1": "0.27",
            "comparison_status": "historical; seed/run differs",
            "notes": "same cohort",
        },
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    runs = build_historical_summary_runs(path)

    assert len(runs) == 1
    run = runs[0]
    assert dict(run.metrics)["cancer_screening_auc"] == 0.45
    assert dict(run.metrics)["cancer_type_id_macro_auc"] == 0.48
    assert all(key not in dict(run.params) for key in ("input", "output", "source_uri"))


def test_finite_metric_rejects_nonfinite_values() -> None:
    assert finite_metric("0.75") == 0.75
    assert finite_metric("nan") is None
    assert finite_metric(math.inf) is None
    assert finite_metric("not-a-number") is None


def test_allowed_artifact_excludes_patient_and_array_outputs() -> None:
    assert allowed_artifact(Path("REPORT.md"))
    assert allowed_artifact(Path("oof_metrics.csv"))
    assert not allowed_artifact(Path("patient_oof_predictions.csv"))
    assert not allowed_artifact(Path("oof_prediction_arrays.npz"))
    assert not allowed_artifact(Path("repeat_metrics_mc.csv"))
