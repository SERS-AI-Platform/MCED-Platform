from __future__ import annotations

from pathlib import Path

from scripts.db.aecd_model_lineage_timeline_core import TimelineRow
from scripts.db.aecd_model_lineage_timeline_outputs import (
    write_timeline_csv,
    write_timeline_outputs,
)


def test_write_timeline_csv_preserves_bom_and_change_names(tmp_path: Path) -> None:
    # Given: one source-time timeline row with a changed hyperparameter.
    row = TimelineRow(
        run_id="run-1",
        registry_model_name="family-a",
        registry_model_version="2",
        source_dir="results/v002",
        source_model_version="v002",
        model_display_name="Model A",
        source_timestamp="2026-03-02T10:00:00",
        source_timestamp_epoch=1772445600.0,
        source_time_status="observed",
        history_order=2,
        hyperparameter_change_count=1,
        changed_hyperparameters=("config_learning_rate",),
        cancer_screening_auc=0.72,
        cancer_type_id_auc=None,
    )
    output = tmp_path / "model_lineage_timeline.csv"

    # When: the timeline CSV is written.
    write_timeline_csv((row,), output)

    # Then: Excel-compatible UTF-8-SIG and machine-readable change fields are present.
    content = output.read_bytes()
    assert content.startswith(b"\xef\xbb\xbf")
    text = content.decode("utf-8-sig")
    assert "hyperparameter_change_count" in text
    assert "config_learning_rate" in text


def test_write_timeline_outputs_documents_mlflow_visibility_steps(tmp_path: Path) -> None:
    # Given: a minimal source-time timeline with no parameter transitions.
    row = TimelineRow(
        run_id="run-1",
        registry_model_name="family-a",
        registry_model_version="1",
        source_dir="results/v001",
        source_model_version="v001",
        model_display_name="Model A",
        source_timestamp="2026-03-01T10:00:00",
        source_timestamp_epoch=1772359200.0,
        source_time_status="observed",
        history_order=1,
        hyperparameter_change_count=0,
        changed_hyperparameters=(),
        cancer_screening_auc=0.70,
        cancer_type_id_auc=None,
    )

    # When: the complete timeline report is written.
    write_timeline_outputs((row,), (), tmp_path)

    # Then: the report explains the UI filters needed to reveal the historical run.
    report = (tmp_path / "MODEL_LINEAGE_TIMELINE.md").read_text(encoding="utf-8")
    assert "Model training" in report
    assert "All time" in report
    assert "lineage_hyperparameter_change_count" in report
