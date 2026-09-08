from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

from scripts.db.aecd_model_lineage_timeline_core import ParameterChangeRow, TimelineRow

TIMELINE_FIELDS = (
    "run_id",
    "registry_model_name",
    "registry_model_version",
    "source_dir",
    "source_model_version",
    "model_display_name",
    "source_timestamp",
    "source_timestamp_epoch",
    "source_time_status",
    "history_order",
    "hyperparameter_change_count",
    "changed_hyperparameters",
    "cancer_screening_auc",
    "cancer_type_id_auc",
)
PARAMETER_FIELDS = ("registry_model_name", "parameter_name", "change_count")


def _number(value: float | None) -> str:
    return "" if value is None else f"{value:.10g}"


def write_timeline_csv(rows: Sequence[TimelineRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TIMELINE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "run_id": row.run_id,
                    "registry_model_name": row.registry_model_name,
                    "registry_model_version": row.registry_model_version,
                    "source_dir": row.source_dir,
                    "source_model_version": row.source_model_version,
                    "model_display_name": row.model_display_name,
                    "source_timestamp": row.source_timestamp,
                    "source_timestamp_epoch": _number(row.source_timestamp_epoch),
                    "source_time_status": row.source_time_status,
                    "history_order": str(row.history_order),
                    "hyperparameter_change_count": str(row.hyperparameter_change_count),
                    "changed_hyperparameters": "|".join(row.changed_hyperparameters),
                    "cancer_screening_auc": _number(row.cancer_screening_auc),
                    "cancer_type_id_auc": _number(row.cancer_type_id_auc),
                }
            )


def write_parameter_change_csv(rows: Sequence[ParameterChangeRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PARAMETER_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "registry_model_name": row.registry_model_name,
                    "parameter_name": row.parameter_name,
                    "change_count": str(row.change_count),
                }
            )


def write_timeline_outputs(
    rows: Sequence[TimelineRow],
    changes: Sequence[ParameterChangeRow],
    report_dir: Path,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    write_timeline_csv(rows, report_dir / "model_lineage_timeline.csv")
    write_parameter_change_csv(changes, report_dir / "parameter_change_matrix.csv")
    observed_dates = [row.source_timestamp for row in rows if row.source_time_status == "observed"]
    date_range = f"{min(observed_dates)} to {max(observed_dates)}" if observed_dates else "not available"
    family_count = len({row.registry_model_name for row in rows})
    observed_count = sum(row.source_time_status == "observed" for row in rows)
    missing_count = len(rows) - observed_count
    (report_dir / "MODEL_LINEAGE_TIMELINE.md").write_text(
        "\n".join(
            [
                "# Historical model lineage timeline",
                "",
                f"- Historical bundles: {len(rows)}",
                f"- Registered model families: {family_count}",
                f"- Source timestamps observed: {observed_count}",
                f"- Source timestamps missing or invalid: {missing_count}",
                f"- Source time range: `{date_range}`",
                f"- Hyperparameter transition rows: {len(changes)}",
                "- Timeline order is calculated within each registered model family from the historical source timestamp.",
                "- Timezone-naive source timestamps are interpreted as UTC and retain their original text in the CSV.",
                "- MLflow backfill time is not used as the experiment chronology.",
                "- Historical checkpoint bundles remain non-serving artifacts; this timeline does not validate deployment readiness.",
                "- Cancer Screening metrics may be hospital/measurement-confounded and are not a cross-hospital generalization claim.",
                "- Direct model comparison requires identical aggregation mode, cancer set, non-cancer set, and sample count.",
                "",
                "Open `model_lineage_timeline.csv` for the version-level mapping and `parameter_change_matrix.csv` for the family-by-parameter change counts.",
                "",
                "## MLflow visibility",
                "",
                "1. Open experiment `aecd-model-lineage-timeline` (ID 6 in the default local store).",
                "2. Select `Model training`, choose `All time`, and clear any saved search expression.",
                "3. Open `historical_model_lineage_timeline_v1` and select the `timeline` artifact folder.",
                "4. In source experiment `aecd-model-artifact-backfill`, use `Model training` and `All time`; chart metric history names include `lineage_history_order`, `lineage_hyperparameter_change_count`, and `lineage_cancer_screening_auc`.",
            ]
        ),
        encoding="utf-8",
    )
