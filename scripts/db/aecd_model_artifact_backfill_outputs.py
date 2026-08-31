from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path


def write_backfill_report(rows: Sequence[Mapping[str, str]], report_dir: Path, git_sha: str) -> None:
    """Write the auditable CSV and Markdown inventory for one backfill execution."""
    report_dir.mkdir(parents=True, exist_ok=True)
    fields = (
        "run_id",
        "registry_model_name",
        "registry_model_version",
        "source_dir",
        "source_model_version",
        "model_display_name",
        "model_file_count",
        "metadata_status",
        "parameter_change_count",
        "model_bundle_sha256",
    )
    with (report_dir / "backfill_inventory.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    model_count = len({row["registry_model_name"] for row in rows})
    metadata_count = sum(row["metadata_status"] == "available" for row in rows)
    (report_dir / "REPORT.md").write_text(
        "\n".join(
            [
                "# Historical model artifact backfill",
                "",
                f"- Backfilled bundles: {len(rows)}",
                f"- Registered model families: {model_count}",
                f"- Bundles with metadata: {metadata_count}",
                f"- Reconstruction Git SHA: `{git_sha}`",
                "- Model Registry entries are raw legacy checkpoint bundles and are explicitly marked non-serving.",
                "- Historical source phase remains `pre_change_unverified`; this does not assert reagent identity.",
                "",
                "See `backfill_inventory.csv` for run-to-artifact and source-version mapping.",
            ]
        ),
        encoding="utf-8",
    )
