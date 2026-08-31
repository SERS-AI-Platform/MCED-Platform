#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mlflow>=3.1,<4",
#     "numpy>=1.24",
#     "openpyxl>=3.1",
# ]
# ///

# ─── How to run ───
# Scan all configured source roots without writing a database:
#     uv run scripts/db/aecd_source_inventory.py
# Register the aggregate screening run in local MLflow:
#     uv run scripts/db/aecd_source_inventory.py --log-mlflow
# ──────────────────

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.db.aecd_measurement_scan import discover_source_specs, scan_sources  # noqa: E402
from scripts.db.aecd_measurement_sources import MeasurementRecord, MeasurementScan  # noqa: E402

Scalar: TypeAlias = str | int | float | bool | None
Row: TypeAlias = dict[str, Scalar]


@dataclass(slots=True)
class _Bucket:
    valid_files: int = 0
    measurement_keys: set[str] = field(default_factory=set)
    source_batches: set[str] = field(default_factory=set)
    dates: set[str] = field(default_factory=set)
    groups: Counter[str] = field(default_factory=Counter)
    reagent_names: set[str] = field(default_factory=set)


def screening_eligibility(record: MeasurementRecord) -> str:
    if record.is_averaged:
        return "exclude_derived_average"
    if record.artifact_role in {"calibration_control", "background", "equipment_test", "metabolite_reference"}:
        return "exclude_reference_or_control"
    if record.source_domain in {"mapping", "thermo", "medical", "remeasurement"}:
        return "include_in_candidate_pool"
    return "exclude_unclassified_source"


def _bucket_key(record: MeasurementRecord) -> tuple[str, str, str, str, str, str]:
    return (
        record.source_domain,
        record.source_kind,
        record.preparation,
        record.reagent_phase,
        record.artifact_role,
        screening_eligibility(record),
    )


def aggregate_scan(scan: MeasurementScan) -> tuple[list[Row], list[Row], dict[str, Scalar]]:
    buckets: dict[tuple[str, str, str, str, str, str], _Bucket] = defaultdict(_Bucket)
    for record in scan.records:
        bucket = buckets[_bucket_key(record)]
        bucket.valid_files += 1
        bucket.measurement_keys.add(record.measurement_key)
        bucket.source_batches.add(record.source_batch)
        if record.acquisition_date is not None:
            bucket.dates.add(record.acquisition_date)
        if record.group_code is not None:
            bucket.groups[record.group_code] += 1
        if record.reagent_name is not None:
            bucket.reagent_names.add(record.reagent_name)

    rows: list[Row] = []
    for key in sorted(buckets):
        domain, kind, preparation, phase, role, eligibility = key
        bucket = buckets[key]
        names = sorted(bucket.reagent_names)
        rows.append(
            {
                "source_domain": domain,
                "source_kind": kind,
                "preparation": preparation,
                "reagent_phase": phase,
                "reagent_name_status": "observed" if names else "not_recorded",
                "artifact_role": role,
                "screening_eligibility": eligibility,
                "valid_file_count": bucket.valid_files,
                "measurement_key_count": len(bucket.measurement_keys),
                "source_batch_count": len(bucket.source_batches),
                "date_min": min(bucket.dates) if bucket.dates else "",
                "date_max": max(bucket.dates) if bucket.dates else "",
                "group_counts": json.dumps(dict(sorted(bucket.groups.items())), ensure_ascii=False),
            }
        )

    rejection_counts: Counter[tuple[str, str]] = Counter(
        (item.source_domain, item.reason_code) for item in scan.rejected
    )
    rejection_rows = [
        {"source_domain": domain, "reason_code": reason, "rejected_file_count": count}
        for (domain, reason), count in sorted(rejection_counts.items())
    ]
    candidate_count = sum(
        int(row["valid_file_count"])
        for row in rows
        if row["screening_eligibility"] == "include_in_candidate_pool"
    )
    summary: dict[str, Scalar] = {
        "total_files_seen": scan.total_files,
        "valid_files": len(scan.records),
        "rejected_files": len(scan.rejected),
        "averages_excluded": scan.averages_excluded,
        "total_points": scan.points,
        "screening_candidate_files": candidate_count,
        "manifest_sha256": scan.manifest_sha256,
    }
    return rows, rejection_rows, summary


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sidecar(relative: str) -> bool:
    lower = relative.casefold()
    return lower.endswith(".identifier") or lower.endswith(":zone.identifier")


def archive_alias_summary(repo_root: Path) -> dict[str, Scalar]:
    archive = repo_root / "data" / "환원제_변경후_데이터.zip"
    mapping_root = repo_root / "data" / "mapping"
    if not archive.is_file() or not mapping_root.is_dir():
        return {"status": "not_found", "archive_member_count": 0, "filesystem_file_count": 0}

    filesystem: dict[str, int] = {
        path.relative_to(mapping_root).as_posix(): path.stat().st_size
        for path in mapping_root.rglob("*")
        if path.is_file() and not _is_sidecar(path.relative_to(mapping_root).as_posix())
    }
    archive_members: dict[str, int] = {}
    with zipfile.ZipFile(archive) as source:
        for info in source.infolist():
            name = info.filename.replace("\\", "/")
            if name.endswith("/") or _is_sidecar(name):
                continue
            if name.startswith("mapping/"):
                name = name[len("mapping/") :]
            archive_members[name] = info.file_size
    common = filesystem.keys() & archive_members.keys()
    size_mismatches = sum(filesystem[name] != archive_members[name] for name in common)
    status = (
        "duplicate_of_mapping_post_change"
        if len(common) == len(filesystem) == len(archive_members) and size_mismatches == 0
        else "requires_review"
    )
    return {
        "status": status,
        "archive_sha256": _sha256_file(archive),
        "archive_member_count": len(archive_members),
        "filesystem_file_count": len(filesystem),
        "matching_member_count": len(common),
        "size_mismatch_count": size_mismatches,
    }


def _write_csv(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=keys)
        if keys:
            writer.writeheader()
            writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_report(output: Path, summary: dict[str, Scalar], archive: dict[str, Scalar]) -> None:
    lines = [
        "# AECD all-source screening inventory",
        "",
        "## Scope",
        "",
        "- Clinical candidate pool includes mapping, historical Thermo, Medical Raman, and remeasurement repeats.",
        "- `mapping` is classified as `post_change_verified` from the observed measurement reagent lot.",
        "- Historical source roots are `pre_change_unverified`: they are temporally earlier than the current mapping cohort, but the old reducing-agent identity is not present in source metadata.",
        "- Finite raw/replicate files are retained; no spectral QC filter is applied here.",
        "- `_ave` files, calibration/reference files, equipment tests, and metabolite references are recorded but excluded from the screening candidate pool.",
        "",
        "## Aggregate result",
        "",
        f"- Files seen: {summary['total_files_seen']}",
        f"- Valid finite files: {summary['valid_files']}",
        f"- Rejected files: {summary['rejected_files']}",
        f"- Derived averages excluded: {summary['averages_excluded']}",
        f"- Candidate-pool files: {summary['screening_candidate_files']}",
        f"- Manifest SHA-256: `{summary['manifest_sha256']}`",
        "",
        "## Duplicate archive check",
        "",
        f"- Status: `{archive['status']}`",
        f"- Archive members after sidecar exclusion: {archive.get('archive_member_count', 0)}",
        f"- Filesystem mapping files after sidecar exclusion: {archive.get('filesystem_file_count', 0)}",
        f"- Matching members: {archive.get('matching_member_count', 0)}",
        f"- Size mismatches: {archive.get('size_mismatch_count', 0)}",
        "",
        "The CSV and JSON files in this directory contain aggregate counts only; patient identifiers, raw filenames, source URIs, and spectral arrays are not exported.",
        "",
        "Cancer Screening AUC remains subject to the documented hospital/measurement confounding disclaimer.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _log_mlflow(output: Path, summary: dict[str, Scalar], archive: dict[str, Scalar], tracking_db: Path) -> str:
    import mlflow
    from mlflow.tracking import MlflowClient

    tracking_db.parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{tracking_db.resolve()}")
    experiment_name = "aecd-all-source-screening"
    mlflow.set_experiment(experiment_name)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment was not created: {experiment_name}")
    lineage_key = f"aecd-all-source-screening:{summary['manifest_sha256']}:{archive.get('archive_sha256', 'none')}"
    existing = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.lineage_key = '{lineage_key}'",
        max_results=1,
    )
    if existing:
        return existing[0].info.run_id
    with mlflow.start_run(run_name="all_source_screening_20260827") as run:
        mlflow.log_params(
            {
                "source_scope": "all_configured_source_roots",
                "averages_included": "false",
                "qc_policy": "finite_value_gate_only",
                "reagent_phase_policy": "post_verified_pre_unverified",
                "archive_alias_status": str(archive["status"]),
            }
        )
        mlflow.log_metrics(
            {
                key: float(value)
                for key, value in summary.items()
                if key != "manifest_sha256" and isinstance(value, (int, float))
            }
        )
        mlflow.set_tags(
            {
                "lineage_key": lineage_key,
                "purpose": "source-screening",
                "data_policy": "aggregate_only_no_patient_ids_or_raw_uris",
                "cancer_screening_disclaimer": "hospital_measurement_confounding_possible",
            }
        )
        mlflow.log_artifacts(str(output), artifact_path="source_screening")
        return run.info.run_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen all configured AECD source roots")
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "results" / "data_source_screening_20260827_v1",
    )
    parser.add_argument("--mlflow-db", type=Path, default=REPO / "mlflow.db")
    parser.add_argument("--log-mlflow", action="store_true")
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    output = args.output.resolve()
    scan = scan_sources(repo_root, discover_source_specs(repo_root), include_averages=False)
    rows, rejection_rows, summary = aggregate_scan(scan)
    archive = archive_alias_summary(repo_root)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "source_inventory.csv", rows)
    _write_csv(output / "rejection_summary.csv", rejection_rows)
    _write_json(output / "source_inventory.json", {"summary": summary, "archive_alias": archive, "sources": rows, "rejections": rejection_rows})
    _write_report(output, summary, archive)
    run_id = _log_mlflow(output, summary, archive, args.mlflow_db) if args.log_mlflow else None
    print(json.dumps({"output_dir": str(output), "summary": summary, "archive_alias": archive, "mlflow_run_id": run_id}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
