#!/usr/bin/env python3
"""Summarize coverage.xml by MCED-Platform development process.

The CI gate currently measures the importable ``sers`` package with pytest-cov.
This report keeps that numeric scope honest by grouping the measured files into
the product/process buckets used in project status reviews.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ProcessBucket:
    key: str
    label: str
    patterns: tuple[str, ...]
    target: str


@dataclass(frozen=True)
class FileCoverage:
    filename: str
    covered: int
    statements: int

    @property
    def percent(self) -> float:
        if self.statements == 0:
            return 100.0
        return self.covered / self.statements * 100.0


@dataclass(frozen=True)
class ProcessCoverage:
    bucket: ProcessBucket
    files: tuple[FileCoverage, ...]

    @property
    def covered(self) -> int:
        return sum(file.covered for file in self.files)

    @property
    def statements(self) -> int:
        return sum(file.statements for file in self.files)

    @property
    def percent(self) -> float | None:
        if self.statements == 0:
            return None
        return self.covered / self.statements * 100.0


PROCESS_BUCKETS: tuple[ProcessBucket, ...] = (
    ProcessBucket(
        "cli_workflows",
        "CLI workflow surface",
        ("src/sers/cli/",),
        "Gate entrypoints and user-facing workflow commands.",
    ),
    ProcessBucket(
        "config_public_api",
        "Config + public package API",
        ("src/sers/__init__.py", "src/sers/config.py", "src/sers/logging_config.py"),
        "Keep as a stable import/config contract.",
    ),
    ProcessBucket(
        "data_io_validation",
        "Data I/O + validation",
        ("src/sers/io.py", "src/sers/validation.py"),
        "Raise before broader external data onboarding.",
    ),
    ProcessBucket(
        "preprocessing_signal",
        "Preprocessing + signal processing",
        ("src/sers/preprocessing.py", "src/sers/signal.py"),
        "Core algorithm path; ratchet toward high coverage.",
    ),
    ProcessBucket(
        "quality_control",
        "Quality control",
        ("src/sers/qc/",),
        "Clinical suitability logic; keep ratcheting upward.",
    ),
    ProcessBucket(
        "risk_scoring",
        "SSI/CTI risk scoring",
        ("src/sers/scoring.py",),
        "High-confidence unit surface.",
    ),
    ProcessBucket(
        "model_helpers",
        "Model helper imports",
        ("src/sers/models/",),
        "Currently mostly import-path scaffolding; test when used actively.",
    ),
    ProcessBucket(
        "analysis_visualization",
        "Analysis + visualization",
        ("src/sers/analysis.py", "src/sers/visualization.py", "src/sers/visualization/"),
        "Important for reporting, but not yet a CI-gated release surface.",
    ),
    ProcessBucket(
        "calibration_transfer",
        "Calibration transfer",
        ("src/sers/calibration_transfer.py",),
        "Open gate for instrument-transfer validation.",
    ),
    ProcessBucket(
        "staging_ingest_utilities",
        "Staging/ingest utilities",
        ("src/sers/ingest_ypan.py", "src/sers/reingest_staging.py", "src/sers/update_lun_dates.py"),
        "Operational utilities; test before making them release-critical.",
    ),
)

UNCATEGORIZED_BUCKET = ProcessBucket(
    "uncategorized_sers",
    "Uncategorized sers package files",
    (),
    "Add a process bucket if this becomes non-empty.",
)

UNMEASURED_PROCESS_NOTES: tuple[tuple[str, str, str], ...] = (
    (
        "deployment_software",
        "scripts/deployment/**",
        "Software product code exists in this repo, but is outside the current --cov=sers gate.",
    ),
    (
        "dashboard_workspace",
        "solum-dashboard",
        "Dashboard workspace is documented as a separate/local asset, not tracked as code here.",
    ),
)


def normalize_filename(filename: str) -> str:
    return filename.replace("\\", "/").lstrip("./")


def classify_file(filename: str) -> ProcessBucket:
    normalized = normalize_filename(filename)
    for bucket in PROCESS_BUCKETS:
        if any(normalized == pattern or normalized.startswith(pattern) for pattern in bucket.patterns):
            return bucket
    return UNCATEGORIZED_BUCKET


def parse_coverage_xml(path: Path) -> tuple[FileCoverage, ...]:
    root = ET.parse(path).getroot()
    by_filename: dict[str, list[tuple[int, int]]] = {}

    for class_node in root.findall(".//class"):
        filename = normalize_filename(class_node.get("filename", ""))
        if not filename:
            continue

        covered = 0
        statements = 0
        for line in class_node.findall("./lines/line"):
            statements += 1
            if int(line.get("hits", "0")) > 0:
                covered += 1

        by_filename.setdefault(filename, []).append((covered, statements))

    files = []
    for filename, chunks in sorted(by_filename.items()):
        covered = sum(chunk[0] for chunk in chunks)
        statements = sum(chunk[1] for chunk in chunks)
        files.append(FileCoverage(filename, covered, statements))
    return tuple(files)


def build_process_report(files: Iterable[FileCoverage]) -> tuple[ProcessCoverage, ...]:
    grouped: dict[str, list[FileCoverage]] = {
        bucket.key: [] for bucket in (*PROCESS_BUCKETS, UNCATEGORIZED_BUCKET)
    }
    buckets_by_key = {bucket.key: bucket for bucket in (*PROCESS_BUCKETS, UNCATEGORIZED_BUCKET)}

    for file in files:
        bucket = classify_file(file.filename)
        grouped[bucket.key].append(file)

    return tuple(
        ProcessCoverage(buckets_by_key[key], tuple(files_for_bucket))
        for key, files_for_bucket in grouped.items()
        if files_for_bucket or key != UNCATEGORIZED_BUCKET.key
    )


def format_percent(percent: float | None) -> str:
    if percent is None:
        return "n/a"
    return f"{percent:.1f}%"


def format_markdown(report: Iterable[ProcessCoverage]) -> str:
    lines = [
        "## Coverage By Process",
        "",
        "| Process | Files | Covered / Stmts | Coverage | Interpretation |",
        "|---|---:|---:|---:|---|",
    ]

    for process in report:
        lines.append(
            "| {label} | {files} | {covered}/{statements} | {percent} | {target} |".format(
                label=process.bucket.label,
                files=len(process.files),
                covered=process.covered,
                statements=process.statements,
                percent=format_percent(process.percent),
                target=process.bucket.target,
            )
        )

    lines.extend(["", "### Not In Current Numeric Coverage Scope", ""])
    for key, path, note in UNMEASURED_PROCESS_NOTES:
        lines.append(f"- `{key}` (`{path}`): {note}")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "coverage_xml",
        type=Path,
        nargs="?",
        default=Path("coverage.xml"),
        help="Path to pytest-cov XML report.",
    )
    args = parser.parse_args(argv)

    if not args.coverage_xml.exists():
        print(f"coverage XML not found: {args.coverage_xml}", file=sys.stderr)
        return 2

    files = parse_coverage_xml(args.coverage_xml)
    report = build_process_report(files)
    print(format_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
