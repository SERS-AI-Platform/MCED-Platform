#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create an organized copy of Ian's SERS and clinical data tree.

The source tree is intentionally left unchanged. This script copies data into a
study-design-oriented structure and writes manifests that preserve the source
path, destination path, checksums, and best-effort sample metadata.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


DEFAULT_SOURCE = Path("/Users/ian/Downloads/data/data")
DEFAULT_DEST = Path("/Users/ian/Downloads/data/organized_data")

SKIP_DIR_NAMES = {".claude", ".venv", "__pycache__"}
SKIP_FILE_NAMES = {".DS_Store"}
SKIP_SUFFIXES = {".pyc"}

TOP_LEVELS = {
    "readme": "00_README",
    "clinical": "01_clinical_metadata",
    "pooled": "02_sers_primary_pooled_acquisition",
    "balanced": "03_sers_date_lot_balanced_acquisition",
    "equipment": "04_machine_repeatability_tests",
    "not_yet": "05_not_yet_analyzed",
    "supporting": "06_supporting_or_previous_outputs",
    "manifests": "99_manifests",
}

NOT_YET_ANALYZED_DIRS = {
    "20260407_Bladder_1mW_0.05s_Ave 100",
}

DATE_RE = re.compile(r"(20\d{6})")
POST_OP_RE = re.compile(r"(?i)(^|[\s_/.-])po\.?\s+")
NF_RE = re.compile(r"(?i)(^|[\s_/.-])nf($|[\s_.-])")
SUBJECT_RE = re.compile(
    r"(?i)(?:po\.\s*)?"
    r"(?P<group>YNOR|YPAN|CPAN|SPAN|BLC|BLA|PRO|BRE|OVA|LUN|NOR|DIA|H\.?\s?D\.?|HBP|CRC)"
    r"\s*(?P<num>\d+)"
    r"(?:\s*NF)?"
    r"(?:[_\-\s](?P<rep>\d+|ave))?"
)


@dataclass(frozen=True)
class PlannedFile:
    source: Path
    relative: Path
    dest: Path | None
    size: int
    mtime_iso: str
    extension: str
    source_bucket: str
    source_category: str
    acquisition_design: str
    skip_reason: str


def rel_parts(relative: Path) -> tuple[str, ...]:
    return tuple(relative.parts)


def has_skipped_component(relative: Path) -> str:
    for part in rel_parts(relative):
        if part in SKIP_DIR_NAMES:
            return f"skipped_directory:{part}"
    return ""


def skip_reason(relative: Path) -> str:
    component_reason = has_skipped_component(relative)
    if component_reason:
        return component_reason
    name = relative.name
    if name in SKIP_FILE_NAMES:
        return f"skipped_file:{name}"
    if name.endswith("Zone.Identifier"):
        return "skipped_os_metadata:Zone.Identifier"
    if relative.suffix.lower() in SKIP_SUFFIXES:
        return f"skipped_cache_suffix:{relative.suffix.lower()}"
    return ""


def category_from_destination(dest: Path | None) -> tuple[str, str]:
    if dest is None:
        return ("skipped", "skipped")
    # The destination passed here is absolute. Use the first organized-data child
    # by scanning for the known top-level folder names.
    for part in dest.parts:
        if part == TOP_LEVELS["clinical"]:
            return ("clinical_metadata", "clinical_metadata")
        if part == TOP_LEVELS["pooled"]:
            return ("sers_primary_pooled_acquisition", "pooled_initial")
        if part == TOP_LEVELS["balanced"]:
            return ("sers_date_lot_balanced_acquisition", "date_lot_balanced")
        if part == TOP_LEVELS["equipment"]:
            return ("machine_repeatability_tests", "machine_repeatability")
        if part == TOP_LEVELS["not_yet"]:
            return ("not_yet_analyzed", "not_yet_analyzed")
        if part == TOP_LEVELS["supporting"]:
            return ("supporting_or_previous_outputs", "supporting_previous_output")
        if part == TOP_LEVELS["readme"]:
            return ("readme", "readme")
        if part == TOP_LEVELS["manifests"]:
            return ("manifest", "manifest")
    return ("unknown", "unknown")


def destination_for(source_root: Path, dest_root: Path, relative: Path) -> Path | None:
    reason = skip_reason(relative)
    if reason:
        return None

    parts = rel_parts(relative)
    if not parts:
        return None
    top = parts[0]
    rest = Path(*parts[1:]) if len(parts) > 1 else Path()

    if top == "clinical_data":
        return dest_root / TOP_LEVELS["clinical"] / top / rest
    if top == "exclusions":
        return dest_root / TOP_LEVELS["clinical"] / top / rest

    if top == "raw_data":
        if len(parts) > 1 and parts[1] in NOT_YET_ANALYZED_DIRS:
            return dest_root / TOP_LEVELS["not_yet"] / top / rest
        return dest_root / TOP_LEVELS["pooled"] / "thermo" / top / rest

    if top == "raw_data_medical":
        return dest_root / TOP_LEVELS["pooled"] / "medical" / top / rest

    if top.startswith("20260319_Bladder"):
        return dest_root / TOP_LEVELS["pooled"] / "thermo" / top / rest

    if top == "Thermo":
        return dest_root / TOP_LEVELS["balanced"] / "thermo" / top / rest
    if top == "임상데이터":
        return dest_root / TOP_LEVELS["balanced"] / "thermo" / top / rest
    if top == "Medical":
        return dest_root / TOP_LEVELS["balanced"] / "medical" / top / rest
    if top == "Handheld":
        return dest_root / TOP_LEVELS["balanced"] / "handheld" / top / rest

    if top == "equipment_test_data":
        return dest_root / TOP_LEVELS["equipment"] / top / rest

    if top in {
        "clinical_dashboard",
        "processed",
        "Metabolite analysis_Thermo",
        "raw_data-DESKTOP-8VL414N",
    }:
        return dest_root / TOP_LEVELS["supporting"] / top / rest

    return dest_root / TOP_LEVELS["supporting"] / top / rest


def iso_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")


def plan_files(source_root: Path, dest_root: Path) -> list[PlannedFile]:
    planned: list[PlannedFile] = []
    for root, dirs, files in os.walk(source_root):
        dirs.sort()
        files.sort()
        root_path = Path(root)
        for file_name in files:
            source = root_path / file_name
            if not source.is_file():
                continue
            relative = source.relative_to(source_root)
            reason = skip_reason(relative)
            dest = destination_for(source_root, dest_root, relative)
            source_category, acquisition_design = category_from_destination(dest)
            parts = rel_parts(relative)
            source_bucket = parts[0] if parts else ""
            stat = source.stat()
            planned.append(
                PlannedFile(
                    source=source,
                    relative=relative,
                    dest=dest,
                    size=stat.st_size,
                    mtime_iso=iso_mtime(source),
                    extension=source.suffix.lower(),
                    source_bucket=source_bucket,
                    source_category=source_category,
                    acquisition_design=acquisition_design,
                    skip_reason=reason,
                )
            )
    return planned


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_and_hash(source: Path, dest: Path, chunk_size: int = 1024 * 1024) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp-organizing")
    if tmp.exists():
        tmp.unlink()

    digest = hashlib.sha256()
    try:
        with source.open("rb") as src_handle, tmp.open("wb") as dst_handle:
            for chunk in iter(lambda: src_handle.read(chunk_size), b""):
                digest.update(chunk)
                dst_handle.write(chunk)
        shutil.copystat(source, tmp, follow_symlinks=True)
        os.replace(tmp, dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    return digest.hexdigest()


def infer_instrument(relative: Path, acquisition_design: str) -> str:
    parts = rel_parts(relative)
    if not parts:
        return ""
    top = parts[0]
    if top in {"raw_data", "Thermo", "임상데이터"} or top.startswith("20260319_Bladder"):
        return "thermo"
    if top in {"raw_data_medical", "Medical"}:
        return "medical"
    if top == "Handheld":
        return "handheld"
    if top == "equipment_test_data" and len(parts) > 1:
        machine = parts[1].lower()
        if machine.startswith("medical"):
            return "medical"
        return machine
    if acquisition_design == "clinical_metadata":
        return ""
    return ""


def infer_date(relative: Path) -> str:
    text = str(relative)
    match = DATE_RE.search(text)
    if not match:
        return ""
    raw = match.group(1)
    return f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]}"


def infer_sample_role(relative: Path, acquisition_design: str) -> str:
    if acquisition_design == "clinical_metadata":
        return "metadata"
    if acquisition_design == "machine_repeatability":
        return "repeatability_test"
    if acquisition_design == "supporting_previous_output":
        return "supporting_artifact"

    parts_lower = [p.lower() for p in rel_parts(relative)]
    name_lower = relative.name.lower()
    stem_lower = relative.stem.lower()

    if any("blank" in part for part in parts_lower):
        return "qc_blank"
    if any("ref_ps" in part for part in parts_lower) or re.match(r"ps\s+\d+", stem_lower):
        return "qc_reference_ps"
    if any("ref_si" in part for part in parts_lower) or re.match(r"si\s+\d+", stem_lower):
        return "qc_reference_si"
    if any("calibration" in part or "control" in part for part in parts_lower):
        return "qc_calibration"
    if name_lower.endswith((".csv", ".xlsx", ".xlsm", ".xls")):
        if acquisition_design in {"pooled_initial", "date_lot_balanced", "not_yet_analyzed"}:
            return "patient_sample"
    return "other"


def infer_fasting_status(relative: Path) -> str:
    text = str(relative)
    if NF_RE.search(text):
        return "non_fasting"
    return "assumed_fasting_or_unspecified"


def infer_sample_timing(relative: Path) -> str:
    text = str(relative)
    if POST_OP_RE.search(text):
        return "post_op"
    return "pre_or_unspecified"


def normalize_group(group: str) -> str:
    cleaned = re.sub(r"\s+", "", group.upper())
    cleaned = cleaned.replace("H.D", "H.D.").replace("HD.", "H.D.").replace("HD", "H.D.")
    if cleaned == "BLA":
        return "BLC"
    return cleaned


def infer_subject(relative: Path) -> tuple[str, str, str, str]:
    text = relative.stem
    if "_Sample_" in text:
        text = text.split("_Sample_", 1)[0]
    match = SUBJECT_RE.search(text)
    if not match:
        return ("", "", "", "")
    group = normalize_group(match.group("group"))
    sample_num = match.group("num")
    replicate = match.group("rep") or ""
    if replicate.lower() == "ave":
        replicate = "ave"
    base_subject = f"{group} {sample_num}"
    sample_id = f"{group} {sample_num}"
    return (group, sample_num, replicate, base_subject or sample_id)


def infer_label_group(relative: Path, subject_group: str) -> str:
    if subject_group:
        return subject_group
    text = str(relative).lower()
    ordered_patterns = [
        ("H.D.", ["high blood pressure + diabetes", "htn+dm", "h.d"]),
        ("CPAN", ["c-pancreatic", "cpan"]),
        ("SPAN", ["s-pancreatic", "span"]),
        ("YPAN", ["y-pancreatic", "ypan"]),
        ("BLC", ["bladder", "bladdder", "blc", "bla"]),
        ("PRO", ["prostate", "pro "]),
        ("BRE", ["breast", "bre "]),
        ("OVA", ["ovarian", "ova "]),
        ("LUN", ["lung", "lun "]),
        ("CRC", ["colorectal", "crc "]),
        ("DIA", ["diabetes", "dia "]),
        ("HBP", ["high blood pressure", "hbp", "htn"]),
        ("YNOR", ["y-normal", "ynor"]),
        ("NOR", ["normal", "nor "]),
        ("PAN", ["pancreatic"]),
    ]
    for label, needles in ordered_patterns:
        if any(needle in text for needle in needles):
            return label
    return ""


def default_training_decision(
    acquisition_design: str,
    sample_role: str,
    fasting_status: str,
    sample_timing: str,
) -> tuple[bool, str]:
    if sample_role != "patient_sample":
        return (False, f"not_patient_sample:{sample_role}")
    if acquisition_design not in {"pooled_initial", "date_lot_balanced"}:
        return (False, f"excluded_acquisition_design:{acquisition_design}")
    if fasting_status == "non_fasting":
        return (False, "non_fasting")
    if sample_timing == "post_op":
        return (False, "post_op")
    return (True, "")


def modeling_manifest_row(record: PlannedFile) -> dict[str, str]:
    if record.dest is None:
        return {}
    sample_role = infer_sample_role(record.relative, record.acquisition_design)
    fasting_status = infer_fasting_status(record.relative)
    sample_timing = infer_sample_timing(record.relative)
    subject_group, sample_number, replicate_id, base_subject_id = infer_subject(record.relative)
    label_group = infer_label_group(record.relative, subject_group)
    include, reason = default_training_decision(
        record.acquisition_design,
        sample_role,
        fasting_status,
        sample_timing,
    )
    return {
        "original_path": str(record.source),
        "organized_path": str(record.dest),
        "relative_source_path": str(record.relative),
        "source_bucket": record.source_bucket,
        "source_category": record.source_category,
        "acquisition_design": record.acquisition_design,
        "instrument": infer_instrument(record.relative, record.acquisition_design),
        "acquisition_date": infer_date(record.relative),
        "sample_role": sample_role,
        "label_group": label_group,
        "subject_group": subject_group,
        "sample_number": sample_number,
        "replicate_id": replicate_id,
        "base_subject_id": base_subject_id,
        "fasting_status": fasting_status,
        "sample_timing": sample_timing,
        "include_default_training": "true" if include else "false",
        "exclusion_reason": reason,
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_readme(dest_root: Path, source_root: Path, planned: list[PlannedFile]) -> None:
    readme_dir = dest_root / TOP_LEVELS["readme"]
    readme_dir.mkdir(parents=True, exist_ok=True)
    copied_count = sum(1 for item in planned if item.dest is not None)
    skipped_count = sum(1 for item in planned if item.dest is None)
    generated = datetime.now().isoformat(timespec="seconds")

    content = f"""# Organized SERS + Clinical Data Copy

Generated: {generated}

Source of truth:

`{source_root}`

Organized copy:

`{dest_root}`

This directory is a copied, organized view of the original source tree. The
original files under the source tree were not moved or rewritten.

## Study-design structure

- `01_clinical_metadata`: normalized clinical metadata, clean clinical tables,
  exclusion registries, and clinical data documentation.
- `02_sers_primary_pooled_acquisition`: earlier SERS data measured as broad
  grouped acquisitions without explicit date/lot balancing.
- `03_sers_date_lot_balanced_acquisition`: later SERS data measured date-by-date
  across groups to estimate or reduce date/lot/instrument batch effects.
- `04_machine_repeatability_tests`: equipment comparison and repeatability data,
  not default supervised clinical model training data.
- `05_not_yet_analyzed`: copied datasets that are present but intentionally
  excluded from default analysis until enabled.
- `06_supporting_or_previous_outputs`: dashboards, processed outputs, duplicate
  exports, metabolite analysis, and other supporting artifacts.
- `99_manifests`: audit files, checksums, and best-effort modeling manifest.

## Key sample rules

- `NF` in a filename means non-fasting; these files are copied but excluded from
  default fasting-only training.
- `Po.` or `PO` in a filename means post-surgery/post-operation urine sample.
  The base subject identity should match the corresponding non-PO sample.
- Reference, silicon, polystyrene, blank, and calibration/control folders are QC
  roles and are not supervised biological samples.
- Date folders and instrument folders should be treated as batch variables.

## Run summary

- Copied/planned data files: {copied_count}
- Intentionally skipped metadata/cache files: {skipped_count}

See `99_manifests/copy_map.csv`, `99_manifests/source_inventory.csv`, and
`99_manifests/modeling_manifest.csv` for file-level details.
"""
    (readme_dir / "DATASET_MAP.md").write_text(content, encoding="utf-8")


def write_registry(dest_root: Path, source_root: Path) -> None:
    manifests_dir = dest_root / TOP_LEVELS["manifests"]
    manifests_dir.mkdir(parents=True, exist_ok=True)
    content = f"""# Dataset registry for organized SERS + clinical data.
source_root: "{source_root}"
organized_root: "{dest_root}"

acquisition_designs:
  pooled_initial:
    description: "Earlier broad acquisition without explicit date/lot balancing."
    default_training_candidate: true
  date_lot_balanced:
    description: "Later date-wise acquisition across groups to estimate or reduce date/lot effects."
    default_training_candidate: true
  machine_repeatability:
    description: "Repeated equipment comparison measurements; not default clinical training data."
    default_training_candidate: false
  not_yet_analyzed:
    description: "Copied but intentionally excluded until explicitly enabled."
    default_training_candidate: false

top_level_directories:
  clinical_metadata: "01_clinical_metadata"
  primary_pooled_acquisition: "02_sers_primary_pooled_acquisition"
  date_lot_balanced_acquisition: "03_sers_date_lot_balanced_acquisition"
  machine_repeatability_tests: "04_machine_repeatability_tests"
  not_yet_analyzed: "05_not_yet_analyzed"
  supporting_or_previous_outputs: "06_supporting_or_previous_outputs"
  manifests: "99_manifests"

sample_rules:
  non_fasting:
    filename_token: "NF"
    default_training: false
  post_operation:
    filename_tokens: ["Po.", "PO"]
    subject_identity_rule: "Use the same base subject as the matching non-PO sample."
    default_training: false
  qc_roles:
    tokens: ["Blank", "Ref_PS", "Ref_Si", "PS", "Si", "Calibration", "control"]
    supervised_sample: false

known_examples:
  - source_pattern: "YPAN 1 NF"
    fasting_status: "non_fasting"
    default_training: false
  - source_pattern: "Po. YPAN 7"
    base_subject_id: "YPAN 7"
    sample_timing: "post_op"
    default_training: false
  - source_pattern: "0. Blank / 0. Ref_PS / 0. Ref_Si"
    sample_role: "qc"
    supervised_sample: false

batch_variables:
  - instrument
  - acquisition_date
  - source_bucket
  - acquisition_design
  - source_folder
"""
    (manifests_dir / "dataset_registry.yaml").write_text(content, encoding="utf-8")


def write_manifests(
    dest_root: Path,
    source_root: Path,
    planned: list[PlannedFile],
    checksums: dict[Path, str],
) -> None:
    manifests_dir = dest_root / TOP_LEVELS["manifests"]
    manifests_dir.mkdir(parents=True, exist_ok=True)

    inventory_rows: list[dict[str, object]] = []
    copy_rows: list[dict[str, object]] = []
    checksum_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []
    modeling_rows: list[dict[str, object]] = []

    for record in planned:
        copied = record.dest is not None
        inventory_rows.append(
            {
                "original_path": str(record.source),
                "relative_source_path": str(record.relative),
                "destination_path": str(record.dest) if record.dest else "",
                "size_bytes": record.size,
                "mtime_iso": record.mtime_iso,
                "extension": record.extension,
                "source_bucket": record.source_bucket,
                "source_category": record.source_category,
                "acquisition_design": record.acquisition_design,
                "copied": "true" if copied else "false",
                "skip_reason": record.skip_reason,
            }
        )
        if copied and record.dest is not None:
            copy_rows.append(
                {
                    "original_path": str(record.source),
                    "organized_path": str(record.dest),
                    "relative_source_path": str(record.relative),
                    "size_bytes": record.size,
                    "source_category": record.source_category,
                    "acquisition_design": record.acquisition_design,
                }
            )
            checksum_rows.append(
                {
                    "original_path": str(record.source),
                    "organized_path": str(record.dest),
                    "size_bytes": record.size,
                    "sha256": checksums.get(record.dest, ""),
                }
            )
            row = modeling_manifest_row(record)
            if row:
                modeling_rows.append(row)
        else:
            skipped_rows.append(
                {
                    "original_path": str(record.source),
                    "relative_source_path": str(record.relative),
                    "size_bytes": record.size,
                    "skip_reason": record.skip_reason,
                }
            )

    write_csv(
        manifests_dir / "source_inventory.csv",
        inventory_rows,
        [
            "original_path",
            "relative_source_path",
            "destination_path",
            "size_bytes",
            "mtime_iso",
            "extension",
            "source_bucket",
            "source_category",
            "acquisition_design",
            "copied",
            "skip_reason",
        ],
    )
    write_csv(
        manifests_dir / "copy_map.csv",
        copy_rows,
        [
            "original_path",
            "organized_path",
            "relative_source_path",
            "size_bytes",
            "source_category",
            "acquisition_design",
        ],
    )
    write_csv(
        manifests_dir / "checksums_sha256.csv",
        checksum_rows,
        ["original_path", "organized_path", "size_bytes", "sha256"],
    )
    write_csv(
        manifests_dir / "skipped_files.csv",
        skipped_rows,
        ["original_path", "relative_source_path", "size_bytes", "skip_reason"],
    )
    write_csv(
        manifests_dir / "modeling_manifest.csv",
        modeling_rows,
        [
            "original_path",
            "organized_path",
            "relative_source_path",
            "source_bucket",
            "source_category",
            "acquisition_design",
            "instrument",
            "acquisition_date",
            "sample_role",
            "label_group",
            "subject_group",
            "sample_number",
            "replicate_id",
            "base_subject_id",
            "fasting_status",
            "sample_timing",
            "include_default_training",
            "exclusion_reason",
        ],
    )


def detect_collisions(planned: list[PlannedFile]) -> dict[Path, list[Path]]:
    destinations: dict[Path, list[Path]] = {}
    for record in planned:
        if record.dest is None:
            continue
        destinations.setdefault(record.dest, []).append(record.source)
    return {dest: sources for dest, sources in destinations.items() if len(sources) > 1}


def run_copy(source_root: Path, dest_root: Path) -> int:
    if not source_root.exists():
        print(f"Source root does not exist: {source_root}", file=sys.stderr)
        return 2

    planned = plan_files(source_root, dest_root)
    collisions = detect_collisions(planned)
    if collisions:
        collision_rows = []
        for dest, sources in collisions.items():
            for source in sources:
                collision_rows.append({"organized_path": str(dest), "original_path": str(source)})
        write_csv(
            dest_root / TOP_LEVELS["manifests"] / "copy_collisions.csv",
            collision_rows,
            ["organized_path", "original_path"],
        )
        print(
            f"Refusing to copy because {len(collisions)} destination path collisions were found. "
            f"See {dest_root / TOP_LEVELS['manifests'] / 'copy_collisions.csv'}",
            file=sys.stderr,
        )
        return 3

    for folder in TOP_LEVELS.values():
        (dest_root / folder).mkdir(parents=True, exist_ok=True)

    to_copy = [item for item in planned if item.dest is not None]
    checksums: dict[Path, str] = {}
    copied = 0
    reused = 0

    for index, record in enumerate(to_copy, start=1):
        assert record.dest is not None
        dest = record.dest
        if dest.exists():
            if dest.stat().st_size != record.size:
                raise RuntimeError(f"Existing destination has different size: {dest}")
            source_hash = sha256_file(record.source)
            dest_hash = sha256_file(dest)
            if source_hash != dest_hash:
                raise RuntimeError(f"Existing destination checksum differs: {dest}")
            checksums[dest] = source_hash
            reused += 1
        else:
            checksums[dest] = copy_and_hash(record.source, dest)
            copied += 1

        if index % 5000 == 0 or index == len(to_copy):
            print(f"processed {index}/{len(to_copy)} files (copied={copied}, reused={reused})")

    write_readme(dest_root, source_root, planned)
    write_registry(dest_root, source_root)
    write_manifests(dest_root, source_root, planned, checksums)

    print(f"Done. copied={copied}, reused={reused}, skipped={len(planned) - len(to_copy)}")
    print(f"Organized copy: {dest_root}")
    print(f"Manifests: {dest_root / TOP_LEVELS['manifests']}")
    return 0


def verify(dest_root: Path) -> int:
    checksum_path = dest_root / TOP_LEVELS["manifests"] / "checksums_sha256.csv"
    if not checksum_path.exists():
        print(f"Checksum manifest not found: {checksum_path}", file=sys.stderr)
        return 2

    checked = 0
    failures: list[str] = []
    with checksum_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            path = Path(row["organized_path"])
            expected_size = int(row["size_bytes"])
            expected_hash = row["sha256"]
            if not path.exists():
                failures.append(f"missing: {path}")
                continue
            if path.stat().st_size != expected_size:
                failures.append(f"size mismatch: {path}")
                continue
            actual_hash = sha256_file(path)
            if actual_hash != expected_hash:
                failures.append(f"sha256 mismatch: {path}")
                continue
            checked += 1
            if checked % 10000 == 0:
                print(f"verified {checked} files")

    if failures:
        failure_path = dest_root / TOP_LEVELS["manifests"] / "verification_failures.txt"
        failure_path.write_text("\n".join(failures) + "\n", encoding="utf-8")
        print(f"Verification failed for {len(failures)} files. See {failure_path}", file=sys.stderr)
        return 1

    print(f"Verification passed for {checked} files.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = args.source.expanduser().resolve()
    dest_root = args.dest.expanduser().resolve()
    if args.verify_only:
        return verify(dest_root)
    return run_copy(source_root, dest_root)


if __name__ == "__main__":
    raise SystemExit(main())
