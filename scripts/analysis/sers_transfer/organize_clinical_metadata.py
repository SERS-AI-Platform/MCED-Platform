#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build an analysis-focused clinical metadata folder for organized_data.

This script only mutates the organized copy under /Users/ian/Downloads/data.
The original source tree under /Users/ian/Downloads/data/data is not changed.
"""

from __future__ import annotations

import csv
import hashlib
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


SOURCE_CLINICAL = Path("/Users/ian/Downloads/data/data/clinical_data")
SOURCE_EXCLUSIONS = Path("/Users/ian/Downloads/data/data/exclusions")
ORGANIZED_ROOT = Path("/Users/ian/Downloads/data/organized_data")
FINAL_DEST = ORGANIZED_ROOT / "01_clinical_metadata"
STAGING_DEST = ORGANIZED_ROOT / "01_clinical_metadata.__staging__"
TOP_LEVEL_MANIFESTS = ORGANIZED_ROOT / "99_manifests"

SKIP_DIRS = {".venv", "__pycache__"}
SKIP_NAMES = {".DS_Store"}
SKIP_SUFFIXES = {".pyc"}

RAW_DISEASE_DIRS = [
    "1. 전립선암",
    "2. 유방암",
    "3. 난소암",
    "4. 폐암",
    "5. 정상",
    "6. 당뇨",
    "7. 고혈압",
    "8. 당뇨 + 고혈압",
    "9. 대장암",
    "10. 췌장암",
    "11. 방광암",
]

RAW_OVERVIEW_FILES = [
    "20260312_솔루엠_임상테이블 정리.xlsx",
    "솔루엠헬스케어 탐색적 임상연구 정보 ('26.01.19 기준) 2.xlsx",
]

STANDARDIZED_FILES = [
    "standardized/all_clinical_standardized.csv",
    "전체환자_임상정보_정규화.xlsx",
]

CLEAN_FILES = [
    "clean_clinical_data.csv",
    "sample_exclusion_registry.csv",
    "missing_from_clean.csv",
    "master_clinical.csv",
    "standardized/phase_x_clean_clinical.csv",
]

SERS_LINKAGE_FILES = [
    "standardized/sers_clinical_merged.csv",
    "standardized/sers_patient_features.csv",
    "standardized/sers_clinical_correlation.csv",
]

QC_AND_MAPPING_FILES = [
    "CHANGELOG.md",
    "COLUMN_MAPPING_LOG.md",
    "all_disease_column_summary.xlsx",
    "cross_disease_column_mapping.xlsx",
    "normal_column_inventory.xlsx",
    "normal_highfill_column_detail.xlsx",
    "cancer_staging_tables.json",
    "standardized/cancer_date_columns_by_type.json",
    "standardized/clinical_data_report_draft.md",
    "standardized/completeness_matrix.csv",
    "standardized/stage_info_by_cancer_type.xlsx",
    "01_전립선암_column_detail.xlsx",
    "02_유방암_column_detail.xlsx",
    "03_난소암_column_detail.xlsx",
    "04_폐암_column_detail.xlsx",
    "06_당뇨_column_detail.xlsx",
    "07_고혈압_column_detail.xlsx",
    "08_당뇨_고혈압_column_detail.xlsx",
    "09_대장암_column_detail.xlsx",
    "10_췌장암_column_detail.xlsx",
    "11_방광암_column_detail.xlsx",
]

CORE_SCRIPT_FILES = [
    "standardize_clinical.py",
    "link_sers_clinical.py",
    "normalize_excel_files.py",
    "normalize_staging.py",
    "summarize_clinical.py",
    "analyze_clinical_completeness.py",
    "apply_column_mappings.py",
    "inspect_mapping.py",
    "inspect_raw_header.py",
    "debug_header.py",
]

COLUMN_DESCRIPTIONS = {
    "patient_id": ("identity", "Standard patient/sample identifier used for clinical joins."),
    "disease_group": ("identity", "Clinical disease group code."),
    "source_file": ("provenance", "Original clinical source file."),
    "age": ("demographics", "Age at clinical record/sample."),
    "sex": ("demographics", "Sex normalized to M/F where available."),
    "height_cm": ("demographics", "Height in centimeters."),
    "weight_kg": ("demographics", "Weight in kilograms."),
    "bmi": ("demographics", "Body mass index."),
    "bp_systolic": ("vitals", "Systolic blood pressure."),
    "bp_diastolic": ("vitals", "Diastolic blood pressure."),
    "smoking_status": ("lifestyle", "0=never, 1=former, 2=current, blank=unknown."),
    "drinking_status": ("lifestyle", "0=never, 1=former, 2=current, blank=unknown."),
    "past_history": ("history", "Past medical/cancer history text."),
    "diagnosis": ("diagnosis", "Diagnosis text."),
    "diagnosis_date": ("diagnosis", "Diagnosis date."),
    "sample_date": ("diagnosis", "Urine/sample collection date."),
    "surgery_date": ("diagnosis", "Surgery date where available."),
    "pathology": ("cancer", "Pathology text."),
    "stage": ("cancer", "Overall cancer stage where available."),
    "tnm": ("cancer", "Combined TNM string."),
    "t_stage": ("cancer", "T stage."),
    "n_stage": ("cancer", "N stage."),
    "m_stage": ("cancer", "M stage."),
    "metastasis": ("cancer", "Metastasis indicator/text."),
    "treatment": ("treatment", "Treatment text."),
    "fasting": ("sample_context", "Clinical fasting value if present."),
    "surgery_name": ("treatment", "Surgery/procedure name."),
    "chemo_date": ("treatment", "Chemotherapy date."),
    "treatment_detail": ("treatment", "Additional treatment detail."),
    "sample_timing": ("sample_context", "control/pre-op/peri-op/post-op derived timing."),
}

LAB_COLUMNS = {
    "wbc", "rbc", "hb", "hct", "platelet", "neutrophil_pct", "lymphocyte_pct",
    "ast", "alt", "alp", "ggt", "bun", "creatinine", "uric_acid", "glucose",
    "total_protein", "albumin", "total_bilirubin", "ldh", "calcium", "hs_crp",
    "sodium", "potassium", "chloride", "total_cholesterol", "triglyceride",
    "hdl_c", "ldl_c", "hba1c", "afp", "cea", "ca19_9", "psa", "ua_sg",
    "ua_ph", "ua_protein", "ua_glucose", "ua_blood",
}


@dataclass(frozen=True)
class FileRecord:
    source: Path
    dest: Path
    role: str
    clinical_stage: str
    analysis_use: str
    note: str = ""


def should_skip(path: Path) -> bool:
    if any(part in SKIP_DIRS for part in path.parts):
        return True
    if path.name in SKIP_NAMES:
        return True
    if path.name.endswith("Zone.Identifier"):
        return True
    if path.suffix.lower() in SKIP_SUFFIXES:
        return True
    return False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(record: FileRecord) -> str:
    record.dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(record.source, record.dest)
    return sha256_file(record.dest)


def csv_shape(path: Path) -> tuple[int | str, int | str, str]:
    if path.suffix.lower() != ".csv":
        return ("", "", "")
    encodings = ["utf-8-sig", "utf-8", "cp949", "euc-kr", "latin1"]
    last_error = ""
    for encoding in encodings:
        try:
            with path.open("r", newline="", encoding=encoding) as handle:
                reader = csv.reader(handle)
                header = next(reader, [])
                rows = sum(1 for _ in reader)
            return (rows, len(header), encoding)
        except Exception as exc:  # noqa: BLE001 - registry should record fallback attempts
            last_error = type(exc).__name__
    return ("", "", f"unreadable:{last_error}")


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    for encoding in ["utf-8-sig", "utf-8", "cp949", "euc-kr", "latin1"]:
        try:
            with path.open("r", newline="", encoding=encoding) as handle:
                return list(csv.DictReader(handle))
        except UnicodeDecodeError:
            continue
    with path.open("r", newline="", encoding="latin1") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def add_tree(
    records: list[FileRecord],
    source_dir: Path,
    dest_dir: Path,
    role: str,
    clinical_stage: str,
    analysis_use: str,
    note: str = "",
) -> None:
    if not source_dir.exists():
        return
    for root, dirs, files in os.walk(source_dir):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for name in sorted(files):
            source = Path(root) / name
            relative = source.relative_to(source_dir)
            if should_skip(source.relative_to(SOURCE_CLINICAL)):
                continue
            records.append(
                FileRecord(
                    source=source,
                    dest=dest_dir / relative,
                    role=role,
                    clinical_stage=clinical_stage,
                    analysis_use=analysis_use,
                    note=note,
                )
            )


def add_file(
    records: list[FileRecord],
    source: Path,
    dest: Path,
    role: str,
    clinical_stage: str,
    analysis_use: str,
    note: str = "",
) -> None:
    if source.exists() and not should_skip(source):
        records.append(FileRecord(source, dest, role, clinical_stage, analysis_use, note))


def build_records(dest: Path) -> list[FileRecord]:
    records: list[FileRecord] = []

    for disease_dir in RAW_DISEASE_DIRS:
        add_tree(
            records,
            SOURCE_CLINICAL / disease_dir,
            dest / "01_raw_hospital_excels" / disease_dir,
            "raw_hospital_source",
            "raw",
            "source_of_truth",
            "Original hospital/clinical source files.",
        )

    for filename in RAW_OVERVIEW_FILES:
        add_file(
            records,
            SOURCE_CLINICAL / filename,
            dest / "01_raw_hospital_excels" / "00_study_overview" / filename,
            "raw_study_overview",
            "raw",
            "reference",
            "Study overview/source workbook.",
        )

    for rel in STANDARDIZED_FILES:
        add_file(
            records,
            SOURCE_CLINICAL / rel,
            dest / "02_standardized_master" / Path(rel).name,
            "standardized_master",
            "standardized",
            "primary_all_patient_table",
            "Use for full-cohort review and reprocessing.",
        )

    for source in sorted((SOURCE_CLINICAL / "standardized").glob("*_clinical_standardized.csv")):
        if source.name == "all_clinical_standardized.csv":
            continue
        add_file(
            records,
            source,
            dest / "02_standardized_master" / "by_group" / source.name,
            "standardized_group_table",
            "standardized",
            "group_level_review",
            "Disease-group standardized table.",
        )

    for rel in CLEAN_FILES:
        add_file(
            records,
            SOURCE_CLINICAL / rel,
            dest / "03_exclusion_and_clean" / Path(rel).name,
            "clean_or_exclusion_table",
            "clean",
            "default_modeling_or_exclusion_review",
            "Clean table, exclusion registry, or secondary master with exclusion columns.",
        )

    for source in sorted(SOURCE_EXCLUSIONS.glob("*.csv")):
        add_file(
            records,
            source,
            dest / "03_exclusion_and_clean" / "phase_x_external" / source.name,
            "phase_x_exclusion_source",
            "clean",
            "exclusion_review",
            "External/previous Phase X exclusion list.",
        )

    for source in sorted((SOURCE_CLINICAL / "by_cancer").glob("*.xlsx")):
        add_file(
            records,
            source,
            dest / "04_analysis_ready_by_cancer" / source.name,
            "analysis_ready_by_cancer_workbook",
            "analysis_ready",
            "manual_review",
            "Workbook with All, Definite_Excluded, and Clean sheets.",
        )

    for rel in SERS_LINKAGE_FILES:
        add_file(
            records,
            SOURCE_CLINICAL / rel,
            dest / "05_sers_linkage" / Path(rel).name,
            "sers_clinical_linkage",
            "sers_linkage",
            "sers_clinical_analysis",
            "Derived table linking SERS features to clinical records.",
        )

    for rel in QC_AND_MAPPING_FILES:
        add_file(
            records,
            SOURCE_CLINICAL / rel,
            dest / "06_column_mapping_and_qc" / Path(rel).name,
            "column_mapping_or_qc",
            "qc_documentation",
            "documentation",
            "Column mapping, standardization report, completeness, or QC artifact.",
        )

    for rel in CORE_SCRIPT_FILES:
        add_file(
            records,
            SOURCE_CLINICAL / rel,
            dest / "07_reproducibility_scripts" / Path(rel).name,
            "reproducibility_script",
            "script",
            "reproduce_or_inspect",
            "Core script needed to reproduce or inspect clinical processing.",
        )

    return records


def removed_reason(path: Path, kept_sources: set[Path]) -> str:
    if path in kept_sources:
        return ""
    text = str(path.relative_to(SOURCE_CLINICAL))
    name = path.name
    if should_skip(path.relative_to(SOURCE_CLINICAL)):
        return "cache_os_metadata"
    lowered = text.lower()
    if "backup" in lowered:
        return "backup_previous_version"
    if "powerbi" in lowered:
        return "powerbi_export_not_needed_for_analysis_folder"
    if path.suffix.lower() == ".db" or text.startswith("sql/"):
        return "database_export_not_needed_for_analysis_folder"
    if name.startswith("clinical_unified_"):
        return "previous_unified_export"
    if name.startswith("master_clinical.backup"):
        return "backup_previous_version"
    if name in {
        "ingest_clinical_to_sqlite.py",
        "ingest_to_postgres.py",
        "prepare_powerbi_patient_group_files.py",
        "prepare_powerbi_smc_source_files.py",
        "prepare_powerbi_workbooks.py",
    }:
        return "auxiliary_export_script_not_needed_for_analysis_folder"
    return "not_required_for_analysis_ready_lineage"


def write_removed_manifest(dest: Path, kept_sources: set[Path]) -> None:
    rows: list[dict[str, object]] = []
    for root, dirs, files in os.walk(SOURCE_CLINICAL):
        dirs.sort()
        files.sort()
        for name in files:
            source = Path(root) / name
            reason = removed_reason(source, kept_sources)
            if reason:
                rows.append(
                    {
                        "original_path": str(source),
                        "relative_source_path": str(source.relative_to(SOURCE_CLINICAL)),
                        "size_bytes": source.stat().st_size,
                        "reason_not_in_clean_copy": reason,
                    }
                )
    write_csv(
        dest / "99_manifests" / "removed_from_organized_copy.csv",
        rows,
        ["original_path", "relative_source_path", "size_bytes", "reason_not_in_clean_copy"],
    )


def write_registry(dest: Path, records: list[FileRecord], checksums: dict[Path, str]) -> None:
    rows: list[dict[str, object]] = []
    for record in records:
        row_count, column_count, encoding = csv_shape(record.dest)
        rows.append(
            {
                "original_path": str(record.source),
                "organized_path": str(record.dest),
                "relative_organized_path": str(record.dest.relative_to(dest)),
                "role": record.role,
                "clinical_stage": record.clinical_stage,
                "analysis_use": record.analysis_use,
                "rows": row_count,
                "columns": column_count,
                "encoding": encoding,
                "size_bytes": record.dest.stat().st_size,
                "sha256": checksums[record.dest],
                "note": record.note,
            }
        )
    write_csv(
        dest / "99_manifests" / "clinical_file_registry.csv",
        rows,
        [
            "original_path",
            "organized_path",
            "relative_organized_path",
            "role",
            "clinical_stage",
            "analysis_use",
            "rows",
            "columns",
            "encoding",
            "size_bytes",
            "sha256",
            "note",
        ],
    )


def write_lineage(dest: Path) -> None:
    rows = [
        {
            "step": 1,
            "stage": "raw",
            "directory": "01_raw_hospital_excels",
            "primary_files": "disease-group hospital Excel/CSV sources",
            "description": "Original clinical source-of-truth grouped by disease/control cohort.",
            "default_use": "Use only when auditing or regenerating standardized tables.",
        },
        {
            "step": 2,
            "stage": "standardized",
            "directory": "02_standardized_master",
            "primary_files": "all_clinical_standardized.csv; 전체환자_임상정보_정규화.xlsx",
            "description": "Unified 68-column clinical table before definitive exclusions.",
            "default_use": "Use for full-cohort review and preprocessing changes.",
        },
        {
            "step": 3,
            "stage": "clean",
            "directory": "03_exclusion_and_clean",
            "primary_files": "clean_clinical_data.csv; sample_exclusion_registry.csv",
            "description": "Default clean table and exclusion registry.",
            "default_use": "Use clean_clinical_data.csv for default modeling clinical covariates.",
        },
        {
            "step": 4,
            "stage": "analysis_ready",
            "directory": "04_analysis_ready_by_cancer",
            "primary_files": "by_cancer/*.xlsx",
            "description": "Cancer/control workbook views with All, Definite_Excluded, and Clean sheets.",
            "default_use": "Use for manual cancer-specific review.",
        },
        {
            "step": 5,
            "stage": "sers_linkage",
            "directory": "05_sers_linkage",
            "primary_files": "sers_clinical_merged.csv; sers_patient_features.csv",
            "description": "Derived SERS-clinical linkage outputs.",
            "default_use": "Use for SERS-clinical correlation and feature review.",
        },
    ]
    write_csv(
        dest / "99_manifests" / "clinical_lineage.csv",
        rows,
        ["step", "stage", "directory", "primary_files", "description", "default_use"],
    )


def write_group_counts(dest: Path) -> None:
    all_rows = read_csv_dicts(dest / "02_standardized_master" / "all_clinical_standardized.csv")
    clean_rows = read_csv_dicts(dest / "03_exclusion_and_clean" / "clean_clinical_data.csv")
    registry_rows = read_csv_dicts(dest / "03_exclusion_and_clean" / "sample_exclusion_registry.csv")

    all_counts = Counter(row.get("disease_group", "") or "<blank>" for row in all_rows)
    clean_counts = Counter(row.get("disease_group", "") or "<blank>" for row in clean_rows)
    grouped_registry: dict[str, Counter] = defaultdict(Counter)
    for row in registry_rows:
        group = row.get("disease_group", "") or "<blank>"
        category = row.get("exclusion_category", "") or "<blank>"
        grouped_registry[group][category] += 1

    groups = sorted(set(all_counts) | set(clean_counts) | set(grouped_registry))
    rows = []
    for group in groups:
        rows.append(
            {
                "disease_group": group,
                "all_standardized_n": all_counts[group],
                "clean_n": clean_counts[group],
                "definite_exclude_n": grouped_registry[group]["definite_exclude"],
                "review_periop_n": grouped_registry[group]["review_periop"],
                "review_undefined_n": grouped_registry[group]["review_undefined"],
                "registry_total_n": sum(grouped_registry[group].values()),
            }
        )
    write_csv(
        dest / "99_manifests" / "clinical_group_counts.csv",
        rows,
        [
            "disease_group",
            "all_standardized_n",
            "clean_n",
            "definite_exclude_n",
            "review_periop_n",
            "review_undefined_n",
            "registry_total_n",
        ],
    )


def write_column_dictionary(dest: Path) -> None:
    rows = read_csv_dicts(dest / "02_standardized_master" / "all_clinical_standardized.csv")
    columns = list(rows[0].keys()) if rows else []
    output = []
    for column in columns:
        if column in COLUMN_DESCRIPTIONS:
            category, description = COLUMN_DESCRIPTIONS[column]
        elif column in LAB_COLUMNS:
            category, description = ("lab_or_marker", "Clinical lab, tumor marker, or urinalysis value.")
        else:
            category, description = ("other", "Standardized clinical column.")
        non_missing = sum(1 for row in rows if (row.get(column) or "").strip() != "")
        output.append(
            {
                "column": column,
                "category": category,
                "description": description,
                "non_missing_in_all_standardized": non_missing,
                "total_rows": len(rows),
            }
        )
    write_csv(
        dest / "99_manifests" / "clinical_column_dictionary.csv",
        output,
        ["column", "category", "description", "non_missing_in_all_standardized", "total_rows"],
    )


def write_readme(dest: Path) -> None:
    readme_dir = dest / "00_README"
    readme_dir.mkdir(parents=True, exist_ok=True)
    content = f"""# Clinical Metadata Map

Generated: {datetime.now().isoformat(timespec="seconds")}

This folder is an analysis-focused clinical metadata copy. The original source
tree remains unchanged at:

`{SOURCE_CLINICAL}`

## What To Use

- Default modeling clinical table: `03_exclusion_and_clean/clean_clinical_data.csv`
- Full standardized cohort: `02_standardized_master/all_clinical_standardized.csv`
- Current normalized workbook: `02_standardized_master/전체환자_임상정보_정규화.xlsx`
- Exclusion tracking: `03_exclusion_and_clean/sample_exclusion_registry.csv`
- Cancer-specific manual review: `04_analysis_ready_by_cancer/*.xlsx`
- SERS-clinical linkage outputs: `05_sers_linkage/*.csv`

## Folder Lineage

1. `01_raw_hospital_excels`: original hospital/source clinical files.
2. `02_standardized_master`: unified 68-column standardized clinical data.
3. `03_exclusion_and_clean`: exclusion registry and default clean table.
4. `04_analysis_ready_by_cancer`: by-cancer review workbooks.
5. `05_sers_linkage`: derived SERS-clinical linkage tables.
6. `06_column_mapping_and_qc`: mapping docs, QC summaries, changelog.
7. `07_reproducibility_scripts`: core scripts for regeneration/inspection.
8. `99_manifests`: registry, lineage, counts, column dictionary, removed-file log.

PowerBI exports, database exports, backups, virtualenv/cache files, and old
intermediate exports were intentionally removed from this organized copy to keep
the clinical data understandable.
"""
    (readme_dir / "CLINICAL_DATA_MAP.md").write_text(content, encoding="utf-8")


def write_top_level_readme() -> None:
    readme_dir = ORGANIZED_ROOT / "00_README"
    readme_dir.mkdir(parents=True, exist_ok=True)
    content = f"""# Organized SERS + Clinical Data

Updated: {datetime.now().isoformat(timespec="seconds")}

Source of truth:

`/Users/ian/Downloads/data/data`

Organized analysis copy:

`{ORGANIZED_ROOT}`

This is no longer a full archival copy. It is an analysis-oriented organized
view. The original source tree is preserved separately.

## Top-Level Structure

- `01_clinical_metadata`: cleaned clinical lineage folder.
- `02_sers_primary_pooled_acquisition`: earlier broad grouped SERS acquisition.
- `03_sers_date_lot_balanced_acquisition`: later date/lot-balanced SERS acquisition.
- `04_machine_repeatability_tests`: machine repeatability/comparison data.
- `05_not_yet_analyzed`: copied datasets intentionally excluded for now.
- `99_manifests`: original SERS copy manifests.

The previous catch-all supporting-output folder and messy clinical metadata copy
are not treated as analysis inputs. Clinical PowerBI/DB/backups are omitted from
the cleaned clinical folder.
"""
    (readme_dir / "DATASET_MAP.md").write_text(content, encoding="utf-8")


def write_top_level_dataset_registry() -> None:
    TOP_LEVEL_MANIFESTS.mkdir(parents=True, exist_ok=True)
    content = f"""# Dataset registry for organized SERS + clinical analysis copy.
source_root: "/Users/ian/Downloads/data/data"
organized_root: "{ORGANIZED_ROOT}"
updated: "{datetime.now().isoformat(timespec="seconds")}"

copy_policy:
  source_tree_unchanged: true
  organized_copy_is_analysis_view: true
  pruned_from_organized_copy:
    - clinical PowerBI exports
    - clinical database exports
    - clinical backups and older unified exports
    - virtualenv/cache/OS metadata
    - catch-all supporting output folder from the first full copy

top_level_directories:
  clinical_metadata:
    path: "01_clinical_metadata"
    description: "Clean clinical lineage: raw source, standardized master, exclusion/clean, by-cancer review, SERS linkage."
  primary_pooled_acquisition:
    path: "02_sers_primary_pooled_acquisition"
    description: "Earlier grouped SERS acquisition without explicit date/lot balancing."
  date_lot_balanced_acquisition:
    path: "03_sers_date_lot_balanced_acquisition"
    description: "Later date-wise SERS acquisition across groups for date/lot/instrument effect control."
  machine_repeatability_tests:
    path: "04_machine_repeatability_tests"
    description: "Machine repeatability and comparison data; not default supervised clinical training data."
  not_yet_analyzed:
    path: "05_not_yet_analyzed"
    description: "Copied datasets intentionally excluded until enabled."

clinical_defaults:
  default_modeling_table: "01_clinical_metadata/03_exclusion_and_clean/clean_clinical_data.csv"
  full_standardized_table: "01_clinical_metadata/02_standardized_master/all_clinical_standardized.csv"
  normalized_workbook: "01_clinical_metadata/02_standardized_master/전체환자_임상정보_정규화.xlsx"
  exclusion_registry: "01_clinical_metadata/03_exclusion_and_clean/sample_exclusion_registry.csv"

sers_sample_rules:
  non_fasting_token: "NF"
  post_operation_tokens: ["Po.", "PO"]
  qc_reference_tokens: ["Blank", "Ref_PS", "Ref_Si", "PS", "Si", "Calibration"]
"""
    (TOP_LEVEL_MANIFESTS / "dataset_registry.yaml").write_text(content, encoding="utf-8")


def refresh_top_level_manifests() -> None:
    TOP_LEVEL_MANIFESTS.mkdir(parents=True, exist_ok=True)
    stale_files = [
        "source_inventory.csv",
        "copy_map.csv",
        "checksums_sha256.csv",
        "modeling_manifest.csv",
        "skipped_files.csv",
    ]
    for name in stale_files:
        path = TOP_LEVEL_MANIFESTS / name
        if path.exists():
            path.unlink()

    inventory_rows: list[dict[str, object]] = []
    for root, dirs, files in os.walk(ORGANIZED_ROOT):
        dirs[:] = sorted(d for d in dirs if d != STAGING_DEST.name)
        for name in sorted(files):
            path = Path(root) / name
            relative = path.relative_to(ORGANIZED_ROOT)
            if relative.parts and relative.parts[0] == "99_manifests":
                continue
            stat = path.stat()
            inventory_rows.append(
                {
                    "relative_path": str(relative),
                    "top_level": relative.parts[0],
                    "size_bytes": stat.st_size,
                    "extension": path.suffix.lower(),
                    "mtime_iso": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                }
            )

    write_csv(
        TOP_LEVEL_MANIFESTS / "current_organized_inventory.csv",
        inventory_rows,
        ["relative_path", "top_level", "size_bytes", "extension", "mtime_iso"],
    )

    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"file_count": 0, "size_bytes": 0})
    for row in inventory_rows:
        top = str(row["top_level"])
        counts[top]["file_count"] += 1
        counts[top]["size_bytes"] += int(row["size_bytes"])
    count_rows = [
        {"top_level": top, "file_count": data["file_count"], "size_bytes": data["size_bytes"]}
        for top, data in sorted(counts.items())
    ]
    write_csv(
        TOP_LEVEL_MANIFESTS / "current_top_level_counts.csv",
        count_rows,
        ["top_level", "file_count", "size_bytes"],
    )
    write_top_level_dataset_registry()


def validate(dest: Path) -> None:
    all_path = dest / "02_standardized_master" / "all_clinical_standardized.csv"
    clean_path = dest / "03_exclusion_and_clean" / "clean_clinical_data.csv"
    registry_path = dest / "03_exclusion_and_clean" / "sample_exclusion_registry.csv"
    workbook_path = dest / "02_standardized_master" / "전체환자_임상정보_정규화.xlsx"

    all_shape = csv_shape(all_path)
    clean_shape = csv_shape(clean_path)
    registry_shape = csv_shape(registry_path)
    if all_shape[:2] != (1782, 68):
        raise RuntimeError(f"Unexpected all_clinical_standardized.csv shape: {all_shape[:2]}")
    if clean_shape[:2] != (1560, 68):
        raise RuntimeError(f"Unexpected clean_clinical_data.csv shape: {clean_shape[:2]}")
    if registry_shape[:2] != (913, 15):
        raise RuntimeError(f"Unexpected sample_exclusion_registry.csv shape: {registry_shape[:2]}")
    if not workbook_path.exists():
        raise RuntimeError(f"Missing normalized workbook: {workbook_path}")

    registry_rows = read_csv_dicts(registry_path)
    category_counts = Counter(row.get("exclusion_category", "") for row in registry_rows)
    expected = {
        "definite_exclude": 222,
        "review_periop": 676,
        "review_undefined": 15,
    }
    for category, expected_count in expected.items():
        if category_counts[category] != expected_count:
            raise RuntimeError(
                f"Unexpected exclusion count for {category}: {category_counts[category]}"
            )

    by_cancer_files = list((dest / "04_analysis_ready_by_cancer").glob("*.xlsx"))
    if len(by_cancer_files) != 8:
        raise RuntimeError(f"Expected 8 by-cancer workbooks, found {len(by_cancer_files)}")


def replace_final_with_staging() -> None:
    if FINAL_DEST.exists():
        shutil.rmtree(FINAL_DEST)
    STAGING_DEST.rename(FINAL_DEST)


def remove_empty_or_unneeded_supporting_folder() -> None:
    supporting = ORGANIZED_ROOT / "06_supporting_or_previous_outputs"
    if supporting.exists():
        shutil.rmtree(supporting)


def remove_os_metadata_from_organized_copy() -> None:
    for path in ORGANIZED_ROOT.rglob(".DS_Store"):
        if path.is_file():
            path.unlink()


def main() -> int:
    if not SOURCE_CLINICAL.exists():
        raise RuntimeError(f"Missing source clinical folder: {SOURCE_CLINICAL}")
    if STAGING_DEST.exists():
        shutil.rmtree(STAGING_DEST)
    STAGING_DEST.mkdir(parents=True, exist_ok=True)

    records = build_records(STAGING_DEST)
    checksums: dict[Path, str] = {}
    for record in records:
        checksums[record.dest] = copy_file(record)

    kept_sources = {record.source for record in records}
    write_registry(STAGING_DEST, records, checksums)
    write_removed_manifest(STAGING_DEST, kept_sources)
    write_lineage(STAGING_DEST)
    write_group_counts(STAGING_DEST)
    write_column_dictionary(STAGING_DEST)
    write_readme(STAGING_DEST)
    validate(STAGING_DEST)

    replace_final_with_staging()
    remove_empty_or_unneeded_supporting_folder()
    remove_os_metadata_from_organized_copy()
    write_top_level_readme()
    refresh_top_level_manifests()

    print(f"Clinical metadata organized at: {FINAL_DEST}")
    print(f"Kept files: {len(records)}")
    print(f"Removed/omitted source clinical files recorded in: {FINAL_DEST / '99_manifests' / 'removed_from_organized_copy.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
