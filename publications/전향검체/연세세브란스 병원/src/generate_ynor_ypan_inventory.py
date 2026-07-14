#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
# How to run:
# uv run publications/전향검체/연세세브란스\ 병원/src/generate_ynor_ypan_inventory.py
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Final

from ynor_ypan_report import ReportData, write_report

REPO: Final = Path(__file__).resolve().parents[4]
OUT: Final = REPO / "publications" / "전향검체" / "연세세브란스 병원"
TABLES: Final = OUT / "tables"
MAPPING: Final = TABLES / "yonsei_clinical_data_mapping.csv"
CLINICAL: Final = REPO / "data" / "clinical_data" / "standardized"
PRIMARY_ROOTS: Final = {
    "YNOR": REPO / "data" / "raw_data" / "12. Y-Normal (YNOR)",
    "YPAN": REPO / "data" / "raw_data" / "10-3. Y-Pancreatic cancer (YPAN)",
}
REACQUIRED_ROOT: Final = REPO / "data" / "임상데이터"
FILE_RE: Final = re.compile(r"^(YNOR|YPAN)\s+(\d+)_([1-5]|ave)\.CSV$", re.IGNORECASE)
REPEAT_SUBJECT: Final = "YNOR_21"
REPEAT_PRIMARY: Final = "YNOR_48"
PRIVATE_CLINICAL_FIELDS: Final = {
    "patient_id",
    "source_file",
    "disease_group",
    "past_history",
    "diagnosis_date",
    "sample_date",
    "surgery_date",
    "chemo_date",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def scan_acquisitions(root: Path) -> dict[tuple[str, str], set[str]]:
    found: dict[tuple[str, str], set[str]] = {}
    for path in root.rglob("*.CSV"):
        match = FILE_RE.match(path.name)
        if match is None:
            continue
        group, sample_id, replicate = match.groups()
        found.setdefault((group.upper(), str(int(sample_id))), set()).add(replicate.lower())
    return found


def primary_acquisitions() -> dict[tuple[str, str], set[str]]:
    found: dict[tuple[str, str], set[str]] = {}
    for group, root in PRIMARY_ROOTS.items():
        for key, replicates in scan_acquisitions(root).items():
            if key[0] == group:
                found[key] = replicates
    return found


def case_id(group: str, sample_id: str, primary: dict[tuple[str, str], set[str]]) -> str:
    ids = sorted((int(key[1]) for key in primary if key[0] == group))
    prefix = "YON-C" if group == "YNOR" else "YON-P"
    return f"{prefix}-{ids.index(int(sample_id)) + 1:03d}"


def clinical_sources() -> dict[str, list[dict[str, str]]]:
    return {
        group: read_rows(CLINICAL / f"{group}_clinical_standardized.csv")
        for group in ("YNOR", "YPAN")
    }


def build_acquisition_rows(
    primary: dict[tuple[str, str], set[str]],
    reacquired: dict[tuple[str, str], set[str]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for group, sample_id in sorted(primary, key=lambda key: (key[0], int(key[1]))):
        subject_case_id = case_id(group, sample_id, primary)
        for acquisition_type, source in (("primary", primary), ("reacquired", reacquired)):
            replicates = source[(group, sample_id)]
            rows.append(
                {
                    "subject_case_id": subject_case_id,
                    "source_group": group,
                    "solum_label": f"{group} {sample_id}",
                    "acquisition_type": acquisition_type,
                    "n_replicates": str(len(replicates & {"1", "2", "3", "4", "5"})),
                    "has_average_file": "yes" if "ave" in replicates else "no",
                    "used_in_primary_model": "yes" if acquisition_type == "primary" else "no",
                    "legacy_processed_label": (
                        "YNOR 21"
                        if acquisition_type == "reacquired"
                        and group == "YNOR"
                        and sample_id == "48"
                        else ""
                    ),
                }
            )
    return rows


def direct_mapping(mapping: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {
        row["solum_label"]: row
        for row in mapping
        if row["clinical_mapping_status"] == "mapped" and row["subject_id"] != REPEAT_SUBJECT
    }


def year(value: str) -> str:
    match = re.search(r"(?:19|20)\d{2}", value)
    return match.group(0) if match is not None else ""


def build_clinical_rows(
    sources: dict[str, list[dict[str, str]]],
    mapping: list[dict[str, str]],
    primary: dict[tuple[str, str], set[str]],
) -> list[dict[str, str]]:
    linked = direct_mapping(mapping)
    rows: list[dict[str, str]] = []
    clinical_only_index = 0
    for group in ("YNOR", "YPAN"):
        for clinical in sources[group]:
            mapped = linked.get(clinical["patient_id"].replace("_", " "))
            if mapped is None:
                clinical_only_index += 1
                subject_case_id = f"YON-CLINICAL-ONLY-{clinical_only_index:03d}"
                solum_label = ""
                link_status = "clinical_only"
            else:
                subject_case_id = case_id(group, mapped["sample_id"], primary)
                solum_label = mapped["solum_label"]
                link_status = "linked_to_primary_spectrum"
            row = {
                "subject_case_id": subject_case_id,
                "source_group": group,
                "solum_label": solum_label,
                "spectrum_link_status": link_status,
                "analysis_stage": mapped["stage"] if mapped is not None else "",
                "analysis_stage_source": mapped["stage_source"] if mapped is not None else "",
                "analysis_stage_group": mapped["stage_1_2_or_3_4"] if mapped is not None else "",
                "diagnosis_year": year(clinical["diagnosis_date"]),
                "sample_year": year(clinical["sample_date"]),
                "surgery_year": year(clinical["surgery_date"]),
                "chemo_year": year(clinical["chemo_date"]),
            }
            row.update(
                {
                    key: value
                    for key, value in clinical.items()
                    if key not in PRIVATE_CLINICAL_FIELDS
                }
            )
            rows.append(row)
    return rows


def build_processed_rows(
    mapping: list[dict[str, str]],
    primary: dict[tuple[str, str], set[str]],
) -> list[dict[str, str]]:
    primary_case = case_id("YNOR", "48", primary)
    rows: list[dict[str, str]] = []
    for index, mapped in enumerate(mapping, start=1):
        repeat = mapped["subject_id"] == REPEAT_SUBJECT
        subject_case_id = (
            primary_case
            if repeat
            else case_id(mapped["source_group"], mapped["sample_id"], primary)
        )
        rows.append(
            {
                "acquisition_case_id": f"YON-A-{index:03d}",
                "subject_case_id": subject_case_id,
                "source_group": mapped["source_group"],
                "solum_label": mapped["solum_label"],
                "acquisition_role": "repeat_acquisition" if repeat else "primary_acquisition",
                "linked_primary_case_id": primary_case if repeat else "",
                "clinical_mapping_basis": "linked_repeat" if repeat else "direct",
                "analysis_included": "no" if repeat else "yes",
                "n_replicates": mapped["n_replicates"],
            }
        )
    return rows


def build_completeness(sources: dict[str, list[dict[str, str]]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for group, clinical_rows in sources.items():
        for field in clinical_rows[0]:
            if field in {"patient_id", "source_file", "disease_group"}:
                continue
            present = sum(row[field].strip() != "" for row in clinical_rows)
            rows.append(
                {
                    "source_group": group,
                    "clinical_field": field,
                    "nonmissing": str(present),
                    "missing": str(len(clinical_rows) - present),
                    "completeness_pct": f"{100 * present / len(clinical_rows):.1f}",
                }
            )
    return rows


def group_inventory(
    sources: dict[str, list[dict[str, str]]],
    primary: dict[tuple[str, str], set[str]],
    reacquired: dict[tuple[str, str], set[str]],
    processed: list[dict[str, str]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for group in ("YNOR", "YPAN"):
        rows.append(
            {
                "source_group": group,
                "clinical_records": str(len(sources[group])),
                "primary_spectrum_subjects": str(sum(key[0] == group for key in primary)),
                "primary_spectrum_replicates": str(
                    sum(
                        len(reps & {"1", "2", "3", "4", "5"})
                        for key, reps in primary.items()
                        if key[0] == group
                    )
                ),
                "reacquired_subjects": str(sum(key[0] == group for key in reacquired)),
                "reacquired_replicates": str(
                    sum(
                        len(reps & {"1", "2", "3", "4", "5"})
                        for key, reps in reacquired.items()
                        if key[0] == group
                    )
                ),
                "processed_acquisitions": str(
                    sum(row["source_group"] == group for row in processed)
                ),
                "primary_model_subjects": str(
                    sum(
                        row["source_group"] == group and row["analysis_included"] == "yes"
                        for row in processed
                    )
                ),
            }
        )
    return rows


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    mapping = read_rows(MAPPING)
    sources = clinical_sources()
    primary = primary_acquisitions()
    reacquired = scan_acquisitions(REACQUIRED_ROOT)
    acquisitions = build_acquisition_rows(primary, reacquired)
    processed = build_processed_rows(mapping, primary)
    clinical = build_clinical_rows(sources, mapping, primary)
    inventory = group_inventory(sources, primary, reacquired, processed)
    write_rows(TABLES / "yonsei_ynor_ypan_group_inventory.csv", inventory)
    write_rows(TABLES / "yonsei_ynor_ypan_acquisition_inventory.csv", acquisitions)
    write_rows(TABLES / "yonsei_ynor_ypan_processed_inventory.csv", processed)
    write_rows(TABLES / "yonsei_ynor_ypan_clinical_records.csv", clinical)
    write_rows(TABLES / "yonsei_ynor_ypan_clinical_completeness.csv", build_completeness(sources))
    write_report(ReportData(sources, inventory, mapping, OUT / "YNOR_YPAN_DATA_SUMMARY.md"))
    print(f"Wrote YNOR/YPAN inventory to {OUT}")


if __name__ == "__main__":
    main()
