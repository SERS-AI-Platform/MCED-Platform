#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pandas>=2.0",
#     "openpyxl>=3.1",
# ]
# ///

# ─── How to run ───
#   python scripts/db/aecd_clinical_v7/build_staging_csv.py
#   (or: uv run scripts/db/aecd_clinical_v7/build_staging_csv.py)
# ──────────────────

"""Turn the normalized clinical workbook into a load file for ingest.clinical_master_staging.

The staging table is all-TEXT on purpose: load raw first, validate in SQL, then
transform. This script therefore does the minimum needed for a clean COPY:

1. drop rows that cannot become a sample (no solum_label) or are marked Drop,
2. resolve site_code from the label (see site_map.py),
3. normalize dates to ISO so the ``::date`` casts in the load SQL succeed,
4. collapse integral floats back to integers (pandas widens int columns holding
   NaN to float, which would write "69.0" where the workbook says 69).

Outputs, in --output-dir:
  <stem>_staging.csv   plain UTF-8 (no BOM) -- BOM would corrupt the first COPY column
  <stem>_excluded.csv  utf-8-sig, the rows this script refused, with the reason
  <stem>_sitemap.csv   utf-8-sig, label -> site_code, for review
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from site_map import site_for_label  # noqa: E402

DEFAULT_WORKBOOK = Path("/mnt/c/Users/user/Downloads/전체환자_임상정보_정규화_v7.xlsx")
DEFAULT_OUTPUT_DIR = Path("/home/user/SERS-AI/data/processed/aecd_platform_ingest")
DEFAULT_SHEET = "table"
DEFAULT_SOURCE_MAPPING = "clinical_v7_20260902"

LABEL_PATTERN = re.compile(r"^([A-Za-z.\s]+?)\s*_\s*(\d+)$")

# The workbook writes some ranges as "PRO_ 60" and others as "PRO_101" for the
# same cohort. Collapse the space so one sample has one spelling; the space in
# the "H. D._1" prefix is part of the group name and is left alone.
LABEL_SPACE_AFTER_UNDERSCORE = re.compile(r"_\s+")

DATE_COLUMNS = (
    "birth_date",
    "collection_date",
    "post_treatment_collection_date",
    "receipt_date",
    "diagnosis_date",
    "surgery_date",
    "chemo_start_date",
    "chemo_end_date",
)

# Column order of ingest.clinical_master_staging, minus the generated columns.
STAGING_COLUMNS = (
    "source_mapping", "site_code", "group", "solum_label", "source_cancer_file",
    "patient_code", "cancer_type", "sex", "age", "birth_date", "weight_kg",
    "height_cm", "bmi", "smoking_history", "drinking_history", "collection_date",
    "post_treatment_collection_date", "has_post_treatment_sample", "sample_type",
    "sample_amount", "receipt_date", "diagnosis_code", "diagnosis_name",
    "diagnosis_date", "past_history_1", "past_history_2", "past_history_3",
    "past_history_4", "past_history_5", "past_history_6", "past_history_7",
    "past_history_8", "past_history_9", "pathology_result", "histologic_type",
    "gleason", "grade group", "stage", "tnm_stage", "t_stage", "n_stage",
    "m_stage", "metastasis", "surgery_date", "surgery_name", "chemo_start_date",
    "chemo_end_date", "treatment_info",
    "post_surgery_or_chemo_urine_result_available",
    "primary_sample_timing_interpretation", "wbc", "rbc", "hb", "hct", "platelet",
    "neutrophil", "lymphocyte", "monocyte", "eosinophil", "basophil", "ast",
    "alt", "alp", "ggt", "total_bilirubin", "bun", "creatinine", "uric_acid",
    "glucose", "calcium", "sodium", "potassium", "chloride", "cholesterol",
    "triglyceride", "hba1c", "ER", "PR", "HER2", "CA 15-3", "CEA", "ca19_9",
    "ca125", "afp", "psa", "ua_sg", "ua_ph", "ua_protein", "ua_glucose",
    "ua_blood(Heme)", "ua_ketone", "ua_urobilinogen", "ua_nitrite",
    "ua_leukocyte esterase", "Microscopy_WBC", "Microscopy_RBC",
    "Microscopy_Bacteria", "source_row_number", "standardization_notes",
)

DATE_TEXT_FORMATS = ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y%m%d")


def to_text(value: object) -> str | None:
    """Render one workbook cell as staging text, or None for an empty cell."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    return text or None


def to_date_text(value: object) -> tuple[str | None, bool]:
    """Return (ISO date text, parsed_ok). Unparseable values are passed through."""
    text = to_text(value)
    if text is None:
        return None, True
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return text, True
    head = text.split(" ")[0]
    for fmt in DATE_TEXT_FORMATS:
        try:
            return datetime.strptime(head, fmt).date().isoformat(), True
        except ValueError:
            continue
    return text, False


def build(workbook: Path, sheet: str, output_dir: Path, source_mapping: str) -> int:
    frame = pd.read_excel(workbook, sheet_name=sheet)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = workbook.stem

    rows: list[dict[str, str | None]] = []
    excluded: list[dict[str, object]] = []
    site_rows: list[dict[str, object]] = []
    unparsed_dates: list[tuple[int, str, str]] = []

    for position, record in enumerate(frame.to_dict("records"), start=2):
        label = to_text(record.get("solum_label"))
        if label is not None:
            label = LABEL_SPACE_AFTER_UNDERSCORE.sub("_", label)
        group = to_text(record.get("group"))

        if label is None:
            excluded.append({"excel_row": position, "solum_label": None,
                             "group": group, "reason": "solum_label_missing"})
            continue
        if group == "Drop":
            excluded.append({"excel_row": position, "solum_label": label,
                             "group": group, "reason": "group_is_Drop"})
            continue

        match = LABEL_PATTERN.match(label)
        if match is None:
            excluded.append({"excel_row": position, "solum_label": label,
                             "group": group, "reason": "label_format_unrecognized"})
            continue

        site_code = site_for_label(match.group(1), int(match.group(2)))
        if site_code is None:
            excluded.append({"excel_row": position, "solum_label": label,
                             "group": group, "reason": "no_site_range_matched"})
            continue

        patient_code = to_text(record.get("patient_code"))
        if patient_code is None:
            excluded.append({"excel_row": position, "solum_label": label,
                             "group": group, "reason": "patient_code_missing"})
            continue

        row: dict[str, str | None] = {
            "source_mapping": source_mapping,
            "site_code": site_code,
        }
        for column in STAGING_COLUMNS:
            if column in ("source_mapping", "site_code"):
                continue
            value = label if column == "solum_label" else record.get(column)
            if column in DATE_COLUMNS:
                text, parsed = to_date_text(value)
                if not parsed:
                    unparsed_dates.append((position, column, str(text)))
            else:
                text = to_text(value)
            row[column] = text

        rows.append(row)
        site_rows.append({"solum_label": label, "group": group,
                          "patient_code": patient_code, "site_code": site_code})

    staging_path = output_dir / f"{stem}_staging.csv"
    with staging_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(STAGING_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)

    pd.DataFrame(excluded).to_csv(
        output_dir / f"{stem}_excluded.csv", index=False, encoding="utf-8-sig"
    )
    site_frame = pd.DataFrame(site_rows)
    site_frame.to_csv(
        output_dir / f"{stem}_sitemap.csv", index=False, encoding="utf-8-sig"
    )

    print(f"workbook rows          : {len(frame)}")
    print(f"staged rows            : {len(rows)}  -> {staging_path}")
    print(f"excluded rows          : {len(excluded)}")
    for reason, count in (
        pd.DataFrame(excluded)["reason"].value_counts().items() if excluded else []
    ):
        print(f"  {reason:<28} {count}")
    print(f"distinct subjects      : {site_frame.groupby(['site_code','patient_code']).ngroups}")
    print("rows per site          :")
    for site_code, count in site_frame["site_code"].value_counts().sort_index().items():
        print(f"  {site_code:<10} {count}")
    if unparsed_dates:
        print(f"UNPARSED DATE VALUES   : {len(unparsed_dates)} (passed through as text)")
        for excel_row, column, text in unparsed_dates[:20]:
            print(f"  row {excel_row} {column} = {text!r}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--sheet", default=DEFAULT_SHEET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-mapping", default=DEFAULT_SOURCE_MAPPING)
    args = parser.parse_args()
    return build(args.workbook, args.sheet, args.output_dir, args.source_mapping)


if __name__ == "__main__":
    raise SystemExit(main())
