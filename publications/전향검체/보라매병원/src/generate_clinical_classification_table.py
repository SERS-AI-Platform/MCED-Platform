#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy>=1.24",
#   "openpyxl>=3.1",
#   "scipy>=1.10",
# ]
# ///
# Run: uv run publications/전향검체/보라매병원/src/generate_clinical_classification_table.py
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Final

import numpy as np
import openpyxl

REPO: Final = Path(__file__).resolve().parents[4]
OUT: Final = REPO / "publications" / "전향검체" / "보라매병원"
TABLE_DIR: Final = OUT / "tables"
CLINICAL_XLSX: Final = REPO / "data" / "clinical_data" / "보라매 병원 임상정보.xlsx"
PREDICTION_FILES: Final = {
    "three_group": TABLE_DIR / "three_group_oof_predictions.csv",
    "screening_binary": TABLE_DIR / "screening_binary_oof_predictions.csv",
}
OUTPUT: Final = TABLE_DIR / "clinical_classification_analysis.csv"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from boramae_data import MODEL_GRID, build_boramae_subjects, load_clinical_samples  # noqa: E402

CLINICAL_FIELDS: Final = (
    "age",
    "psa",
    "gleason_score",
    "grade_group",
    "pathology_result",
    "stage",
    "ua_sg",
    "ua_ph",
    "ua_protein",
    "ua_blood",
    "ua_leukocyte",
    "creatinine",
    "bun",
    "glucose",
    "hba1c",
    "sample_type",
    "has_post_treatment_sample",
)
SOURCE_FIELD_NAMES: Final = {
    "gleason_score": "Gleason Score",
    "grade_group": "Grade Group",
}
OUTPUT_FIELDS: Final = (
    "case_id",
    "publication_group",
    "true_label",
    "grade_band",
    "age",
    "psa",
    "gleason_score",
    "grade_group",
    "pathology_result",
    "stage",
    "ua_sg",
    "ua_ph",
    "ua_protein",
    "ua_blood",
    "ua_leukocyte",
    "creatinine",
    "bun",
    "glucose",
    "hba1c",
    "sample_type",
    "has_post_treatment_sample",
    "three_group_fold",
    "three_group_pred_label",
    "three_group_prob_control",
    "three_group_prob_biopsy_negative",
    "three_group_prob_cancer",
    "three_group_is_correct",
    "three_group_error_direction",
    "screening_fold",
    "screening_pred_label",
    "screening_prob_non_cancer",
    "screening_prob_cancer",
    "screening_is_correct",
    "screening_error_direction",
    "cancer_misclassified",
)


def read_prediction_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_clinical_rows() -> dict[str, dict[str, str]]:
    workbook = openpyxl.load_workbook(CLINICAL_XLSX, data_only=True, read_only=True)
    sheet = workbook.active
    values = list(sheet.values)
    headers = [str(value) for value in values[0]]
    return {
        str(row[1]): {header: _cell(row[index]) for index, header in enumerate(headers)}
        for row in values[1:]
        if row[1] is not None
    }


def _cell(value: object) -> str:
    return "" if value is None else str(value)


def grade_band(grade: str) -> str:
    if grade in {"1", "2"}:
        return "GG1-2"
    if grade in {"3", "4", "5"}:
        return "GG3-5"
    return "Unknown"


def error_direction(true_label: str, pred_label: str) -> str:
    if true_label == pred_label:
        return "correct"
    if true_label == "Cancer" and pred_label == "Biopsy-negative":
        return "cancer_to_biopsy_negative"
    if true_label == "Cancer" and pred_label == "Control":
        return "cancer_to_control"
    if true_label == "Cancer" and pred_label == "Non-cancer":
        return "cancer_to_non_cancer"
    if true_label == "Control" and pred_label == "Biopsy-negative":
        return "control_to_biopsy_negative"
    if true_label == "Biopsy-negative" and pred_label == "Control":
        return "biopsy_negative_to_control"
    if true_label != "Cancer" and pred_label == "Cancer":
        return "non_cancer_to_cancer"
    return "other_error"


def build_rows() -> list[dict[str, str]]:
    subjects = build_boramae_subjects(load_clinical_samples(), np.load(MODEL_GRID))
    clinical = read_clinical_rows()
    predictions = {name: read_prediction_rows(path) for name, path in PREDICTION_FILES.items()}
    if any(len(rows) != len(subjects) for rows in predictions.values()):
        raise RuntimeError("Prediction row count does not match subject count")
    output: list[dict[str, str]] = []
    for index, subject in enumerate(subjects):
        sample = clinical[subject.sample.label]
        three = predictions["three_group"][index]
        screening = predictions["screening_binary"][index]
        true_label = three["true_label"]
        three_pred = three["pred_label"]
        screening_pred = screening["pred_label"]
        row = {
            "case_id": f"BRM-{int(subject.sample.sample_no):03d}",
            "publication_group": subject.sample.group,
            "true_label": true_label,
            "grade_band": grade_band(sample[SOURCE_FIELD_NAMES["grade_group"]]),
            **{field: sample[SOURCE_FIELD_NAMES.get(field, field)] for field in CLINICAL_FIELDS},
            "three_group_fold": three["fold"],
            "three_group_pred_label": three_pred,
            "three_group_prob_control": three["prob_Control"],
            "three_group_prob_biopsy_negative": three["prob_Biopsy-negative"],
            "three_group_prob_cancer": three["prob_Cancer"],
            "three_group_is_correct": str(true_label == three_pred),
            "three_group_error_direction": error_direction(true_label, three_pred),
            "screening_fold": screening["fold"],
            "screening_pred_label": screening_pred,
            "screening_prob_non_cancer": screening["prob_Non-cancer"],
            "screening_prob_cancer": screening["prob_Cancer"],
            "screening_is_correct": str(screening["true_label"] == screening_pred),
            "screening_error_direction": error_direction(screening["true_label"], screening_pred),
            "cancer_misclassified": str(true_label == "Cancer" and three_pred != "Cancer"),
        }
        output.append(row)
    return output


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(build_rows())
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
