#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["openpyxl>=3.1"]
# ///
# --- How to run ---
# uv run publications/전향검체/연세세브란스\ 병원/src/generate_sharepoint_cohort_package.py
# pyright: reportImplicitRelativeImport=false
from __future__ import annotations

import csv
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Final

from yonsei_sharepoint.outputs import write_archive, write_readme, write_workbook

REPO: Final = Path(__file__).resolve().parents[4]
ROOT: Final = REPO / "publications" / "전향검체" / "연세세브란스 병원"
TABLES: Final = ROOT / "tables"
PACKAGE_NAME: Final = "2026_SERS-AI_Yonsei_Prospective_Cohort_v1.0"
PACKAGE: Final = ROOT / "sharepoint_upload" / PACKAGE_NAME
CSV_DIR: Final = PACKAGE / "csv"
WORKBOOK: Final = PACKAGE / f"{PACKAGE_NAME}.xlsx"
ARCHIVE: Final = PACKAGE.parent / f"{PACKAGE_NAME}.zip"
REACQUIRED_ROOT: Final = REPO / "data" / "임상데이터"
YNOR_TRACE: Final = (
    REPO / "results" / "raw_data_vs_clinical_reference" / "ynor_id_remapping_trace.csv"
)
FILE_RE: Final = re.compile(r"^(YNOR|YPAN)\s+(\d+)_([1-5]|ave)\.CSV$", re.IGNORECASE)
DATE_RE: Final = re.compile(r"^(\d{8})_Urine test$")

Row = dict[str, str]


class CohortDataError(RuntimeError):
    def __init__(self, missing: list[str], unexpected: list[str], invalid: list[str]) -> None:
        details = f"missing={missing}, unexpected={unexpected}, invalid={invalid}"
        super().__init__(f"reacquired measurement inventory mismatch: {details}")


def read_rows(path: Path) -> list[Row]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[Row]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def age_band(value: str) -> str:
    if not value:
        return "미상"
    decade = int(float(value)) // 10 * 10
    return f"{decade}대"


def bmi_band(value: str) -> str:
    if not value:
        return "미상"
    bmi = float(value)
    if bmi < 18.5:
        return "저체중"
    if bmi < 23:
        return "정상"
    if bmi < 25:
        return "과체중"
    return "비만"


def cohort_summary() -> list[Row]:
    source = read_rows(TABLES / "yonsei_ynor_ypan_group_inventory.csv")
    rows = [
        {
            "source_group": row["source_group"],
            "clinical_records": row["clinical_records"],
            "independent_spectrum_subjects": row["primary_spectrum_subjects"],
            "primary_replicates": row["primary_spectrum_replicates"],
            "reacquired_replicates": row["reacquired_replicates"],
            "total_spectrum_replicates": str(
                int(row["primary_spectrum_replicates"]) + int(row["reacquired_replicates"])
            ),
            "processed_acquisitions": row["processed_acquisitions"],
            "primary_model_subjects": row["primary_model_subjects"],
        }
        for row in source
    ]
    rows.append(
        {
            key: "TOTAL" if key == "source_group" else str(sum(int(row[key]) for row in rows))
            for key in rows[0]
        }
    )
    return rows


def reacquired_measurements() -> dict[tuple[str, str], tuple[str, set[str]]]:
    collected: dict[tuple[str, str], tuple[str, set[str]]] = {}
    for path in REACQUIRED_ROOT.rglob("*.CSV"):
        match = FILE_RE.match(path.name)
        if match is None or match.group(3).lower() == "ave":
            continue
        date = next(
            (date_match.group(1) for part in path.parts if (date_match := DATE_RE.match(part))),
            "",
        )
        if not date:
            continue
        key = (match.group(1).upper(), f"{match.group(1).upper()} {match.group(2)}")
        prior_date, replicates = collected.get(key, (date, set()))
        replicates.add(match.group(3))
        collected[key] = (prior_date, replicates)
    return collected


def validate_reacquired_measurements(
    measurements: dict[tuple[str, str], tuple[str, set[str]]],
) -> None:
    source = read_rows(TABLES / "yonsei_ynor_ypan_acquisition_inventory.csv")
    expected = {
        (row["source_group"], row["solum_label"])
        for row in source
        if row["acquisition_type"] == "reacquired"
    }
    actual = set(measurements)
    invalid = sorted(
        f"{group}:{label}"
        for (group, label), (date, replicates) in measurements.items()
        if len(date) != 8 or replicates != {"1", "2", "3", "4", "5"}
    )
    missing = sorted(f"{group}:{label}" for group, label in expected - actual)
    unexpected = sorted(f"{group}:{label}" for group, label in actual - expected)
    if missing or unexpected or invalid:
        raise CohortDataError(missing, unexpected, invalid)


def measurement_schedule(measurements: dict[tuple[str, str], tuple[str, set[str]]]) -> list[Row]:
    grouped: dict[tuple[str, str], list[set[str]]] = defaultdict(lambda: [set(), set()])
    for (group, label), (date, replicates) in measurements.items():
        grouped[(group, date[:6])][0].add(label)
        grouped[(group, date[:6])][1].update(f"{label}_{replicate}" for replicate in replicates)
    primary_counts = {"YNOR": (29, 145), "YPAN": (30, 150)}
    rows = [
        {
            "source_group": group,
            "acquisition_round": "1차",
            "measurement_month": "원자료 메타데이터 없음",
            "subjects": str(counts[0]),
            "replicate_spectra": str(counts[1]),
            "purpose": "primary 분석 기준",
        }
        for group, counts in primary_counts.items()
    ]
    rows.extend(
        {
            "source_group": group,
            "acquisition_round": "2차 재측정",
            "measurement_month": f"{month[:4]}-{month[4:6]}",
            "subjects": str(len(values[0])),
            "replicate_spectra": str(len(values[1])),
            "purpose": "동일 검체 반복 측정",
        }
        for (group, month), values in sorted(grouped.items())
    )
    return rows


def acquisition_inventory(measurements: dict[tuple[str, str], tuple[str, set[str]]]) -> list[Row]:
    source = read_rows(TABLES / "yonsei_ynor_ypan_acquisition_inventory.csv")
    rows: list[Row] = []
    for row in source:
        reacquired = row["acquisition_type"] == "reacquired"
        date = measurements.get((row["source_group"], row["solum_label"]), ("", set()))[0]
        rows.append(
            {
                "subject_case_id": row["subject_case_id"],
                "source_group": row["source_group"],
                "current_solum_label": row["solum_label"],
                "acquisition_round": "2차 재측정" if reacquired else "1차",
                "measurement_month": f"{date[:4]}-{date[4:6]}" if reacquired and date else "",
                "n_replicates": row["n_replicates"],
                "has_average_file": row["has_average_file"],
                "used_in_primary_model": row["used_in_primary_model"],
                "mapping_note": "2차 원라벨 YNOR 21" if row["legacy_processed_label"] else "",
            }
        )
    return rows


def subject_manifest(acquisitions: list[Row]) -> list[Row]:
    counts: dict[str, dict[str, str]] = defaultdict(dict)
    for row in acquisitions:
        counts[row["subject_case_id"]][row["acquisition_round"]] = row["n_replicates"]
    clinical = read_rows(TABLES / "yonsei_ynor_ypan_clinical_records.csv")
    return [
        {
            "subject_case_id": row["subject_case_id"],
            "source_group": row["source_group"],
            "current_solum_label": row["solum_label"],
            "cohort_role": "non-cancer" if row["source_group"] == "YNOR" else "pancreatic cancer",
            "diagnosis_group": row["diagnosis"],
            "analysis_stage_group": row["analysis_stage_group"],
            "sex": row["sex"],
            "age_band": age_band(row["age"]),
            "bmi_band": bmi_band(row["bmi"]),
            "sample_timing": row["sample_timing"],
            "spectrum_status": "임상정보만 존재"
            if row["spectrum_link_status"] == "clinical_only"
            else "1차·2차 측정 존재",
            "primary_replicates": counts[row["subject_case_id"]].get("1차", "0"),
            "reacquired_replicates": counts[row["subject_case_id"]].get("2차 재측정", "0"),
        }
        for row in clinical
    ]


def ynor_id_mapping(acquisitions: list[Row]) -> list[Row]:
    case_ids = {
        row["current_solum_label"]: row["subject_case_id"]
        for row in acquisitions
        if row["source_group"] == "YNOR" and row["acquisition_round"] == "1차"
    }
    return [
        {
            "subject_case_id": case_ids[f"YNOR {row['1차_sample_id']}"],
            "primary_label": f"YNOR {row['1차_sample_id']}",
            "original_reacquired_label": f"YNOR {row['2차_sample_id']}",
            "current_reacquired_label": f"YNOR {row['1차_sample_id']}",
            "measurement_month": f"{row['2차_측정일'][:4]}-{row['2차_측정일'][4:6]}",
            "mapping_status": row["ID_변경여부"],
            "primary_replicates": row["1차_CSV수"],
            "reacquired_replicates": row["2차_CSV수"],
        }
        for row in read_rows(YNOR_TRACE)
    ]


def readme_lines(summary: list[Row]) -> list[str]:
    total = summary[-1]
    return [
        "연세세브란스 YNOR·YPAN 측정 Cohort",
        "문서 버전",
        "v1.0 / 2026-07-14",
        "범위",
        "YNOR 30 임상 레코드와 YPAN 30 임상 레코드",
        "독립 spectrum 피험자",
        f"{total['independent_spectrum_subjects']}명",
        "측정 데이터",
        f"1차 295 + 2차 재측정 295 = {total['total_spectrum_replicates']} spectra",
        "중요 정의",
        "2차 재측정은 신규 피험자가 아니라 동일 59검체의 반복 측정",
        "YNOR 구성",
        "건강한 자 10, 췌장낭종 10, 기타 양성 질환 6, 만성 췌장염 4",
        "ID 주의",
        "2차 원라벨 YNOR 21은 1차 YNOR 48과 동일 검체이며 현재 YNOR 48로 통합",
        "모델 사용",
        "primary 모델은 독립 피험자 59명만 사용; 회차 간 subject 분리 금지",
        "개인정보",
        "직접식별자, 정확한 임상·측정 일자, 자유서술 과거력, 원본 파일 경로 제외",
        "외부 해석",
        "Cancer Screening 성능 산출 시 병원 confound 가능성을 반드시 별도 고지",
    ]


def main() -> None:
    summary = cohort_summary()
    measurements = reacquired_measurements()
    validate_reacquired_measurements(measurements)
    acquisitions = acquisition_inventory(measurements)
    tables = [
        ("Cohort_Summary", summary),
        ("Measurement_Schedule", measurement_schedule(measurements)),
        ("Subject_Manifest", subject_manifest(acquisitions)),
        ("Acquisition_Inventory", acquisitions),
        ("YNOR_ID_Mapping", ynor_id_mapping(acquisitions)),
        ("Clinical_Completeness", read_rows(TABLES / "yonsei_ynor_ypan_clinical_completeness.csv")),
    ]
    if PACKAGE.exists():
        shutil.rmtree(PACKAGE)
    ARCHIVE.unlink(missing_ok=True)
    CSV_DIR.mkdir(parents=True)
    for index, (_, rows) in enumerate(tables, start=1):
        write_rows(CSV_DIR / f"{index:02d}_{tables[index - 1][0].lower()}.csv", rows)
    write_workbook(WORKBOOK, tables, readme_lines(summary))
    readme = PACKAGE / "README_업로드안내.md"
    write_readme(readme, WORKBOOK.name, summary[-1])
    write_archive(
        ARCHIVE, PACKAGE, PACKAGE_NAME, [WORKBOOK, readme, *sorted(CSV_DIR.glob("*.csv"))]
    )
    print(f"Workbook: {WORKBOOK}")
    print(f"Archive: {ARCHIVE}")


if __name__ == "__main__":
    main()
