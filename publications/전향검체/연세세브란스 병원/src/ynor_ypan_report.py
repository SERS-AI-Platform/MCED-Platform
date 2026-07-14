from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, stdev


@dataclass(frozen=True, slots=True)
class ReportData:
    sources: dict[str, list[dict[str, str]]]
    inventory: list[dict[str, str]]
    mapping: list[dict[str, str]]
    output: Path


def number_summary(rows: list[dict[str, str]], field: str) -> str:
    values = [float(row[field]) for row in rows if row[field].strip() != ""]
    return f"{mean(values):.1f} ± {stdev(values):.1f}; median {median(values):.1f}; range {min(values):.1f}-{max(values):.1f}"


def write_report(data: ReportData) -> None:
    ynor, ypan = data.sources["YNOR"], data.sources["YPAN"]
    diagnoses = Counter(row["diagnosis"] for row in ynor)
    sexes = {
        group: Counter(row["sex"] for row in data.sources[group]) for group in ("YNOR", "YPAN")
    }
    stages = Counter(
        row["stage_1_2_or_3_4"] or "missing"
        for row in data.mapping
        if row["source_group"] == "YPAN"
    )
    lines = [
        "# YNOR·YPAN 전체 데이터 및 임상정보 정리",
        "",
        "## 핵심 구분",
        "",
        "- 임상 레코드: YNOR 30명 + YPAN 30명 = 60명",
        "- primary spectrum 독립 피험자: YNOR 29명 + YPAN 30명 = 59명, 각 5 replicate",
        "- reacquired spectrum: 동일 59명에 대해 각 5 replicate가 별도로 존재",
        "- processed acquisition은 60건이지만 `YNOR 21`은 `YNOR 48`의 재측정본이므로 primary 분석은 59명",
        "",
        "## 데이터 층별 수량",
        "",
        "| 군 | 임상 레코드 | Primary subjects / spectra | Reacquired subjects / spectra | Processed acquisitions | Primary model |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in data.inventory:
        lines.append(
            f"| {row['source_group']} | {row['clinical_records']} | {row['primary_spectrum_subjects']} / {row['primary_spectrum_replicates']} | {row['reacquired_subjects']} / {row['reacquired_replicates']} | {row['processed_acquisitions']} | {row['primary_model_subjects']} |"
        )
    lines += [
        "",
        "## 임상 구성",
        "",
        f"- YNOR 진단 구성: 건강한 자 {diagnoses['건강한 자']}명, 췌장낭종 {diagnoses['췌장낭종']}명, 기타 양성 질환 {diagnoses['기타 양성 질환']}명, 만성 췌장염 {diagnoses['만성 췌장염']}명",
        f"- YNOR age: {number_summary(ynor, 'age')}; BMI: {number_summary(ynor, 'bmi')}",
        f"- YPAN age: {number_summary(ypan, 'age')}; BMI: {number_summary(ypan, 'bmi')}",
        f"- 성별: YNOR F {sexes['YNOR']['F']} / M {sexes['YNOR']['M']}, YPAN F {sexes['YPAN']['F']} / M {sexes['YPAN']['M']}",
        f"- YPAN analysis stage: 1-2 {stages['1-2']}명, 3-4 {stages['3-4']}명, missing {stages['missing']}명",
        "- YNOR은 암 병기 대상이 아니며, 15/30명에서 sample timing이 post-op으로 기록되고 15명은 결측",
        "- CA19-9: YNOR 14/30, YPAN 24/30에서 값 보유",
        "",
        "## 해석상 중요사항",
        "",
        "- YNOR은 순수 건강인군이 아니라 건강인·췌장낭종·만성 췌장염·기타 양성 질환이 섞인 non-cancer 군이다.",
        "- YPAN이 YNOR보다 연령이 높고 BMI가 낮아 임상 구성 차이가 spectrum 분류에 영향을 줄 수 있다.",
        "- `YNOR 21`과 `YNOR 48`을 독립 fold로 분리하면 동일 피험자 누수가 생기므로 반드시 subject 단위로 묶거나 repeat를 제외해야 한다.",
        "- 연세 subset은 단일기관 내부 자료다. Cancer Screening 결과를 외부검증 또는 cross-hospital 일반화 근거로 사용하면 안 된다.",
        "",
        "## 생성 표",
        "",
        "- `tables/yonsei_ynor_ypan_group_inventory.csv`",
        "- `tables/yonsei_ynor_ypan_acquisition_inventory.csv`",
        "- `tables/yonsei_ynor_ypan_processed_inventory.csv`",
        "- `tables/yonsei_ynor_ypan_clinical_records.csv`",
        "- `tables/yonsei_ynor_ypan_clinical_completeness.csv`",
        "",
        "직접 환자 ID, 임상 원본 파일명, exact date, 자유서술 과거력은 새 표에서 제외했다. 날짜는 연도만 보존했다.",
    ]
    data.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
