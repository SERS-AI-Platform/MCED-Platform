#!/usr/bin/env python3
"""암종별 통합 Excel 파일 생성.

raw directory에 분산된 임상 파일(예: 폐암 1/2/3, 난소암 1/2, CPAN/YPAN/SPAN)을
암종별로 1개 xlsx로 통합한다. 각 xlsx는 다음 3개 시트를 포함:
  - All: 해당 암종 전체 환자 + exclusion_category 컬럼
  - Definite_Excluded: 확정 제외 환자만
  - Clean: 활용 가능 환자만

출력: data/clinical_data/by_cancer/*.xlsx

Usage:
    python scripts/analysis/build_cancer_xlsx.py
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CLINICAL_CSV = ROOT / "data/clinical_data/standardized/all_clinical_standardized.csv"
REGISTRY_CSV = ROOT / "data/clinical_data/sample_exclusion_registry.csv"
OUTDIR = ROOT / "data/clinical_data/by_cancer"

# 암종별 그룹 정의: 파일명 → (한글 라벨, [disease_group...])
CANCER_GROUPS = {
    "bladder":     ("방광암",   ["BLA"]),
    "breast":      ("유방암",   ["BRE"]),
    "colorectal":  ("대장암",   ["CRC"]),
    "lung":        ("폐암",     ["LUN"]),
    "ovarian":     ("난소암",   ["OVA"]),
    "pancreatic":  ("췌장암",   ["PAN", "SPAN", "YPAN_BENIGN", "YPAN_CP", "YPAN_CYST", "YPAN_NOR"]),
    "prostate":    ("전립선암", ["PRO"]),
    "controls":    ("대조군",   ["NOR", "DIA", "HBP", "H.D."]),
}


def main():
    print("=" * 60)
    print("암종별 Excel 파일 생성")
    print("=" * 60)

    clin = pd.read_csv(CLINICAL_CSV, encoding="utf-8-sig")
    reg = pd.read_csv(REGISTRY_CSV, encoding="utf-8-sig")

    # patient_id → exclusion 정보 매핑 (definite 우선)
    excl_map = {}
    reason_map = {}
    detail_map = {}
    for _, row in reg.sort_values("exclusion_category").iterrows():
        pid = row["patient_id"]
        if pid not in excl_map or row["exclusion_category"] == "definite_exclude":
            excl_map[pid] = row["exclusion_category"]
            reason_map[pid] = row["exclusion_reason"]
            detail_map[pid] = row["detail"]

    clin["exclusion_category"] = clin["patient_id"].map(excl_map).fillna("usable")
    clin["exclusion_reason"] = clin["patient_id"].map(reason_map)
    clin["exclusion_detail"] = clin["patient_id"].map(detail_map)

    OUTDIR.mkdir(parents=True, exist_ok=True)

    for fname, (label, groups) in CANCER_GROUPS.items():
        sub = clin[clin["disease_group"].isin(groups)].copy()
        if len(sub) == 0:
            print(f"  [SKIP] {fname}: 데이터 없음")
            continue

        all_sheet = sub.copy()
        definite = sub[sub["exclusion_category"] == "definite_exclude"].copy()
        clean = sub[sub["exclusion_category"] != "definite_exclude"].copy()

        out_path = OUTDIR / f"{fname}.xlsx"
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            all_sheet.to_excel(writer, sheet_name="All", index=False)
            definite.to_excel(writer, sheet_name="Definite_Excluded", index=False)
            clean.to_excel(writer, sheet_name="Clean", index=False)

        print(f"  ✓ {fname}.xlsx ({label})")
        print(f"      All: {len(all_sheet)}, Definite_Excluded: {len(definite)}, Clean: {len(clean)}")
        print(f"      구성 그룹: {', '.join(groups)}")

    print(f"\n출력 위치: {OUTDIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
