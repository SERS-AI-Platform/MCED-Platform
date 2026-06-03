#!/usr/bin/env python3
"""전체 샘플 제외 레지스트리 및 Clean Clinical Data 생성.

제외 카테고리:
  - definite_exclude: SPAN 전체, post-op 암 환자, 암 병력 대조군
  - review_periop: 수술 ±7일 내 채취 (추후 검토용)
  - review_undefined: sample_timing 미정의 (추후 검토용)

출력:
  1. data/clinical_data/sample_exclusion_registry.csv  — 제외/리뷰 목록
  2. data/clinical_data/clean_clinical_data.csv         — 확정 제외 후 활용 가능 데이터

Usage:
    python scripts/analysis/generate_exclusion_registry.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

from generate_phase_x_exclusions import (
    extract_spectral_id,
    build_lun3_mapping,
    build_ova1_mapping,
    find_postop_cancer,
    find_noncancer_with_cancer_history,
)

CLINICAL_CSV = ROOT / "data/clinical_data/standardized/all_clinical_standardized.csv"
SPECTRA_CSV = ROOT / "results/processed_spectra.csv"
REGISTRY_CSV = ROOT / "data/clinical_data/sample_exclusion_registry.csv"
CLEAN_CSV = ROOT / "data/clinical_data/clean_clinical_data.csv"

# Columns to include in registry for review
META_COLS = [
    "sample_timing", "sample_date", "surgery_date", "diagnosis_date",
    "age", "sex", "stage", "pathology",
]


def resolve_spectral_mapping(patient_id: str, disease_group: str,
                              lun3_map: dict, ova1_map: dict) -> tuple:
    """Clinical patient_id → (spectral_group, spectral_sample_id)."""
    group, sid = extract_spectral_id(patient_id, disease_group)
    if group is not None:
        return group, sid

    # LUN3/OVA1 numeric ID fallback
    pid_str = str(patient_id).strip()
    if disease_group == "LUN" and pid_str in lun3_map:
        return "LUN", lun3_map[pid_str]
    if disease_group == "OVA" and pid_str in ova1_map:
        return "OVA", ova1_map[pid_str]

    return None, None


def find_span_exclusions(clin: pd.DataFrame) -> pd.DataFrame:
    """SPAN 전체를 확정 제외로 식별."""
    span = clin[clin["disease_group"] == "SPAN"].copy()
    return pd.DataFrame({
        "patient_id": span["patient_id"],
        "disease_group": span["disease_group"],
        "sample_timing": span["sample_timing"],
        "exclusion_reason": "SPAN post-op cancer (전체)",
        "detail": "삼성 췌장암 — 수술 후 샘플, 스크리닝 부적합",
    })


def find_periop_patients(clin: pd.DataFrame) -> pd.DataFrame:
    """Peri-op 환자를 리뷰용으로 식별."""
    periop = clin[clin["sample_timing"] == "peri-op"].copy()
    rows = []
    for _, row in periop.iterrows():
        dg = row["disease_group"]
        detail = f"{dg} peri-operative sample (수술 ±7일)"
        if dg == "PRO":
            detail += " — timing은 diagnosis_date fallback (surgery_date 없음)"
        rows.append({
            "patient_id": row["patient_id"],
            "disease_group": dg,
            "sample_timing": row["sample_timing"],
            "exclusion_reason": "peri-operative sample",
            "detail": detail,
        })
    return pd.DataFrame(rows)


def find_undefined_timing(clin: pd.DataFrame) -> pd.DataFrame:
    """sample_timing이 NaN이면서 SPAN이 아닌 환자."""
    mask = clin["sample_timing"].isna() & (clin["disease_group"] != "SPAN")
    undef = clin[mask].copy()
    return pd.DataFrame({
        "patient_id": undef["patient_id"],
        "disease_group": undef["disease_group"],
        "sample_timing": undef["sample_timing"],
        "exclusion_reason": "undefined timing",
        "detail": undef["disease_group"] + " — sample_timing 미정의",
    })


def build_registry(clin: pd.DataFrame, valid_spectral: set,
                    lun3_map: dict, ova1_map: dict) -> pd.DataFrame:
    """전체 제외/리뷰 레지스트리 구축."""
    # 1) 확정 제외: SPAN
    span_excl = find_span_exclusions(clin)
    span_excl["exclusion_category"] = "definite_exclude"

    # 2) 확정 제외: post-op 암 환자 (기존 함수 재사용, PRO 자동 제외)
    postop = find_postop_cancer(clin)
    postop["exclusion_category"] = "definite_exclude"

    # 3) 확정 제외: 암 병력 대조군 (기존 함수 재사용)
    cancer_hist = find_noncancer_with_cancer_history(clin)
    cancer_hist["exclusion_category"] = "definite_exclude"

    # 4) 리뷰용: peri-op
    periop = find_periop_patients(clin)
    periop["exclusion_category"] = "review_periop"

    # 5) 리뷰용: undefined timing (SPAN 제외)
    undef = find_undefined_timing(clin)
    undef["exclusion_category"] = "review_undefined"

    # Combine
    all_flags = pd.concat([span_excl, postop, cancer_hist, periop, undef],
                          ignore_index=True)

    # Spectral mapping
    spec_groups = []
    spec_ids = []
    for _, row in all_flags.iterrows():
        g, sid = resolve_spectral_mapping(
            str(row["patient_id"]), row["disease_group"], lun3_map, ova1_map
        )
        # Check if spectral data actually exists
        if g is not None and (g, sid) not in valid_spectral:
            g, sid = None, None
        spec_groups.append(g)
        spec_ids.append(sid)

    all_flags["spectral_group"] = spec_groups
    all_flags["spectral_sample_id"] = spec_ids

    # Join clinical metadata
    clin_indexed = clin.set_index("patient_id")
    for col in META_COLS:
        if col not in all_flags.columns:
            all_flags[col] = all_flags["patient_id"].map(
                clin_indexed[col].to_dict()
            )

    # Reorder columns
    col_order = [
        "patient_id", "disease_group", "spectral_group", "spectral_sample_id",
        "exclusion_category", "exclusion_reason", "detail",
    ] + META_COLS
    all_flags = all_flags[[c for c in col_order if c in all_flags.columns]]

    return all_flags.sort_values(
        ["exclusion_category", "disease_group", "patient_id"]
    ).reset_index(drop=True)


def build_clean_clinical(clin: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    """확정 제외 환자를 제거한 clean clinical data."""
    definite = registry[registry["exclusion_category"] == "definite_exclude"]
    exclude_ids = set(definite["patient_id"].unique())
    clean = clin[~clin["patient_id"].isin(exclude_ids)].copy()
    return clean.reset_index(drop=True)


def main():
    print("=" * 60)
    print("Sample Exclusion Registry Generator")
    print("=" * 60)

    # 1. Load data
    print("\n[1] Loading clinical data...")
    clin = pd.read_csv(CLINICAL_CSV, encoding="utf-8-sig")
    print(f"  {len(clin)}명 로드")

    print("\n[2] Loading spectral data...")
    spec = pd.read_csv(SPECTRA_CSV, usecols=["group", "sample_id"])
    valid_spectral = set(zip(spec["group"], spec["sample_id"]))
    n_subjects = len(valid_spectral)
    print(f"  {n_subjects} unique (group, sample_id)")

    # 2. Build ID mappings
    print("\n[3] Building ID mappings...")
    lun3_map = build_lun3_mapping()
    ova1_map = build_ova1_mapping()
    print(f"  LUN3: {len(lun3_map)}, OVA1: {len(ova1_map)}")

    # 3. Build registry
    print("\n[4] Building exclusion registry...")
    registry = build_registry(clin, valid_spectral, lun3_map, ova1_map)

    # Summary
    print("\n  === Registry Summary ===")
    for cat in ["definite_exclude", "review_periop", "review_undefined"]:
        sub = registry[registry["exclusion_category"] == cat]
        n_with_spectral = sub["spectral_group"].notna().sum()
        print(f"  {cat}: {len(sub)}명 (스펙트럼 매핑: {n_with_spectral}명)")
        for reason, rsub in sub.groupby("exclusion_reason"):
            print(f"    {reason}: {len(rsub)}명")

    # 4. Build clean data
    print("\n[5] Building clean clinical data...")
    clean = build_clean_clinical(clin, registry)
    print(f"  원본: {len(clin)}명 → 확정 제외 후: {len(clean)}명")
    print(f"  제거: {len(clin) - len(clean)}명")

    # 5. Cross-validate with existing Phase X exclusions
    print("\n[6] Cross-validation with phase_x_exclusions.csv...")
    phase_x_csv = ROOT / "data/exclusions/phase_x_exclusions.csv"
    if phase_x_csv.exists():
        phase_x = pd.read_csv(phase_x_csv)
        definite = registry[registry["exclusion_category"] == "definite_exclude"]
        definite_spectral = set(
            zip(definite["spectral_group"].dropna(), definite["spectral_sample_id"].dropna())
        )
        phase_x_keys = set(zip(phase_x["group"], phase_x["sample_id"]))
        missing = phase_x_keys - definite_spectral
        if missing:
            print(f"  [WARNING] Phase X에 있지만 레지스트리에 없는 항목: {len(missing)}")
            for g, sid in sorted(missing):
                print(f"    {g} {sid}")
        else:
            print(f"  Phase X 66건 모두 레지스트리에 포함 확인")

    # 6. Save
    print("\n[7] Saving...")
    registry.to_csv(REGISTRY_CSV, index=False, encoding="utf-8-sig")
    print(f"  {REGISTRY_CSV}")
    print(f"  → {len(registry)}건")

    clean.to_csv(CLEAN_CSV, index=False, encoding="utf-8-sig")
    print(f"  {CLEAN_CSV}")
    print(f"  → {len(clean)}명")

    # 7. Disease group distribution in clean data
    print("\n  === Clean Data 분포 ===")
    for dg in sorted(clean["disease_group"].unique()):
        n = len(clean[clean["disease_group"] == dg])
        print(f"    {dg}: {n}명")

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
