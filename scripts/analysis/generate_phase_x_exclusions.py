#!/usr/bin/env python3
"""Phase X: 제외 리스트 생성 스크립트.

Post-op 암 환자와 비암 대조군 중 타 암 병력자를 식별하여
exclusion CSV를 생성한다.

Usage:
    python scripts/analysis/generate_phase_x_exclusions.py
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CLINICAL_CSV = ROOT / "data/clinical_data/standardized/all_clinical_standardized.csv"
SPECTRA_CSV = ROOT / "results/processed_spectra.csv"
OUTPUT_CSV = ROOT / "data/exclusions/phase_x_exclusions.csv"

CANCER_GROUPS_CLINICAL = {"PRO", "BRE", "OVA", "LUN", "CRC", "PAN", "BLA"}
NON_CANCER_GROUPS = {"NOR", "DIA", "HBP", "H.D."}

# Clinical disease_group → spectral group name
GROUP_MAP = {
    "BLA": "BLC",
    # PAN is handled specially: CPAN and YPAN are separate spectral groups
}


def extract_spectral_id(patient_id: str, disease_group: str) -> tuple:
    """Map clinical patient_id to (spectral_group, spectral_sample_id).

    Returns (group, sample_id) or (None, None) if unmappable.
    """
    spectral_group = GROUP_MAP.get(disease_group, disease_group)

    # Standard format: "GROUP N", "GROUP  N", "H. D. N"
    m = re.match(r"^([A-Z][A-Z.\s]*?)\s+(\d+)$", patient_id.strip())
    if m:
        prefix = m.group(1).strip()
        sid = int(m.group(2))
        # CPAN/YPAN stay as-is in spectral data (not mapped to PAN)
        if prefix in ("CPAN", "YPAN"):
            return prefix, sid
        # "H. D." → "H.D." in spectral data
        if "H." in prefix and "D." in prefix:
            return "H.D.", sid
        return spectral_group, sid

    # BLA format: "1기~2기-N" → BLC N, "3기~4기-N" → BLC (290+N)
    m = re.match(r"^(\d)기~(\d)기-(\d+)$", patient_id.strip())
    if m:
        stage_prefix = int(m.group(1))
        n = int(m.group(3))
        if stage_prefix >= 3:
            return "BLC", 290 + n  # 3기~4기 starts at 291
        return "BLC", n

    # Numeric-only (LUN3, OVA1): cannot map without external lookup
    return None, None


def build_lun3_mapping() -> dict:
    """Build mapping from LUN3 clinical patient_id (제공자bCODE) to spectral sample_id.

    LUN3 (폐암 3) patients are in spectral files LUN 201-300,
    ordered by row in the Excel sheet.
    """
    excel_path = ROOT / "data/clinical_data/4. 폐암/SMCXD06_폐암 3.xlsx"
    if not excel_path.exists():
        return {}
    df = pd.read_excel(excel_path, sheet_name="인구학적정보 및 암 관련 정보",
                       header=0, engine="openpyxl")
    mapping = {}
    for i, (_, row) in enumerate(df.iterrows()):
        pid = str(row.get("제공자:제공자bCODE", "")).strip()
        if pid and pid != "nan":
            mapping[pid] = 201 + i  # LUN 201-300
    return mapping


def build_ova1_mapping() -> dict:
    """Build mapping from OVA1 clinical patient_id (제공자bCODE) to spectral sample_id.

    OVA1 (난소암 1) patients are in spectral files OVA 1-30,
    ordered by row in the Excel sheet.
    """
    excel_path = ROOT / "data/clinical_data/3. 난소암/SMCXD01_난소암 1.xlsx"
    if not excel_path.exists():
        return {}
    df = pd.read_excel(excel_path, sheet_name="C56 임상정보",
                       header=0, engine="openpyxl")
    mapping = {}
    for i, (_, row) in enumerate(df.iterrows()):
        pid = str(row.get("제공자bCODE", "")).strip()
        if pid and pid != "nan":
            mapping[pid] = 1 + i  # OVA 1-30
    return mapping


def find_postop_cancer(clin: pd.DataFrame) -> pd.DataFrame:
    """Identify post-op cancer patients.

    PRO is excluded because it has no surgery_date — its "post-op" label
    comes from diagnosis_date fallback, which does not indicate surgical status.
    """
    mask = (
        clin["disease_group"].isin(CANCER_GROUPS_CLINICAL)
        & (clin["sample_timing"] == "post-op")
        & (clin["disease_group"] != "PRO")  # PRO: no surgery_date, diagnosis fallback only
    )
    postop = clin[mask][["patient_id", "disease_group", "sample_timing"]].copy()
    postop["exclusion_reason"] = "post-op cancer"
    postop["detail"] = postop["disease_group"] + " post-operative sample"
    return postop


def find_noncancer_with_cancer_history(clin: pd.DataFrame) -> pd.DataFrame:
    """Identify non-cancer controls with prior cancer history.

    Parses past_history for '과거력(암)-XXX=예' patterns.
    Excludes false positives: 없음, 없슴, 심혈관.
    """
    nc = clin[clin["disease_group"].isin(NON_CANCER_GROUPS)].copy()
    results = []
    for _, row in nc.iterrows():
        ph = row.get("past_history")
        if pd.isna(ph):
            continue
        mentions = re.findall(r"과거력\(암\)-([^;]+)", str(ph))
        cancers = []
        for m in mentions:
            m_clean = m.strip()
            if any(x in m_clean for x in ["없음", "없슴", "심혈관"]):
                continue
            # "예" alone or "예[XXX]" with actual cancer → include
            if "예" in m_clean:
                cancers.append(m_clean)
        if cancers:
            results.append({
                "patient_id": row["patient_id"],
                "disease_group": row["disease_group"],
                "sample_timing": row.get("sample_timing"),
                "exclusion_reason": "non-cancer with cancer history",
                "detail": "; ".join(cancers),
            })
    return pd.DataFrame(results)


def map_to_spectral(exclusions: pd.DataFrame,
                    lun3_map: dict, ova1_map: dict,
                    valid_spectral: set) -> pd.DataFrame:
    """Map exclusion rows to spectral (group, sample_id) and filter by what exists."""
    rows = []
    unmapped = []
    for _, row in exclusions.iterrows():
        pid = row["patient_id"]
        dg = row["disease_group"]

        group, sid = extract_spectral_id(pid, dg)

        # Try special mappings for numeric IDs
        if group is None:
            pid_str = str(pid).strip()
            if dg == "LUN" and pid_str in lun3_map:
                group, sid = "LUN", lun3_map[pid_str]
            elif dg == "OVA" and pid_str in ova1_map:
                group, sid = "OVA", ova1_map[pid_str]
            else:
                unmapped.append((pid, dg))
                continue

        # Check if this patient exists in spectral data
        if (group, sid) not in valid_spectral:
            continue

        rows.append({
            "group": group,
            "sample_id": sid,
            "exclusion_reason": row["exclusion_reason"],
            "detail": row["detail"],
            "clinical_patient_id": pid,
            "clinical_disease_group": dg,
        })

    if unmapped:
        print(f"\n  [WARNING] {len(unmapped)} patients could not be mapped:")
        for pid, dg in unmapped:
            print(f"    {pid} ({dg})")

    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("Phase X Exclusion List Generator")
    print("=" * 60)

    # Load data
    print("\n[1] Loading clinical data...")
    clin = pd.read_csv(CLINICAL_CSV)
    print(f"  {len(clin)} patients loaded")

    print("\n[2] Loading spectral data (for intersection)...")
    spec = pd.read_csv(SPECTRA_CSV, usecols=["group", "sample_id"])
    valid_spectral = set(zip(spec["group"], spec["sample_id"]))
    n_subjects = len(set(zip(spec["group"], spec["sample_id"])))
    print(f"  {n_subjects} unique (group, sample_id) pairs")

    # Build special mappings
    print("\n[3] Building ID mappings...")
    lun3_map = build_lun3_mapping()
    ova1_map = build_ova1_mapping()
    print(f"  LUN3 mapping: {len(lun3_map)} patients")
    print(f"  OVA1 mapping: {len(ova1_map)} patients")

    # Find exclusion candidates
    print("\n[4] Identifying exclusion candidates...")

    # Category A: Post-op cancer
    postop = find_postop_cancer(clin)
    print(f"  Post-op cancer (clinical): {len(postop)}")
    for g, sub in postop.groupby("disease_group"):
        print(f"    {g}: {len(sub)}")

    # Category B: Non-cancer with cancer history
    cancer_hist = find_noncancer_with_cancer_history(clin)
    print(f"  Non-cancer with cancer history (clinical): {len(cancer_hist)}")
    for g, sub in cancer_hist.groupby("disease_group"):
        print(f"    {g}: {len(sub)}")

    # Combine
    all_exclusions = pd.concat([postop, cancer_hist], ignore_index=True)
    print(f"  Total exclusion candidates (clinical): {len(all_exclusions)}")

    # Map to spectral IDs and filter
    print("\n[5] Mapping to spectral data...")
    mapped = map_to_spectral(all_exclusions, lun3_map, ova1_map, valid_spectral)
    print(f"  Mapped to spectral data: {len(mapped)} patients")

    if len(mapped) > 0:
        print("\n  By exclusion reason:")
        for reason, sub in mapped.groupby("exclusion_reason"):
            print(f"    {reason}: {len(sub)}")
            for g, gsub in sub.groupby("group"):
                print(f"      {g}: {len(gsub)}")

    # Count excluded spectra
    spec_full = pd.read_csv(SPECTRA_CSV, usecols=["group", "sample_id"])
    excl_keys = set(zip(mapped["group"], mapped["sample_id"]))
    n_spectra = len(spec_full[
        spec_full.apply(lambda r: (r["group"], r["sample_id"]) in excl_keys, axis=1)
    ])
    print(f"\n  Total spectra to be excluded: {n_spectra}")

    # Save
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    mapped.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[6] Saved to: {OUTPUT_CSV}")
    print(f"  {len(mapped)} exclusion entries")

    return mapped


if __name__ == "__main__":
    result = main()
