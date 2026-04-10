#!/usr/bin/env python3
"""Build master_clinical.csv: standardized clinical + spectral mapping + exclusion info.

- Source: data/clinical_data/standardized/all_clinical_standardized.csv
- Drops: sample_timing (unreliable)
- Adds:  spectral_group, spectral_sample_id, in_spectra,
         exclusion_category, exclusion_reason, exclusion_detail
- Output: data/clinical_data/master_clinical.csv

Also prints full spectrum<->clinical mapping verification.
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/analysis"))
from generate_phase_x_exclusions import (
    extract_spectral_id,
    build_lun3_mapping,
    build_ova1_mapping,
)

STD = ROOT / "data/clinical_data/standardized/all_clinical_standardized.csv"
REG = ROOT / "data/clinical_data/sample_exclusion_registry.csv"
SPECTRA = ROOT / "results/processed_spectra.csv"
OUT = ROOT / "data/clinical_data/master_clinical.csv"

DROP_COLS = ["sample_timing"]


def build_bre_mapping() -> dict:
    """Map BRE clinical patient_id (제공자bCODE) → spectral sample_id (1-30) via box-order sheet."""
    p = ROOT / "data/clinical_data/2. 유방암/SMCXD01_유방암.xlsx"
    if not p.exists():
        return {}
    box = pd.read_excel(p, sheet_name="box 순서 및 정도관리")
    mapping = {}
    for _, r in box.iterrows():
        code = r.get("제공자:제공자bCODE")
        label = str(r.get("SoluM Label", "")).strip()
        if pd.isna(code) or not label.startswith("BRE"):
            continue
        try:
            sid = int(label.split()[1])
            mapping[str(int(code))] = sid
        except (ValueError, IndexError):
            continue
    return mapping


def map_to_spectral(df: pd.DataFrame) -> pd.DataFrame:
    lun3 = build_lun3_mapping()
    ova1 = build_ova1_mapping()
    bre = build_bre_mapping()
    sgroup, ssid = [], []
    for _, r in df.iterrows():
        pid = str(r["patient_id"]).strip()
        dg = str(r["disease_group"]).strip()
        g, sid = extract_spectral_id(pid, dg)
        if g is None:
            if dg == "LUN" and pid in lun3:
                g, sid = "LUN", lun3[pid]
            elif dg == "OVA" and pid in ova1:
                g, sid = "OVA", ova1[pid]
            elif dg == "BRE" and pid in bre:
                g, sid = "BRE", bre[pid]
        sgroup.append(g)
        ssid.append(sid)
    df = df.copy()
    df["spectral_group"] = sgroup
    df["spectral_sample_id"] = ssid
    return df


def main():
    print(f"[load] {STD.relative_to(ROOT)}")
    clin = pd.read_csv(STD, encoding="utf-8-sig", low_memory=False)
    print(f"  rows: {len(clin)}")

    # Drop unreliable timing
    for c in DROP_COLS:
        if c in clin.columns:
            clin = clin.drop(columns=c)

    # Map to spectral
    clin = map_to_spectral(clin)

    # Load spectra unique keys
    spectra = pd.read_csv(SPECTRA, usecols=["group", "sample_id"]).drop_duplicates()
    spectra_keys = set(zip(spectra["group"].astype(str), spectra["sample_id"].astype(int)))
    print(f"[spectra] unique patients: {len(spectra_keys)}")

    # in_spectra flag
    def _in(row):
        if pd.isna(row["spectral_group"]) or pd.isna(row["spectral_sample_id"]):
            return False
        return (str(row["spectral_group"]), int(row["spectral_sample_id"])) in spectra_keys

    clin["in_spectra"] = clin.apply(_in, axis=1)

    # Join exclusion registry
    reg = pd.read_csv(REG, encoding="utf-8-sig")
    keep = ["patient_id", "exclusion_category", "exclusion_reason", "detail"]
    reg_small = reg[[c for c in keep if c in reg.columns]].rename(columns={"detail": "exclusion_detail"})
    clin = clin.merge(reg_small, on="patient_id", how="left")

    # Reorder: identification → demographics → diagnosis → dates → labs → spectral/exclusion
    front = [
        "patient_id", "disease_group", "source_file",
        "spectral_group", "spectral_sample_id", "in_spectra",
        "exclusion_category", "exclusion_reason", "exclusion_detail",
        "age", "sex", "height_cm", "weight_kg", "bmi",
        "diagnosis", "diagnosis_date", "surgery_date", "sample_date",
    ]
    rest = [c for c in clin.columns if c not in front]
    clin = clin[[c for c in front if c in clin.columns] + rest]

    clin.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n[written] {OUT.relative_to(ROOT)}  ({len(clin)} rows, {len(clin.columns)} cols)")

    # ===== Verification =====
    print("\n" + "=" * 60)
    print("MAPPING VERIFICATION")
    print("=" * 60)

    print("\n[1] disease_group counts (clinical):")
    print(clin["disease_group"].value_counts(dropna=False).to_string())
    print(f"  TOTAL: {len(clin)}")

    print("\n[2] in_spectra by disease_group:")
    ct = pd.crosstab(clin["disease_group"], clin["in_spectra"], margins=True)
    print(ct.to_string())

    print("\n[3] spectral_group counts (mapped from clinical):")
    print(clin["spectral_group"].value_counts(dropna=False).to_string())

    print("\n[4] Spectra-side coverage (1,628 spectral patients):")
    clin_keys = set(
        (str(g), int(s))
        for g, s in zip(clin["spectral_group"], clin["spectral_sample_id"])
        if pd.notna(g) and pd.notna(s)
    )
    covered = spectra_keys & clin_keys
    spectra_only = spectra_keys - clin_keys
    clin_only = clin_keys - spectra_keys
    print(f"  spectra ∩ clinical : {len(covered)}")
    print(f"  spectra only       : {len(spectra_only)}  (in spectra, no clinical row)")
    print(f"  clinical only      : {len(clin_only)}  (clinical mapped key, not in spectra)")
    print(f"  clinical unmapped  : {clin['spectral_group'].isna().sum()}  (no spectral key)")

    if spectra_only:
        from collections import Counter
        print("\n  spectra-only by group:")
        for g, c in sorted(Counter(g for g, _ in spectra_only).items(), key=lambda x: -x[1]):
            print(f"    {g:<6} {c}")
        print("  spectra-only IDs:")
        for g, sid in sorted(spectra_only):
            print(f"    {g} {sid}")

    if clin_only:
        from collections import Counter
        print("\n  clinical-only (mapped but missing in spectra) by group:")
        for g, c in sorted(Counter(g for g, _ in clin_only).items(), key=lambda x: -x[1]):
            print(f"    {g:<6} {c}")
        clin_only_rows = clin[clin.apply(
            lambda r: pd.notna(r["spectral_group"]) and pd.notna(r["spectral_sample_id"])
            and (str(r["spectral_group"]), int(r["spectral_sample_id"])) in clin_only, axis=1)]
        print(clin_only_rows[["patient_id","disease_group","spectral_group","spectral_sample_id"]].to_string(index=False))

    if clin["spectral_group"].isna().any():
        print("\n  unmapped clinical rows (no spectral key):")
        un = clin[clin["spectral_group"].isna()]
        print(un["disease_group"].value_counts().to_string())

    print("\n[5] exclusion summary:")
    print(clin["exclusion_category"].fillna("(usable)").value_counts().to_string())


if __name__ == "__main__":
    main()
