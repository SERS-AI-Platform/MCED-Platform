#!/usr/bin/env python3
"""Trace 1,628 spectral patients vs 1,560 clean clinical → list missing 68.

For each spectral (group, sample_id) not present in clean_clinical_data.csv,
look up the corresponding row in:
  - sample_exclusion_registry.csv (to see exclusion_reason)
  - all_clinical_standardized.csv (to confirm clinical row exists at all)

Outputs:
  data/clinical_data/missing_from_clean.csv
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

CLEAN = ROOT / "data/clinical_data/clean_clinical_data.csv"
STD = ROOT / "data/clinical_data/standardized/all_clinical_standardized.csv"
REG = ROOT / "data/clinical_data/sample_exclusion_registry.csv"
SPECTRA = ROOT / "results/processed_spectra.csv"
OUT = ROOT / "data/clinical_data/missing_from_clean.csv"


def map_clinical_to_spectral(df: pd.DataFrame) -> pd.DataFrame:
    lun3 = build_lun3_mapping()
    ova1 = build_ova1_mapping()
    rows = []
    for _, r in df.iterrows():
        pid = str(r["patient_id"]).strip()
        dg = str(r["disease_group"]).strip()
        g, sid = extract_spectral_id(pid, dg)
        if g is None:
            # Try LUN3 / OVA1 numeric-only mappings
            if dg == "LUN" and pid in lun3:
                g, sid = "LUN", lun3[pid]
            elif dg == "OVA" and pid in ova1:
                g, sid = "OVA", ova1[pid]
        rows.append((pid, dg, g, sid))
    out = pd.DataFrame(rows, columns=["patient_id", "disease_group", "sgroup", "ssid"])
    return out


def main():
    spectra = pd.read_csv(SPECTRA, usecols=["group", "sample_id"]).drop_duplicates()
    spectra_keys = set(zip(spectra["group"].astype(str), spectra["sample_id"].astype(int)))
    print(f"[spectra] unique patients: {len(spectra_keys)}")

    clean = pd.read_csv(CLEAN, encoding="utf-8-sig")
    print(f"[clean]   rows: {len(clean)}")
    clean_map = map_clinical_to_spectral(clean)
    unmapped_clean = clean_map[clean_map["sgroup"].isna()]
    print(f"[clean]   unmapped (no spectral key): {len(unmapped_clean)}")
    clean_keys = set(
        (g, int(s)) for g, s in zip(clean_map["sgroup"], clean_map["ssid"]) if pd.notna(g) and pd.notna(s)
    )
    print(f"[clean]   spectral keys covered: {len(clean_keys)}")

    missing = spectra_keys - clean_keys
    print(f"\n[MISSING] in spectra but not in clean: {len(missing)}")

    # Group breakdown
    from collections import Counter
    cnt = Counter(g for g, _ in missing)
    print("  by group:")
    for g, c in sorted(cnt.items(), key=lambda x: -x[1]):
        print(f"    {g:<6} {c}")

    # Reverse-lookup: for each missing key, find row in standardized + registry
    std = pd.read_csv(STD, encoding="utf-8-sig", low_memory=False)
    std_map = map_clinical_to_spectral(std[["patient_id", "disease_group"]])
    std = std.assign(sgroup=std_map["sgroup"].values, ssid=std_map["ssid"].values)

    reg = pd.read_csv(REG, encoding="utf-8-sig")
    if "spectral_group" in reg.columns and "spectral_sample_id" in reg.columns:
        reg_keyed = reg.dropna(subset=["spectral_group", "spectral_sample_id"]).copy()
        reg_keyed["ssid"] = reg_keyed["spectral_sample_id"].astype(int)
        reg_lookup = {
            (str(g), int(s)): (cat, reason, det)
            for g, s, cat, reason, det in zip(
                reg_keyed["spectral_group"],
                reg_keyed["ssid"],
                reg_keyed.get("exclusion_category", [""] * len(reg_keyed)),
                reg_keyed.get("exclusion_reason", [""] * len(reg_keyed)),
                reg_keyed.get("detail", [""] * len(reg_keyed)),
            )
        }
    else:
        reg_lookup = {}

    out_rows = []
    for g, sid in sorted(missing):
        std_hit = std[(std["sgroup"] == g) & (std["ssid"] == sid)]
        if len(std_hit):
            r = std_hit.iloc[0]
            pid = r["patient_id"]
            dg = r["disease_group"]
            timing = r.get("sample_timing", "")
            in_std = True
        else:
            pid, dg, timing, in_std = "", "", "", False
        cat, reason, detail = reg_lookup.get((g, sid), ("", "", ""))
        out_rows.append(
            dict(
                sgroup=g,
                ssid=sid,
                in_standardized=in_std,
                patient_id=pid,
                disease_group=dg,
                sample_timing=timing,
                exclusion_category=cat,
                exclusion_reason=reason,
                detail=detail,
            )
        )

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n[written] {OUT.relative_to(ROOT)}  ({len(out_df)} rows)")

    # Summary by reason
    print("\n[by exclusion_reason]")
    rc = out_df["exclusion_reason"].fillna("(not in registry)").replace("", "(not in registry)").value_counts()
    for k, v in rc.items():
        print(f"  {k:<40} {v}")
    print("\n[by in_standardized]")
    print(out_df["in_standardized"].value_counts().to_string())


if __name__ == "__main__":
    main()
