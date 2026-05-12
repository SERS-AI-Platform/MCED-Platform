"""Stage analysis - Step 5: Update master_clinical.csv from clinical_unified.

Authoritative source: clinical_unified_202604172056.csv
Target: data/clinical_data/master_clinical.csv

Columns to replace / add for 5-cancer rows (CRC, CPAN, PRO, OVA, LUN):
  - stage             <- cancer_stage (new)
  - tnm               <- tnm_stage (new)
  - t_stage, n_stage, m_stage  (same names)
  - smoking_status    <- smoking (new)   [raw string, keeps unified values]
  - drinking_status   <- drinking (new)
  - cancer_stage_group (NEW column)
  - prior_cancer_flag  (NEW column)
  - smoking_years, smoking_per_day, drinking_days (NEW columns)

Manual override per user instruction:
  - CRC solum_label 271..300 -> cancer_stage_group='advanced'
    (these are absent from unified; rows exist in master only)

Preserved (NOT touched): source_file, patient_id, disease_group,
spectral_group, spectral_sample_id, age, sex, bmi, blood_*, etc.

Backup created at data/clinical_data/master_clinical.backup_YYYYMMDD_HHMMSS.csv
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import shutil
import pandas as pd
import numpy as np

ROOT = Path("/home/user/SERS-AI")
CLI_UNIFIED = ROOT / "data/clinical_data/clinical_unified_202604172056.csv"
MASTER = ROOT / "data/clinical_data/master_clinical.csv"

TARGET_GROUPS = ["CRC", "CPAN", "PRO", "OVA", "LUN"]

# unified column  -> master column (rename)
REPLACE_MAP = {
    "cancer_stage": "stage",
    "tnm_stage": "tnm",
    "t_stage": "t_stage",
    "n_stage": "n_stage",
    "m_stage": "m_stage",
    "smoking": "smoking_status",
    "drinking": "drinking_status",
}
# added as-is (same names)
NEW_COLS = ["cancer_stage_group", "prior_cancer_flag",
            "smoking_years", "smoking_per_day", "drinking_days"]


def backup(path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = path.with_name(f"{path.stem}.backup_{ts}{path.suffix}")
    shutil.copy2(path, dst)
    print(f"[backup] {dst}")
    return dst


def load_unified_slim() -> pd.DataFrame:
    cli = pd.read_csv(CLI_UNIFIED, encoding="cp949", low_memory=False)
    extracted = cli["solum_label"].astype(str).str.extract(r"^([A-Z]+)\s+(\d+)$")
    cli["spectral_group"] = extracted[0]
    cli["spectral_sample_id"] = pd.to_numeric(extracted[1], errors="coerce")
    cli = cli[cli["spectral_group"].isin(TARGET_GROUPS)].copy()
    # dedup per patient
    cli = cli.drop_duplicates(subset=["spectral_group", "spectral_sample_id"])

    keep = (["spectral_group", "spectral_sample_id"]
            + list(REPLACE_MAP.keys()) + NEW_COLS)
    keep = [c for c in keep if c in cli.columns]
    out = cli[keep].copy()
    print(f"[unified] slim patient-level rows: {len(out)}")
    return out


def main() -> None:
    backup(MASTER)

    m = pd.read_csv(MASTER, low_memory=False)
    print(f"[master] original shape: {m.shape}")

    u = load_unified_slim()

    # Ensure new columns exist in master (filled with NaN)
    for src_col, tgt_col in REPLACE_MAP.items():
        if tgt_col not in m.columns:
            m[tgt_col] = pd.NA
    for c in NEW_COLS:
        if c not in m.columns:
            m[c] = pd.NA

    # Key normalization
    m["spectral_sample_id"] = pd.to_numeric(m["spectral_sample_id"], errors="coerce")
    u["spectral_sample_id"] = pd.to_numeric(u["spectral_sample_id"], errors="coerce")

    # Merge: add suffixed columns from u, then overwrite master target columns
    u_renamed = u.rename(columns={src: f"_u_{tgt}" for src, tgt in REPLACE_MAP.items()})
    for c in NEW_COLS:
        if c in u_renamed.columns:
            u_renamed = u_renamed.rename(columns={c: f"_u_{c}"})

    merged = m.merge(u_renamed, on=["spectral_group", "spectral_sample_id"], how="left")

    # Apply replacements only where _u_ is non-null (preserves master for non-5-cancer rows)
    updated_rows_summary = {}
    for src_col, tgt_col in REPLACE_MAP.items():
        u_col = f"_u_{tgt_col}"
        if u_col not in merged.columns:
            continue
        mask = merged[u_col].notna()
        merged.loc[mask, tgt_col] = merged.loc[mask, u_col]
        updated_rows_summary[tgt_col] = int(mask.sum())
        merged.drop(columns=[u_col], inplace=True)
    for c in NEW_COLS:
        u_col = f"_u_{c}"
        if u_col not in merged.columns:
            continue
        mask = merged[u_col].notna()
        merged.loc[mask, c] = merged.loc[mask, u_col]
        updated_rows_summary[c] = int(mask.sum())
        merged.drop(columns=[u_col], inplace=True)

    print("\n[updated rows per column]")
    for k, v in updated_rows_summary.items():
        print(f"  {k:20s}: {v}")

    # Manual override: CRC 271..300 -> cancer_stage_group='advanced'
    crc_mask = ((merged["spectral_group"] == "CRC")
                & (merged["spectral_sample_id"].between(271, 300)))
    n_override = int(crc_mask.sum())
    merged.loc[crc_mask, "cancer_stage_group"] = "advanced"
    print(f"\n[override] CRC 271-300 -> cancer_stage_group='advanced': {n_override} rows")

    # Save
    merged.to_csv(MASTER, index=False, encoding="utf-8-sig")
    print(f"\n[saved] {MASTER}  shape={merged.shape}")

    # Quick summary
    print("\n[summary] cancer_stage_group on 5-cancer rows in master:")
    sub = merged[merged["spectral_group"].isin(TARGET_GROUPS)]
    print(sub.groupby(["spectral_group", "cancer_stage_group"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
