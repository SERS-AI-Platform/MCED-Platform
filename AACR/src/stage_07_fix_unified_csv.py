"""Stage analysis - Step 7: Fix original clinical_unified_202604172056.csv in place.

Fixes:
  1. CRC: 30 duplicated rows (CRC 1..30 appearing twice).
     Pool with LARGER id (id 830-range) -> rename to CRC 271..300
     with cancer_stage_group = 'advanced'.
     Mapping: id-largest row of each duplicate pair -> CRC (N+270).
  2. LUN: 170 rows with solum_label NaN (SMCXD06_폐암 2.xlsx).
     Assign them the 170 integers in 1..300 that are not already used
     by the 130 LUN rows that have solum_label.
     Assignment is ordered by row index, ascending.

Backup created in both WSL and Windows OneDrive locations before write.
Final output: both files overwritten with encoding=cp949 (original encoding).

Expected total after fix: CRC 300 + CPAN 70 + PRO 100 + OVA 70 + LUN 300 = 840 unique solum_label rows.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import shutil
import pandas as pd
import numpy as np

SRC_WSL = Path("/home/user/SERS-AI/data/clinical_data/clinical_unified_202604172056.csv")
SRC_WIN = Path("/mnt/c/Users/user/OneDrive - solum/바탕 화면/clinical_unified_202604172056.csv")


def backup(path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = path.with_name(f"{path.stem}.backup_{ts}{path.suffix}")
    shutil.copy2(path, dst)
    print(f"[backup] {dst}")
    return dst


def fix_crc(df: pd.DataFrame) -> dict:
    """Rename id-830-range duplicated CRC 1..30 rows -> CRC 271..300 + advanced."""
    col = df[df["disease_group"] == "COL"].copy()
    rename_info = []
    for n in range(1, 31):
        lbl = f"CRC {n}"
        rows = col[col["solum_label"] == lbl].sort_values("id", ascending=False)
        if len(rows) < 2:
            print(f"  warning: {lbl} has only {len(rows)} row(s); skipping")
            continue
        target_idx = rows.index[0]   # largest id (id 830-range)
        new_label = f"CRC {n + 270}"
        df.at[target_idx, "solum_label"] = new_label
        df.at[target_idx, "cancer_stage_group"] = "advanced"
        rename_info.append({
            "id": int(df.at[target_idx, "id"]),
            "old_label": lbl,
            "new_label": new_label,
        })
    return {"n_renamed": len(rename_info), "rename_info": rename_info}


def fix_lun(df: pd.DataFrame) -> dict:
    """Assign unused 1..300 numbers to the 170 LUN rows with NaN solum_label."""
    lun_mask = df["disease_group"] == "LUN"
    lun = df[lun_mask]

    existing = lun["solum_label"].dropna().astype(str).str.extract(r"LUN\s+(\d+)")[0]
    used_nums = set(pd.to_numeric(existing, errors="coerce").dropna().astype(int).tolist())
    all_nums = set(range(1, 301))
    avail = sorted(all_nums - used_nums)
    missing_idx = df.index[lun_mask & df["solum_label"].isna()].tolist()

    assert len(missing_idx) == len(avail), \
        f"LUN: missing {len(missing_idx)} rows vs {len(avail)} available numbers"

    for i, idx in enumerate(missing_idx):
        df.at[idx, "solum_label"] = f"LUN {avail[i]}"

    return {"n_assigned": len(missing_idx), "available_nums_used": avail}


def verify(df: pd.DataFrame) -> None:
    print("\n[verify] totals per disease_group:")
    print(df["disease_group"].value_counts())

    print("\n[verify] unique solum_label per target cancer prefix:")
    for g, prefix in [("COL", "CRC"), ("CPAN", "CPAN"), ("PRO", "PRO"),
                      ("OVA", "OVA"), ("LUN", "LUN")]:
        sub = df[df["disease_group"] == g]
        nonnull = sub["solum_label"].notna().sum()
        uniq = sub["solum_label"].nunique()
        print(f"  {prefix}: rows={len(sub)}, non-null solum_label={nonnull}, unique={uniq}")

    print("\n[verify] cancer_stage_group on 5-cancer:")
    five = df[df["disease_group"].isin(["COL", "CPAN", "PRO", "OVA", "LUN"])]
    xt = five.groupby(["disease_group", "cancer_stage_group"]).size().unstack(fill_value=0)
    print(xt)
    total = five["solum_label"].notna().sum()
    print(f"\n[verify] Total 5-cancer rows with non-null solum_label: {total} (expect 840)")


def main() -> None:
    backup(SRC_WSL)
    if SRC_WIN.exists():
        backup(SRC_WIN)

    df = pd.read_csv(SRC_WSL, encoding="cp949", low_memory=False)

    r_crc = fix_crc(df)
    print(f"\n[CRC fix] renamed {r_crc['n_renamed']} rows (CRC 1..30 -> CRC 271..300)")
    for info in r_crc["rename_info"][:5]:
        print(f"  id={info['id']}  {info['old_label']} -> {info['new_label']}")
    print("  ...")

    r_lun = fix_lun(df)
    print(f"\n[LUN fix] assigned solum_label to {r_lun['n_assigned']} rows")
    print(f"  number range used (first 10): {r_lun['available_nums_used'][:10]}")

    verify(df)

    df.to_csv(SRC_WSL, index=False, encoding="cp949")
    print(f"\n[saved] {SRC_WSL}")
    if SRC_WIN.exists():
        df.to_csv(SRC_WIN, index=False, encoding="cp949")
        print(f"[saved] {SRC_WIN}")


if __name__ == "__main__":
    main()
