"""Stage analysis - Step 1: Merge clinical + spectra.

Source:
  clinical: data/clinical_data/clinical_unified_202604172056.csv (encoding=cp949)
  spectra : results/processed_spectra.csv

Rules:
  - 5 cancers: CRC, CPAN, PRO, OVA, LUN (from solum_label prefix)
  - Override: CRC #271-#300 -> advanced (user-specified override)
  - unknown stage kept (excluded downstream per figure)

Output:
  AACR/data/stage_merged.csv         (long form, 1 row = 1 spectrum)
  AACR/data/stage_clinical_unique.csv (1 row = 1 patient)
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/home/user/SERS-AI")
CLI_PATH = ROOT / "data/clinical_data/clinical_unified_202604172056.csv"
SPECTRA_PATH = ROOT / "results/processed_spectra.csv"
OUT_DIR = ROOT / "AACR/data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_GROUPS = ["CRC", "CPAN", "PRO", "OVA", "LUN"]

def load_clinical() -> pd.DataFrame:
    cli = pd.read_csv(CLI_PATH, encoding="cp949", low_memory=False)
    # solum_label e.g. "CRC 271" -> split into prefix + number
    extracted = cli["solum_label"].astype(str).str.extract(r"^([A-Z]+)\s+(\d+)$")
    cli["group"] = extracted[0]
    cli["sid_num"] = pd.to_numeric(extracted[1], errors="coerce")
    cli = cli[cli["group"].isin(TARGET_GROUPS)].copy()

    # User override: CRC 271-300 -> advanced
    ov = (cli["group"] == "CRC") & (cli["sid_num"].between(271, 300))
    n_override = int(ov.sum())
    cli.loc[ov, "cancer_stage_group"] = "advanced"
    print(f"[override] CRC 271-300 -> advanced: {n_override} rows")

    # Dedup per patient (group, sid_num)
    cli_u = cli.drop_duplicates(subset=["group", "sid_num"]).copy()
    print(f"[clinical] rows (all timings): {len(cli)}, unique patients: {len(cli_u)}")
    return cli, cli_u


def load_spectra() -> pd.DataFrame:
    sp = pd.read_csv(SPECTRA_PATH)
    sp = sp[sp["group"].isin(TARGET_GROUPS)].copy()
    sp["sid_num"] = pd.to_numeric(sp["sample_id"], errors="coerce").astype("Int64")
    print(f"[spectra] rows in 5 cancers: {len(sp)}")
    return sp


def main() -> None:
    cli_all, cli_u = load_clinical()
    sp = load_spectra()

    keep_cols = ["group", "sid_num", "cancer_stage_group",
                 "age", "sex", "bmi", "tnm_stage", "cancer_stage"]
    merged = sp.merge(
        cli_u[keep_cols],
        on=["group", "sid_num"],
        how="left",
    )

    n_miss = merged["cancer_stage_group"].isna().sum()
    print(f"[merge] total spectra rows: {len(merged)}, unmerged (no clinical): {n_miss}")

    merged.to_csv(OUT_DIR / "stage_merged.csv", index=False, encoding="utf-8-sig")
    cli_u.to_csv(OUT_DIR / "stage_clinical_unique.csv", index=False, encoding="utf-8-sig")

    # Summary
    print("\n[summary] 5-cancer x stage group (unique patients):")
    xtab = cli_u.groupby(["group", "cancer_stage_group"]).size().unstack(fill_value=0)
    xtab = xtab.reindex(TARGET_GROUPS)
    for col in ["early", "advanced", "unknown"]:
        if col not in xtab.columns:
            xtab[col] = 0
    xtab = xtab[["early", "advanced", "unknown"]]
    xtab["total"] = xtab.sum(axis=1)
    print(xtab)
    xtab.to_csv(OUT_DIR / "stage_crosstab.csv", encoding="utf-8-sig")


if __name__ == "__main__":
    main()
