"""
Automated Clinical Data Standardization Pipeline.

Reads all clinical .xlsx/.xlsm files from configured cancer groups,
standardizes column names (Korean -> English), and outputs a unified
all_clinical_standardized.csv plus per-disease CSVs.

This script wraps the per-disease loaders from
data/clinical_data/standardize_clinical.py, driven by config/config.yaml
group definitions. It can be run repeatedly — output is fully regenerated
each time.

Usage:
    python scripts/standardize_clinical_data.py
    python scripts/standardize_clinical_data.py --output-dir /custom/path
    python scripts/standardize_clinical_data.py --dry-run

Output:
    data/clinical_data/standardized/all_clinical_standardized.csv
    data/clinical_data/standardized/{GROUP}_clinical_standardized.csv  (per disease)

Requires:
    pandas, openpyxl, pyyaml
"""

import argparse
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — ensure we can import from src/ and data/clinical_data/
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "data" / "clinical_data"))

from sers.config import load_config, CLINICAL_DATA_DIR

# Import the per-disease loaders from the existing standardize_clinical.py
from standardize_clinical import (
    STD_COLUMNS,
    load_smcxd03,
    load_smcxd01_compact,
    load_smcxd06_cancer,
    load_lung2,
    load_lung3,
    normalize_date,
    safe_float,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File registry: maps clinical files to their loader function + params
# ---------------------------------------------------------------------------
# Each entry: (relative_path, loader_function, loader_kwargs)
# relative_path is under CLINICAL_DATA_DIR


def build_file_registry(clinical_dir: Path) -> list[dict]:
    """Build the registry of clinical files and their loaders.

    This is the single source of truth for which files exist and how
    to read them. When a new disease group is added, add an entry here.
    """
    base = clinical_dir
    registry = []

    # ── SMCXD03/05: 546-column health screening format ──────────────────
    smcxd03_entries = [
        ("5. 정상/SMCXD03_정상인 1.xlsx", "정상인", "NOR", "NOR"),
        ("5. 정상/SMCXD03_정상인 2.xlsx", "정상인", "NOR", "NOR"),
        ("6. 당뇨/SMCXD03_당뇨 1.xlsx", "당뇨", "DIA", "DIA"),
        ("6. 당뇨/SMCXD03_당뇨 2.xlsx", "당뇨", "DIA", "DIA"),
        ("7. 고혈압/SMCXD03_고혈압.xlsx", "고혈압", "HBP", "HBP"),
        ("8. 당뇨 + 고혈압/SMCXD05_당뇨+고혈압.xlsx", "당뇨+고혈압", "H.D.", "HD"),
    ]
    for rel_path, sheet, group, prefix in smcxd03_entries:
        registry.append({
            "path": base / rel_path,
            "group": group,
            "loader": "smcxd03",
            "kwargs": {"sheet": sheet, "disease_group": group, "id_prefix": prefix},
        })

    # ── SMCXD01: Compact cancer format ──────────────────────────────────
    smcxd01_entries = [
        ("1. 전립선암/SMCXD01_전립선암 임상정보.xlsx", "Sheet1", "PRO"),
        ("2. 유방암/SMCXD01_유방암.xlsx", "C50 임상정보", "BRE"),
        ("3. 난소암/SMCXD01_난소암 1.xlsx", "C56 임상정보", "OVA"),
        ("3. 난소암/SMCXD01_난소암 2.xlsx", "난소", "OVA"),
        ("4. 폐암/SMCXD01_폐암 1.xlsx", "폐", "LUN"),
    ]
    for rel_path, sheet, group in smcxd01_entries:
        registry.append({
            "path": base / rel_path,
            "group": group,
            "loader": "smcxd01_compact",
            "kwargs": {"sheet": sheet, "disease_group": group},
        })

    # ── Lung cancer special formats ─────────────────────────────────────
    registry.append({
        "path": base / "4. 폐암/SMCXD06_폐암 2.xlsx",
        "group": "LUN",
        "loader": "lung2",
        "kwargs": {},
    })
    registry.append({
        "path": base / "4. 폐암/SMCXD06_폐암 3.xlsx",
        "group": "LUN",
        "loader": "lung3",
        "kwargs": {},
    })

    # ── SMCXD06: Mid-form cancer format ─────────────────────────────────
    smcxd06_entries = [
        ("9. 대장암/SMCXD06_대장암.xlsx", "대장암1~2기270명", "CRC", 0),
        ("9. 대장암/SMCXD06_대장암.xlsx", "대장암3~4기30명", "CRC", 0),
        ("10. 췌장암/CPAN/SMCMD06_췌장암.xlsx", "췌장암70명", "PAN", 0),
        ("11. 방광암/SMCXD06_방광암.xlsm", "분양명단", "BLA", 1),
    ]
    for rel_path, sheet, group, hrow in smcxd06_entries:
        registry.append({
            "path": base / rel_path,
            "group": group,
            "loader": "smcxd06_cancer",
            "kwargs": {"sheet": sheet, "disease_group": group, "header_row": hrow},
        })

    return registry


# ---------------------------------------------------------------------------
# Loader dispatch
# ---------------------------------------------------------------------------
LOADERS = {
    "smcxd03": load_smcxd03,
    "smcxd01_compact": load_smcxd01_compact,
    "smcxd06_cancer": load_smcxd06_cancer,
    "lung2": load_lung2,
    "lung3": load_lung3,
}


def run_loader(entry: dict) -> pd.DataFrame | None:
    """Run a single file loader. Returns DataFrame or None on failure."""
    filepath = entry["path"]
    loader_name = entry["loader"]
    kwargs = entry["kwargs"]

    if not filepath.exists():
        log.warning(f"MISSING: {filepath}")
        return None

    loader_fn = LOADERS[loader_name]

    try:
        if loader_name in ("lung2", "lung3"):
            df = loader_fn(str(filepath), **kwargs)
        else:
            df = loader_fn(str(filepath), **kwargs)
        return df
    except Exception as e:
        log.error(f"FAILED loading {filepath.name}: {e}")
        return None


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------
def postprocess(merged: pd.DataFrame) -> pd.DataFrame:
    """Apply domain-specific fixes and computed columns."""

    # 1. Fix PRO (prostate): height_cm and weight_kg are swapped in source
    pro_mask = merged["disease_group"] == "PRO"
    if pro_mask.any():
        pro_h = merged.loc[pro_mask, "height_cm"].copy()
        pro_w = merged.loc[pro_mask, "weight_kg"].copy()
        merged.loc[pro_mask, "height_cm"] = pro_w
        merged.loc[pro_mask, "weight_kg"] = pro_h
        merged.loc[pro_mask, "bmi"] = None
        log.info(f"[FIX] PRO height/weight swapped for {pro_mask.sum()} rows")

    # 2. Compute BMI where missing
    mask = (
        merged["bmi"].isna()
        & merged["weight_kg"].notna()
        & merged["height_cm"].notna()
    )
    if mask.any():
        h_m = merged.loc[mask, "height_cm"] / 100
        merged.loc[mask, "bmi"] = (merged.loc[mask, "weight_kg"] / (h_m ** 2)).round(1)
        log.info(f"[BMI] Computed BMI for {mask.sum()} rows")

    # 3. Compute sample_timing (pre-op / post-op / peri-op / control)
    merged["sample_timing"] = None
    control_groups = {"NOR", "DIA", "HBP", "H.D."}

    for idx, row in merged.iterrows():
        sample_dt = pd.to_datetime(row.get("sample_date"), errors="coerce")
        if pd.isna(sample_dt):
            if row.get("disease_group") in control_groups:
                merged.at[idx, "sample_timing"] = "control"
            continue

        if row.get("disease_group") in control_groups:
            merged.at[idx, "sample_timing"] = "control"
            continue

        # Try surgery_date first
        ref_dt = pd.to_datetime(row.get("surgery_date"), errors="coerce")

        # BLA: parse surgery date from treatment text
        if pd.isna(ref_dt) and row.get("disease_group") == "BLA":
            treatment_text = row.get("treatment")
            if pd.notna(treatment_text):
                surg_matches = re.findall(
                    r"(\d{8})\s*-?\s*수술", str(treatment_text)
                )
                if surg_matches:
                    try:
                        ref_dt = pd.to_datetime(surg_matches[0], format="%Y%m%d")
                    except Exception:
                        pass

        # Fallback to diagnosis_date
        if pd.isna(ref_dt):
            ref_dt = pd.to_datetime(row.get("diagnosis_date"), errors="coerce")

        if pd.isna(ref_dt):
            continue

        delta_days = (ref_dt - sample_dt).days
        if delta_days > 7:
            merged.at[idx, "sample_timing"] = "pre-op"
        elif delta_days < -7:
            merged.at[idx, "sample_timing"] = "post-op"
        else:
            merged.at[idx, "sample_timing"] = "peri-op"

    return merged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Standardize all clinical data into unified format"
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "data" / "clinical_data" / "standardized"),
        help="Output directory for standardized CSVs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files to process without actually writing output",
    )
    args = parser.parse_args()

    t_start = time.time()
    log.info("=" * 60)
    log.info("Clinical Data Standardization Pipeline")
    log.info("=" * 60)

    # Load project config
    config = load_config()
    configured_groups = set(config.folder_to_group.values())
    log.info(f"Configured groups from config.yaml: {sorted(configured_groups)}")

    # Build file registry
    clinical_dir = CLINICAL_DATA_DIR
    log.info(f"Clinical data directory: {clinical_dir}")
    registry = build_file_registry(clinical_dir)
    log.info(f"File registry: {len(registry)} entries")

    if args.dry_run:
        log.info("\n[DRY RUN] Files to process:")
        for entry in registry:
            exists = entry["path"].exists()
            status = "OK" if exists else "MISSING"
            log.info(f"  [{status}] {entry['path'].name} -> {entry['group']}")
        return

    # Process each file
    all_dfs = []
    issues = []

    for entry in registry:
        filepath = entry["path"]
        group = entry["group"]

        log.info(f"  Loading {filepath.name} [{group}] ...")
        df = run_loader(entry)

        if df is None:
            issues.append(f"SKIPPED: {filepath.name} (missing or failed)")
            continue

        if df.empty:
            issues.append(f"EMPTY: {filepath.name}")
            continue

        all_dfs.append(df)
        log.info(f"    -> {len(df)} rows")

    if not all_dfs:
        log.error("No data loaded! Check file paths.")
        sys.exit(1)

    # Merge
    log.info(f"\nMerging {len(all_dfs)} dataframes...")
    merged = pd.concat(all_dfs, ignore_index=True)

    # Ensure all standard columns exist
    for col in STD_COLUMNS:
        if col not in merged.columns:
            merged[col] = None

    # Reorder to standard column order
    merged = merged[STD_COLUMNS]

    # Post-processing
    log.info("Applying post-processing...")
    merged = postprocess(merged)

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Full merged CSV
    merged_path = out_dir / "all_clinical_standardized.csv"
    merged.to_csv(merged_path, index=False, encoding="utf-8-sig")
    log.info(f"\nMerged output: {merged_path}")
    log.info(f"  Total: {len(merged)} rows x {len(merged.columns)} cols")

    # Per-disease CSVs
    for group, gdf in merged.groupby("disease_group"):
        gpath = out_dir / f"{group}_clinical_standardized.csv"
        gdf.to_csv(gpath, index=False, encoding="utf-8-sig")
        log.info(f"  {group}: {len(gdf)} rows -> {gpath.name}")

    # Summary report
    log.info(f"\n{'=' * 60}")
    log.info("SUMMARY")
    log.info(f"{'=' * 60}")

    # Group counts
    log.info("\nRows per disease group:")
    for group, count in merged["disease_group"].value_counts().sort_index().items():
        in_config = group in configured_groups or group in ("BLA", "PAN")
        flag = "" if in_config else " [NOT IN CONFIG]"
        log.info(f"  {group:6s}: {count:5d}{flag}")

    # Column fill rates
    log.info("\nColumn fill rates (non-null):")
    for col in STD_COLUMNS:
        n = merged[col].notna().sum()
        pct = n / len(merged) * 100
        if pct > 0:
            log.info(f"  {col:25s}: {n:5d}/{len(merged)} ({pct:5.1f}%)")

    # Sample timing summary
    log.info("\nSample timing:")
    timing_counts = merged["sample_timing"].value_counts(dropna=False)
    for t, c in timing_counts.items():
        label = t if pd.notna(t) else "unknown"
        log.info(f"  {label:12s}: {c}")

    # Issues
    if issues:
        log.info(f"\nISSUES ({len(issues)}):")
        for issue in issues:
            log.warning(f"  {issue}")

    elapsed = time.time() - t_start
    log.info(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
