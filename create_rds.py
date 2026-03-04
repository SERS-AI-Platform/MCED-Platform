"""
create_rds.py — Build an RDS file combining raw SERS spectra with clinical info.

Output: data/sers_clinical.rds
  - Each row = one sample (averaged across replicates)
  - Columns: sample_id, group, category, hospital, clinical info, spectral intensities

Usage:
    python create_rds.py
"""

import sys
import re
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr
from scipy.interpolate import interp1d

# ── Setup ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sers.config import RAW_DATA_DIR, CLINICAL_DATA_DIR, load_config
from sers.io import read_spectrum, parse_filename, find_spectra

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────

# Common wavenumber grid (fingerprint region)
WN_MIN = 400.0
WN_MAX = 2200.0
WN_NPOINTS = 901  # 2 cm⁻¹ resolution

AVERAGED_FOLDER_NAMES = ["Averaged data", "Average data"]

# Clinical files to use (avoid duplicates — prefer *_patients.xlsx)
CLINICAL_FILES = {
    "Norm_patients.xlsx": "NOR",
    "DIA_patients.xlsx": "DIA",
    "HBP_patients.xlsx": "HBP",
    "H_D_patients.xlsx": "H.D.",
    "Lung_patients.xlsx": "LUN",
    "CRC_patients.xlsx": "CRC",
    "3. Colorectal cancer 임상정보.xlsx": None,  # skip (duplicate of CRC_patients)
    "2. Lung cancer 임상정보.xlsx": None,         # skip (duplicate of Lung_patients)
    "4. High blood pressure 임상정보.xlsx": None,  # skip (duplicate of HBP_patients)
    "5. Diabetes 임상정보.xlsx": None,             # skip (duplicate of DIA_patients)
}

# Patient ID column candidates
PID_CANDIDATES = [
    "SoluM Label", "SoluM label", "solum label",
    "Patient ID", "patient_id", "ID", "Label", "No", "No.", "번호",
]

# Core clinical columns to extract (Korean → English mapping)
# These are the most common across all clinical files
CLINICAL_COLUMN_MAP = {
    # Sex
    "성별": "sex",
    "제공자:성별": "sex",
    # Age
    "나이": "age",
    "제공자상세:최초참여시나이": "age",
    # Weight
    "체중": "weight",
    # Height
    "신장": "height",
    # Smoking
    "흡연력": "smoking",
    "흡연": "smoking",
    "흡연상태": "smoking",
    # Drinking
    "음주력": "drinking",
    "음주": "drinking",
    "음주상태": "drinking",
    # Diagnosis
    "진단명": "diagnosis",
    # Diagnosis date
    "진단일": "diagnosis_date",
    # Stage
    "Stage": "stage",
    # TNM
    "TNM": "tnm",
    # Metastasis
    "Metastasis": "metastasis",
    # Past history
    "과거력": "past_history",
    # Blood test
    "혈액검사결과": "blood_test",
    # BMI
    "BMI": "bmi",
    "체질량지수": "bmi",
    # Birth year (for computing age if 나이 is missing)
    "생년": "birth_year",
    # T stage
    "인체자원그룹:T STAGE": "t_stage",
    # N stage
    "인체자원그룹:N_STAGE2": "n_stage",
    # M stage
    "인체자원그룹:M_STAGE": "m_stage",
    # Comorbidities
    "과거 질병력\xa0(동반질환)": "comorbidity",
    # Surgery date
    "암 수술일": "surgery_date",
    # Fasting
    "공복여부": "fasting",
    # Sample collection date
    "검체수집일": "collection_date",
    "인체자원:자원접수일": "collection_date",
}


def make_wavenumber_grid():
    """Create the common wavenumber grid."""
    return np.linspace(WN_MIN, WN_MAX, WN_NPOINTS)


def load_averaged_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load an averaged spectrum file."""
    return read_spectrum(path)


def average_replicates(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """Average replicate spectra onto a common grid."""
    spectra = []
    for p in paths:
        try:
            x, y = read_spectrum(p)
            spectra.append((x, y))
        except Exception:
            continue

    if not spectra:
        raise ValueError("No valid spectra to average")

    # Use first spectrum's x as reference
    x_ref = spectra[0][0]
    y_all = []

    for x, y in spectra:
        if np.array_equal(x, x_ref):
            y_all.append(y)
        else:
            f = interp1d(x, y, kind="linear", bounds_error=False, fill_value=np.nan)
            y_all.append(f(x_ref))

    return x_ref, np.nanmean(y_all, axis=0)


def resample_to_grid(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Resample spectrum onto common wavenumber grid."""
    # Only interpolate within the data range
    mask = (grid >= x.min()) & (grid <= x.max())
    y_out = np.full(len(grid), np.nan)

    if mask.any():
        f = interp1d(x, y, kind="linear", bounds_error=False, fill_value=np.nan)
        y_out[mask] = f(grid[mask])

    return y_out


def load_all_spectra(config) -> pd.DataFrame:
    """Load all averaged spectra, resample to common grid, return as DataFrame."""
    grid = make_wavenumber_grid()
    folder_to_group = config.folder_to_group

    rows = []

    for folder_name, group_code in folder_to_group.items():
        folder_path = RAW_DATA_DIR / folder_name
        if not folder_path.is_dir():
            log.warning(f"Folder not found: {folder_path}")
            continue

        log.info(f"[{group_code}] {folder_name}")

        # Try averaged data first
        avg_dir = None
        for avg_name in AVERAGED_FOLDER_NAMES:
            candidate = folder_path / avg_name
            if candidate.is_dir():
                avg_dir = candidate
                break

        if avg_dir is not None:
            # Load pre-averaged files (deduplicate for case-insensitive OS)
            avg_files = sorted(
                {f.resolve(): f for f in
                 list(avg_dir.glob("*.CSV")) + list(avg_dir.glob("*.csv"))
                }.values()
            )
            for fpath in avg_files:
                stem = fpath.stem
                # Parse "GROUP ID_ave" pattern
                m = re.match(
                    r"^([A-Za-z]+(?:\.[A-Za-z]+)*\.?)\s*(\d+)_ave$",
                    stem, re.IGNORECASE,
                )
                if not m:
                    continue

                sample_id = int(m.group(2))
                sample_label = f"{group_code} {sample_id}"

                try:
                    x, y = read_spectrum(fpath)
                    y_grid = resample_to_grid(x, y, grid)
                    rows.append({
                        "sample_id": sample_label,
                        "group": group_code,
                        "sample_num": sample_id,
                        **{f"wn_{wn:.1f}": val for wn, val in zip(grid, y_grid)},
                    })
                except Exception as e:
                    log.warning(f"  Failed: {fpath.name}: {e}")

            log.info(f"  {len([r for r in rows if r['group'] == group_code])} averaged spectra loaded")
        else:
            # No averaged folder — compute average from replicates
            log.info(f"  No averaged folder, computing from replicates...")
            csv_files = sorted(
                {f.resolve(): f for f in
                 list(folder_path.glob("*.CSV")) + list(folder_path.glob("*.csv"))
                }.values()
            )

            # Group by sample_id
            sample_files: dict[int, list[Path]] = {}
            for fpath in csv_files:
                try:
                    spec_id = parse_filename(fpath, fallback_group=group_code)
                    sid = int(spec_id.sample_id)
                    sample_files.setdefault(sid, []).append(fpath)
                except ValueError:
                    continue

            for sid, paths in sorted(sample_files.items()):
                sample_label = f"{group_code} {sid}"
                try:
                    x, y = average_replicates(paths)
                    y_grid = resample_to_grid(x, y, grid)
                    rows.append({
                        "sample_id": sample_label,
                        "group": group_code,
                        "sample_num": sid,
                        **{f"wn_{wn:.1f}": val for wn, val in zip(grid, y_grid)},
                    })
                except Exception as e:
                    log.warning(f"  Failed to average {sample_label}: {e}")

            log.info(f"  {len([r for r in rows if r['group'] == group_code])} samples averaged from replicates")

    df = pd.DataFrame(rows)
    log.info(f"Total: {len(df)} samples loaded")
    return df


def load_clinical_data() -> pd.DataFrame:
    """Load clinical data from Excel files, extract core columns."""
    if not CLINICAL_DATA_DIR.is_dir():
        log.warning(f"Clinical data dir not found: {CLINICAL_DATA_DIR}")
        return pd.DataFrame()

    all_clinical = []

    for filename, group_code in CLINICAL_FILES.items():
        if group_code is None:
            continue  # skip duplicates

        fpath = CLINICAL_DATA_DIR / filename
        if not fpath.exists():
            log.warning(f"Clinical file not found: {fpath}")
            continue

        log.info(f"Loading clinical: {filename} → {group_code}")

        try:
            df = pd.read_excel(fpath, engine="openpyxl")
        except Exception as e:
            log.warning(f"  Failed to read: {e}")
            continue

        if df.empty:
            continue

        # Find patient ID column
        pid_col = None
        for cand in PID_CANDIDATES:
            if cand in df.columns:
                pid_col = cand
                break

        if pid_col is None:
            log.warning(f"  No patient ID column found in {filename}")
            continue

        # Extract mapped clinical columns
        clinical_rows = []
        for _, row in df.iterrows():
            raw_pid = row[pid_col]
            if pd.isna(raw_pid):
                continue
            sample_id = str(raw_pid).strip()

            clin = {"sample_id": sample_id}

            for src_col, dst_col in CLINICAL_COLUMN_MAP.items():
                if src_col in df.columns:
                    val = row[src_col]
                    if pd.notna(val):
                        clin[dst_col] = val

            clinical_rows.append(clin)

        if clinical_rows:
            clin_df = pd.DataFrame(clinical_rows)
            all_clinical.append(clin_df)
            log.info(f"  {len(clin_df)} clinical records loaded")

    if not all_clinical:
        return pd.DataFrame(columns=["sample_id"])

    # Concatenate all clinical data
    result = pd.concat(all_clinical, ignore_index=True)

    # Standardize sex column
    if "sex" in result.columns:
        sex_map = {"M": "M", "F": "F", "남": "M", "여": "F", "남자": "M", "여자": "F", "male": "M", "female": "F"}
        result["sex"] = result["sex"].astype(str).str.strip().map(
            lambda x: sex_map.get(x, x)
        )

    # Compute BMI if not present but weight and height are
    if "bmi" not in result.columns and "weight" in result.columns and "height" in result.columns:
        w = pd.to_numeric(result["weight"], errors="coerce")
        h = pd.to_numeric(result["height"], errors="coerce") / 100  # cm → m
        result["bmi"] = (w / (h ** 2)).round(1)
    elif "bmi" not in result.columns:
        result["bmi"] = np.nan

    log.info(f"Total clinical: {len(result)} records, {len(result.columns)} columns")
    return result


def main():
    log.info("=" * 60)
    log.info("Creating RDS file: raw spectra + clinical info")
    log.info("=" * 60)

    config = load_config()

    # Step 1: Load all spectra
    log.info("\n=== Step 1: Loading spectral data ===")
    spectra_df = load_all_spectra(config)

    # Step 2: Load clinical data
    log.info("\n=== Step 2: Loading clinical data ===")
    clinical_df = load_clinical_data()

    # Step 3: Merge
    log.info("\n=== Step 3: Merging spectral + clinical data ===")

    # Add category info from display.category_map
    spectra_df["category"] = spectra_df["group"].map(config.display.category_map)

    # Add hospital info from display.group_metadata
    hospital_map = {}
    for group_code, meta in config.display.group_metadata.items():
        if isinstance(meta, dict):
            if "hospital" in meta:
                hospital_map[group_code] = meta["hospital"]
            elif "hospitals" in meta:
                hospitals = [h.get("hospital", "") if isinstance(h, dict) else str(h)
                             for h in meta["hospitals"]]
                hospital_map[group_code] = "; ".join(hospitals)
    spectra_df["hospital"] = spectra_df["group"].map(hospital_map)

    # Merge with clinical data
    if not clinical_df.empty and "sample_id" in clinical_df.columns:
        merged = spectra_df.merge(clinical_df, on="sample_id", how="left")
    else:
        merged = spectra_df
        log.warning("No clinical data to merge")

    # Reorder columns: metadata first, then spectral data
    wn_cols = [c for c in merged.columns if c.startswith("wn_")]
    meta_cols = [c for c in merged.columns if not c.startswith("wn_")]
    merged = merged[meta_cols + wn_cols]

    # Sort by group and sample number
    group_order = config.display.group_order if hasattr(config.display, "group_order") else None
    if group_order:
        merged["_group_order"] = merged["group"].map(
            {g: i for i, g in enumerate(group_order)}
        )
        merged = merged.sort_values(["_group_order", "sample_num"]).drop(
            columns=["_group_order"]
        )
    else:
        merged = merged.sort_values(["group", "sample_num"])

    merged = merged.reset_index(drop=True)

    # Summary
    log.info(f"\nFinal dataset: {merged.shape[0]} samples × {merged.shape[1]} columns")
    log.info(f"  Spectral columns: {len(wn_cols)} ({WN_MIN}-{WN_MAX} cm⁻¹)")
    log.info(f"  Metadata columns: {len(meta_cols)}")
    log.info(f"\nSamples per group:")
    for grp, count in merged.groupby("group", sort=False).size().items():
        has_clin = merged[merged["group"] == grp]["sex"].notna().sum() if "sex" in merged.columns else 0
        log.info(f"  {grp:6s}: {count:4d} samples, {has_clin:4d} with clinical info")

    # Step 4: Save as RDS
    out_path = PROJECT_ROOT / "data" / "sers_clinical.rds"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"\n=== Step 4: Saving to {out_path} ===")
    pyreadr.write_rds(str(out_path), merged)
    log.info(f"RDS file saved: {out_path}")
    log.info(f"File size: {out_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Also save a CSV summary (metadata only, no spectral data)
    summary_path = PROJECT_ROOT / "data" / "sers_clinical_summary.csv"
    merged[meta_cols].to_csv(summary_path, index=False, encoding="utf-8-sig")
    log.info(f"Summary CSV saved: {summary_path}")

    log.info("\nDone!")


if __name__ == "__main__":
    main()
