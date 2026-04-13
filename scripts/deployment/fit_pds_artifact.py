#!/usr/bin/env python3
"""
Fit and save PDS calibration artifacts for production models.

Fits Piecewise Direct Standardization (half_window=31, ridge=0.01) on ALL
paired Thermo↔Medical samples (n=1569), for BOTH directions, onto whatever
wavenumber grid the target production model uses.

Usage:
    python scripts/deployment/fit_pds_artifact.py  \
        --grid models/production_stacking/common_grid.npy  \
        --out  models/production_stacking/calibration/pds.npz

By default writes artifacts for BOTH production/ and production_stacking/ models.
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sers.calibration_transfer import PDSTransfer, paired_correlation, paired_rmse
from src.sers.io import find_spectra, parse_filename, read_spectrum
from src.sers.preprocessing import preprocess_single_spectrum
# Reuse constants + folder maps from the analysis script
from scripts.analysis.cross_instrument_calibration import (
    MEDICAL_FOLDER_MAP, MEDICAL_SHIFT, NOR_MAX_REPLICATE, THERMO_FOLDER_MAP,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

THERMO_DIR = PROJECT_ROOT / "data" / "raw_data"
MEDICAL_DIR = PROJECT_ROOT / "data" / "raw_data_medical"

BEST_PDS = {"half_window": 31, "ridge": 0.01}


def _preprocess_thermo(x, y, grid):
    return preprocess_single_spectrum(
        x, y, grid,
        do_trim=True, trim_region=(400, 2200),
        do_smooth=True, smooth_window=11, smooth_poly=3,
        do_baseline=True, baseline_window=101,
        normalization="snv",
    )


def _preprocess_medical(x, y, grid):
    return preprocess_single_spectrum(
        x, y, grid,
        do_trim=True, trim_region=(400, 2200),
        do_smooth=False,
        do_baseline=True, baseline_window=101,
        normalization="snv",
    )


def load_domain(data_dir, folder_map, pattern, prep_fn, grid,
                wavenumber_shift=0.0, read_from_subdir=None) -> pd.DataFrame:
    rows = []
    feat_cols = [f"x_{wn:.4f}" for wn in grid]
    for folder_name, group in folder_map.items():
        folder = data_dir / folder_name
        source = folder / read_from_subdir if read_from_subdir else folder
        if not source.is_dir():
            continue
        files = find_spectra(source, pattern=pattern, recursive=False)
        if not files and pattern == "*.csv":
            files = find_spectra(source, pattern="*.CSV", recursive=False)
        files = [f for f in files
                 if "_ave" not in f.stem.lower()
                 and "zone.identifier" not in f.name.lower()
                 and "multidata" not in f.stem.lower()]
        for fp in files:
            try:
                sid = parse_filename(fp, fallback_group=group)
                if sid.group == "NOR" and sid.replicate > NOR_MAX_REPLICATE:
                    continue
                x, y = read_spectrum(fp)
                if wavenumber_shift != 0.0:
                    x = x + wavenumber_shift
                y_proc = prep_fn(x, y, grid)
                row = {"group": sid.group, "sample_id": sid.sample_id,
                       "replicate": sid.replicate}
                row.update(dict(zip(feat_cols, y_proc)))
                rows.append(row)
            except Exception:
                pass
    return pd.DataFrame(rows)


def fit_and_save_pds(grid_path: Path, out_path: Path):
    logger.info(f"\n== Fitting PDS for grid: {grid_path}")
    grid = np.load(grid_path)
    logger.info(f"   grid: {grid.shape} [{grid[0]:.2f}, {grid[-1]:.2f}]")
    feat_cols = [f"x_{wn:.4f}" for wn in grid]

    logger.info("   loading Thermo …")
    df_t_rep = load_domain(THERMO_DIR, THERMO_FOLDER_MAP, "*.CSV",
                            _preprocess_thermo, grid)
    logger.info(f"     {len(df_t_rep)} spectra")

    logger.info("   loading Medical …")
    df_m_rep = load_domain(MEDICAL_DIR, MEDICAL_FOLDER_MAP, "*.txt",
                            _preprocess_medical, grid,
                            wavenumber_shift=MEDICAL_SHIFT,
                            read_from_subdir="Background")
    logger.info(f"     {len(df_m_rep)} spectra")

    # Aggregate to sample-level mean
    df_t = df_t_rep.groupby(["group", "sample_id"], as_index=False)[feat_cols].mean()
    df_m = df_m_rep.groupby(["group", "sample_id"], as_index=False)[feat_cols].mean()

    # Paired intersection
    key_t = set(zip(df_t["group"], df_t["sample_id"]))
    key_m = set(zip(df_m["group"], df_m["sample_id"]))
    paired_keys = sorted(key_t & key_m)
    logger.info(f"   paired samples: {len(paired_keys)}")

    df_t_idx = df_t.set_index(["group", "sample_id"])
    df_m_idx = df_m.set_index(["group", "sample_id"])
    X_t = df_t_idx.loc[paired_keys, feat_cols].values
    X_m = df_m_idx.loc[paired_keys, feat_cols].values

    # Fit bidirectional PDS on ALL paired samples (for production, use everything)
    logger.info(f"   fitting PDS w={BEST_PDS['half_window']} ridge={BEST_PDS['ridge']} …")
    pds_m2t = PDSTransfer(**BEST_PDS).fit(X_m, X_t)  # Medical → Thermo
    pds_t2m = PDSTransfer(**BEST_PDS).fit(X_t, X_m)  # Thermo → Medical

    # In-sample diagnostics (all paired, since no held-out here)
    rmse_raw = paired_rmse(X_m, X_t)
    corr_raw = paired_correlation(X_m, X_t)
    rmse_m2t = paired_rmse(pds_m2t.transform(X_m), X_t)
    corr_m2t = paired_correlation(pds_m2t.transform(X_m), X_t)
    rmse_t2m = paired_rmse(pds_t2m.transform(X_t), X_m)
    corr_t2m = paired_correlation(pds_t2m.transform(X_t), X_m)

    logger.info(f"   raw     RMSE={rmse_raw:.4f} corr={corr_raw:.4f}")
    logger.info(f"   M→T PDS RMSE={rmse_m2t:.4f} corr={corr_m2t:.4f}")
    logger.info(f"   T→M PDS RMSE={rmse_t2m:.4f} corr={corr_t2m:.4f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        grid=grid,
        F_medical_to_thermo=pds_m2t.F,
        mu_s_medical_to_thermo=pds_m2t.mu_s,
        mu_t_medical_to_thermo=pds_m2t.mu_t,
        F_thermo_to_medical=pds_t2m.F,
        mu_s_thermo_to_medical=pds_t2m.mu_s,
        mu_t_thermo_to_medical=pds_t2m.mu_t,
    )

    meta = {
        "version": "1.0",
        "fit_date": datetime.now().isoformat(timespec="seconds"),
        "method": "Piecewise Direct Standardization (PDS)",
        "hyperparameters": BEST_PDS,
        "n_paired_samples": int(len(paired_keys)),
        "n_features": int(grid.size),
        "grid_range_cm1": [float(grid[0]), float(grid[-1])],
        "medical_wavenumber_shift_cm1": MEDICAL_SHIFT,
        "in_sample_metrics": {
            "raw": {"rmse": rmse_raw, "mean_correlation": corr_raw},
            "medical_to_thermo": {"rmse": rmse_m2t, "mean_correlation": corr_m2t},
            "thermo_to_medical": {"rmse": rmse_t2m, "mean_correlation": corr_t2m},
        },
        "source_phase": "CIC-2 (2026-04-07) — PDS hp sweep best config",
        "training_cohort_source": "All paired samples from Thermo+Medical datasets (sample-level mean aggregation)",
    }
    meta_path = out_path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"   saved → {out_path}")
    logger.info(f"           {meta_path}")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=Path, default=None,
                    help="Specific common_grid.npy to fit to. If omitted, fits for "
                         "both production/ and production_stacking/.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output .npz path (required with --grid)")
    args = ap.parse_args()

    if args.grid is not None:
        if args.out is None:
            raise SystemExit("--out is required when --grid is given")
        fit_and_save_pds(args.grid, args.out)
    else:
        for prod_dir in ["production", "production_stacking"]:
            grid_path = PROJECT_ROOT / "models" / prod_dir / "common_grid.npy"
            out_path = PROJECT_ROOT / "models" / prod_dir / "calibration" / "pds.npz"
            if grid_path.exists():
                fit_and_save_pds(grid_path, out_path)


if __name__ == "__main__":
    main()
