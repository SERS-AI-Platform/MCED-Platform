"""
Extract representative SERS spectra for the grid analysis dashboard.

Reads one raw spectrum per cancer type (+ NOR), preprocesses on both
dynamic and fixed grids, and saves as JSON for dashboard embedding.

Usage:
    cd /home/user/SERS-AI
    python -m scripts.extract_dashboard_spectra
"""

import sys
import json
import logging
from pathlib import Path
from collections import OrderedDict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.sers.config import load_config, RAW_DATA_DIR
from src.sers.io import (
    find_spectra, read_spectrum, parse_filename,
    make_common_grid, make_fixed_grid,
)
from src.sers.preprocessing import (
    trim_spectrum, smooth, baseline_correction, normalize_spectrum,
    resample, FINGERPRINT_REGION,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Groups to extract (one spectrum each)
TARGET_GROUPS = ["PRO", "BRE", "OVA", "LUN", "CRC", "CPAN", "SPAN", "BLC", "NOR"]

# Max raw points for JSON (downsample)
RAW_DOWNSAMPLE_TARGET = 500


def preprocess_spectrum(x, y, grid, trim_region=FINGERPRINT_REGION):
    """Standard preprocessing: trim -> smooth -> baseline -> SNV -> resample."""
    x_t, y_proc = trim_spectrum(x, y.copy(), region=trim_region)
    y_proc = smooth(y_proc, window_length=11, polyorder=3)
    y_proc = baseline_correction(y_proc, window=101)
    y_proc = normalize_spectrum(y_proc, method="snv")
    y_grid = resample(x_t, y_proc, grid)
    return y_grid


def main():
    config = load_config()
    folder_to_group = config.folder_to_group

    # Reverse mapping: group -> folder name
    group_to_folder = {v: k for k, v in folder_to_group.items()}

    # --- Step 1: Find one representative file per group ---
    representative_files = {}
    all_x_arrays = []

    for group in TARGET_GROUPS:
        folder_name = group_to_folder.get(group)
        if folder_name is None:
            logger.warning(f"No folder mapping for group {group}, skipping")
            continue

        folder_path = RAW_DATA_DIR / folder_name
        if not folder_path.is_dir():
            logger.warning(f"Folder not found: {folder_path}")
            continue

        files = find_spectra(folder_path, pattern="*.csv", recursive=False)
        # Also try *.CSV
        if not files:
            files = find_spectra(folder_path, pattern="*.CSV", recursive=False)
        if not files:
            logger.warning(f"No CSV files in {folder_path}")
            continue

        # Pick the first file
        chosen = files[0]
        try:
            x, y = read_spectrum(chosen)
            spec_id = parse_filename(chosen, fallback_group=group)
            representative_files[group] = {
                "path": chosen,
                "sample_id": spec_id.sample_id,
                "x": x,
                "y": y,
            }
            all_x_arrays.append(x)
            logger.info(f"  {group}: {chosen.name} ({len(x)} points)")
        except Exception as e:
            logger.error(f"  Failed to read {chosen}: {e}")

    if not representative_files:
        logger.error("No spectra found!")
        sys.exit(1)

    logger.info(f"Found {len(representative_files)} representative spectra")

    # --- Step 2: Build grids ---
    # Dynamic grid: from all raw x arrays, trimmed to fingerprint
    dynamic_grid_full = make_common_grid(all_x_arrays)
    fp_mask = (dynamic_grid_full >= FINGERPRINT_REGION[0]) & (dynamic_grid_full <= FINGERPRINT_REGION[1])
    dynamic_grid = dynamic_grid_full[fp_mask]
    logger.info(f"Dynamic grid: {len(dynamic_grid)} points ({dynamic_grid[0]:.2f} - {dynamic_grid[-1]:.2f})")

    # Fixed grid: from config
    fixed_grid = make_fixed_grid(config)
    if fixed_grid is None:
        # Fallback: 402.0-2198.0, 935 points
        fixed_grid = np.linspace(402.0, 2198.0, 935)
    logger.info(f"Fixed grid: {len(fixed_grid)} points ({fixed_grid[0]:.2f} - {fixed_grid[-1]:.2f})")

    # --- Step 3: Build output data ---
    raw_spectra_out = []
    dynamic_spectra_out = []
    fixed_spectra_out = []

    for group in TARGET_GROUPS:
        if group not in representative_files:
            continue

        info = representative_files[group]
        x, y = info["x"], info["y"]
        sample_id = info["sample_id"]

        # Raw: downsample for JSON size
        n_raw = len(x)
        if n_raw > RAW_DOWNSAMPLE_TARGET:
            step = max(1, n_raw // RAW_DOWNSAMPLE_TARGET)
            x_ds = x[::step]
            y_ds = y[::step]
        else:
            x_ds = x
            y_ds = y

        raw_spectra_out.append({
            "group": group,
            "sample_id": sample_id,
            "x": [round(float(v), 2) for v in x_ds],
            "y": [round(float(v), 2) for v in y_ds],
        })

        # Preprocess on dynamic grid
        try:
            y_dyn = preprocess_spectrum(x, y, dynamic_grid)
            dynamic_spectra_out.append({
                "group": group,
                "sample_id": sample_id,
                "y": [round(float(v), 6) for v in y_dyn],
            })
        except Exception as e:
            logger.error(f"Dynamic preprocess failed for {group}: {e}")

        # Preprocess on fixed grid
        try:
            y_fix = preprocess_spectrum(x, y, fixed_grid)
            fixed_spectra_out.append({
                "group": group,
                "sample_id": sample_id,
                "y": [round(float(v), 6) for v in y_fix],
            })
        except Exception as e:
            logger.error(f"Fixed preprocess failed for {group}: {e}")

    # --- Step 4: Build JSON ---
    output = {
        "raw_spectra": raw_spectra_out,
        "dynamic_grid": {
            "grid": [round(float(v), 4) for v in dynamic_grid],
            "spectra": dynamic_spectra_out,
        },
        "fixed_grid": {
            "grid": [round(float(v), 4) for v in fixed_grid],
            "spectra": fixed_spectra_out,
        },
    }

    # Save
    out_path = PROJECT_ROOT / "results" / "grid_analysis" / "dashboard_spectra.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f)

    # Stats
    file_size_kb = out_path.stat().st_size / 1024
    logger.info(f"Saved to {out_path} ({file_size_kb:.1f} KB)")
    logger.info(f"  Raw spectra: {len(raw_spectra_out)} groups")
    logger.info(f"  Dynamic grid: {len(dynamic_grid)} points, {len(dynamic_spectra_out)} spectra")
    logger.info(f"  Fixed grid: {len(fixed_grid)} points, {len(fixed_spectra_out)} spectra")


if __name__ == "__main__":
    main()
