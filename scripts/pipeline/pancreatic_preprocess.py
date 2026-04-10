"""
Pancreatic Cancer Binary Classification - Data Preprocessing

Builds a dataset for PAN vs Normal binary classification:
- PAN: CPAN (70 patients) + YPAN (30 patients) = 100 patients
- Normal: NOR (100 patients) + YNOR (29 patients) = 129 patients

Includes wavenumber calibration correction:
- Detects reference peak (~1001 cm⁻¹, urea symmetric stretch)
- Shifts each spectrum's x-axis to align reference peak
- Eliminates instrument calibration drift between institutions

Uses the same QC/preprocessing pipeline as the base model.
Output: results/pancreatic/processed_spectra.csv, metadata.csv
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.sers.config import load_config
from src.sers.io import find_spectra, read_spectrum, parse_filename, make_common_grid, make_fixed_grid
from src.sers.preprocessing import preprocess_spectra, save_processed_spectra

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────
RAW_DATA_DIR = Path("data/raw_data")
OUTPUT_DIR = Path("results/pancreatic")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Folders to include and their group mappings
FOLDER_MAP = {
    "10-1. C-Pancreatic cancer (70개)": "CPAN",
    "10-3. Y-Pancreatic cancer (YPAN)": "YPAN",
    "5. Normal (100개)": "NOR",
    "12. Y-Normal (YNOR)": "YNOR",
}

# Binary label mapping: PAN (cancer) vs NOR (normal)
BINARY_MAP = {
    "CPAN": "PAN",
    "YPAN": "PAN",
    "NOR": "NOR",
    "YNOR": "NOR",
}


# ── Wavenumber Calibration ─────────────────────────────────────────
REFERENCE_PEAK_WN = 1001.4  # Urea symmetric C-N stretch (literature value)
SEARCH_WINDOW = 20.0        # ± cm⁻¹ search range


def find_reference_peak(x, y, target_wn=REFERENCE_PEAK_WN, window=SEARCH_WINDOW):
    """
    Find the position of the reference peak nearest to target_wn.

    Uses light smoothing + peak detection in a local window.
    Returns the detected peak position in cm⁻¹, or None if not found.
    """
    mask = (x >= target_wn - window) & (x <= target_wn + window)
    x_win = x[mask]
    y_win = y[mask]
    if len(x_win) < 5:
        return None

    # Light smoothing to avoid noise peaks
    if len(y_win) >= 7:
        y_smooth = savgol_filter(y_win, window_length=7, polyorder=2)
    else:
        y_smooth = y_win

    peaks, props = find_peaks(y_smooth, distance=5)
    if len(peaks) > 0:
        best = peaks[np.argmax(y_smooth[peaks])]
        # Sub-pixel refinement: parabolic interpolation around peak
        if 1 <= best < len(x_win) - 1:
            y0, y1, y2 = y_smooth[best - 1], y_smooth[best], y_smooth[best + 1]
            denom = 2 * (2 * y1 - y0 - y2)
            if abs(denom) > 1e-10:
                offset = (y0 - y2) / denom
                return x_win[best] + offset * (x_win[1] - x_win[0])
        return float(x_win[best])
    else:
        # Fallback: just use max
        return float(x_win[np.argmax(y_smooth)])


def calibrate_spectrum(x, y, target_wn=REFERENCE_PEAK_WN):
    """
    Shift x-axis so that the reference peak aligns to target_wn.
    Returns (x_shifted, y, shift_applied).
    """
    detected = find_reference_peak(x, y, target_wn)
    if detected is None:
        return x, y, 0.0
    shift = target_wn - detected
    return x + shift, y, shift


def main():
    logger.info("=" * 60)
    logger.info("Pancreatic Cancer - Data Preprocessing")
    logger.info("  (with wavenumber calibration correction)")
    logger.info("=" * 60)

    config = load_config("config/config.yaml")

    # ── Step 1: Load raw spectra ───────────────────────────────────
    raw_spectra = {}
    metadata_rows = []

    for folder_name, group in FOLDER_MAP.items():
        folder_path = RAW_DATA_DIR / folder_name
        if not folder_path.exists():
            logger.warning(f"Folder not found: {folder_path}")
            continue

        files = find_spectra(folder_path, recursive=False)
        logger.info(f"  {group}: {len(files)} files in {folder_name}")

        for fp in files:
            try:
                spec_id = parse_filename(fp, fallback_group=group)
                x, y = read_spectrum(fp)
                key = (spec_id.group, spec_id.sample_id, spec_id.replicate)
                raw_spectra[key] = (x, y)
                metadata_rows.append({
                    "file": fp.name,
                    "group": spec_id.group,
                    "sample_id": spec_id.sample_id,
                    "replicate": spec_id.replicate,
                    "n_points": len(x),
                    "x_min": x.min(),
                    "x_max": x.max(),
                })
            except Exception as e:
                logger.warning(f"  Failed: {fp.name}: {e}")

    meta_df = pd.DataFrame(metadata_rows)
    logger.info(f"\nLoaded {len(raw_spectra)} spectra total")
    logger.info(f"Group counts:\n{meta_df.groupby('group')['sample_id'].nunique()}")

    # ── Step 1.5: Wavenumber calibration correction ────────────────
    logger.info(f"\n[Calibration] Reference peak: {REFERENCE_PEAK_WN} cm⁻¹ (urea C-N stretch)")
    shifts_by_group = {}
    calibrated_spectra = {}
    cal_failed = 0

    for key, (x, y) in raw_spectra.items():
        x_cal, y_cal, shift = calibrate_spectrum(x, y)
        calibrated_spectra[key] = (x_cal, y_cal)
        group = key[0]
        shifts_by_group.setdefault(group, []).append(shift)

    for group, shifts in sorted(shifts_by_group.items()):
        shifts = np.array(shifts)
        logger.info(f"  {group:>4s}: mean shift = {shifts.mean():+.4f} cm⁻¹, "
                     f"std = {shifts.std():.4f}, range = [{shifts.min():+.4f}, {shifts.max():+.4f}]")

    raw_spectra = calibrated_spectra

    # ── Step 2: Build common grid & preprocess ─────────────────────
    fixed = make_fixed_grid(config)
    if fixed is not None:
        grid = fixed
        logger.info(f"Using fixed grid: {len(grid)} points ({grid[0]:.1f} – {grid[-1]:.1f} cm⁻¹)")
    else:
        x_arrays = [v[0] for v in raw_spectra.values()]
        grid = make_common_grid(x_arrays)

    processed, stats_df, proc_grid = preprocess_spectra(
        raw_spectra, grid, config, qc_passed_keys=None
    )
    logger.info(f"Preprocessed {len(processed)} spectra on {len(proc_grid)}-point grid")

    # ── Step 3: Save with binary labels ────────────────────────────
    rows = []
    for key, y_proc in processed.items():
        group, sid, rep = key
        binary_label = BINARY_MAP[group]
        row = {
            "group": group,
            "binary_label": binary_label,
            "sample_id": sid,
            "replicate": rep,
        }
        for i, xv in enumerate(proc_grid):
            row[f"x_{xv:.2f}"] = y_proc[i]
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "processed_spectra.csv", index=False)

    # Save metadata
    meta_df["binary_label"] = meta_df["group"].map(BINARY_MAP)
    meta_df.to_csv(OUTPUT_DIR / "metadata.csv", index=False)

    # Summary
    summary = df.groupby(["binary_label", "group"]).agg(
        patients=("sample_id", "nunique"),
        spectra=("sample_id", "count"),
    )
    logger.info(f"\n=== Dataset Summary ===\n{summary}")
    logger.info(f"\nTotal: {df['sample_id'].nunique()} patients, {len(df)} spectra")
    logger.info(f"Output: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
