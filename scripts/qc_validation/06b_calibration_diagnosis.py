"""
Phase 6b — Calibration Failure Diagnosis

목적: 현재 calibration이 |shift|>5 cm⁻¹로 잡은 22개 케이스를 시각화해서
"잘못 잡힌 peak"가 어떤 모양/위치에 있는지 파악한다.
이 정보로 어떤 가드(prominence/2nd-deriv/window)가 적합한지 결정한다.

산출:
    results/preprocessing_validation/calibration_diagnosis_grid.png
    results/preprocessing_validation/calibration_failure_summary.csv
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks, peak_prominences

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.sers.config import load_config, RAW_DATA_DIR, RESULTS_DIR
from src.sers import read_spectrum, parse_filename
from src.sers.io import find_spectra

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("cal_diag")

OUT_DIR = Path(RESULTS_DIR) / "preprocessing_validation"
SHIFT_CSV = OUT_DIR / "preproc_calibration_shifts_w20.csv"
TARGET_WN = 1001.4
WINDOW = 20.0
PLOT_HALFWIDTH = 50.0  # cm⁻¹ around target for plotting
LARGE_SHIFT = 5.0


def load_one(config, group: str, sid, rep) -> tuple[np.ndarray, np.ndarray] | None:
    """Find and read a single spectrum file by (group, sample_id, replicate)."""
    data_dir = Path(RAW_DATA_DIR)
    for fp in find_spectra(data_dir, pattern="*.csv"):
        try:
            spec_id = parse_filename(
                fp,
                fallback_group=config.folder_to_group.get(fp.parent.name, "UNK"),
            )
        except Exception:
            continue
        if (spec_id.group == group and str(spec_id.sample_id) == str(sid)
                and int(spec_id.replicate) == int(rep)):
            return read_spectrum(fp)
    return None


def analyze_window(x: np.ndarray, y: np.ndarray) -> dict:
    """Replicate current find_reference_peak logic + capture all candidates."""
    mask = (x >= TARGET_WN - WINDOW) & (x <= TARGET_WN + WINDOW)
    x_win, y_win = x[mask], y[mask]
    if len(x_win) < 5:
        return {}
    y_smooth = savgol_filter(y_win, window_length=7, polyorder=2) if len(y_win) >= 7 else y_win
    peaks, _ = find_peaks(y_smooth, distance=5)
    if len(peaks) == 0:
        idx = int(np.argmax(y_smooth))
        return dict(picked_x=float(x_win[idx]), picked_y=float(y_smooth[idx]),
                    n_candidates=0, prominence=np.nan, picked_idx_in_peaks=-1,
                    candidates_x=[], candidates_y=[], candidates_prom=[])
    proms, _, _ = peak_prominences(y_smooth, peaks)
    picked_local = peaks[int(np.argmax(y_smooth[peaks]))]
    return dict(
        picked_x=float(x_win[picked_local]),
        picked_y=float(y_smooth[picked_local]),
        n_candidates=int(len(peaks)),
        prominence=float(proms[int(np.argmax(y_smooth[peaks]))]),
        candidates_x=[float(x_win[p]) for p in peaks],
        candidates_y=[float(y_smooth[p]) for p in peaks],
        candidates_prom=[float(p) for p in proms],
        x_win=x_win, y_win=y_win, y_smooth=y_smooth,
    )


def main():
    if not SHIFT_CSV.exists():
        raise FileNotFoundError(f"{SHIFT_CSV} not found — run 06_preprocessing_validation.py first")

    df = pd.read_csv(SHIFT_CSV)
    failures = df[np.abs(df["shift_cm1"]) > LARGE_SHIFT].copy()
    logger.info(f"Loaded {len(df)} shifts; {len(failures)} with |shift|>{LARGE_SHIFT}")

    if len(failures) == 0:
        logger.info("No failures to diagnose.")
        return

    config = load_config(str(PROJECT_ROOT / "config" / "config.yaml"))

    # Plot grid
    n = len(failures)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).flatten()

    summary_rows = []
    for ax, (_, row) in zip(axes, failures.iterrows()):
        spec = load_one(config, row["group"], row["sample_id"], row["replicate"])
        if spec is None:
            ax.set_title(f"{row['group']}-{row['sample_id']}-{row['replicate']}\nFILE NOT FOUND")
            ax.axis("off")
            continue
        x, y = spec
        info = analyze_window(x, y)

        # plot wider context
        plot_mask = (x >= TARGET_WN - PLOT_HALFWIDTH) & (x <= TARGET_WN + PLOT_HALFWIDTH)
        ax.plot(x[plot_mask], y[plot_mask], "k-", lw=0.8, label="raw")
        if "y_smooth" in info:
            ax.plot(info["x_win"], info["y_smooth"], "b-", lw=1.0, label="smooth")
        ax.axvline(TARGET_WN, color="green", ls="--", lw=0.8, label="target 1001.4")
        ax.axvspan(TARGET_WN - WINDOW, TARGET_WN + WINDOW, alpha=0.08, color="orange")
        if info:
            ax.axvline(info["picked_x"], color="red", ls=":", lw=1.2, label="picked")
            for cx, cy in zip(info.get("candidates_x", []), info.get("candidates_y", [])):
                ax.plot(cx, cy, "ro", ms=3, alpha=0.4)
        ax.set_title(
            f"{row['group']}-{row['sample_id']}-{row['replicate']}  "
            f"shift={row['shift_cm1']:+.1f}\n"
            f"picked={info.get('picked_x', float('nan')):.1f}  "
            f"prom={info.get('prominence', float('nan')):.1f}",
            fontsize=8,
        )
        ax.tick_params(labelsize=7)
        if ax is axes[0]:
            ax.legend(fontsize=6, loc="upper right")

        summary_rows.append({
            "group": row["group"], "sample_id": row["sample_id"],
            "replicate": row["replicate"], "shift_cm1": row["shift_cm1"],
            "picked_x": info.get("picked_x"),
            "n_candidates": info.get("n_candidates"),
            "prominence": info.get("prominence"),
            "max_y_in_window": float(np.max(info["y_win"])) if "y_win" in info else None,
        })

    for ax in axes[len(failures):]:
        ax.axis("off")

    fig.suptitle(f"Calibration failures (|shift|>{LARGE_SHIFT} cm⁻¹) — {len(failures)} cases", fontsize=11)
    fig.tight_layout()
    out_png = OUT_DIR / "calibration_diagnosis_grid.png"
    fig.savefig(out_png, dpi=130)
    logger.info(f"  → {out_png}")

    sum_df = pd.DataFrame(summary_rows)
    sum_csv = OUT_DIR / "calibration_failure_summary.csv"
    sum_df.to_csv(sum_csv, index=False)
    logger.info(f"  → {sum_csv}")

    # Quick stats
    if "prominence" in sum_df:
        logger.info(f"\nFailure prominence stats:")
        logger.info(f"  median={sum_df['prominence'].median():.1f}  "
                    f"mean={sum_df['prominence'].mean():.1f}  "
                    f"max={sum_df['prominence'].max():.1f}")
        logger.info(f"  # candidates: median={sum_df['n_candidates'].median():.0f}  "
                    f"max={sum_df['n_candidates'].max():.0f}")


if __name__ == "__main__":
    main()
