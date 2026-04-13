"""Analyze wavenumber grid properties of all raw spectrum CSV files."""
import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR = Path("/home/user/SERS-AI/data/raw_data")
OUT_DIR = Path("/home/user/SERS-AI/results/grid_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

rows = []

for folder in sorted(RAW_DIR.iterdir()):
    if not folder.is_dir():
        continue
    # Find all .csv/.CSV files (excluding subdirectories like "Averaged data")
    csv_files = []
    for f in folder.iterdir():
        if f.is_file() and f.suffix.lower() == ".csv":
            csv_files.append(f)
    csv_files.sort()

    for fpath in csv_files:
        try:
            df = pd.read_csv(fpath, sep=None, engine="python", header=None, usecols=[0, 1])
            df.columns = ["raman_shift", "intensity"]
            df["raman_shift"] = pd.to_numeric(df["raman_shift"], errors="coerce")
            df["intensity"] = pd.to_numeric(df["intensity"], errors="coerce")
            df = df.dropna(subset=["raman_shift"]).sort_values("raman_shift").reset_index(drop=True)

            n_pts = len(df)
            x_min = df["raman_shift"].min()
            x_max = df["raman_shift"].max()
            steps = np.diff(df["raman_shift"].values)
            mean_step = steps.mean() if len(steps) > 0 else np.nan
            min_step = steps.min() if len(steps) > 0 else np.nan
            max_step = steps.max() if len(steps) > 0 else np.nan

            rows.append({
                "folder": folder.name,
                "file": fpath.name,
                "n_points": n_pts,
                "x_min": round(x_min, 4),
                "x_max": round(x_max, 4),
                "mean_step": round(mean_step, 4),
                "min_step": round(min_step, 4),
                "max_step": round(max_step, 4),
            })
        except Exception as e:
            rows.append({
                "folder": folder.name,
                "file": fpath.name,
                "n_points": np.nan,
                "x_min": np.nan,
                "x_max": np.nan,
                "mean_step": np.nan,
                "min_step": np.nan,
                "max_step": np.nan,
            })
            print(f"  ERROR reading {fpath.name}: {e}")

per_file = pd.DataFrame(rows)
per_file.to_csv(OUT_DIR / "raw_grid_stats.csv", index=False)
print(f"Per-file stats saved: {OUT_DIR / 'raw_grid_stats.csv'}  ({len(per_file)} files)\n")

# --- Summary by folder ---
print("=" * 110)
print(f"{'Folder':<42} {'Files':>5}  {'n_pts(min-max)':>16} {'uniq_n':>6}  "
      f"{'x_min range':>20}  {'x_max range':>20}  {'step(min/mean/max)':>22}")
print("=" * 110)

for folder_name, grp in per_file.groupby("folder", sort=True):
    n_files = len(grp)
    n_min = int(grp["n_points"].min())
    n_max = int(grp["n_points"].max())
    n_med = int(grp["n_points"].median())
    n_unique = grp["n_points"].nunique()

    xmin_lo = grp["x_min"].min()
    xmin_hi = grp["x_min"].max()
    xmax_lo = grp["x_max"].min()
    xmax_hi = grp["x_max"].max()

    step_lo = grp["mean_step"].min()
    step_hi = grp["mean_step"].max()
    step_avg = grp["mean_step"].mean()

    flag = " ***" if n_unique > 1 else ""

    print(f"{folder_name:<42} {n_files:>5}  {n_min:>7}-{n_max:<7}  {n_unique:>5}  "
          f"{xmin_lo:>9.2f}-{xmin_hi:<9.2f}  {xmax_lo:>9.2f}-{xmax_hi:<9.2f}  "
          f"{step_lo:>6.3f}/{step_avg:>6.3f}/{step_hi:>6.3f}{flag}")

print("=" * 110)

# Overall
print(f"\nOVERALL: {len(per_file)} files across {per_file['folder'].nunique()} folders")
print(f"  n_points:  min={int(per_file['n_points'].min())}, max={int(per_file['n_points'].max())}, "
      f"unique values={sorted(per_file['n_points'].unique().astype(int).tolist())}")
print(f"  x_min:     {per_file['x_min'].min():.4f} to {per_file['x_min'].max():.4f}")
print(f"  x_max:     {per_file['x_max'].min():.4f} to {per_file['x_max'].max():.4f}")
print(f"  mean_step: {per_file['mean_step'].min():.4f} to {per_file['mean_step'].max():.4f} "
      f"(overall mean: {per_file['mean_step'].mean():.4f})")

# Flag inconsistencies
inconsistent = per_file.groupby("folder")["n_points"].nunique()
inconsistent = inconsistent[inconsistent > 1]
if len(inconsistent):
    print(f"\n*** INCONSISTENT n_points in {len(inconsistent)} folder(s):")
    for fname, cnt in inconsistent.items():
        sub = per_file[per_file["folder"] == fname]
        vals = sorted(sub["n_points"].unique().astype(int).tolist())
        print(f"  {fname}: {cnt} unique values: {vals}")
else:
    print("\nAll folders have consistent n_points across files.")
