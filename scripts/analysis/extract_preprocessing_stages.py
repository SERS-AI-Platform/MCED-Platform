#!/usr/bin/env python3
"""Extract preprocessing stages for ALL 1,628+ subjects."""

import sys, json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.sers.preprocessing import (
    trim_spectrum, smooth, baseline_correction,
    normalize_spectrum, resample,
)
from src.sers.config import load_config, RAW_DATA_DIR
from src.sers.io import find_spectra, parse_filename, read_spectrum

BASE = Path(__file__).resolve().parents[2]
OUT = BASE / "results" / "qc_diagnostic"
OUT.mkdir(exist_ok=True)

config = load_config(BASE / "config" / "config.yaml")

# --- Load raw spectra ---
print("Loading raw spectra...")
data_dir = RAW_DATA_DIR
raw_spectra = {}
for fp in find_spectra(data_dir, pattern="*.csv"):
    try:
        spec_id = parse_filename(fp, fallback_group=config.folder_to_group.get(fp.parent.name, "UNK"))
        x, y = read_spectrum(fp)
        raw_spectra[(spec_id.group, spec_id.sample_id, spec_id.replicate)] = (x, y)
    except Exception:
        pass
print(f"  Loaded {len(raw_spectra)} spectra")

# --- QC stats ---
qc_stats = pd.read_csv(BASE / "results" / "qc_stats.csv")
RSD_THRESH = 5.0
CORR_THRESH = 0.95
qc_stats["qc_pass"] = (qc_stats["mean_rsd"] <= RSD_THRESH) & (qc_stats["mean_corr"] >= CORR_THRESH)

# --- Grid & params ---
fg = config.preprocessing.fixed_grid
grid = np.linspace(fg["x_min"], fg["x_max"], fg["n_points"])
trim_region = (400, 2200)
sw = config.preprocessing.smooth_window
sp = config.preprocessing.smooth_poly
bw = config.preprocessing.baseline_window

# Key stages only (skip trimmed and resampled to save space)
KEY_STAGES = ["raw", "smoothed", "baseline_corrected", "normalized"]
N_PTS = 50  # points per spectrum

def extract_stages(x, y):
    stages = {}
    stages["raw"] = (x.copy(), y.copy())
    xt, yt = trim_spectrum(x, y, region=trim_region)
    ys = smooth(yt, window_length=sw, polyorder=sp)
    stages["smoothed"] = (xt.copy(), ys.copy())
    baseline = pd.Series(ys).rolling(bw, center=True, min_periods=1).min().to_numpy()
    yb = ys - baseline
    stages["baseline_corrected"] = (xt.copy(), yb.copy())
    stages["_baseline_curve"] = (xt.copy(), baseline.copy())
    yn = normalize_spectrum(yb, method="snv")
    stages["normalized"] = (xt.copy(), yn.copy())
    return stages

def downsample(x, y, max_pts=N_PTS):
    if len(x) <= max_pts:
        return [round(float(v), 1) for v in x], [round(float(v), 3) for v in y]
    idx = np.linspace(0, len(x)-1, max_pts, dtype=int)
    return [round(float(x[i]), 1) for i in idx], [round(float(y[i]), 3) for i in idx]

# --- Process ALL subjects ---
result = {
    "config": {
        "smooth_window": sw, "smooth_poly": sp, "baseline_window": bw,
        "normalization": "snv", "trim_region": list(trim_region),
        "grid_points": len(grid), "rsd_threshold": RSD_THRESH, "corr_threshold": CORR_THRESH,
    },
    "stages": KEY_STAGES,
    "samples": [],
}

total = len(qc_stats)
for idx, (_, row) in enumerate(qc_stats.iterrows()):
    grp = row["group"]
    sid = row["sample_id"]
    qc_pass = bool(row["qc_pass"])

    sample_data = {
        "g": grp,
        "id": str(sid),
        "s": "pass" if qc_pass else "fail",
        "rsd": round(float(row["mean_rsd"]), 2),
        "cor": round(float(row["mean_corr"]), 4),
        "r": [],  # replicates
    }

    for rep in range(1, 6):
        key = (grp, str(sid), rep)
        if key not in raw_spectra:
            key = (grp, int(sid) if str(sid).isdigit() else sid, rep)
        if key not in raw_spectra:
            continue

        x_raw, y_raw = raw_spectra[key]
        stages = extract_stages(x_raw, y_raw)

        rep_data = {"n": rep, "d": {}}
        for sn in KEY_STAGES:
            xs, ys = stages[sn]
            xd, yd = downsample(xs, ys)
            rep_data["d"][sn] = [xd, yd]  # compact: [x_array, y_array]

        # Baseline curve for smoothed stage
        if "_baseline_curve" in stages:
            xb, yb = stages["_baseline_curve"]
            _, ybd = downsample(xb, yb)
            rep_data["d"]["_bl"] = ybd  # same x as smoothed, only y

        sample_data["r"].append(rep_data)

    if sample_data["r"]:
        result["samples"].append(sample_data)

    if (idx + 1) % 200 == 0:
        print(f"  {idx+1}/{total} samples processed...")

print(f"\nExtracted {len(result['samples'])} samples")

# Summary
from collections import Counter
grp_counts = Counter()
for s in result["samples"]:
    grp_counts[s["g"]] += 1
for g in sorted(grp_counts):
    p = sum(1 for s in result["samples"] if s["g"]==g and s["s"]=="pass")
    f = sum(1 for s in result["samples"] if s["g"]==g and s["s"]=="fail")
    print(f"  {g:>5s}: {p:3d} pass + {f:3d} fail = {grp_counts[g]}")

out_path = OUT / "preprocessing_stages_full.json"
with open(out_path, "w") as f:
    json.dump(result, f, separators=(",", ":"))

sz = out_path.stat().st_size / 1024 / 1024
print(f"\nSaved to {out_path}")
print(f"  Size: {sz:.1f} MB")
print("Done!")
