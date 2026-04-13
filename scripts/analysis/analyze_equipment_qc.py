"""
Cross-Equipment QC Analysis — AECD Standard

Computes AECD-standard RSD (std/max*100) and pairwise correlation
for each equipment at protocol-specific replicate counts:
  - Mira P: 5 replicates
  - Thermo DXR3xi: 5 replicates
  - NS200: 5 replicates
  - RamCheck-A1: 6 replicates

Also computes QC at 20 replicates for reference.
"""

import sys
import json
import logging
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.interpolate import interp1d

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "equipment_test_data"
FINGERPRINT = (400, 2200)
GRID = np.linspace(FINGERPRINT[0], FINGERPRINT[1], 900)

EQUIPMENT = {
    "handheld":    {"name": "Mira P",       "protocol_reps": 5},
    "thermo":      {"name": "Thermo DXR3xi","protocol_reps": 5},
    "nanoscope":   {"name": "NS200",        "protocol_reps": 5},
    "medical_raw": {"name": "RamCheck-A1",  "protocol_reps": 6},
}

RSD_THRESHOLD = 5.0
CORR_THRESHOLD = 0.95


def read_handheld(path):
    metadata = {}
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().strip('"').split('","')
            if len(parts) == 2:
                metadata[parts[0]] = parts[1]
    first_wn = float(metadata.get("Firstwavenumber", 400))
    last_wn = float(metadata.get("LastWavenumber", 2300))
    intensities = [float(v) for v in metadata["Intensities"].split(",")]
    return np.linspace(first_wn, last_wn, len(intensities)), np.array(intensities)


def read_two_column(path):
    import pandas as pd
    df = pd.read_csv(path, sep=None, engine="python", header=None, usecols=[0, 1])
    df.columns = ["wn", "intensity"]
    df = df.apply(pd.to_numeric, errors="coerce").dropna().sort_values("wn")
    return df["wn"].to_numpy(), df["intensity"].to_numpy()


def resample(wn, intensity):
    f = interp1d(wn, intensity, kind='linear', bounds_error=False, fill_value=np.nan)
    return f(GRID)


def load_data():
    data = {}
    for eq_key, info in EQUIPMENT.items():
        eq_dir = DATA_DIR / eq_key / "1. NOR"
        if not eq_dir.exists():
            continue
        files = []
        for p in ["*.csv", "*.CSV", "*.txt"]:
            files.extend(eq_dir.glob(p))
        files = [f for f in files if "Background" not in str(f)]

        samples = defaultdict(dict)
        for fpath in files:
            m = re.search(r"(\d+)_(\d+)", fpath.stem)
            if not m:
                continue
            sid, rep = m.group(1), int(m.group(2))
            try:
                if eq_key == "handheld":
                    wn, intensity = read_handheld(fpath)
                else:
                    wn, intensity = read_two_column(fpath)
                mask = (wn >= FINGERPRINT[0]) & (wn <= FINGERPRINT[1])
                if mask.sum() < 50:
                    continue
                resampled = resample(wn[mask], intensity[mask])
                if not np.isnan(resampled).all():
                    samples[sid][rep] = resampled
            except Exception:
                continue

        data[eq_key] = dict(samples)
        logger.info(f"  {info['name']}: {len(samples)} samples loaded")
    return data


def compute_qc(spectra_matrix):
    """Compute AECD-standard QC metrics for a replicate matrix (n_reps x n_points).
    RSD = mean(std_per_wn / max_intensity) * 100
    Corr = mean pairwise Pearson correlation
    """
    valid_cols = ~np.isnan(spectra_matrix).any(axis=0)
    M = spectra_matrix[:, valid_cols]
    if M.shape[1] < 10 or M.shape[0] < 2:
        return None, None

    mean_spec = M.mean(axis=0)
    std_spec = M.std(axis=0, ddof=1)
    max_intensity = mean_spec.max()

    if max_intensity > 1e-10:
        rsd = float(np.mean(std_spec / max_intensity * 100))
    else:
        rsd = float('inf')

    n = len(M)
    corr_matrix = np.corrcoef(M)
    corrs = [corr_matrix[i, j] for i in range(n) for j in range(i+1, n)]
    corr = float(np.mean(corrs)) if corrs else 0.0

    return rsd, corr


def analyze(data, n_reps):
    """Analyze QC at a specific replicate count using first n_reps replicates."""
    results = {}
    for eq_key, info in EQUIPMENT.items():
        if eq_key not in data:
            continue
        samples = data[eq_key]
        rsds, corrs = [], []
        for sid, reps in samples.items():
            rep_keys = sorted(reps.keys())[:n_reps]
            if len(rep_keys) < n_reps:
                continue
            M = np.array([reps[k] for k in rep_keys])
            rsd, corr = compute_qc(M)
            if rsd is not None:
                rsds.append(rsd)
                corrs.append(corr)

        if not rsds:
            continue

        rsd_arr = np.array(rsds)
        corr_arr = np.array(corrs)
        pass_rsd = int((rsd_arr < RSD_THRESHOLD).sum())
        pass_corr = int((corr_arr > CORR_THRESHOLD).sum())
        pass_both = int(((rsd_arr < RSD_THRESHOLD) & (corr_arr > CORR_THRESHOLD)).sum())
        n = len(rsds)

        results[eq_key] = {
            "name": info["name"],
            "n_samples": n,
            "n_reps": n_reps,
            "rsd_mean": round(float(rsd_arr.mean()), 2),
            "rsd_median": round(float(np.median(rsd_arr)), 2),
            "rsd_lt5": pass_rsd,
            "rsd_lt10": int((rsd_arr < 10).sum()),
            "corr_mean": round(float(corr_arr.mean()), 4),
            "corr_gt95": pass_corr,
            "corr_gt99": int((corr_arr > 0.99).sum()),
            "qc_pass": pass_both,
            "qc_rate": round(pass_both / n * 100, 1),
            "per_rsd": [round(float(v), 2) for v in rsd_arr],
            "per_corr": [round(float(v), 4) for v in corr_arr],
        }
    return results


def main():
    logger.info("=" * 60)
    logger.info("  Cross-Equipment QC Analysis (AECD Standard: RSD=std/max)")
    logger.info("=" * 60)

    data = load_data()

    out_dir = PROJECT_ROOT / "results" / "equipment_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # QC at protocol replicates
    logger.info("\n--- QC at Protocol Replicates ---")
    for eq_key, info in EQUIPMENT.items():
        n = info["protocol_reps"]
        res = analyze(data, n)
        if eq_key in res:
            r = res[eq_key]
            logger.info(f"  {r['name']:>15s} ({n} reps): RSD {r['rsd_mean']:.2f}% (med {r['rsd_median']:.2f}%), "
                       f"Corr {r['corr_mean']:.4f}, "
                       f"RSD<5%: {r['rsd_lt5']}/{r['n_samples']}, "
                       f"Corr>0.95: {r['corr_gt95']}/{r['n_samples']}, "
                       f"QC pass: {r['qc_pass']}/{r['n_samples']} ({r['qc_rate']}%)")
            all_results[f"{eq_key}_protocol"] = r

    # QC at 20 replicates (reference)
    logger.info("\n--- QC at 20 Replicates (Reference) ---")
    res20 = analyze(data, 20)
    for eq_key in EQUIPMENT:
        if eq_key in res20:
            r = res20[eq_key]
            logger.info(f"  {r['name']:>15s} (20 reps): RSD {r['rsd_mean']:.2f}% (med {r['rsd_median']:.2f}%), "
                       f"Corr {r['corr_mean']:.4f}, "
                       f"RSD<5%: {r['rsd_lt5']}/{r['n_samples']}, "
                       f"Corr>0.95: {r['corr_gt95']}/{r['n_samples']}, "
                       f"QC pass: {r['qc_pass']}/{r['n_samples']} ({r['qc_rate']}%)")
            all_results[f"{eq_key}_20reps"] = r

    # QC at various replicate counts (for convergence)
    logger.info("\n--- QC Convergence (RSD<5% pass rate at N reps) ---")
    convergence = {}
    for eq_key, info in EQUIPMENT.items():
        conv = []
        for n_reps in range(2, 21):
            res = analyze(data, n_reps)
            if eq_key in res:
                r = res[eq_key]
                conv.append({
                    "n_reps": n_reps,
                    "rsd_mean": r["rsd_mean"],
                    "rsd_median": r["rsd_median"],
                    "corr_mean": r["corr_mean"],
                    "rsd_pass_rate": round(r["rsd_lt5"] / r["n_samples"] * 100, 1),
                    "qc_pass_rate": r["qc_rate"],
                })
        convergence[eq_key] = conv
        logger.info(f"  {info['name']}: computed for 2-20 reps")

    all_results["convergence"] = convergence

    with open(out_dir / "equipment_qc_aecd.json", "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nSaved to: {out_dir / 'equipment_qc_aecd.json'}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
