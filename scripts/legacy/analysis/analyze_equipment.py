"""
Cross-Equipment SERS Spectral Analysis

Analyzes spectral reproducibility across 5 instruments using
the same 100 NOR samples × 20 replicates:
  1. Metrohm Mira P (handheld)
  2. Nanoscope RamCheck-A1 raw (medical_raw)
  3. Nanoscope RamCheck-A1 background-removed (medical_removed)
  4. Nanoscope NS200 (nanoscope)
  5. Thermo DXR3xi (thermo)

Analysis:
  A. Spectral characteristics per instrument (mean, std, SNR)
  B. Inter-replicate reproducibility (RSD, pairwise correlation)
  C. Minimum replicate number analysis (convergence of mean CV)
  D. Cross-instrument spectral similarity
"""

import sys
import json
import logging
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "equipment_test_data"
FINGERPRINT = (400, 2200)

EQUIPMENT_INFO = {
    "handheld":        {"name": "Metrohm Mira P",           "type": "Handheld"},
    "medical_raw":     {"name": "Nanoscope RamCheck-A1 (raw)", "type": "Medical"},
    "medical_removed": {"name": "Nanoscope RamCheck-A1 (bg-removed)", "type": "Medical"},
    "nanoscope":       {"name": "Nanoscope NS200",           "type": "Benchtop"},
    "thermo":          {"name": "Thermo DXR3xi",             "type": "Benchtop"},
}


# =============================================================================
# File readers per equipment
# =============================================================================
def read_handheld(path):
    """Metrohm Mira P: key-value CSV with 'Intensities' field."""
    metadata = {}
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().strip('"').split('","')
            if len(parts) == 2:
                key, val = parts
                metadata[key] = val

    first_wn = float(metadata.get("Firstwavenumber", 400))
    last_wn = float(metadata.get("LastWavenumber", 2300))
    intensities = [float(v) for v in metadata["Intensities"].split(",")]
    wavenumbers = np.linspace(first_wn, last_wn, len(intensities))
    return wavenumbers, np.array(intensities)


def read_two_column(path):
    """Standard 2-column (wavenumber, intensity) format."""
    df = pd.read_csv(path, sep=None, engine="python", header=None, usecols=[0, 1])
    df.columns = ["wn", "intensity"]
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    df = df.sort_values("wn")
    return df["wn"].to_numpy(), df["intensity"].to_numpy()


def read_spectrum_for_equipment(path, equipment):
    """Route to correct reader based on equipment type."""
    if equipment == "handheld":
        return read_handheld(path)
    else:
        return read_two_column(path)


def parse_sample_replicate(filename):
    """Extract sample_id and replicate from filename."""
    stem = Path(filename).stem
    # Handle handheld: "NOR 100_1(1)_Sample_..."
    m = re.search(r"(\d+)_(\d+)", stem)
    if m:
        return m.group(1), int(m.group(2))
    return None, None


# =============================================================================
# Load all equipment data
# =============================================================================
def load_equipment_data():
    """Load all spectra from all equipment, trimmed to fingerprint region."""
    data = {}  # equipment -> {sample_id -> {rep -> (wn, intensity)}}

    for eq_folder, info in EQUIPMENT_INFO.items():
        eq_dir = DATA_DIR / eq_folder / "1. NOR"
        if not eq_dir.exists():
            logger.warning(f"  {eq_folder}: directory not found")
            continue

        # Find spectrum files
        patterns = ["*.csv", "*.CSV", "*.txt"]
        files = []
        for p in patterns:
            files.extend(eq_dir.glob(p))
        # Exclude Background subfolder files
        files = [f for f in files if "Background" not in str(f)]

        samples = defaultdict(dict)
        loaded, failed = 0, 0
        for fpath in files:
            sid, rep = parse_sample_replicate(fpath.name)
            if sid is None:
                continue
            try:
                wn, intensity = read_spectrum_for_equipment(fpath, eq_folder)
                # Trim to fingerprint region
                mask = (wn >= FINGERPRINT[0]) & (wn <= FINGERPRINT[1])
                if mask.sum() < 50:
                    continue
                samples[sid][rep] = (wn[mask], intensity[mask])
                loaded += 1
            except Exception as e:
                failed += 1
                continue

        data[eq_folder] = dict(samples)
        logger.info(f"  {eq_folder} ({info['name']}): {loaded} spectra, "
                    f"{len(samples)} samples, {failed} failed")

    return data


# =============================================================================
# Create common grid and resample
# =============================================================================
def resample_to_grid(wn, intensity, grid):
    """Resample spectrum to common grid."""
    f = interp1d(wn, intensity, kind='linear', bounds_error=False, fill_value=np.nan)
    return f(grid)


# =============================================================================
# A. Spectral characteristics
# =============================================================================
def analyze_spectral_characteristics(data, grid):
    """Per-equipment: mean spectrum, std, SNR."""
    results = {}
    for eq, samples in data.items():
        all_spectra = []
        for sid, reps in samples.items():
            for rep, (wn, intensity) in reps.items():
                resampled = resample_to_grid(wn, intensity, grid)
                if not np.isnan(resampled).all():
                    all_spectra.append(resampled)

        if not all_spectra:
            continue
        spectra = np.array(all_spectra)
        # Remove rows that are mostly NaN (>50%)
        nan_frac = np.isnan(spectra).sum(axis=1) / spectra.shape[1]
        spectra = spectra[nan_frac < 0.5]
        if len(spectra) == 0:
            continue

        mean_spec = np.nanmean(spectra, axis=0)
        std_spec = np.nanstd(spectra, axis=0)
        snr = np.nanmean(mean_spec) / np.nanmean(std_spec) if np.nanmean(std_spec) > 0 else 0

        results[eq] = {
            "n_spectra": len(spectra),
            "n_samples": len(samples),
            "mean_intensity": float(np.nanmean(mean_spec)),
            "std_intensity": float(np.nanmean(std_spec)),
            "snr": float(snr),
            "dynamic_range": float(np.nanmax(mean_spec) - np.nanmin(mean_spec)),
            "intensity_range": f"{np.nanmin(mean_spec):.0f} - {np.nanmax(mean_spec):.0f}",
        }
    return results


# =============================================================================
# B. Inter-replicate reproducibility
# =============================================================================
def analyze_reproducibility(data, grid):
    """Per-equipment: RSD and pairwise correlation across replicates."""
    results = {}
    for eq, samples in data.items():
        sample_cvs = []
        sample_corrs = []

        for sid, reps in samples.items():
            if len(reps) < 2:
                continue
            spectra = []
            for rep, (wn, intensity) in reps.items():
                resampled = resample_to_grid(wn, intensity, grid)
                if not np.isnan(resampled).all():
                    spectra.append(resampled)

            if len(spectra) < 2:
                continue
            spectra = np.array(spectra)
            valid_cols = ~np.isnan(spectra).any(axis=0)
            spectra = spectra[:, valid_cols]

            if spectra.shape[1] < 10:
                continue

            # CV per wavenumber point, then mean
            mean = np.mean(spectra, axis=0)
            std = np.std(spectra, axis=0)
            cv = np.where(np.abs(mean) > 1e-6, std / np.abs(mean) * 100, 0)
            sample_cvs.append(np.mean(cv))

            # Pairwise correlation
            corr_matrix = np.corrcoef(spectra)
            n = len(spectra)
            corrs = [corr_matrix[i, j] for i in range(n) for j in range(i+1, n)]
            sample_corrs.append(np.mean(corrs))

        results[eq] = {
            "mean_cv": float(np.mean(sample_cvs)) if sample_cvs else float('nan'),
            "std_cv": float(np.std(sample_cvs)) if sample_cvs else float('nan'),
            "median_cv": float(np.median(sample_cvs)) if sample_cvs else float('nan'),
            "mean_corr": float(np.mean(sample_corrs)) if sample_corrs else float('nan'),
            "std_corr": float(np.std(sample_corrs)) if sample_corrs else float('nan'),
            "n_samples_evaluated": len(sample_cvs),
            "cv_below_5pct": sum(1 for c in sample_cvs if c < 5.0),
            "cv_below_10pct": sum(1 for c in sample_cvs if c < 10.0),
            "corr_above_095": sum(1 for c in sample_corrs if c > 0.95),
        }
    return results


# =============================================================================
# C. Minimum replicate analysis
# =============================================================================
def analyze_minimum_replicates(data, grid, max_reps=20):
    """How many replicates are needed for stable mean spectrum?"""
    results = {}

    for eq, samples in data.items():
        # For each n_reps (1..20), compute mean CV of the averaged spectrum
        convergence = []
        for n_rep in range(1, max_reps + 1):
            cvs_at_n = []
            for sid, reps in samples.items():
                rep_keys = sorted(reps.keys())
                if len(rep_keys) < max_reps:
                    continue

                # Bootstrap: take first n_rep replicates, compute mean
                # Then compare multiple subsets
                all_rep_spectra = []
                for rk in rep_keys[:max_reps]:
                    wn, intensity = reps[rk]
                    resampled = resample_to_grid(wn, intensity, grid)
                    if not np.isnan(resampled).all():
                        all_rep_spectra.append(resampled)

                if len(all_rep_spectra) < max_reps:
                    continue

                all_rep_spectra = np.array(all_rep_spectra)
                valid_cols = ~np.isnan(all_rep_spectra).any(axis=0)
                all_rep_spectra = all_rep_spectra[:, valid_cols]

                if all_rep_spectra.shape[1] < 10:
                    continue

                # Monte Carlo: 10 random subsets of size n_rep
                subset_means = []
                rng = np.random.RandomState(42)
                n_trials = min(20, int(__import__('math').comb(len(all_rep_spectra), n_rep)))
                for _ in range(n_trials):
                    idx = rng.choice(len(all_rep_spectra), n_rep, replace=False)
                    subset_means.append(np.mean(all_rep_spectra[idx], axis=0))

                if len(subset_means) < 2:
                    continue

                subset_means = np.array(subset_means)
                grand_mean = np.mean(subset_means, axis=0)
                grand_std = np.std(subset_means, axis=0)
                cv = np.where(np.abs(grand_mean) > 1e-6,
                             grand_std / np.abs(grand_mean) * 100, 0)
                cvs_at_n.append(np.mean(cv))

            if cvs_at_n:
                convergence.append({
                    "n_reps": n_rep,
                    "mean_cv": float(np.mean(cvs_at_n)),
                    "std_cv": float(np.std(cvs_at_n)),
                })

        # Find minimum n where CV < threshold
        min_reps_2pct = None
        min_reps_1pct = None
        for c in convergence:
            if min_reps_2pct is None and c["mean_cv"] < 2.0:
                min_reps_2pct = c["n_reps"]
            if min_reps_1pct is None and c["mean_cv"] < 1.0:
                min_reps_1pct = c["n_reps"]

        results[eq] = {
            "convergence": convergence,
            "min_reps_cv_2pct": min_reps_2pct,
            "min_reps_cv_1pct": min_reps_1pct,
        }
    return results


# =============================================================================
# D. Cross-instrument similarity
# =============================================================================
def analyze_cross_instrument(data, grid):
    """Compare mean spectra across instruments (after SNV normalization)."""
    mean_spectra = {}
    for eq, samples in data.items():
        all_spectra = []
        for sid, reps in samples.items():
            for rep, (wn, intensity) in reps.items():
                resampled = resample_to_grid(wn, intensity, grid)
                if not np.isnan(resampled).all():
                    all_spectra.append(resampled)
        if all_spectra:
            spectra = np.array(all_spectra)
            nan_frac = np.isnan(spectra).sum(axis=1) / spectra.shape[1]
            spectra = spectra[nan_frac < 0.5]
            if len(spectra) == 0:
                continue
            mean = np.nanmean(spectra, axis=0)
            # Replace remaining NaN with interpolation
            nans = np.isnan(mean)
            if nans.all():
                continue
            if nans.any():
                mean[nans] = np.interp(grid[nans], grid[~nans], mean[~nans])
            # SNV normalize for fair comparison
            mean_snv = (mean - np.mean(mean)) / np.std(mean)
            mean_spectra[eq] = mean_snv

    # Pairwise correlation matrix
    eq_names = sorted(mean_spectra.keys())
    n = len(eq_names)
    corr_matrix = np.eye(n)
    for i in range(n):
        for j in range(i+1, n):
            valid = ~(np.isnan(mean_spectra[eq_names[i]]) | np.isnan(mean_spectra[eq_names[j]]))
            if valid.sum() > 10:
                c = np.corrcoef(mean_spectra[eq_names[i]][valid],
                               mean_spectra[eq_names[j]][valid])[0, 1]
                corr_matrix[i, j] = c
                corr_matrix[j, i] = c

    return eq_names, corr_matrix


# =============================================================================
# Main
# =============================================================================
def main():
    logger.info("=" * 64)
    logger.info("  Cross-Equipment SERS Spectral Analysis")
    logger.info("=" * 64)

    # Load data
    logger.info("\n[1] Loading equipment data...")
    data = load_equipment_data()

    # Create common grid (fingerprint region)
    grid = np.linspace(FINGERPRINT[0], FINGERPRINT[1], 900)

    # A. Spectral characteristics
    logger.info("\n[2] Spectral characteristics...")
    char = analyze_spectral_characteristics(data, grid)
    logger.info(f"\n  {'Equipment':<35s} {'Spectra':>8s} {'Samples':>8s} {'Mean I':>10s} {'Std I':>10s} {'SNR':>8s} {'Dyn Range':>10s}")
    logger.info(f"  {'-'*89}")
    for eq in EQUIPMENT_INFO:
        if eq in char:
            c = char[eq]
            name = EQUIPMENT_INFO[eq]["name"]
            logger.info(f"  {name:<35s} {c['n_spectra']:>8d} {c['n_samples']:>8d} "
                       f"{c['mean_intensity']:>10.0f} {c['std_intensity']:>10.0f} "
                       f"{c['snr']:>8.1f} {c['dynamic_range']:>10.0f}")

    # B. Reproducibility
    logger.info("\n[3] Inter-replicate reproducibility...")
    repro = analyze_reproducibility(data, grid)
    logger.info(f"\n  {'Equipment':<35s} {'Mean CV%':>9s} {'Med CV%':>9s} {'Mean Corr':>10s} {'CV<5%':>7s} {'CV<10%':>7s} {'Corr>0.95':>10s}")
    logger.info(f"  {'-'*97}")
    for eq in EQUIPMENT_INFO:
        if eq in repro:
            r = repro[eq]
            name = EQUIPMENT_INFO[eq]["name"]
            n = r["n_samples_evaluated"]
            logger.info(f"  {name:<35s} {r['mean_cv']:>8.2f}% {r['median_cv']:>8.2f}% "
                       f"{r['mean_corr']:>10.4f} "
                       f"{r['cv_below_5pct']:>4d}/{n:<3d} "
                       f"{r['cv_below_10pct']:>4d}/{n:<3d} "
                       f"{r['corr_above_095']:>5d}/{n:<3d}")

    # C. Minimum replicates
    logger.info("\n[4] Minimum replicate analysis...")
    min_reps = analyze_minimum_replicates(data, grid)
    logger.info(f"\n  {'Equipment':<35s} {'Min for CV<2%':>14s} {'Min for CV<1%':>14s}")
    logger.info(f"  {'-'*63}")
    for eq in EQUIPMENT_INFO:
        if eq in min_reps:
            mr = min_reps[eq]
            name = EQUIPMENT_INFO[eq]["name"]
            v2 = str(mr["min_reps_cv_2pct"]) if mr["min_reps_cv_2pct"] else ">20"
            v1 = str(mr["min_reps_cv_1pct"]) if mr["min_reps_cv_1pct"] else ">20"
            logger.info(f"  {name:<35s} {v2:>14s} {v1:>14s}")

    # Convergence details
    logger.info("\n  Convergence curve (mean CV% at each replicate count):")
    header = f"  {'n_reps':>6s}"
    for eq in EQUIPMENT_INFO:
        if eq in min_reps:
            header += f"  {EQUIPMENT_INFO[eq]['name'][:15]:>15s}"
    logger.info(header)
    for n_rep in [1, 2, 3, 4, 5, 7, 10, 15, 20]:
        row = f"  {n_rep:>6d}"
        for eq in EQUIPMENT_INFO:
            if eq in min_reps:
                conv = min_reps[eq]["convergence"]
                match = [c for c in conv if c["n_reps"] == n_rep]
                if match:
                    row += f"  {match[0]['mean_cv']:>14.2f}%"
                else:
                    row += f"  {'N/A':>15s}"
        logger.info(row)

    # D. Cross-instrument similarity
    logger.info("\n[5] Cross-instrument similarity (SNV-normalized mean spectra)...")
    eq_names, corr_matrix = analyze_cross_instrument(data, grid)
    logger.info(f"\n  Correlation matrix:")
    header = f"  {'':>35s}"
    for eq in eq_names:
        header += f"  {EQUIPMENT_INFO[eq]['name'][:12]:>12s}"
    logger.info(header)
    for i, eq_i in enumerate(eq_names):
        row = f"  {EQUIPMENT_INFO[eq_i]['name']:<35s}"
        for j in range(len(eq_names)):
            row += f"  {corr_matrix[i,j]:>12.4f}"
        logger.info(row)

    # Save results
    out_dir = PROJECT_ROOT / "results" / "equipment_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "equipment_info": {eq: {**info, "folder": eq} for eq, info in EQUIPMENT_INFO.items()},
        "spectral_characteristics": char,
        "reproducibility": repro,
        "minimum_replicates": {eq: {
            "min_reps_cv_2pct": v["min_reps_cv_2pct"],
            "min_reps_cv_1pct": v["min_reps_cv_1pct"],
            "convergence": v["convergence"],
        } for eq, v in min_reps.items()},
        "cross_instrument_correlation": {
            "equipment_order": eq_names,
            "correlation_matrix": corr_matrix.tolist(),
        },
    }

    with open(out_dir / "equipment_analysis.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"\n  Results saved to: {out_dir / 'equipment_analysis.json'}")
    logger.info("\n  Done.")


if __name__ == "__main__":
    main()
