"""Link SERS spectrum data with standardized clinical data.

Joins SERS metadata (group + sample_id) to clinical patient records,
producing a merged dataset and correlation analysis.

Matching rules:
  - PRO, NOR, DIA, HBP, LUN, CRC: "{group} {sample_id}" == patient_id
  - CPAN: "CPAN {sample_id}" == patient_id (disease_group=PAN)
  - H.D.: "H.D. {sample_id}" -> "H. D. {sample_id}" (normalize spaces)
  - BRE, OVA: order-based matching (SERS sample_id 1..N -> clinical row order)
  - BLA, SPAN, UNK: no clinical data available

Usage:
    python link_sers_clinical.py
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd


def build_sers_patient_mapping(meta: pd.DataFrame, clin: pd.DataFrame) -> dict:
    """Build mapping: (sers_group, sers_sample_id) -> clinical patient_id."""
    mapping = {}

    # 1. Direct match groups: SERS "{group} {sample_id}" == clinical patient_id
    direct_groups = ["PRO", "NOR", "DIA", "HBP", "LUN", "CRC"]
    clin_pid_set = set(clin["patient_id"].tolist())

    for g in direct_groups:
        sers_sids = meta[meta["group"] == g]["sample_id"].unique()
        for sid in sers_sids:
            pid = f"{g} {sid}"
            if pid in clin_pid_set:
                mapping[(g, int(sid))] = pid

    # 2. CPAN -> PAN
    sers_sids = meta[meta["group"] == "CPAN"]["sample_id"].unique()
    for sid in sers_sids:
        pid = f"CPAN {sid}"
        if pid in clin_pid_set:
            mapping[("CPAN", int(sid))] = pid

    # 3. H.D.: SERS "H.D." -> Clinical "H. D."
    sers_sids = meta[meta["group"] == "H.D."]["sample_id"].unique()
    for sid in sers_sids:
        pid = f"H. D. {sid}"
        if pid in clin_pid_set:
            mapping[("H.D.", int(sid))] = pid

    # 4. BRE, OVA: order-based (SERS sample 1..N -> clinical row 1..N)
    for sers_group, clin_group in [("BRE", "BRE"), ("OVA", "OVA")]:
        sers_sids = sorted(meta[meta["group"] == sers_group]["sample_id"].unique())
        clin_pids = clin[clin["disease_group"] == clin_group]["patient_id"].tolist()
        for sid in sers_sids:
            idx = int(sid) - 1  # 1-based -> 0-based
            if 0 <= idx < len(clin_pids):
                mapping[(sers_group, int(sid))] = clin_pids[idx]

    return mapping


def compute_sers_features(spectra: pd.DataFrame) -> pd.DataFrame:
    """Compute per-patient aggregate SERS spectral features.

    From replicate spectra, compute: mean intensity, std, peak positions, etc.
    """
    # Identify wavenumber columns (x_401.81, x_403.74, ...)
    wn_cols = [c for c in spectra.columns if c.startswith("x_")]
    wavenumbers = np.array([float(c[2:]) for c in wn_cols])

    records = []
    for (group, sid), gdf in spectra.groupby(["group", "sample_id"]):
        vals = gdf[wn_cols].values  # (n_replicates, n_wavenumbers)
        mean_spec = vals.mean(axis=0)
        std_spec = vals.std(axis=0)

        # Key spectral features
        peak_idx = np.argmax(mean_spec)
        peak_wn = wavenumbers[peak_idx]
        peak_int = mean_spec[peak_idx]
        total_int = np.trapezoid(mean_spec, wavenumbers)
        mean_int = mean_spec.mean()

        # Key SERS band regions (approximate)
        def band_mean(wn_low, wn_high):
            mask = (wavenumbers >= wn_low) & (wavenumbers <= wn_high)
            if mask.sum() == 0:
                return np.nan
            return mean_spec[mask].mean()

        records.append({
            "group": group,
            "sample_id": int(sid),
            "n_replicates": len(gdf),
            "peak_wavenumber": peak_wn,
            "peak_intensity": peak_int,
            "total_intensity": total_int,
            "mean_intensity": mean_int,
            "std_intensity": std_spec.mean(),
            # Key SERS regions
            "band_600_650": band_mean(600, 650),    # C-S stretch
            "band_720_760": band_mean(720, 760),    # C-N stretch
            "band_1000_1050": band_mean(1000, 1050), # phenylalanine
            "band_1200_1300": band_mean(1200, 1300), # amide III
            "band_1400_1500": band_mean(1400, 1500), # CH2 deformation
            "band_1600_1700": band_mean(1600, 1700), # amide I / C=C
        })

    return pd.DataFrame(records)


def main():
    base = Path(".")
    out_dir = Path("data/clinical_data/standardized")

    # Load data
    print("Loading data...")
    spectra = pd.read_csv(base / "results" / "processed_spectra.csv")
    meta = pd.read_csv(base / "results" / "metadata_raw.csv")
    clin = pd.read_csv(out_dir / "all_clinical_standardized.csv")

    print(f"  Spectra: {spectra.shape}")
    print(f"  Metadata: {meta.shape}")
    print(f"  Clinical: {clin.shape}")

    # Build mapping
    print("\nBuilding SERS -> Clinical mapping...")
    mapping = build_sers_patient_mapping(meta, clin)
    print(f"  Mapped: {len(mapping)} patient pairs")

    # Show per-group match counts
    group_counts = {}
    for (g, sid), pid in mapping.items():
        group_counts[g] = group_counts.get(g, 0) + 1
    for g in sorted(group_counts):
        sers_n = meta[meta["group"] == g]["sample_id"].nunique()
        print(f"    {g:6s}: {group_counts[g]:3d}/{sers_n:3d} matched")

    # Unmatched SERS groups
    all_sers_groups = set(meta["group"].unique())
    matched_groups = set(g for g, _ in mapping.keys())
    unmatched = all_sers_groups - matched_groups
    if unmatched:
        for g in sorted(unmatched):
            n = meta[meta["group"] == g]["sample_id"].nunique()
            print(f"    {g:6s}: 0/{n:3d} (no clinical data)")

    # Compute SERS features
    print("\nComputing SERS spectral features...")
    sers_features = compute_sers_features(spectra)
    print(f"  Features computed for {len(sers_features)} patient-samples")

    # Add patient_id to SERS features
    sers_features["patient_id"] = sers_features.apply(
        lambda r: mapping.get((r["group"], r["sample_id"])), axis=1
    )
    matched_features = sers_features[sers_features["patient_id"].notna()].copy()
    print(f"  With clinical match: {len(matched_features)}")

    # Merge with clinical data
    print("\nMerging SERS features with clinical data...")
    merged = matched_features.merge(clin, on="patient_id", how="left", suffixes=("_sers", "_clin"))

    # Save merged dataset
    merged_path = out_dir / "sers_clinical_merged.csv"
    merged.to_csv(merged_path, index=False, encoding="utf-8-sig")
    print(f"  Saved: {merged_path} ({len(merged)} rows x {len(merged.columns)} cols)")

    # Save SERS features separately
    features_path = out_dir / "sers_patient_features.csv"
    sers_features.to_csv(features_path, index=False)
    print(f"  Saved: {features_path}")

    # ── Correlation analysis ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Correlation Analysis: SERS features vs Clinical values")
    print(f"{'='*60}")

    sers_cols = [
        "peak_intensity", "total_intensity", "mean_intensity", "std_intensity",
        "band_600_650", "band_720_760", "band_1000_1050",
        "band_1200_1300", "band_1400_1500", "band_1600_1700",
    ]
    clin_cols = [
        "age", "bmi", "wbc", "rbc", "hb", "hct", "platelet",
        "ast", "alt", "alp", "ggt", "bun", "creatinine", "glucose",
        "total_cholesterol", "hba1c", "total_bilirubin", "uric_acid", "calcium",
        "cea", "ca19_9", "psa",
    ]

    # Filter columns that exist and have data
    avail_sers = [c for c in sers_cols if c in merged.columns]
    avail_clin = [c for c in clin_cols if c in merged.columns and merged[c].notna().sum() > 10]

    if avail_sers and avail_clin:
        corr_data = merged[avail_sers + avail_clin].astype(float)
        corr_matrix = corr_data.corr()

        # Extract SERS-clinical cross-correlations
        cross_corr = corr_matrix.loc[avail_sers, avail_clin]

        # Save full correlation matrix
        corr_path = out_dir / "sers_clinical_correlation.csv"
        cross_corr.to_csv(corr_path, encoding="utf-8-sig")
        print(f"\nCorrelation matrix saved: {corr_path}")

        # Show top correlations
        print(f"\nTop correlations (|r| > 0.15):")
        for sers_c in avail_sers:
            for clin_c in avail_clin:
                r = cross_corr.loc[sers_c, clin_c]
                n = corr_data[[sers_c, clin_c]].dropna().shape[0]
                if abs(r) > 0.15 and n >= 20:
                    print(f"  {sers_c:25s} vs {clin_c:20s}: r={r:+.3f} (n={n})")

        # Per-group summary
        print(f"\n{'='*60}")
        print("Per-group SERS feature summary:")
        for g, gdf in merged.groupby("group"):
            n = len(gdf)
            mi = gdf["mean_intensity"].mean()
            pi = gdf["peak_intensity"].mean()
            print(f"  {g:6s} (n={n:3d}): mean_int={mi:.2f}, peak_int={pi:.2f}")

    # ── Summary stats ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  Total SERS patients: {len(sers_features)}")
    print(f"  With clinical data:  {len(matched_features)}")
    print(f"  Without clinical:    {len(sers_features) - len(matched_features)}")
    print(f"  Clinical variables:  {len(avail_clin)}")
    print(f"  SERS features:       {len(avail_sers)}")

    # Disease distribution in merged
    print(f"\n  Disease distribution (merged):")
    for g, gdf in merged.groupby("disease_group"):
        print(f"    {g:6s}: {len(gdf)} patients")


if __name__ == "__main__":
    main()
