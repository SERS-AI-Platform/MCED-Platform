#!/usr/bin/env python3
"""
Direct Peak Intensity ↔ Blood Marker Correlation
=================================================
NMF component 대신, 대사체에 매핑된 특정 wavenumber 피크 강도를
직접 혈액 마커와 비교. SERS가 실제 대사체를 측정하는지 생물학적 검증.

Usage:
    python scripts/analysis/peak_marker_direct_correlation.py
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROJECT = Path("/home/user/SERS-AI")
CLINICAL = PROJECT / "data/clinical_data/standardized/all_clinical_standardized.csv"
SPECTRA = PROJECT / "results/processed_spectra.csv"
OUT_DIR = PROJECT / "results/figures/peak_marker_direct_correlation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Known metabolite peak assignments (from metabolite_profiling) ──
# Format: (wavenumber_center, bandwidth, metabolite_name, corresponding_blood_marker)
PEAK_MARKER_PAIRS = [
    # Direct biochemical correspondence
    (683, 10, "Creatinine (SERS)", "creatinine"),
    (795, 10, "Hippuric acid (SERS)", None),  # no blood equivalent
    (999, 10, "Phenylalanine (SERS)", None),  # no blood Phe
    (1130, 15, "Glucose (SERS)", "glucose"),
    (1340, 15, "Adenine/Nucleotide (SERS)", None),
    (724, 10, "Adenine (SERS)", None),

    # Indirect: peak regions vs related markers
    (683, 10, "Creatinine (SERS)", "bun"),       # creatinine ↔ BUN (both renal)
    (683, 10, "Creatinine (SERS)", "uric_acid"),  # renal function cluster
    (1648, 15, "Amide I/Protein (SERS)", "total_protein"),
    (1648, 15, "Amide I/Protein (SERS)", "albumin"),
    (1550, 15, "Amide II (SERS)", "total_protein"),
    (1550, 15, "Amide II (SERS)", "albumin"),

    # Tumor markers (where available)
    (683, 10, "Creatinine (SERS)", "cea"),
    (795, 10, "Hippuric acid (SERS)", "cea"),     # CRC ↔ hippuric acid ↔ CEA?
    (724, 10, "Adenine (SERS)", "ca19_9"),        # PAN ↔ adenine depletion ↔ CA19-9?
    (1648, 15, "Amide I (SERS)", "psa"),          # PRO ↔ protein ↔ PSA?
    (1648, 15, "Amide I (SERS)", "afp"),
]

# Extended: scan ALL wavenumber bands vs ALL blood markers
BLOOD_MARKERS = {
    # Renal
    "creatinine": "Blood Creatinine",
    "bun": "BUN",
    "uric_acid": "Uric Acid",
    # Liver
    "ast": "AST",
    "alt": "ALT",
    "alp": "ALP",
    "ggt": "GGT",
    "total_bilirubin": "Total Bilirubin",
    "ldh": "LDH",
    # Metabolic
    "glucose": "Glucose",
    "hba1c": "HbA1c",
    "total_cholesterol": "Total Cholesterol",
    "triglyceride": "Triglyceride",
    # Protein
    "total_protein": "Total Protein",
    "albumin": "Albumin",
    # Inflammation
    "wbc": "WBC",
    "hs_crp": "hs-CRP",
    # Tumor markers
    "cea": "CEA",
    "ca19_9": "CA 19-9",
    "psa": "PSA",
    "afp": "AFP",
}


def load_spectra_with_clinical():
    """Load mean-aggregated spectra merged with clinical data."""
    print("Loading spectra...")
    spec = pd.read_csv(SPECTRA)

    # Identify wavenumber columns
    wn_cols = [c for c in spec.columns if c.startswith("x_")]
    wavenumbers = np.array([float(c.replace("x_", "")) for c in wn_cols])

    # Mean-aggregate per patient (across replicates)
    meta_cols = ["group", "sample_id"]
    spec_mean = spec.groupby(meta_cols)[wn_cols].mean().reset_index()
    print(f"  {len(spec_mean)} subjects after mean aggregation")

    # Build patient_id for clinical merge
    # BRE/OVA need hospital ID mapping
    from nmf_tumor_marker_correlation import build_id_mapping
    id_map = build_id_mapping()

    patient_ids = []
    for _, row in spec_mean.iterrows():
        group = row["group"]
        sid = str(int(row["sample_id"]))
        solum_label = f"{group} {sid}"
        if solum_label in id_map:
            patient_ids.append(id_map[solum_label])
        elif group == "CPAN":
            patient_ids.append(f"CPAN {sid}")
        else:
            patient_ids.append(f"{group} {sid}")
    spec_mean["patient_id"] = patient_ids

    # Merge clinical
    clin = pd.read_csv(CLINICAL)
    merged = spec_mean.merge(clin, on="patient_id", how="left", suffixes=("", "_clin"))
    print(f"  {merged['creatinine'].notna().sum()} with blood creatinine")
    print(f"  {merged['glucose'].notna().sum()} with blood glucose")
    print(f"  {merged['cea'].notna().sum()} with CEA")

    return merged, wn_cols, wavenumbers


def extract_peak_intensity(df, wn_cols, wavenumbers, center, bandwidth):
    """Extract mean intensity in a wavenumber band."""
    mask = (wavenumbers >= center - bandwidth) & (wavenumbers <= center + bandwidth)
    if mask.sum() == 0:
        return None
    cols = [wn_cols[i] for i in np.where(mask)[0]]
    return df[cols].mean(axis=1)


def hypothesis_driven_analysis(df, wn_cols, wavenumbers):
    """Test specific metabolite peak ↔ blood marker hypotheses."""
    print("\n" + "=" * 70)
    print("PART 1: Hypothesis-Driven Peak ↔ Blood Marker Correlations")
    print("=" * 70)

    results = []
    seen = set()

    for center, bw, peak_name, marker in PEAK_MARKER_PAIRS:
        if marker is None:
            continue
        key = (center, marker)
        if key in seen:
            continue
        seen.add(key)

        peak_vals = extract_peak_intensity(df, wn_cols, wavenumbers, center, bw)
        if peak_vals is None:
            continue

        subset = df.dropna(subset=[marker]).copy()
        if len(subset) < 10:
            continue

        peak_sub = peak_vals.loc[subset.index]
        r, p = stats.spearmanr(peak_sub, subset[marker])

        results.append({
            "peak": peak_name,
            "center_cm": center,
            "blood_marker": BLOOD_MARKERS.get(marker, marker),
            "marker_col": marker,
            "r": r,
            "p": p,
            "n": len(subset),
            "abs_r": abs(r),
        })

    res = pd.DataFrame(results)
    if len(res) > 0:
        reject, pvals_corr, _, _ = multipletests(res["p"], method="fdr_bh")
        res["p_fdr"] = pvals_corr
        res["significant"] = reject
        res = res.sort_values("abs_r", ascending=False)

        print(f"\n  {len(res)} hypothesis tests, {res['significant'].sum()} significant (FDR < 0.05)")
        print(res[["peak", "blood_marker", "r", "p", "p_fdr", "n", "significant"]].to_string(index=False))

    return res


def genome_wide_scan(df, wn_cols, wavenumbers):
    """Scan ALL wavenumber bands against ALL blood markers."""
    print("\n" + "=" * 70)
    print("PART 2: Genome-Wide Scan — Every 20 cm⁻¹ Band × Every Blood Marker")
    print("=" * 70)

    # Create bands every 20 cm⁻¹
    band_centers = np.arange(wavenumbers.min() + 10, wavenumbers.max() - 10, 20)
    bandwidth = 10

    results = []
    for marker_col, marker_name in BLOOD_MARKERS.items():
        subset = df.dropna(subset=[marker_col])
        if len(subset) < 15:
            continue

        for center in band_centers:
            peak_vals = extract_peak_intensity(df, wn_cols, wavenumbers, center, bandwidth)
            if peak_vals is None:
                continue
            peak_sub = peak_vals.loc[subset.index]
            r, p = stats.spearmanr(peak_sub, subset[marker_col])
            results.append({
                "band_center": center,
                "marker": marker_name,
                "marker_col": marker_col,
                "r": r,
                "p": p,
                "n": len(subset),
                "abs_r": abs(r),
            })

    res = pd.DataFrame(results)
    if len(res) == 0:
        print("  No tests performed.")
        return res

    reject, pvals_corr, _, _ = multipletests(res["p"], method="fdr_bh")
    res["p_fdr"] = pvals_corr
    res["significant"] = reject
    res = res.sort_values("abs_r", ascending=False)

    sig = res[res["significant"]]
    print(f"\n  Total tests: {len(res)}")
    print(f"  Significant (FDR < 0.05): {len(sig)}")

    if len(sig) > 0:
        print(f"\n  Top 20 significant correlations:")
        print(sig[["band_center", "marker", "r", "p_fdr", "n"]].head(20).to_string(index=False))
    else:
        # Show top uncorrected anyway
        print(f"\n  Top 20 by uncorrected p-value (none pass FDR):")
        top = res.head(20)
        print(top[["band_center", "marker", "r", "p", "p_fdr", "n"]].to_string(index=False))

    return res


def within_group_scan(df, wn_cols, wavenumbers):
    """Within-group correlations to remove confounding by disease status."""
    print("\n" + "=" * 70)
    print("PART 3: Within-NOR Correlations (n=100, removes disease confounding)")
    print("=" * 70)

    nor = df[df["group"] == "NOR"].copy()
    print(f"  NOR subjects: {len(nor)}")

    band_centers = np.arange(wavenumbers.min() + 10, wavenumbers.max() - 10, 20)

    # Focus on markers with good coverage in NOR
    nor_markers = {}
    for mc, mn in BLOOD_MARKERS.items():
        n_avail = nor[mc].notna().sum()
        if n_avail >= 50:
            nor_markers[mc] = mn
            print(f"  {mn}: {n_avail} available")

    results = []
    for marker_col, marker_name in nor_markers.items():
        subset = nor.dropna(subset=[marker_col])
        for center in band_centers:
            peak_vals = extract_peak_intensity(df, wn_cols, wavenumbers, center, 10)
            if peak_vals is None:
                continue
            peak_sub = peak_vals.loc[subset.index]
            r, p = stats.spearmanr(peak_sub, subset[marker_col])
            results.append({
                "band_center": center,
                "marker": marker_name,
                "marker_col": marker_col,
                "r": r,
                "p": p,
                "n": len(subset),
                "abs_r": abs(r),
            })

    res = pd.DataFrame(results)
    if len(res) == 0:
        return res

    reject, pvals_corr, _, _ = multipletests(res["p"], method="fdr_bh")
    res["p_fdr"] = pvals_corr
    res["significant"] = reject
    res = res.sort_values("abs_r", ascending=False)

    sig = res[res["significant"]]
    print(f"\n  Total tests: {len(res)}")
    print(f"  Significant (FDR < 0.05): {len(sig)}")

    if len(sig) > 0:
        print(f"\n  Significant within-NOR correlations:")
        print(sig[["band_center", "marker", "r", "p_fdr", "n"]].head(20).to_string(index=False))

    # Always show top regardless
    print(f"\n  Top 20 (regardless of significance):")
    print(res[["band_center", "marker", "r", "p", "n"]].head(20).to_string(index=False))

    return res


def plot_results(hyp_res, scan_res, nor_res, wavenumbers):
    """Generate visualization."""
    # 1. Manhattan-style plot: -log10(p) across wavenumbers for each marker
    if len(scan_res) > 0:
        top_markers = scan_res.groupby("marker_col")["abs_r"].max().nlargest(6).index
        fig, axes = plt.subplots(len(top_markers), 1, figsize=(14, 3 * len(top_markers)),
                                  sharex=True)
        if len(top_markers) == 1:
            axes = [axes]

        for ax, mc in zip(axes, top_markers):
            sub = scan_res[scan_res["marker_col"] == mc].sort_values("band_center")
            mn = BLOOD_MARKERS.get(mc, mc)

            # Color by sign of correlation
            colors = ["#d62728" if r < 0 else "#2ca02c" for r in sub["r"]]
            ax.bar(sub["band_center"], -np.log10(sub["p"]), width=18,
                   color=colors, alpha=0.7)
            ax.axhline(-np.log10(0.05), color="gray", ls="--", lw=0.8, label="p=0.05")

            # Mark FDR threshold
            fdr_sig = sub[sub["significant"]]
            if len(fdr_sig) > 0:
                ax.scatter(fdr_sig["band_center"],
                          -np.log10(fdr_sig["p"]),
                          color="gold", s=50, zorder=5, marker="*", label="FDR<0.05")

            ax.set_ylabel(f"-log₁₀(p)")
            ax.set_title(f"Spectral bands vs {mn} (n={sub['n'].iloc[0]})", fontsize=10)
            ax.legend(fontsize=7, loc="upper right")

        axes[-1].set_xlabel("Wavenumber (cm⁻¹)")
        plt.suptitle("Spectral Band ↔ Blood Marker Correlations (Manhattan Plot)",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "manhattan_scan.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  Saved: {OUT_DIR / 'manhattan_scan.png'}")

    # 2. Within-NOR correlation profile
    if len(nor_res) > 0:
        top_markers = nor_res.groupby("marker_col")["abs_r"].max().nlargest(6).index
        fig, axes = plt.subplots(len(top_markers), 1, figsize=(14, 2.5 * len(top_markers)),
                                  sharex=True)
        if len(top_markers) == 1:
            axes = [axes]

        for ax, mc in zip(axes, top_markers):
            sub = nor_res[nor_res["marker_col"] == mc].sort_values("band_center")
            mn = BLOOD_MARKERS.get(mc, mc)
            ax.plot(sub["band_center"], sub["r"], color="#1f77b4", lw=1.2)
            ax.fill_between(sub["band_center"], sub["r"], 0, alpha=0.15, color="#1f77b4")
            ax.axhline(0, color="black", lw=0.5)
            ax.set_ylabel(f"ρ vs {mn}", fontsize=8)
            ax.set_ylim(-0.5, 0.5)

            # Annotate peaks
            top3 = sub.nlargest(3, "abs_r")
            for _, row in top3.iterrows():
                ax.annotate(f"{row['band_center']:.0f}\nρ={row['r']:.2f}",
                           xy=(row["band_center"], row["r"]),
                           fontsize=6, ha="center", color="red")

        axes[-1].set_xlabel("Wavenumber (cm⁻¹)")
        plt.suptitle("Within-NOR: Spectral Profile ↔ Blood Marker Correlation",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "nor_correlation_profile.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {OUT_DIR / 'nor_correlation_profile.png'}")

    # 3. Hypothesis-driven scatter plots
    if len(hyp_res) > 0:
        top = hyp_res.nlargest(6, "abs_r")
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()

        for idx, (_, row) in enumerate(top.iterrows()):
            ax = axes[idx]
            center = row["center_cm"]
            marker_col = row["marker_col"]
            # Recompute for scatter
            # (simplified - just show NOR data)
            ax.set_title(f"{row['peak']} vs {row['blood_marker']}\nρ={row['r']:.3f}, p={row['p']:.3e}, n={row['n']}",
                        fontsize=8)
            ax.set_xlabel(row["peak"], fontsize=8)
            ax.set_ylabel(row["blood_marker"], fontsize=8)

        for idx in range(len(top), 6):
            axes[idx].set_visible(False)

        plt.suptitle("Hypothesis-Driven: Peak Intensity vs Blood Marker", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "hypothesis_scatters.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {OUT_DIR / 'hypothesis_scatters.png'}")


def main():
    import sys
    sys.path.insert(0, str(PROJECT / "scripts/analysis"))

    df, wn_cols, wavenumbers = load_spectra_with_clinical()

    # Part 1: Hypothesis-driven
    hyp_res = hypothesis_driven_analysis(df, wn_cols, wavenumbers)

    # Part 2: Genome-wide scan
    scan_res = genome_wide_scan(df, wn_cols, wavenumbers)

    # Part 3: Within-NOR (cleanest test)
    nor_res = within_group_scan(df, wn_cols, wavenumbers)

    # Visualize
    print("\n" + "=" * 70)
    print("Generating plots...")
    plot_results(hyp_res, scan_res, nor_res, wavenumbers)

    # Save all results
    for name, res in [("hypothesis", hyp_res), ("scan", scan_res), ("within_nor", nor_res)]:
        if len(res) > 0:
            res.to_csv(OUT_DIR / f"{name}_correlations.csv", index=False)
            print(f"  Saved: {OUT_DIR / f'{name}_correlations.csv'}")

    print(f"\nAll outputs: {OUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
