#!/usr/bin/env python3
"""
YPAN Peak ↔ Clinical Marker Correlation Analysis
=================================================
YPAN(세브란스 췌장암) 환자의 SERS 스펙트럼 peak 강도와
종양 표지자(CEA, CA19-9), 간기능(AST, ALT, T.Bil), 음주/흡연력의
상관성을 분석.

Analysis:
    1. Peak intensity extraction (known metabolite peaks)
    2. Spearman correlation: peak intensity ↔ tumor markers / liver function
    3. Peak intensity by smoking/drinking status (Kruskal-Wallis)
    4. Heatmap: all peaks × all clinical variables
    5. Scatter plots for significant correlations

Usage:
    python scripts/analysis/ypan_peak_clinical_correlation.py
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parents[2]
SPECTRA_PATH = PROJECT / "results/data/processed_spectra_cal_newqc.csv"
CLINICAL_PATH = PROJECT / "data/clinical_data/standardized/YPAN_clinical_standardized.csv"
OUT_DIR = PROJECT / "results/figures/ypan_peak_clinical"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Known metabolite peaks (cm⁻¹) and assignments ─────────────────────
PEAKS = {
    442: "Uric acid (NH₂ def.)",
    527: "Urea (ring def.)",
    605: "Creatinine (ring def.)",
    618: "C-S stretch",
    683: "Creatinine (ring breathing)",
    724: "Adenine (ring breathing)",
    754: "Tryptophan (ring)",
    795: "Hippuric acid (CH₂ rock)",
    849: "Tyr/Trp Fermi resonance",
    895: "C-C stretch",
    999: "Phenylalanine (ring)",
    1050: "Glycogen (C-O)",
    1130: "Uric acid (C-N)",
    1148: "C-N/C-O-C stretch",
    1231: "Amide III",
    1293: "CH₂ twist / Lipid",
    1352: "Trp / Nucleic acid",
    1420: "Creatinine (C=C)",
    1449: "CH₂ deformation / Lipid",
    1597: "C=C / Purine ring",
    1651: "Amide I (protein)",
    1680: "C=O / Nucleic acid",
}

# Peak bandwidth for integration (±bw cm⁻¹)
PEAK_BW = 8  # ±8 cm⁻¹

# Clinical variables to correlate
TUMOR_MARKERS = ["cea", "ca19_9"]
LIVER_MARKERS = ["ast", "alt", "total_bilirubin"]
CONTINUOUS_MARKERS = TUMOR_MARKERS + LIVER_MARKERS
MARKER_LABELS = {
    "cea": "CEA (ng/mL)",
    "ca19_9": "CA 19-9 (U/mL)",
    "ast": "AST (U/L)",
    "alt": "ALT (U/L)",
    "total_bilirubin": "T.Bil (mg/dL)",
}

CATEGORICAL_VARS = {
    "smoking_status": {0: "Non-smoker", 1: "Ex-smoker", 2: "Current"},
    "drinking_status": {0: "Non-drinker", 1: "Ex-drinker", 2: "Current"},
}


def extract_peak_intensities(
    spectra_df: pd.DataFrame, wavenumber_cols: list[str]
) -> pd.DataFrame:
    """Extract peak intensities by averaging within ±PEAK_BW of each peak center."""
    wn_values = np.array([float(c.replace("x_", "")) for c in wavenumber_cols])
    intensity_matrix = spectra_df[wavenumber_cols].values

    peak_data = {}
    for center, label in PEAKS.items():
        mask = (wn_values >= center - PEAK_BW) & (wn_values <= center + PEAK_BW)
        if mask.sum() == 0:
            continue
        peak_data[f"{center}"] = intensity_matrix[:, mask].mean(axis=1)

    return pd.DataFrame(peak_data, index=spectra_df.index)


def main():
    print("=" * 70)
    print("YPAN Peak ↔ Clinical Marker Correlation Analysis")
    print("=" * 70)

    # ── Load data ──────────────────────────────────────────────────────
    spectra = pd.read_csv(SPECTRA_PATH)
    clinical = pd.read_csv(CLINICAL_PATH)

    # Filter YPAN spectra
    ypan_spectra = spectra[spectra["group"] == "YPAN"].copy()
    print(f"\nYPAN spectra: {len(ypan_spectra)} rows, {ypan_spectra['sample_id'].nunique()} patients")
    print(f"Clinical data: {len(clinical)} patients")

    # Wavenumber columns
    wn_cols = [c for c in spectra.columns if c.startswith("x_")]

    # Extract peak intensities (per-spectrum)
    peak_df = extract_peak_intensities(ypan_spectra, wn_cols)
    peak_df["sample_id"] = ypan_spectra["sample_id"].values
    peak_df["replicate"] = ypan_spectra["replicate"].values

    # Average replicates → patient-level
    peak_patient = peak_df.groupby("sample_id")[list(PEAKS.keys().__str__() for _ in [])].mean()
    # Proper groupby on peak columns
    peak_cols = [str(k) for k in PEAKS.keys() if str(k) in peak_df.columns]
    peak_patient = peak_df.groupby("sample_id")[peak_cols].mean().reset_index()
    print(f"Patient-level peak matrix: {peak_patient.shape}")

    # Merge with clinical — patient_id is "YPAN 4", sample_id is int 4
    clinical["_merge_id"] = clinical["patient_id"].str.extract(r"(\d+)$").astype(int)
    merged = peak_patient.merge(clinical, left_on="sample_id", right_on="_merge_id", how="inner")
    print(f"Merged (spectra ∩ clinical): {len(merged)} patients")

    if len(merged) < 5:
        print("ERROR: Too few matched patients. Check ID alignment.")
        return

    # ══════════════════════════════════════════════════════════════════════
    # 1. Spearman correlation: peaks × continuous markers
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("1. Spearman Correlation: Peak Intensity ↔ Tumor Markers / Liver")
    print("─" * 70)

    results = []
    for marker in CONTINUOUS_MARKERS:
        if marker not in merged.columns or merged[marker].isna().all():
            print(f"  Skipping {marker} (no data)")
            continue
        for peak_wn in peak_cols:
            valid = merged[[peak_wn, marker]].dropna()
            if len(valid) < 5:
                continue
            rho, pval = stats.spearmanr(valid[peak_wn], valid[marker])
            results.append({
                "peak_cm1": int(peak_wn),
                "metabolite": PEAKS[int(peak_wn)],
                "marker": marker,
                "marker_label": MARKER_LABELS.get(marker, marker),
                "n": len(valid),
                "rho": rho,
                "p_value": pval,
            })

    corr_df = pd.DataFrame(results)
    if len(corr_df) > 0:
        # FDR correction
        _, corr_df["p_adj"], _, _ = multipletests(corr_df["p_value"], method="fdr_bh")
        corr_df["sig"] = corr_df["p_adj"] < 0.05
        corr_df = corr_df.sort_values("p_value")

        # Print significant
        sig = corr_df[corr_df["sig"]]
        print(f"\n  Total tests: {len(corr_df)}, Significant (FDR < 0.05): {len(sig)}")
        if len(sig) > 0:
            print("\n  Significant correlations:")
            for _, row in sig.iterrows():
                print(f"    {row['peak_cm1']} cm⁻¹ ({row['metabolite']}) ↔ {row['marker_label']}: "
                      f"ρ={row['rho']:.3f}, p_adj={row['p_adj']:.4f}")
        else:
            print("  No FDR-significant correlations found.")
            # Show top uncorrected
            print("\n  Top 10 uncorrected (p < 0.1):")
            top = corr_df[corr_df["p_value"] < 0.1].head(10)
            for _, row in top.iterrows():
                print(f"    {row['peak_cm1']} cm⁻¹ ({row['metabolite']}) ↔ {row['marker_label']}: "
                      f"ρ={row['rho']:.3f}, p={row['p_value']:.4f}")

        corr_df.to_csv(OUT_DIR / "peak_marker_spearman.csv", index=False)

    # ══════════════════════════════════════════════════════════════════════
    # 2. Heatmap: peaks × clinical markers
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("2. Correlation Heatmap")
    print("─" * 70)

    heatmap_data = pd.DataFrame(index=peak_cols, columns=CONTINUOUS_MARKERS, dtype=float)
    pval_data = pd.DataFrame(index=peak_cols, columns=CONTINUOUS_MARKERS, dtype=float)

    for marker in CONTINUOUS_MARKERS:
        if marker not in merged.columns or merged[marker].isna().all():
            continue
        for peak_wn in peak_cols:
            valid = merged[[peak_wn, marker]].dropna()
            if len(valid) < 5:
                continue
            rho, pval = stats.spearmanr(valid[peak_wn], valid[marker])
            heatmap_data.loc[peak_wn, marker] = rho
            pval_data.loc[peak_wn, marker] = pval

    heatmap_data = heatmap_data.dropna(how="all", axis=1).astype(float)

    if heatmap_data.shape[1] > 0:
        fig, ax = plt.subplots(figsize=(8, 12))
        # Annotation: rho values with significance stars
        annot = heatmap_data.copy().round(2).astype(str)
        for col in pval_data.columns:
            if col in annot.columns:
                for idx in annot.index:
                    p = pval_data.loc[idx, col]
                    if pd.notna(p) and p < 0.01:
                        annot.loc[idx, col] += "**"
                    elif pd.notna(p) and p < 0.05:
                        annot.loc[idx, col] += "*"

        # Row labels: wavenumber + metabolite
        row_labels = [f"{wn} ({PEAKS[int(wn)]})" for wn in heatmap_data.index]
        col_labels = [MARKER_LABELS.get(c, c) for c in heatmap_data.columns]

        sns.heatmap(
            heatmap_data.values.astype(float),
            xticklabels=col_labels,
            yticklabels=row_labels,
            annot=annot.values,
            fmt="",
            cmap="RdBu_r",
            center=0,
            vmin=-0.7, vmax=0.7,
            linewidths=0.5,
            ax=ax,
        )
        ax.set_title("YPAN: Peak Intensity ↔ Clinical Marker (Spearman ρ)\n*p<0.05, **p<0.01", fontsize=12)
        ax.set_xlabel("")
        ax.set_ylabel("SERS Peak (cm⁻¹)")
        plt.tight_layout()
        fig.savefig(OUT_DIR / "heatmap_peak_marker_correlation.png", dpi=200)
        plt.close()
        print(f"  Saved: {OUT_DIR / 'heatmap_peak_marker_correlation.png'}")

    # ══════════════════════════════════════════════════════════════════════
    # 3. Smoking / Drinking ↔ Peak intensity (Kruskal-Wallis)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("3. Smoking / Drinking Status ↔ Peak Intensity")
    print("─" * 70)

    cat_results = []
    for cat_var, labels in CATEGORICAL_VARS.items():
        if cat_var not in merged.columns:
            continue
        print(f"\n  [{cat_var}] Distribution: {merged[cat_var].value_counts().to_dict()}")

        for peak_wn in peak_cols:
            groups_data = []
            group_names = []
            for val, name in labels.items():
                subset = merged.loc[merged[cat_var] == val, peak_wn].dropna()
                if len(subset) >= 2:
                    groups_data.append(subset.values)
                    group_names.append(f"{name} (n={len(subset)})")

            if len(groups_data) < 2:
                continue

            # Kruskal-Wallis (non-parametric ANOVA)
            if len(groups_data) == 2:
                stat, pval = stats.mannwhitneyu(groups_data[0], groups_data[1], alternative="two-sided")
                test_name = "Mann-Whitney U"
            else:
                stat, pval = stats.kruskal(*groups_data)
                test_name = "Kruskal-Wallis"

            cat_results.append({
                "variable": cat_var,
                "peak_cm1": int(peak_wn),
                "metabolite": PEAKS[int(peak_wn)],
                "test": test_name,
                "statistic": stat,
                "p_value": pval,
                "group_means": {name: f"{g.mean():.4f}" for name, g in zip(group_names, groups_data)},
            })

    cat_df = pd.DataFrame(cat_results)
    if len(cat_df) > 0:
        for var in CATEGORICAL_VARS:
            sub = cat_df[cat_df["variable"] == var].copy()
            if len(sub) == 0:
                continue
            _, sub["p_adj"], _, _ = multipletests(sub["p_value"], method="fdr_bh")
            sub["sig"] = sub["p_adj"] < 0.05
            sub = sub.sort_values("p_value")

            sig_count = sub["sig"].sum()
            print(f"\n  {var}: {len(sub)} tests, {sig_count} FDR-significant")
            top = sub.head(5)
            for _, row in top.iterrows():
                star = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else ""
                print(f"    {row['peak_cm1']} cm⁻¹ ({row['metabolite']}): "
                      f"p={row['p_value']:.4f}{star}, p_adj={row['p_adj']:.4f}")
                print(f"      means: {row['group_means']}")

        cat_df.to_csv(OUT_DIR / "peak_smoking_drinking_kruskal.csv", index=False)

    # ══════════════════════════════════════════════════════════════════════
    # 4. Scatter plots for top correlations
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("4. Scatter Plots (Top Correlations)")
    print("─" * 70)

    if len(corr_df) > 0:
        # Plot top 6 by |rho|
        top_corr = corr_df.nlargest(6, "rho", keep="first") if corr_df["rho"].abs().max() > 0 else corr_df.head(6)
        top_corr = corr_df.reindex(corr_df["rho"].abs().nlargest(6).index)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, (_, row) in enumerate(top_corr.iterrows()):
            if idx >= 6:
                break
            ax = axes[idx]
            pk = str(row["peak_cm1"])
            mk = row["marker"]
            valid = merged[[pk, mk]].dropna()

            ax.scatter(valid[pk], valid[mk], alpha=0.7, edgecolors="k", linewidth=0.5, s=50)

            # Trend line
            if len(valid) >= 3:
                z = np.polyfit(valid[pk].values, valid[mk].values, 1)
                x_line = np.linspace(valid[pk].min(), valid[pk].max(), 100)
                ax.plot(x_line, np.polyval(z, x_line), "r--", alpha=0.7)

            star = ""
            if row["p_value"] < 0.001:
                star = "***"
            elif row["p_value"] < 0.01:
                star = "**"
            elif row["p_value"] < 0.05:
                star = "*"

            ax.set_xlabel(f"{row['peak_cm1']} cm⁻¹\n({row['metabolite']})", fontsize=9)
            ax.set_ylabel(row["marker_label"], fontsize=9)
            ax.set_title(f"ρ={row['rho']:.3f}, p={row['p_value']:.3f}{star}", fontsize=10)

        for idx in range(len(top_corr), 6):
            axes[idx].set_visible(False)

        fig.suptitle("YPAN: Top Peak ↔ Clinical Marker Correlations", fontsize=14, y=1.02)
        plt.tight_layout()
        fig.savefig(OUT_DIR / "scatter_top_correlations.png", dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {OUT_DIR / 'scatter_top_correlations.png'}")

    # ══════════════════════════════════════════════════════════════════════
    # 5. Box plots: smoking/drinking × top peaks
    # ══════════════════════════════════════════════════════════════════════
    if len(cat_df) > 0:
        for cat_var, labels in CATEGORICAL_VARS.items():
            sub = cat_df[cat_df["variable"] == cat_var].sort_values("p_value")
            top_peaks = sub.head(4)["peak_cm1"].values

            if len(top_peaks) == 0:
                continue

            n_plots = min(4, len(top_peaks))
            fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 5))
            if n_plots == 1:
                axes = [axes]

            for i, peak_wn in enumerate(top_peaks[:n_plots]):
                ax = axes[i]
                pk = str(peak_wn)
                plot_data = []
                for val, name in labels.items():
                    subset = merged.loc[merged[cat_var] == val, pk].dropna()
                    for v in subset:
                        plot_data.append({"status": name, "intensity": v})

                if plot_data:
                    pdf = pd.DataFrame(plot_data)
                    sns.boxplot(data=pdf, x="status", y="intensity", ax=ax, palette="Set2")
                    sns.stripplot(data=pdf, x="status", y="intensity", ax=ax,
                                  color="black", alpha=0.5, size=4)

                pval = sub[sub["peak_cm1"] == peak_wn]["p_value"].values[0]
                ax.set_title(f"{peak_wn} cm⁻¹\n({PEAKS[peak_wn]})\np={pval:.4f}", fontsize=9)
                ax.set_xlabel("")
                ax.set_ylabel("Peak intensity (SNV)")

            display_name = "Smoking" if cat_var == "smoking_status" else "Drinking"
            fig.suptitle(f"YPAN: {display_name} Status ↔ Peak Intensity (Top 4)", fontsize=13)
            plt.tight_layout()
            fig.savefig(OUT_DIR / f"boxplot_{cat_var}_peaks.png", dpi=200, bbox_inches="tight")
            plt.close()
            print(f"  Saved: {OUT_DIR / f'boxplot_{cat_var}_peaks.png'}")

    # ══════════════════════════════════════════════════════════════════════
    # 6. CA19-9 stratified spectrum comparison
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("5. CA19-9 High vs Low Spectrum Comparison")
    print("─" * 70)

    if "ca19_9" in merged.columns and merged["ca19_9"].notna().sum() >= 10:
        ca19_9_median = merged["ca19_9"].median()
        high = merged[merged["ca19_9"] >= ca19_9_median]
        low = merged[merged["ca19_9"] < ca19_9_median]
        print(f"  CA19-9 median: {ca19_9_median:.1f}")
        print(f"  High (≥median): n={len(high)}, Low (<median): n={len(low)}")

        wn_values = np.array([float(c.replace("x_", "")) for c in wn_cols])

        # Get patient-level mean spectra
        ypan_patient = ypan_spectra.groupby("sample_id")[wn_cols].mean().reset_index()
        clin_ca = clinical[["_merge_id", "ca19_9"]].copy()
        ypan_patient = ypan_patient.merge(
            clin_ca, left_on="sample_id", right_on="_merge_id", how="inner"
        )

        high_spec = ypan_patient[ypan_patient["ca19_9"] >= ca19_9_median][wn_cols].values
        low_spec = ypan_patient[ypan_patient["ca19_9"] < ca19_9_median][wn_cols].values

        if len(high_spec) > 0 and len(low_spec) > 0:
            fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

            # Mean spectra
            ax = axes[0]
            h_mean = high_spec.mean(axis=0)
            l_mean = low_spec.mean(axis=0)
            h_sem = high_spec.std(axis=0) / np.sqrt(len(high_spec))
            l_sem = low_spec.std(axis=0) / np.sqrt(len(low_spec))

            ax.plot(wn_values, h_mean, color="#E53935", label=f"CA19-9 High (n={len(high_spec)})", linewidth=1.2)
            ax.fill_between(wn_values, h_mean - h_sem, h_mean + h_sem, color="#E53935", alpha=0.15)
            ax.plot(wn_values, l_mean, color="#1E88E5", label=f"CA19-9 Low (n={len(low_spec)})", linewidth=1.2)
            ax.fill_between(wn_values, l_mean - l_sem, l_mean + l_sem, color="#1E88E5", alpha=0.15)
            ax.set_ylabel("Intensity (SNV)")
            ax.legend(fontsize=10)
            ax.set_title(f"YPAN Mean Spectra: CA19-9 High vs Low (cutoff={ca19_9_median:.0f} U/mL)", fontsize=12)

            # Difference spectrum with peak labels
            ax2 = axes[1]
            diff = h_mean - l_mean
            ax2.plot(wn_values, diff, color="#333333", linewidth=1)
            ax2.axhline(0, color="gray", linestyle="--", linewidth=0.5)
            ax2.fill_between(wn_values, diff, 0, where=diff > 0, color="#E53935", alpha=0.3)
            ax2.fill_between(wn_values, diff, 0, where=diff < 0, color="#1E88E5", alpha=0.3)
            ax2.set_xlabel("Raman Shift (cm⁻¹)")
            ax2.set_ylabel("Δ Intensity\n(High − Low)")

            # Mark known peaks
            for center in PEAKS:
                if wn_values.min() <= center <= wn_values.max():
                    idx = np.argmin(np.abs(wn_values - center))
                    ax2.axvline(center, color="gray", alpha=0.2, linewidth=0.5)

            plt.tight_layout()
            fig.savefig(OUT_DIR / "spectra_ca19_9_high_vs_low.png", dpi=200, bbox_inches="tight")
            plt.close()
            print(f"  Saved: {OUT_DIR / 'spectra_ca19_9_high_vs_low.png'}")

    # ══════════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print(f"Results saved to: {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
