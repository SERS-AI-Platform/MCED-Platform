#!/usr/bin/env python3
"""
NMF Component ↔ Tumor Marker Correlation Analysis
==================================================
NMF W matrix (per-sample component weights)와 종양마커(CEA, CA19-9, PSA, AFP)의
상관성을 분석하여 SERS 스펙트럼이 종양마커 정보를 캡처하는지 검증.

Usage:
    python scripts/analysis/nmf_tumor_marker_correlation.py
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT = Path("/home/user/SERS-AI")
NMF_DIR = PROJECT / "metabolite_profiling/experiments/results/nmf"
PREP_DIR = PROJECT / "metabolite_profiling/experiments/results/prepared_data"
CLINICAL = PROJECT / "data/clinical_data/standardized/all_clinical_standardized.csv"
OUT_DIR = PROJECT / "results/figures/nmf_tumor_marker_correlation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# NMF models to analyze
NMF_MODELS = {
    "Frobenius k=20": "nmf_frob_k20.npz",
    "Thermo k=20": "nmf_thermo_k20.npz",
}

# Group name mapping: NMF metadata → clinical data
GROUP_MAP = {"CPAN": "PAN"}  # NMF uses CPAN, clinical uses PAN

TUMOR_MARKERS = ["cea", "ca19_9", "psa", "afp"]
MARKER_LABELS = {"cea": "CEA", "ca19_9": "CA 19-9", "psa": "PSA", "afp": "AFP"}

# Also check indirect markers (lab values)
LAB_MARKERS = ["ldh", "alp", "ggt", "hs_crp", "wbc", "creatinine", "glucose"]
LAB_LABELS = {
    "ldh": "LDH", "alp": "ALP", "ggt": "GGT", "hs_crp": "hs-CRP",
    "wbc": "WBC", "creatinine": "Creatinine", "glucose": "Glucose"
}

ALL_MARKERS = TUMOR_MARKERS + LAB_MARKERS
ALL_LABELS = {**MARKER_LABELS, **LAB_LABELS}


def build_id_mapping() -> dict:
    """Build SoluM Label → hospital_id mapping for BRE and OVA."""
    mapping = {}

    # BRE: SoluM Label in col 4, hospital_id in col 1
    bre_raw = pd.read_excel(
        PROJECT / "data/clinical_data/2. 유방암/SMCXD01_유방암.xlsx",
        header=None, skiprows=2
    )
    for _, row in bre_raw.iterrows():
        hid = str(row[1]).strip().split(".")[0] if pd.notna(row[1]) else None
        label = str(row[4]).strip() if pd.notna(row[4]) else None
        if hid and label and label.startswith("BRE"):
            mapping[label] = hid

    # OVA file 1 (OVA 1-30): hospital_id in col 1, label in col 4
    ova1 = pd.read_excel(
        PROJECT / "data/clinical_data/3. 난소암/SMCXD01_난소암 1.xlsx",
        header=None, skiprows=2
    )
    for _, row in ova1.iterrows():
        hid = str(row[1]).strip().split(".")[0] if pd.notna(row[1]) else None
        label = str(row[4]).strip() if pd.notna(row[4]) else None
        if hid and label and label.startswith("OVA"):
            mapping[label] = hid

    # OVA file 2 (OVA 31-70): hospital_id in col 3, label in col 20
    ova2 = pd.read_excel(
        PROJECT / "data/clinical_data/3. 난소암/SMCXD01_난소암 2.xlsx",
        header=None, skiprows=1
    )
    for _, row in ova2.iterrows():
        hid = str(row[3]).strip().split(".")[0] if pd.notna(row[3]) else None
        label = str(row[20]).strip() if pd.notna(row[20]) else None
        if hid and label and label.startswith("OVA"):
            mapping[label] = hid

    return mapping


def load_nmf_with_clinical(nmf_file: str) -> pd.DataFrame:
    """Load NMF W matrix and merge with clinical data."""
    # Load NMF
    data = np.load(NMF_DIR / nmf_file)
    W = data["W"]  # (n_samples, k)
    n_samples, k = W.shape

    # Load metadata
    meta = pd.read_csv(PREP_DIR / "metadata_mean.csv")
    assert len(meta) == n_samples, f"Metadata {len(meta)} != W {n_samples}"

    # Build patient_id with proper mapping
    # Groups with simple "GROUP N" format: NOR, PRO, LUN, CRC
    # CPAN → clinical patient_id is "CPAN N" (disease_group=PAN)
    # BLC → clinical uses different format (unmatchable for now)
    # BRE, OVA → hospital IDs, need SoluM Label mapping
    id_map = build_id_mapping()

    patient_ids = []
    for _, row in meta.iterrows():
        group, sid = row["group"], str(int(row["sample_id"]))
        solum_label = f"{group} {sid}"

        if solum_label in id_map:
            # BRE/OVA: use hospital ID
            patient_ids.append(id_map[solum_label])
        elif group == "CPAN":
            # CPAN in NMF → "CPAN N" in clinical
            patient_ids.append(f"CPAN {sid}")
        else:
            # NOR, PRO, LUN, CRC, BLC: "GROUP N"
            patient_ids.append(f"{group} {sid}")

    meta["patient_id"] = patient_ids

    # Add NMF components
    comp_cols = [f"C{i}" for i in range(k)]
    for i, col in enumerate(comp_cols):
        meta[col] = W[:, i]

    # Merge with clinical
    clin = pd.read_csv(CLINICAL)
    merged = meta.merge(clin, on="patient_id", how="left", suffixes=("", "_clin"))

    return merged, comp_cols


def compute_correlations(df: pd.DataFrame, comp_cols: list, markers: list) -> pd.DataFrame:
    """Compute Spearman correlations between all NMF components and markers."""
    results = []
    for marker in markers:
        for comp in comp_cols:
            subset = df.dropna(subset=[marker, comp])
            if len(subset) < 10:
                continue
            r, p = stats.spearmanr(subset[comp], subset[marker])
            results.append({
                "component": comp,
                "marker": marker,
                "marker_label": ALL_LABELS.get(marker, marker),
                "r": r,
                "p": p,
                "n": len(subset),
                "abs_r": abs(r),
            })

    res = pd.DataFrame(results)
    if len(res) == 0:
        return res

    # FDR correction (Benjamini-Hochberg)
    reject, pvals_corrected, _, _ = multipletests(res["p"], method="fdr_bh")
    res["p_fdr"] = pvals_corrected
    res["significant"] = reject
    return res.sort_values("abs_r", ascending=False)


def compute_group_correlations(df: pd.DataFrame, comp_cols: list, markers: list) -> pd.DataFrame:
    """Compute correlations within each disease group separately."""
    results = []
    for group in df["group"].unique():
        gdf = df[df["group"] == group]
        for marker in markers:
            for comp in comp_cols:
                subset = gdf.dropna(subset=[marker, comp])
                if len(subset) < 8:
                    continue
                r, p = stats.spearmanr(subset[comp], subset[marker])
                results.append({
                    "group": group,
                    "component": comp,
                    "marker": marker,
                    "marker_label": ALL_LABELS.get(marker, marker),
                    "r": r,
                    "p": p,
                    "n": len(subset),
                    "abs_r": abs(r),
                })

    res = pd.DataFrame(results)
    if len(res) > 0:
        reject, pvals_corrected, _, _ = multipletests(res["p"], method="fdr_bh")
        res["p_fdr"] = pvals_corrected
        res["significant"] = reject
    return res.sort_values("abs_r", ascending=False)


def plot_correlation_heatmap(corr_df: pd.DataFrame, comp_cols: list, markers: list,
                              title: str, save_path: Path):
    """Heatmap of component × marker correlations."""
    marker_labels = [ALL_LABELS.get(m, m) for m in markers]

    # Pivot to matrix
    pivot_r = pd.DataFrame(0.0, index=comp_cols, columns=markers)
    pivot_sig = pd.DataFrame("", index=comp_cols, columns=markers)

    for _, row in corr_df.iterrows():
        pivot_r.loc[row["component"], row["marker"]] = row["r"]
        if row["significant"]:
            pivot_sig.loc[row["component"], row["marker"]] = "*"
            if row["p_fdr"] < 0.001:
                pivot_sig.loc[row["component"], row["marker"]] = "***"
            elif row["p_fdr"] < 0.01:
                pivot_sig.loc[row["component"], row["marker"]] = "**"

    fig, ax = plt.subplots(figsize=(max(8, len(markers) * 0.9), max(6, len(comp_cols) * 0.35)))
    im = ax.imshow(pivot_r.values, cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")

    # Annotations
    for i in range(len(comp_cols)):
        for j in range(len(markers)):
            val = pivot_r.iloc[i, j]
            sig = pivot_sig.iloc[i, j]
            if val != 0:
                color = "white" if abs(val) > 0.35 else "black"
                ax.text(j, i, f"{val:.2f}{sig}", ha="center", va="center",
                        fontsize=7, color=color, fontweight="bold" if sig else "normal")

    ax.set_xticks(range(len(markers)))
    ax.set_xticklabels(marker_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(comp_cols)))
    ax.set_yticklabels(comp_cols, fontsize=8)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    plt.colorbar(im, ax=ax, label="Spearman ρ", shrink=0.8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_top_scatters(df: pd.DataFrame, corr_df: pd.DataFrame, top_n: int = 12,
                       title_prefix: str = "", save_path: Path = None):
    """Scatter plots for top correlated component-marker pairs."""
    top = corr_df.head(top_n)
    if len(top) == 0:
        print("  No significant correlations for scatter plots.")
        return

    ncols = 4
    nrows = (len(top) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = axes.flatten() if nrows > 1 else [axes] if ncols == 1 else axes.flatten()

    colors = {
        "BLC": "#1f77b4", "BRE": "#ff7f0e", "CRC": "#2ca02c", "LUN": "#d62728",
        "NOR": "#9467bd", "OVA": "#8c564b", "CPAN": "#e377c2", "PRO": "#7f7f7f",
    }

    for idx, (_, row) in enumerate(top.iterrows()):
        ax = axes[idx]
        comp, marker = row["component"], row["marker"]
        subset = df.dropna(subset=[marker, comp])

        for grp in subset["group"].unique():
            gdf = subset[subset["group"] == grp]
            ax.scatter(gdf[comp], gdf[marker], s=15, alpha=0.6,
                      color=colors.get(grp, "#333"), label=grp, edgecolors="none")

        ax.set_xlabel(comp, fontsize=9)
        ax.set_ylabel(ALL_LABELS.get(marker, marker), fontsize=9)

        sig_str = f"{'*' if row['significant'] else ''}"
        ax.set_title(f"ρ={row['r']:.3f} (n={row['n']}){sig_str}", fontsize=9)
        ax.legend(fontsize=6, markerscale=0.8, loc="best")

    # Hide unused axes
    for idx in range(len(top), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f"{title_prefix} — Top {len(top)} Correlations", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_group_heatmap(group_corr: pd.DataFrame, save_path: Path):
    """Heatmap of top correlations broken down by disease group."""
    # Filter significant only
    sig = group_corr[group_corr["significant"]].copy()
    if len(sig) == 0:
        print("  No group-level significant correlations.")
        return

    # Show top 20
    top = sig.head(20)
    top["label"] = top["group"] + " | " + top["component"] + " ↔ " + top["marker_label"]

    fig, ax = plt.subplots(figsize=(8, max(4, len(top) * 0.35)))
    colors_list = ["#d62728" if r < 0 else "#2ca02c" for r in top["r"]]
    bars = ax.barh(range(len(top)), top["r"], color=colors_list, alpha=0.8)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["label"], fontsize=8)
    ax.set_xlabel("Spearman ρ", fontsize=10)
    ax.set_title("Top Group-Level Significant Correlations (FDR < 0.05)", fontsize=11, fontweight="bold")
    ax.axvline(0, color="black", lw=0.5)

    # Annotate n
    for i, (_, row) in enumerate(top.iterrows()):
        ax.text(row["r"] + (0.01 if row["r"] >= 0 else -0.01), i,
                f"n={row['n']}", va="center", ha="left" if row["r"] >= 0 else "right", fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    print("=" * 70)
    print("NMF Component ↔ Tumor Marker Correlation Analysis")
    print("=" * 70)

    for model_name, nmf_file in NMF_MODELS.items():
        print(f"\n{'─' * 50}")
        print(f"Model: {model_name}")
        print(f"{'─' * 50}")

        df, comp_cols = load_nmf_with_clinical(nmf_file)

        # Sample overlap summary
        print(f"\n[Sample Overlap Summary]")
        for marker in ALL_MARKERS:
            avail = df.dropna(subset=[marker])
            if len(avail) > 0:
                groups = avail["group"].value_counts().to_dict()
                print(f"  {ALL_LABELS.get(marker, marker):12s}: {len(avail):4d} samples — {groups}")

        # ── 1. Global correlations (all groups pooled) ──
        print(f"\n[1] Global Correlations (all groups pooled)")

        # Primary tumor markers
        corr_tumor = compute_correlations(df, comp_cols, TUMOR_MARKERS)
        if len(corr_tumor) > 0:
            sig_t = corr_tumor[corr_tumor["significant"]]
            print(f"  Tumor markers: {len(sig_t)}/{len(corr_tumor)} significant (FDR < 0.05)")
            if len(sig_t) > 0:
                print(sig_t[["component", "marker_label", "r", "p_fdr", "n"]].head(10).to_string(index=False))

            safe_name = model_name.replace(" ", "_").replace("=", "")
            plot_correlation_heatmap(
                corr_tumor, comp_cols, TUMOR_MARKERS,
                f"{model_name} — NMF Components × Tumor Markers",
                OUT_DIR / f"heatmap_tumor_{safe_name}.png"
            )
            plot_top_scatters(
                df, corr_tumor, top_n=12, title_prefix=f"{model_name} — Tumor Markers",
                save_path=OUT_DIR / f"scatter_tumor_{safe_name}.png"
            )

        # Lab markers
        corr_lab = compute_correlations(df, comp_cols, LAB_MARKERS)
        if len(corr_lab) > 0:
            sig_l = corr_lab[corr_lab["significant"]]
            print(f"\n  Lab markers: {len(sig_l)}/{len(corr_lab)} significant (FDR < 0.05)")
            if len(sig_l) > 0:
                print(sig_l[["component", "marker_label", "r", "p_fdr", "n"]].head(10).to_string(index=False))

            safe_name = model_name.replace(" ", "_").replace("=", "")
            plot_correlation_heatmap(
                corr_lab, comp_cols, LAB_MARKERS,
                f"{model_name} — NMF Components × Lab Markers",
                OUT_DIR / f"heatmap_lab_{safe_name}.png"
            )

        # Combined all markers
        corr_all = compute_correlations(df, comp_cols, ALL_MARKERS)

        # ── 2. Group-level correlations ──
        print(f"\n[2] Group-Level Correlations")
        group_corr = compute_group_correlations(df, comp_cols, ALL_MARKERS)
        if len(group_corr) > 0:
            sig_g = group_corr[group_corr["significant"]]
            print(f"  Total: {len(sig_g)}/{len(group_corr)} significant (FDR < 0.05)")
            if len(sig_g) > 0:
                print(sig_g[["group", "component", "marker_label", "r", "p_fdr", "n"]].head(15).to_string(index=False))

            safe_name = model_name.replace(" ", "_").replace("=", "")
            plot_group_heatmap(group_corr, OUT_DIR / f"group_barplot_{safe_name}.png")

        # ── 3. Key biological questions ──
        print(f"\n[3] Key Biological Questions")

        # Q1: Which component best predicts CA19-9? (PAN relevance)
        ca19_corr = corr_all[corr_all["marker"] == "ca19_9"].head(5)
        if len(ca19_corr) > 0:
            print(f"\n  Q1. Top components correlated with CA 19-9:")
            print(ca19_corr[["component", "r", "p_fdr", "n"]].to_string(index=False))

        # Q2: Which component best predicts CEA? (CRC/LUN relevance)
        cea_corr = corr_all[corr_all["marker"] == "cea"].head(5)
        if len(cea_corr) > 0:
            print(f"\n  Q2. Top components correlated with CEA:")
            print(cea_corr[["component", "r", "p_fdr", "n"]].to_string(index=False))

        # Q3: Creatinine spectral component vs blood creatinine?
        creat_corr = corr_all[corr_all["marker"] == "creatinine"].head(5)
        if len(creat_corr) > 0:
            print(f"\n  Q3. Top components correlated with blood Creatinine:")
            print(creat_corr[["component", "r", "p_fdr", "n"]].to_string(index=False))

    # ── 4. Summary CSV ──
    print(f"\n{'=' * 70}")
    print("Saving summary CSV...")
    # Rerun for both models and combine
    all_results = []
    for model_name, nmf_file in NMF_MODELS.items():
        df, comp_cols = load_nmf_with_clinical(nmf_file)
        corr = compute_correlations(df, comp_cols, ALL_MARKERS)
        corr["model"] = model_name
        all_results.append(corr)

    combined = pd.concat(all_results, ignore_index=True)
    csv_path = OUT_DIR / "nmf_marker_correlations.csv"
    combined.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # Summary stats
    sig_combined = combined[combined["significant"]]
    print(f"\n  Total tests: {len(combined)}")
    print(f"  Significant (FDR < 0.05): {len(sig_combined)}")
    if len(sig_combined) > 0:
        print(f"  Max |ρ|: {sig_combined['abs_r'].max():.3f}")
        print(f"  Top 5 overall:")
        print(sig_combined[["model", "component", "marker_label", "r", "p_fdr", "n"]].head(5).to_string(index=False))

    print(f"\nAll outputs saved to: {OUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
