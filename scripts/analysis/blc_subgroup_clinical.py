"""
BLC (Bladder Cancer) intra-group subgroup analysis with clinical data overlay.

Workflow:
  1. Load BLC spectra + clinical data, link by sample_id
  2. UMAP embedding of spectra
  3. HDBSCAN clustering to find spectral subgroups
  4. Overlay all available clinical variables on UMAP
  5. Statistical tests (Kruskal-Wallis / Fisher) between clusters
  6. Generate multi-panel publication figure

Usage:
    python scripts/analysis/blc_subgroup_clinical.py
"""

import re
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats

# Korean font
_KO_FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
if _KO_FONT.exists():
    fm.fontManager.addfont(str(_KO_FONT))
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

warnings.filterwarnings("ignore", category=FutureWarning)

# ═══════════════════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════════════════
ROOT = Path(__file__).resolve().parents[2]
SPECTRA_PATH = ROOT / "results" / "processed_spectra.csv"
CLINICAL_PATH = ROOT / "data" / "clinical_data" / "by_cancer" / "bladder.xlsx"
OUTPUT_DIR = ROOT / "results" / "blc_subgroup_analysis"

# ═══════════════════════════════════════════════════════════════════════════════
# Clinical variable definitions
# ═══════════════════════════════════════════════════════════════════════════════
NUMERIC_VARS = [
    "age", "bmi", "bp_systolic", "bp_diastolic",
    "wbc", "rbc", "hb", "hct", "platelet",
    "neutrophil_pct", "lymphocyte_pct",
    "ast", "alt", "alp", "ggt",
    "bun", "creatinine", "uric_acid", "glucose",
    "total_bilirubin", "calcium",
    "total_cholesterol", "triglyceride",
    "ua_sg", "ua_ph",
    "potassium", "chloride",
]

CATEGORICAL_VARS = [
    "sex", "smoking_status", "drinking_status",
    "t_stage", "sample_timing",
    "pathology_group",  # simplified pathology
    "ua_protein", "ua_blood", "ua_glucose",
]

# Display-friendly names
VAR_LABELS = {
    "age": "Age", "bmi": "BMI",
    "bp_systolic": "Systolic BP", "bp_diastolic": "Diastolic BP",
    "wbc": "WBC", "rbc": "RBC", "hb": "Hemoglobin", "hct": "Hematocrit",
    "platelet": "Platelet", "neutrophil_pct": "Neutrophil %",
    "lymphocyte_pct": "Lymphocyte %",
    "ast": "AST", "alt": "ALT", "alp": "ALP", "ggt": "GGT",
    "bun": "BUN", "creatinine": "Creatinine", "uric_acid": "Uric Acid",
    "glucose": "Glucose", "total_bilirubin": "Bilirubin",
    "calcium": "Calcium", "total_cholesterol": "Cholesterol",
    "triglyceride": "Triglyceride",
    "ua_sg": "Urine SG", "ua_ph": "Urine pH",
    "potassium": "K", "chloride": "Cl",
    "sex": "Sex", "smoking_status": "Smoking",
    "drinking_status": "Drinking", "t_stage": "T-Stage",
    "sample_timing": "Sample Timing",
    "pathology_group": "Pathology", "ua_protein": "Urine Protein",
    "ua_blood": "Urine Blood", "ua_glucose": "Urine Glucose",
}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Data loading & linkage
# ═══════════════════════════════════════════════════════════════════════════════
def load_and_link():
    """Load BLC spectra and clinical data, return merged DataFrame."""
    # Spectra
    spec = pd.read_csv(SPECTRA_PATH)
    blc = spec[spec["group"] == "BLC"].copy()
    feat_cols = [c for c in blc.columns if c.startswith("x_")]
    print(f"BLC spectra: {len(blc)} rows, {blc['sample_id'].nunique()} subjects")

    # Aggregate to subject-level (mean of replicates)
    blc_agg = blc.groupby("sample_id")[feat_cols].mean().reset_index()
    print(f"After mean aggregation: {len(blc_agg)} subjects")

    # Clinical
    clin = pd.read_excel(CLINICAL_PATH, sheet_name="Clean", engine="openpyxl")
    # Extract numeric ID from patient_id pattern "1기~2기-{N}" or "3기~4기-{N}"
    clin["sample_id"] = clin["patient_id"].astype(str).apply(
        lambda x: int(m.group(1)) if (m := re.search(r"-(\d+)$", x)) else None
    )
    clin = clin.dropna(subset=["sample_id"])
    clin["sample_id"] = clin["sample_id"].astype(int)

    # Simplify pathology
    clin["pathology_group"] = clin["pathology"].apply(_simplify_pathology)

    # Simplify t_stage
    clin["t_stage"] = clin["t_stage"].apply(_simplify_t_stage)

    # Merge
    merged = blc_agg.merge(clin, on="sample_id", how="inner")
    print(f"Linked: {len(merged)} subjects (spectra ∩ clinical)")
    return merged, feat_cols


def _simplify_pathology(p):
    if pd.isna(p):
        return "Unknown"
    p = str(p).upper().strip().rstrip(",")
    if "NONINVASIVE" in p or "NON-INVASIVE" in p:
        return "Non-invasive Papillary UC"
    if "PAPILLARY" in p:
        return "Papillary UC"
    if "UROTHELIAL" in p:
        return "UC (non-papillary)"
    return "Other"


def _simplify_t_stage(t):
    if pd.isna(t):
        return "Unknown"
    t = str(t).upper().strip()
    if t.startswith("TA"):
        return "Ta"
    if t.startswith("T1"):
        return "T1"
    if t.startswith("T2"):
        return "T2"
    if t.startswith("T3") or t.startswith("T4"):
        return "T3/T4"
    return t


# ═══════════════════════════════════════════════════════════════════════════════
# 2. UMAP embedding + HDBSCAN clustering
# ═══════════════════════════════════════════════════════════════════════════════
def embed_and_cluster(df, feat_cols, min_cluster_size=15):
    """UMAP 2D embedding + HDBSCAN clustering."""
    from umap import UMAP
    from hdbscan import HDBSCAN

    X = df[feat_cols].values

    # UMAP
    reducer = UMAP(n_components=2, n_neighbors=20, min_dist=0.1,
                   metric="euclidean", random_state=42)
    emb = reducer.fit_transform(X)
    df["umap_1"] = emb[:, 0]
    df["umap_2"] = emb[:, 1]

    # HDBSCAN
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size,
                        min_samples=5, metric="euclidean")
    labels = clusterer.fit_predict(emb)
    df["cluster"] = labels
    n_clusters = len(set(labels) - {-1})
    n_noise = (labels == -1).sum()
    print(f"HDBSCAN: {n_clusters} clusters found, {n_noise} noise points")
    for c in sorted(set(labels)):
        tag = "noise" if c == -1 else f"C{c}"
        print(f"  {tag}: {(labels == c).sum()} subjects")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Statistical tests
# ═══════════════════════════════════════════════════════════════════════════════
def run_statistical_tests(df):
    """Compare clusters on all clinical variables. Returns results DataFrame."""
    # Exclude noise for clean comparisons
    df_clean = df[df["cluster"] >= 0].copy()
    clusters = sorted(df_clean["cluster"].unique())
    if len(clusters) < 2:
        print("WARNING: fewer than 2 clusters, skipping statistical tests")
        return pd.DataFrame()

    results = []

    # Numeric: Kruskal-Wallis
    for var in NUMERIC_VARS:
        if var not in df_clean.columns:
            continue
        groups = [df_clean.loc[df_clean["cluster"] == c, var].dropna() for c in clusters]
        groups = [g for g in groups if len(g) >= 3]
        if len(groups) < 2:
            continue
        stat, pval = stats.kruskal(*groups)
        # Effect size: eta-squared
        n_total = sum(len(g) for g in groups)
        eta_sq = (stat - len(groups) + 1) / (n_total - len(groups)) if n_total > len(groups) else 0
        medians = {f"C{c}": df_clean.loc[df_clean["cluster"] == c, var].median()
                   for c in clusters}
        results.append({
            "variable": var,
            "label": VAR_LABELS.get(var, var),
            "type": "numeric",
            "test": "Kruskal-Wallis",
            "statistic": stat,
            "p_value": pval,
            "eta_squared": max(eta_sq, 0),
            **medians,
        })

    # Categorical: Fisher exact (2×k) or Chi-squared
    for var in CATEGORICAL_VARS:
        if var not in df_clean.columns:
            continue
        valid = df_clean[[var, "cluster"]].dropna()
        if len(valid) < 10:
            continue
        ct = pd.crosstab(valid[var], valid["cluster"])
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            continue
        stat, pval, dof, expected = stats.chi2_contingency(ct)
        # Cramér's V
        n = ct.sum().sum()
        k = min(ct.shape) - 1
        cramers_v = np.sqrt(stat / (n * k)) if n * k > 0 else 0
        # Mode per cluster
        modes = {f"C{c}": valid.loc[valid["cluster"] == c, var].mode().iloc[0]
                 if len(valid.loc[valid["cluster"] == c, var]) > 0 else "NA"
                 for c in clusters}
        results.append({
            "variable": var,
            "label": VAR_LABELS.get(var, var),
            "type": "categorical",
            "test": "Chi-squared",
            "statistic": stat,
            "p_value": pval,
            "cramers_v": cramers_v,
            **modes,
        })

    result_df = pd.DataFrame(results)
    if len(result_df) > 0:
        result_df["p_adj"] = _bh_correction(result_df["p_value"].values)
        result_df = result_df.sort_values("p_value")
    return result_df


def _bh_correction(pvals):
    """Benjamini-Hochberg FDR correction."""
    n = len(pvals)
    ranked = np.argsort(pvals)
    adjusted = np.zeros(n)
    for i, idx in enumerate(ranked[::-1]):
        if i == 0:
            adjusted[idx] = pvals[idx]
        else:
            adjusted[idx] = min(pvals[idx] * n / (n - i), adjusted[ranked[n - i]])
    return np.minimum(adjusted, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Visualization
# ═══════════════════════════════════════════════════════════════════════════════
CLUSTER_COLORS = ["#E53935", "#1E88E5", "#43A047", "#FB8C00", "#8E24AA", "#00ACC1"]
NOISE_COLOR = "#BDBDBD"


def make_figure(df, stat_results, output_dir):
    """Generate multi-panel figure: UMAP + clinical overlays + stats table."""
    output_dir.mkdir(parents=True, exist_ok=True)

    clusters = sorted(df["cluster"].unique())
    real_clusters = [c for c in clusters if c >= 0]
    cmap = {c: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, c in enumerate(real_clusters)}
    cmap[-1] = NOISE_COLOR
    colors = [cmap[c] for c in df["cluster"]]

    # --- Figure 1: Main UMAP with cluster labels ---
    fig1, ax = plt.subplots(figsize=(8, 7))
    for c in clusters:
        mask = df["cluster"] == c
        label = f"C{c} (n={mask.sum()})" if c >= 0 else f"Noise (n={mask.sum()})"
        ax.scatter(df.loc[mask, "umap_1"], df.loc[mask, "umap_2"],
                   c=cmap[c], label=label, s=30, alpha=0.7, edgecolors="white", linewidth=0.3)
    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title("BLC Spectral Subgroups (UMAP + HDBSCAN)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="best")
    ax.set_aspect("equal")
    fig1.tight_layout()
    fig1.savefig(output_dir / "01_umap_clusters.png", dpi=200, bbox_inches="tight")
    plt.close(fig1)

    # --- Figure 2: Top significant numeric variables overlaid on UMAP ---
    sig_numeric = stat_results[
        (stat_results["type"] == "numeric") & (stat_results["p_adj"] < 0.05)
    ].head(9)

    if len(sig_numeric) > 0:
        n_panels = len(sig_numeric)
        ncols = min(3, n_panels)
        nrows = (n_panels + ncols - 1) // ncols
        fig2, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
        axes = np.atleast_2d(axes)
        for i, (_, row) in enumerate(sig_numeric.iterrows()):
            ax = axes[i // ncols, i % ncols]
            var = row["variable"]
            vals = df[var].values
            valid_mask = ~np.isnan(vals)
            sc = ax.scatter(
                df.loc[valid_mask, "umap_1"], df.loc[valid_mask, "umap_2"],
                c=vals[valid_mask], cmap="RdYlBu_r", s=20, alpha=0.7,
                edgecolors="white", linewidth=0.2,
            )
            plt.colorbar(sc, ax=ax, shrink=0.8)
            pstr = f"p={row['p_adj']:.1e}" if row["p_adj"] < 0.001 else f"p={row['p_adj']:.3f}"
            ax.set_title(f"{row['label']} ({pstr})", fontsize=11, fontweight="bold")
            ax.set_xlabel("UMAP 1", fontsize=9)
            ax.set_ylabel("UMAP 2", fontsize=9)
        # Hide unused axes
        for i in range(n_panels, nrows * ncols):
            axes[i // ncols, i % ncols].set_visible(False)
        fig2.suptitle("Significant Numeric Variables on UMAP (BH-adjusted p < 0.05)",
                      fontsize=13, fontweight="bold", y=1.02)
        fig2.tight_layout()
        fig2.savefig(output_dir / "02_umap_numeric_overlay.png", dpi=200, bbox_inches="tight")
        plt.close(fig2)

    # --- Figure 3: Categorical overlays ---
    sig_cat = stat_results[
        (stat_results["type"] == "categorical") & (stat_results["p_adj"] < 0.1)
    ]
    # Always show key categoricals even if not significant
    key_cats = ["t_stage", "sample_timing", "sex", "pathology_group", "ua_blood", "ua_protein"]
    cat_to_plot = list(dict.fromkeys(
        list(sig_cat["variable"]) + [v for v in key_cats if v in df.columns]
    ))[:6]

    if cat_to_plot:
        ncols = min(3, len(cat_to_plot))
        nrows = (len(cat_to_plot) + ncols - 1) // ncols
        fig3, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
        axes = np.atleast_2d(axes)
        for i, var in enumerate(cat_to_plot):
            ax = axes[i // ncols, i % ncols]
            categories = df[var].dropna().unique()
            cat_colors = plt.cm.Set2(np.linspace(0, 1, len(categories)))
            for j, cat in enumerate(sorted(categories, key=str)):
                mask = df[var] == cat
                ax.scatter(df.loc[mask, "umap_1"], df.loc[mask, "umap_2"],
                           c=[cat_colors[j]], label=str(cat), s=20, alpha=0.6,
                           edgecolors="white", linewidth=0.2)
            ax.legend(fontsize=7, loc="best", markerscale=1.5)
            ax.set_title(VAR_LABELS.get(var, var), fontsize=11, fontweight="bold")
            ax.set_xlabel("UMAP 1", fontsize=9)
            ax.set_ylabel("UMAP 2", fontsize=9)
        for i in range(len(cat_to_plot), nrows * ncols):
            axes[i // ncols, i % ncols].set_visible(False)
        fig3.suptitle("Categorical Variables on UMAP", fontsize=13, fontweight="bold", y=1.02)
        fig3.tight_layout()
        fig3.savefig(output_dir / "03_umap_categorical_overlay.png", dpi=200, bbox_inches="tight")
        plt.close(fig3)

    # --- Figure 4: Boxplots of top variables by cluster ---
    if len(stat_results) > 0:
        top_vars = stat_results[stat_results["type"] == "numeric"].head(8)
        if len(top_vars) > 0:
            df_clean = df[df["cluster"] >= 0].copy()
            ncols = min(4, len(top_vars))
            nrows = (len(top_vars) + ncols - 1) // ncols
            fig4, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
            axes = np.atleast_1d(axes).flatten()
            for i, (_, row) in enumerate(top_vars.iterrows()):
                ax = axes[i]
                var = row["variable"]
                data_by_cluster = [
                    df_clean.loc[df_clean["cluster"] == c, var].dropna()
                    for c in real_clusters
                ]
                bp = ax.boxplot(data_by_cluster, labels=[f"C{c}" for c in real_clusters],
                                patch_artist=True, widths=0.6)
                for j, patch in enumerate(bp["boxes"]):
                    patch.set_facecolor(CLUSTER_COLORS[j % len(CLUSTER_COLORS)])
                    patch.set_alpha(0.6)
                pstr = f"p={row['p_adj']:.1e}" if row["p_adj"] < 0.001 else f"p={row['p_adj']:.3f}"
                ax.set_title(f"{row['label']}\n({pstr})", fontsize=10)
                ax.set_ylabel(row["label"], fontsize=9)
            for i in range(len(top_vars), len(axes)):
                axes[i].set_visible(False)
            fig4.suptitle("Cluster Comparison: Top Variables (Kruskal-Wallis)",
                          fontsize=13, fontweight="bold")
            fig4.tight_layout()
            fig4.savefig(output_dir / "04_cluster_boxplots.png", dpi=200, bbox_inches="tight")
            plt.close(fig4)

    # --- Figure 5: Cluster composition stacked bars ---
    df_clean = df[df["cluster"] >= 0].copy()
    if len(real_clusters) >= 2:
        comp_vars = ["t_stage", "sample_timing", "sex", "pathology_group"]
        comp_vars = [v for v in comp_vars if v in df_clean.columns]
        if comp_vars:
            fig5, axes = plt.subplots(1, len(comp_vars), figsize=(5 * len(comp_vars), 4))
            if len(comp_vars) == 1:
                axes = [axes]
            for ax, var in zip(axes, comp_vars):
                ct = pd.crosstab(df_clean["cluster"], df_clean[var], normalize="index")
                ct.index = [f"C{c}" for c in ct.index]
                ct.plot(kind="bar", stacked=True, ax=ax, colormap="Set2", edgecolor="white")
                ax.set_title(VAR_LABELS.get(var, var), fontsize=11, fontweight="bold")
                ax.set_ylabel("Proportion", fontsize=10)
                ax.set_xlabel("Cluster", fontsize=10)
                ax.legend(fontsize=7, bbox_to_anchor=(1.0, 1.0))
                ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            fig5.suptitle("Cluster Composition by Clinical Variables",
                          fontsize=13, fontweight="bold")
            fig5.tight_layout()
            fig5.savefig(output_dir / "05_cluster_composition.png", dpi=200, bbox_inches="tight")
            plt.close(fig5)

    print(f"\nFigures saved to {output_dir}/")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("BLC Subgroup Analysis with Clinical Data")
    print("=" * 60)

    # 1. Load & link
    df, feat_cols = load_and_link()

    # 2. Embed & cluster
    df = embed_and_cluster(df, feat_cols)

    # 3. Statistical tests
    stat_results = run_statistical_tests(df)

    # 4. Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if len(stat_results) > 0:
        stat_results.to_csv(OUTPUT_DIR / "statistical_tests.csv", index=False)
        print(f"\n{'='*60}")
        print("SIGNIFICANT RESULTS (BH-adjusted p < 0.05):")
        print(f"{'='*60}")
        sig = stat_results[stat_results["p_adj"] < 0.05]
        if len(sig) > 0:
            for _, row in sig.iterrows():
                direction = ""
                if row["type"] == "numeric":
                    cluster_cols = [c for c in row.index if c.startswith("C")]
                    if cluster_cols:
                        vals = {c: row[c] for c in cluster_cols if pd.notna(row[c])}
                        direction = " | ".join(f"{k}={v:.1f}" for k, v in vals.items())
                print(f"  {row['label']:20s}  p_adj={row['p_adj']:.4f}  {direction}")
        else:
            print("  (no variables reached significance after FDR correction)")

        print(f"\nAll p < 0.10:")
        marginal = stat_results[stat_results["p_adj"] < 0.10]
        for _, row in marginal.iterrows():
            print(f"  {row['label']:20s}  p_adj={row['p_adj']:.4f}  [{row['test']}]")

    # 5. Visualize
    make_figure(df, stat_results, OUTPUT_DIR)

    # 6. Save cluster assignments
    out_cols = ["sample_id", "cluster", "umap_1", "umap_2"] + \
               [v for v in NUMERIC_VARS + CATEGORICAL_VARS if v in df.columns]
    df[out_cols].to_csv(OUTPUT_DIR / "blc_cluster_assignments.csv", index=False)
    print(f"\nCluster assignments saved to {OUTPUT_DIR / 'blc_cluster_assignments.csv'}")


if __name__ == "__main__":
    main()
