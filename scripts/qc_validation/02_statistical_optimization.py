"""
Phase 2: Statistical Optimization of QC Thresholds

Performs comprehensive statistical analysis:
  2-1. Fine-grained threshold sweep (620 combinations)
  2-2. Sensitivity analysis (gradient)
  2-3. RSD-Correlation independence analysis
  2-4. Per-cancer pass rate analysis
  2-5. Utility curve (Pareto front)
  2-6. Bootstrap CI for optimal threshold
  2-7. Replicate convergence analysis

Usage:
    cd /home/user/SERS-AI
    PYTHONPATH=. python scripts/qc_validation/02_statistical_optimization.py
"""

import sys
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_qc_threshold_experiment import (
    aggregate_medoid, run_cv, get_feature_cols, load_data
)
from models.model import ModelConfig


def filter_by_qc_fast(df, qc, rsd_thresh, corr_thresh, _merge_cache={}):
    """Optimized filter_by_qc using merge instead of apply."""
    if rsd_thresh == float("inf") and corr_thresh == 0.0:
        return df.copy()

    passed = qc.copy()
    if rsd_thresh < float("inf"):
        passed = passed[passed["mean_rsd"] <= rsd_thresh]
    if corr_thresh > 0.0:
        passed = passed[passed["mean_corr"] >= corr_thresh]

    passed_keys = passed[["group", "sample_id"]].copy()
    passed_keys["sample_id"] = passed_keys["sample_id"].astype(str)

    # Build merge key on df if not cached
    cache_key = id(df)
    if cache_key not in _merge_cache:
        df_keys = df[["group", "sample_id"]].copy()
        df_keys["sample_id"] = df_keys["sample_id"].astype(str)
        _merge_cache[cache_key] = df_keys

    df_keys = _merge_cache[cache_key]
    mask = pd.merge(df_keys.reset_index(), passed_keys, on=["group", "sample_id"], how="inner")["index"]
    return df.loc[mask].copy()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

OUT_DIR = PROJECT_ROOT / "results" / "qc_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

QC_STATS_PATH = PROJECT_ROOT / "results" / "fixed_grid" / "qc_stats.csv"

# Current thresholds
CURRENT_RSD = 5.0
CURRENT_CORR = 0.95


# =============================================================================
# 2-1. Fine-Grained Threshold Sweep
# =============================================================================
def run_fine_sweep(df, qc, config, feature_cols):
    """620-combination sweep: RSD [2..20,inf] x Corr [0.70..0.99,0.0]"""
    logger.info("=" * 60)
    logger.info("  2-1. Fine-Grained Threshold Sweep")
    logger.info("=" * 60)

    rsd_vals = list(range(2, 21)) + [float("inf")]  # 20 values
    corr_vals = [round(0.70 + i * 0.02, 2) for i in range(15)] + [0.0]  # 16 values (0.02 step)

    results = []
    total = len(rsd_vals) * len(corr_vals)

    for i, rsd_t in enumerate(rsd_vals):
        for j, corr_t in enumerate(corr_vals):
            idx = i * len(corr_vals) + j + 1
            if idx % 50 == 0:
                logger.info(f"  [{idx}/{total}] RSD={rsd_t}, Corr={corr_t}")

            df_f = filter_by_qc_fast(df, qc, rsd_t, corr_t)
            n_samples = df_f.groupby(["group", "sample_id"]).ngroups
            n_spectra = len(df_f)

            row = {
                "rsd_threshold": rsd_t,
                "corr_threshold": corr_t,
                "n_samples": n_samples,
                "n_spectra": n_spectra,
            }

            if n_samples < 20:
                row.update({"auc_s1_mean": np.nan, "auc_s1_std": np.nan,
                            "f1_s2_mean": np.nan, "f1_s2_std": np.nan})
                results.append(row)
                continue

            df_agg = aggregate_medoid(df_f, feature_cols)
            cv_result = run_cv(df_agg, config, feature_cols)

            if cv_result is None:
                row.update({"auc_s1_mean": np.nan, "auc_s1_std": np.nan,
                            "f1_s2_mean": np.nan, "f1_s2_std": np.nan})
            else:
                auc_arr = np.array([x for x in cv_result["auc_s1_folds"] if not np.isnan(x)])
                f1_arr = np.array([x for x in cv_result["f1_s2_folds"] if not np.isnan(x)])
                row["auc_s1_mean"] = float(auc_arr.mean()) if len(auc_arr) else np.nan
                row["auc_s1_std"] = float(auc_arr.std()) if len(auc_arr) else np.nan
                row["f1_s2_mean"] = float(f1_arr.mean()) if len(f1_arr) else np.nan
                row["f1_s2_std"] = float(f1_arr.std()) if len(f1_arr) else np.nan

            results.append(row)

    df_results = pd.DataFrame(results)
    csv_path = OUT_DIR / "fine_sweep_results.csv"
    df_results.to_csv(csv_path, index=False)
    logger.info(f"Saved: {csv_path} ({len(results)} combinations)")

    # Plot heatmaps
    plot_heatmaps(df_results)

    return df_results


def plot_heatmaps(df_sweep):
    """Generate AUC and F1 heatmaps from sweep results."""
    # Filter out inf and 0.0 for clean heatmap
    df_plot = df_sweep[
        (df_sweep["rsd_threshold"] != float("inf")) &
        (df_sweep["corr_threshold"] != 0.0)
    ].copy()

    rsd_vals = sorted(df_plot["rsd_threshold"].unique())
    corr_vals = sorted(df_plot["corr_threshold"].unique())

    for metric, label in [("auc_s1_mean", "AUC (Cancer Screening)"),
                          ("f1_s2_mean", "F1 (Cancer Type ID)"),
                          ("n_samples", "Sample Count")]:
        fig, ax = plt.subplots(figsize=(14, 6))

        pivot = df_plot.pivot(index="rsd_threshold", columns="corr_threshold", values=metric)
        pivot = pivot.reindex(index=rsd_vals, columns=corr_vals)

        cmap = "YlOrRd" if metric == "n_samples" else "RdYlGn"
        im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, origin="lower")
        plt.colorbar(im, ax=ax, label=label)

        ax.set_xticks(range(len(corr_vals)))
        ax.set_xticklabels([f"{v:.2f}" for v in corr_vals], rotation=45, fontsize=7)
        ax.set_yticks(range(len(rsd_vals)))
        ax.set_yticklabels([f"{v}" for v in rsd_vals], fontsize=8)
        ax.set_xlabel("Correlation Threshold")
        ax.set_ylabel("RSD Threshold (%)")
        ax.set_title(f"QC Threshold Sweep — {label}")

        # Mark current threshold
        if CURRENT_RSD in rsd_vals and CURRENT_CORR in corr_vals:
            ri = rsd_vals.index(CURRENT_RSD)
            ci = corr_vals.index(CURRENT_CORR)
            ax.plot(ci, ri, "k*", markersize=15, markeredgewidth=1.5)
            ax.annotate("Current", (ci, ri), textcoords="offset points",
                        xytext=(10, 10), fontsize=9, fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color="black"))

        plt.tight_layout()
        fname = f"heatmap_{metric}.png"
        fig.savefig(OUT_DIR / fname, dpi=150)
        plt.close(fig)
        logger.info(f"  Saved: {fname}")


# =============================================================================
# 2-2. Sensitivity Analysis
# =============================================================================
def run_sensitivity_analysis(df_sweep):
    """Compute numerical gradient of performance w.r.t. threshold changes."""
    logger.info("\n" + "=" * 60)
    logger.info("  2-2. Sensitivity Analysis")
    logger.info("=" * 60)

    df_plot = df_sweep[
        (df_sweep["rsd_threshold"] != float("inf")) &
        (df_sweep["corr_threshold"] != 0.0)
    ].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1D sensitivity: Fix one, vary the other
    for ax, fix_name, fix_val, vary_name in [
        (axes[0], "corr_threshold", CURRENT_CORR, "rsd_threshold"),
        (axes[1], "rsd_threshold", CURRENT_RSD, "corr_threshold"),
    ]:
        subset = df_plot[df_plot[fix_name] == fix_val].sort_values(vary_name)
        if len(subset) < 3:
            continue

        x = subset[vary_name].values
        for metric, color, label in [
            ("auc_s1_mean", "blue", "AUC (Screening)"),
            ("f1_s2_mean", "red", "F1 (Type ID)"),
        ]:
            y = subset[metric].values
            ax.plot(x, y, f"-o", color=color, label=label, markersize=4)

        # Mark current
        current_val = CURRENT_RSD if vary_name == "rsd_threshold" else CURRENT_CORR
        ax.axvline(current_val, color="gray", linestyle="--", alpha=0.7, label="Current")

        # Secondary axis for sample count
        ax2 = ax.twinx()
        ax2.fill_between(x, subset["n_samples"].values, alpha=0.1, color="green")
        ax2.set_ylabel("Sample Count", color="green", fontsize=9)

        ax.set_xlabel(vary_name.replace("_", " ").title())
        ax.set_ylabel("Performance")
        fix_label = f"{fix_name}={fix_val}"
        ax.set_title(f"Sensitivity (fixed {fix_label})")
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "sensitivity_analysis.png", dpi=150)
    plt.close(fig)
    logger.info("  Saved: sensitivity_analysis.png")


# =============================================================================
# 2-3. RSD-Correlation Independence
# =============================================================================
def run_independence_analysis(qc):
    """Analyze whether RSD and Correlation metrics are independent."""
    logger.info("\n" + "=" * 60)
    logger.info("  2-3. RSD-Correlation Independence")
    logger.info("=" * 60)

    rsd = qc["mean_rsd"].values
    corr = qc["mean_corr"].values

    from scipy.stats import pearsonr, spearmanr
    pr, pp = pearsonr(rsd, corr)
    sr, sp = spearmanr(rsd, corr)

    logger.info(f"  Pearson r = {pr:.4f} (p = {pp:.2e})")
    logger.info(f"  Spearman rho = {sr:.4f} (p = {sp:.2e})")

    # Failure mode Venn diagram data
    rsd_fail = rsd >= CURRENT_RSD
    corr_fail = corr < CURRENT_CORR
    both_fail = rsd_fail & corr_fail
    rsd_only = rsd_fail & ~corr_fail
    corr_only = ~rsd_fail & corr_fail
    neither = ~rsd_fail & ~corr_fail

    n = len(rsd)
    venn_data = {
        "total": n,
        "pass_both": int(neither.sum()),
        "fail_rsd_only": int(rsd_only.sum()),
        "fail_corr_only": int(corr_only.sum()),
        "fail_both": int(both_fail.sum()),
        "pearson_r": round(pr, 4),
        "spearman_rho": round(sr, 4),
    }

    logger.info(f"  Pass both: {venn_data['pass_both']} ({venn_data['pass_both']/n*100:.1f}%)")
    logger.info(f"  Fail RSD only: {venn_data['fail_rsd_only']} ({venn_data['fail_rsd_only']/n*100:.1f}%)")
    logger.info(f"  Fail Corr only: {venn_data['fail_corr_only']} ({venn_data['fail_corr_only']/n*100:.1f}%)")
    logger.info(f"  Fail both: {venn_data['fail_both']} ({venn_data['fail_both']/n*100:.1f}%)")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter
    ax = axes[0]
    colors = np.where(neither, "green", np.where(both_fail, "red",
                      np.where(rsd_only, "orange", "purple")))
    ax.scatter(rsd, corr, c=colors, alpha=0.4, s=10)
    ax.axvline(CURRENT_RSD, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(CURRENT_CORR, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Mean RSD (%)")
    ax.set_ylabel("Mean Correlation")
    ax.set_title(f"RSD vs Correlation (Pearson r={pr:.3f})")

    # Legend
    from matplotlib.patches import Patch
    legend_items = [
        Patch(color="green", label=f"Pass both ({venn_data['pass_both']})"),
        Patch(color="orange", label=f"Fail RSD only ({venn_data['fail_rsd_only']})"),
        Patch(color="purple", label=f"Fail Corr only ({venn_data['fail_corr_only']})"),
        Patch(color="red", label=f"Fail both ({venn_data['fail_both']})"),
    ]
    ax.legend(handles=legend_items, fontsize=8)

    # Bar chart for Venn data
    ax = axes[1]
    categories = ["Pass Both", "Fail RSD\nOnly", "Fail Corr\nOnly", "Fail Both"]
    counts = [venn_data["pass_both"], venn_data["fail_rsd_only"],
              venn_data["fail_corr_only"], venn_data["fail_both"]]
    pcts = [c / n * 100 for c in counts]
    bars = ax.bar(categories, pcts, color=["green", "orange", "purple", "red"], alpha=0.7)
    for bar, pct, cnt in zip(bars, pcts, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{cnt}\n({pct:.1f}%)", ha="center", fontsize=9)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("QC Failure Mode Distribution")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "rsd_corr_independence.png", dpi=150)
    plt.close(fig)
    logger.info("  Saved: rsd_corr_independence.png")

    return venn_data


# =============================================================================
# 2-4. Per-Cancer Pass Rate
# =============================================================================
def run_per_cancer_analysis(qc):
    """Analyze QC pass rates per cancer group at various thresholds."""
    logger.info("\n" + "=" * 60)
    logger.info("  2-4. Per-Cancer Pass Rate Analysis")
    logger.info("=" * 60)

    groups = qc["group"].unique()
    rsd_thresholds = [3, 5, 7, 10, 15, float("inf")]
    corr_thresholds = [0.99, 0.95, 0.90, 0.85, 0.0]

    # RSD sweep (fix corr at current)
    rows_rsd = []
    for grp in sorted(groups):
        grp_data = qc[qc["group"] == grp]
        for rsd_t in rsd_thresholds:
            passed = grp_data[
                (grp_data["mean_rsd"] <= rsd_t) & (grp_data["mean_corr"] >= CURRENT_CORR)
            ]
            rows_rsd.append({
                "group": grp,
                "rsd_threshold": rsd_t,
                "n_total": len(grp_data),
                "n_pass": len(passed),
                "pass_rate": len(passed) / len(grp_data) * 100 if len(grp_data) > 0 else 0,
            })

    # Corr sweep (fix rsd at current)
    rows_corr = []
    for grp in sorted(groups):
        grp_data = qc[qc["group"] == grp]
        for corr_t in corr_thresholds:
            passed = grp_data[
                (grp_data["mean_rsd"] <= CURRENT_RSD) & (grp_data["mean_corr"] >= corr_t)
            ]
            rows_corr.append({
                "group": grp,
                "corr_threshold": corr_t,
                "n_total": len(grp_data),
                "n_pass": len(passed),
                "pass_rate": len(passed) / len(grp_data) * 100 if len(grp_data) > 0 else 0,
            })

    df_rsd = pd.DataFrame(rows_rsd)
    df_corr = pd.DataFrame(rows_corr)

    # Save
    df_rsd.to_csv(OUT_DIR / "per_cancer_passrate_rsd.csv", index=False)
    df_corr.to_csv(OUT_DIR / "per_cancer_passrate_corr.csv", index=False)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, df_pr, sweep_col, title in [
        (axes[0], df_rsd, "rsd_threshold", "Pass Rate by RSD Threshold (Corr=0.95)"),
        (axes[1], df_corr, "corr_threshold", "Pass Rate by Corr Threshold (RSD=5%)"),
    ]:
        pivot = df_pr.pivot(index="group", columns=sweep_col, values="pass_rate")
        pivot.plot(kind="bar", ax=ax, width=0.8)
        ax.set_ylabel("Pass Rate (%)")
        ax.set_title(title)
        ax.set_xlabel("")
        ax.legend(title=sweep_col, fontsize=7, bbox_to_anchor=(1.0, 1.0))
        ax.set_ylim(0, 105)
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "per_cancer_passrate.png", dpi=150)
    plt.close(fig)
    logger.info("  Saved: per_cancer_passrate.png")

    # Summary at current thresholds
    current = df_rsd[df_rsd["rsd_threshold"] == CURRENT_RSD][["group", "n_total", "n_pass", "pass_rate"]]
    logger.info("\n  Current threshold pass rates:")
    for _, row in current.sort_values("pass_rate").iterrows():
        logger.info(f"    {row['group']:>5}: {row['n_pass']:>4}/{row['n_total']:<4} ({row['pass_rate']:.1f}%)")

    return df_rsd, df_corr


# =============================================================================
# 2-5. Utility Curve (Pareto Front)
# =============================================================================
def run_pareto_analysis(df_sweep):
    """Plot sample retention vs performance and find Pareto front."""
    logger.info("\n" + "=" * 60)
    logger.info("  2-5. Utility Curve (Pareto Front)")
    logger.info("=" * 60)

    df = df_sweep.dropna(subset=["auc_s1_mean", "f1_s2_mean"]).copy()
    total_samples = df["n_samples"].max()
    df["retention_pct"] = df["n_samples"] / total_samples * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, label in [
        (axes[0], "auc_s1_mean", "AUC (Cancer Screening)"),
        (axes[1], "f1_s2_mean", "F1 (Cancer Type ID)"),
    ]:
        ax.scatter(df["retention_pct"], df[metric], alpha=0.3, s=15, c="gray")

        # Pareto front
        sorted_df = df.sort_values("retention_pct")
        pareto_x, pareto_y = [], []
        best_perf = -1
        for _, row in sorted_df.iterrows():
            if row[metric] > best_perf:
                best_perf = row[metric]
                pareto_x.append(row["retention_pct"])
                pareto_y.append(row[metric])
        ax.plot(pareto_x, pareto_y, "r-o", markersize=5, label="Pareto front")

        # Mark current operating point
        current = df[
            (df["rsd_threshold"] == CURRENT_RSD) & (df["corr_threshold"] == CURRENT_CORR)
        ]
        if len(current) > 0:
            cx = current.iloc[0]["retention_pct"]
            cy = current.iloc[0][metric]
            ax.plot(cx, cy, "k*", markersize=15)
            ax.annotate(f"Current\n({cx:.0f}%, {cy:.3f})", (cx, cy),
                        textcoords="offset points", xytext=(15, -15), fontsize=9,
                        arrowprops=dict(arrowstyle="->"))

        ax.set_xlabel("Sample Retention (%)")
        ax.set_ylabel(label)
        ax.set_title(f"Retention vs {label}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "utility_pareto.png", dpi=150)
    plt.close(fig)
    logger.info("  Saved: utility_pareto.png")


# =============================================================================
# 2-6. Bootstrap CI for Optimal Threshold
# =============================================================================
def run_bootstrap_ci(df, qc, config, feature_cols, n_boot=200):
    """Bootstrap to estimate CI of optimal threshold."""
    logger.info("\n" + "=" * 60)
    logger.info(f"  2-6. Bootstrap CI ({n_boot} resamples)")
    logger.info("=" * 60)

    # Coarse grid for speed
    rsd_grid = [3, 5, 7, 10, 15]
    corr_grid = [0.80, 0.85, 0.90, 0.95]

    sample_ids = qc[["group", "sample_id"]].drop_duplicates()
    optimal_rsd_auc, optimal_corr_auc = [], []
    optimal_rsd_f1, optimal_corr_f1 = [], []

    for b in range(n_boot):
        if (b + 1) % 50 == 0:
            logger.info(f"  Bootstrap {b+1}/{n_boot}")

        # Resample subjects with replacement
        boot_ids = sample_ids.sample(n=len(sample_ids), replace=True)
        boot_keys = set(zip(boot_ids["group"], boot_ids["sample_id"].astype(str)))

        mask = df.apply(lambda r: (r["group"], str(r["sample_id"])) in boot_keys, axis=1)
        df_boot = df[mask].copy()
        qc_mask = qc.apply(lambda r: (r["group"], str(r["sample_id"])) in boot_keys, axis=1)
        qc_boot = qc[qc_mask].copy()

        best_auc, best_f1 = -1, -1
        best_auc_params, best_f1_params = (5, 0.95), (5, 0.95)

        for rsd_t in rsd_grid:
            for corr_t in corr_grid:
                df_f = filter_by_qc_fast(df_boot, qc_boot, rsd_t, corr_t)
                n_samp = df_f.groupby(["group", "sample_id"]).ngroups
                if n_samp < 20:
                    continue

                df_agg = aggregate_medoid(df_f, feature_cols)
                cv_res = run_cv(df_agg, config, feature_cols)
                if cv_res is None:
                    continue

                auc_arr = np.array([x for x in cv_res["auc_s1_folds"] if not np.isnan(x)])
                f1_arr = np.array([x for x in cv_res["f1_s2_folds"] if not np.isnan(x)])

                if len(auc_arr) > 0 and auc_arr.mean() > best_auc:
                    best_auc = auc_arr.mean()
                    best_auc_params = (rsd_t, corr_t)
                if len(f1_arr) > 0 and f1_arr.mean() > best_f1:
                    best_f1 = f1_arr.mean()
                    best_f1_params = (rsd_t, corr_t)

        optimal_rsd_auc.append(best_auc_params[0])
        optimal_corr_auc.append(best_auc_params[1])
        optimal_rsd_f1.append(best_f1_params[0])
        optimal_corr_f1.append(best_f1_params[1])

    # Results
    results = {
        "auc_optimal_rsd": {"mean": np.mean(optimal_rsd_auc), "std": np.std(optimal_rsd_auc),
                            "median": np.median(optimal_rsd_auc),
                            "ci_lower": np.percentile(optimal_rsd_auc, 2.5),
                            "ci_upper": np.percentile(optimal_rsd_auc, 97.5)},
        "auc_optimal_corr": {"mean": np.mean(optimal_corr_auc), "std": np.std(optimal_corr_auc),
                             "median": np.median(optimal_corr_auc),
                             "ci_lower": np.percentile(optimal_corr_auc, 2.5),
                             "ci_upper": np.percentile(optimal_corr_auc, 97.5)},
        "f1_optimal_rsd": {"mean": np.mean(optimal_rsd_f1), "std": np.std(optimal_rsd_f1),
                           "median": np.median(optimal_rsd_f1),
                           "ci_lower": np.percentile(optimal_rsd_f1, 2.5),
                           "ci_upper": np.percentile(optimal_rsd_f1, 97.5)},
        "f1_optimal_corr": {"mean": np.mean(optimal_corr_f1), "std": np.std(optimal_corr_f1),
                            "median": np.median(optimal_corr_f1),
                            "ci_lower": np.percentile(optimal_corr_f1, 2.5),
                            "ci_upper": np.percentile(optimal_corr_f1, 97.5)},
    }

    for key, vals in results.items():
        logger.info(f"  {key}: median={vals['median']:.2f}, 95%CI=[{vals['ci_lower']:.2f}, {vals['ci_upper']:.2f}]")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, rsd_list, corr_list, title in [
        (axes[0], optimal_rsd_auc, optimal_corr_auc, "Optimal for AUC"),
        (axes[1], optimal_rsd_f1, optimal_corr_f1, "Optimal for F1"),
    ]:
        ax.scatter(rsd_list, corr_list, alpha=0.3, s=20, c="blue")
        ax.plot(CURRENT_RSD, CURRENT_CORR, "r*", markersize=20, label="Current (5, 0.95)")
        ax.set_xlabel("Optimal RSD Threshold (%)")
        ax.set_ylabel("Optimal Corr Threshold")
        ax.set_title(f"Bootstrap Distribution — {title}")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "bootstrap_ci.png", dpi=150)
    plt.close(fig)
    logger.info("  Saved: bootstrap_ci.png")

    # Save raw data
    pd.DataFrame({
        "rsd_optimal_auc": optimal_rsd_auc,
        "corr_optimal_auc": optimal_corr_auc,
        "rsd_optimal_f1": optimal_rsd_f1,
        "corr_optimal_f1": optimal_corr_f1,
    }).to_csv(OUT_DIR / "bootstrap_raw.csv", index=False)

    return results


# =============================================================================
# 2-7. Replicate Convergence
# =============================================================================
def run_convergence_analysis(qc):
    """Analyze whether 5 replicates is sufficient using existing QC stats."""
    logger.info("\n" + "=" * 60)
    logger.info("  2-7. Replicate Convergence Analysis")
    logger.info("=" * 60)

    # Use existing QC stats to analyze stability
    # Since we have 5 replicates, we can simulate 2,3,4,5 reps
    # by subsampling from qc_stats correlation matrices
    # For now, use the pre-computed stats to show distribution
    n_reps = qc["n_reps"].values
    logger.info(f"  Replicate distribution: mean={n_reps.mean():.1f}, min={n_reps.min()}, max={n_reps.max()}")

    # Distribution of QC metrics
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.hist(qc["mean_rsd"], bins=50, alpha=0.7, color="steelblue", edgecolor="white")
    ax.axvline(CURRENT_RSD, color="red", linestyle="--", label=f"Threshold = {CURRENT_RSD}%")
    pct_pass = (qc["mean_rsd"] < CURRENT_RSD).mean() * 100
    ax.set_title(f"RSD Distribution (pass rate: {pct_pass:.1f}%)")
    ax.set_xlabel("Mean RSD (%)")
    ax.set_ylabel("Count")
    ax.legend()

    ax = axes[1]
    ax.hist(qc["mean_corr"], bins=50, alpha=0.7, color="darkorange", edgecolor="white")
    ax.axvline(CURRENT_CORR, color="red", linestyle="--", label=f"Threshold = {CURRENT_CORR}")
    pct_pass = (qc["mean_corr"] >= CURRENT_CORR).mean() * 100
    ax.set_title(f"Correlation Distribution (pass rate: {pct_pass:.1f}%)")
    ax.set_xlabel("Mean Correlation")
    ax.set_ylabel("Count")
    ax.legend()

    plt.tight_layout()
    fig.savefig(OUT_DIR / "convergence.png", dpi=150)
    plt.close(fig)
    logger.info("  Saved: convergence.png")


# =============================================================================
# Main
# =============================================================================
def main():
    logger.info("=" * 60)
    logger.info("  Phase 2: Statistical Optimization of QC Thresholds")
    logger.info("=" * 60)

    config = ModelConfig(group_aliases={"PAN": ["CPAN", "YPAN"]})
    df, qc = load_data()
    feature_cols = get_feature_cols(df)

    # Load QC stats
    qc_stats = pd.read_csv(QC_STATS_PATH)
    logger.info(f"  Loaded: {len(df)} spectra, {len(qc)} QC rows, {len(qc_stats)} QC stats")

    # 2-1. Fine-grained sweep
    df_sweep = run_fine_sweep(df, qc, config, feature_cols)

    # 2-2. Sensitivity analysis
    run_sensitivity_analysis(df_sweep)

    # 2-3. Independence analysis
    venn = run_independence_analysis(qc_stats)

    # 2-4. Per-cancer pass rate
    run_per_cancer_analysis(qc_stats)

    # 2-5. Pareto front
    run_pareto_analysis(df_sweep)

    # 2-6. Bootstrap CI
    bootstrap_results = run_bootstrap_ci(df, qc_stats, config, feature_cols, n_boot=100)

    # 2-7. Convergence
    run_convergence_analysis(qc_stats)

    logger.info("\n" + "=" * 60)
    logger.info("  Phase 2: Complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
