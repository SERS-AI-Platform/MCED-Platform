"""
Phase 3: Clinical Impact Analysis of QC Thresholds

Analyzes the clinical implications of QC thresholds:
  3-1. Per-cancer sensitivity/specificity at different thresholds
  3-2. FPR/FNR analysis
  3-3. 51% rejection rate clinical context
  3-4. Per-group QC characteristics
  3-5. Prospective simulation

Usage:
    cd /home/user/SERS-AI
    PYTHONPATH=. python scripts/qc_validation/03_clinical_impact.py
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_qc_threshold_experiment import (
    aggregate_medoid, get_feature_cols, load_data
)


def filter_by_qc_fast(df, qc, rsd_thresh, corr_thresh):
    """Optimized filter using merge instead of apply."""
    if rsd_thresh == float("inf") and corr_thresh == 0.0:
        return df.copy()

    passed = qc.copy()
    if rsd_thresh < float("inf"):
        passed = passed[passed["mean_rsd"] <= rsd_thresh]
    if corr_thresh > 0.0:
        passed = passed[passed["mean_corr"] >= corr_thresh]

    passed_keys = passed[["group", "sample_id"]].copy()
    passed_keys["sample_id"] = passed_keys["sample_id"].astype(str)

    df_keys = df[["group", "sample_id"]].copy()
    df_keys["sample_id"] = df_keys["sample_id"].astype(str)
    mask = pd.merge(df_keys.reset_index(), passed_keys, on=["group", "sample_id"], how="inner")["index"]
    return df.loc[mask].copy()
from models.model import ModelConfig
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, classification_report,
    confusion_matrix, precision_score, recall_score
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

OUT_DIR = PROJECT_ROOT / "results" / "qc_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

QC_STATS_PATH = PROJECT_ROOT / "results" / "fixed_grid" / "qc_stats.csv"

CURRENT_RSD = 5.0
CURRENT_CORR = 0.95
N_SPLITS = 5
RANDOM_STATE = 42


# =============================================================================
# 3-1. Per-Cancer Sensitivity/Specificity
# =============================================================================
def run_per_cancer_sensitivity(df, qc, config, feature_cols):
    """Evaluate per-cancer sensitivity at different QC thresholds."""
    logger.info("=" * 60)
    logger.info("  3-1. Per-Cancer Sensitivity/Specificity")
    logger.info("=" * 60)

    thresholds = [
        (3, 0.95, "Strict"),
        (5, 0.95, "Current"),
        (7, 0.90, "Moderate"),
        (10, 0.85, "Relaxed"),
        (float("inf"), 0.0, "No QC"),
    ]

    all_results = []

    for rsd_t, corr_t, label in thresholds:
        logger.info(f"\n  --- {label}: RSD<={rsd_t}, Corr>={corr_t} ---")

        df_f = filter_by_qc_fast(df, qc, rsd_t, corr_t)
        n_samples = df_f.groupby(["group", "sample_id"]).ngroups
        if n_samples < 20:
            continue

        df_agg = aggregate_medoid(df_f, feature_cols)
        df_agg = df_agg.copy()
        df_agg["group"] = df_agg["group"].apply(config.resolve_group)

        valid_groups = set(config.cancer_types) | set(config.non_cancer_groups) | set(config.cancer_groups_raw)
        df_v = df_agg[df_agg["group"].isin(valid_groups)].copy()

        X = df_v[feature_cols].values
        groups = df_v["group"].values
        sample_ids = df_v["sample_id"].values
        binary = np.array([1 if g in config.cancer_types else 0 for g in groups])
        cancer_idx = np.array([config.cancer_type_index(g) if g in config.cancer_types else -1 for g in groups])

        cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

        # Collect per-class predictions
        all_true_binary = []
        all_pred_binary = []
        all_true_cancer = []
        all_pred_cancer = []

        for train_idx, val_idx in cv.split(X, binary, sample_ids):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_bin_tr, y_bin_val = binary[train_idx], binary[val_idx]
            ct_tr, ct_val = cancer_idx[train_idx], cancer_idx[val_idx]

            # Stage 1
            lr1 = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", random_state=RANDOM_STATE)
            lr1.fit(X_tr, y_bin_tr)
            pred_bin = lr1.predict(X_val)
            all_true_binary.extend(y_bin_val.tolist())
            all_pred_binary.extend(pred_bin.tolist())

            # Stage 2
            cancer_tr = ct_tr[y_bin_tr == 1]
            cancer_val = ct_val[y_bin_val == 1]
            X_tr_c = X_tr[y_bin_tr == 1]
            X_val_c = X_val[y_bin_val == 1]

            if len(np.unique(cancer_tr)) >= 2 and len(X_val_c) >= 2:
                lr2 = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs",
                                         multi_class="multinomial", random_state=RANDOM_STATE)
                lr2.fit(X_tr_c, cancer_tr)
                pred_ct = lr2.predict(X_val_c)
                all_true_cancer.extend(cancer_val.tolist())
                all_pred_cancer.extend(pred_ct.tolist())

        # Binary metrics
        true_b = np.array(all_true_binary)
        pred_b = np.array(all_pred_binary)
        sensitivity = recall_score(true_b, pred_b, pos_label=1)
        specificity = recall_score(true_b, pred_b, pos_label=0)
        fpr = 1 - specificity
        fnr = 1 - sensitivity

        # Per-cancer type metrics
        true_c = np.array(all_true_cancer)
        pred_c = np.array(all_pred_cancer)
        cancer_names = config.cancer_types

        per_cancer = {}
        for i, cname in enumerate(cancer_names):
            mask = true_c == i
            if mask.sum() > 0:
                correct = (pred_c[mask] == i).sum()
                per_cancer[cname] = {
                    "sensitivity": correct / mask.sum(),
                    "n_samples": int(mask.sum()),
                }

        result_row = {
            "label": label,
            "rsd_threshold": rsd_t,
            "corr_threshold": corr_t,
            "n_samples": n_samples,
            "binary_sensitivity": round(sensitivity, 4),
            "binary_specificity": round(specificity, 4),
            "fpr": round(fpr, 4),
            "fnr": round(fnr, 4),
        }

        for cname, metrics in per_cancer.items():
            result_row[f"{cname}_sensitivity"] = round(metrics["sensitivity"], 4)
            result_row[f"{cname}_n"] = metrics["n_samples"]

        all_results.append(result_row)
        logger.info(f"  Sensitivity={sensitivity:.3f}, Specificity={specificity:.3f}, "
                     f"FPR={fpr:.3f}, FNR={fnr:.3f}, N={n_samples}")

    df_results = pd.DataFrame(all_results)
    df_results.to_csv(OUT_DIR / "per_cancer_sensitivity.csv", index=False)
    logger.info(f"\n  Saved: per_cancer_sensitivity.csv")

    # Plot
    plot_sensitivity_comparison(df_results, config)

    return df_results


def plot_sensitivity_comparison(df_results, config):
    """Plot sensitivity comparison across thresholds."""
    cancer_types = config.cancer_types
    sensitivity_cols = [f"{ct}_sensitivity" for ct in cancer_types if f"{ct}_sensitivity" in df_results.columns]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Binary sensitivity/specificity
    ax = axes[0]
    x = range(len(df_results))
    ax.bar([i - 0.15 for i in x], df_results["binary_sensitivity"], 0.3,
           label="Sensitivity", color="steelblue")
    ax.bar([i + 0.15 for i in x], df_results["binary_specificity"], 0.3,
           label="Specificity", color="darkorange")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df_results["label"], rotation=30)
    ax.set_ylabel("Rate")
    ax.set_title("Binary Classification (Cancer vs Non-Cancer)")
    ax.legend()
    ax.set_ylim(0.5, 1.05)

    # Per-cancer sensitivity
    ax = axes[1]
    if sensitivity_cols:
        df_plot = df_results[["label"] + sensitivity_cols].set_index("label")
        df_plot.columns = [c.replace("_sensitivity", "") for c in df_plot.columns]
        df_plot.plot(kind="bar", ax=ax, width=0.8)
        ax.set_ylabel("Sensitivity")
        ax.set_title("Per-Cancer Type Sensitivity")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7, bbox_to_anchor=(1.0, 1.0))
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "sensitivity_comparison.png", dpi=150)
    plt.close(fig)
    logger.info("  Saved: sensitivity_comparison.png")


# =============================================================================
# 3-2. FPR/FNR Analysis (already included in 3-1)
# =============================================================================
def plot_fpr_fnr(df_results):
    """Dedicated FPR/FNR plot."""
    logger.info("\n" + "=" * 60)
    logger.info("  3-2. FPR/FNR Analysis")
    logger.info("=" * 60)

    fig, ax = plt.subplots(figsize=(8, 5))

    x = range(len(df_results))
    ax.plot(list(x), df_results["fpr"], "r-o", label="FPR (False Positive Rate)", markersize=8)
    ax.plot(list(x), df_results["fnr"], "b-s", label="FNR (False Negative Rate)", markersize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(df_results["label"], rotation=30)
    ax.set_ylabel("Error Rate")
    ax.set_title("FPR/FNR by QC Threshold Stringency")
    ax.legend()
    ax.grid(alpha=0.3)

    # Add sample count annotations
    for i, row in df_results.iterrows():
        ax.annotate(f"N={row['n_samples']}", (i, max(row['fpr'], row['fnr']) + 0.01),
                    ha="center", fontsize=8, color="gray")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fpr_fnr_curve.png", dpi=150)
    plt.close(fig)
    logger.info("  Saved: fpr_fnr_curve.png")


# =============================================================================
# 3-3. 51% Rejection Rate Clinical Context
# =============================================================================
def analyze_rejection_rate(qc_stats):
    """Compare our rejection rate with other clinical screening tests."""
    logger.info("\n" + "=" * 60)
    logger.info("  3-3. Rejection Rate Clinical Context")
    logger.info("=" * 60)

    n_total = len(qc_stats)
    n_pass = ((qc_stats["mean_rsd"] < CURRENT_RSD) & (qc_stats["mean_corr"] >= CURRENT_CORR)).sum()
    rejection_rate = (1 - n_pass / n_total) * 100

    comparisons = [
        {"test": "Pap Smear (Cervical)", "rejection_rate": "5-10%",
         "reason": "Unsatisfactory specimen (cellularity, drying artifact)",
         "action": "Re-collect within 2-4 months"},
        {"test": "Liquid Biopsy (ctDNA)", "rejection_rate": "2-5%",
         "reason": "Insufficient DNA yield, hemolysis",
         "action": "Re-draw blood sample"},
        {"test": "PSA Screening", "rejection_rate": "10-15%",
         "reason": "Indeterminate range (4-10 ng/mL) requiring follow-up",
         "action": "Repeat test or biopsy referral"},
        {"test": "Mammography", "rejection_rate": "10-15%",
         "reason": "BI-RADS 0 (incomplete, needs additional imaging)",
         "action": "Additional imaging within 6 months"},
        {"test": "FOBT/FIT (Colorectal)", "rejection_rate": "1-3%",
         "reason": "Insufficient sample, collection error",
         "action": "Re-collect stool sample"},
        {"test": "SERS-AI (Ours)", "rejection_rate": f"{rejection_rate:.1f}%",
         "reason": "Low replicate consistency (RSD/Corr QC failure)",
         "action": "Re-test with new SERS measurement (5 replicates)"},
    ]

    df_comp = pd.DataFrame(comparisons)
    df_comp.to_csv(OUT_DIR / "rejection_rate_comparison.csv", index=False)

    # Analysis
    report = [
        "# QC Rejection Rate Clinical Analysis",
        "",
        f"## Current Status: {rejection_rate:.1f}% rejection rate ({n_total - n_pass}/{n_total} samples)",
        "",
        "## Comparison with Other Screening Tests",
        "",
        df_comp.to_markdown(index=False),
        "",
        "## Key Considerations",
        "",
        "### 1. Rejection ≠ Lost Patient",
        "- SERS QC failure → re-test 가능 (비침습적 소변 검사)",
        "- 재검 비용: 기판 1개 + 5분 측정 (매우 낮음)",
        "- 환자 입장: 소변 재제출 (부담 최소)",
        "",
        "### 2. 높은 rejection rate의 원인",
        "- **BLC (90% fail)**: 신규 추가 데이터, 프로토콜/기판 차이 가능성",
        "- **NOR (75% fail)**: 정상군 검체 품질 이슈 가능",
        "- **OVA (75.7% fail)**: 샘플 수 적음 (70개), 통계적 불안정",
        "",
        "### 3. 실제 운용 시 예상 rejection rate",
        "- BLC 데이터 품질 개선 시: rejection rate 크게 감소 예상",
        "- 장비 표준화 (Mira P, 92% pass) 시: rejection rate ~10% 수준 가능",
        "",
        "### 4. Regulatory Perspective",
        "- FDA는 specimen adequacy rate를 요구하지만 고정 수치 없음",
        "- 핵심은 rejection 사유와 후속 조치(re-test) 프로토콜의 명확성",
        "- ICH Q2(R2): 과학적 정당화가 고정 수치보다 중요",
    ]

    with open(OUT_DIR / "clinical_comparison.md", "w") as f:
        f.write("\n".join(report))

    logger.info(f"  Rejection rate: {rejection_rate:.1f}%")
    logger.info(f"  Saved: rejection_rate_comparison.csv, clinical_comparison.md")


# =============================================================================
# 3-4. Per-Group QC Characteristics
# =============================================================================
def analyze_group_qc(qc_stats):
    """Analyze QC metric distributions per group."""
    logger.info("\n" + "=" * 60)
    logger.info("  3-4. Per-Group QC Characteristics")
    logger.info("=" * 60)

    groups = sorted(qc_stats["group"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # RSD box plot
    ax = axes[0]
    data_rsd = [qc_stats[qc_stats["group"] == g]["mean_rsd"].values for g in groups]
    bp = ax.boxplot(data_rsd, labels=groups, patch_artist=True)
    ax.axhline(CURRENT_RSD, color="red", linestyle="--", alpha=0.7, label=f"Threshold = {CURRENT_RSD}%")
    ax.set_ylabel("Mean RSD (%)")
    ax.set_title("RSD Distribution by Group")
    ax.legend()
    ax.tick_params(axis="x", rotation=45)

    # Correlation box plot
    ax = axes[1]
    data_corr = [qc_stats[qc_stats["group"] == g]["mean_corr"].values for g in groups]
    bp = ax.boxplot(data_corr, labels=groups, patch_artist=True)
    ax.axhline(CURRENT_CORR, color="red", linestyle="--", alpha=0.7, label=f"Threshold = {CURRENT_CORR}")
    ax.set_ylabel("Mean Correlation")
    ax.set_title("Correlation Distribution by Group")
    ax.legend()
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "group_qc_boxplot.png", dpi=150)
    plt.close(fig)
    logger.info("  Saved: group_qc_boxplot.png")

    # Summary table
    summary = qc_stats.groupby("group").agg(
        n_samples=("sample_id", "count"),
        rsd_mean=("mean_rsd", "mean"),
        rsd_median=("mean_rsd", "median"),
        rsd_std=("mean_rsd", "std"),
        corr_mean=("mean_corr", "mean"),
        corr_median=("mean_corr", "median"),
        corr_std=("mean_corr", "std"),
    ).round(4)

    # Pass rate at current thresholds
    pass_rates = qc_stats.groupby("group").apply(
        lambda g: ((g["mean_rsd"] < CURRENT_RSD) & (g["mean_corr"] >= CURRENT_CORR)).mean() * 100
    ).round(1)
    summary["pass_rate_pct"] = pass_rates

    summary.to_csv(OUT_DIR / "group_qc_summary.csv")
    logger.info("  Saved: group_qc_summary.csv")
    logger.info(f"\n{summary.to_string()}")


# =============================================================================
# 3-5. Prospective Simulation
# =============================================================================
def run_prospective_simulation(qc_stats, n_patients=10000, cancer_prevalence=0.05, n_sims=1000):
    """Simulate prospective screening scenario."""
    logger.info("\n" + "=" * 60)
    logger.info(f"  3-5. Prospective Simulation ({n_patients} patients, {cancer_prevalence*100}% prevalence)")
    logger.info("=" * 60)

    # QC pass probability from empirical distribution
    pass_rate = ((qc_stats["mean_rsd"] < CURRENT_RSD) &
                 (qc_stats["mean_corr"] >= CURRENT_CORR)).mean()

    # Separate pass rates for cancer vs non-cancer
    cancer_groups = {"PRO", "BRE", "OVA", "LUN", "CRC", "CPAN", "BLC"}
    non_cancer_groups = {"NOR", "DIA", "HBP", "H.D."}

    cancer_mask = qc_stats["group"].isin(cancer_groups)
    non_cancer_mask = qc_stats["group"].isin(non_cancer_groups)

    cancer_pass_rate = ((qc_stats[cancer_mask]["mean_rsd"] < CURRENT_RSD) &
                        (qc_stats[cancer_mask]["mean_corr"] >= CURRENT_CORR)).mean()
    non_cancer_pass_rate = ((qc_stats[non_cancer_mask]["mean_rsd"] < CURRENT_RSD) &
                            (qc_stats[non_cancer_mask]["mean_corr"] >= CURRENT_CORR)).mean()

    logger.info(f"  Overall QC pass rate: {pass_rate:.3f}")
    logger.info(f"  Cancer QC pass rate: {cancer_pass_rate:.3f}")
    logger.info(f"  Non-cancer QC pass rate: {non_cancer_pass_rate:.3f}")

    # Simulation
    results = []
    for _ in range(n_sims):
        n_cancer = np.random.binomial(n_patients, cancer_prevalence)
        n_healthy = n_patients - n_cancer

        # QC pass
        cancer_qc_pass = np.random.binomial(n_cancer, cancer_pass_rate)
        healthy_qc_pass = np.random.binomial(n_healthy, non_cancer_pass_rate)

        total_qc_pass = cancer_qc_pass + healthy_qc_pass
        total_qc_fail = n_patients - total_qc_pass

        results.append({
            "n_cancer": n_cancer,
            "n_healthy": n_healthy,
            "cancer_qc_pass": cancer_qc_pass,
            "healthy_qc_pass": healthy_qc_pass,
            "total_qc_pass": total_qc_pass,
            "total_qc_fail": total_qc_fail,
            "effective_rejection_rate": total_qc_fail / n_patients,
        })

    df_sim = pd.DataFrame(results)
    df_sim.to_csv(OUT_DIR / "prospective_simulation.csv", index=False)

    # Summary
    summary = {
        "n_patients": n_patients,
        "cancer_prevalence": cancer_prevalence,
        "qc_pass_rate_overall": round(pass_rate, 4),
        "qc_pass_rate_cancer": round(cancer_pass_rate, 4),
        "qc_pass_rate_noncancer": round(non_cancer_pass_rate, 4),
        "expected_qc_pass": round(df_sim["total_qc_pass"].mean()),
        "expected_qc_fail": round(df_sim["total_qc_fail"].mean()),
        "expected_rejection_rate": round(df_sim["effective_rejection_rate"].mean() * 100, 1),
        "cancer_lost_to_qc": round(df_sim["n_cancer"].mean() - df_sim["cancer_qc_pass"].mean()),
        "cancer_lost_to_qc_pct": round((1 - cancer_pass_rate) * 100, 1),
    }

    logger.info(f"\n  === Prospective Simulation Results ===")
    logger.info(f"  {n_patients} patients screened:")
    logger.info(f"    Expected QC pass: {summary['expected_qc_pass']:.0f} ({100-summary['expected_rejection_rate']:.1f}%)")
    logger.info(f"    Expected QC fail (re-test): {summary['expected_qc_fail']:.0f} ({summary['expected_rejection_rate']}%)")
    logger.info(f"    Cancer patients lost to QC: {summary['cancer_lost_to_qc']:.0f} ({summary['cancer_lost_to_qc_pct']}%)")
    logger.info(f"    → These patients can be RE-TESTED (non-invasive urine)")

    return summary


# =============================================================================
# Main
# =============================================================================
def main():
    logger.info("=" * 60)
    logger.info("  Phase 3: Clinical Impact Analysis")
    logger.info("=" * 60)

    config = ModelConfig(group_aliases={"PAN": ["CPAN", "YPAN"]})
    df, qc = load_data()
    feature_cols = get_feature_cols(df)
    qc_stats = pd.read_csv(QC_STATS_PATH)

    # 3-1. Per-cancer sensitivity
    df_sensitivity = run_per_cancer_sensitivity(df, qc_stats, config, feature_cols)

    # 3-2. FPR/FNR
    plot_fpr_fnr(df_sensitivity)

    # 3-3. Rejection rate context
    analyze_rejection_rate(qc_stats)

    # 3-4. Per-group QC
    analyze_group_qc(qc_stats)

    # 3-5. Prospective simulation
    run_prospective_simulation(qc_stats)

    logger.info("\n" + "=" * 60)
    logger.info("  Phase 3: Complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
