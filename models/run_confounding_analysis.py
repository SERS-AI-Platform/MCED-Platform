"""
Confounding Analysis: Is SERS detecting cancer or demographics?

Key concern:
    - Normal subjects are much younger (mean 45) than cancer patients (mean 65-73)
    - PRO is 100% male, OVA/BRE 100% female
    - The model might learn age/sex proxies instead of metabolite patterns

Tests:
    1. Demographics-only classifier (age, sex, BMI → cancer type)
    2. Correlation between spectral features and age/sex
    3. Age-matched subgroup analysis
    4. SERS + demographics fusion (does adding age/sex improve?)
    5. Age-stratified performance (young vs old cancer patients)

Usage:
    python models/run_confounding_analysis.py
"""

from __future__ import annotations

import sys
import logging
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, confusion_matrix,
    classification_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.train import (
    load_processed_spectra, get_feature_columns,
    apply_class_selection, resolve_aliases, aggregate_replicates,
    create_labels, ModelConfig,
)
from src.sers.config import RESULTS_DIR

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import yaml
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

CANCER_TYPES = ["PRO", "LUN", "CRC", "CPAN", "OVA"]
NON_CANCER = ["NOR", "DIA", "HBP", "H.D."]


def load_clinical_data():
    """Load and prepare clinical data."""
    clin = pd.read_csv(PROJECT_ROOT / "data" / "clinical_data" / "standardized" / "all_clinical_standardized.csv")

    # Map PAN -> CPAN to match spectral data
    clin["disease_group"] = clin["disease_group"].replace({"PAN": "CPAN"})

    # Encode sex
    clin["sex_numeric"] = (clin["sex"] == "M").astype(float)

    return clin


def merge_spectral_clinical(df_spec, feat_cols, clin):
    """Merge spectral data with clinical demographics."""
    # Build sample-level clinical lookup
    clin_lookup = {}
    for _, row in clin.iterrows():
        pid = row["patient_id"]
        clin_lookup[pid] = {
            "age": row["age"],
            "sex": row["sex"],
            "sex_numeric": row["sex_numeric"],
            "bmi": row.get("bmi", np.nan),
        }

    # Try matching: spectral sample_id format is like "100", clinical is like "PRO 100"
    ages, sexes, sex_nums, bmis = [], [], [], []
    matched = 0

    for _, row in df_spec.iterrows():
        grp = row["group"]
        sid = row["sample_id"]

        # Try different key formats
        keys_to_try = [
            f"{grp} {sid}",
            f"{grp}{sid}",
            str(sid),
        ]

        found = False
        for key in keys_to_try:
            if key in clin_lookup:
                c = clin_lookup[key]
                ages.append(c["age"])
                sexes.append(c["sex"])
                sex_nums.append(c["sex_numeric"])
                bmis.append(c["bmi"])
                matched += 1
                found = True
                break

        if not found:
            ages.append(np.nan)
            sexes.append(np.nan)
            sex_nums.append(np.nan)
            bmis.append(np.nan)

    df_spec = df_spec.copy()
    df_spec["age"] = ages
    df_spec["sex"] = sexes
    df_spec["sex_numeric"] = sex_nums
    df_spec["bmi"] = bmis

    logger.info(f"  Matched {matched}/{len(df_spec)} spectra with clinical data")
    logger.info(f"  Age available: {df_spec['age'].notna().sum()}")
    logger.info(f"  BMI available: {df_spec['bmi'].notna().sum()}")

    return df_spec


def run_demographics_only(X_demo, bl, ctl, sample_ids, groups, mc, label="demographics"):
    """Train classifier using only demographics (no spectra)."""
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=mc.random_state)

    n = len(X_demo)
    val_bp = np.full(n, np.nan)
    val_cp = np.full((n, mc.n_cancer_types), np.nan)

    for fold_i, (train_idx, val_idx) in enumerate(cv.split(X_demo, bl, sample_ids)):
        X_tr, X_va = X_demo[train_idx], X_demo[val_idx]
        bl_tr, bl_va = bl[train_idx], bl[val_idx]
        ctl_tr, ctl_va = ctl[train_idx], ctl[val_idx]

        # Stage 1: binary
        m1 = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, solver="saga",
                           class_weight="balanced", random_state=mc.random_state + fold_i))
        m1.fit(X_tr, bl_tr)
        val_bp[val_idx] = m1.predict_proba(X_va)[:, 1]

        # Stage 2: cancer type
        cancer_mask = ctl_tr >= 0
        if cancer_mask.sum() > 0:
            present = np.unique(ctl_tr[cancer_mask])
            if len(present) > 1:
                local_labels = np.searchsorted(present, ctl_tr[cancer_mask])
                m2 = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, solver="saga",
                                   class_weight="balanced", random_state=mc.random_state + fold_i + 100))
                m2.fit(X_tr[cancer_mask], local_labels)
                local_prob = m2.predict_proba(X_va)
                for j, cls in enumerate(present):
                    val_cp[val_idx, cls] = local_prob[:, j]

    # Compute metrics
    valid = ~np.isnan(val_bp)
    bp_v = val_bp[valid]
    bt_v = bl[valid]
    bpred = (bp_v > 0.5).astype(int)

    result = {
        "label": label,
        "s1_auc": float(roc_auc_score(bt_v, bp_v)),
        "s1_accuracy": float(accuracy_score(bt_v, bpred)),
    }

    tn, fp, fn, tp = confusion_matrix(bt_v, bpred, labels=[0, 1]).ravel()
    result["s1_sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0
    result["s1_specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0

    cancer_mask = ctl[valid] >= 0
    if cancer_mask.sum() > 0:
        cp_v = val_cp[valid][cancer_mask]
        ct_v = ctl[valid][cancer_mask]
        valid_cp = ~np.isnan(cp_v).any(axis=1)
        if valid_cp.sum() > 0:
            cpred = cp_v[valid_cp].argmax(axis=1)
            result["s2_accuracy"] = float(accuracy_score(ct_v[valid_cp], cpred))
            result["s2_f1_macro"] = float(f1_score(ct_v[valid_cp], cpred, average="macro", zero_division=0))

    return result


def compute_spectral_age_correlation(X, ages, feat_cols):
    """Compute correlation between each spectral feature and age."""
    valid = ~np.isnan(ages)
    X_v = X[valid]
    ages_v = ages[valid]

    correlations = []
    for i in range(X_v.shape[1]):
        corr = np.corrcoef(X_v[:, i], ages_v)[0, 1]
        correlations.append(corr)

    return np.array(correlations)


def main():
    out_dir = PROJECT_ROOT / "results" / "training" / "confounding_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(out_dir / "confounding.log", mode="w", encoding="utf-8")],
    )

    t0 = datetime.now()

    logger.info("=" * 64)
    logger.info("  Confounding Analysis: Age/Sex/BMI vs SERS")
    logger.info("=" * 64)

    # Load data
    with open("config/config.yaml", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    df = load_processed_spectra()
    feat_cols = get_feature_columns(df)
    n_feat = len(feat_cols)

    mc = ModelConfig.from_pipeline_config(raw_cfg, n_spectral_features=n_feat)
    mc = apply_class_selection(mc, cancer_types=CANCER_TYPES, non_cancer_groups=NON_CANCER)

    df = resolve_aliases(df, mc)
    df_agg = aggregate_replicates(df, feat_cols, "none")

    # Load clinical
    logger.info("\n[Step 1] Loading clinical data...")
    clin = load_clinical_data()
    df_merged = merge_spectral_clinical(df_agg, feat_cols, clin)

    # Create labels (same as standard pipeline)
    X, bl, ctl, sample_ids, groups_arr = create_labels(df_merged, mc)

    # Extract demographics
    ages = df_merged.loc[df_merged["group"].isin(CANCER_TYPES + NON_CANCER), "age"].values
    sex_nums = df_merged.loc[df_merged["group"].isin(CANCER_TYPES + NON_CANCER), "sex_numeric"].values
    bmis = df_merged.loc[df_merged["group"].isin(CANCER_TYPES + NON_CANCER), "bmi"].values

    logger.info(f"\n  Total spectra: {len(X)}")
    logger.info(f"  With age: {np.sum(~np.isnan(ages))}")
    logger.info(f"  With sex: {np.sum(~np.isnan(sex_nums))}")
    logger.info(f"  With BMI: {np.sum(~np.isnan(bmis))}")

    # ================================================================
    # Test 1: Demographics-only classifier
    # ================================================================
    logger.info("\n" + "=" * 64)
    logger.info("  Test 1: Demographics-only Classifier")
    logger.info("=" * 64)

    # Fill missing BMI with median for this test
    bmi_filled = bmis.copy()
    bmi_filled[np.isnan(bmi_filled)] = np.nanmedian(bmi_filled)

    # Age + Sex only
    X_age_sex = np.column_stack([ages, sex_nums])
    valid_demo = ~np.isnan(X_age_sex).any(axis=1)
    logger.info(f"  Valid age+sex samples: {valid_demo.sum()}/{len(X_age_sex)}")

    # Use only valid samples
    idx_valid = np.where(valid_demo)[0]
    X_as = X_age_sex[idx_valid]
    bl_v = bl[idx_valid]
    ctl_v = ctl[idx_valid]
    sid_v = sample_ids[idx_valid]

    result_age_sex = run_demographics_only(X_as, bl_v, ctl_v, sid_v, groups_arr[idx_valid], mc,
                                            label="age+sex")
    logger.info(f"  Age+Sex only:")
    logger.info(f"    S1 AUC: {result_age_sex['s1_auc']:.4f}")
    logger.info(f"    S1 Accuracy: {result_age_sex['s1_accuracy']:.4f}")
    logger.info(f"    S1 Sensitivity: {result_age_sex['s1_sensitivity']:.4f}")
    logger.info(f"    S1 Specificity: {result_age_sex['s1_specificity']:.4f}")
    logger.info(f"    S2 F1 macro: {result_age_sex.get('s2_f1_macro', 'N/A')}")

    # Age + Sex + BMI
    X_asb = np.column_stack([ages, sex_nums, bmi_filled])
    valid_asb = ~np.isnan(X_asb).any(axis=1)
    idx_asb = np.where(valid_asb)[0]

    result_age_sex_bmi = run_demographics_only(X_asb[idx_asb], bl[idx_asb], ctl[idx_asb],
                                                sample_ids[idx_asb], groups_arr[idx_asb], mc,
                                                label="age+sex+bmi")
    logger.info(f"\n  Age+Sex+BMI:")
    logger.info(f"    S1 AUC: {result_age_sex_bmi['s1_auc']:.4f}")
    logger.info(f"    S1 Accuracy: {result_age_sex_bmi['s1_accuracy']:.4f}")
    logger.info(f"    S2 F1 macro: {result_age_sex_bmi.get('s2_f1_macro', 'N/A')}")

    # Age only
    X_age = ages[idx_valid].reshape(-1, 1)
    result_age = run_demographics_only(X_age, bl_v, ctl_v, sid_v, groups_arr[idx_valid], mc,
                                        label="age_only")
    logger.info(f"\n  Age only:")
    logger.info(f"    S1 AUC: {result_age['s1_auc']:.4f}")

    # SERS only (on same valid subset for fair comparison)
    X_sers_sub = X[idx_valid]
    result_sers = run_demographics_only(X_sers_sub, bl_v, ctl_v, sid_v, groups_arr[idx_valid], mc,
                                         label="sers_only")
    logger.info(f"\n  SERS only (same subset):")
    logger.info(f"    S1 AUC: {result_sers['s1_auc']:.4f}")
    logger.info(f"    S2 F1 macro: {result_sers.get('s2_f1_macro', 'N/A')}")

    # SERS + Demographics
    X_sers_demo = np.column_stack([X[idx_valid], X_as])
    result_fusion = run_demographics_only(X_sers_demo, bl_v, ctl_v, sid_v, groups_arr[idx_valid], mc,
                                           label="sers+age+sex")
    logger.info(f"\n  SERS + Age + Sex:")
    logger.info(f"    S1 AUC: {result_fusion['s1_auc']:.4f}")
    logger.info(f"    S2 F1 macro: {result_fusion.get('s2_f1_macro', 'N/A')}")

    # ================================================================
    # Test 2: Spectral-Age Correlation
    # ================================================================
    logger.info("\n" + "=" * 64)
    logger.info("  Test 2: Spectral Feature — Age Correlation")
    logger.info("=" * 64)

    corr_age = compute_spectral_age_correlation(X, ages, feat_cols)
    logger.info(f"  Mean |correlation| with age: {np.nanmean(np.abs(corr_age)):.4f}")
    logger.info(f"  Max  |correlation| with age: {np.nanmax(np.abs(corr_age)):.4f}")
    logger.info(f"  Features with |corr| > 0.2: {np.sum(np.abs(corr_age) > 0.2)}/{len(corr_age)}")
    logger.info(f"  Features with |corr| > 0.3: {np.sum(np.abs(corr_age) > 0.3)}/{len(corr_age)}")

    # ================================================================
    # Test 3: Age Distribution Analysis
    # ================================================================
    logger.info("\n" + "=" * 64)
    logger.info("  Test 3: Age Distribution by Group")
    logger.info("=" * 64)

    groups_in_data = df_merged.loc[df_merged["group"].isin(CANCER_TYPES + NON_CANCER)]
    age_stats = groups_in_data.groupby("group")["age"].agg(["mean", "std", "min", "max", "count"])
    for grp in CANCER_TYPES + NON_CANCER:
        if grp in age_stats.index:
            row = age_stats.loc[grp]
            logger.info(f"  {grp:>5s}: mean={row['mean']:.1f} +/- {row['std']:.1f} "
                        f"[{row['min']:.0f}-{row['max']:.0f}] n={row['count']:.0f}")

    # ================================================================
    # Test 4: Age-matched Analysis
    # ================================================================
    logger.info("\n" + "=" * 64)
    logger.info("  Test 4: Age-matched Subgroup Analysis")
    logger.info("=" * 64)

    # Restrict to age 50-80 where there's overlap between cancer and normal
    age_mask = (ages >= 50) & (ages <= 80) & (~np.isnan(ages))
    idx_matched = np.where(age_mask)[0]

    logger.info(f"  Age 50-80 subset: {len(idx_matched)}/{len(X)} samples")

    # Check group distribution in matched subset
    matched_groups = groups_in_data.iloc[idx_matched] if len(idx_matched) <= len(groups_in_data) else None
    if matched_groups is not None:
        grp_counts = matched_groups["group"].value_counts()
        for grp in CANCER_TYPES + NON_CANCER:
            if grp in grp_counts.index:
                logger.info(f"    {grp}: {grp_counts[grp]}")

    if len(idx_matched) > 100:
        X_matched = X[idx_matched]
        bl_matched = bl[idx_matched]
        ctl_matched = ctl[idx_matched]
        sid_matched = sample_ids[idx_matched]
        grp_matched = groups_arr[idx_matched]

        result_matched = run_demographics_only(X_matched, bl_matched, ctl_matched,
                                                sid_matched, grp_matched, mc,
                                                label="sers_age_matched")
        logger.info(f"\n  SERS on age-matched subset (50-80):")
        logger.info(f"    S1 AUC: {result_matched['s1_auc']:.4f}")
        logger.info(f"    S2 F1 macro: {result_matched.get('s2_f1_macro', 'N/A')}")

        # Demographics-only on same subset
        ages_m = ages[idx_matched]
        sex_m = sex_nums[idx_matched]
        valid_m = ~np.isnan(ages_m) & ~np.isnan(sex_m)
        idx_m2 = np.where(valid_m)[0]

        if len(idx_m2) > 50:
            X_demo_matched = np.column_stack([ages_m[idx_m2], sex_m[idx_m2]])
            result_demo_matched = run_demographics_only(X_demo_matched, bl_matched[idx_m2],
                                                         ctl_matched[idx_m2], sid_matched[idx_m2],
                                                         grp_matched[idx_m2], mc,
                                                         label="demo_age_matched")
            logger.info(f"\n  Demographics-only on age-matched subset (50-80):")
            logger.info(f"    S1 AUC: {result_demo_matched['s1_auc']:.4f}")

    # ================================================================
    # Summary Table
    # ================================================================
    logger.info("\n" + "=" * 64)
    logger.info("  SUMMARY: Confounding Analysis")
    logger.info("=" * 64)

    all_results = [result_age, result_age_sex, result_age_sex_bmi,
                   result_sers, result_fusion]
    if 'result_matched' in dir():
        all_results.append(result_matched)
    if 'result_demo_matched' in dir():
        all_results.append(result_demo_matched)

    logger.info(f"\n  {'Model':<30s} {'S1 AUC':>8s} {'S1 Acc':>8s} {'S2 F1':>8s}")
    logger.info("  " + "-" * 60)
    for r in all_results:
        s2 = f"{r.get('s2_f1_macro', 0):.4f}" if 's2_f1_macro' in r else "  N/A "
        logger.info(f"  {r['label']:<30s} {r['s1_auc']:>8.4f} {r['s1_accuracy']:>8.4f} {s2:>8s}")

    # ================================================================
    # Plots
    # ================================================================
    logger.info("\n  Generating plots...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # (0,0) Age distribution by group
    ax = axes[0, 0]
    group_order = NON_CANCER + CANCER_TYPES
    colors_grp = {"NOR": "#43A047", "DIA": "#66BB6A", "HBP": "#81C784", "H.D.": "#A5D6A7",
                  "PRO": "#E53935", "LUN": "#EF5350", "CRC": "#F44336", "CPAN": "#E57373", "OVA": "#EF9A9A"}
    age_data = []
    labels_grp = []
    for grp in group_order:
        sub = groups_in_data[groups_in_data["group"] == grp]["age"].dropna()
        if len(sub) > 0:
            age_data.append(sub.values)
            labels_grp.append(grp)

    bp = ax.boxplot(age_data, labels=labels_grp, patch_artist=True, widths=0.6)
    for i, grp in enumerate(labels_grp):
        bp["boxes"][i].set_facecolor(colors_grp.get(grp, "#999999"))
        bp["boxes"][i].set_alpha(0.7)
    ax.axhline(50, color="gray", linestyle=":", linewidth=0.5)
    ax.axhline(80, color="gray", linestyle=":", linewidth=0.5)
    ax.axvspan(0.5, len(NON_CANCER) + 0.5, alpha=0.05, color="green")
    ax.axvspan(len(NON_CANCER) + 0.5, len(group_order) + 0.5, alpha=0.05, color="red")
    ax.set_ylabel("Age")
    ax.set_title("Age Distribution by Group\n(Green=Control, Red=Cancer)")
    ax.grid(True, alpha=0.3)

    # (0,1) Sex distribution by group
    ax = axes[0, 1]
    sex_data = []
    for grp in group_order:
        sub = groups_in_data[groups_in_data["group"] == grp]
        male_pct = (sub["sex"] == "M").mean() * 100 if len(sub) > 0 else 0
        sex_data.append(male_pct)
    bars = ax.bar(range(len(group_order)), sex_data,
                  color=[colors_grp.get(g, "#999") for g in group_order], alpha=0.7)
    ax.set_xticks(range(len(group_order)))
    ax.set_xticklabels(group_order)
    ax.set_ylabel("Male %")
    ax.set_title("Sex Distribution by Group")
    ax.axhline(50, color="gray", linestyle="--", linewidth=1)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    for i, v in enumerate(sex_data):
        ax.text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=8)

    # (0,2) Model comparison bar chart
    ax = axes[0, 2]
    comparison_labels = ["Age\nonly", "Age+Sex", "Age+Sex\n+BMI", "SERS\nonly", "SERS+\nAge+Sex"]
    comparison_s1 = [result_age["s1_auc"], result_age_sex["s1_auc"], result_age_sex_bmi["s1_auc"],
                     result_sers["s1_auc"], result_fusion["s1_auc"]]
    comparison_s2 = [result_age.get("s2_f1_macro", 0), result_age_sex.get("s2_f1_macro", 0),
                     result_age_sex_bmi.get("s2_f1_macro", 0), result_sers.get("s2_f1_macro", 0),
                     result_fusion.get("s2_f1_macro", 0)]

    x = np.arange(len(comparison_labels))
    w = 0.35
    ax.bar(x - w/2, comparison_s1, w, label="S1 AUC", color="#1976D2", alpha=0.8)
    ax.bar(x + w/2, comparison_s2, w, label="S2 F1", color="#E53935", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_labels, fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Demographics vs SERS Performance")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # (1,0) Spectral-Age correlation heatmap
    ax = axes[1, 0]
    wavenumbers = [float(c.replace("x_", "")) for c in feat_cols]
    ax.plot(wavenumbers, corr_age, color="#9C27B0", linewidth=0.8)
    ax.fill_between(wavenumbers, 0, corr_age,
                    where=(corr_age > 0), color="#E53935", alpha=0.2)
    ax.fill_between(wavenumbers, 0, corr_age,
                    where=(corr_age < 0), color="#1976D2", alpha=0.2)
    ax.axhline(0, color="gray", linestyle=":")
    ax.axhline(0.2, color="orange", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.axhline(-0.2, color="orange", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Pearson Correlation with Age")
    ax.set_title(f"Spectral Features vs Age\n(mean |r|={np.nanmean(np.abs(corr_age)):.3f})")
    ax.grid(True, alpha=0.3)

    # (1,1) Age-matched comparison
    ax = axes[1, 1]
    if 'result_matched' in dir() and 'result_demo_matched' in dir():
        match_labels = ["Full\nDemographics", "Full\nSERS", "Matched\nDemographics", "Matched\nSERS"]
        match_s1 = [result_age_sex["s1_auc"], result_sers["s1_auc"],
                    result_demo_matched["s1_auc"], result_matched["s1_auc"]]
        bars = ax.bar(range(len(match_labels)), match_s1,
                      color=["#FF9800", "#1976D2", "#FF9800", "#1976D2"], alpha=0.7,
                      edgecolor=["none", "none", "black", "black"], linewidth=2)
        ax.set_xticks(range(len(match_labels)))
        ax.set_xticklabels(match_labels, fontsize=9)
        ax.set_ylabel("S1 AUC")
        ax.set_title("Full vs Age-Matched (50-80)\n(bordered = age-matched)")
        ax.set_ylim(0.5, 1.05)
        ax.grid(True, alpha=0.3)
        for i, v in enumerate(match_s1):
            ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")

    # (1,2) Interpretation summary
    ax = axes[1, 2]
    ax.axis("off")

    demo_s1 = result_age_sex["s1_auc"]
    sers_s1 = result_sers["s1_auc"]
    fusion_s1 = result_fusion["s1_auc"]
    demo_s2 = result_age_sex.get("s2_f1_macro", 0)
    sers_s2 = result_sers.get("s2_f1_macro", 0)

    summary_text = (
        f"CONFOUNDING ANALYSIS SUMMARY\n"
        f"{'='*40}\n\n"
        f"Demographics alone:\n"
        f"  S1 AUC: {demo_s1:.3f}  S2 F1: {demo_s2:.3f}\n\n"
        f"SERS alone:\n"
        f"  S1 AUC: {sers_s1:.3f}  S2 F1: {sers_s2:.3f}\n\n"
        f"SERS + Demographics:\n"
        f"  S1 AUC: {fusion_s1:.3f}\n\n"
    )

    if demo_s1 > 0.75:
        summary_text += (
            f"WARNING: Demographics alone achieves\n"
            f"S1 AUC {demo_s1:.3f}. Age/sex confounding\n"
            f"is a significant concern.\n\n"
        )

    if fusion_s1 - sers_s1 < 0.005:
        summary_text += (
            f"SERS already captures demographic\n"
            f"signal (fusion adds < 0.5% AUC).\n"
        )
    else:
        summary_text += (
            f"Adding demographics improves SERS\n"
            f"by {(fusion_s1-sers_s1)*100:.1f}% AUC.\n"
        )

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=10, fontfamily="monospace", verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))

    plt.suptitle("Confounding Analysis: Is SERS Detecting Cancer or Demographics?",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(out_dir / "confounding_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out_dir / 'confounding_analysis.png'}")

    # Save results
    summary = {
        "experiment": "confounding_analysis",
        "timestamp": datetime.now().isoformat(),
        "results": all_results,
        "age_correlation": {
            "mean_abs": float(np.nanmean(np.abs(corr_age))),
            "max_abs": float(np.nanmax(np.abs(corr_age))),
            "n_above_0.2": int(np.sum(np.abs(corr_age) > 0.2)),
            "n_above_0.3": int(np.sum(np.abs(corr_age) > 0.3)),
        },
        "elapsed": str(datetime.now() - t0),
    }
    with open(out_dir / "confounding_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"\n  Elapsed: {datetime.now() - t0}")
    logger.info(f"  Output: {out_dir}/")
    logger.info("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
