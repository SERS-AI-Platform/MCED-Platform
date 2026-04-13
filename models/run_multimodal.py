"""
Multimodal Experiment: SERS + Clinical Features

Fusion strategies:
    1. Early fusion: concatenate clinical features to spectral features → single LR
    2. Late fusion: separate SERS model + clinical model → blend predictions
    3. Subset with lab values: NOR/DIA/HBP/H.D. + LUN (binary only)

Clinical feature tiers:
    Tier 1 (all groups): age, sex, BMI
    Tier 2 (controls + LUN only): age, sex, BMI + WBC, RBC, Hb, platelet, AST, ALT,
                                   creatinine, glucose, total_cholesterol, calcium,
                                   total_bilirubin, uric_acid

Usage:
    python models/run_multimodal.py
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.train import (
    load_processed_spectra, get_feature_columns,
    apply_class_selection, resolve_aliases, aggregate_replicates,
    create_labels, ModelConfig,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import yaml
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

CANCER_TYPES = ["PRO", "LUN", "CRC", "CPAN", "OVA"]
NON_CANCER = ["NOR", "DIA", "HBP", "H.D."]

TIER1_FEATURES = ["age", "sex_numeric", "bmi"]
TIER2_LABS = ["wbc", "rbc", "hb", "platelet", "ast", "alt",
              "creatinine", "glucose", "total_cholesterol", "calcium",
              "total_bilirubin", "uric_acid"]


def load_clinical():
    clin = pd.read_csv(PROJECT_ROOT / "data" / "clinical_data" / "standardized" / "all_clinical_standardized.csv")
    clin["disease_group"] = clin["disease_group"].replace({"PAN": "CPAN"})
    clin["sex_numeric"] = (clin["sex"] == "M").astype(float)
    return clin


def merge_clinical(df_spec, clin, clinical_cols):
    """Merge spectral df with clinical columns. Returns arrays."""
    clin_lookup = {}
    for _, row in clin.iterrows():
        pid = row["patient_id"]
        clin_lookup[pid] = {c: row.get(c, np.nan) for c in clinical_cols}

    clin_arrays = {c: [] for c in clinical_cols}
    for _, row in df_spec.iterrows():
        key = f"{row['group']} {row['sample_id']}"
        if key in clin_lookup:
            for c in clinical_cols:
                clin_arrays[c].append(clin_lookup[key][c])
        else:
            for c in clinical_cols:
                clin_arrays[c].append(np.nan)

    for c in clinical_cols:
        df_spec = df_spec.copy()
        df_spec[c] = clin_arrays[c]

    return df_spec


def run_cv_lr(X, bl, ctl, sample_ids, n_cancer_types, seed=42, n_splits=5):
    """Run 5-fold CV with LR, return metrics."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    n = len(X)
    val_bp = np.full(n, np.nan)
    val_cp = np.full((n, n_cancer_types), 1.0 / n_cancer_types)

    for fold_i, (tr, va) in enumerate(cv.split(X, bl, sample_ids)):
        # Stage 1
        m1 = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, max_iter=1000, solver="saga", class_weight="balanced", random_state=seed + fold_i))
        m1.fit(X[tr], bl[tr])
        val_bp[va] = m1.predict_proba(X[va])[:, 1]

        # Stage 2
        cancer_mask = ctl[tr] >= 0
        if cancer_mask.sum() > 0:
            present = np.unique(ctl[tr][cancer_mask])
            if len(present) > 1:
                local_labels = np.searchsorted(present, ctl[tr][cancer_mask])
                m2 = make_pipeline(StandardScaler(), LogisticRegression(
                    C=1.0, max_iter=1000, solver="saga", class_weight="balanced",
                    random_state=seed + fold_i + 100))
                m2.fit(X[tr][cancer_mask], local_labels)
                probs = m2.predict_proba(X[va])
                for j, cls in enumerate(present):
                    val_cp[va, cls] = probs[:, j]

    # Metrics
    valid = ~np.isnan(val_bp)
    bp_v, bt_v = val_bp[valid], bl[valid]
    bpred = (bp_v > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(bt_v, bpred, labels=[0, 1]).ravel()

    result = {
        "s1_auc": float(roc_auc_score(bt_v, bp_v)),
        "s1_accuracy": float(accuracy_score(bt_v, bpred)),
        "s1_sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0,
        "s1_specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0,
    }

    cm = ctl[valid] >= 0
    if cm.sum() > 0:
        cpred = val_cp[valid][cm].argmax(axis=1)
        result["s2_accuracy"] = float(accuracy_score(ctl[valid][cm], cpred))
        result["s2_f1_macro"] = float(f1_score(ctl[valid][cm], cpred, average="macro", zero_division=0))
        try:
            result["s2_auc"] = float(roc_auc_score(ctl[valid][cm], val_cp[valid][cm],
                                                    multi_class="ovr", average="macro"))
        except ValueError:
            result["s2_auc"] = float("nan")

    return result, val_bp, val_cp


def run_late_fusion(val_bp_sers, val_cp_sers, val_bp_clin, val_cp_clin,
                    bl, ctl, alphas):
    """Blend SERS and clinical model predictions at various weights."""
    results = []
    for alpha in alphas:
        bp = alpha * val_bp_sers + (1 - alpha) * val_bp_clin
        cp = alpha * val_cp_sers + (1 - alpha) * val_cp_clin

        valid = ~np.isnan(bp)
        bp_v, bt_v = bp[valid], bl[valid]
        bpred = (bp_v > 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(bt_v, bpred, labels=[0, 1]).ravel()

        r = {
            "alpha": alpha,
            "s1_auc": float(roc_auc_score(bt_v, bp_v)),
            "s1_accuracy": float(accuracy_score(bt_v, bpred)),
        }

        cm = ctl[valid] >= 0
        if cm.sum() > 0:
            cpred = cp[valid][cm].argmax(axis=1)
            r["s2_f1_macro"] = float(f1_score(ctl[valid][cm], cpred, average="macro", zero_division=0))

        results.append(r)
    return results


def main():
    out_dir = PROJECT_ROOT / "results" / "training" / "multimodal"
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(out_dir / "multimodal.log", mode="w", encoding="utf-8")],
    )

    t0 = datetime.now()

    logger.info("=" * 64)
    logger.info("  Multimodal Experiment: SERS + Clinical Features")
    logger.info("=" * 64)

    # Load spectral data
    with open("config/config.yaml", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    df = load_processed_spectra()
    feat_cols = get_feature_columns(df)
    mc = ModelConfig.from_pipeline_config(raw_cfg, n_spectral_features=len(feat_cols))
    mc = apply_class_selection(mc, cancer_types=CANCER_TYPES, non_cancer_groups=NON_CANCER)
    df = resolve_aliases(df, mc)
    df_agg = aggregate_replicates(df, feat_cols, "none")

    # Load clinical
    clin = load_clinical()
    all_clinical_cols = TIER1_FEATURES + TIER2_LABS
    df_merged = merge_clinical(df_agg, clin, all_clinical_cols)

    # Create labels
    X, bl, ctl, sample_ids, groups_arr = create_labels(df_merged, mc)

    # Extract clinical features
    used_rows = df_merged["group"].isin(CANCER_TYPES + NON_CANCER)
    df_used = df_merged[used_rows].reset_index(drop=True)

    tier1_data = df_used[TIER1_FEATURES].values.astype(float)
    tier1_valid = ~np.isnan(tier1_data).any(axis=1)

    # Fill missing BMI with median
    bmi_col = TIER1_FEATURES.index("bmi")
    bmi_median = np.nanmedian(tier1_data[:, bmi_col])
    tier1_filled = tier1_data.copy()
    tier1_filled[np.isnan(tier1_filled[:, bmi_col]), bmi_col] = bmi_median
    tier1_valid_filled = ~np.isnan(tier1_filled).any(axis=1)

    logger.info(f"\n  Spectra: {len(X)}")
    logger.info(f"  Tier 1 valid (age+sex+bmi): {tier1_valid_filled.sum()}/{len(X)}")

    # ================================================================
    # Experiment 1: Full cohort (5 cancers, 4 controls)
    # ================================================================
    logger.info("\n" + "=" * 64)
    logger.info("  Experiment 1: Full Cohort — Early Fusion")
    logger.info("=" * 64)

    idx1 = np.where(tier1_valid_filled)[0]
    X_s = X[idx1]
    X_t1 = tier1_filled[idx1]
    bl_s = bl[idx1]
    ctl_s = ctl[idx1]
    sid_s = sample_ids[idx1]

    logger.info(f"  Samples with Tier 1 clinical: {len(idx1)}")

    # 1a: SERS only
    logger.info("\n  [1a] SERS only...")
    r_sers, bp_sers, cp_sers = run_cv_lr(X_s, bl_s, ctl_s, sid_s, mc.n_cancer_types, seed=mc.random_state)

    # 1b: Clinical only (Tier 1)
    logger.info("  [1b] Clinical only (age+sex+bmi)...")
    r_clin, bp_clin, cp_clin = run_cv_lr(X_t1, bl_s, ctl_s, sid_s, mc.n_cancer_types, seed=mc.random_state)

    # 1c: Early fusion (SERS + Tier 1)
    logger.info("  [1c] Early fusion (SERS + age+sex+bmi)...")
    X_early = np.hstack([X_s, X_t1])
    r_early, bp_early, cp_early = run_cv_lr(X_early, bl_s, ctl_s, sid_s, mc.n_cancer_types, seed=mc.random_state)

    # 1d: Late fusion (blend SERS + clinical predictions)
    logger.info("  [1d] Late fusion (blend sweep)...")
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    late_results = run_late_fusion(bp_sers, cp_sers, bp_clin, cp_clin, bl_s, ctl_s, alphas)

    best_late = max(late_results, key=lambda r: r.get("s1_auc", 0))

    logger.info(f"\n  {'Model':<35s} {'S1 AUC':>8s} {'S1 Acc':>8s} {'S1 Sens':>8s} {'S1 Spec':>8s} {'S2 F1':>8s}")
    logger.info("  " + "-" * 75)
    for label, r in [("SERS only", r_sers), ("Clinical only (age+sex+bmi)", r_clin),
                     ("Early fusion (SERS+clinical)", r_early),
                     (f"Late fusion (alpha={best_late['alpha']:.1f})", best_late)]:
        s2 = f"{r.get('s2_f1_macro', 0):.4f}"
        sens = f"{r.get('s1_sensitivity', 0):.4f}"
        spec = f"{r.get('s1_specificity', 0):.4f}"
        logger.info(f"  {label:<35s} {r['s1_auc']:>8.4f} {r['s1_accuracy']:>8.4f} {sens:>8s} {spec:>8s} {s2:>8s}")

    # ================================================================
    # Experiment 2: Subset with Lab Values (NOR/DIA/HBP/H.D. + LUN)
    # ================================================================
    logger.info("\n" + "=" * 64)
    logger.info("  Experiment 2: Lab Subset (Controls + LUN) — Binary Only")
    logger.info("=" * 64)

    lab_groups = ["NOR", "DIA", "HBP", "H.D.", "LUN"]
    lab_mask = df_used["group"].isin(lab_groups)
    tier2_cols = TIER1_FEATURES + TIER2_LABS
    tier2_data = df_used[tier2_cols].values.astype(float)

    # Fill missing lab values with per-column median for rows that have SOME labs
    # First identify rows in lab groups that have age+sex (Tier1)
    has_tier1 = ~np.isnan(tier2_data[:, :3]).any(axis=1)
    has_any_lab = ~np.isnan(tier2_data[:, 3:]).all(axis=1)  # has at least one lab
    lab_candidate = lab_mask.values & has_tier1

    # For lab columns, fill NaN with column median computed from available data
    tier2_filled = tier2_data.copy()
    for j in range(3, tier2_filled.shape[1]):  # skip age/sex/bmi
        col_vals = tier2_filled[lab_candidate, j]
        col_median = np.nanmedian(col_vals) if np.any(~np.isnan(col_vals)) else 0
        tier2_filled[np.isnan(tier2_filled[:, j]) & lab_candidate, j] = col_median

    # BMI fill
    bmi_med = np.nanmedian(tier2_filled[:, 2])
    tier2_filled[np.isnan(tier2_filled[:, 2]), 2] = bmi_med

    lab_valid = lab_candidate & (~np.isnan(tier2_filled).any(axis=1))
    idx2 = np.where(lab_valid)[0]

    logger.info(f"  Groups: {lab_groups}")
    logger.info(f"  Samples with full lab data: {len(idx2)}")

    if len(idx2) > 100:
        X_s2 = X[idx2]
        X_t2 = tier2_filled[idx2]
        bl_s2 = bl[idx2]
        sid_s2 = sample_ids[idx2]
        # Binary only (no cancer subtype — only LUN is cancer here)
        ctl_s2 = np.full(len(idx2), -1)  # No stage 2

        grp_counts = df_used.iloc[idx2]["group"].value_counts()
        for g in lab_groups:
            if g in grp_counts.index:
                logger.info(f"    {g}: {grp_counts[g]}")

        # 2a: SERS only
        logger.info("\n  [2a] SERS only (lab subset)...")
        r_sers2, _, _ = run_cv_lr(X_s2, bl_s2, ctl_s2, sid_s2, 1, seed=mc.random_state)

        # 2b: Clinical (Tier 1 only)
        X_t1_sub = tier2_data[idx2][:, :3]  # age, sex, bmi
        logger.info("  [2b] Clinical (age+sex+bmi, lab subset)...")
        r_clin2a, _, _ = run_cv_lr(X_t1_sub, bl_s2, ctl_s2, sid_s2, 1, seed=mc.random_state)

        # 2c: Clinical (Tier 1 + labs)
        logger.info("  [2c] Clinical (age+sex+bmi+labs)...")
        r_clin2b, _, _ = run_cv_lr(X_t2, bl_s2, ctl_s2, sid_s2, 1, seed=mc.random_state)

        # 2d: Early fusion (SERS + Tier 1)
        logger.info("  [2d] Early fusion (SERS + age+sex+bmi)...")
        r_early2a, _, _ = run_cv_lr(np.hstack([X_s2, X_t1_sub]), bl_s2, ctl_s2, sid_s2, 1, seed=mc.random_state)

        # 2e: Early fusion (SERS + Tier 1 + labs)
        logger.info("  [2e] Early fusion (SERS + all clinical)...")
        r_early2b, _, _ = run_cv_lr(np.hstack([X_s2, X_t2]), bl_s2, ctl_s2, sid_s2, 1, seed=mc.random_state)

        # 2f: Labs only (no SERS)
        logger.info("  [2f] Labs only (no SERS, no demographics)...")
        X_labs_only = tier2_filled[idx2][:, 3:]  # just lab values
        r_labs_only, _, _ = run_cv_lr(X_labs_only, bl_s2, ctl_s2, sid_s2, 1, seed=mc.random_state)

        logger.info(f"\n  {'Model':<40s} {'S1 AUC':>8s} {'S1 Acc':>8s}")
        logger.info("  " + "-" * 55)
        for label, r in [("SERS only", r_sers2),
                         ("Clinical (age+sex+bmi)", r_clin2a),
                         ("Labs only (12 values)", r_labs_only),
                         ("Clinical (age+sex+bmi+labs)", r_clin2b),
                         ("Early fusion (SERS+demographics)", r_early2a),
                         ("Early fusion (SERS+demographics+labs)", r_early2b)]:
            logger.info(f"  {label:<40s} {r['s1_auc']:>8.4f} {r['s1_accuracy']:>8.4f}")

    # ================================================================
    # Plots
    # ================================================================
    logger.info("\n  Generating plots...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Full cohort comparison
    ax = axes[0]
    models_full = [
        ("Clinical\nonly", r_clin["s1_auc"], r_clin.get("s2_f1_macro", 0)),
        ("SERS\nonly", r_sers["s1_auc"], r_sers.get("s2_f1_macro", 0)),
        ("Early\nfusion", r_early["s1_auc"], r_early.get("s2_f1_macro", 0)),
    ]
    x = np.arange(len(models_full))
    w = 0.35
    ax.bar(x - w/2, [m[1] for m in models_full], w, label="S1 AUC", color="#1976D2", alpha=0.8)
    ax.bar(x + w/2, [m[2] for m in models_full], w, label="S2 F1", color="#E53935", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in models_full])
    ax.set_ylabel("Score")
    ax.set_title("Full Cohort: Early Fusion\n(SERS + age/sex/BMI)")
    ax.legend()
    ax.set_ylim(0.2, 1.05)
    ax.grid(True, alpha=0.3)
    for i, m in enumerate(models_full):
        ax.text(i - w/2, m[1] + 0.01, f"{m[1]:.3f}", ha="center", fontsize=8)
        ax.text(i + w/2, m[2] + 0.01, f"{m[2]:.3f}", ha="center", fontsize=8)

    # Plot 2: Late fusion sweep
    ax = axes[1]
    late_s1 = [r["s1_auc"] for r in late_results]
    late_s2 = [r.get("s2_f1_macro", 0) for r in late_results]
    ax.plot(alphas, late_s1, "o-", color="#1976D2", label="S1 AUC", linewidth=2)
    ax.plot(alphas, late_s2, "s-", color="#E53935", label="S2 F1", linewidth=2)
    ax.set_xlabel("alpha (SERS weight)")
    ax.set_ylabel("Score")
    ax.set_title("Late Fusion Sweep\n(alpha=1: pure SERS, alpha=0: pure clinical)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)

    # Plot 3: Lab subset comparison
    ax = axes[2]
    if len(idx2) > 100:
        models_lab = [
            ("Clinical\n(demo)", r_clin2a["s1_auc"]),
            ("Labs\nonly", r_labs_only["s1_auc"]),
            ("Clinical\n(demo+labs)", r_clin2b["s1_auc"]),
            ("SERS\nonly", r_sers2["s1_auc"]),
            ("SERS+\ndemo", r_early2a["s1_auc"]),
            ("SERS+\ndemo+labs", r_early2b["s1_auc"]),
        ]
        colors = ["#FF9800", "#FF9800", "#FF9800", "#1976D2", "#9C27B0", "#9C27B0"]
        x = np.arange(len(models_lab))
        bars = ax.bar(x, [m[1] for m in models_lab], color=colors, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([m[0] for m in models_lab], fontsize=8)
        ax.set_ylabel("S1 AUC")
        ax.set_title(f"Lab Subset (Controls + LUN, n={len(idx2)})\nBinary: Cancer vs Non-cancer")
        ax.set_ylim(0.5, 1.05)
        ax.grid(True, alpha=0.3)
        for i, m in enumerate(models_lab):
            ax.text(i, m[1] + 0.01, f"{m[1]:.3f}", ha="center", fontsize=8, fontweight="bold")

    plt.suptitle("Multimodal: SERS + Clinical Features", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "multimodal_results.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Save summary
    summary = {
        "experiment": "multimodal",
        "timestamp": datetime.now().isoformat(),
        "full_cohort": {
            "sers_only": r_sers,
            "clinical_only": r_clin,
            "early_fusion": r_early,
            "late_fusion_best": best_late,
            "late_fusion_all": late_results,
        },
        "elapsed": str(datetime.now() - t0),
    }
    if len(idx2) > 100:
        summary["lab_subset"] = {
            "n_samples": len(idx2),
            "sers_only": r_sers2,
            "clinical_demo": r_clin2a,
            "labs_only": r_labs_only,
            "clinical_full": r_clin2b,
            "early_fusion_demo": r_early2a,
            "early_fusion_full": r_early2b,
        }

    with open(out_dir / "multimodal_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"\n  Elapsed: {datetime.now() - t0}")
    logger.info(f"  Output: {out_dir}/")
    logger.info("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
