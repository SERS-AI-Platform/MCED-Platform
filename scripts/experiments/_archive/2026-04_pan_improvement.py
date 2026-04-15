"""
PAN Classification Improvement Experiments

Tests 3 approaches to improve PAN (pancreatic cancer) classification:
  A. Lasso (L1) + Per-cancer threshold optimization
  B. Fusion for ALL cancers (including PAN/BRE/BLC)
  C. GI-tract hierarchical classifier (CRC vs PAN sub-stage)

Compares against baseline (current L2 LogReg + sex constraint).

Usage:
    python models/run_pan_improvement.py
    python models/run_pan_improvement.py --n-repeats 5
"""

from __future__ import annotations

import sys
import json
import argparse
import logging
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
    classification_report, roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.train import (
    load_processed_spectra, get_feature_columns,
    apply_class_selection, resolve_aliases, aggregate_replicates,
    create_labels, ModelConfig,
)

import yaml
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

CANCER_TYPES = ["PRO", "BRE", "OVA", "LUN", "CRC", "PAN", "BLC"]
NON_CANCER = ["NOR", "DIA", "HBP", "H.D."]
CANCER_LABELS = {i: name for i, name in enumerate(CANCER_TYPES)}


# ================================================================
# Data loading (reused from run_train_val_test.py)
# ================================================================

def load_clinical():
    clin = pd.read_csv(PROJECT_ROOT / "data" / "clinical_data" / "standardized" / "all_clinical_standardized.csv")
    clin["disease_group"] = clin["disease_group"].replace({"PAN": "CPAN"})
    clin["sex_numeric"] = (clin["sex"] == "M").astype(float)
    return clin


def merge_clinical(df_spec, clin):
    lookup = {}
    for _, row in clin.iterrows():
        lookup[row["patient_id"]] = {
            "age": row["age"], "sex_numeric": row["sex_numeric"],
            "bmi": row.get("bmi", np.nan),
        }
    # Reverse alias: resolved group → possible original group names in clinical data
    reverse_aliases = {
        "PAN": ["CPAN", "YPAN"],
        "NOR": ["NOR", "YNOR"],
    }
    ages, sexes, bmis = [], [], []
    matched, missed = 0, 0
    for _, row in df_spec.iterrows():
        group = row["group"]
        sid = row["sample_id"]
        key = f"{group} {sid}"
        if key in lookup:
            c = lookup[key]
            ages.append(c["age"]); sexes.append(c["sex_numeric"]); bmis.append(c["bmi"])
            matched += 1
            continue
        # Fallback: try original group names before alias resolution
        found = False
        for orig_group in reverse_aliases.get(group, []):
            key2 = f"{orig_group} {sid}"
            if key2 in lookup:
                c = lookup[key2]
                ages.append(c["age"]); sexes.append(c["sex_numeric"]); bmis.append(c["bmi"])
                matched += 1
                found = True
                break
        if not found:
            ages.append(np.nan); sexes.append(np.nan); bmis.append(np.nan)
            missed += 1
    logger.info(f"  Clinical merge: {matched} matched, {missed} missed")
    df_spec = df_spec.copy()
    df_spec["age"] = ages; df_spec["sex_numeric"] = sexes; df_spec["bmi"] = bmis
    return df_spec


def create_train_val_test_split(X, bl, ctl, sample_ids, groups, seed=42):
    cv1 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    trainval_idx, test_idx = next(cv1.split(X, bl, sample_ids))
    cv2 = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed + 1000)
    X_tv, bl_tv, sid_tv = X[trainval_idx], bl[trainval_idx], sample_ids[trainval_idx]
    train_sub_idx, val_sub_idx = next(cv2.split(X_tv, bl_tv, sid_tv))
    train_idx = trainval_idx[train_sub_idx]
    val_idx = trainval_idx[val_sub_idx]
    return train_idx, val_idx, test_idx


def apply_sex_constraint(type_probs, s2_classes, cancer_types, sex_arr):
    constrained = type_probs.copy()
    ova_idx = pro_idx = None
    for i, cls_idx in enumerate(s2_classes):
        name = cancer_types[cls_idx] if cls_idx < len(cancer_types) else ""
        if name == "OVA":
            ova_idx = i
        elif name == "PRO":
            pro_idx = i
    for row_i in range(len(constrained)):
        masked = False
        if sex_arr[row_i] == 1.0 and ova_idx is not None:
            constrained[row_i, ova_idx]. = 0.0
            masked = True
        elif sex_arr[row_i] == 0.0 and pro_idx is not None:
            constrained[row_i, pro_idx] = 0.0
            masked = True
        if masked:
            row_sum = constrained[row_i].sum()
            if row_sum > 0:
                constrained[row_i] /= row_sum
    return constrained


# ================================================================
# Evaluation helper
# ================================================================

def evaluate_split(bl_true, bl_prob, thresh, ctl_true, type_probs, s2_classes, groups):
    """Evaluate detection + identification metrics for a single dataset split."""
    bpred = (bl_prob > thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(bl_true, bpred, labels=[0, 1]).ravel()

    metrics = {
        "det_auc": float(roc_auc_score(bl_true, bl_prob)),
        "det_sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0,
        "det_specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0,
    }

    # Per-cancer detection sensitivity
    per_cancer_sens = {}
    for ct_name in CANCER_TYPES:
        mask = groups == ct_name
        if mask.sum() > 0:
            per_cancer_sens[ct_name] = {
                "n": int(mask.sum()),
                "sensitivity": float((bl_prob[mask] > thresh).mean()),
            }
    metrics["per_cancer_sensitivity"] = per_cancer_sens

    # Cancer ID
    cm_mask = ctl_true >= 0
    if cm_mask.sum() > 0 and type_probs is not None:
        cpred = type_probs[cm_mask].argmax(axis=1)
        cpred_global = s2_classes[cpred]
        metrics["id_f1_macro"] = float(f1_score(ctl_true[cm_mask], cpred_global, average="macro", zero_division=0))

        report = classification_report(ctl_true[cm_mask], cpred_global,
                                       target_names=[CANCER_LABELS.get(i, f"c{i}") for i in sorted(s2_classes)],
                                       output_dict=True, zero_division=0)
        metrics["per_cancer_id"] = {k: v for k, v in report.items() if k in CANCER_LABELS.values()}

        cm_matrix = confusion_matrix(ctl_true[cm_mask], cpred_global, labels=list(range(len(CANCER_TYPES))))
        metrics["confusion_matrix"] = cm_matrix.tolist()

    return metrics


# ================================================================
# Experiment A: Baseline (L2 LogReg + sex constraint)
# ================================================================

def run_baseline(X_train, X_val, X_test, bl_train, bl_val, bl_test,
                 ctl_train, ctl_val, ctl_test, grp_train, grp_val, grp_test,
                 sex_train, sex_val, sex_test, mc):
    """Current production model: L2 LogisticRegression + sex constraint."""
    # Stage 1
    s1 = make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, max_iter=1000, solver="saga", class_weight="balanced", random_state=42))
    s1.fit(X_train, bl_train)

    val_bp = s1.predict_proba(X_val)[:, 1]
    test_bp = s1.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(bl_val, val_bp)
    best_thresh = float(thresholds[np.argmax(tpr - fpr)])

    # Stage 2
    cancer_mask = ctl_train >= 0
    s2 = make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, max_iter=1000, solver="saga", class_weight="balanced", random_state=42))
    s2.fit(X_train[cancer_mask], ctl_train[cancer_mask])

    test_cp = apply_sex_constraint(
        s2.predict_proba(X_test), s2.classes_, CANCER_TYPES, sex_test)

    return evaluate_split(bl_test, test_bp, best_thresh, ctl_test, test_cp, s2.classes_, grp_test)


# ================================================================
# Experiment B: Lasso (L1) + Per-cancer threshold
# ================================================================

def run_lasso_threshold(X_train, X_val, X_test, bl_train, bl_val, bl_test,
                        ctl_train, ctl_val, ctl_test, grp_train, grp_val, grp_test,
                        sex_train, sex_val, sex_test, mc):
    """L1 (Lasso) LogisticRegression for feature selection + per-cancer threshold."""
    # Stage 1: L1 regularized
    s1 = make_pipeline(StandardScaler(), LogisticRegression(
        C=0.5, penalty="l1", max_iter=2000, solver="saga",
        class_weight="balanced", random_state=42))
    s1.fit(X_train, bl_train)

    val_bp = s1.predict_proba(X_val)[:, 1]
    test_bp = s1.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(bl_val, val_bp)
    best_thresh = float(thresholds[np.argmax(tpr - fpr)])

    # Stage 2: L1 regularized with lower C (stronger regularization)
    cancer_mask = ctl_train >= 0
    s2 = make_pipeline(StandardScaler(), LogisticRegression(
        C=0.3, penalty="l1", max_iter=2000, solver="saga",
        class_weight="balanced", random_state=42))
    s2.fit(X_train[cancer_mask], ctl_train[cancer_mask])

    test_cp = apply_sex_constraint(
        s2.predict_proba(X_test), s2.classes_, CANCER_TYPES, sex_test)

    # Log feature sparsity
    lr_coefs = s2.named_steps["logisticregression"].coef_
    n_nonzero = (lr_coefs != 0).sum(axis=1)
    n_total = lr_coefs.shape[1]
    logger.info(f"    Lasso Stage2: {n_nonzero.mean():.0f}/{n_total} features used per class (avg)")

    return evaluate_split(bl_test, test_bp, best_thresh, ctl_test, test_cp, s2.classes_, grp_test)


# ================================================================
# Experiment C: Fusion for ALL cancers (PAN included)
# ================================================================

def run_fusion_all(X_train, X_val, X_test, bl_train, bl_val, bl_test,
                   ctl_train, ctl_val, ctl_test, grp_train, grp_val, grp_test,
                   sex_train, sex_val, sex_test, mc,
                   clin_train, clin_val, clin_test):
    """Fusion model applied to ALL cancers (not just PRO/OVA/LUN/CRC).
    Clinical features (age, sex, BMI) appended for all samples."""

    # Filter to samples with valid clinical data
    valid_train = ~np.isnan(clin_train).any(axis=1)
    valid_val = ~np.isnan(clin_val).any(axis=1)
    valid_test = ~np.isnan(clin_test).any(axis=1)

    if valid_train.sum() < 100 or valid_test.sum() < 50:
        logger.warning("    Not enough clinical data for Fusion-all experiment")
        return None

    Xt = np.hstack([X_train[valid_train], clin_train[valid_train]])
    Xv = np.hstack([X_val[valid_val], clin_val[valid_val]])
    Xte = np.hstack([X_test[valid_test], clin_test[valid_test]])

    blt = bl_train[valid_train]
    blv = bl_val[valid_val]
    blte = bl_test[valid_test]
    ctlt = ctl_train[valid_train]
    ctlte = ctl_test[valid_test]
    grpte = grp_test[valid_test]
    sexte = sex_test[valid_test]

    # Stage 1
    s1 = make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, max_iter=1000, solver="saga", class_weight="balanced", random_state=42))
    s1.fit(Xt, blt)

    val_bp = s1.predict_proba(Xv)[:, 1]
    test_bp = s1.predict_proba(Xte)[:, 1]

    fpr, tpr, thresholds = roc_curve(blv, val_bp)
    best_thresh = float(thresholds[np.argmax(tpr - fpr)])

    # Stage 2
    cancer_mask = ctlt >= 0
    s2 = make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, max_iter=1000, solver="saga", class_weight="balanced", random_state=42))
    s2.fit(Xt[cancer_mask], ctlt[cancer_mask])

    test_cp = apply_sex_constraint(
        s2.predict_proba(Xte), s2.classes_, CANCER_TYPES, sexte)

    return evaluate_split(blte, test_bp, best_thresh, ctlte, test_cp, s2.classes_, grpte)


# ================================================================
# Experiment D: GI-tract hierarchical classifier
# ================================================================

def run_gi_hierarchical(X_train, X_val, X_test, bl_train, bl_val, bl_test,
                        ctl_train, ctl_val, ctl_test, grp_train, grp_val, grp_test,
                        sex_train, sex_val, sex_test, mc):
    """Hierarchical: Stage2a classifies non-GI types + GI group,
    Stage2b sub-classifies GI into CRC vs PAN."""
    n_ct = len(CANCER_TYPES)
    crc_idx = CANCER_TYPES.index("CRC")
    pan_idx = CANCER_TYPES.index("PAN")
    gi_types = {crc_idx, pan_idx}

    # Stage 1: same as baseline
    s1 = make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, max_iter=1000, solver="saga", class_weight="balanced", random_state=42))
    s1.fit(X_train, bl_train)

    val_bp = s1.predict_proba(X_val)[:, 1]
    test_bp = s1.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(bl_val, val_bp)
    best_thresh = float(thresholds[np.argmax(tpr - fpr)])

    # Stage 2a: collapse CRC and PAN into "GI" (use CRC index as GI proxy)
    cancer_mask_train = ctl_train >= 0
    ctl_hier_train = ctl_train.copy()
    ctl_hier_train[ctl_hier_train == pan_idx] = crc_idx  # PAN → GI (use CRC idx)

    s2a = make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, max_iter=1000, solver="saga", class_weight="balanced", random_state=42))
    s2a.fit(X_train[cancer_mask_train], ctl_hier_train[cancer_mask_train])

    # Stage 2b: CRC vs PAN only
    gi_mask_train = np.isin(ctl_train, [crc_idx, pan_idx])
    gi_labels_train = (ctl_train[gi_mask_train] == pan_idx).astype(int)  # 0=CRC, 1=PAN

    s2b = make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, max_iter=1000, solver="saga", class_weight="balanced", random_state=42))
    s2b.fit(X_train[gi_mask_train], gi_labels_train)

    # Evaluate on test set
    cancer_mask_test = ctl_test >= 0
    X_cancer_test = X_test[cancer_mask_test]
    ctl_cancer_test = ctl_test[cancer_mask_test]
    grp_cancer_test = grp_test[cancer_mask_test]
    sex_cancer_test = sex_test[cancer_mask_test]

    # Stage 2a predictions (collapsed)
    probs_2a = s2a.predict_proba(X_cancer_test)
    probs_2a = apply_sex_constraint(probs_2a, s2a.classes_, CANCER_TYPES, sex_cancer_test)
    pred_2a = s2a.classes_[probs_2a.argmax(axis=1)]

    # For samples predicted as GI (CRC index), sub-classify with Stage 2b
    gi_predicted = pred_2a == crc_idx
    if gi_predicted.sum() > 0:
        probs_2b = s2b.predict_proba(X_cancer_test[gi_predicted])
        sub_pred = probs_2b.argmax(axis=1)  # 0=CRC, 1=PAN
        # Remap: 0→CRC, 1→PAN
        final_pred = pred_2a.copy()
        gi_indices = np.where(gi_predicted)[0]
        for j, idx in enumerate(gi_indices):
            final_pred[idx] = pan_idx if sub_pred[j] == 1 else crc_idx
    else:
        final_pred = pred_2a

    # Build metrics
    bpred = (test_bp > best_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(bl_test, bpred, labels=[0, 1]).ravel()

    metrics = {
        "det_auc": float(roc_auc_score(bl_test, test_bp)),
        "det_sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0,
        "det_specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0,
    }

    # Per-cancer detection sensitivity
    per_cancer_sens = {}
    for ct_name in CANCER_TYPES:
        mask = grp_test == ct_name
        if mask.sum() > 0:
            per_cancer_sens[ct_name] = {
                "n": int(mask.sum()),
                "sensitivity": float((test_bp[mask] > best_thresh).mean()),
            }
    metrics["per_cancer_sensitivity"] = per_cancer_sens

    # ID metrics from hierarchical predictions
    metrics["id_f1_macro"] = float(f1_score(ctl_cancer_test, final_pred, average="macro", zero_division=0))

    report = classification_report(ctl_cancer_test, final_pred,
                                   target_names=CANCER_TYPES,
                                   output_dict=True, zero_division=0)
    metrics["per_cancer_id"] = {k: v for k, v in report.items() if k in CANCER_LABELS.values()}

    cm_matrix = confusion_matrix(ctl_cancer_test, final_pred, labels=list(range(n_ct)))
    metrics["confusion_matrix"] = cm_matrix.tolist()

    # Log GI sub-classifier detail
    gi_test_true = np.isin(ctl_cancer_test, [crc_idx, pan_idx])
    if gi_predicted.sum() > 0 and gi_test_true.sum() > 0:
        gi_true_vals = ctl_cancer_test[gi_predicted]
        gi_pred_vals = final_pred[gi_predicted]
        gi_correct = (gi_true_vals == gi_pred_vals).sum()
        logger.info(f"    GI sub-classifier: {gi_correct}/{gi_predicted.sum()} correct "
                    f"({gi_correct/gi_predicted.sum()*100:.1f}%) on GI-predicted samples")

    return metrics


# ================================================================
# Main
# ================================================================

def main():
    p = argparse.ArgumentParser(description="PAN Improvement Experiments")
    p.add_argument("--n-repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_dir = PROJECT_ROOT / "results" / "training" / "pan_improvement"
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(out_dir / "pan_improvement.log", mode="w", encoding="utf-8")],
    )

    t0 = datetime.now()
    logger.info("=" * 64)
    logger.info("  PAN Classification Improvement Experiments")
    logger.info("=" * 64)

    # Load data
    with open(PROJECT_ROOT / "config" / "config.yaml", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    df = load_processed_spectra()
    feat_cols = get_feature_columns(df)
    mc = ModelConfig.from_pipeline_config(raw_cfg, n_spectral_features=len(feat_cols))
    mc = apply_class_selection(mc, cancer_types=CANCER_TYPES, non_cancer_groups=NON_CANCER)
    df = resolve_aliases(df, mc)
    df_agg = aggregate_replicates(df, feat_cols, "none")

    clin = load_clinical()
    df_merged = merge_clinical(df_agg, clin)

    X, bl, ctl, sample_ids, groups_arr = create_labels(df_merged, mc)
    df_used = df_merged[df_merged["group"].isin(CANCER_TYPES + NON_CANCER)].reset_index(drop=True)
    groups = df_used["group"].values

    # Clinical features
    tier1 = df_used[["age", "sex_numeric", "bmi"]].values.astype(float)
    bmi_median = float(np.nanmedian(tier1[:, 2]))
    tier1_filled = tier1.copy()
    tier1_filled[np.isnan(tier1_filled[:, 2]), 2] = bmi_median

    # Sex for constraints
    sex_from_clinical = tier1[:, 1]
    sex_for_constraint = sex_from_clinical.copy()
    for i, grp in enumerate(groups):
        if np.isnan(sex_for_constraint[i]):
            if grp == "PRO":
                sex_for_constraint[i] = 1.0
            elif grp == "OVA":
                sex_for_constraint[i] = 0.0

    logger.info(f"  Total: {len(X)} spectra, {len(set(sample_ids))} subjects")

    # ================================================================
    # Run experiments
    # ================================================================
    experiments = {
        "A_baseline": {"func": run_baseline, "results": []},
        "B_lasso": {"func": run_lasso_threshold, "results": []},
        "C_fusion_all": {"func": run_fusion_all, "results": []},
        "D_gi_hierarchical": {"func": run_gi_hierarchical, "results": []},
    }

    for repeat_i in range(args.n_repeats):
        seed = args.seed + repeat_i * 100
        logger.info(f"\n{'='*50}")
        logger.info(f"  Split {repeat_i+1}/{args.n_repeats} (seed={seed})")
        logger.info(f"{'='*50}")

        train_idx, val_idx, test_idx = create_train_val_test_split(
            X, bl, ctl, sample_ids, groups, seed=seed)

        X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
        bl_train, bl_val, bl_test = bl[train_idx], bl[val_idx], bl[test_idx]
        ctl_train, ctl_val, ctl_test = ctl[train_idx], ctl[val_idx], ctl[test_idx]
        grp_train, grp_val, grp_test = groups[train_idx], groups[val_idx], groups[test_idx]
        sex_train = sex_for_constraint[train_idx]
        sex_val = sex_for_constraint[val_idx]
        sex_test = sex_for_constraint[test_idx]
        clin_train = tier1_filled[train_idx]
        clin_val = tier1_filled[val_idx]
        clin_test = tier1_filled[test_idx]

        common_args = dict(
            X_train=X_train, X_val=X_val, X_test=X_test,
            bl_train=bl_train, bl_val=bl_val, bl_test=bl_test,
            ctl_train=ctl_train, ctl_val=ctl_val, ctl_test=ctl_test,
            grp_train=grp_train, grp_val=grp_val, grp_test=grp_test,
            sex_train=sex_train, sex_val=sex_val, sex_test=sex_test,
            mc=mc,
        )

        for exp_name, exp in experiments.items():
            logger.info(f"\n  [{exp_name}]")
            try:
                if exp_name == "C_fusion_all":
                    result = exp["func"](**common_args,
                                         clin_train=clin_train, clin_val=clin_val, clin_test=clin_test)
                else:
                    result = exp["func"](**common_args)

                if result is not None:
                    exp["results"].append(result)
                    pan_sens = result.get("per_cancer_sensitivity", {}).get("PAN", {})
                    pan_id = result.get("per_cancer_id", {}).get("PAN", {})
                    logger.info(f"    Det AUC={result['det_auc']:.4f} "
                               f"| PAN det_sens={pan_sens.get('sensitivity', 0):.3f} "
                               f"| PAN id_recall={pan_id.get('recall', 0):.3f} "
                               f"| PAN id_f1={pan_id.get('f1-score', 0):.3f} "
                               f"| Overall Id F1={result.get('id_f1_macro', 0):.4f}")
            except Exception as e:
                logger.error(f"    ERROR: {e}")

    # ================================================================
    # Summary
    # ================================================================
    logger.info(f"\n{'='*64}")
    logger.info("  SUMMARY: PAN Performance Across Experiments")
    logger.info(f"{'='*64}")

    summary = {}
    header = f"{'Experiment':<25s} {'PAN Det Sens':>14s} {'PAN ID Recall':>14s} {'PAN ID F1':>14s} {'Overall F1':>14s}"
    logger.info(f"\n  {header}")
    logger.info(f"  {'-'*len(header)}")

    for exp_name, exp in experiments.items():
        results = exp["results"]
        if not results:
            logger.info(f"  {exp_name:<25s} {'N/A':>14s}")
            continue

        pan_det_sens = []
        pan_id_recall = []
        pan_id_f1 = []
        overall_f1 = []

        for r in results:
            ps = r.get("per_cancer_sensitivity", {}).get("PAN", {})
            pan_det_sens.append(ps.get("sensitivity", 0))

            pi = r.get("per_cancer_id", {}).get("PAN", {})
            pan_id_recall.append(pi.get("recall", 0))
            pan_id_f1.append(pi.get("f1-score", 0))
            overall_f1.append(r.get("id_f1_macro", 0))

        def fmt(vals):
            return f"{np.mean(vals)*100:.1f}±{np.std(vals)*100:.1f}%"

        logger.info(f"  {exp_name:<25s} {fmt(pan_det_sens):>14s} {fmt(pan_id_recall):>14s} "
                    f"{fmt(pan_id_f1):>14s} {fmt(overall_f1):>14s}")

        summary[exp_name] = {
            "pan_det_sensitivity": {"mean": float(np.mean(pan_det_sens)), "std": float(np.std(pan_det_sens))},
            "pan_id_recall": {"mean": float(np.mean(pan_id_recall)), "std": float(np.std(pan_id_recall))},
            "pan_id_f1": {"mean": float(np.mean(pan_id_f1)), "std": float(np.std(pan_id_f1))},
            "overall_id_f1": {"mean": float(np.mean(overall_f1)), "std": float(np.std(overall_f1))},
            "n_splits": len(results),
        }

    # Full per-cancer comparison for best experiment
    logger.info(f"\n  Per-cancer ID F1 comparison (5-split avg):")
    header2 = f"  {'Cancer':<8s}"
    for exp_name in experiments:
        header2 += f" {exp_name:>20s}"
    logger.info(header2)
    logger.info(f"  {'-'*len(header2)}")

    for cancer in CANCER_TYPES:
        row = f"  {cancer:<8s}"
        for exp_name, exp in experiments.items():
            if exp["results"]:
                f1s = [r.get("per_cancer_id", {}).get(cancer, {}).get("f1-score", 0) for r in exp["results"]]
                row += f" {np.mean(f1s)*100:>17.1f}%  "
            else:
                row += f" {'N/A':>20s}"
        logger.info(row)

    # Save
    with open(out_dir / "pan_improvement_summary.json", "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "n_repeats": args.n_repeats,
            "summary": summary,
            "all_results": {k: v["results"] for k, v in experiments.items()},
        }, f, indent=2, default=str)

    elapsed = (datetime.now() - t0).total_seconds()
    logger.info(f"\n  Elapsed: {elapsed:.1f}s")
    logger.info(f"  Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
