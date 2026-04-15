"""
Train / Validation / Test Split Experiment

Instead of 5-fold CV, use a proper held-out test set:
    - Train (60%): model training
    - Validation (20%): threshold tuning, model selection
    - Test (20%): final unbiased evaluation (touched ONCE)

Split is at SUBJECT level (no replicate leakage).
Uses StratifiedGroupKFold to create balanced splits.

Usage:
    python models/run_train_val_test.py
    python models/run_train_val_test.py --n-repeats 5  # multiple random splits
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
    classification_report, roc_curve, average_precision_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.train import (
    load_processed_spectra, get_feature_columns,
    apply_class_selection, resolve_aliases, aggregate_replicates,
    apply_patient_exclusions, create_labels, ModelConfig,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import yaml
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

CANCER_TYPES = ["PRO", "BRE", "OVA", "LUN", "CRC", "PAN", "BLC"]
NON_CANCER = ["NOR", "DIA", "HBP", "H.D."]
CANCER_LABELS = {i: name for i, name in enumerate(CANCER_TYPES)}


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
    """Create 60/20/20 train/val/test split at subject level."""
    # First split: 80% trainval / 20% test
    cv1 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    trainval_idx, test_idx = next(cv1.split(X, bl, sample_ids))

    # Second split: 75% train / 25% val (of trainval = 60/20 of total)
    cv2 = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed + 1000)
    X_tv, bl_tv, sid_tv = X[trainval_idx], bl[trainval_idx], sample_ids[trainval_idx]
    train_sub_idx, val_sub_idx = next(cv2.split(X_tv, bl_tv, sid_tv))

    train_idx = trainval_idx[train_sub_idx]
    val_idx = trainval_idx[val_sub_idx]

    return train_idx, val_idx, test_idx


def apply_sex_constraint(type_probs, s2_classes, cancer_types, sex_arr):
    """Zero out biologically impossible cancer types based on sex and renormalize.

    Males cannot have OVA (ovarian), females cannot have PRO (prostate).
    """
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
        if sex_arr[row_i] == 1.0 and ova_idx is not None:  # Male → mask OVA
            constrained[row_i, ova_idx] = 0.0
            masked = True
        elif sex_arr[row_i] == 0.0 and pro_idx is not None:  # Female → mask PRO
            constrained[row_i, pro_idx] = 0.0
            masked = True
        if masked:
            row_sum = constrained[row_i].sum()
            if row_sum > 0:
                constrained[row_i] /= row_sum
    return constrained


def train_and_evaluate(X, bl, ctl, groups_arr, train_idx, val_idx, test_idx,
                       mc, clinical_features=None, sex_numeric=None, label=""):
    """Train LR on train, tune on val, evaluate on test."""
    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    bl_train, bl_val, bl_test = bl[train_idx], bl[val_idx], bl[test_idx]
    ctl_train, ctl_val, ctl_test = ctl[train_idx], ctl[val_idx], ctl[test_idx]
    grp_train = groups_arr[train_idx]
    grp_val = groups_arr[val_idx]
    grp_test = groups_arr[test_idx]

    # Append clinical features if provided
    if clinical_features is not None:
        cf_train = clinical_features[train_idx]
        cf_val = clinical_features[val_idx]
        cf_test = clinical_features[test_idx]
        X_train = np.hstack([X_train, cf_train])
        X_val = np.hstack([X_val, cf_val])
        X_test = np.hstack([X_test, cf_test])

    n_ct = mc.n_cancer_types

    # ── Stage 1: Cancer Detection ──
    s1 = make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, max_iter=1000, solver="saga", class_weight="balanced", random_state=42))
    s1.fit(X_train, bl_train)

    val_bp = s1.predict_proba(X_val)[:, 1]
    test_bp = s1.predict_proba(X_test)[:, 1]

    # Optimize threshold on validation set
    fpr, tpr, thresholds = roc_curve(bl_val, val_bp)
    j_scores = tpr - fpr
    best_thresh = float(thresholds[np.argmax(j_scores)])

    # ── Stage 2: Cancer Identification ──
    cancer_train = ctl_train >= 0
    s2 = make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, max_iter=1000, solver="saga", class_weight="balanced", random_state=42))
    s2.fit(X_train[cancer_train], ctl_train[cancer_train])

    val_cp = s2.predict_proba(X_val)
    test_cp = s2.predict_proba(X_test)

    # ── Apply sex-based constraint if sex info is available ──
    if sex_numeric is not None:
        sex_train = sex_numeric[train_idx]
        sex_val = sex_numeric[val_idx]
        sex_test = sex_numeric[test_idx]
        s2_classes = s2.classes_
        train_cp_constrained = apply_sex_constraint(s2.predict_proba(X_train), s2_classes, CANCER_TYPES, sex_train)
        val_cp = apply_sex_constraint(val_cp, s2_classes, CANCER_TYPES, sex_val)
        test_cp = apply_sex_constraint(test_cp, s2_classes, CANCER_TYPES, sex_test)
    else:
        train_cp_constrained = s2.predict_proba(X_train)

    # ── Evaluate on each set ──
    results = {}
    for set_name, bp, cp, bt, ct, grps in [
        ("train", s1.predict_proba(X_train)[:, 1], train_cp_constrained, bl_train, ctl_train, grp_train),
        ("val", val_bp, val_cp, bl_val, ctl_val, grp_val),
        ("test", test_bp, test_cp, bl_test, ctl_test, grp_test),
    ]:
        bpred = (bp > best_thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(bt, bpred, labels=[0, 1]).ravel()

        metrics = {
            "n_samples": len(bt),
            "n_subjects": len(set(range(len(bt)))),  # approximate
            "threshold": best_thresh,
            "det_auc": float(roc_auc_score(bt, bp)),
            "det_pr_auc": float(average_precision_score(bt, bp)),
            "det_accuracy": float(accuracy_score(bt, bpred)),
            "det_sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0,
            "det_specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0,
            "det_f1": float(f1_score(bt, bpred)),
        }

        # Per-cancer sensitivity
        per_cancer = {}
        for ct_name in CANCER_TYPES:
            mask = grps == ct_name
            if mask.sum() > 0:
                per_cancer[ct_name] = {
                    "n": int(mask.sum()),
                    "sensitivity": float((bp[mask] > best_thresh).mean()),
                }
        metrics["per_cancer_sensitivity"] = per_cancer

        # Per-control specificity
        per_control = {}
        for ctrl_name in NON_CANCER:
            mask = grps == ctrl_name
            if mask.sum() > 0:
                per_control[ctrl_name] = {
                    "n": int(mask.sum()),
                    "specificity": float((bp[mask] <= best_thresh).mean()),
                }
        metrics["per_control_specificity"] = per_control

        # Cancer identification
        cm_mask = ct >= 0
        if cm_mask.sum() > 0:
            s2_classes = s2.classes_
            cpred = cp[cm_mask].argmax(axis=1)
            # Map local predictions back to global class indices
            cpred_global = s2_classes[cpred]
            metrics["id_accuracy"] = float(accuracy_score(ct[cm_mask], cpred_global))
            metrics["id_f1_macro"] = float(f1_score(ct[cm_mask], cpred_global, average="macro", zero_division=0))
            try:
                metrics["id_auc"] = float(roc_auc_score(ct[cm_mask], cp[cm_mask], multi_class="ovr", average="macro"))
            except ValueError:
                metrics["id_auc"] = float("nan")
            try:
                per_class_ap = []
                for local_idx, global_cls in enumerate(s2_classes):
                    cls_bg = (ct[cm_mask] == global_cls).astype(int)
                    if cls_bg.sum() > 0:
                        per_class_ap.append(average_precision_score(cls_bg, cp[cm_mask][:, local_idx]))
                metrics["id_pr_auc"] = float(np.mean(per_class_ap)) if per_class_ap else float("nan")
            except (ValueError, IndexError):
                metrics["id_pr_auc"] = float("nan")

            # Per-cancer F1
            present_labels = sorted(set(ct[cm_mask]) | set(cpred_global))
            report = classification_report(ct[cm_mask], cpred_global,
                                           labels=present_labels,
                                           target_names=[CANCER_LABELS.get(i, f"c{i}") for i in present_labels],
                                           output_dict=True, zero_division=0)
            metrics["per_cancer_id"] = {k: v for k, v in report.items()
                                        if k in CANCER_LABELS.values()}

            # Confusion matrix
            cm_matrix = confusion_matrix(ct[cm_mask], cpred_global, labels=list(range(n_ct)))
            metrics["confusion_matrix"] = cm_matrix.tolist()

        results[set_name] = metrics

    return results, best_thresh, s1, s2


def main():
    p = argparse.ArgumentParser(description="Train/Val/Test Split Experiment")
    p.add_argument("--n-repeats", type=int, default=5, help="Number of random splits")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--exclude-patients", default=None,
                   help="CSV file with (group, sample_id) columns to exclude")
    args = p.parse_args()

    exp_name = "train_val_test_7cancer_clean" if args.exclude_patients else "train_val_test_7cancer"
    out_dir = PROJECT_ROOT / "results" / "training" / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(out_dir / "train_val_test.log", mode="w", encoding="utf-8")],
    )

    t0 = datetime.now()

    logger.info("=" * 64)
    logger.info("  Train / Validation / Test Split Experiment")
    logger.info("=" * 64)
    logger.info(f"  Splits: {args.n_repeats} random repeats")
    logger.info(f"  Ratio: 60% train / 20% val / 20% test")

    # Load data
    with open(PROJECT_ROOT / "config" / "config.yaml", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    df = load_processed_spectra()
    feat_cols = get_feature_columns(df)
    mc = ModelConfig.from_pipeline_config(raw_cfg, n_spectral_features=len(feat_cols))
    mc = apply_class_selection(mc, cancer_types=CANCER_TYPES, non_cancer_groups=NON_CANCER)
    df = resolve_aliases(df, mc)
    if args.exclude_patients:
        logger.info("Applying patient exclusions...")
        df = apply_patient_exclusions(df, args.exclude_patients)
    df_agg = aggregate_replicates(df, feat_cols, "none")

    # Clinical
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
    clin_valid = ~np.isnan(tier1_filled).any(axis=1)

    # Sex array for biological constraints (derive from group for samples without clinical data)
    sex_from_clinical = tier1[:, 1]  # sex_numeric column
    # For samples missing clinical sex, infer from cancer group: PRO=male, OVA=female
    sex_for_constraint = sex_from_clinical.copy()
    for i, grp in enumerate(groups):
        if np.isnan(sex_for_constraint[i]):
            if grp == "PRO":
                sex_for_constraint[i] = 1.0  # Male
            elif grp == "OVA":
                sex_for_constraint[i] = 0.0  # Female

    logger.info(f"  Total: {len(X)} spectra, {len(set(sample_ids))} subjects")
    logger.info(f"  With clinical: {clin_valid.sum()}")

    # ================================================================
    # Run multiple random splits
    # ================================================================
    all_sers_results = []
    all_fusion_results = []

    for repeat_i in range(args.n_repeats):
        seed = args.seed + repeat_i * 100
        logger.info(f"\n{'='*50}")
        logger.info(f"  Split {repeat_i+1}/{args.n_repeats} (seed={seed})")
        logger.info(f"{'='*50}")

        train_idx, val_idx, test_idx = create_train_val_test_split(
            X, bl, ctl, sample_ids, groups, seed=seed)

        # Count subjects per set
        for name, idx in [("Train", train_idx), ("Val", val_idx), ("Test", test_idx)]:
            n_subj = len(set(sample_ids[idx]))
            n_cancer = bl[idx].sum()
            n_ctrl = (bl[idx] == 0).sum()
            grp_counts = pd.Series(groups[idx]).value_counts()
            logger.info(f"  {name:>5s}: {len(idx):5d} spectra, {n_subj:4d} subjects "
                        f"(cancer={n_cancer}, control={n_ctrl})")

        # ── SERS only ──
        logger.info(f"\n  [SERS only]")
        sers_results, thresh_sers, _, _ = train_and_evaluate(
            X, bl, ctl, groups, train_idx, val_idx, test_idx, mc,
            sex_numeric=sex_for_constraint, label="sers")

        logger.info(f"    Threshold (from val): {thresh_sers:.4f}")
        for s in ["train", "val", "test"]:
            r = sers_results[s]
            logger.info(f"    {s:>5s}: Det AUC={r['det_auc']:.4f} Sens={r['det_sensitivity']:.4f} "
                        f"Spec={r['det_specificity']:.4f} | Id F1={r.get('id_f1_macro', 0):.4f}")

        all_sers_results.append(sers_results)

        # ── Fusion (SERS + age/sex/BMI) ── only on samples with clinical data
        idx_clin = np.where(clin_valid)[0]
        train_clin = np.intersect1d(train_idx, idx_clin)
        val_clin = np.intersect1d(val_idx, idx_clin)
        test_clin = np.intersect1d(test_idx, idx_clin)

        if len(train_clin) > 100 and len(val_clin) > 20 and len(test_clin) > 20:
            logger.info(f"\n  [Fusion (SERS + age/sex/BMI)]")
            logger.info(f"    Train={len(train_clin)} Val={len(val_clin)} Test={len(test_clin)}")
            fusion_results, thresh_fus, _, _ = train_and_evaluate(
                X, bl, ctl, groups, train_clin, val_clin, test_clin, mc,
                clinical_features=tier1_filled, sex_numeric=sex_for_constraint,
                label="fusion")

            for s in ["train", "val", "test"]:
                r = fusion_results[s]
                logger.info(f"    {s:>5s}: Det AUC={r['det_auc']:.4f} Sens={r['det_sensitivity']:.4f} "
                            f"Spec={r['det_specificity']:.4f} | Id F1={r.get('id_f1_macro', 0):.4f}")
            all_fusion_results.append(fusion_results)

    # ================================================================
    # Aggregate results across repeats
    # ================================================================
    logger.info(f"\n{'='*64}")
    logger.info("  AGGREGATED RESULTS (mean +/- std across splits)")
    logger.info(f"{'='*64}")

    def aggregate(results_list, set_name):
        metrics = {}
        for key in ["det_auc", "det_sensitivity", "det_specificity", "det_accuracy", "id_f1_macro", "id_auc"]:
            vals = [r[set_name].get(key, np.nan) for r in results_list if set_name in r]
            vals = [v for v in vals if not np.isnan(v)]
            if vals:
                metrics[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                                "min": float(np.min(vals)), "max": float(np.max(vals))}
        return metrics

    for model_name, results_list in [("SERS only", all_sers_results), ("Fusion", all_fusion_results)]:
        if not results_list:
            continue
        logger.info(f"\n  {model_name} ({len(results_list)} splits):")
        logger.info(f"  {'Set':>5s} {'Det AUC':>16s} {'Sensitivity':>16s} {'Specificity':>16s} {'Id F1':>16s}")
        logger.info("  " + "-" * 75)
        for set_name in ["train", "val", "test"]:
            agg = aggregate(results_list, set_name)
            parts = []
            for key in ["det_auc", "det_sensitivity", "det_specificity", "id_f1_macro"]:
                if key in agg:
                    parts.append(f"{agg[key]['mean']:.3f}+/-{agg[key]['std']:.3f}")
                else:
                    parts.append("  N/A  ")
            logger.info(f"  {set_name:>5s} {'  '.join(parts)}")

    # Per-cancer test sensitivity (from last split for detail)
    if all_sers_results:
        last = all_sers_results[-1]["test"]
        if "per_cancer_sensitivity" in last:
            logger.info(f"\n  Per-cancer TEST sensitivity (last split, SERS only):")
            for ct, info in last["per_cancer_sensitivity"].items():
                logger.info(f"    {ct}: {info['sensitivity']:.3f} (n={info['n']})")

        if "confusion_matrix" in last:
            cm = np.array(last["confusion_matrix"])
            logger.info(f"\n  TEST Confusion Matrix (last split):")
            header = "        " + "  ".join(f"{CANCER_LABELS[i]:>6s}" for i in range(len(CANCER_LABELS)))
            logger.info(f"  {header}")
            for i in range(cm.shape[0]):
                row = "  ".join(f"{cm[i, j]:>6d}" for j in range(cm.shape[1]))
                logger.info(f"  {CANCER_LABELS[i]:<6s}  {row}")

    # ================================================================
    # Plots
    # ================================================================
    logger.info("\n  Generating plots...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # (0,0) Train vs Val vs Test AUC across splits
    ax = axes[0, 0]
    for model_name, results_list, color in [("SERS", all_sers_results, "#1976D2"),
                                             ("Fusion", all_fusion_results, "#9C27B0")]:
        if not results_list:
            continue
        for set_name, marker, alpha in [("train", "o", 0.4), ("val", "s", 0.7), ("test", "D", 1.0)]:
            vals = [r[set_name]["det_auc"] for r in results_list]
            x = range(1, len(vals) + 1)
            ax.scatter(x, vals, marker=marker, color=color, alpha=alpha, s=40,
                       label=f"{model_name} {set_name}" if model_name == "SERS" else None)
    ax.set_xlabel("Split #")
    ax.set_ylabel("Detection AUC")
    ax.set_title("Detection AUC: Train/Val/Test")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.9, 1.01)

    # (0,1) Train vs Val vs Test Id F1
    ax = axes[0, 1]
    for model_name, results_list, color in [("SERS", all_sers_results, "#1976D2"),
                                             ("Fusion", all_fusion_results, "#9C27B0")]:
        if not results_list:
            continue
        for set_name, marker, alpha in [("train", "o", 0.4), ("val", "s", 0.7), ("test", "D", 1.0)]:
            vals = [r[set_name].get("id_f1_macro", 0) for r in results_list]
            x = range(1, len(vals) + 1)
            ax.scatter(x, vals, marker=marker, color=color, alpha=alpha, s=40)
    ax.set_xlabel("Split #")
    ax.set_ylabel("Identification F1 Macro")
    ax.set_title("Identification F1: Train/Val/Test")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.7, 1.01)

    # (0,2) Comparison bar chart (mean test)
    ax = axes[0, 2]
    models = []
    det_aucs = []
    id_f1s = []
    for model_name, results_list in [("SERS\nonly", all_sers_results), ("Fusion\n(+clinical)", all_fusion_results)]:
        if results_list:
            test_auc = np.mean([r["test"]["det_auc"] for r in results_list])
            test_f1 = np.mean([r["test"].get("id_f1_macro", 0) for r in results_list])
            models.append(model_name)
            det_aucs.append(test_auc)
            id_f1s.append(test_f1)

    if models:
        x = np.arange(len(models))
        w = 0.3
        ax.bar(x - w/2, det_aucs, w, label="Det AUC", color="#1976D2", alpha=0.8)
        ax.bar(x + w/2, id_f1s, w, label="Id F1", color="#E53935", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_ylabel("Score")
        ax.set_title("TEST Set Performance\n(mean across splits)")
        ax.legend()
        ax.set_ylim(0.7, 1.02)
        ax.grid(True, alpha=0.3)
        for i in range(len(models)):
            ax.text(i - w/2, det_aucs[i] + 0.005, f"{det_aucs[i]:.3f}", ha="center", fontsize=9, fontweight="bold")
            ax.text(i + w/2, id_f1s[i] + 0.005, f"{id_f1s[i]:.3f}", ha="center", fontsize=9, fontweight="bold")

    # (1,0) Per-cancer test sensitivity (from all splits)
    ax = axes[1, 0]
    if all_sers_results:
        cancer_sens = {ct: [] for ct in CANCER_TYPES}
        for res in all_sers_results:
            pcs = res["test"].get("per_cancer_sensitivity", {})
            for ct in CANCER_TYPES:
                if ct in pcs:
                    cancer_sens[ct].append(pcs[ct]["sensitivity"])

        positions = range(len(CANCER_TYPES))
        bp_data = [cancer_sens[ct] for ct in CANCER_TYPES]
        CANCER_COLORS_LOCAL = {"PRO": "#E91E63", "BRE": "#FF69B4", "OVA": "#AB47BC", "LUN": "#42A5F5",
                               "CRC": "#EF5350", "PAN": "#FFA726", "BLC": "#7E57C2"}
        bplot = ax.boxplot(bp_data, labels=CANCER_TYPES, patch_artist=True, widths=0.5)
        for i, ct in enumerate(CANCER_TYPES):
            bplot["boxes"][i].set_facecolor(CANCER_COLORS_LOCAL[ct])
            bplot["boxes"][i].set_alpha(0.7)
        ax.set_ylabel("Sensitivity")
        ax.set_title(f"TEST: Per-Cancer Detection Sensitivity\n({args.n_repeats} splits)")
        ax.axhline(0.95, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.6, 1.05)

    # (1,1) Overfitting check: train vs test gap
    ax = axes[1, 1]
    if all_sers_results:
        train_aucs = [r["train"]["det_auc"] for r in all_sers_results]
        test_aucs = [r["test"]["det_auc"] for r in all_sers_results]
        train_f1s = [r["train"].get("id_f1_macro", 0) for r in all_sers_results]
        test_f1s = [r["test"].get("id_f1_macro", 0) for r in all_sers_results]

        x = range(1, len(train_aucs) + 1)
        ax.plot(x, train_aucs, "o-", color="#1976D2", alpha=0.5, label="Train Det AUC")
        ax.plot(x, test_aucs, "D-", color="#1976D2", label="Test Det AUC")
        ax.plot(x, train_f1s, "o-", color="#E53935", alpha=0.5, label="Train Id F1")
        ax.plot(x, test_f1s, "D-", color="#E53935", label="Test Id F1")
        ax.set_xlabel("Split #")
        ax.set_ylabel("Score")
        ax.set_title("Overfitting Check\n(Train vs Test gap)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.8, 1.02)

    # (1,2) Confusion matrix (last split test)
    ax = axes[1, 2]
    if all_sers_results and "confusion_matrix" in all_sers_results[-1]["test"]:
        cm = np.array(all_sers_results[-1]["test"]["confusion_matrix"])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)
        im = ax.imshow(cm_norm, cmap="YlOrRd", vmin=0, vmax=1)
        n_cls = cm.shape[0]
        ax.set_xticks(range(n_cls))
        ax.set_yticks(range(n_cls))
        ax.set_xticklabels([CANCER_LABELS[i] for i in range(n_cls)], fontsize=9)
        ax.set_yticklabels([CANCER_LABELS[i] for i in range(n_cls)], fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("TEST: Confusion Matrix (last split)")
        for i in range(n_cls):
            for j in range(n_cls):
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]:.0%})",
                        ha="center", va="center", fontsize=8, color=color)
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle("Train / Validation / Test Split Experiment",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    from src.sers.config import FIG_DIR
    fig_out = FIG_DIR / "training" / exp_name
    fig_out.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_out / "train_val_test_results.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Save summary
    summary = {
        "experiment": "train_val_test",
        "timestamp": datetime.now().isoformat(),
        "n_repeats": args.n_repeats,
        "split_ratio": "60/20/20",
        "sers_results": all_sers_results,
        "fusion_results": all_fusion_results,
        "elapsed": str(datetime.now() - t0),
    }
    with open(out_dir / "train_val_test_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"\n  Elapsed: {datetime.now() - t0}")
    logger.info(f"  Output: {out_dir}/")
    logger.info("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
