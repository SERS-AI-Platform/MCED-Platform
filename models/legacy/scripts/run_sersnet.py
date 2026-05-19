"""
SERS-Net Experiment — Ensemble (80% LR + 20% ResNet18) + Demographics Fusion + Sex Constraint

Architecture:
    - Stage 1 (Cancer Screening):
        LR_fusion trained on SERS + age/sex/BMI → binary prob P_LR
        ResNet18 trained on SERS only → binary prob P_RN
        Blend: P = 0.8 * P_LR + 0.2 * P_RN

    - Stage 2 (Cancer Type ID):
        LR_fusion trained on SERS + age/sex/BMI → type probs P_LR
        ResNet18 trained on SERS only → type logits → softmax → P_RN
        Blend: P = 0.8 * P_LR + 0.2 * P_RN
        Sex constraint: Males → OVA=0, Females → PRO=0, renormalize

    - Evaluation: held-out test (60/20/20) × N repeats

Usage:
    python models/run_sersnet.py
    python models/run_sersnet.py --n-repeats 5 --alpha 0.8
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
import torch
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, confusion_matrix,
    classification_report, roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.model import ModelConfig, SERSDataset, SERSCancerDetector
from models.train import (
    load_processed_spectra, get_feature_columns,
    apply_class_selection, resolve_aliases, aggregate_replicates,
    create_labels, train_fold,
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
    reverse_aliases = {"PAN": ["CPAN", "YPAN"], "NOR": ["NOR", "YNOR"]}
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
            constrained[row_i, ova_idx] = 0.0
            masked = True
        elif sex_arr[row_i] == 0.0 and pro_idx is not None:
            constrained[row_i, pro_idx] = 0.0
            masked = True
        if masked:
            row_sum = constrained[row_i].sum()
            if row_sum > 0:
                constrained[row_i] /= row_sum
    return constrained


def train_lr(X_train, X_val, X_test, bl_train, bl_val, ctl_train, mc, seed=42):
    """Train LR for both stages, return probabilities for all sets."""
    # Stage 1
    s1 = make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, max_iter=1000, solver="saga", class_weight="balanced", random_state=seed))
    s1.fit(X_train, bl_train)
    bp_train = s1.predict_proba(X_train)[:, 1]
    bp_val = s1.predict_proba(X_val)[:, 1]
    bp_test = s1.predict_proba(X_test)[:, 1]

    # Stage 2
    cancer_mask = ctl_train >= 0
    n_ct = mc.n_cancer_types
    s2_probs = {}
    s2_model = None

    if cancer_mask.sum() > 0:
        present = np.unique(ctl_train[cancer_mask])
        if len(present) > 1:
            s2_model = make_pipeline(StandardScaler(), LogisticRegression(
                C=1.0, max_iter=1000, solver="saga", class_weight="balanced", random_state=seed + 100))
            s2_model.fit(X_train[cancer_mask], ctl_train[cancer_mask])

            for name, data in [("train", X_train), ("val", X_val), ("test", X_test)]:
                raw_probs = s2_model.predict_proba(data)
                # Expand to full class space
                full_probs = np.full((len(data), n_ct), 1.0 / n_ct)
                for j, cls in enumerate(s2_model.classes_):
                    full_probs[:, cls] = raw_probs[:, j]
                s2_probs[name] = full_probs

    if not s2_probs:
        for name, data in [("train", X_train), ("val", X_val), ("test", X_test)]:
            s2_probs[name] = np.full((len(data), n_ct), 1.0 / n_ct)

    return {
        "bp": {"train": bp_train, "val": bp_val, "test": bp_test},
        "cp": s2_probs,
        "s1_model": s1,
        "s2_model": s2_model,
    }


def train_resnet(X_train, X_val, bl_train, bl_val, ctl_train, ctl_val,
                 mc, device, seed=42):
    """Train ResNet18-1D, return probabilities for train and val sets."""
    train_ds = SERSDataset(X_train, bl_train, ctl_train, augment=True)
    val_ds = SERSDataset(X_val, bl_val, ctl_val, augment=False)

    runtime = {
        "num_workers": 4 if device.type == "cuda" else 0,
        "pin_memory": device.type == "cuda",
        "prefetch_factor": 2 if device.type == "cuda" else None,
        "use_amp": device.type == "cuda",
    }

    model = SERSCancerDetector(mc).to(device)
    fold_result = train_fold(model, train_ds, val_ds, mc, device, 0, runtime)

    train_m = fold_result["train_metrics"]
    val_m = fold_result["val_metrics"]

    # Convert logits to probabilities
    train_cancer_probs = torch.softmax(torch.tensor(train_m["cancer_logits"]), dim=-1).numpy()
    val_cancer_probs = torch.softmax(torch.tensor(val_m["cancer_logits"]), dim=-1).numpy()

    return {
        "bp": {"train": train_m["binary_prob"], "val": val_m["binary_prob"]},
        "cp": {"train": train_cancer_probs, "val": val_cancer_probs},
        "model": fold_result.get("model", model),
    }


def predict_resnet_on_set(model, X, bl, ctl, device):
    """Run trained ResNet18 on a dataset and return predictions."""
    ds = SERSDataset(X, bl, ctl, augment=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False,
                                          num_workers=0, pin_memory=False)
    model.eval()
    all_bp, all_cl = [], []
    with torch.no_grad():
        for batch in loader:
            spec = batch["spectra"].to(device)
            out = model(spec)
            all_bp.append(out["binary_prob"].cpu().numpy())
            all_cl.append(out["cancer_logits"].cpu().numpy())

    bp = np.concatenate(all_bp).squeeze()
    cl = np.concatenate(all_cl)
    cancer_probs = torch.softmax(torch.tensor(cl), dim=-1).numpy()
    return bp, cancer_probs


def evaluate_sersnet(bp_blend, cp_blend, bl, ctl, groups, threshold, cancer_types, n_ct):
    """Evaluate blended predictions."""
    bpred = (bp_blend > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(bl, bpred, labels=[0, 1]).ravel()

    metrics = {
        "n_samples": len(bl),
        "threshold": threshold,
        "det_auc": float(roc_auc_score(bl, bp_blend)),
        "det_accuracy": float(accuracy_score(bl, bpred)),
        "det_sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0,
        "det_specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0,
        "det_f1": float(f1_score(bl, bpred)),
    }

    # Per-cancer sensitivity
    per_cancer = {}
    for ct_name in cancer_types:
        mask = groups == ct_name
        if mask.sum() > 0:
            per_cancer[ct_name] = {
                "n": int(mask.sum()),
                "sensitivity": float((bp_blend[mask] > threshold).mean()),
            }
    metrics["per_cancer_sensitivity"] = per_cancer

    # Per-control specificity
    per_control = {}
    for ctrl_name in NON_CANCER:
        mask = groups == ctrl_name
        if mask.sum() > 0:
            per_control[ctrl_name] = {
                "n": int(mask.sum()),
                "specificity": float((bp_blend[mask] <= threshold).mean()),
            }
    metrics["per_control_specificity"] = per_control

    # Cancer identification
    cm_mask = ctl >= 0
    if cm_mask.sum() > 0:
        cpred = cp_blend[cm_mask].argmax(axis=1)
        metrics["id_accuracy"] = float(accuracy_score(ctl[cm_mask], cpred))
        metrics["id_f1_macro"] = float(f1_score(ctl[cm_mask], cpred, average="macro", zero_division=0))
        try:
            metrics["id_auc"] = float(roc_auc_score(ctl[cm_mask], cp_blend[cm_mask],
                                                     multi_class="ovr", average="macro"))
        except ValueError:
            metrics["id_auc"] = float("nan")

        # Per-cancer F1
        present_labels = sorted(set(ctl[cm_mask]) | set(cpred))
        report = classification_report(ctl[cm_mask], cpred,
                                       labels=present_labels,
                                       target_names=[CANCER_LABELS.get(i, f"c{i}") for i in present_labels],
                                       output_dict=True, zero_division=0)
        metrics["per_cancer_id"] = {k: v for k, v in report.items() if k in CANCER_LABELS.values()}

        cm_matrix = confusion_matrix(ctl[cm_mask], cpred, labels=list(range(n_ct)))
        metrics["confusion_matrix"] = cm_matrix.tolist()

    return metrics


def main():
    p = argparse.ArgumentParser(description="SERS-Net Experiment")
    p.add_argument("--n-repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=0.8, help="LR weight (1-alpha = ResNet weight)")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    out_dir = PROJECT_ROOT / "results" / "training" / "sersnet"
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(out_dir / "sersnet.log", mode="w", encoding="utf-8")],
    )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    t0 = datetime.now()
    alpha = args.alpha

    logger.info("=" * 64)
    logger.info("  SERS-Net: Ensemble (LR+ResNet18) + Fusion + Sex Constraint")
    logger.info("=" * 64)
    logger.info(f"  Alpha (LR weight): {alpha}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Repeats: {args.n_repeats}")
    logger.info(f"  ResNet18 epochs: {args.epochs}")

    with open(PROJECT_ROOT / "config" / "config.yaml", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    df = load_processed_spectra()
    feat_cols = get_feature_columns(df)
    mc = ModelConfig.from_pipeline_config(raw_cfg, n_spectral_features=len(feat_cols))
    mc = apply_class_selection(mc, cancer_types=CANCER_TYPES, non_cancer_groups=NON_CANCER)
    mc = ModelConfig(**{**mc.__dict__, "n_epochs": args.epochs})
    df = resolve_aliases(df, mc)
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

    # Sex array for constraint
    sex_for_constraint = tier1[:, 1].copy()
    for i, grp in enumerate(groups):
        if np.isnan(sex_for_constraint[i]):
            if grp == "PRO":
                sex_for_constraint[i] = 1.0
            elif grp == "OVA":
                sex_for_constraint[i] = 0.0

    logger.info(f"  Total: {len(X)} spectra, {len(set(sample_ids))} subjects")
    logger.info(f"  With clinical: {clin_valid.sum()}")
    logger.info(f"  Cancer types: {CANCER_TYPES}")
    logger.info(f"  Non-cancer: {NON_CANCER}")

    n_ct = mc.n_cancer_types

    # ================================================================
    # Run experiments
    # ================================================================
    all_sersnet_results = []
    all_lr_sers_results = []   # baseline: LR SERS-only + sex constraint
    all_lr_fusion_results = []  # baseline: LR Fusion + sex constraint

    for repeat_i in range(args.n_repeats):
        seed = args.seed + repeat_i * 100
        logger.info(f"\n{'='*60}")
        logger.info(f"  Repeat {repeat_i+1}/{args.n_repeats} (seed={seed})")
        logger.info(f"{'='*60}")

        train_idx, val_idx, test_idx = create_train_val_test_split(
            X, bl, ctl, sample_ids, groups, seed=seed)

        for name, idx in [("Train", train_idx), ("Val", val_idx), ("Test", test_idx)]:
            n_subj = len(set(sample_ids[idx]))
            logger.info(f"  {name:>5s}: {len(idx):5d} spectra, {n_subj:4d} subjects")

        # Restrict to samples with clinical data for SERS-Net and fusion
        idx_clin = np.where(clin_valid)[0]
        train_clin = np.intersect1d(train_idx, idx_clin)
        val_clin = np.intersect1d(val_idx, idx_clin)
        test_clin = np.intersect1d(test_idx, idx_clin)

        logger.info(f"  Clinical subset: train={len(train_clin)} val={len(val_clin)} test={len(test_clin)}")

        # ── 1. LR SERS-only baseline (with sex constraint) ──
        logger.info(f"\n  [1/3] LR SERS-only + sex constraint")
        lr_sers = train_lr(X[train_idx], X[val_idx], X[test_idx],
                           bl[train_idx], bl[val_idx], ctl[train_idx], mc, seed)

        # Optimize threshold on val
        fpr, tpr, thresholds = roc_curve(bl[val_idx], lr_sers["bp"]["val"])
        best_thresh_sers = float(thresholds[np.argmax(tpr - fpr)])

        # Apply sex constraint to Stage 2
        if lr_sers["s2_model"] is not None:
            s2_classes = lr_sers["s2_model"].classes_ if hasattr(lr_sers["s2_model"], "classes_") else lr_sers["s2_model"][-1].classes_
        else:
            s2_classes = np.arange(n_ct)
        cp_test_sers = apply_sex_constraint(lr_sers["cp"]["test"], s2_classes, CANCER_TYPES, sex_for_constraint[test_idx])

        sers_metrics = evaluate_sersnet(lr_sers["bp"]["test"], cp_test_sers,
                                        bl[test_idx], ctl[test_idx], groups[test_idx],
                                        best_thresh_sers, CANCER_TYPES, n_ct)
        all_lr_sers_results.append(sers_metrics)
        logger.info(f"    Det AUC={sers_metrics['det_auc']:.4f} | Id F1={sers_metrics.get('id_f1_macro',0):.4f}")

        # ── 2. LR Fusion baseline (with sex constraint) ──
        logger.info(f"\n  [2/3] LR Fusion + sex constraint")
        X_fus_train = np.hstack([X[train_clin], tier1_filled[train_clin]])
        X_fus_val = np.hstack([X[val_clin], tier1_filled[val_clin]])
        X_fus_test = np.hstack([X[test_clin], tier1_filled[test_clin]])

        lr_fusion = train_lr(X_fus_train, X_fus_val, X_fus_test,
                             bl[train_clin], bl[val_clin], ctl[train_clin], mc, seed)

        fpr_f, tpr_f, thresholds_f = roc_curve(bl[val_clin], lr_fusion["bp"]["val"])
        best_thresh_fus = float(thresholds_f[np.argmax(tpr_f - fpr_f)])

        if lr_fusion["s2_model"] is not None:
            s2_classes_fus = lr_fusion["s2_model"][-1].classes_ if hasattr(lr_fusion["s2_model"][-1], "classes_") else np.arange(n_ct)
        else:
            s2_classes_fus = np.arange(n_ct)
        cp_test_fus = apply_sex_constraint(lr_fusion["cp"]["test"], s2_classes_fus, CANCER_TYPES, sex_for_constraint[test_clin])

        fusion_metrics = evaluate_sersnet(lr_fusion["bp"]["test"], cp_test_fus,
                                          bl[test_clin], ctl[test_clin], groups[test_clin],
                                          best_thresh_fus, CANCER_TYPES, n_ct)
        all_lr_fusion_results.append(fusion_metrics)
        logger.info(f"    Det AUC={fusion_metrics['det_auc']:.4f} | Id F1={fusion_metrics.get('id_f1_macro',0):.4f}")

        # ── 3. SERS-Net: Ensemble + Fusion + Sex Constraint ──
        logger.info(f"\n  [3/3] SERS-Net (Ensemble + Fusion + Sex Constraint)")
        logger.info(f"    Training ResNet18 on FULL SERS data (no clinical restriction)...")

        # Train ResNet18 on FULL dataset (SERS-only, no demographics needed)
        rn_result = train_resnet(X[train_idx], X[val_idx],
                                 bl[train_idx], bl[val_idx],
                                 ctl[train_idx], ctl[val_idx],
                                 mc, device, seed)

        # Get ResNet18 predictions on clinical-matched test set (for blending with LR fusion)
        rn_model = rn_result["model"]
        rn_bp_test, rn_cp_test = predict_resnet_on_set(
            rn_model, X[test_clin], bl[test_clin], ctl[test_clin], device)

        logger.info(f"    ResNet18 val AUC: {roc_auc_score(bl[val_idx], rn_result['bp']['val']):.4f}")

        # Blend: alpha * LR_fusion + (1-alpha) * ResNet18
        sersnet_bp_test = alpha * lr_fusion["bp"]["test"] + (1 - alpha) * rn_bp_test

        # For Stage 2, blend LR fusion probs + ResNet probs
        sersnet_cp_test = alpha * lr_fusion["cp"]["test"] + (1 - alpha) * rn_cp_test

        # Optimize threshold on blended val predictions
        # ResNet val is on full val set, LR fusion val is on clinical subset
        # Use clinical-matched val for threshold tuning
        rn_bp_val_clin, _ = predict_resnet_on_set(
            rn_model, X[val_clin], bl[val_clin], ctl[val_clin], device)
        sersnet_bp_val = alpha * lr_fusion["bp"]["val"] + (1 - alpha) * rn_bp_val_clin
        fpr_sn, tpr_sn, thresholds_sn = roc_curve(bl[val_clin], sersnet_bp_val)
        best_thresh_sn = float(thresholds_sn[np.argmax(tpr_sn - fpr_sn)])

        # Apply sex constraint
        sersnet_cp_test_constrained = apply_sex_constraint(
            sersnet_cp_test, np.arange(n_ct), CANCER_TYPES, sex_for_constraint[test_clin])

        sersnet_metrics = evaluate_sersnet(sersnet_bp_test, sersnet_cp_test_constrained,
                                           bl[test_clin], ctl[test_clin], groups[test_clin],
                                           best_thresh_sn, CANCER_TYPES, n_ct)
        all_sersnet_results.append(sersnet_metrics)
        logger.info(f"    Det AUC={sersnet_metrics['det_auc']:.4f} | Id F1={sersnet_metrics.get('id_f1_macro',0):.4f}")

    # ================================================================
    # Aggregate results
    # ================================================================
    logger.info(f"\n{'='*64}")
    logger.info("  AGGREGATED RESULTS (mean ± std across splits)")
    logger.info(f"{'='*64}")

    def agg_metric(results_list, key):
        vals = [r.get(key, np.nan) for r in results_list]
        vals = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
        if vals:
            return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        return {"mean": float("nan"), "std": float("nan")}

    summary_table = []
    for model_name, results_list in [
        ("LR SERS + sex const.", all_lr_sers_results),
        ("LR Fusion + sex const.", all_lr_fusion_results),
        ("SERS-Net (Ensemble+Fusion+Sex)", all_sersnet_results),
    ]:
        if not results_list:
            continue
        det_auc = agg_metric(results_list, "det_auc")
        det_sens = agg_metric(results_list, "det_sensitivity")
        det_spec = agg_metric(results_list, "det_specificity")
        id_f1 = agg_metric(results_list, "id_f1_macro")
        id_auc = agg_metric(results_list, "id_auc")

        logger.info(f"\n  {model_name}:")
        logger.info(f"    Det AUC:  {det_auc['mean']:.4f} ± {det_auc['std']:.4f}")
        logger.info(f"    Det Sens: {det_sens['mean']:.4f} ± {det_sens['std']:.4f}")
        logger.info(f"    Det Spec: {det_spec['mean']:.4f} ± {det_spec['std']:.4f}")
        logger.info(f"    Id F1:    {id_f1['mean']:.4f} ± {id_f1['std']:.4f}")
        logger.info(f"    Id AUC:   {id_auc['mean']:.4f} ± {id_auc['std']:.4f}")

        summary_table.append({
            "model": model_name,
            "det_auc": det_auc, "det_sensitivity": det_sens,
            "det_specificity": det_spec, "id_f1_macro": id_f1, "id_auc": id_auc,
        })

    # Per-cancer test breakdown (last SERS-Net split)
    if all_sersnet_results:
        last = all_sersnet_results[-1]
        if "per_cancer_sensitivity" in last:
            logger.info(f"\n  SERS-Net per-cancer TEST sensitivity (last split):")
            for ct, info in last["per_cancer_sensitivity"].items():
                logger.info(f"    {ct}: {info['sensitivity']:.3f} (n={info['n']})")

        if "per_cancer_id" in last:
            logger.info(f"\n  SERS-Net per-cancer TEST F1 (last split):")
            for ct, info in last["per_cancer_id"].items():
                if isinstance(info, dict) and "f1-score" in info:
                    logger.info(f"    {ct}: F1={info['f1-score']:.3f} P={info['precision']:.3f} R={info['recall']:.3f}")

        if "confusion_matrix" in last:
            cm = np.array(last["confusion_matrix"])
            logger.info(f"\n  SERS-Net TEST Confusion Matrix (last split):")
            header = "        " + "  ".join(f"{CANCER_LABELS[i]:>6s}" for i in range(min(cm.shape[0], len(CANCER_LABELS))))
            logger.info(f"  {header}")
            for i in range(cm.shape[0]):
                if i in CANCER_LABELS:
                    row = "  ".join(f"{cm[i, j]:>6d}" for j in range(cm.shape[1]))
                    logger.info(f"  {CANCER_LABELS[i]:<6s}  {row}")

    # ================================================================
    # Save
    # ================================================================
    summary = {
        "experiment": "sersnet",
        "model_name": "SERS-Net",
        "description": "Ensemble (80% LR + 20% ResNet18) + Demographics Fusion + Sex Constraint",
        "timestamp": datetime.now().isoformat(),
        "alpha": alpha,
        "n_repeats": args.n_repeats,
        "cancer_types": CANCER_TYPES,
        "non_cancer_groups": NON_CANCER,
        "split_ratio": "60/20/20",
        "summary_table": summary_table,
        "sersnet_results": all_sersnet_results,
        "lr_sers_results": all_lr_sers_results,
        "lr_fusion_results": all_lr_fusion_results,
        "elapsed": str(datetime.now() - t0),
    }
    with open(out_dir / "sersnet_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"\n  Elapsed: {datetime.now() - t0}")
    logger.info(f"  Output: {out_dir}/")
    logger.info("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
