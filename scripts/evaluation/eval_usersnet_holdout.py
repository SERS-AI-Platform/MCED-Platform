#!/usr/bin/env python3
"""
Stacking V2 — Holdout Evaluation (Train / Val / Test)

Subject-level stratified split:
  - Train 60% → fit base models + meta-learner (OOF within train)
  - Val   20% → threshold tuning, early stopping proxy
  - Test  20% → final unbiased performance report

Usage:
    python models/eval_stacking_holdout.py
    python models/eval_stacking_holdout.py --split 70 15 15
"""

from __future__ import annotations

import sys
import json
import argparse
import logging
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, roc_curve, classification_report,
    confusion_matrix, precision_recall_fscore_support,
)

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sers.preprocessing import trim_spectrum, baseline_correction, normalize_spectrum, resample
from src.sers.io import find_spectra, read_spectrum, parse_filename
from sers.models.usersnet.stacking import build_classifier
from models.train_stacking import (
    extract_peak_features, train_base_model_ext,
    EXTENDED_BASE_MODELS, KNOWN_PEAKS,
)
from sers.models.usersnet.stacking import apply_sex_constraint, infer_sex_from_groups

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────
CANCER_TYPES = ["PRO", "LUN", "CRC", "PAN", "OVA", "BRE", "BLC"]
NON_CANCER = ["NOR", "DIA", "HBP", "H.D."]
ALL_GROUPS = CANCER_TYPES + NON_CANCER
SG_WINDOW, SG_POLY = 11, 3

THERMO_MAP = {
    "1. Prostate cancer (100개)": "PRO",
    "2. Breast cancer (30개)": "BRE",
    "3. Ovarian cancer (70개)": "OVA",
    "4. Lung cancer (300개)": "LUN",
    "5. Normal (100개)": "NOR",
    "6. Diabetes (100개)": "DIA",
    "7. High blood pressure (100개)": "HBP",
    "8. High blood pressure + Diabetes (100개)": "H.D.",
    "9. Colorectal cancer (300개)": "CRC",
    "10-1. C-Pancreatic cancer (70개)": "CPAN",
    "10-3. Y-Pancreatic cancer (YPAN)": "YPAN",
    "11 BLC (299개)": "BLC",
    "12. Y-Normal (YNOR)": "YNOR",
}
GROUP_ALIASES = {"CPAN": "PAN", "YPAN": "PAN", "YNOR": "NOR"}

OUTPUT_DIR = PROJECT_ROOT / "results" / "training" / "stacking_v2_holdout"
FIG_DIR = PROJECT_ROOT / "figures" / "training" / "stacking_v2_holdout"


# ─────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────
def preprocess_channel(x, y, grid, deriv_order):
    y_sg = savgol_filter(y, SG_WINDOW, SG_POLY, deriv=deriv_order)
    x_tr, y_tr = trim_spectrum(x.copy(), y_sg, region=(400, 2200))
    if deriv_order == 0:
        y_tr = baseline_correction(y_tr, window=101)
    y_tr = normalize_spectrum(y_tr, method="snv")
    return resample(x_tr, y_tr, grid)


def load_data(grid):
    """Load all spectra → 3-channel arrays + metadata with unique subject IDs."""
    data_dir = PROJECT_ROOT / "data" / "raw_data"
    all_X, meta_rows, failed = [], [], 0

    for folder_name, group in THERMO_MAP.items():
        folder = data_dir / folder_name
        if not folder.is_dir():
            continue
        files = find_spectra(folder, pattern="*.CSV", recursive=False)
        if not files:
            for p in ["*.csv", "*.txt"]:
                files = find_spectra(folder, pattern=p, recursive=False)
                if files:
                    break
        files = [f for f in files
                 if "_ave" not in f.stem.lower()
                 and "zone.identifier" not in f.name.lower()]

        for fp in files:
            try:
                sid = parse_filename(fp, fallback_group=group)
                x, y = read_spectrum(fp)
                ch0 = preprocess_channel(x, y, grid, 0)
                ch1 = preprocess_channel(x, y, grid, 1)
                ch2 = preprocess_channel(x, y, grid, 2)
                all_X.append(np.stack([ch0, ch1, ch2], axis=0))

                g = GROUP_ALIASES.get(sid.group, sid.group)
                # Prefix YPAN/YNOR sample_ids to disambiguate from CPAN/NOR
                raw_group = sid.group
                sample_id = sid.sample_id
                if raw_group in ("YPAN", "YNOR"):
                    sample_id = f"{raw_group}_{sample_id}"

                meta_rows.append({
                    "group": g,
                    "sample_id": sample_id,
                    "replicate": sid.replicate,
                })
            except Exception:
                failed += 1

    X = np.stack(all_X, axis=0).astype(np.float32)
    df = pd.DataFrame(meta_rows)
    logger.info(f"Loaded {len(X)} spectra ({failed} failed), shape {X.shape}")
    return X, df


def aggregate_mean(X, df):
    """Mean-aggregate replicates per subject."""
    df = df.copy()
    df["uid"] = df["group"] + "_" + df["sample_id"].astype(str)
    uids = df["uid"].unique()

    X_agg, meta_agg = [], []
    for uid in uids:
        mask = df["uid"] == uid
        X_agg.append(X[mask.values].mean(axis=0))
        row = df[mask].iloc[0]
        meta_agg.append({"group": row["group"], "sample_id": row["sample_id"],
                         "uid": uid})

    return np.stack(X_agg), pd.DataFrame(meta_agg)


def create_labels(df):
    """Create binary + type labels, filter to valid groups."""
    valid = df["group"].isin(ALL_GROUPS)
    idx = np.where(valid.values)[0]
    groups = df.loc[valid, "group"].values
    binary = np.array([1 if g in CANCER_TYPES else 0 for g in groups])
    type_labels = np.full(len(groups), -1)
    for i, g in enumerate(groups):
        if g in CANCER_TYPES:
            type_labels[i] = CANCER_TYPES.index(g)
    uids = df.loc[valid, "uid"].values
    return idx, binary, type_labels, groups, uids


# ─────────────────────────────────────────────────────────────────
# Subject-level stratified split
# ─────────────────────────────────────────────────────────────────
def stratified_subject_split(groups, uids, binary, train_pct, val_pct, seed=42):
    """Split subjects into train/val/test with stratification on group."""
    rng = np.random.RandomState(seed)

    # Get unique subjects
    uid_to_idx = {}
    uid_to_group = {}
    uid_to_binary = {}
    for i, (uid, grp, b) in enumerate(zip(uids, groups, binary)):
        if uid not in uid_to_idx:
            uid_to_idx[uid] = []
            uid_to_group[uid] = grp
            uid_to_binary[uid] = b
        uid_to_idx[uid].append(i)

    unique_uids = np.array(list(uid_to_idx.keys()))
    unique_groups = np.array([uid_to_group[u] for u in unique_uids])

    logger.info(f"Total subjects: {len(unique_uids)}")

    # Stratified split by group
    test_pct = 100 - train_pct - val_pct
    train_uids, val_uids, test_uids = [], [], []

    for grp in sorted(set(unique_groups)):
        grp_mask = unique_groups == grp
        grp_uids = unique_uids[grp_mask].copy()
        rng.shuffle(grp_uids)

        n = len(grp_uids)
        n_train = max(1, int(n * train_pct / 100))
        n_val = max(1, int(n * val_pct / 100))
        # Rest goes to test
        train_uids.extend(grp_uids[:n_train])
        val_uids.extend(grp_uids[n_train:n_train + n_val])
        test_uids.extend(grp_uids[n_train + n_val:])

    train_uids = set(train_uids)
    val_uids = set(val_uids)
    test_uids = set(test_uids)

    train_idx = [i for uid in train_uids for i in uid_to_idx[uid]]
    val_idx = [i for uid in val_uids for i in uid_to_idx[uid]]
    test_idx = [i for uid in test_uids for i in uid_to_idx[uid]]

    return (np.array(sorted(train_idx)),
            np.array(sorted(val_idx)),
            np.array(sorted(test_idx)),
            len(train_uids), len(val_uids), len(test_uids))


# ─────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────
def compute_thresholds(y_true, y_prob):
    """Compute operating thresholds from probabilities."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    # Balanced (Youden's J)
    j_scores = tpr - fpr
    balanced_idx = np.argmax(j_scores)
    balanced_thresh = float(thresholds[balanced_idx])

    # Screening: target Sensitivity >= 95%
    sens_idx = np.argmin(np.abs(tpr - 0.95))
    screening_thresh = float(thresholds[sens_idx])

    # Confirmatory: target Specificity >= 95%
    spec_idx = np.argmin(np.abs((1 - fpr) - 0.95))
    confirmatory_thresh = float(thresholds[spec_idx])

    # IMPORTANT: screening should have LOWER threshold, confirmatory HIGHER
    # If inverted, swap them
    if screening_thresh > confirmatory_thresh:
        logger.warning(f"Thresholds inverted! screening={screening_thresh:.4f} > "
                       f"confirmatory={confirmatory_thresh:.4f}. Swapping.")
        screening_thresh, confirmatory_thresh = confirmatory_thresh, screening_thresh

    return {
        "screening": screening_thresh,
        "balanced": balanced_thresh,
        "confirmatory": confirmatory_thresh,
    }


def eval_split(y_bin, y_type, s1_prob, s2_prob, groups, split_name,
               thresholds=None):
    """Evaluate Stage 1 + Stage 2 on a split."""
    result = {"split": split_name, "n_samples": len(y_bin)}

    # Stage 1: AUC
    auc = roc_auc_score(y_bin, s1_prob)
    result["s1_auc"] = round(auc, 4)

    # Stage 1: per-threshold metrics
    if thresholds:
        for mode, thresh in thresholds.items():
            preds = (s1_prob > thresh).astype(int)
            tp = ((preds == 1) & (y_bin == 1)).sum()
            tn = ((preds == 0) & (y_bin == 0)).sum()
            fp = ((preds == 1) & (y_bin == 0)).sum()
            fn = ((preds == 0) & (y_bin == 1)).sum()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            result[f"s1_{mode}_thresh"] = round(thresh, 4)
            result[f"s1_{mode}_sens"] = round(sens, 4)
            result[f"s1_{mode}_spec"] = round(spec, 4)

    # Stage 2: cancer type F1 (cancer samples only)
    cancer_mask = y_bin == 1
    if cancer_mask.sum() > 0:
        s2_preds = s2_prob[cancer_mask].argmax(axis=1)
        f1_macro = f1_score(y_type[cancer_mask], s2_preds,
                            average="macro", zero_division=0)
        result["s2_f1_macro"] = round(f1_macro, 4)

        # Per-class F1
        prec, rec, f1, sup = precision_recall_fscore_support(
            y_type[cancer_mask], s2_preds,
            labels=list(range(len(CANCER_TYPES))),
            zero_division=0,
        )
        for ci, ct in enumerate(CANCER_TYPES):
            result[f"s2_{ct}_f1"] = round(float(f1[ci]), 4)
            result[f"s2_{ct}_n"] = int(sup[ci])

        # Confusion matrix
        cm = confusion_matrix(y_type[cancer_mask], s2_preds,
                              labels=list(range(len(CANCER_TYPES))))
        result["s2_confusion_matrix"] = cm.tolist()

    return result


def plot_holdout_results(results, thresholds, output_dir):
    """Generate evaluation figures."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Figure 1: AUC + F1 bar chart across splits ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    splits = [r["split"] for r in results]
    aucs = [r["s1_auc"] for r in results]
    f1s = [r.get("s2_f1_macro", 0) for r in results]
    colors = ["#4CAF50", "#FF9800", "#F44336"]

    ax = axes[0]
    bars = ax.bar(splits, aucs, color=colors, alpha=0.85, width=0.5)
    ax.set_ylabel("AUC")
    ax.set_title("Stage 1: Cancer Detection AUC", fontweight="bold")
    ax.set_ylim(0.5, 1.05)
    ax.axhline(0.95, color="red", ls="--", alpha=0.5, label="0.95")
    for b, v in zip(bars, aucs):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.4f}",
                ha="center", fontsize=11, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    bars = ax.bar(splits, f1s, color=colors, alpha=0.85, width=0.5)
    ax.set_ylabel("F1 Macro")
    ax.set_title("Stage 2: Cancer Type F1 Macro", fontweight="bold")
    ax.set_ylim(0.0, 1.05)
    ax.axhline(0.90, color="red", ls="--", alpha=0.5, label="0.90 target")
    for b, v in zip(bars, f1s):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.4f}",
                ha="center", fontsize=11, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "holdout_auc_f1.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 2: Per-threshold Sensitivity/Specificity on val & test ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ai, split_name in enumerate(["Val", "Test"]):
        r = [x for x in results if x["split"] == split_name]
        if not r:
            continue
        r = r[0]
        ax = axes[ai]
        modes = list(thresholds.keys())
        sens_vals = [r.get(f"s1_{m}_sens", 0) for m in modes]
        spec_vals = [r.get(f"s1_{m}_spec", 0) for m in modes]

        x_pos = np.arange(len(modes))
        w = 0.35
        ax.bar(x_pos - w / 2, sens_vals, w, label="Sensitivity", color="#2196F3", alpha=0.85)
        ax.bar(x_pos + w / 2, spec_vals, w, label="Specificity", color="#FF5722", alpha=0.85)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"{m}\n(t={thresholds[m]:.3f})" for m in modes], fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_title(f"{split_name} — Sens/Spec by Mode", fontweight="bold")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        for i in range(len(modes)):
            ax.text(x_pos[i] - w / 2, sens_vals[i] + 0.02, f"{sens_vals[i]:.3f}",
                    ha="center", fontsize=8)
            ax.text(x_pos[i] + w / 2, spec_vals[i] + 0.02, f"{spec_vals[i]:.3f}",
                    ha="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_dir / "holdout_sens_spec.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 3: Test set confusion matrix ──
    test_r = [x for x in results if x["split"] == "Test"]
    if test_r and "s2_confusion_matrix" in test_r[0]:
        cm = np.array(test_r[0]["s2_confusion_matrix"])
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
        ax.set_xticks(range(len(CANCER_TYPES)))
        ax.set_yticks(range(len(CANCER_TYPES)))
        ax.set_xticklabels(CANCER_TYPES, fontsize=10)
        ax.set_yticklabels(CANCER_TYPES, fontsize=10)
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("True", fontsize=12)
        ax.set_title("Test Set — Cancer Type Confusion Matrix", fontweight="bold")
        for i in range(len(CANCER_TYPES)):
            for j in range(len(CANCER_TYPES)):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color=color, fontsize=11, fontweight="bold")
        fig.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        fig.savefig(output_dir / "holdout_confusion_matrix.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)

    logger.info(f"Figures saved to {output_dir}")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", nargs=3, type=int, default=[60, 20, 20],
                        help="Train/Val/Test percentage (default: 60 20 20)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inner-folds", type=int, default=5,
                        help="CV folds within train set for OOF")
    args = parser.parse_args()

    train_pct, val_pct, test_pct = args.split
    assert train_pct + val_pct + test_pct == 100

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    t0 = datetime.now()

    logger.info("=" * 64)
    logger.info("  Stacking V2 — Holdout Evaluation")
    logger.info(f"  Split: Train {train_pct}% / Val {val_pct}% / Test {test_pct}%")
    logger.info("=" * 64)

    # ── Step 1: Load data ──
    grid = np.linspace(402.0, 2198.0, 935)
    X_raw, df_raw = load_data(grid)
    X_agg, df_agg = aggregate_mean(X_raw, df_raw)
    idx, y_bin, y_type, groups, uids = create_labels(df_agg)
    X = X_agg[idx]
    groups = groups  # already filtered
    logger.info(f"After aggregation & filter: {len(X)} subjects "
                f"({y_bin.sum()} cancer, {(y_bin == 0).sum()} non-cancer)")

    # Group distribution
    for g in sorted(set(groups)):
        n = (groups == g).sum()
        logger.info(f"  {g}: {n}")

    # ── Step 2: Subject-level stratified split ──
    train_idx, val_idx, test_idx, n_tr, n_va, n_te = stratified_subject_split(
        groups, uids, y_bin, train_pct, val_pct, seed=args.seed,
    )
    logger.info(f"\nSplit result:")
    logger.info(f"  Train: {len(train_idx)} subjects ({n_tr} unique) "
                f"— cancer {y_bin[train_idx].sum()}, non-cancer {(y_bin[train_idx] == 0).sum()}")
    logger.info(f"  Val:   {len(val_idx)} subjects ({n_va} unique) "
                f"— cancer {y_bin[val_idx].sum()}, non-cancer {(y_bin[val_idx] == 0).sum()}")
    logger.info(f"  Test:  {len(test_idx)} subjects ({n_te} unique) "
                f"— cancer {y_bin[test_idx].sum()}, non-cancer {(y_bin[test_idx] == 0).sum()}")

    X_train, y_bin_train, y_type_train = X[train_idx], y_bin[train_idx], y_type[train_idx]
    X_val, y_bin_val, y_type_val = X[val_idx], y_bin[val_idx], y_type[val_idx]
    X_test, y_bin_test, y_type_test = X[test_idx], y_bin[test_idx], y_type[test_idx]
    groups_train = groups[train_idx]
    groups_val = groups[val_idx]
    groups_test = groups[test_idx]
    uids_train = uids[train_idx]

    n_classes = len(CANCER_TYPES)
    n_base = len(EXTENDED_BASE_MODELS)

    # ── Step 3: Extract peak features ──
    logger.info("\nExtracting peak features (Voigt fitting)...")
    X_peak_train = extract_peak_features(X_train[:, 0, :], grid)
    X_peak_val = extract_peak_features(X_val[:, 0, :], grid)
    X_peak_test = extract_peak_features(X_test[:, 0, :], grid)
    logger.info(f"  Peak features: {X_peak_train.shape[1]} dims")

    # ── Step 4: OOF predictions within train set ──
    logger.info(f"\n--- OOF within train ({args.inner_folds}-fold) ---")
    oof_s1 = np.zeros((len(X_train), n_base))
    oof_s2 = np.zeros((len(X_train), n_base, n_classes))

    cv = StratifiedGroupKFold(n_splits=args.inner_folds, shuffle=True,
                              random_state=args.seed)

    for fold_idx, (tr_i, va_i) in enumerate(cv.split(X_train, y_bin_train, uids_train)):
        logger.info(f"  Inner fold {fold_idx}: train={len(tr_i)}, val={len(va_i)}")
        pk_tr = X_peak_train[tr_i]
        pk_va = X_peak_train[va_i]

        for bi, (name, spec) in enumerate(EXTENDED_BASE_MODELS.items()):
            s1_p, s2_p = train_base_model_ext(
                spec, X_train[tr_i], y_bin_train[tr_i], y_type_train[tr_i],
                X_train[va_i], n_classes,
                X_peak_train=pk_tr, X_peak_val=pk_va,
            )
            oof_s1[va_i, bi] = s1_p
            oof_s2[va_i, bi, :] = s2_p

    # ── Step 5: Train base models on FULL train set ──
    logger.info(f"\n--- Training {n_base} base models on full train set ---")
    base_models = {}
    for name, spec in EXTENDED_BASE_MODELS.items():
        ch = spec["channels"]
        model_type = spec["model"]

        if ch == "peak":
            X_tr = X_peak_train
        else:
            X_tr = X_train[:, ch, :].reshape(len(X_train), -1)

        s1 = build_classifier(model_type, "binary")
        s1.fit(X_tr, y_bin_train)

        cancer_mask = y_bin_train == 1
        s2 = build_classifier(model_type, "multiclass", n_classes)
        s2.fit(X_tr[cancer_mask], y_type_train[cancer_mask])

        base_models[name] = {"s1": s1, "s2": s2, "spec": spec}
        logger.info(f"  {name}: trained")

    # ── Step 6: Train meta-learner on OOF predictions ──
    logger.info("\n--- Training meta-learner on train OOF ---")
    meta_s1_features_train = oof_s1  # (N_train, 10)
    meta_s2_features_train = oof_s2.reshape(len(X_train), -1)  # (N_train, 70)
    meta_all_train = np.hstack([meta_s1_features_train, meta_s2_features_train])

    meta_s1 = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.5, penalty="elasticnet", l1_ratio=0.5,
            max_iter=2000, solver="saga", class_weight="balanced",
        ),
    )
    meta_s1.fit(meta_s1_features_train, y_bin_train)

    cancer_mask_train = y_bin_train == 1
    meta_s2 = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.5, penalty="elasticnet", l1_ratio=0.5,
            max_iter=2000, solver="saga", class_weight="balanced",
            multi_class="multinomial",
        ),
    )
    meta_s2.fit(meta_all_train[cancer_mask_train], y_type_train[cancer_mask_train])

    # ── Step 7: Predict on train (OOF), val, test ──
    logger.info("\n--- Generating predictions ---")

    def predict_split(X_split, X_peak_split):
        """Get base model predictions for a split."""
        n = len(X_split)
        s1_preds = np.zeros((n, n_base))
        s2_preds = np.zeros((n, n_base, n_classes))

        for bi, (name, bm) in enumerate(base_models.items()):
            ch = bm["spec"]["channels"]
            if ch == "peak":
                X_in = X_peak_split
            else:
                X_in = X_split[:, ch, :].reshape(n, -1)

            s1_preds[:, bi] = bm["s1"].predict_proba(X_in)[:, 1]

            raw_s2 = bm["s2"].predict_proba(X_in)
            for i, cls in enumerate(bm["s2"].classes_):
                if cls < n_classes:
                    s2_preds[:, bi, cls] = raw_s2[:, i]

        return s1_preds, s2_preds

    # Val predictions
    val_base_s1, val_base_s2 = predict_split(X_val, X_peak_val)
    val_meta_s1_feat = val_base_s1
    val_meta_all = np.hstack([val_meta_s1_feat, val_base_s2.reshape(len(X_val), -1)])
    val_s1_prob = meta_s1.predict_proba(val_meta_s1_feat)[:, 1]
    val_s2_prob = meta_s2.predict_proba(val_meta_all)
    # Map back to full class indices
    val_s2_full = np.zeros((len(X_val), n_classes))
    s2_classes = meta_s2[-1].classes_ if hasattr(meta_s2[-1], "classes_") else range(n_classes)
    for i, cls in enumerate(s2_classes):
        if cls < n_classes:
            val_s2_full[:, cls] = val_s2_prob[:, i]

    # Test predictions
    test_base_s1, test_base_s2 = predict_split(X_test, X_peak_test)
    test_meta_s1_feat = test_base_s1
    test_meta_all = np.hstack([test_meta_s1_feat, test_base_s2.reshape(len(X_test), -1)])
    test_s1_prob = meta_s1.predict_proba(test_meta_s1_feat)[:, 1]
    test_s2_prob = meta_s2.predict_proba(test_meta_all)
    test_s2_full = np.zeros((len(X_test), n_classes))
    for i, cls in enumerate(s2_classes):
        if cls < n_classes:
            test_s2_full[:, cls] = test_s2_prob[:, i]

    # Train (OOF) predictions through meta
    train_s1_prob = meta_s1.predict_proba(meta_s1_features_train)[:, 1]
    train_s2_prob_raw = meta_s2.predict_proba(meta_all_train)
    train_s2_full = np.zeros((len(X_train), n_classes))
    for i, cls in enumerate(s2_classes):
        if cls < n_classes:
            train_s2_full[:, cls] = train_s2_prob_raw[:, i]

    # ── Step 7b: Apply sex constraint ──
    sex_train = infer_sex_from_groups(groups_train)
    sex_val = infer_sex_from_groups(groups_val)
    sex_test = infer_sex_from_groups(groups_test)
    train_s2_full = apply_sex_constraint(train_s2_full, CANCER_TYPES, sex_train)
    val_s2_full = apply_sex_constraint(val_s2_full, CANCER_TYPES, sex_val)
    test_s2_full = apply_sex_constraint(test_s2_full, CANCER_TYPES, sex_test)
    logger.info("  Applied sex constraint (PRO→mask OVA, OVA→mask PRO)")

    # ── Step 8: Compute thresholds on Val set ──
    logger.info("\n--- Computing thresholds on Val set ---")
    thresholds = compute_thresholds(y_bin_val, val_s1_prob)
    for mode, thresh in thresholds.items():
        logger.info(f"  {mode}: {thresh:.4f}")

    # ── Step 9: Evaluate all splits ──
    logger.info("\n--- Evaluation Results ---")
    results = []

    r_train = eval_split(y_bin_train, y_type_train, train_s1_prob, train_s2_full,
                         groups_train, "Train (OOF→meta)", thresholds)
    results.append(r_train)

    r_val = eval_split(y_bin_val, y_type_val, val_s1_prob, val_s2_full,
                       groups_val, "Val", thresholds)
    results.append(r_val)

    r_test = eval_split(y_bin_test, y_type_test, test_s1_prob, test_s2_full,
                        groups_test, "Test", thresholds)
    results.append(r_test)

    # Print summary
    logger.info(f"\n{'=' * 72}")
    logger.info(f"  HOLDOUT EVALUATION SUMMARY")
    logger.info(f"  Train {train_pct}% ({r_train['n_samples']}) / "
                f"Val {val_pct}% ({r_val['n_samples']}) / "
                f"Test {test_pct}% ({r_test['n_samples']})")
    logger.info(f"{'=' * 72}")
    logger.info(f"  {'Split':<18} {'S1 AUC':>8} {'S2 F1':>8}")
    logger.info(f"  {'-' * 18} {'-' * 8} {'-' * 8}")
    for r in results:
        logger.info(f"  {r['split']:<18} {r['s1_auc']:>8.4f} {r.get('s2_f1_macro', 0):>8.4f}")

    logger.info(f"\n  Operating Thresholds (tuned on Val):")
    for mode, thresh in thresholds.items():
        logger.info(f"    {mode:>14s}: threshold={thresh:.4f}")

    logger.info(f"\n  Test Set — Per-mode Performance:")
    for mode in thresholds:
        sens = r_test.get(f"s1_{mode}_sens", 0)
        spec = r_test.get(f"s1_{mode}_spec", 0)
        logger.info(f"    {mode:>14s}: Sens={sens:.4f}, Spec={spec:.4f}")

    logger.info(f"\n  Test Set — Per-cancer-type F1:")
    for ct in CANCER_TYPES:
        f1 = r_test.get(f"s2_{ct}_f1", 0)
        n = r_test.get(f"s2_{ct}_n", 0)
        logger.info(f"    {ct}: F1={f1:.4f} (n={n})")

    # ── Step 10: Save ──
    report = {
        "timestamp": datetime.now().isoformat(),
        "split_pct": {"train": train_pct, "val": val_pct, "test": test_pct},
        "seed": args.seed,
        "inner_folds": args.inner_folds,
        "n_base_models": n_base,
        "base_model_names": list(EXTENDED_BASE_MODELS.keys()),
        "thresholds": thresholds,
        "results": results,
        "duration_sec": (datetime.now() - t0).total_seconds(),
    }
    report_path = OUTPUT_DIR / "holdout_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"\nReport saved: {report_path}")

    # Figures
    plot_holdout_results(results, thresholds, FIG_DIR)

    logger.info(f"\nTotal time: {(datetime.now() - t0).total_seconds():.0f}s")
    logger.info("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
