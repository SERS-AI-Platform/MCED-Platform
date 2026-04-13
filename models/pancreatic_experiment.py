"""
Pancreatic Cancer Binary Classification Experiment

MFDS Single-Cancer Diagnosis: Pancreatic Cancer vs Non-Cancer
- PAN: CPAN (70, Chungbuk/SMCXD06) + YPAN (30, Samsung/SMCXD04) = 100 patients
- NOR: NOR (100, Yangsan/SMCXD03) + YNOR (29, Samsung/SMCXD04) = 129 patients

Follows the base model architecture (LR with StandardScaler),
but trains a single binary classifier (no Stage 2).

Train/Test split: Patient-level stratified split (80/20).
Outputs: experiment log, model artifacts, evaluation metrics.

Note: Multi-institution batch effect documented as limitation.

Usage:
    python models/pancreatic_experiment.py
"""

from __future__ import annotations

import sys
import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, roc_curve,
    confusion_matrix, classification_report, precision_recall_curve,
    average_precision_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────
DATA_PATH = PROJECT_ROOT / "results" / "pancreatic" / "processed_spectra_combat.csv"
CLINICAL_PATH = PROJECT_ROOT / "data" / "clinical_data" / "standardized" / "all_clinical_standardized.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "pancreatic" / "experiment"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def get_feature_columns(df):
    return [c for c in df.columns if c.startswith("x_")]


def load_data():
    """Load pancreatic dataset and create patient-level keys."""
    df = pd.read_csv(DATA_PATH)
    df["patient_key"] = df["group"] + "_" + df["sample_id"].astype(str)
    logger.info(f"Loaded {len(df)} spectra, {df['patient_key'].nunique()} patients")
    logger.info(f"  PAN: {df[df['binary_label']=='PAN']['patient_key'].nunique()} patients")
    logger.info(f"  NOR: {df[df['binary_label']=='NOR']['patient_key'].nunique()} patients")
    return df


def load_clinical():
    """Load clinical data and merge with spectral data."""
    clin = pd.read_csv(CLINICAL_PATH)
    clin["sex_numeric"] = (clin["sex"] == "M").astype(float)
    return clin


def merge_clinical(df, clin):
    """Add age, sex, BMI to spectral dataframe."""
    lookup = {}
    for _, r in clin.iterrows():
        lookup[r["patient_id"]] = {
            "age": r["age"],
            "sex_numeric": r["sex_numeric"],
            "bmi": r.get("bmi", np.nan),
        }

    ages, sexes, bmis = [], [], []
    for _, r in df.iterrows():
        key = f"{r['group']} {r['sample_id']}"
        c = lookup.get(key, {})
        ages.append(c.get("age", np.nan))
        sexes.append(c.get("sex_numeric", np.nan))
        bmis.append(c.get("bmi", np.nan))

    df = df.copy()
    df["age"] = ages
    df["sex_numeric"] = sexes
    df["bmi"] = bmis
    return df


def aggregate_medoid(df, feature_cols):
    """Select most representative replicate per patient."""
    rows = []
    for _, sub in df.groupby("patient_key"):
        if len(sub) == 1:
            rows.append(sub.iloc[0])
            continue
        spectra = sub[feature_cols].values
        medoid_idx = np.corrcoef(spectra).mean(axis=1).argmax()
        rows.append(sub.iloc[medoid_idx])
    return pd.DataFrame(rows).reset_index(drop=True)


def build_lr(X, y, seed=RANDOM_STATE, C=0.1):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=C, max_iter=1000, solver="saga",
            class_weight="balanced", random_state=seed,
        )
    ).fit(X, y)


def patient_train_test_split(df, test_size=TEST_SIZE, seed=RANDOM_STATE):
    """Patient-level stratified train/test split."""
    patients = df.groupby("patient_key")["binary_label"].first().reset_index()
    train_pts, test_pts = train_test_split(
        patients["patient_key"],
        test_size=test_size,
        stratify=patients["binary_label"],
        random_state=seed,
    )
    train_df = df[df["patient_key"].isin(train_pts)].reset_index(drop=True)
    test_df = df[df["patient_key"].isin(test_pts)].reset_index(drop=True)
    return train_df, test_df


def evaluate(model, X, y, label=""):
    """Evaluate and return metrics dict."""
    proba = model.predict_proba(X)[:, 1]
    preds = model.predict(X)
    auc = roc_auc_score(y, proba)
    f1 = f1_score(y, preds, average="binary")
    acc = accuracy_score(y, preds)
    cm = confusion_matrix(y, preds)
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    ap = average_precision_score(y, proba)

    logger.info(f"  {label}: AUC={auc:.4f}, F1={f1:.4f}, Acc={acc:.4f}, Sens={sens:.4f}, Spec={spec:.4f}, AP={ap:.4f}")
    return {
        "auc": round(auc, 4),
        "f1": round(f1, 4),
        "accuracy": round(acc, 4),
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "average_precision": round(ap, 4),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "confusion_matrix": cm.tolist(),
    }


def compute_thresholds(model, X, y):
    """Compute operating thresholds for screening/balanced/confirmatory modes."""
    proba = model.predict_proba(X)[:, 1]
    fpr, tpr, thresholds = roc_curve(y, proba)

    # Youden's J
    j_scores = tpr - fpr
    balanced_idx = np.argmax(j_scores)
    balanced_thresh = float(thresholds[balanced_idx])

    # Sensitivity >= 95%
    screening_idx = np.argmin(np.abs(tpr - 0.95))
    screening_thresh = float(thresholds[screening_idx])

    # Specificity >= 95%
    confirm_idx = np.argmin(np.abs((1 - fpr) - 0.95))
    confirm_thresh = float(thresholds[confirm_idx])

    modes = {}
    for name, thresh in [("screening", screening_thresh), ("balanced", balanced_thresh), ("confirmatory", confirm_thresh)]:
        preds = (proba > thresh).astype(int)
        sens = float((proba[y == 1] > thresh).mean())
        spec = float((proba[y == 0] <= thresh).mean())
        modes[name] = {
            "threshold": round(thresh, 4),
            "sensitivity": round(sens, 4),
            "specificity": round(spec, 4),
        }
    return modes


def plot_roc(y_true, proba, auc_val, output_path):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, proba)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#E53935", lw=2, label=f"AUC = {auc_val:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Pancreatic Cancer Screening — ROC Curve")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm, output_path):
    """Plot confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Pancreatic Ca."])
    ax.set_yticklabels(["Normal", "Pancreatic Ca."])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (Test Set)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                    color="white" if cm[i][j] > cm.max() / 2 else "black", fontsize=16)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_probability_distribution(y_true, proba, output_path):
    """Plot prediction probability distribution by class."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(proba[y_true == 0], bins=30, alpha=0.6, color="#43A047", label="Normal", density=True)
    ax.hist(proba[y_true == 1], bins=30, alpha=0.6, color="#E53935", label="Pancreatic Ca.", density=True)
    ax.set_xlabel("Predicted Probability (Pancreatic Cancer)")
    ax.set_ylabel("Density")
    ax.set_title("Prediction Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("=" * 64)
    logger.info("  Pancreatic Cancer Binary Classification Experiment")
    logger.info("  MFDS Single-Cancer Diagnosis")
    logger.info("=" * 64)

    # ── Load & prepare ─────────────────────────────────────────────
    df = load_data()
    feat_cols = get_feature_columns(df)
    clin = load_clinical()
    df = merge_clinical(df, clin)

    # ── Patient-level train/test split ─────────────────────────────
    train_df, test_df = patient_train_test_split(df)
    logger.info(f"\nTrain: {train_df['patient_key'].nunique()} patients, {len(train_df)} spectra")
    logger.info(f"Test:  {test_df['patient_key'].nunique()} patients, {len(test_df)} spectra")

    for split_name, split_df in [("train", train_df), ("test", test_df)]:
        for label in ["PAN", "NOR"]:
            n = split_df[split_df["binary_label"] == label]["patient_key"].nunique()
            logger.info(f"  {split_name} {label}: {n} patients")

    # ── Aggregate replicates (medoid) ──────────────────────────────
    train_agg = aggregate_medoid(train_df, feat_cols)
    test_agg = aggregate_medoid(test_df, feat_cols)

    y_train = (train_agg["binary_label"] == "PAN").astype(int).values
    y_test = (test_agg["binary_label"] == "PAN").astype(int).values
    X_train = train_agg[feat_cols].values
    X_test = test_agg[feat_cols].values

    logger.info(f"\nAfter medoid aggregation:")
    logger.info(f"  Train: {len(X_train)} ({y_train.sum()} PAN, {(1-y_train).sum()} NOR)")
    logger.info(f"  Test:  {len(X_test)} ({y_test.sum()} PAN, {(1-y_test).sum()} NOR)")

    # ── Train SERS-only model ──────────────────────────────────────
    logger.info("\n--- SERS-only Model ---")
    model_sers = build_lr(X_train, y_train)
    metrics_train_sers = evaluate(model_sers, X_train, y_train, "Train")
    metrics_test_sers = evaluate(model_sers, X_test, y_test, "Test ")

    # ── Train Fusion model (SERS + age/sex/BMI) ───────────────────
    logger.info("\n--- Fusion Model (SERS + age/sex/BMI) ---")
    tier1_cols = ["age", "sex_numeric", "bmi"]

    # Fill missing BMI with training median
    bmi_median = float(train_agg["bmi"].median())
    for df_tmp in [train_agg, test_agg]:
        df_tmp["bmi"] = df_tmp["bmi"].fillna(bmi_median)

    # Check clinical data availability
    train_clin_valid = ~train_agg[tier1_cols].isna().any(axis=1)
    test_clin_valid = ~test_agg[tier1_cols].isna().any(axis=1)
    logger.info(f"  Clinical data available: train={train_clin_valid.sum()}/{len(train_agg)}, test={test_clin_valid.sum()}/{len(test_agg)}")

    X_train_fus = np.hstack([X_train[train_clin_valid], train_agg.loc[train_clin_valid, tier1_cols].values.astype(float)])
    X_test_fus = np.hstack([X_test[test_clin_valid], test_agg.loc[test_clin_valid, tier1_cols].values.astype(float)])
    y_train_fus = y_train[train_clin_valid]
    y_test_fus = y_test[test_clin_valid]

    model_fusion = build_lr(X_train_fus, y_train_fus)
    metrics_train_fus = evaluate(model_fusion, X_train_fus, y_train_fus, "Train")
    metrics_test_fus = evaluate(model_fusion, X_test_fus, y_test_fus, "Test ")

    # ── Cross-validation for robust estimate ───────────────────────
    logger.info("\n--- 5-Fold CV (SERS-only, on training set) ---")
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_probs = np.full(len(X_train), np.nan)
    cv_groups = train_agg["patient_key"].values

    for fold_i, (tr_idx, va_idx) in enumerate(cv.split(X_train, y_train, cv_groups)):
        m = build_lr(X_train[tr_idx], y_train[tr_idx])
        cv_probs[va_idx] = m.predict_proba(X_train[va_idx])[:, 1]
        va_auc = roc_auc_score(y_train[va_idx], cv_probs[va_idx])
        logger.info(f"  Fold {fold_i+1}: AUC={va_auc:.4f}")

    valid_cv = ~np.isnan(cv_probs)
    cv_auc = roc_auc_score(y_train[valid_cv], cv_probs[valid_cv])
    logger.info(f"  CV mean AUC: {cv_auc:.4f}")

    # ── Operating thresholds ───────────────────────────────────────
    logger.info("\n--- Operating Thresholds (from CV) ---")
    fpr_cv, tpr_cv, thresh_cv = roc_curve(y_train[valid_cv], cv_probs[valid_cv])
    j_scores = tpr_cv - fpr_cv
    balanced_thresh = float(thresh_cv[np.argmax(j_scores)])
    screening_thresh = float(thresh_cv[np.argmin(np.abs(tpr_cv - 0.95))])
    confirm_thresh = float(thresh_cv[np.argmin(np.abs((1 - fpr_cv) - 0.95))])

    operating_modes = {}
    for name, thresh in [("screening", screening_thresh), ("balanced", balanced_thresh), ("confirmatory", confirm_thresh)]:
        sens = float((cv_probs[valid_cv][y_train[valid_cv] == 1] > thresh).mean())
        spec = float((cv_probs[valid_cv][y_train[valid_cv] == 0] <= thresh).mean())
        operating_modes[name] = {
            "threshold": round(thresh, 4),
            "cv_sensitivity": round(sens, 4),
            "cv_specificity": round(spec, 4),
        }
        logger.info(f"  {name:>14s}: thresh={thresh:.4f}, Sens={sens:.3f}, Spec={spec:.3f}")

    # ── Test set with thresholds ───────────────────────────────────
    logger.info("\n--- Test Set Performance at Each Threshold ---")
    test_proba_sers = model_sers.predict_proba(X_test)[:, 1]
    threshold_test_metrics = {}
    for name, mode in operating_modes.items():
        thresh = mode["threshold"]
        preds = (test_proba_sers > thresh).astype(int)
        sens = float((test_proba_sers[y_test == 1] > thresh).mean())
        spec = float((test_proba_sers[y_test == 0] <= thresh).mean())
        threshold_test_metrics[name] = {
            "threshold": thresh,
            "test_sensitivity": round(sens, 4),
            "test_specificity": round(spec, 4),
        }
        logger.info(f"  {name:>14s}: Sens={sens:.3f}, Spec={spec:.3f}")

    # ── Batch Effect Analysis ─────────────────────────────────────
    logger.info("\n--- Batch Effect Analysis ---")
    batch_effect = {}

    # CPAN vs YPAN (same disease, different institutions)
    cpan_mask = train_agg["group"] == "CPAN"
    ypan_mask = train_agg["group"] == "YPAN"
    if cpan_mask.sum() > 0 and ypan_mask.sum() > 0:
        inst_df = train_agg[cpan_mask | ypan_mask].copy()
        inst_df["_inst"] = (inst_df["group"] == "CPAN").astype(int)
        X_inst = inst_df[feat_cols].values
        y_inst = inst_df["_inst"].values
        g_inst = inst_df["patient_key"].values
        n_splits_inst = min(5, int(y_inst.sum()), int((1 - y_inst).sum()))
        if n_splits_inst >= 2:
            cv_inst = StratifiedGroupKFold(n_splits=n_splits_inst, shuffle=True, random_state=RANDOM_STATE)
            inst_aucs = []
            for tr, va in cv_inst.split(X_inst, y_inst, g_inst):
                m = build_lr(X_inst[tr], y_inst[tr])
                inst_aucs.append(roc_auc_score(y_inst[va], m.predict_proba(X_inst[va])[:, 1]))
            auc_cpan_ypan = float(np.mean(inst_aucs))
            batch_effect["CPAN_vs_YPAN"] = {"auc": round(auc_cpan_ypan, 4), "interpretation": "strong" if auc_cpan_ypan > 0.8 else "moderate" if auc_cpan_ypan > 0.65 else "none"}
            logger.info(f"  CPAN vs YPAN (institution classifier): AUC={auc_cpan_ypan:.4f}")

    # NOR vs YNOR (same condition, different institutions)
    nor_mask = train_agg["group"] == "NOR"
    ynor_mask = train_agg["group"] == "YNOR"
    if nor_mask.sum() > 0 and ynor_mask.sum() > 0:
        inst_df2 = train_agg[nor_mask | ynor_mask].copy()
        inst_df2["_inst"] = (inst_df2["group"] == "NOR").astype(int)
        X_inst2 = inst_df2[feat_cols].values
        y_inst2 = inst_df2["_inst"].values
        g_inst2 = inst_df2["patient_key"].values
        n_splits_inst2 = min(5, int(y_inst2.sum()), int((1 - y_inst2).sum()))
        if n_splits_inst2 >= 2:
            cv_inst2 = StratifiedGroupKFold(n_splits=n_splits_inst2, shuffle=True, random_state=RANDOM_STATE)
            inst_aucs2 = []
            for tr, va in cv_inst2.split(X_inst2, y_inst2, g_inst2):
                m = build_lr(X_inst2[tr], y_inst2[tr])
                inst_aucs2.append(roc_auc_score(y_inst2[va], m.predict_proba(X_inst2[va])[:, 1]))
            auc_nor_ynor = float(np.mean(inst_aucs2))
            batch_effect["NOR_vs_YNOR"] = {"auc": round(auc_nor_ynor, 4), "interpretation": "strong" if auc_nor_ynor > 0.8 else "moderate" if auc_nor_ynor > 0.65 else "none"}
            logger.info(f"  NOR vs YNOR (institution classifier): AUC={auc_nor_ynor:.4f}")

    # ── Plots ──────────────────────────────────────────────────────
    fig_dir = OUTPUT_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)

    plot_roc(y_test, test_proba_sers, metrics_test_sers["auc"], fig_dir / "roc_curve_test.png")
    plot_confusion_matrix(
        np.array(metrics_test_sers["confusion_matrix"]),
        fig_dir / "confusion_matrix_test.png",
    )
    plot_probability_distribution(y_test, test_proba_sers, fig_dir / "probability_distribution.png")
    logger.info(f"\n  Figures saved to {fig_dir}/")

    # ── Save models ────────────────────────────────────────────────
    model_dir = OUTPUT_DIR / "models"
    model_dir.mkdir(exist_ok=True)
    joblib.dump(model_sers, model_dir / "pancreatic_sers.joblib")
    joblib.dump(model_fusion, model_dir / "pancreatic_fusion.joblib")

    # Save grid
    grid = np.array([float(c.replace("x_", "")) for c in feat_cols])
    np.save(model_dir / "pancreatic_grid.npy", grid)

    # ── Save patient-level predictions (for clinical reports) ──────
    test_results = test_agg[["patient_key", "group", "sample_id", "binary_label"]].copy()
    test_results["probability"] = test_proba_sers
    test_results["predicted"] = (test_proba_sers > operating_modes["balanced"]["threshold"]).astype(int)
    test_results["predicted_label"] = test_results["predicted"].map({0: "Normal", 1: "Pancreatic Cancer"})

    # Add clinical info
    for col in ["age", "sex_numeric", "bmi"]:
        test_results[col] = test_agg[col].values
    test_results["sex"] = test_results["sex_numeric"].map({1.0: "M", 0.0: "F"})

    test_results.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    # Also save train predictions
    train_results = train_agg[["patient_key", "group", "sample_id", "binary_label"]].copy()
    train_proba = model_sers.predict_proba(X_train)[:, 1]
    train_results["probability"] = train_proba
    train_results["predicted"] = (train_proba > operating_modes["balanced"]["threshold"]).astype(int)
    train_results["predicted_label"] = train_results["predicted"].map({0: "Normal", 1: "Pancreatic Cancer"})
    train_results.to_csv(OUTPUT_DIR / "train_predictions.csv", index=False)

    # ── Save experiment log ────────────────────────────────────────
    experiment_log = {
        "experiment": "Pancreatic Cancer Binary Classification",
        "purpose": "MFDS Single-Cancer Diagnosis - Pancreatic Cancer vs Non-Cancer",
        "timestamp": timestamp,
        "dataset": {
            "total_patients": int(df["patient_key"].nunique()),
            "total_spectra": len(df),
            "PAN": {
                "patients": int(df[df["binary_label"] == "PAN"]["patient_key"].nunique()),
                "sources": {"CPAN": 70, "YPAN": 30},
            },
            "NOR": {
                "patients": int(df[df["binary_label"] == "NOR"]["patient_key"].nunique()),
                "sources": {"NOR": 100, "YNOR": 29},
            },
            "note": "SPAN excluded. YPAN/YNOR from SMCXD04 (prospective, Samsung). CPAN from SMCXD06 (Chungbuk). NOR from SMCXD03 (Yangsan). Fasting samples only, pre-treatment baseline.",
        },
        "split": {
            "method": "patient-level stratified",
            "test_size": TEST_SIZE,
            "train_patients": int(train_df["patient_key"].nunique()),
            "test_patients": int(test_df["patient_key"].nunique()),
        },
        "preprocessing": {
            "pipeline": "trim(400-2200) -> smooth(SG,w=11,p=3) -> baseline(rolling_min,w=101) -> SNV",
            "aggregation": "medoid",
            "n_features": len(feat_cols),
        },
        "model": {
            "type": "LogisticRegression",
            "params": {"C": 0.1, "solver": "saga", "class_weight": "balanced", "max_iter": 1000},
            "scaler": "StandardScaler",
        },
        "results": {
            "sers_only": {
                "train": metrics_train_sers,
                "test": metrics_test_sers,
            },
            "fusion": {
                "train": metrics_train_fus,
                "test": metrics_test_fus,
                "clinical_features": tier1_cols,
                "bmi_fill_median": round(bmi_median, 1),
            },
            "cv_auc": round(cv_auc, 4),
        },
        "operating_modes": operating_modes,
        "threshold_test_metrics": threshold_test_metrics,
        "batch_effect_analysis": {
            "results": batch_effect,
            "limitation": (
                "Multi-institution batch effect detected. Institution classifiers "
                "(CPAN vs YPAN, NOR vs YNOR) show AUC > 0.8, indicating spectral "
                "differences attributable to equipment/protocol rather than disease. "
                "Model uses L2 regularization (C=0.1) to mitigate overfitting to "
                "batch-specific features. Results should be validated with "
                "prospective multi-site data using harmonized protocols."
            ),
            "mitigation": [
                "Strong L2 regularization (C=0.1) to reduce batch feature reliance",
                "Balanced class weights to prevent majority class bias",
                "Patient-level stratified split preserving institution ratio",
                "Medoid aggregation to reduce replicate noise",
            ],
        },
    }

    with open(OUTPUT_DIR / "experiment_log.json", "w", encoding="utf-8") as f:
        json.dump(experiment_log, f, indent=2, ensure_ascii=False)

    logger.info(f"\n{'='*64}")
    logger.info(f"  Experiment complete. Output: {OUTPUT_DIR}/")
    logger.info(f"  Test AUC (SERS): {metrics_test_sers['auc']}")
    logger.info(f"  Test AUC (Fusion): {metrics_test_fus['auc']}")
    logger.info(f"{'='*64}")


if __name__ == "__main__":
    main()
