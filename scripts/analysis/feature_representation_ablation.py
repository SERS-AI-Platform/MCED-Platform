"""
Feature Representation Ablation Study

Compares 4 feature extraction methods × 3 models:
  (a) Full spectrum (current baseline) — 935 raw intensity points
  (b) 1st derivative — rate of change at each point
  (c) 2nd derivative — curvature, enhances peak resolution
  (d) Peak features — area, height, FWHM for detected peaks

Models: LR, Random Forest, XGBoost
Evaluation: 5-fold StratifiedGroupKFold CV, mean aggregation

Usage:
    python scripts/analysis/feature_representation_ablation.py
"""

from __future__ import annotations

import sys
import logging
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter, peak_widths
from scipy.integrate import trapezoid
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, classification_report, confusion_matrix,
)
from sklearn.pipeline import make_pipeline

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.sers.config import RESULTS_DIR

logger = logging.getLogger(__name__)

# =============================================================================
# 1. Data Loading (reuse train.py conventions)
# =============================================================================

def load_data(csv_path=None):
    if csv_path is None:
        csv_path = RESULTS_DIR / "processed_spectra.csv"
    df = pd.read_csv(csv_path)
    feat_cols = [c for c in df.columns if c.startswith("x_")]
    wavenumbers = np.array([float(c.replace("x_", "")) for c in feat_cols])
    return df, feat_cols, wavenumbers


def aggregate_mean(df, feat_cols):
    meta_cols = ["group", "sample_id"]
    return df.groupby(meta_cols, as_index=False).agg(
        {**{c: "mean" for c in feat_cols}, "replicate": "count"}
    )


# =============================================================================
# 2. Feature Extraction Methods
# =============================================================================

def extract_full_spectrum(X, wavenumbers):
    """(a) Current baseline: raw intensity at each point."""
    return X, [f"x_{w:.2f}" for w in wavenumbers]


def extract_1st_derivative(X, wavenumbers):
    """(b) 1st derivative: dI/dν using Savitzky-Golay."""
    # SG derivative: window=11, poly=3, deriv=1
    dX = np.apply_along_axis(
        lambda y: savgol_filter(y, window_length=11, polyorder=3, deriv=1),
        axis=1, arr=X,
    )
    names = [f"d1_{w:.2f}" for w in wavenumbers]
    return dX, names


def extract_2nd_derivative(X, wavenumbers):
    """(c) 2nd derivative: d²I/dν² — resolves overlapping peaks."""
    d2X = np.apply_along_axis(
        lambda y: savgol_filter(y, window_length=11, polyorder=3, deriv=2),
        axis=1, arr=X,
    )
    names = [f"d2_{w:.2f}" for w in wavenumbers]
    return d2X, names


def extract_peak_features(X, wavenumbers):
    """(d) Peak-based features: detect peaks, extract area/height/FWHM.

    Uses a consensus approach:
    1. Find peaks in the mean spectrum to define peak positions
    2. For each sample, extract features at those positions
    """
    mean_spectrum = X.mean(axis=0)

    # Detect peaks in mean spectrum with moderate prominence
    prominence = np.ptp(mean_spectrum) * 0.03  # 3% of range
    peaks, props = find_peaks(
        mean_spectrum,
        distance=10,  # ~20 cm⁻¹ minimum separation
        prominence=prominence,
    )

    if len(peaks) == 0:
        # Fallback: take top 20 highest points
        peaks = np.argsort(mean_spectrum)[-20:]
        peaks = np.sort(peaks)

    logger.info(f"  Detected {len(peaks)} peaks in mean spectrum")
    for i, p in enumerate(peaks):
        logger.info(f"    Peak {i+1}: {wavenumbers[p]:.1f} cm⁻¹ (intensity={mean_spectrum[p]:.4f})")

    # Define integration windows around each peak
    # Use peak_widths to estimate FWHM from mean spectrum
    widths_result = peak_widths(mean_spectrum, peaks, rel_height=0.5)
    fwhm_points = widths_result[0]  # width in points

    feature_names = []
    all_features = []

    for i, (peak_idx, fwhm_pt) in enumerate(zip(peaks, fwhm_points)):
        wn = wavenumbers[peak_idx]
        # Integration window: ±2×FWHM (points), minimum ±3 points
        half_window = max(int(np.ceil(fwhm_pt)), 3)
        lo = max(0, peak_idx - half_window)
        hi = min(len(wavenumbers), peak_idx + half_window + 1)

        # Extract per-sample features
        heights = X[:, peak_idx]  # peak height
        areas = np.array([
            trapezoid(X[s, lo:hi], wavenumbers[lo:hi])
            for s in range(len(X))
        ])
        # FWHM per sample (approximate: interpolated)
        sample_fwhms = []
        for s in range(len(X)):
            try:
                w = peak_widths(X[s], [peak_idx], rel_height=0.5)
                # Convert points to cm⁻¹
                dw = np.mean(np.diff(wavenumbers[lo:hi])) if hi - lo > 1 else 1.93
                sample_fwhms.append(w[0][0] * abs(dw))
            except Exception:
                sample_fwhms.append(fwhm_pt * 1.93)
        sample_fwhms = np.array(sample_fwhms)

        feature_names.extend([
            f"peak_{wn:.0f}_height",
            f"peak_{wn:.0f}_area",
            f"peak_{wn:.0f}_fwhm",
        ])
        all_features.extend([heights, areas, sample_fwhms])

    # Add peak ratios for key pairs (if enough peaks)
    if len(peaks) >= 2:
        for i in range(len(peaks)):
            for j in range(i + 1, min(i + 3, len(peaks))):  # adjacent pairs only
                wn_i = wavenumbers[peaks[i]]
                wn_j = wavenumbers[peaks[j]]
                ratio = X[:, peaks[i]] / (X[:, peaks[j]] + 1e-10)
                feature_names.append(f"ratio_{wn_i:.0f}_{wn_j:.0f}")
                all_features.append(ratio)

    X_peak = np.column_stack(all_features)
    return X_peak, feature_names


FEATURE_METHODS = {
    "full_spectrum": extract_full_spectrum,
    "1st_derivative": extract_1st_derivative,
    "2nd_derivative": extract_2nd_derivative,
    "peak_features": extract_peak_features,
}


# =============================================================================
# 3. Models
# =============================================================================

def build_models(random_state=42):
    models = {
        "LR": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0, max_iter=1000, solver="saga",
                class_weight="balanced", random_state=random_state,
            ),
        ),
        "RF": make_pipeline(
            StandardScaler(),
            RandomForestClassifier(
                n_estimators=300, max_depth=None,
                class_weight="balanced", random_state=random_state,
                n_jobs=4,
            ),
        ),
    }
    if HAS_XGB:
        models["XGB"] = make_pipeline(
            StandardScaler(),
            XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                n_jobs=4, tree_method="hist", verbosity=0,
                random_state=random_state, eval_metric="mlogloss",
            ),
        )
    return models


# =============================================================================
# 4. Evaluation (matches train.py two-stage protocol)
# =============================================================================

CANCER_TYPES = ["PRO", "BRE", "OVA", "LUN", "CRC", "PAN", "BLC"]
NON_CANCER = ["NOR", "DIA", "TBN"]


def run_cv(X, groups, sample_ids, feature_name, model_name, model_factory,
           n_splits=5, random_state=42):
    """Run two-stage CV evaluation."""
    # Labels
    cancer_set = set(CANCER_TYPES)
    binary_labels = np.array([1 if g in cancer_set else 0 for g in groups])
    cancer_type_labels = np.array([
        CANCER_TYPES.index(g) if g in cancer_set else -1 for g in groups
    ])

    valid_mask = np.isin(groups, CANCER_TYPES + NON_CANCER)
    X = X[valid_mask]
    groups = groups[valid_mask]
    sample_ids = sample_ids[valid_mask]
    binary_labels = binary_labels[valid_mask]
    cancer_type_labels = cancer_type_labels[valid_mask]

    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_det_aucs = []
    fold_type_f1s = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, binary_labels, groups=sample_ids)):
        X_train, X_test = X[train_idx], X[test_idx]
        yb_train, yb_test = binary_labels[train_idx], binary_labels[test_idx]
        yc_train, yc_test = cancer_type_labels[train_idx], cancer_type_labels[test_idx]

        # Stage 1: Binary (cancer vs non-cancer)
        model_s1 = model_factory()
        model_s1.fit(X_train, yb_train)

        if hasattr(model_s1, "predict_proba"):
            s1_prob = model_s1.predict_proba(X_test)[:, 1]
        else:
            s1_prob = model_s1.decision_function(X_test)

        try:
            det_auc = roc_auc_score(yb_test, s1_prob)
        except ValueError:
            det_auc = np.nan
        fold_det_aucs.append(det_auc)

        # Stage 2: Cancer type ID (cancer samples only)
        cancer_train = yc_train >= 0
        cancer_test = yc_test >= 0

        if cancer_train.sum() > 0 and cancer_test.sum() > 0:
            present_classes = sorted(set(yc_train[cancer_train]))
            if len(present_classes) >= 2:
                # Remap labels to 0..N-1 for XGBoost compatibility
                label_map = {c: i for i, c in enumerate(present_classes)}
                inv_map = {i: c for c, i in label_map.items()}
                yc_train_local = np.array([label_map[c] for c in yc_train[cancer_train]])
                yc_test_local = np.array([label_map.get(c, -1) for c in yc_test[cancer_test]])
                # Exclude test samples with unseen classes
                seen_mask = yc_test_local >= 0
                if seen_mask.sum() == 0:
                    fold_type_f1s.append(np.nan)
                    continue
                model_s2 = model_factory()
                model_s2.fit(X_train[cancer_train], yc_train_local)
                yc_pred_local = model_s2.predict(X_test[cancer_test][seen_mask])
                # Map back for F1
                yc_pred = np.array([inv_map[p] for p in yc_pred_local])
                yc_true = np.array([inv_map[t] for t in yc_test_local[seen_mask]])
                type_f1 = f1_score(
                    yc_true, yc_pred,
                    average="macro", zero_division=0,
                )
            else:
                type_f1 = np.nan
        else:
            type_f1 = np.nan
        fold_type_f1s.append(type_f1)

    return {
        "feature": feature_name,
        "model": model_name,
        "n_features": X.shape[1],
        "det_auc_mean": np.nanmean(fold_det_aucs),
        "det_auc_std": np.nanstd(fold_det_aucs),
        "type_f1_mean": np.nanmean(fold_type_f1s),
        "type_f1_std": np.nanstd(fold_type_f1s),
        "fold_det_aucs": fold_det_aucs,
        "fold_type_f1s": fold_type_f1s,
    }


# =============================================================================
# 5. Main
# =============================================================================

def main():
    t0 = datetime.now()
    out_dir = RESULTS_DIR / "feature_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(out_dir / "ablation.log", mode="w", encoding="utf-8"),
        ],
    )

    logger.info("=" * 64)
    logger.info("  Feature Representation Ablation Study")
    logger.info("=" * 64)

    # Load data
    logger.info("\n[1] Loading data...")
    df, feat_cols, wavenumbers = load_data()

    # Mean aggregation (best known)
    logger.info("\n[2] Aggregating replicates (mean)...")
    df_agg = aggregate_mean(df, feat_cols)

    X_raw = df_agg[feat_cols].values
    groups = df_agg["group"].values
    sample_ids = df_agg["sample_id"].values

    logger.info(f"  Samples: {len(X_raw)}, Wavenumber range: {wavenumbers[0]:.1f}–{wavenumbers[-1]:.1f} cm⁻¹")

    # Extract features for each method
    logger.info("\n[3] Extracting features...")
    feature_sets = {}
    for name, method in FEATURE_METHODS.items():
        logger.info(f"\n  --- {name} ---")
        X_feat, feat_names = method(X_raw, wavenumbers)
        feature_sets[name] = (X_feat, feat_names)
        logger.info(f"  Shape: {X_feat.shape} ({len(feat_names)} features)")

        # Check for NaN/Inf
        n_nan = np.isnan(X_feat).sum()
        n_inf = np.isinf(X_feat).sum()
        if n_nan > 0 or n_inf > 0:
            logger.warning(f"  ⚠ NaN={n_nan}, Inf={n_inf} — replacing with 0")
            X_feat = np.nan_to_num(X_feat, nan=0.0, posinf=0.0, neginf=0.0)
            feature_sets[name] = (X_feat, feat_names)

    # Run all combinations
    logger.info("\n[4] Running CV experiments...")
    results = []

    for feat_name, (X_feat, _) in feature_sets.items():
        models = build_models()
        for model_name, model_pipeline in models.items():
            logger.info(f"\n  {feat_name} × {model_name}...")

            # Create factory that returns fresh model each call
            def make_model(mp=model_pipeline):
                from sklearn.base import clone
                return clone(mp)

            result = run_cv(
                X_feat, groups, sample_ids,
                feat_name, model_name, make_model,
            )
            results.append(result)
            logger.info(
                f"    Det AUC: {result['det_auc_mean']:.4f}±{result['det_auc_std']:.4f} | "
                f"Type F1: {result['type_f1_mean']:.4f}±{result['type_f1_std']:.4f} | "
                f"Features: {result['n_features']}"
            )

    # Summary table
    logger.info("\n" + "=" * 90)
    logger.info("  RESULTS SUMMARY")
    logger.info("=" * 90)

    summary_rows = []
    for r in results:
        row = {
            "Feature": r["feature"],
            "Model": r["model"],
            "N_Features": r["n_features"],
            "Det_AUC": f"{r['det_auc_mean']:.4f}±{r['det_auc_std']:.4f}",
            "Type_F1": f"{r['type_f1_mean']:.4f}±{r['type_f1_std']:.4f}",
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    logger.info(f"\n{summary_df.to_string(index=False)}")

    # Save results
    summary_df.to_csv(out_dir / "ablation_results.csv", index=False)

    # Detailed JSON
    detail = []
    for r in results:
        d = {k: v for k, v in r.items() if k not in ("fold_det_aucs", "fold_type_f1s")}
        d["fold_det_aucs"] = [float(x) for x in r["fold_det_aucs"]]
        d["fold_type_f1s"] = [float(x) for x in r["fold_type_f1s"]]
        detail.append(d)
    with open(out_dir / "ablation_detail.json", "w") as f:
        json.dump(detail, f, indent=2)

    # Best result
    best = max(results, key=lambda r: r["type_f1_mean"])
    logger.info(f"\n  ★ Best Type F1: {best['feature']} × {best['model']} "
                f"= {best['type_f1_mean']:.4f}±{best['type_f1_std']:.4f}")

    best_det = max(results, key=lambda r: r["det_auc_mean"])
    logger.info(f"  ★ Best Det AUC: {best_det['feature']} × {best_det['model']} "
                f"= {best_det['det_auc_mean']:.4f}±{best_det['det_auc_std']:.4f}")

    elapsed = datetime.now() - t0
    logger.info(f"\n  Completed in {elapsed.total_seconds():.0f}s")
    logger.info(f"  Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
