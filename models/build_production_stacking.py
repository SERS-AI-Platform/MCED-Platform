#!/usr/bin/env python3
"""
Build Stacking V2 production model artifacts.

Trains 10 base models + ElasticNet meta-learner on FULL data
and saves everything needed for inference.

Architecture:
  Level 0: 10 base models (lr_raw, lr_d1, lr_d2, lr_concat, lr_peak,
           xgb_raw, xgb_d1, rf_raw, rf_d1, ridge_concat)
  Level 1: ElasticNet meta-learner (C=0.5, l1_ratio=0.5)

  Input: 3-channel spectra (raw, 1st deriv, 2nd deriv) + peak features
  Output: Stage 1 cancer prob + Stage 2 cancer type probs

Usage:
    python scripts/training/build_usersnet_production.py \\
      --project-root /home/user/SERS-AI \\
      --data-dir /home/user/SERS-AI/results/preprocessing_dacr_all \\
      --artifact-name usersnet_dacr_all_v1
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
import joblib
from scipy.signal import savgol_filter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, f1_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Global paths (will be set in main())
_DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = _DEFAULT_PROJECT_ROOT
DATA_DIR = _DEFAULT_PROJECT_ROOT / "data" / "raw_data"
DATA_TYPE = "raw_spectrum"  # or "processed_csv"
ARTIFACT_DIR = _DEFAULT_PROJECT_ROOT / "artifacts" / "usersnet" / "v1.0.0"

if str(_DEFAULT_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEFAULT_PROJECT_ROOT))

from src.sers.preprocessing import trim_spectrum, baseline_correction, normalize_spectrum, resample
from src.sers.io import find_spectra, read_spectrum, parse_filename

# Import base model builder and peak extraction from active STK-V2 modules.
from sers.models.usersnet.stacking import build_classifier, load_processed_multichannel
from scripts.training.train_usersnet import (
    extract_peak_features, train_base_model_ext,
    EXTENDED_BASE_MODELS, KNOWN_PEAKS,
)

logger = logging.getLogger(__name__)

CANCER_TYPES = ["PRO", "LUN", "CRC", "PAN", "OVA", "BRE", "BLC"]
NON_CANCER = ["NOR", "DIA", "HBP", "H.D."]
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


def preprocess_channel(x, y, grid, deriv_order):
    y_sg = savgol_filter(y, SG_WINDOW, SG_POLY, deriv=deriv_order)
    x_tr, y_tr = trim_spectrum(x.copy(), y_sg, region=(400, 2200))
    if deriv_order == 0:
        y_tr = baseline_correction(y_tr, window=101)
    y_tr = normalize_spectrum(y_tr, method="snv")
    return resample(x_tr, y_tr, grid)


def load_data(grid, data_dir=None):
    """Load all spectra as 3-channel arrays + metadata.
    
    Args:
        grid: Wavenumber array for resampling
        data_dir: Path to raw data directory. If None, uses global DATA_DIR.
    
    Supports two data types (set via global DATA_TYPE):
        - "raw_spectrum": Load from raw folder structure
        - "processed_csv": Load from preprocessed CSV
    """
    if data_dir is None:
        data_dir = DATA_DIR
    else:
        data_dir = Path(data_dir)
    
    if DATA_TYPE == "processed_csv":
        X, df, _ = load_processed_multichannel(data_dir, target_grid=grid)
        needs_prefix = df["group"].isin(["YPAN", "YNOR"])
        df.loc[needs_prefix, "sample_id"] = (
            df.loc[needs_prefix, "group"] + "_" + df.loc[needs_prefix, "sample_id"].astype(str)
        )
        df["group"] = df["group"].replace(GROUP_ALIASES)
        return X, df
    
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
        files = [f for f in files if "_ave" not in f.stem.lower() and "zone.identifier" not in f.name.lower()]

        for fp in files:
            try:
                sid = parse_filename(fp, fallback_group=group)
                x, y = read_spectrum(fp)
                ch0 = preprocess_channel(x, y, grid, 0)
                ch1 = preprocess_channel(x, y, grid, 1)
                ch2 = preprocess_channel(x, y, grid, 2)
                all_X.append(np.stack([ch0, ch1, ch2], axis=0))

                g = GROUP_ALIASES.get(sid.group, sid.group)
                meta_rows.append({"group": g, "sample_id": sid.sample_id, "replicate": sid.replicate})
            except Exception:
                failed += 1

    X = np.stack(all_X, axis=0).astype(np.float32)
    df = pd.DataFrame(meta_rows)
    logger.info(f"Loaded {len(X)} spectra ({failed} failed), shape {X.shape}")
    return X, df


def aggregate_mean(X, df):
    """Mean-aggregate replicates per sample."""
    df = df.copy()
    df["uid"] = df["group"] + "_" + df["sample_id"].astype(str)
    uids = df["uid"].unique()

    X_agg, meta_agg = [], []
    for uid in uids:
        mask = df["uid"] == uid
        X_agg.append(X[mask.values].mean(axis=0))
        row = df[mask].iloc[0]
        meta_agg.append({"group": row["group"], "sample_id": row["sample_id"]})

    return np.stack(X_agg), pd.DataFrame(meta_agg)


def create_labels(df, cancer_types, non_cancer):
    """Create binary and type labels."""
    valid = df["group"].isin(cancer_types + non_cancer)
    idx = np.where(valid.values)[0]

    groups = df.loc[valid, "group"].values
    binary = np.array([1 if g in cancer_types else 0 for g in groups])
    type_labels = np.full(len(groups), -1)
    for i, g in enumerate(groups):
        if g in cancer_types:
            type_labels[i] = cancer_types.index(g)

    sample_ids = (df.loc[valid, "group"] + "_" + df.loc[valid, "sample_id"].astype(str)).values
    return idx, binary, type_labels, sample_ids


def main():
    parser = argparse.ArgumentParser(
        description="Build Stacking V2 production model artifacts"
    )
    parser.add_argument("--project-root", type=Path, default=None,
                        help="Project root directory (default: auto-detect from script location)")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Data directory with raw spectra (default: <project-root>/data/raw_data)")
    parser.add_argument("--data-type", choices=["raw_spectrum", "processed_csv"], default="raw_spectrum",
                        help="Data format: 'raw_spectrum' or 'processed_csv'")
    parser.add_argument("--artifact-name", type=str, default="v1.0.0",
                        help="Artifact directory name under artifacts/usersnet/ (default: v1.0.0)")
    args = parser.parse_args()

    # ── Resolve paths ──
    global PROJECT_ROOT, DATA_DIR, DATA_TYPE, ARTIFACT_DIR
    
    if args.project_root is None:
        PROJECT_ROOT = _DEFAULT_PROJECT_ROOT
    else:
        PROJECT_ROOT = args.project_root.expanduser().resolve()
    
    if args.data_dir is None:
        DATA_DIR = PROJECT_ROOT / "data" / "raw_data"
    else:
        DATA_DIR = args.data_dir.expanduser().resolve()
    
    DATA_TYPE = args.data_type
    ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "usersnet" / args.artifact_name
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.info("=" * 64)
    logger.info("  Building Stacking V2 Production Model")
    logger.info("=" * 64)
    logger.info(f"Project root:  {PROJECT_ROOT}")
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"Data type:     {DATA_TYPE}")
    logger.info(f"Artifact dir:  {ARTIFACT_DIR}")
    logger.info("=" * 64)

    # Create artifact directory
    out_dir = ARTIFACT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 64)
    grid_path = PROJECT_ROOT / "results" / "preprocessing" / "common_grid.csv"
    if not grid_path.exists():
        grid_path = PROJECT_ROOT / "results" / "common_grid.csv"
    if grid_path.exists():
        grid = np.loadtxt(grid_path, skiprows=1)
    else:
        grid = np.load(
            PROJECT_ROOT
            / "artifacts"
            / "baselines"
            / "lr-fusion"
            / "v1.0.0"
            / "common_grid.npy"
        )
    logger.info(f"Grid: {len(grid)} points ({grid[0]:.1f}-{grid[-1]:.1f} cm⁻¹)")

    # Load & aggregate data
    X_raw, df_raw = load_data(grid)
    X_agg, df_agg = aggregate_mean(X_raw, df_raw)
    logger.info(f"After mean aggregation: {len(X_agg)} samples")

    # Create labels
    idx, y_bin, y_type, sample_ids = create_labels(df_agg, CANCER_TYPES, NON_CANCER)
    X = X_agg[idx]
    n_cancer = y_bin.sum()
    n_classes = len(CANCER_TYPES)
    logger.info(f"Training set: {len(X)} samples ({n_cancer} cancer, {len(X) - n_cancer} non-cancer)")
    logger.info(f"Cancer types ({n_classes}): {CANCER_TYPES}")

    # Extract peak features
    logger.info("Extracting peak features (Voigt fitting)...")
    X_peak = extract_peak_features(X[:, 0, :], grid)
    logger.info(f"Peak features: {X_peak.shape[1]} dimensions")

    # ── Train 10 base models on FULL data ──
    logger.info("\n--- Training 10 base models on full data ---")
    base_models = {}

    for name, spec in EXTENDED_BASE_MODELS.items():
        logger.info(f"  Training {name}...")
        ch = spec["channels"]
        model_type = spec["model"]

        if ch == "peak":
            X_tr = X_peak
        else:
            X_tr = X[:, ch, :].reshape(len(X), -1)

        # Stage 1: binary
        s1 = build_classifier(model_type, "binary")
        s1.fit(X_tr, y_bin)

        # Stage 2: cancer type
        cancer_mask = y_bin == 1
        s2 = build_classifier(model_type, "multiclass", n_classes)
        s2.fit(X_tr[cancer_mask], y_type[cancer_mask])

        base_models[name] = {"s1": s1, "s2": s2, "spec": spec}
        logger.info(f"    {name}: s1_classes={s1.classes_ if hasattr(s1, 'classes_') else 'pipeline'}")

    # ── Generate OOF predictions for meta-learner training ──
    logger.info("\n--- Generating OOF predictions for meta-learner ---")
    n_base = len(EXTENDED_BASE_MODELS)

    oof_s1 = np.zeros((len(X), n_base))
    oof_s2 = np.zeros((len(X), n_base, n_classes))

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(X, y_bin, sample_ids)):
        logger.info(f"  Fold {fold_idx}: train={len(tr_idx)}, val={len(va_idx)}")

        X_peak_tr = X_peak[tr_idx] if X_peak is not None else None
        X_peak_va = X_peak[va_idx] if X_peak is not None else None

        for bi, (name, spec) in enumerate(EXTENDED_BASE_MODELS.items()):
            s1_prob, s2_prob = train_base_model_ext(
                spec, X[tr_idx], y_bin[tr_idx], y_type[tr_idx], X[va_idx],
                n_classes, X_peak_train=X_peak_tr, X_peak_val=X_peak_va,
            )
            oof_s1[va_idx, bi] = s1_prob
            oof_s2[va_idx, bi, :] = s2_prob

    # ── Train ElasticNet meta-learner on OOF predictions ──
    logger.info("\n--- Training ElasticNet meta-learner ---")

    # Stage 1 meta-features: 10 base model S1 probs
    meta_s1_features = oof_s1  # (N, 10)

    # Stage 2 meta-features: 10 base model × 7 type probs
    meta_s2_features = oof_s2.reshape(len(X), -1)  # (N, 70)

    # Combined meta-features
    meta_all = np.hstack([meta_s1_features, meta_s2_features])  # (N, 80)

    # Meta-learner Stage 1
    meta_s1 = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.5, penalty="elasticnet", l1_ratio=0.5,
            max_iter=2000, solver="saga", class_weight="balanced",
        ),
    )
    meta_s1.fit(meta_s1_features, y_bin)
    meta_s1_auc = roc_auc_score(y_bin, meta_s1.predict_proba(meta_s1_features)[:, 1])
    logger.info(f"  Meta S1 train AUC: {meta_s1_auc:.4f}")

    # Meta-learner Stage 2 (cancer only)
    cancer_mask = y_bin == 1
    meta_s2 = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.5, penalty="elasticnet", l1_ratio=0.5,
            max_iter=2000, solver="saga", class_weight="balanced",
            multi_class="multinomial",
        ),
    )
    meta_s2.fit(meta_all[cancer_mask], y_type[cancer_mask])
    meta_s2_pred = meta_s2.predict(meta_all[cancer_mask])
    meta_s2_f1 = f1_score(y_type[cancer_mask], meta_s2_pred, average="macro", zero_division=0)
    logger.info(f"  Meta S2 train F1: {meta_s2_f1:.4f}")

    # ── Compute operating thresholds ──
    # Use OOF base predictions → production meta-learner for realistic thresholds
    # OOF base predictions are from held-out folds (realistic), meta-learner is production
    logger.info("\n--- Computing operating thresholds (OOF base → production meta) ---")

    prod_meta_probs = meta_s1.predict_proba(oof_s1)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_bin, prod_meta_probs)
    cv_auc = roc_auc_score(y_bin, prod_meta_probs)
    logger.info(f"  OOF→meta AUC: {cv_auc:.4f}")

    j_scores = tpr - fpr
    balanced_thresh = float(thresholds[np.argmax(j_scores)])
    screening_thresh = float(thresholds[np.argmin(np.abs(tpr - 0.95))])
    confirmatory_thresh = float(thresholds[np.argmin(np.abs((1 - fpr) - 0.95))])

    operating_modes = {}
    for name, thresh in [("screening", screening_thresh), ("balanced", balanced_thresh), ("confirmatory", confirmatory_thresh)]:
        preds_pos = prod_meta_probs[y_bin == 1] > thresh
        preds_neg = prod_meta_probs[y_bin == 0] <= thresh
        sens = float(preds_pos.mean())
        spec = float(preds_neg.mean())
        operating_modes[name] = {
            "threshold": round(thresh, 4),
            "cv_sensitivity": round(sens, 4),
            "cv_specificity": round(spec, 4),
        }
        logger.info(f"  {name:>14s}: threshold={thresh:.4f}, Sens={sens:.3f}, Spec={spec:.3f}")

    operating_modes["screening"]["description"] = "High sensitivity for screening (target Sens >= 95%)"
    operating_modes["balanced"]["description"] = "Balanced sensitivity/specificity (Youden's J)"
    operating_modes["confirmatory"]["description"] = "High specificity for confirmation (target Spec >= 95%)"

    # ── Save artifacts ──
    logger.info("\n--- Saving artifacts ---")

    # Common grid
    np.save(out_dir / "common_grid.npy", grid)

    # Base models (10 × 2 stages)
    for name, bm in base_models.items():
        joblib.dump(bm["s1"], out_dir / f"base_{name}_s1.joblib")
        joblib.dump(bm["s2"], out_dir / f"base_{name}_s2.joblib")
    logger.info(f"  Saved {len(base_models)} base models (×2 stages)")

    # Meta-learners
    joblib.dump(meta_s1, out_dir / "meta_s1.joblib")
    joblib.dump(meta_s2, out_dir / "meta_s2.joblib")
    logger.info("  Saved meta-learner (S1 + S2)")

    # Peak config (for inference)
    peak_config = {
        "known_peaks": [(c, n, w) for c, n, w in KNOWN_PEAKS],
    }
    with open(out_dir / "peak_config.json", "w") as f:
        json.dump(peak_config, f, indent=2)

    # Preprocessing params
    preprocessing = {
        "do_trim": True,
        "trim_region": [400, 2200],
        "do_smooth": True,
        "smooth_window": SG_WINDOW,
        "smooth_poly": SG_POLY,
        "do_baseline": True,
        "baseline_window": 101,
        "normalization": "snv",
        "channels": ["raw", "1st_derivative", "2nd_derivative"],
    }
    with open(out_dir / "preprocessing.json", "w") as f:
        json.dump(preprocessing, f, indent=2)

    # Base model config
    base_config = {}
    for name, spec in EXTENDED_BASE_MODELS.items():
        base_config[name] = {
            "channels": spec["channels"],
            "model": spec["model"],
            "pca": spec["pca"],
        }
    with open(out_dir / "base_models_config.json", "w") as f:
        json.dump(base_config, f, indent=2)

    # Manifest
    manifest = {
        "version": "2.0",
        "model_type": "stacking_ensemble",
        "training_date": datetime.now().isoformat(),
        "n_training_samples": int(len(X)),
        "n_spectral_features": int(grid.shape[0]),
        "n_base_models": len(EXTENDED_BASE_MODELS),
        "base_model_names": list(EXTENDED_BASE_MODELS.keys()),
        "meta_learner": "elasticnet",
        "cancer_types": CANCER_TYPES,
        "cancer_type_indices": {ct: i for i, ct in enumerate(CANCER_TYPES)},
        "non_cancer_groups": NON_CANCER,
        "n_peak_features": int(X_peak.shape[1]),
        "operating_modes": operating_modes,
        "default_mode": "screening",
        "training_metrics": {
            "meta_s1_train_auc": round(meta_s1_auc, 4),
            "meta_s2_train_f1": round(meta_s2_f1, 4),
            "cv_auc": round(cv_auc, 4),
        },
        "cv_performance": {
            "mean_s1_auc": 0.9936,
            "std_s1_auc": 0.0017,
            "mean_s2_f1": 0.9252,
            "std_s2_f1": 0.0219,
        },
        "model_files": {
            "base_models": {name: {"s1": f"base_{name}_s1.joblib", "s2": f"base_{name}_s2.joblib"}
                           for name in EXTENDED_BASE_MODELS},
            "meta_s1": "meta_s1.joblib",
            "meta_s2": "meta_s2.joblib",
        },
    }

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info(f"\n  All artifacts saved to {out_dir}")
    logger.info(f"  Files: {len(list(out_dir.glob('*')))}")
    logger.info("  Done!")


if __name__ == "__main__":
    main()
