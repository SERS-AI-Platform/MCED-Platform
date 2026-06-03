"""
Build production model artifacts.

Trains final LR models on ALL data (no CV split) and saves
everything needed for inference to an LR-fusion artifact directory.

Saves:
    - SERS-only models (stage1 + stage2)
    - Fusion models (SERS + age/sex/BMI, stage1 + stage2)
    - Common grid, preprocessing params, manifest

Usage:
    python models/legacy/scripts/build_production_model.py
    python models/legacy/scripts/build_production_model.py --output-dir artifacts/baselines/lr-fusion/v1.0.0
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
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.legacy.scripts.train import (
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
TIER1_FEATURES = ["age", "sex_numeric", "bmi"]


def load_clinical():
    clin = pd.read_csv(PROJECT_ROOT / "data" / "clinical_data" / "standardized" / "all_clinical_standardized.csv")
    clin["disease_group"] = clin["disease_group"].replace({"PAN": "CPAN"})
    clin["sex_numeric"] = (clin["sex"] == "M").astype(float)
    return clin


def merge_clinical(df_spec, clin):
    """Merge spectral df with clinical age/sex/bmi.

    Handles three ID formats:
    - Standard: patient_id = 'GROUP SAMPLE_ID' (PRO, OVA, LUN, CRC, CPAN, NOR, etc.)
    - BLC: clinical disease_group='BLA', patient_ids='1기~2기-N' → match by row position
    - BRE: clinical patient_ids are hospital codes → match by row position
    """
    # Standard lookup (GROUP SAMPLE_ID format)
    lookup = {}
    for _, row in clin.iterrows():
        lookup[row["patient_id"]] = {
            "age": row["age"],
            "sex_numeric": row["sex_numeric"],
            "bmi": row.get("bmi", np.nan),
        }

    # Positional lookup for groups with non-standard patient_id formats
    # CSV row order = spectral sample_id order
    pos_lookup = {}
    for grp_spec, grp_clin in [("BLC", "BLA"), ("BRE", "BRE"), ("OVA", "OVA")]:
        grp_rows = clin[clin["disease_group"] == grp_clin].reset_index(drop=True)
        for idx, (_, row) in enumerate(grp_rows.iterrows()):
            pos_key = f"{grp_spec} {idx + 1}"
            pos_lookup[pos_key] = {
                "age": row["age"],
                "sex_numeric": row["sex_numeric"],
                "bmi": row.get("bmi", np.nan),
            }

    # Alias mapping: spectral group name after resolve_aliases → original clinical key prefix
    ALIAS_TO_CLINICAL = {"PAN": "CPAN"}

    ages, sexes, bmis = [], [], []
    for _, row in df_spec.iterrows():
        key = f"{row['group']} {row['sample_id']}"
        # Also try alias key (e.g., PAN→CPAN) and normalized H.D. key
        alias_grp = ALIAS_TO_CLINICAL.get(row["group"], row["group"])
        alias_key = f"{alias_grp} {row['sample_id']}"
        # H.D. in clinical has extra spaces: "H. D. N"
        hd_key = f"H. D. {row['sample_id']}" if row["group"] == "H.D." else None
        c = lookup.get(key) or lookup.get(alias_key) or pos_lookup.get(key)
        if not c and hd_key:
            c = lookup.get(hd_key)
        if c:
            ages.append(c["age"])
            sexes.append(c["sex_numeric"])
            bmis.append(c["bmi"])
        else:
            ages.append(np.nan)
            sexes.append(np.nan)
            bmis.append(np.nan)

    df_spec = df_spec.copy()
    df_spec["age"] = ages
    df_spec["sex_numeric"] = sexes
    df_spec["bmi"] = bmis
    return df_spec


def build_lr(X, y, seed=42):
    """Build a single LR pipeline."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=1000, solver="saga",
                           class_weight="balanced", random_state=seed)
    ).fit(X, y)


def main():
    p = argparse.ArgumentParser(description="Build production model artifacts")
    p.add_argument("--output-dir", default="models/production")
    args = p.parse_args()

    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.info("=" * 64)
    logger.info("  Building Production Model Artifacts")
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

    # Clinical
    clin = load_clinical()
    df_merged = merge_clinical(df_agg, clin)

    X, bl, ctl, sample_ids, groups_arr = create_labels(df_merged, mc)

    df_used = df_merged[df_merged["group"].isin(CANCER_TYPES + NON_CANCER)].reset_index(drop=True)
    tier1 = df_used[TIER1_FEATURES].values.astype(float)

    # Fill BMI with median
    bmi_median = float(np.nanmedian(tier1[:, 2]))
    tier1_filled = tier1.copy()
    tier1_filled[np.isnan(tier1_filled[:, 2]), 2] = bmi_median

    # Identify samples with clinical data
    clin_valid = ~np.isnan(tier1_filled).any(axis=1)
    idx_clin = np.where(clin_valid)[0]

    logger.info(f"  Total spectra: {len(X)}")
    logger.info(f"  With clinical: {len(idx_clin)}")
    logger.info(f"  Cancer types: {list(mc.cancer_types)}")
    logger.info(f"  Non-cancer: {list(mc.non_cancer_groups)}")
    logger.info(f"  BMI fill median: {bmi_median:.1f}")

    # ── Train SERS-only models ──
    logger.info("\n  Training SERS-only models...")

    s1_sers = build_lr(X, bl)
    s1_sers_auc = roc_auc_score(bl, s1_sers.predict_proba(X)[:, 1])
    logger.info(f"    Stage 1 (SERS): train AUC = {s1_sers_auc:.4f}")

    cancer_mask = ctl >= 0
    s2_sers = build_lr(X[cancer_mask], ctl[cancer_mask])
    s2_pred = s2_sers.predict(X[cancer_mask])
    s2_sers_f1 = f1_score(ctl[cancer_mask], s2_pred, average="macro", zero_division=0)
    logger.info(f"    Stage 2 (SERS): train F1 = {s2_sers_f1:.4f}")

    # ── Train Fusion models ──
    logger.info("\n  Training fusion models (SERS + age/sex/BMI)...")

    X_fus = np.hstack([X[idx_clin], tier1_filled[idx_clin]])
    bl_fus = bl[idx_clin]
    ctl_fus = ctl[idx_clin]

    s1_fusion = build_lr(X_fus, bl_fus)
    s1_fus_auc = roc_auc_score(bl_fus, s1_fusion.predict_proba(X_fus)[:, 1])
    logger.info(f"    Stage 1 (fusion): train AUC = {s1_fus_auc:.4f}")

    cancer_fus = ctl_fus >= 0
    X_fus_cancer = X_fus[cancer_fus]
    s2_fusion = build_lr(X_fus_cancer, ctl_fus[cancer_fus])
    s2_fus_pred = s2_fusion.predict(X_fus_cancer)
    s2_fus_f1 = f1_score(ctl_fus[cancer_fus], s2_fus_pred, average="macro", zero_division=0)
    logger.info(f"    Stage 2 (fusion): train F1 = {s2_fus_f1:.4f}")

    # ── Compute operating thresholds via internal CV ──
    logger.info("\n  Computing operating thresholds (5-fold internal CV)...")

    from sklearn.model_selection import StratifiedGroupKFold
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    cv_probs = np.full(len(X), np.nan)
    for _, (tr, va) in enumerate(cv.split(X, bl, sample_ids)):
        m = build_lr(X[tr], bl[tr], seed=42)
        cv_probs[va] = m.predict_proba(X[va])[:, 1]

    valid = ~np.isnan(cv_probs)
    fpr, tpr, thresholds = roc_curve(bl[valid], cv_probs[valid])

    def find_threshold_for_sensitivity(target_sens):
        idx = np.argmin(np.abs(tpr - target_sens))
        return float(thresholds[idx])

    def find_threshold_for_specificity(target_spec):
        idx = np.argmin(np.abs((1 - fpr) - target_spec))
        return float(thresholds[idx])

    # Youden's J (balanced)
    j_scores = tpr - fpr
    balanced_thresh = float(thresholds[np.argmax(j_scores)])

    # Target-based thresholds
    screening_thresh = find_threshold_for_sensitivity(0.95)
    confirmatory_thresh = find_threshold_for_specificity(0.95)

    operating_modes = {
        "screening": {
            "threshold": round(screening_thresh, 4),
            "description": "High sensitivity for screening (target Sens >= 95%)",
        },
        "balanced": {
            "threshold": round(balanced_thresh, 4),
            "description": "Balanced sensitivity/specificity (Youden's J)",
        },
        "confirmatory": {
            "threshold": round(confirmatory_thresh, 4),
            "description": "High specificity for confirmation (target Spec >= 95%)",
        },
    }

    for mode_name, mode_info in operating_modes.items():
        t = mode_info["threshold"]
        preds = (cv_probs[valid] > t).astype(int)
        sens = float((cv_probs[valid][bl[valid] == 1] > t).mean())
        spec = float((cv_probs[valid][bl[valid] == 0] <= t).mean())
        mode_info["cv_sensitivity"] = round(sens, 4)
        mode_info["cv_specificity"] = round(spec, 4)
        logger.info(f"    {mode_name:>14s}: threshold={t:.4f} → Sens={sens:.3f}, Spec={spec:.3f}")

    # ── Save common grid ──
    grid = np.loadtxt(PROJECT_ROOT / "results" / "common_grid.csv", skiprows=1)
    np.save(out_dir / "common_grid.npy", grid)
    logger.info(f"\n  Grid: {len(grid)} points ({grid[0]:.1f}-{grid[-1]:.1f} cm-1)")

    # ── Save models ──
    joblib.dump(s1_sers, out_dir / "stage1_sers.joblib")
    joblib.dump(s2_sers, out_dir / "stage2_sers.joblib")
    joblib.dump(s1_fusion, out_dir / "stage1_fusion.joblib")
    joblib.dump(s2_fusion, out_dir / "stage2_fusion.joblib")

    # ── Save preprocessing params ──
    prep_cfg = raw_cfg.get("preprocessing", {})
    preprocessing = {
        "do_trim": True,
        "trim_region": [400, 2200],
        "do_smooth": prep_cfg.get("do_smooth", True),
        "smooth_window": prep_cfg.get("smooth_window", 11),
        "smooth_poly": prep_cfg.get("smooth_poly", 3),
        "do_baseline": True,
        "baseline_window": prep_cfg.get("baseline_window", 101),
        "normalization": "snv",
    }
    with open(out_dir / "preprocessing.json", "w") as f:
        json.dump(preprocessing, f, indent=2)

    # ── Save manifest ──
    manifest = {
        "version": "1.0",
        "training_date": datetime.now().isoformat(),
        "n_training_samples": int(len(X)),
        "n_spectral_features": int(X.shape[1]),
        "cancer_types": list(mc.cancer_types),
        "non_cancer_groups": list(mc.non_cancer_groups),
        "cancer_type_indices": {name: int(i) for i, name in enumerate(mc.cancer_types)},
        "clinical_features": TIER1_FEATURES,
        "bmi_fill_median": bmi_median,
        "model_files": {
            "stage1_sers": "stage1_sers.joblib",
            "stage2_sers": "stage2_sers.joblib",
            "stage1_fusion": "stage1_fusion.joblib",
            "stage2_fusion": "stage2_fusion.joblib",
        },
        "training_metrics": {
            "sers_only": {"s1_auc": s1_sers_auc, "s2_f1_macro": s2_sers_f1},
            "fusion": {"s1_auc": s1_fus_auc, "s2_f1_macro": s2_fus_f1},
        },
        "stage2_classes": s2_sers.classes_.tolist(),
        "operating_modes": operating_modes,
        "default_mode": "screening",
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    logger.info(f"\n  Artifacts saved to: {out_dir}/")
    for p_file in sorted(out_dir.iterdir()):
        size_kb = p_file.stat().st_size / 1024
        logger.info(f"    {p_file.name:>30s}  {size_kb:>8.1f} KB")

    logger.info("\n" + "=" * 64)
    logger.info("  Production models ready!")
    logger.info(f"  Next: python sers_predict.py <spectrum.csv>")
    logger.info("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
