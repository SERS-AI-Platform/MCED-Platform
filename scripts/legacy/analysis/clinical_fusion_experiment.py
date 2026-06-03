"""
Clinical Information Fusion Experiment

Compares 4-view fusion (spectrum + derivative + peak + clinical)
against baselines on matched cohort.

Views:
  1. Full spectrum (935)
  2. 1st derivative (935)
  3. Peak features (75) — Voigt-fitted at 17 known SERS peaks
  4. Clinical (age, sex_M, bmi) — 3 features

Conditions:
  A) SERS only: full spectrum (935)
  B) SERS 3-view: spectrum + deriv + peak (1945)
  C) Clinical only: age + sex + bmi (3)
  D) SERS + Clinical early fusion: spectrum + clinical (938)
  E) 3-view + Clinical: spectrum + deriv + peak + clinical (1948)
  F) 1st deriv + Clinical (938)

Models: LR (+ sex constraint for Stage 2)
Evaluation: 5-fold StratifiedGroupKFold CV, mean aggregation

Usage:
    python scripts/analysis/clinical_fusion_experiment.py
"""

from __future__ import annotations

import sys
import logging
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from scipy.special import voigt_profile
from scipy.integrate import trapezoid
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, f1_score

import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.sers.config import RESULTS_DIR

logger = logging.getLogger(__name__)

CANCER_TYPES = ["PRO", "OVA", "LUN", "CRC", "PAN"]  # Only types with clinical data
NON_CANCER = ["NOR", "DIA", "HBP", "H.D."]
GROUP_ALIASES = {"PAN": ["CPAN", "YPAN"], "NOR": ["YNOR"]}

# 17 known SERS peaks
KNOWN_PEAKS = [
    (448.1, "ring_deform", 15), (538.7, "SS_stretch", 15),
    (617.7, "CS_stretch", 15), (683.3, "creatinine", 15),
    (723.8, "adenine", 15), (795.1, "hippuric", 15),
    (849.1, "tyrosine", 15), (895.4, "uric_acid", 15),
    (933.9, "creatinine2", 15), (999.5, "phe_urea", 15),
    (1147.9, "uric_CN", 15), (1230.8, "amide_III", 20),
    (1292.5, "CH2_twist", 15), (1352.3, "trp_fermi", 15),
    (1448.7, "CH2_deform", 20), (1597.1, "purine_CC", 20),
    (1651.1, "amide_I", 20),
]


# =============================================================================
# Feature Extraction
# =============================================================================

def extract_1st_derivative(X):
    return np.apply_along_axis(
        lambda y: savgol_filter(y, window_length=11, polyorder=3, deriv=1),
        axis=1, arr=X,
    )


def voigt_func(x, amplitude, center, sigma, gamma):
    return amplitude * voigt_profile(x - center, sigma, gamma)


def extract_peak_features(X, wavenumbers):
    n_samples = len(X)
    n_peaks = len(KNOWN_PEAKS)
    peak_areas = np.zeros((n_samples, n_peaks))
    peak_heights = np.zeros((n_samples, n_peaks))
    peak_fwhms = np.zeros((n_samples, n_peaks))
    peak_shifts = np.zeros((n_samples, n_peaks))

    for pi, (center, name, half_w) in enumerate(KNOWN_PEAKS):
        lo_wn, hi_wn = center - half_w * 1.5, center + half_w * 1.5
        mask = (wavenumbers >= lo_wn) & (wavenumbers <= hi_wn)
        if mask.sum() < 5:
            continue
        x_region = wavenumbers[mask]
        for si in range(n_samples):
            y_region = X[si, mask]
            y_shifted = y_region - np.min(y_region)
            try:
                amp_guess = max(np.max(y_shifted), 1e-6)
                p0 = [amp_guess, center, half_w / 3, half_w / 3]
                bounds_lo = [0, center - half_w, 0.1, 0.1]
                bounds_hi = [amp_guess * 10, center + half_w, half_w * 2, half_w * 2]
                popt, _ = curve_fit(voigt_func, x_region, y_shifted,
                                   p0=p0, bounds=(bounds_lo, bounds_hi), maxfev=2000)
                amp, ctr, sigma, gamma = popt
                y_fit = voigt_func(x_region, *popt)
                area = trapezoid(y_fit, x_region)
                fL = 2 * gamma
                fG = 2 * sigma * np.sqrt(2 * np.log(2))
                fwhm = 0.5346 * fL + np.sqrt(0.2166 * fL**2 + fG**2)
            except (RuntimeError, ValueError):
                amp, ctr, area, fwhm = 0.0, center, 0.0, 0.0
            peak_areas[si, pi] = area
            peak_heights[si, pi] = amp
            peak_fwhms[si, pi] = fwhm
            peak_shifts[si, pi] = ctr - center

    features = [peak_areas, peak_heights, peak_fwhms, peak_shifts]
    peak_name_to_idx = {p[1]: i for i, p in enumerate(KNOWN_PEAKS)}
    ratio_pairs = [
        ("phe_urea", "adenine"), ("phe_urea", "creatinine"),
        ("hippuric", "creatinine"), ("CS_stretch", "creatinine"),
        ("amide_I", "CH2_deform"), ("adenine", "purine_CC"),
        ("tyrosine", "phe_urea"),
    ]
    for na, nb in ratio_pairs:
        ia, ib = peak_name_to_idx[na], peak_name_to_idx[nb]
        features.append(peak_areas[:, ia:ia+1] / (peak_areas[:, ib:ib+1] + 1e-10))
    return np.nan_to_num(np.hstack(features), nan=0.0, posinf=0.0, neginf=0.0)


# =============================================================================
# Data Loading & Merging
# =============================================================================

def load_and_merge():
    """Load spectral + clinical data, merge, return matched cohort."""
    # Spectra
    spec_df = pd.read_csv(RESULTS_DIR / "processed_spectra.csv")
    feat_cols = [c for c in spec_df.columns if c.startswith("x_")]
    wavenumbers = np.array([float(c.replace("x_", "")) for c in feat_cols])

    # Mean aggregation
    meta_cols = ["group", "sample_id"]
    agg_df = spec_df.groupby(meta_cols, as_index=False).agg(
        {**{c: "mean" for c in feat_cols}, "replicate": "count"}
    )

    # Build merge key
    def make_key(row):
        if row["group"] == "H.D.":
            return f"H. D. {row['sample_id']}"
        return f"{row['group']} {row['sample_id']}"

    agg_df["merge_key"] = agg_df.apply(make_key, axis=1)

    # Clinical
    clin_df = pd.read_csv(
        PROJECT_ROOT / "data/clinical_data/standardized/all_clinical_standardized.csv",
        encoding="utf-8-sig",
    )

    # Merge
    merged = agg_df.merge(
        clin_df[["patient_id", "age", "sex", "bmi"]],
        left_on="merge_key", right_on="patient_id", how="inner",
    )

    # Resolve aliases
    def resolve_group(g):
        for alias, members in GROUP_ALIASES.items():
            if g in members:
                return alias
        return g

    merged["group_resolved"] = merged["group"].map(resolve_group)

    # Filter to valid groups (excl SPAN)
    valid_groups = set(CANCER_TYPES) | set(NON_CANCER)
    merged = merged[merged["group_resolved"].isin(valid_groups)].copy()

    # Drop rows missing clinical
    merged = merged.dropna(subset=["age", "bmi"])

    # Encode sex
    merged["sex_M"] = (merged["sex"] == "M").astype(float)

    logger.info(f"  Matched cohort: {len(merged)} samples")
    logger.info(f"  Groups: {merged['group_resolved'].value_counts().sort_index().to_dict()}")

    return merged, feat_cols, wavenumbers


# =============================================================================
# Sex Constraint (from Phase Q)
# =============================================================================

def apply_sex_constraint(predictions, sex_array, cancer_types):
    """PRO only for M, OVA/BRE only for F."""
    pred = predictions.copy()
    pro_idx = cancer_types.index("PRO") if "PRO" in cancer_types else None
    ova_idx = cancer_types.index("OVA") if "OVA" in cancer_types else None

    if pro_idx is not None:
        # Female cannot have PRO
        female_mask = sex_array == 0  # sex_M = 0 → female
        female_pro = female_mask & (pred == pro_idx)
        if female_pro.any():
            pred[female_pro] = -1  # Will be re-assigned

    if ova_idx is not None:
        # Male cannot have OVA
        male_mask = sex_array == 1
        male_ova = male_mask & (pred == ova_idx)
        if male_ova.any():
            pred[male_ova] = -1

    return pred


# =============================================================================
# CV Evaluation
# =============================================================================

def run_cv(X, groups, sample_ids, sex_arr, condition_name,
           n_splits=5, random_state=42, use_sex_constraint=True):
    """Two-stage CV with optional sex constraint."""
    cancer_set = set(CANCER_TYPES)
    bl = np.array([1 if g in cancer_set else 0 for g in groups])
    ctl = np.array([CANCER_TYPES.index(g) if g in cancer_set else -1 for g in groups])

    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    aucs, f1s, f1s_noconstraint = [], [], []

    for ti, vi in skf.split(X, bl, groups=sample_ids):
        # Stage 1
        m1 = make_pipeline(StandardScaler(),
                          LogisticRegression(C=1.0, max_iter=1000, solver="saga",
                                           class_weight="balanced", random_state=random_state))
        m1.fit(X[ti], bl[ti])
        prob = m1.predict_proba(X[vi])[:, 1]
        try:
            aucs.append(roc_auc_score(bl[vi], prob))
        except ValueError:
            aucs.append(np.nan)

        # Stage 2
        cm_tr, cm_te = ctl[ti] >= 0, ctl[vi] >= 0
        present = sorted(set(ctl[ti][cm_tr]))
        if len(present) >= 2 and cm_te.sum() > 0:
            lm = {c: i for i, c in enumerate(present)}
            im = {i: c for c, i in lm.items()}
            yt = np.array([lm[c] for c in ctl[ti][cm_tr]])
            yte = np.array([lm.get(c, -1) for c in ctl[vi][cm_te]])
            seen = yte >= 0
            if seen.sum() > 0:
                m2 = make_pipeline(StandardScaler(),
                                  LogisticRegression(C=1.0, max_iter=1000, solver="saga",
                                                    class_weight="balanced", random_state=random_state))
                m2.fit(X[ti][cm_tr], yt)
                pred = m2.predict(X[vi][cm_te][seen])

                # F1 without constraint
                yp = np.array([im[p] for p in pred])
                ytrue = np.array([im[t] for t in yte[seen]])
                f1_no = f1_score(ytrue, yp, average="macro", zero_division=0)
                f1s_noconstraint.append(f1_no)

                # F1 with sex constraint
                if use_sex_constraint:
                    pred_constrained = apply_sex_constraint(
                        yp, sex_arr[vi][cm_te][seen], CANCER_TYPES
                    )
                    # Re-assign -1 predictions to next best
                    needs_fix = pred_constrained == -1
                    if needs_fix.any():
                        probs = m2.predict_proba(X[vi][cm_te][seen][needs_fix])
                        for idx_local in range(needs_fix.sum()):
                            sorted_classes = np.argsort(probs[idx_local])[::-1]
                            for sc in sorted_classes:
                                candidate = im[sc]
                                sex_val = sex_arr[vi][cm_te][seen][needs_fix][idx_local]
                                if candidate == "PRO" and sex_val == 0:
                                    continue
                                if candidate == "OVA" and sex_val == 1:
                                    continue
                                pred_constrained[np.where(needs_fix)[0][idx_local]] = candidate
                                break
                    f1_c = f1_score(ytrue, pred_constrained, average="macro", zero_division=0)
                    f1s.append(f1_c)
                else:
                    f1s.append(f1_no)
            else:
                f1s.append(np.nan)
                f1s_noconstraint.append(np.nan)
        else:
            f1s.append(np.nan)
            f1s_noconstraint.append(np.nan)

    return {
        "condition": condition_name,
        "n_features": X.shape[1],
        "det_auc_mean": np.nanmean(aucs),
        "det_auc_std": np.nanstd(aucs),
        "type_f1_mean": np.nanmean(f1s),
        "type_f1_std": np.nanstd(f1s),
        "type_f1_noconstraint_mean": np.nanmean(f1s_noconstraint),
        "fold_aucs": aucs,
        "fold_f1s": f1s,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    t0 = datetime.now()
    out_dir = RESULTS_DIR / "clinical_fusion"
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(out_dir / "experiment.log", mode="w", encoding="utf-8"),
        ],
    )

    logger.info("=" * 64)
    logger.info("  Clinical Fusion Experiment")
    logger.info("=" * 64)

    # Load & merge
    logger.info("\n[1] Loading & merging data...")
    merged, feat_cols, wavenumbers = load_and_merge()

    X_spec = merged[feat_cols].values
    groups = merged["group_resolved"].values
    sample_ids = merged["sample_id"].values
    sex_arr = merged["sex_M"].values
    clinical = merged[["age", "sex_M", "bmi"]].values

    # Extract views
    logger.info("\n[2] Extracting features...")
    logger.info("  1st derivative...")
    X_deriv = extract_1st_derivative(X_spec)
    logger.info("  Peak features (Voigt fitting)...")
    X_peak = extract_peak_features(X_spec, wavenumbers)

    # Build feature combinations
    conditions = {
        "A: spectrum_only":       X_spec,
        "B: 1st_deriv_only":     X_deriv,
        "C: clinical_only":      clinical,
        "D: spectrum+clinical":  np.hstack([X_spec, clinical]),
        "E: deriv+clinical":     np.hstack([X_deriv, clinical]),
        "F: 3view":              np.hstack([X_spec, X_deriv, X_peak]),
        "G: 3view+clinical":     np.hstack([X_spec, X_deriv, X_peak, clinical]),
        "H: deriv+peak+clinical": np.hstack([X_deriv, X_peak, clinical]),
    }

    # Run experiments
    logger.info(f"\n[3] Running {len(conditions)} conditions...")
    results = []
    for cond_name, X_feat in conditions.items():
        logger.info(f"\n  {cond_name} ({X_feat.shape[1]} features)...")
        r = run_cv(X_feat, groups, sample_ids, sex_arr, cond_name)
        results.append(r)
        logger.info(f"    Det AUC: {r['det_auc_mean']:.4f}±{r['det_auc_std']:.4f}")
        logger.info(f"    Type F1: {r['type_f1_mean']:.4f}±{r['type_f1_std']:.4f} "
                    f"(no constraint: {r['type_f1_noconstraint_mean']:.4f})")

    # Summary
    logger.info("\n" + "=" * 90)
    logger.info("  RESULTS SUMMARY (5-cancer, sex constraint, matched cohort)")
    logger.info("=" * 90)

    rows = []
    for r in results:
        rows.append({
            "Condition": r["condition"],
            "Features": r["n_features"],
            "Det_AUC": f"{r['det_auc_mean']:.4f}±{r['det_auc_std']:.4f}",
            "Type_F1": f"{r['type_f1_mean']:.4f}±{r['type_f1_std']:.4f}",
            "F1_noSexC": f"{r['type_f1_noconstraint_mean']:.4f}",
        })

    summary_df = pd.DataFrame(rows)
    logger.info(f"\n{summary_df.to_string(index=False)}")

    # Save
    summary_df.to_csv(out_dir / "results.csv", index=False)
    detail = [{
        "condition": r["condition"],
        "n_features": r["n_features"],
        "det_auc_mean": r["det_auc_mean"],
        "type_f1_mean": r["type_f1_mean"],
        "fold_aucs": [float(x) for x in r["fold_aucs"]],
        "fold_f1s": [float(x) for x in r["fold_f1s"]],
    } for r in results]
    with open(out_dir / "detail.json", "w") as f:
        json.dump(detail, f, indent=2)

    # Best
    best = max(results, key=lambda r: r["type_f1_mean"])
    logger.info(f"\n  ★ Best Type F1: {best['condition']} "
                f"= {best['type_f1_mean']:.4f}±{best['type_f1_std']:.4f}")

    # Cancer type breakdown
    logger.info(f"\n  Cancer types used: {CANCER_TYPES}")
    logger.info(f"  Non-cancer: {NON_CANCER}")
    logger.info(f"  Note: BLC, BRE excluded (no clinical data mapping)")
    logger.info(f"  Note: LUN partially included (200/300)")

    elapsed = datetime.now() - t0
    logger.info(f"\n  Completed in {elapsed.total_seconds():.0f}s")
    logger.info(f"  Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
