"""
Clinical Fusion Experiment — Full Cohort (using clinical_utils.py)

Uses existing build_id_mappings() for BLC/BRE/OVA/H.D. mapping.
All 1,628 subjects (7-cancer + 4 non-cancer, SPAN excluded).

Usage:
    python scripts/analysis/clinical_fusion_full.py
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
from sers.models.usersnet.clinical_fusion import load_clinical, merge_clinical_features

logger = logging.getLogger(__name__)

CANCER_TYPES = ["PRO", "BRE", "OVA", "LUN", "CRC", "PAN", "BLC"]
NON_CANCER = ["NOR", "DIA", "HBP", "H.D."]
GROUP_ALIASES = {"PAN": ["CPAN", "YPAN"], "NOR": ["YNOR"]}

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
    n_samples, n_peaks = len(X), len(KNOWN_PEAKS)
    peak_areas = np.zeros((n_samples, n_peaks))
    peak_heights = np.zeros((n_samples, n_peaks))
    peak_fwhms = np.zeros((n_samples, n_peaks))
    peak_shifts = np.zeros((n_samples, n_peaks))

    for pi, (center, name, half_w) in enumerate(KNOWN_PEAKS):
        mask = (wavenumbers >= center - half_w * 1.5) & (wavenumbers <= center + half_w * 1.5)
        if mask.sum() < 5:
            continue
        x_region = wavenumbers[mask]
        for si in range(n_samples):
            y_shifted = X[si, mask] - np.min(X[si, mask])
            try:
                amp_guess = max(np.max(y_shifted), 1e-6)
                popt, _ = curve_fit(voigt_func, x_region, y_shifted,
                                   p0=[amp_guess, center, half_w/3, half_w/3],
                                   bounds=([0, center-half_w, 0.1, 0.1],
                                          [amp_guess*10, center+half_w, half_w*2, half_w*2]),
                                   maxfev=2000)
                amp, ctr, sigma, gamma = popt
                y_fit = voigt_func(x_region, *popt)
                area = trapezoid(y_fit, x_region)
                fL, fG = 2*gamma, 2*sigma*np.sqrt(2*np.log(2))
                fwhm = 0.5346*fL + np.sqrt(0.2166*fL**2 + fG**2)
            except (RuntimeError, ValueError):
                amp, ctr, area, fwhm = 0.0, center, 0.0, 0.0
            peak_areas[si, pi] = area
            peak_heights[si, pi] = amp
            peak_fwhms[si, pi] = fwhm
            peak_shifts[si, pi] = ctr - center

    features = [peak_areas, peak_heights, peak_fwhms, peak_shifts]
    pn2i = {p[1]: i for i, p in enumerate(KNOWN_PEAKS)}
    for na, nb in [("phe_urea","adenine"),("phe_urea","creatinine"),
                   ("hippuric","creatinine"),("CS_stretch","creatinine"),
                   ("amide_I","CH2_deform"),("adenine","purine_CC"),("tyrosine","phe_urea")]:
        features.append(peak_areas[:, pn2i[na]:pn2i[na]+1] / (peak_areas[:, pn2i[nb]:pn2i[nb]+1] + 1e-10))
    return np.nan_to_num(np.hstack(features), nan=0.0, posinf=0.0, neginf=0.0)


# =============================================================================
# CV Evaluation
# =============================================================================

def apply_sex_constraint(pred, sex_arr, cancer_types, probs_matrix):
    """Enforce PRO→M only, OVA→F only. Re-assign via next-best probability."""
    pred = pred.copy()
    pro_idx = cancer_types.index("PRO") if "PRO" in cancer_types else None
    ova_idx = cancer_types.index("OVA") if "OVA" in cancer_types else None

    for i in range(len(pred)):
        if pro_idx is not None and pred[i] == pro_idx and sex_arr[i] == 0:
            # Female predicted PRO → re-assign
            sorted_c = np.argsort(probs_matrix[i])[::-1]
            for c in sorted_c:
                if c == pro_idx:
                    continue
                if ova_idx is not None and c == ova_idx and sex_arr[i] == 1:
                    continue
                pred[i] = c
                break
        elif ova_idx is not None and pred[i] == ova_idx and sex_arr[i] == 1:
            # Male predicted OVA → re-assign
            sorted_c = np.argsort(probs_matrix[i])[::-1]
            for c in sorted_c:
                if c == ova_idx:
                    continue
                if pro_idx is not None and c == pro_idx and sex_arr[i] == 0:
                    continue
                pred[i] = c
                break
    return pred


def run_cv(X, groups, sample_ids, sex_arr, condition_name,
           n_splits=5, random_state=42):
    cancer_set = set(CANCER_TYPES)
    bl = np.array([1 if g in cancer_set else 0 for g in groups])
    ctl = np.array([CANCER_TYPES.index(g) if g in cancer_set else -1 for g in groups])

    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    aucs, f1s, f1s_nc = [], [], []

    for ti, vi in skf.split(X, bl, groups=sample_ids):
        m1 = make_pipeline(StandardScaler(),
                          LogisticRegression(C=1.0, max_iter=1000, solver="saga",
                                           class_weight="balanced", random_state=random_state))
        m1.fit(X[ti], bl[ti])
        prob = m1.predict_proba(X[vi])[:, 1]
        try:
            aucs.append(roc_auc_score(bl[vi], prob))
        except ValueError:
            aucs.append(np.nan)

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
                pred_local = m2.predict(X[vi][cm_te][seen])
                probs_local = m2.predict_proba(X[vi][cm_te][seen])

                yp = np.array([im[p] for p in pred_local])
                ytrue = np.array([im[t] for t in yte[seen]])
                f1_nc = f1_score(ytrue, yp, average="macro", zero_division=0)
                f1s_nc.append(f1_nc)

                # Sex constraint
                yp_c = apply_sex_constraint(yp, sex_arr[vi][cm_te][seen], CANCER_TYPES, probs_local)
                f1_c = f1_score(ytrue, yp_c, average="macro", zero_division=0)
                f1s.append(f1_c)
            else:
                f1s.append(np.nan); f1s_nc.append(np.nan)
        else:
            f1s.append(np.nan); f1s_nc.append(np.nan)

    return {
        "condition": condition_name,
        "n_features": X.shape[1],
        "det_auc": f"{np.nanmean(aucs):.4f}±{np.nanstd(aucs):.4f}",
        "type_f1": f"{np.nanmean(f1s):.4f}±{np.nanstd(f1s):.4f}",
        "type_f1_nc": f"{np.nanmean(f1s_nc):.4f}",
        "det_auc_mean": np.nanmean(aucs),
        "type_f1_mean": np.nanmean(f1s),
        "fold_aucs": [float(x) for x in aucs],
        "fold_f1s": [float(x) for x in f1s],
    }


# =============================================================================
# Main
# =============================================================================

def main():
    t0 = datetime.now()
    out_dir = RESULTS_DIR / "clinical_fusion_full"
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
    logger.info("  Clinical Fusion — Full Cohort (7-cancer)")
    logger.info("=" * 64)

    # Load spectra
    logger.info("\n[1] Loading spectra...")
    spec_df = pd.read_csv(RESULTS_DIR / "processed_spectra.csv")
    feat_cols = [c for c in spec_df.columns if c.startswith("x_")]
    wavenumbers = np.array([float(c.replace("x_", "")) for c in feat_cols])

    # Aggregate
    agg_df = spec_df.groupby(["group", "sample_id"], as_index=False).agg(
        {**{c: "mean" for c in feat_cols}, "replicate": "count"}
    )

    # Resolve aliases
    def resolve(g):
        for alias, members in GROUP_ALIASES.items():
            if g in members:
                return alias
        return g

    agg_df["group_raw"] = agg_df["group"]
    agg_df["group"] = agg_df["group"].map(resolve)

    # Filter valid groups (excl SPAN)
    valid = set(CANCER_TYPES) | set(NON_CANCER)
    agg_df = agg_df[agg_df["group"].isin(valid)].reset_index(drop=True)
    logger.info(f"  Samples after filtering: {len(agg_df)}")
    logger.info(f"  Groups: {agg_df['group'].value_counts().sort_index().to_dict()}")

    # Load clinical using clinical_utils
    logger.info("\n[2] Loading clinical data (clinical_utils.py)...")
    clin = load_clinical(PROJECT_ROOT)
    clinical_cols = ["age", "sex_numeric", "bmi"]
    clinical_data = merge_clinical_features(agg_df, clin, clinical_cols, PROJECT_ROOT)

    # Check match rate
    n_with_age = (~np.isnan(clinical_data[:, 0])).sum() if np.isnan(clinical_data[:, 0]).any() else len(clinical_data)
    logger.info(f"  Clinical features shape: {clinical_data.shape}")

    # Extract spectral features
    X_spec = agg_df[feat_cols].values
    groups = agg_df["group"].values
    sample_ids = agg_df["sample_id"].values
    sex_arr = clinical_data[:, 1]  # sex_numeric

    logger.info("\n[3] Extracting features...")
    logger.info("  1st derivative...")
    X_deriv = extract_1st_derivative(X_spec)
    logger.info("  Peak features (Voigt fitting)...")
    X_peak = extract_peak_features(X_spec, wavenumbers)

    # Build conditions
    conditions = {
        "A: spectrum":           X_spec,
        "B: 1st_deriv":         X_deriv,
        "C: clinical":          clinical_data,
        "D: spec+clin":         np.hstack([X_spec, clinical_data]),
        "E: deriv+clin":        np.hstack([X_deriv, clinical_data]),
        "F: 3view":             np.hstack([X_spec, X_deriv, X_peak]),
        "G: 3view+clin":        np.hstack([X_spec, X_deriv, X_peak, clinical_data]),
        "H: deriv+peak+clin":   np.hstack([X_deriv, X_peak, clinical_data]),
    }

    # Run
    logger.info(f"\n[4] Running {len(conditions)} conditions (5-fold CV, sex constraint)...")
    results = []
    for name, X_feat in conditions.items():
        logger.info(f"\n  {name} ({X_feat.shape[1]} features)...")
        r = run_cv(X_feat, groups, sample_ids, sex_arr, name)
        results.append(r)
        logger.info(f"    Det AUC: {r['det_auc']}")
        logger.info(f"    Type F1: {r['type_f1']} (no constraint: {r['type_f1_nc']})")

    # Summary
    logger.info("\n" + "=" * 90)
    logger.info("  RESULTS — 7-cancer, full cohort, sex constraint, mean agg")
    logger.info("=" * 90)

    rows = []
    for r in results:
        rows.append({
            "Condition": r["condition"],
            "Feat": r["n_features"],
            "Det_AUC": r["det_auc"],
            "Type_F1": r["type_f1"],
            "F1_noC": r["type_f1_nc"],
        })

    df = pd.DataFrame(rows)
    logger.info(f"\n{df.to_string(index=False)}")

    # Save
    df.to_csv(out_dir / "results.csv", index=False)
    with open(out_dir / "detail.json", "w") as f:
        json.dump(results, f, indent=2)

    best = max(results, key=lambda r: r["type_f1_mean"])
    logger.info(f"\n  ★ Best: {best['condition']} = {best['type_f1']}")

    elapsed = datetime.now() - t0
    logger.info(f"\n  Completed in {elapsed.total_seconds():.0f}s")


if __name__ == "__main__":
    main()
