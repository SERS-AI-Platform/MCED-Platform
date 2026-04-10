"""
Peak Fitting Ablation Study — Literature-Based Voigt Fitting

Uses 17 known SERS peak positions from metabolite profiling data
and fits Voigt profiles to extract physically meaningful features:
  - Peak area (∝ concentration)
  - Peak height
  - FWHM (molecular environment)
  - Peak position shift (Δν from reference)

Compares against full spectrum and 1st derivative baselines.

Models: LR, RF, XGBoost
Evaluation: 5-fold StratifiedGroupKFold CV, mean aggregation

Usage:
    python scripts/analysis/peak_fitting_ablation.py
"""

from __future__ import annotations

import sys
import logging
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import voigt_profile
from scipy.signal import savgol_filter
from scipy.integrate import trapezoid
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score
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
# 1. Known Peak Table (from metabolite_profiling + literature)
# =============================================================================

# 17 known SERS peaks from metabolite_peak_assignments.csv
# + 10 biologically-defined bands from metabolite_band_statistics.csv
# Merged with literature cancer biomarker assignments

KNOWN_PEAKS = [
    # (center_cm1, name, band_half_width_cm1, biological_assignment)
    (448.1,  "ring_deform",     15, "Uric acid ring deformation / Creatinine"),
    (538.7,  "SS_stretch",      15, "S-S disulfide / Cysteine / Adenine"),
    (617.7,  "CS_stretch",      15, "C-S stretch (Cysteine, thiol)"),
    (683.3,  "creatinine",      15, "Creatinine / Guanine / Purine ring"),
    (723.8,  "adenine",         15, "Adenine / Hypoxanthine ring breathing"),
    (795.1,  "hippuric",        15, "Hippuric acid / Kynurenine"),
    (849.1,  "tyrosine",        15, "Tyrosine Fermi resonance / Cysteine"),
    (895.4,  "uric_acid",       15, "Uric acid C-O-H / C-C stretch"),
    (933.9,  "creatinine2",     15, "Creatinine C-C-N / C-N stretch"),
    (999.5,  "phe_urea",        15, "Phenylalanine ring / Urea C-N (dominant)"),
    (1147.9, "uric_CN",         15, "Uric acid C-N / Xanthine"),
    (1230.8, "amide_III",       20, "Amide III / Tryptophan"),
    (1292.5, "CH2_twist",       15, "CH₂ twist (proteins) / nucleic acids"),
    (1352.3, "trp_fermi",       15, "Tryptophan Fermi / Adenine-Guanine"),
    (1448.7, "CH2_deform",      20, "CH₂/CH₃ deformation (lipids, proteins)"),
    (1597.1, "purine_CC",       20, "Purine C=C / Adenine / Phenylalanine"),
    (1651.1, "amide_I",         20, "Amide I (protein secondary structure)"),
]


# =============================================================================
# 2. Voigt Profile Fitting
# =============================================================================

def voigt_func(x, amplitude, center, sigma, gamma):
    """Voigt profile: convolution of Gaussian (sigma) and Lorentzian (gamma).

    Physical meaning:
    - Gaussian component: inhomogeneous broadening (temperature, disorder)
    - Lorentzian component: natural linewidth (lifetime broadening)
    - Voigt = real-world Raman peak shape
    """
    return amplitude * voigt_profile(x - center, sigma, gamma)


def fit_single_peak_voigt(x_region, y_region, center_guess, half_width):
    """Fit a single Voigt profile to a spectral region.

    Returns: (amplitude, center, sigma, gamma, area, fwhm, success)
    """
    try:
        # Initial guesses
        amp_guess = np.max(y_region) - np.min(y_region)
        if amp_guess < 1e-10:
            amp_guess = 1e-6
        sigma_guess = half_width / 3
        gamma_guess = half_width / 3

        p0 = [amp_guess, center_guess, sigma_guess, gamma_guess]

        # Bounds
        bounds_lo = [0, center_guess - half_width, 0.1, 0.1]
        bounds_hi = [amp_guess * 10, center_guess + half_width, half_width * 2, half_width * 2]

        popt, pcov = curve_fit(
            voigt_func, x_region, y_region,
            p0=p0, bounds=(bounds_lo, bounds_hi),
            maxfev=2000,
        )

        amplitude, center, sigma, gamma = popt

        # Calculate area (analytical: amplitude × 1 for normalized Voigt)
        # Numerical integration is more reliable
        y_fit = voigt_func(x_region, *popt)
        area = trapezoid(y_fit, x_region)

        # FWHM approximation for Voigt (Thompson et al. 1987)
        fL = 2 * gamma
        fG = 2 * sigma * np.sqrt(2 * np.log(2))
        fwhm = 0.5346 * fL + np.sqrt(0.2166 * fL**2 + fG**2)

        return amplitude, center, sigma, gamma, area, fwhm, True

    except (RuntimeError, ValueError, TypeError):
        return 0.0, center_guess, 1.0, 1.0, 0.0, 0.0, False


def fit_single_peak_gaussian(x_region, y_region, center_guess, half_width):
    """Simpler Gaussian fit as fallback."""
    try:
        def gauss(x, a, mu, sig):
            return a * np.exp(-0.5 * ((x - mu) / sig) ** 2)

        amp_guess = np.max(y_region)
        p0 = [amp_guess, center_guess, half_width / 2]
        bounds_lo = [0, center_guess - half_width, 0.5]
        bounds_hi = [amp_guess * 10, center_guess + half_width, half_width * 2]

        popt, _ = curve_fit(gauss, x_region, y_region, p0=p0,
                           bounds=(bounds_lo, bounds_hi), maxfev=1000)
        a, mu, sig = popt
        area = a * sig * np.sqrt(2 * np.pi)
        fwhm = 2 * sig * np.sqrt(2 * np.log(2))
        return a, mu, sig, 0.0, area, fwhm, True

    except (RuntimeError, ValueError):
        return 0.0, center_guess, 1.0, 0.0, 0.0, 0.0, False


# =============================================================================
# 3. Feature Extraction Methods
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


def extract_full_spectrum(X, wavenumbers):
    return X, [f"x_{w:.2f}" for w in wavenumbers]


def extract_1st_derivative(X, wavenumbers):
    dX = np.apply_along_axis(
        lambda y: savgol_filter(y, window_length=11, polyorder=3, deriv=1),
        axis=1, arr=X,
    )
    return dX, [f"d1_{w:.2f}" for w in wavenumbers]


def extract_voigt_peak_features(X, wavenumbers, use_gaussian_fallback=True):
    """Extract Voigt-fitted peak features at 17 known positions.

    Per peak: area, height, fwhm, position_shift (4 features)
    + Inter-peak ratios for biologically relevant pairs
    """
    n_samples = len(X)
    feature_names = []
    all_features = []

    fit_stats = {"voigt_ok": 0, "gaussian_fallback": 0, "failed": 0}

    # First pass: fit all peaks for all samples
    peak_areas = np.zeros((n_samples, len(KNOWN_PEAKS)))
    peak_heights = np.zeros((n_samples, len(KNOWN_PEAKS)))
    peak_fwhms = np.zeros((n_samples, len(KNOWN_PEAKS)))
    peak_shifts = np.zeros((n_samples, len(KNOWN_PEAKS)))

    for pi, (center, name, half_w, _) in enumerate(KNOWN_PEAKS):
        # Extract region
        lo_wn = center - half_w * 1.5
        hi_wn = center + half_w * 1.5
        mask = (wavenumbers >= lo_wn) & (wavenumbers <= hi_wn)

        if mask.sum() < 5:
            logger.warning(f"  Peak {name} ({center} cm⁻¹): too few points ({mask.sum()})")
            continue

        x_region = wavenumbers[mask]

        for si in range(n_samples):
            y_region = X[si, mask]

            # Shift baseline to 0 (local minimum)
            y_shifted = y_region - np.min(y_region)

            amp, ctr, sigma, gamma, area, fwhm, ok = fit_single_peak_voigt(
                x_region, y_shifted, center, half_w
            )

            if ok:
                fit_stats["voigt_ok"] += 1
            elif use_gaussian_fallback:
                amp, ctr, sigma, gamma, area, fwhm, ok = fit_single_peak_gaussian(
                    x_region, y_shifted, center, half_w
                )
                if ok:
                    fit_stats["gaussian_fallback"] += 1
                else:
                    fit_stats["failed"] += 1
            else:
                fit_stats["failed"] += 1

            peak_areas[si, pi] = area
            peak_heights[si, pi] = amp
            peak_fwhms[si, pi] = fwhm
            peak_shifts[si, pi] = ctr - center

    # Build feature matrix
    for pi, (center, name, _, _) in enumerate(KNOWN_PEAKS):
        feature_names.extend([
            f"{name}_{center:.0f}_area",
            f"{name}_{center:.0f}_height",
            f"{name}_{center:.0f}_fwhm",
            f"{name}_{center:.0f}_shift",
        ])
        all_features.extend([
            peak_areas[:, pi],
            peak_heights[:, pi],
            peak_fwhms[:, pi],
            peak_shifts[:, pi],
        ])

    # Biologically meaningful ratios
    # Based on metabolite_band_statistics.csv significant bands
    ratio_pairs = [
        ("phe_urea", "adenine",    "phe_adenine_ratio"),      # Phe/Adenine
        ("phe_urea", "creatinine", "phe_creatinine_ratio"),   # Phe/Creatinine (normalization)
        ("hippuric", "creatinine", "hippuric_creat_ratio"),   # Hippuric/Creatinine
        ("CS_stretch","creatinine","CS_creat_ratio"),         # C-S/Creatinine (thiol stress)
        ("amide_I",  "CH2_deform", "amideI_CH2_ratio"),      # Protein/Lipid
        ("adenine",  "purine_CC",  "adenine_purine_ratio"),   # Purine metabolism
        ("tyrosine", "phe_urea",   "tyr_phe_ratio"),         # Tyr/Phe (hydroxylation)
    ]

    peak_name_to_idx = {p[1]: i for i, p in enumerate(KNOWN_PEAKS)}
    for name_a, name_b, ratio_name in ratio_pairs:
        idx_a = peak_name_to_idx[name_a]
        idx_b = peak_name_to_idx[name_b]
        # Use area ratio (most physically meaningful)
        ratio = peak_areas[:, idx_a] / (peak_areas[:, idx_b] + 1e-10)
        feature_names.append(ratio_name)
        all_features.append(ratio)

    X_peaks = np.column_stack(all_features)

    logger.info(f"  Voigt fit stats: {fit_stats}")
    logger.info(f"  Total features: {len(feature_names)} "
                f"({len(KNOWN_PEAKS)} peaks × 4 + {len(ratio_pairs)} ratios)")

    return X_peaks, feature_names


def extract_band_integration(X, wavenumbers):
    """Simple band area integration at known positions (no fitting).

    Fastest approach — just integrates intensity in each band window.
    """
    feature_names = []
    all_features = []

    for center, name, half_w, _ in KNOWN_PEAKS:
        lo_wn = center - half_w
        hi_wn = center + half_w
        mask = (wavenumbers >= lo_wn) & (wavenumbers <= hi_wn)

        if mask.sum() < 3:
            continue

        x_region = wavenumbers[mask]

        # Band area (trapezoid integration)
        areas = np.array([trapezoid(X[s, mask], x_region) for s in range(len(X))])
        # Band max intensity
        heights = X[:, mask].max(axis=1)
        # Band mean intensity
        means = X[:, mask].mean(axis=1)

        feature_names.extend([
            f"{name}_{center:.0f}_band_area",
            f"{name}_{center:.0f}_band_max",
            f"{name}_{center:.0f}_band_mean",
        ])
        all_features.extend([areas, heights, means])

    # Same biologically meaningful ratios (using band areas)
    peak_name_to_area = {}
    for center, name, half_w, _ in KNOWN_PEAKS:
        lo_wn = center - half_w
        hi_wn = center + half_w
        mask = (wavenumbers >= lo_wn) & (wavenumbers <= hi_wn)
        if mask.sum() >= 3:
            peak_name_to_area[name] = np.array([
                trapezoid(X[s, mask], wavenumbers[mask]) for s in range(len(X))
            ])

    ratio_pairs = [
        ("phe_urea", "adenine",    "phe_adenine_ratio"),
        ("phe_urea", "creatinine", "phe_creatinine_ratio"),
        ("hippuric", "creatinine", "hippuric_creat_ratio"),
        ("CS_stretch","creatinine","CS_creat_ratio"),
        ("amide_I",  "CH2_deform", "amideI_CH2_ratio"),
        ("adenine",  "purine_CC",  "adenine_purine_ratio"),
        ("tyrosine", "phe_urea",   "tyr_phe_ratio"),
    ]

    for name_a, name_b, ratio_name in ratio_pairs:
        if name_a in peak_name_to_area and name_b in peak_name_to_area:
            ratio = peak_name_to_area[name_a] / (peak_name_to_area[name_b] + 1e-10)
            feature_names.append(ratio_name)
            all_features.append(ratio)

    X_band = np.column_stack(all_features)
    return X_band, feature_names


def extract_combined(X, wavenumbers):
    """1st derivative (full) + Voigt peak features — best of both worlds."""
    X_deriv, names_deriv = extract_1st_derivative(X, wavenumbers)
    X_peak, names_peak = extract_voigt_peak_features(X, wavenumbers)

    X_combined = np.hstack([X_deriv, X_peak])
    names_combined = names_deriv + names_peak
    return X_combined, names_combined


# =============================================================================
# 4. Models & Evaluation (same as feature_representation_ablation.py)
# =============================================================================

CANCER_TYPES = ["PRO", "BRE", "OVA", "LUN", "CRC", "PAN", "BLC"]
NON_CANCER = ["NOR", "DIA", "TBN"]


def build_models(random_state=42):
    models = {
        "LR": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=1000, solver="saga",
                             class_weight="balanced", random_state=random_state),
        ),
        "RF": make_pipeline(
            StandardScaler(),
            RandomForestClassifier(n_estimators=300, max_depth=None,
                                  class_weight="balanced", random_state=random_state, n_jobs=4),
        ),
    }
    if HAS_XGB:
        models["XGB"] = make_pipeline(
            StandardScaler(),
            XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                         subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                         n_jobs=4, tree_method="hist", verbosity=0,
                         random_state=random_state, eval_metric="mlogloss"),
        )
    return models


def run_cv(X, groups, sample_ids, feature_name, model_name, model_factory,
           n_splits=5, random_state=42):
    cancer_set = set(CANCER_TYPES)
    binary_labels = np.array([1 if g in cancer_set else 0 for g in groups])
    cancer_type_labels = np.array([
        CANCER_TYPES.index(g) if g in cancer_set else -1 for g in groups
    ])

    valid_mask = np.isin(groups, CANCER_TYPES + NON_CANCER)
    X, groups, sample_ids = X[valid_mask], groups[valid_mask], sample_ids[valid_mask]
    binary_labels, cancer_type_labels = binary_labels[valid_mask], cancer_type_labels[valid_mask]

    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_det_aucs, fold_type_f1s = [], []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, binary_labels, groups=sample_ids)):
        X_train, X_test = X[train_idx], X[test_idx]
        yb_train, yb_test = binary_labels[train_idx], binary_labels[test_idx]
        yc_train, yc_test = cancer_type_labels[train_idx], cancer_type_labels[test_idx]

        # Stage 1: Binary
        model_s1 = model_factory()
        model_s1.fit(X_train, yb_train)
        s1_prob = model_s1.predict_proba(X_test)[:, 1]
        try:
            det_auc = roc_auc_score(yb_test, s1_prob)
        except ValueError:
            det_auc = np.nan
        fold_det_aucs.append(det_auc)

        # Stage 2: Cancer type
        cancer_train = yc_train >= 0
        cancer_test = yc_test >= 0
        if cancer_train.sum() > 0 and cancer_test.sum() > 0:
            present = sorted(set(yc_train[cancer_train]))
            if len(present) >= 2:
                label_map = {c: i for i, c in enumerate(present)}
                inv_map = {i: c for c, i in label_map.items()}
                yc_tr_local = np.array([label_map[c] for c in yc_train[cancer_train]])
                yc_te_local = np.array([label_map.get(c, -1) for c in yc_test[cancer_test]])
                seen = yc_te_local >= 0
                if seen.sum() > 0:
                    model_s2 = model_factory()
                    model_s2.fit(X_train[cancer_train], yc_tr_local)
                    pred_local = model_s2.predict(X_test[cancer_test][seen])
                    yc_pred = np.array([inv_map[p] for p in pred_local])
                    yc_true = np.array([inv_map[t] for t in yc_te_local[seen]])
                    type_f1 = f1_score(yc_true, yc_pred, average="macro", zero_division=0)
                else:
                    type_f1 = np.nan
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
    out_dir = RESULTS_DIR / "peak_fitting_ablation"
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
    logger.info("  Peak Fitting Ablation — Literature-Based Voigt Fitting")
    logger.info("=" * 64)

    # Load
    logger.info("\n[1] Loading data...")
    df, feat_cols, wavenumbers = load_data()
    df_agg = aggregate_mean(df, feat_cols)
    X_raw = df_agg[feat_cols].values
    groups = df_agg["group"].values
    sample_ids = df_agg["sample_id"].values
    logger.info(f"  Samples: {len(X_raw)}, Range: {wavenumbers[0]:.1f}–{wavenumbers[-1]:.1f} cm⁻¹")

    # Feature extraction
    METHODS = {
        "full_spectrum":     extract_full_spectrum,
        "1st_derivative":    extract_1st_derivative,
        "band_integration":  extract_band_integration,
        "voigt_fitting":     extract_voigt_peak_features,
        "deriv+voigt":       extract_combined,
    }

    logger.info("\n[2] Extracting features...")
    feature_sets = {}
    for name, method in METHODS.items():
        logger.info(f"\n  === {name} ===")
        X_feat, feat_names = method(X_raw, wavenumbers)
        n_nan = np.isnan(X_feat).sum() + np.isinf(X_feat).sum()
        if n_nan > 0:
            logger.warning(f"  ⚠ {n_nan} NaN/Inf values → replacing with 0")
            X_feat = np.nan_to_num(X_feat, nan=0.0, posinf=0.0, neginf=0.0)
        feature_sets[name] = (X_feat, feat_names)
        logger.info(f"  Shape: {X_feat.shape}")

    # Run experiments
    logger.info("\n[3] Running CV experiments...")
    results = []
    for feat_name, (X_feat, _) in feature_sets.items():
        models = build_models()
        for model_name, model_pipeline in models.items():
            logger.info(f"\n  {feat_name} × {model_name}...")
            def make_model(mp=model_pipeline):
                from sklearn.base import clone
                return clone(mp)

            result = run_cv(X_feat, groups, sample_ids, feat_name, model_name, make_model)
            results.append(result)
            logger.info(
                f"    Det AUC: {result['det_auc_mean']:.4f}±{result['det_auc_std']:.4f} | "
                f"Type F1: {result['type_f1_mean']:.4f}±{result['type_f1_std']:.4f} | "
                f"Features: {result['n_features']}"
            )

    # Summary
    logger.info("\n" + "=" * 95)
    logger.info("  RESULTS SUMMARY")
    logger.info("=" * 95)

    rows = []
    for r in results:
        rows.append({
            "Feature": r["feature"],
            "Model": r["model"],
            "N_Feat": r["n_features"],
            "Det_AUC": f"{r['det_auc_mean']:.4f}±{r['det_auc_std']:.4f}",
            "Type_F1": f"{r['type_f1_mean']:.4f}±{r['type_f1_std']:.4f}",
            "det_auc_raw": r["det_auc_mean"],
            "type_f1_raw": r["type_f1_mean"],
        })

    summary_df = pd.DataFrame(rows)
    logger.info(f"\n{summary_df[['Feature','Model','N_Feat','Det_AUC','Type_F1']].to_string(index=False)}")

    # Save
    summary_df[['Feature','Model','N_Feat','Det_AUC','Type_F1']].to_csv(
        out_dir / "results.csv", index=False
    )

    detail = []
    for r in results:
        d = {k: v for k, v in r.items() if k not in ("fold_det_aucs", "fold_type_f1s")}
        d["fold_det_aucs"] = [float(x) for x in r["fold_det_aucs"]]
        d["fold_type_f1s"] = [float(x) for x in r["fold_type_f1s"]]
        detail.append(d)
    with open(out_dir / "detail.json", "w") as f:
        json.dump(detail, f, indent=2)

    # Best results
    best_f1 = max(results, key=lambda r: r["type_f1_mean"])
    best_auc = max(results, key=lambda r: r["det_auc_mean"])
    logger.info(f"\n  ★ Best Type F1: {best_f1['feature']} × {best_f1['model']} "
                f"= {best_f1['type_f1_mean']:.4f}±{best_f1['type_f1_std']:.4f} ({best_f1['n_features']} feat)")
    logger.info(f"  ★ Best Det AUC: {best_auc['feature']} × {best_auc['model']} "
                f"= {best_auc['det_auc_mean']:.4f}±{best_auc['det_auc_std']:.4f}")

    # Efficiency analysis
    logger.info("\n  --- Efficiency: Features vs Performance ---")
    for r in sorted(results, key=lambda r: r["n_features"]):
        if r["model"] == "LR":  # LR only for clean comparison
            logger.info(f"    {r['feature']:20s}: {r['n_features']:5d} feat → F1 {r['type_f1_mean']:.4f}")

    elapsed = datetime.now() - t0
    logger.info(f"\n  Completed in {elapsed.total_seconds():.0f}s")
    logger.info(f"  Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
