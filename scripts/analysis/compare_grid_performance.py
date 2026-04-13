"""
Grid Consistency Experiment: Dynamic Grid vs Fixed Grid

Compares model performance between:
  1. Dynamic grid (make_common_grid — data-dependent)
  2. Fixed grid (config-defined — 402.0–2198.0, 935 points)

Both use identical preprocessing (rolling min baseline, SNV, Savgol).
Evaluation: 5-fold StratifiedGroupKFold, Logistic Regression.
8-class cancer set (PRO, BRE, OVA, LUN, CRC, CPAN, SPAN, BLC).

Metrics: Cancer Screening AUC, Cancer Screening F1, Cancer Type ID F1-macro
"""

import sys
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.sers.config import load_config, RAW_DATA_DIR
from src.sers.io import find_spectra, read_spectrum, parse_filename, make_common_grid, make_fixed_grid
from src.sers.preprocessing import (
    trim_spectrum, smooth, baseline_correction, normalize_spectrum, resample,
    FINGERPRINT_REGION,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

CANCER_TYPES = ("PRO", "OVA", "LUN", "CRC", "PAN", "BLC")
NON_CANCER = ("NOR", "DIA", "HBP", "H.D.")


def preprocess_spectrum(x, y, grid, trim_region=FINGERPRINT_REGION):
    """Standard preprocessing: trim → smooth → baseline → SNV → resample."""
    x, y_proc = trim_spectrum(x, y.copy(), region=trim_region)
    y_proc = smooth(y_proc, window_length=11, polyorder=3)
    y_proc = baseline_correction(y_proc, window=101)
    y_proc = normalize_spectrum(y_proc, method="snv")
    y_grid = resample(x, y_proc, grid)
    return y_grid


def build_dataset(raw_spectra, grid, trim_region=FINGERPRINT_REGION):
    """Build feature matrix from raw spectra on given grid."""
    mask = (grid >= trim_region[0]) & (grid <= trim_region[1])
    proc_grid = grid[mask]

    rows = []
    for (group, sample_id, replicate), (x, y) in raw_spectra.items():
        try:
            y_grid = preprocess_spectrum(x, y, proc_grid, trim_region)
            rows.append({
                "group": group,
                "sample_id": sample_id,
                "replicate": replicate,
                "features": y_grid,
            })
        except Exception as e:
            logger.warning(f"  Skip {group}/{sample_id}/{replicate}: {e}")
    return rows, proc_grid


def aggregate_medoid(rows):
    """Medoid aggregation per sample, create binary + cancer-type labels."""
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["group"], r["sample_id"])].append(r["features"])

    X_list, groups, sample_ids = [], [], []
    for (group, sid), spectra in grouped.items():
        spectra = np.array(spectra)
        if len(spectra) == 1:
            chosen = spectra[0]
        else:
            corr = np.corrcoef(spectra)
            medoid_idx = corr.mean(axis=1).argmax()
            chosen = spectra[medoid_idx]
        X_list.append(chosen)
        groups.append(group)
        sample_ids.append(sid)

    X = np.array(X_list)
    groups = np.array(groups)
    sample_ids = np.array(sample_ids)

    valid = np.isin(groups, list(CANCER_TYPES) + list(NON_CANCER))
    X, groups, sample_ids = X[valid], groups[valid], sample_ids[valid]

    binary = np.array([1 if g in CANCER_TYPES else 0 for g in groups])
    type_map = {t: i for i, t in enumerate(CANCER_TYPES)}
    cancer_type = np.array([type_map.get(g, -1) for g in groups])

    return X, binary, cancer_type, groups, sample_ids


def evaluate(name, X, binary, cancer_type, groups, sample_ids, n_splits=5):
    """5-fold StratifiedGroupKFold evaluation."""
    logger.info(f"\n{'='*60}")
    logger.info(f"  {name}")
    logger.info(f"  Samples: {len(X)} | Features: {X.shape[1]}")
    logger.info(f"  Cancer: {(binary==1).sum()} | Non-cancer: {(binary==0).sum()}")
    logger.info(f"{'='*60}")

    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    s1_aucs, s1_f1s, s2_f1s = [], [], []
    fold_details = []

    for fold_i, (train_idx, val_idx) in enumerate(skf.split(X, binary, groups=sample_ids)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = binary[train_idx], binary[val_idx]
        ct_train, ct_val = cancer_type[train_idx], cancer_type[val_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        # Cancer Screening (binary)
        clf_s1 = LogisticRegression(C=1.0, max_iter=1000, solver="saga",
                                     class_weight="balanced", random_state=42)
        clf_s1.fit(X_train_s, y_train)
        prob_s1 = clf_s1.predict_proba(X_val_s)[:, 1]
        pred_s1 = (prob_s1 > 0.5).astype(int)

        auc_s1 = roc_auc_score(y_val, prob_s1) if len(np.unique(y_val)) > 1 else float("nan")
        f1_s1 = f1_score(y_val, pred_s1, average="binary")

        # Cancer Type ID (multiclass)
        cancer_train = ct_train >= 0
        cancer_val = ct_val >= 0
        f1_s2 = float("nan")
        if cancer_train.sum() > 5 and cancer_val.sum() > 0:
            clf_s2 = LogisticRegression(C=1.0, max_iter=1000, solver="saga",
                                         class_weight="balanced", random_state=42,
                                         multi_class="multinomial")
            clf_s2.fit(X_train_s[cancer_train], ct_train[cancer_train])
            pred_s2 = clf_s2.predict(X_val_s[cancer_val])
            f1_s2 = f1_score(ct_val[cancer_val], pred_s2, average="macro", zero_division=0)

        s1_aucs.append(auc_s1)
        s1_f1s.append(f1_s1)
        s2_f1s.append(f1_s2)

        fold_details.append({
            "fold": fold_i + 1,
            "s1_auc": round(auc_s1, 4),
            "s1_f1": round(f1_s1, 4),
            "s2_f1_macro": round(f1_s2, 4),
        })

        logger.info(f"  Fold {fold_i+1}: Screening AUC={auc_s1:.4f}  "
                     f"Screening F1={f1_s1:.4f}  Type ID F1={f1_s2:.4f}")

    results = {
        "method": name,
        "screening_auc_mean": round(np.nanmean(s1_aucs), 4),
        "screening_auc_std": round(np.nanstd(s1_aucs), 4),
        "screening_f1_mean": round(np.nanmean(s1_f1s), 4),
        "screening_f1_std": round(np.nanstd(s1_f1s), 4),
        "type_id_f1_mean": round(np.nanmean(s2_f1s), 4),
        "type_id_f1_std": round(np.nanstd(s2_f1s), 4),
        "n_features": int(X.shape[1]),
        "n_samples": int(len(X)),
        "folds": fold_details,
    }

    logger.info(f"\n  >> {name} Summary:")
    logger.info(f"     Screening AUC:  {results['screening_auc_mean']:.4f} ± {results['screening_auc_std']:.4f}")
    logger.info(f"     Screening F1:   {results['screening_f1_mean']:.4f} ± {results['screening_f1_std']:.4f}")
    logger.info(f"     Type ID F1:     {results['type_id_f1_mean']:.4f} ± {results['type_id_f1_std']:.4f}")

    return results


def main():
    logger.info("=" * 60)
    logger.info("  Grid Consistency Experiment")
    logger.info("  Dynamic Grid vs Fixed Grid — Model Performance Comparison")
    logger.info("=" * 60)

    config = load_config("config/config.yaml")
    data_dir = Path(RAW_DATA_DIR)
    files = list(find_spectra(data_dir, pattern="*.csv"))
    logger.info(f"Found {len(files)} spectrum files")

    raw_spectra = {}
    for f in files:
        try:
            spec_id = parse_filename(
                f, fallback_group=config.folder_to_group.get(f.parent.name, "UNK")
            )
            x, y = read_spectrum(f)
            raw_spectra[(spec_id.group, spec_id.sample_id, spec_id.replicate)] = (x, y)
        except Exception:
            continue
    logger.info(f"Loaded {len(raw_spectra)} spectra")

    # ── Grid 1: Dynamic (original behaviour) ──
    logger.info("\n\n[1/2] Dynamic Grid (make_common_grid)")
    x_arrays = [x for x, y in raw_spectra.values()]
    dynamic_grid = make_common_grid(x_arrays)
    logger.info(f"  Grid: {len(dynamic_grid)} pts, "
                f"{dynamic_grid[0]:.2f} – {dynamic_grid[-1]:.2f} cm⁻¹")

    t0 = time.time()
    rows_dyn, grid_dyn = build_dataset(raw_spectra, dynamic_grid)
    X_dyn, bin_dyn, ct_dyn, grp_dyn, sid_dyn = aggregate_medoid(rows_dyn)
    t_dyn = time.time() - t0
    logger.info(f"  Preprocessing: {t_dyn:.1f}s | Grid points after trim: {grid_dyn.shape[0]}")
    res_dyn = evaluate("Dynamic Grid", X_dyn, bin_dyn, ct_dyn, grp_dyn, sid_dyn)
    res_dyn["preprocessing_time_s"] = round(t_dyn, 1)
    res_dyn["grid_range"] = f"{dynamic_grid[0]:.2f} – {dynamic_grid[-1]:.2f}"
    res_dyn["grid_points_raw"] = len(dynamic_grid)

    # ── Grid 2: Fixed (config-defined) ──
    logger.info("\n\n[2/2] Fixed Grid (config-defined)")
    fixed_grid = make_fixed_grid(config)
    if fixed_grid is None:
        logger.error("No fixed_grid defined in config! Add preprocessing.fixed_grid to config.yaml.")
        sys.exit(1)
    logger.info(f"  Grid: {len(fixed_grid)} pts, "
                f"{fixed_grid[0]:.2f} – {fixed_grid[-1]:.2f} cm⁻¹")

    t0 = time.time()
    rows_fix, grid_fix = build_dataset(raw_spectra, fixed_grid)
    X_fix, bin_fix, ct_fix, grp_fix, sid_fix = aggregate_medoid(rows_fix)
    t_fix = time.time() - t0
    logger.info(f"  Preprocessing: {t_fix:.1f}s | Grid points after trim: {grid_fix.shape[0]}")
    res_fix = evaluate("Fixed Grid", X_fix, bin_fix, ct_fix, grp_fix, sid_fix)
    res_fix["preprocessing_time_s"] = round(t_fix, 1)
    res_fix["grid_range"] = f"{fixed_grid[0]:.2f} – {fixed_grid[-1]:.2f}"
    res_fix["grid_points_raw"] = len(fixed_grid)

    # ── Comparison ──
    logger.info("\n\n" + "=" * 60)
    logger.info("  COMPARISON: Dynamic Grid vs Fixed Grid")
    logger.info("=" * 60)

    diff_auc = res_fix["screening_auc_mean"] - res_dyn["screening_auc_mean"]
    diff_f1 = res_fix["type_id_f1_mean"] - res_dyn["type_id_f1_mean"]

    logger.info(f"                     {'Dynamic':>12s}  {'Fixed':>12s}  {'Δ':>8s}")
    logger.info(f"  Features:          {res_dyn['n_features']:>12d}  {res_fix['n_features']:>12d}")
    logger.info(f"  Screening AUC:     {res_dyn['screening_auc_mean']:>12.4f}  "
                f"{res_fix['screening_auc_mean']:>12.4f}  {diff_auc:>+8.4f}")
    logger.info(f"  Type ID F1:        {res_dyn['type_id_f1_mean']:>12.4f}  "
                f"{res_fix['type_id_f1_mean']:>12.4f}  {diff_f1:>+8.4f}")
    logger.info("=" * 60)

    if abs(diff_auc) < 0.005 and abs(diff_f1) < 0.01:
        logger.info("  ✓ Fixed grid achieves comparable performance — safe to adopt.")
    elif diff_auc > 0 and diff_f1 > 0:
        logger.info("  ✓ Fixed grid IMPROVES performance!")
    else:
        logger.info("  ⚠ Performance difference detected — investigate before adopting.")

    # Save results
    out_dir = PROJECT_ROOT / "results" / "grid_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    comparison = {
        "experiment": "grid_consistency",
        "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "cancer_types": list(CANCER_TYPES),
        "non_cancer": list(NON_CANCER),
        "n_folds": 5,
        "model": "LogisticRegression (C=1.0, saga, balanced)",
        "dynamic_grid": res_dyn,
        "fixed_grid": res_fix,
        "delta": {
            "screening_auc": round(diff_auc, 4),
            "type_id_f1": round(diff_f1, 4),
        },
    }

    with open(out_dir / "grid_comparison_results.json", "w") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    logger.info(f"\n  Results saved: {out_dir / 'grid_comparison_results.json'}")


if __name__ == "__main__":
    main()
