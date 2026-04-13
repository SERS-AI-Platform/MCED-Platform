"""
Baseline Correction Method Comparison: Rolling Minimum vs ALS

Compares two baseline correction approaches on the full SERS dataset:
  1. Rolling Minimum (current pipeline, window=101)
  2. ALS - Asymmetric Least Squares (Eilers & Boelens 2005)

Evaluation: 5-fold StratifiedGroupKFold with Logistic Regression
(fastest reliable model, known best single-model performer).

Metrics: Stage 1 AUC, Stage 1 F1, Stage 2 F1-macro
"""

import sys
import logging
import time
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.sers.config import load_config, RAW_DATA_DIR
from src.sers.io import find_spectra, read_spectrum, parse_filename, make_common_grid
from src.sers.preprocessing import (
    trim_spectrum, smooth, baseline_correction, normalize_spectrum, resample,
    FINGERPRINT_REGION,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Cancer / non-cancer group definitions ──
CANCER_TYPES = ("PRO", "BRE", "OVA", "LUN", "CRC", "PAN", "SPAN", "BLC")
NON_CANCER = ("NOR", "DIA", "HBP", "H.D.")


# =============================================================================
# Baseline correction implementations
# =============================================================================
def baseline_rolling_min(y, window=101):
    """Current pipeline: rolling minimum."""
    return baseline_correction(y, window=window)


def baseline_als(y, lam=1e6, p=0.01, niter=10):
    """Asymmetric Least Squares (Eilers & Boelens 2005)."""
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    w = np.ones(L)
    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.T)
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y <= z)
    return y - z


# =============================================================================
# Preprocessing with pluggable baseline
# =============================================================================
def preprocess_spectrum(x, y, grid, baseline_fn, trim_region=FINGERPRINT_REGION):
    """Preprocess a single spectrum with a given baseline correction function."""
    x, y_proc = trim_spectrum(x, y.copy(), region=trim_region)
    y_proc = smooth(y_proc, window_length=11, polyorder=3)
    y_proc = baseline_fn(y_proc)
    y_proc = normalize_spectrum(y_proc, method="snv")
    y_grid = resample(x, y_proc, grid)
    return y_grid


def build_dataset(raw_spectra, grid, baseline_fn, trim_region=FINGERPRINT_REGION):
    """Build feature matrix using specified baseline method."""
    # Trim grid to fingerprint region
    mask = (grid >= trim_region[0]) & (grid <= trim_region[1])
    proc_grid = grid[mask]

    rows = []
    for (group, sample_id, replicate), (x, y) in raw_spectra.items():
        try:
            y_grid = preprocess_spectrum(x, y, proc_grid, baseline_fn, trim_region)
            rows.append({
                "group": group,
                "sample_id": sample_id,
                "replicate": replicate,
                "features": y_grid,
            })
        except Exception as e:
            logger.warning(f"  Skip {group}/{sample_id}/{replicate}: {e}")
    return rows, proc_grid


# =============================================================================
# Aggregation + label creation
# =============================================================================
def aggregate_and_label(rows):
    """Medoid aggregation per sample, then create binary + cancer-type labels."""
    # Group by (group, sample_id) → medoid selection
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

    # Filter to known groups
    valid = np.isin(groups, list(CANCER_TYPES) + list(NON_CANCER))
    X, groups, sample_ids = X[valid], groups[valid], sample_ids[valid]

    # Labels
    binary = np.array([1 if g in CANCER_TYPES else 0 for g in groups])
    type_map = {t: i for i, t in enumerate(CANCER_TYPES)}
    cancer_type = np.array([type_map.get(g, -1) for g in groups])

    return X, binary, cancer_type, groups, sample_ids


# =============================================================================
# Evaluation
# =============================================================================
def evaluate_baseline_method(name, X, binary, cancer_type, groups, sample_ids, n_splits=5):
    """5-fold StratifiedGroupKFold with Logistic Regression."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating: {name}")
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

        # Stage 1: Binary
        clf_s1 = LogisticRegression(C=1.0, max_iter=1000, solver="saga",
                                     class_weight="balanced", random_state=42)
        clf_s1.fit(X_train_s, y_train)
        prob_s1 = clf_s1.predict_proba(X_val_s)[:, 1]
        pred_s1 = (prob_s1 > 0.5).astype(int)

        auc_s1 = roc_auc_score(y_val, prob_s1) if len(np.unique(y_val)) > 1 else float("nan")
        f1_s1 = f1_score(y_val, pred_s1, average="binary")

        # Stage 2: Cancer type (train on cancer samples only)
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
            "s1_auc": auc_s1,
            "s1_f1": f1_s1,
            "s2_f1_macro": f1_s2,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
        })

        logger.info(f"  Fold {fold_i+1}: S1 AUC={auc_s1:.4f}  S1 F1={f1_s1:.4f}  S2 F1={f1_s2:.4f}")

    results = {
        "method": name,
        "s1_auc_mean": np.nanmean(s1_aucs),
        "s1_auc_std": np.nanstd(s1_aucs),
        "s1_f1_mean": np.nanmean(s1_f1s),
        "s1_f1_std": np.nanstd(s1_f1s),
        "s2_f1_mean": np.nanmean(s2_f1s),
        "s2_f1_std": np.nanstd(s2_f1s),
        "folds": fold_details,
    }

    logger.info(f"\n  >> {name} Summary:")
    logger.info(f"     S1 AUC:      {results['s1_auc_mean']:.4f} ± {results['s1_auc_std']:.4f}")
    logger.info(f"     S1 F1:       {results['s1_f1_mean']:.4f} ± {results['s1_f1_std']:.4f}")
    logger.info(f"     S2 F1-macro: {results['s2_f1_mean']:.4f} ± {results['s2_f1_std']:.4f}")

    return results


# =============================================================================
# Main
# =============================================================================
def main():
    logger.info("=" * 60)
    logger.info("Baseline Correction Comparison: Rolling Min vs ALS")
    logger.info("=" * 60)

    # Load config & raw data
    config = load_config("config/config.yaml")
    data_dir = Path(RAW_DATA_DIR)
    files = list(find_spectra(data_dir, pattern="*.csv"))
    logger.info(f"Found {len(files)} spectrum files")

    raw_spectra = {}
    for f in files:
        try:
            spec_id = parse_filename(f, fallback_group=config.folder_to_group.get(f.parent.name, "UNK"))
            x, y = read_spectrum(f)
            raw_spectra[(spec_id.group, spec_id.sample_id, spec_id.replicate)] = (x, y)
        except Exception as e:
            continue
    logger.info(f"Loaded {len(raw_spectra)} spectra")

    # Common grid
    x_arrays = [x for x, y in raw_spectra.values()]
    common_grid = make_common_grid(x_arrays)

    # ── Method 1: Rolling Minimum (current) ──
    logger.info("\n\n[1/3] Building dataset: Rolling Minimum (window=101)")
    t0 = time.time()
    rows_rm, grid_rm = build_dataset(raw_spectra, common_grid, baseline_rolling_min)
    X_rm, bin_rm, ct_rm, grp_rm, sid_rm = aggregate_and_label(rows_rm)
    t_rm = time.time() - t0
    logger.info(f"  Preprocessing time: {t_rm:.1f}s")
    res_rm = evaluate_baseline_method("Rolling Minimum", X_rm, bin_rm, ct_rm, grp_rm, sid_rm)
    res_rm["preprocess_time_s"] = t_rm

    # ── Method 2: ALS (default params) ──
    logger.info("\n\n[2/3] Building dataset: ALS (lam=1e6, p=0.01, niter=10)")
    t0 = time.time()
    rows_als, grid_als = build_dataset(raw_spectra, common_grid,
                                        lambda y: baseline_als(y, lam=1e6, p=0.01, niter=10))
    X_als, bin_als, ct_als, grp_als, sid_als = aggregate_and_label(rows_als)
    t_als = time.time() - t0
    logger.info(f"  Preprocessing time: {t_als:.1f}s")
    res_als = evaluate_baseline_method("ALS (lam=1e6, p=0.01)", X_als, bin_als, ct_als, grp_als, sid_als)
    res_als["preprocess_time_s"] = t_als

    # ── Method 3: ALS with different params ──
    logger.info("\n\n[3/3] Building dataset: ALS (lam=1e7, p=0.001, niter=15)")
    t0 = time.time()
    rows_als2, grid_als2 = build_dataset(raw_spectra, common_grid,
                                          lambda y: baseline_als(y, lam=1e7, p=0.001, niter=15))
    X_als2, bin_als2, ct_als2, grp_als2, sid_als2 = aggregate_and_label(rows_als2)
    t_als2 = time.time() - t0
    logger.info(f"  Preprocessing time: {t_als2:.1f}s")
    res_als2 = evaluate_baseline_method("ALS (lam=1e7, p=0.001)", X_als2, bin_als2, ct_als2, grp_als2, sid_als2)
    res_als2["preprocess_time_s"] = t_als2

    # ── Final comparison table ──
    logger.info("\n\n" + "=" * 70)
    logger.info("FINAL COMPARISON")
    logger.info("=" * 70)

    comparison = []
    for r in [res_rm, res_als, res_als2]:
        comparison.append({
            "Method": r["method"],
            "S1 AUC": f"{r['s1_auc_mean']:.4f} ± {r['s1_auc_std']:.4f}",
            "S1 F1": f"{r['s1_f1_mean']:.4f} ± {r['s1_f1_std']:.4f}",
            "S2 F1-macro": f"{r['s2_f1_mean']:.4f} ± {r['s2_f1_std']:.4f}",
            "Time (s)": f"{r['preprocess_time_s']:.1f}",
        })

    df_cmp = pd.DataFrame(comparison)
    logger.info(f"\n{df_cmp.to_string(index=False)}")

    # Save results
    output_path = PROJECT_ROOT / "results" / "baseline_comparison.csv"
    df_cmp.to_csv(output_path, index=False)
    logger.info(f"\nResults saved to: {output_path}")

    # Determine winner
    all_res = [res_rm, res_als, res_als2]
    best_s1 = max(all_res, key=lambda r: r["s1_auc_mean"])
    best_s2 = max(all_res, key=lambda r: r["s2_f1_mean"])

    logger.info(f"\n  Best S1 AUC:      {best_s1['method']} ({best_s1['s1_auc_mean']:.4f})")
    logger.info(f"  Best S2 F1-macro: {best_s2['method']} ({best_s2['s2_f1_mean']:.4f})")


if __name__ == "__main__":
    main()
