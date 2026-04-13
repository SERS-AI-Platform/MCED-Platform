"""
Baseline Correction Comparison — Phase Q Pipeline

Uses the exact same evaluation as Phase Q (run_train_val_test.py):
  - 60/20/20 train/val/test split at subject level
  - Logistic Regression + sex-based biological constraint
  - Aggregation: none (all spectra)
  - 5 random repeats
  - Metrics: Det AUC, Det Sensitivity/Specificity, Id F1-macro

Compares:
  1. Rolling Minimum (current, window=101)
  2. ALS (lam=1e6, p=0.01, niter=10)
  3. ALS (lam=1e7, p=0.001, niter=15)
"""

from __future__ import annotations

import sys
import json
import logging
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.sers.config import load_config, RAW_DATA_DIR
from src.sers.io import find_spectra, read_spectrum, parse_filename, make_common_grid
from src.sers.preprocessing import (
    trim_spectrum, smooth, baseline_correction, normalize_spectrum, resample,
    FINGERPRINT_REGION, save_processed_spectra,
)

# Reuse Phase Q pipeline components
from models.train import (
    load_processed_spectra, get_feature_columns,
    apply_class_selection, resolve_aliases, aggregate_replicates,
    create_labels, ModelConfig,
)
from models.run_train_val_test import (
    load_clinical, merge_clinical,
    create_train_val_test_split, apply_sex_constraint,
    train_and_evaluate,
    CANCER_TYPES, NON_CANCER,
)

import yaml
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# ALS Baseline Correction
# =============================================================================
def baseline_als(y, lam=1e6, p=0.01, niter=10):
    """Asymmetric Least Squares baseline correction."""
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2)).tocsc()
    w = np.ones(L)
    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.T)
        z = spsolve(Z.tocsc(), w * y)
        w = p * (y > z) + (1 - p) * (y <= z)
    return y - z


# =============================================================================
# Build processed_spectra.csv with custom baseline
# =============================================================================
def build_processed_csv(raw_spectra, common_grid, baseline_fn, config, output_path):
    """Preprocess all spectra with given baseline function, save as CSV."""
    trim_region = FINGERPRINT_REGION
    mask = (common_grid >= trim_region[0]) & (common_grid <= trim_region[1])
    proc_grid = common_grid[mask]

    processed = {}
    for key, (x, y) in raw_spectra.items():
        try:
            x_t, y_t = trim_spectrum(x, y.copy(), region=trim_region)
            y_t = smooth(y_t, window_length=11, polyorder=3)
            y_t = baseline_fn(y_t)
            y_t = normalize_spectrum(y_t, method="snv")
            y_grid = resample(x_t, y_t, proc_grid)
            processed[key] = y_grid
        except Exception:
            continue

    save_processed_spectra(processed, proc_grid, output_path=output_path)
    return output_path


# =============================================================================
# Run Phase Q evaluation on a given processed_spectra.csv
# =============================================================================
def run_phase_q_evaluation(csv_path, method_name, n_repeats=5, seed=42):
    """Replicate Phase Q train/val/test evaluation pipeline."""
    logger.info(f"\n{'='*64}")
    logger.info(f"  Phase Q Evaluation: {method_name}")
    logger.info(f"{'='*64}")

    with open(PROJECT_ROOT / "config" / "config.yaml", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    df = load_processed_spectra(csv_path)
    feat_cols = get_feature_columns(df)
    mc = ModelConfig.from_pipeline_config(raw_cfg, n_spectral_features=len(feat_cols))
    mc = apply_class_selection(mc, cancer_types=CANCER_TYPES, non_cancer_groups=NON_CANCER)
    df = resolve_aliases(df, mc)
    df_agg = aggregate_replicates(df, feat_cols, "none")

    # Clinical data
    clin = load_clinical()
    df_merged = merge_clinical(df_agg, clin)

    X, bl, ctl, sample_ids, groups_arr = create_labels(df_merged, mc)
    df_used = df_merged[df_merged["group"].isin(list(CANCER_TYPES) + list(NON_CANCER))].reset_index(drop=True)
    groups = df_used["group"].values

    # Sex array for biological constraint
    tier1 = df_used[["age", "sex_numeric", "bmi"]].values.astype(float)
    sex_from_clinical = tier1[:, 1]
    sex_for_constraint = sex_from_clinical.copy()
    for i, grp in enumerate(groups):
        if np.isnan(sex_for_constraint[i]):
            if grp == "PRO":
                sex_for_constraint[i] = 1.0
            elif grp == "OVA":
                sex_for_constraint[i] = 0.0

    logger.info(f"  Total: {len(X)} spectra, {len(set(sample_ids))} subjects")
    logger.info(f"  Cancer: {(bl==1).sum()}, Non-cancer: {(bl==0).sum()}")

    all_results = []
    for repeat_i in range(n_repeats):
        s = seed + repeat_i * 100
        train_idx, val_idx, test_idx = create_train_val_test_split(
            X, bl, ctl, sample_ids, groups, seed=s)

        results, thresh, _, _ = train_and_evaluate(
            X, bl, ctl, groups, train_idx, val_idx, test_idx, mc,
            sex_numeric=sex_for_constraint, label=method_name)

        t = results["test"]
        logger.info(f"  Split {repeat_i+1}: Det AUC={t['det_auc']:.4f} "
                    f"Sens={t['det_sensitivity']:.4f} Spec={t['det_specificity']:.4f} "
                    f"Id F1={t.get('id_f1_macro', 0):.4f}")
        all_results.append(results)

    # Aggregate
    test_metrics = {}
    for key in ["det_auc", "det_sensitivity", "det_specificity", "id_f1_macro"]:
        vals = [r["test"].get(key, np.nan) for r in all_results]
        vals = [v for v in vals if not np.isnan(v)]
        if vals:
            test_metrics[key] = {"mean": np.mean(vals), "std": np.std(vals)}

    logger.info(f"\n  >> {method_name} TEST Summary ({n_repeats} splits):")
    for key in ["det_auc", "det_sensitivity", "det_specificity", "id_f1_macro"]:
        if key in test_metrics:
            m = test_metrics[key]
            logger.info(f"     {key:>20s}: {m['mean']:.4f} ± {m['std']:.4f}")

    # Per-cancer F1 from last split
    last_test = all_results[-1]["test"]
    if "per_cancer_id" in last_test:
        logger.info(f"\n  Per-cancer F1 (last split):")
        for ct_name in CANCER_TYPES:
            if ct_name in last_test["per_cancer_id"]:
                info = last_test["per_cancer_id"][ct_name]
                logger.info(f"    {ct_name}: F1={info['f1-score']:.3f} "
                           f"(prec={info['precision']:.3f}, recall={info['recall']:.3f}, n={info['support']})")

    return test_metrics, all_results


# =============================================================================
# Main
# =============================================================================
def main():
    logger.info("=" * 64)
    logger.info("  Baseline Correction Comparison — Phase Q Pipeline")
    logger.info("  (Rolling Min vs ALS, with sex constraint)")
    logger.info("=" * 64)

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
        except Exception:
            continue
    logger.info(f"Loaded {len(raw_spectra)} spectra")

    x_arrays = [x for x, y in raw_spectra.values()]
    common_grid = make_common_grid(x_arrays)

    results_dir = PROJECT_ROOT / "results" / "baseline_comparison"
    results_dir.mkdir(parents=True, exist_ok=True)

    methods = [
        ("Rolling_Minimum",
         lambda y: baseline_correction(y, window=101),
         {}),
        ("ALS_lam1e6_p0.01",
         lambda y: baseline_als(y, lam=1e6, p=0.01, niter=10),
         {"lam": 1e6, "p": 0.01, "niter": 10}),
        ("ALS_lam1e7_p0.001",
         lambda y: baseline_als(y, lam=1e7, p=0.001, niter=15),
         {"lam": 1e7, "p": 0.001, "niter": 15}),
    ]

    summary_rows = []
    all_method_results = {}

    for method_name, baseline_fn, params in methods:
        logger.info(f"\n\n{'#'*64}")
        logger.info(f"  Preprocessing with: {method_name}")
        logger.info(f"{'#'*64}")

        csv_path = results_dir / f"processed_spectra_{method_name}.csv"

        t0 = time.time()
        build_processed_csv(raw_spectra, common_grid, baseline_fn, config, csv_path)
        preprocess_time = time.time() - t0
        logger.info(f"  Preprocessing time: {preprocess_time:.1f}s")

        t0 = time.time()
        test_metrics, split_results = run_phase_q_evaluation(
            csv_path, method_name, n_repeats=5, seed=42)
        eval_time = time.time() - t0

        row = {
            "method": method_name,
            "preprocess_time_s": round(preprocess_time, 1),
            "eval_time_s": round(eval_time, 1),
        }
        for key in ["det_auc", "det_sensitivity", "det_specificity", "id_f1_macro"]:
            if key in test_metrics:
                row[f"{key}_mean"] = round(test_metrics[key]["mean"], 4)
                row[f"{key}_std"] = round(test_metrics[key]["std"], 4)
        row["params"] = str(params)

        summary_rows.append(row)
        all_method_results[method_name] = split_results

    # ================================================================
    # Final Comparison Table
    # ================================================================
    logger.info(f"\n\n{'='*80}")
    logger.info("  FINAL COMPARISON — Phase Q Pipeline (TEST set, 5 splits)")
    logger.info(f"{'='*80}")

    header = f"{'Method':<25s} {'Det AUC':>16s} {'Sensitivity':>16s} {'Specificity':>16s} {'Id F1-macro':>16s} {'Time':>6s}"
    logger.info(f"\n  {header}")
    logger.info(f"  {'-'*len(header)}")

    for row in summary_rows:
        line = (f"  {row['method']:<25s} "
                f"{row.get('det_auc_mean', 0):.4f}±{row.get('det_auc_std', 0):.4f}  "
                f"{row.get('det_sensitivity_mean', 0):.4f}±{row.get('det_sensitivity_std', 0):.4f}  "
                f"{row.get('det_specificity_mean', 0):.4f}±{row.get('det_specificity_std', 0):.4f}  "
                f"{row.get('id_f1_macro_mean', 0):.4f}±{row.get('id_f1_macro_std', 0):.4f}  "
                f"{row['preprocess_time_s']:>5.1f}s")
        logger.info(line)

    # Determine best
    best_auc = max(summary_rows, key=lambda r: r.get("det_auc_mean", 0))
    best_f1 = max(summary_rows, key=lambda r: r.get("id_f1_macro_mean", 0))

    logger.info(f"\n  Best Det AUC:    {best_auc['method']} ({best_auc.get('det_auc_mean', 0):.4f})")
    logger.info(f"  Best Id F1:      {best_f1['method']} ({best_f1.get('id_f1_macro_mean', 0):.4f})")

    # Save summary
    df_summary = pd.DataFrame(summary_rows)
    summary_path = results_dir / "baseline_comparison_phaseQ.csv"
    df_summary.to_csv(summary_path, index=False)
    logger.info(f"\n  Results saved to: {summary_path}")

    # Save detailed JSON
    detail = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "baseline_correction_comparison",
        "pipeline": "Phase Q (train/val/test, sex constraint)",
        "n_repeats": 5,
        "methods": summary_rows,
    }
    with open(results_dir / "baseline_comparison_phaseQ.json", "w") as f:
        json.dump(detail, f, indent=2, default=str)

    logger.info("\n  Done.")


if __name__ == "__main__":
    main()
