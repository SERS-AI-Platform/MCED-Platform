"""
QC Threshold Sweep Experiment

Sweeps (RSD threshold, Corr threshold) combinations and measures
classification performance (Binary AUC, Type ID F1) via 5-fold CV
with Logistic Regression.

Requires:
  - results/qc_experiment/processed_spectra.csv (all samples, --skip-qc)
  - results/qc_stats.csv (per-sample QC metrics)

Usage:
    cd /home/user/SERS-AI
    PYTHONPATH=. python scripts/run_qc_threshold_experiment.py
"""

import sys
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import label_binarize

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.model import ModelConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Config ──
ALL_SPECTRA = PROJECT_ROOT / "results" / "qc_experiment" / "processed_spectra.csv"
QC_STATS = PROJECT_ROOT / "results" / "qc_stats.csv"
OUT_DIR = PROJECT_ROOT / "results" / "qc_experiment"

RSD_THRESHOLDS = [3, 5, 7, 10, 15, 20, float("inf")]
CORR_THRESHOLDS = [0.95, 0.90, 0.85, 0.80, 0.0]

N_SPLITS = 5
RANDOM_STATE = 42


def load_data():
    df = pd.read_csv(ALL_SPECTRA)
    qc = pd.read_csv(QC_STATS)
    logger.info(f"Loaded {len(df)} spectra, {len(qc)} QC stats")
    return df, qc


def get_feature_cols(df):
    return [c for c in df.columns if c.startswith("x_")]


def aggregate_medoid(df, feature_cols):
    rows = []
    for (group, sid), sub in df.groupby(["group", "sample_id"]):
        if len(sub) == 1:
            rows.append(sub.iloc[0])
            continue
        spectra = sub[feature_cols].values
        corr_mat = np.corrcoef(spectra)
        medoid_idx = corr_mat.mean(axis=1).argmax()
        rows.append(sub.iloc[medoid_idx])
    return pd.DataFrame(rows).reset_index(drop=True)


def filter_by_qc(df, qc, rsd_thresh, corr_thresh):
    if rsd_thresh == float("inf") and corr_thresh == 0.0:
        return df.copy()

    passed = qc.copy()
    if rsd_thresh < float("inf"):
        passed = passed[passed["mean_rsd"] <= rsd_thresh]
    if corr_thresh > 0.0:
        passed = passed[passed["mean_corr"] >= corr_thresh]

    keys = set(zip(passed["group"], passed["sample_id"].astype(str)))
    mask = df.apply(lambda r: (r["group"], str(r["sample_id"])) in keys, axis=1)
    return df[mask].copy()


def run_cv(df_agg, config, feature_cols):
    # Resolve aliases (e.g., CPAN/YPAN → PAN)
    df_agg = df_agg.copy()
    df_agg["group"] = df_agg["group"].apply(config.resolve_group)

    valid_groups = set(config.cancer_types) | set(config.non_cancer_groups) | set(config.cancer_groups_raw)
    df_v = df_agg[df_agg["group"].isin(valid_groups)].copy()

    if len(df_v) < 20:
        return None

    X = df_v[feature_cols].values
    groups = df_v["group"].values
    sample_ids = df_v["sample_id"].values

    binary = np.array([1 if g in config.cancer_types else 0 for g in groups])
    cancer_idx = np.array([config.cancer_type_index(g) if g in config.cancer_types else -1 for g in groups])

    if len(np.unique(binary)) < 2:
        return None

    n_cancer_types = len(config.cancer_types)
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    fold_auc_s1 = []
    fold_f1_s2 = []

    for train_idx, val_idx in cv.split(X, binary, sample_ids):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_bin_tr, y_bin_val = binary[train_idx], binary[val_idx]
        ct_tr, ct_val = cancer_idx[train_idx], cancer_idx[val_idx]

        # Stage 1: Binary
        lr1 = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", random_state=RANDOM_STATE)
        lr1.fit(X_tr, y_bin_tr)
        prob1 = lr1.predict_proba(X_val)[:, 1]

        if len(np.unique(y_bin_val)) >= 2:
            auc1 = roc_auc_score(y_bin_val, prob1)
        else:
            auc1 = float("nan")
        fold_auc_s1.append(auc1)

        # Stage 2: Cancer type (cancer samples only)
        cancer_tr = ct_tr[y_bin_tr == 1]
        cancer_val = ct_val[y_bin_val == 1]
        X_tr_c = X_tr[y_bin_tr == 1]
        X_val_c = X_val[y_bin_val == 1]

        present_tr = np.unique(cancer_tr)
        present_val = np.unique(cancer_val)

        if len(present_tr) >= 2 and len(present_val) >= 2 and len(X_val_c) >= 5:
            lr2 = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs",
                                     multi_class="multinomial", random_state=RANDOM_STATE)
            lr2.fit(X_tr_c, cancer_tr)
            pred2 = lr2.predict(X_val_c)
            f1 = f1_score(cancer_val, pred2, average="macro", zero_division=0)
        else:
            f1 = float("nan")
        fold_f1_s2.append(f1)

    return {
        "auc_s1_folds": fold_auc_s1,
        "f1_s2_folds": fold_f1_s2,
    }


def main():
    logger.info("=" * 60)
    logger.info("  QC Threshold Sweep Experiment")
    logger.info("=" * 60)

    config = ModelConfig(
        group_aliases={"PAN": ["CPAN", "YPAN"]},
    )
    df, qc = load_data()
    feature_cols = get_feature_cols(df)

    results = []

    total = len(RSD_THRESHOLDS) * len(CORR_THRESHOLDS)
    i = 0

    for rsd_t in RSD_THRESHOLDS:
        for corr_t in CORR_THRESHOLDS:
            i += 1
            rsd_label = "inf" if rsd_t == float("inf") else str(rsd_t)
            logger.info(f"\n[{i}/{total}] RSD<={rsd_label}%, Corr>={corr_t}")

            df_filtered = filter_by_qc(df, qc, rsd_t, corr_t)
            n_spectra = len(df_filtered)
            n_samples = df_filtered.groupby(["group", "sample_id"]).ngroups

            if n_samples < 20:
                logger.warning(f"  Only {n_samples} samples — skipping")
                results.append({
                    "rsd_threshold": rsd_t, "corr_threshold": corr_t,
                    "n_samples": n_samples, "n_spectra": n_spectra,
                    "auc_s1_mean": None, "auc_s1_std": None,
                    "f1_s2_mean": None, "f1_s2_std": None,
                    "skipped": True,
                })
                continue

            # Per-group counts
            group_counts = df_filtered.groupby("group")["sample_id"].nunique().to_dict()

            # Aggregate
            df_agg = aggregate_medoid(df_filtered, feature_cols)

            # CV
            cv_result = run_cv(df_agg, config, feature_cols)

            if cv_result is None:
                logger.warning(f"  CV failed (not enough data)")
                results.append({
                    "rsd_threshold": rsd_t, "corr_threshold": corr_t,
                    "n_samples": n_samples, "n_spectra": n_spectra,
                    "auc_s1_mean": None, "auc_s1_std": None,
                    "f1_s2_mean": None, "f1_s2_std": None,
                    "skipped": True,
                })
                continue

            auc_arr = np.array([x for x in cv_result["auc_s1_folds"] if not np.isnan(x)])
            f1_arr = np.array([x for x in cv_result["f1_s2_folds"] if not np.isnan(x)])

            row = {
                "rsd_threshold": rsd_t,
                "corr_threshold": corr_t,
                "n_samples": n_samples,
                "n_spectra": n_spectra,
                "auc_s1_mean": round(float(auc_arr.mean()), 4) if len(auc_arr) else None,
                "auc_s1_std": round(float(auc_arr.std()), 4) if len(auc_arr) else None,
                "f1_s2_mean": round(float(f1_arr.mean()), 4) if len(f1_arr) else None,
                "f1_s2_std": round(float(f1_arr.std()), 4) if len(f1_arr) else None,
                "auc_folds": cv_result["auc_s1_folds"],
                "f1_folds": cv_result["f1_s2_folds"],
                "group_counts": group_counts,
                "skipped": False,
            }
            results.append(row)

            logger.info(f"  Samples: {n_samples}, Groups: {group_counts}")
            logger.info(f"  AUC(S1): {row['auc_s1_mean']:.4f} ± {row['auc_s1_std']:.4f}")
            logger.info(f"  F1(S2):  {row['f1_s2_mean']:.4f} ± {row['f1_s2_std']:.4f}")

    # Save results
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # CSV (without list columns)
    df_results = pd.DataFrame([{k: v for k, v in r.items() if k not in ("auc_folds", "f1_folds", "group_counts")}
                                for r in results])
    csv_path = OUT_DIR / "sweep_results.csv"
    df_results.to_csv(csv_path, index=False)
    logger.info(f"\nSaved CSV: {csv_path}")

    # JSON (full detail)
    json_path = OUT_DIR / "sweep_results.json"
    # Convert inf to string for JSON
    for r in results:
        if r["rsd_threshold"] == float("inf"):
            r["rsd_threshold"] = "inf"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved JSON: {json_path}")

    # Print summary table
    logger.info("\n" + "=" * 80)
    logger.info("  SUMMARY")
    logger.info("=" * 80)
    logger.info(f"{'RSD':>6} {'Corr':>6} {'N':>6} {'AUC(S1)':>12} {'F1(S2)':>12}")
    logger.info("-" * 50)
    for r in results:
        if r["skipped"]:
            continue
        rsd_label = "inf" if r["rsd_threshold"] in (float("inf"), "inf") else f"{r['rsd_threshold']}"
        logger.info(f"{rsd_label:>6} {r['corr_threshold']:>6.2f} {r['n_samples']:>6} "
                    f"{r['auc_s1_mean']:>6.4f}±{r['auc_s1_std']:.4f} "
                    f"{r['f1_s2_mean']:>6.4f}±{r['f1_s2_std']:.4f}")

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
