#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "matplotlib",
#   "numpy",
#   "pandas",
#   "pyyaml",
#   "scikit-learn",
#   "scipy",
#   "xgboost",
# ]
# ///
# ─── How to run ───
# python scripts/analysis/yonsei_prospective_current_model.py
# python scripts/analysis/yonsei_prospective_current_model.py --bootstrap 5000
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.training.train_usersnet import (  # noqa: E402
    EXTENDED_BASE_MODELS,
    extract_peak_features,
    peak_feature_names,
)
from src.sers.models.usersnet.stacking import (  # noqa: E402
    build_classifier,
    configure_runtime,
    load_processed_multichannel,
)

INPUT_SPECTRA = PROJECT_ROOT / "results/kao_20260610_updated_cohort/processed_spectra.csv"
INPUT_MANIFEST = PROJECT_ROOT / "results/kao_20260610_updated_cohort/cohort_subject_manifest.csv"
REFERENCE_RUN = PROJECT_ROOT / "results/training/stacking_v2_kao_20260610_updated"
OUT_DIR = PROJECT_ROOT / "results/yonsei_prospective_current_model"
YONSEI_GROUPS = ("YPAN", "YNOR")


def aggregate_yonsei() -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    grid = np.linspace(402.0, 2198.0, 933)
    spectra, meta, grid = load_processed_multichannel(INPUT_SPECTRA, target_grid=grid)
    keep = meta["group"].isin(YONSEI_GROUPS).to_numpy()
    spectra = spectra[keep]
    meta = meta.loc[keep].reset_index(drop=True)
    meta["source_group"] = meta["group"].astype(str)
    meta["subject_id"] = meta["source_group"] + "_" + meta["sample_id"].astype(str)
    rows: list[dict[str, str]] = []
    aggregated = []
    for subject_id, sub in meta.groupby("subject_id", sort=True):
        idx = sub.index.to_numpy()
        source_group = str(sub["source_group"].iloc[0])
        sample_id = str(sub["sample_id"].iloc[0])
        rows.append(
            {
                "subject_id": subject_id,
                "source_group": source_group,
                "sample_id": sample_id,
                "model_group": "PAN" if source_group == "YPAN" else "NOR",
                "analysis_group": "Cancer" if source_group == "YPAN" else "Control",
                "n_replicates": str(len(idx)),
            }
        )
        aggregated.append(spectra[idx].mean(axis=0))
    return np.stack(aggregated).astype(np.float32), pd.DataFrame(rows), grid


def feature_matrix(
    spec: dict[str, object],
    spectra: np.ndarray,
    peak_features: np.ndarray,
) -> np.ndarray:
    channels = spec["channels"]
    if channels == "peak":
        return peak_features
    return spectra[:, channels, :].reshape(len(spectra), -1)


def oof_base_predictions(
    spectra: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    grid: np.ndarray,
    n_splits: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
    peak_features = extract_peak_features(spectra[:, 0, :], grid)
    np.savez_compressed(
        OUT_DIR / "peak_features.npz",
        X_peak=peak_features,
        feature_names=np.asarray(peak_feature_names(), dtype=object),
    )
    feature_matrices = {
        name: feature_matrix(spec, spectra, peak_features)
        for name, spec in EXTENDED_BASE_MODELS.items()
    }
    outer_folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    inner_folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)
    predictions: dict[str, np.ndarray] = {
        name: np.full(len(labels), np.nan, dtype=float) for name in EXTENDED_BASE_MODELS
    }
    meta_predictions = np.full(len(labels), np.nan, dtype=float)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(outer_folds.split(spectra, labels), start=1):
        meta_train_columns = []
        meta_test_columns = []
        for name, spec in EXTENDED_BASE_MODELS.items():
            x_all = feature_matrices[name]
            train_inner_prob = np.full(len(train_idx), np.nan, dtype=float)
            for inner_train, inner_test in inner_folds.split(x_all[train_idx], labels[train_idx]):
                model = build_classifier(str(spec["model"]), "binary")
                model.fit(x_all[train_idx][inner_train], labels[train_idx][inner_train])
                train_inner_prob[inner_test] = model.predict_proba(x_all[train_idx][inner_test])[
                    :, 1
                ]
            model = build_classifier(str(spec["model"]), "binary")
            model.fit(x_all[train_idx], labels[train_idx])
            test_prob = model.predict_proba(x_all[test_idx])[:, 1]
            predictions[name][test_idx] = test_prob
            meta_train_columns.append(train_inner_prob)
            meta_test_columns.append(test_prob)
        meta_train = np.column_stack(meta_train_columns)
        meta_test = np.column_stack(meta_test_columns)
        meta = make_pipeline(
            StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        )
        meta.fit(meta_train, labels[train_idx])
        meta_predictions[test_idx] = meta.predict_proba(meta_test)[:, 1]
        rows.append(
            {
                "fold": fold,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "test_subjects": ";".join(subject_ids[test_idx].astype(str)),
            }
        )
    return pd.DataFrame(rows), predictions, meta_predictions


def binary_metrics(
    labels: np.ndarray, prob: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    return {
        "auroc": roc_auc_score(labels, prob),
        "pr_auc": average_precision_score(labels, prob),
        "accuracy": accuracy_score(labels, pred),
        "sensitivity": recall_score(labels, pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "precision_ppv": precision_score(labels, pred, zero_division=0),
        "npv": tn / (tn + fn) if (tn + fn) else float("nan"),
        "f1": f1_score(labels, pred, zero_division=0),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "threshold": threshold,
    }


def bootstrap_ci(
    labels: np.ndarray,
    prob: np.ndarray,
    n_bootstrap: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(20260710)
    point = binary_metrics(labels, prob)
    draws: dict[str, list[float]] = {
        key: [] for key in point if key not in {"tp", "fp", "tn", "fn", "threshold"}
    }
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(labels), len(labels))
        if len(np.unique(labels[idx])) < 2:
            continue
        metrics = binary_metrics(labels[idx], prob[idx])
        for key in draws:
            draws[key].append(metrics[key])
    rows = []
    for key, values in draws.items():
        arr = np.asarray(values, dtype=float)
        rows.append(
            {
                "metric": key,
                "estimate": point[key],
                "ci_lower": float(np.nanpercentile(arr, 2.5)),
                "ci_upper": float(np.nanpercentile(arr, 97.5)),
            }
        )
    return pd.DataFrame(rows)


def write_outputs(
    cohort: pd.DataFrame,
    fold_table: pd.DataFrame,
    base_predictions: dict[str, np.ndarray],
    meta_prob: np.ndarray,
    labels: np.ndarray,
    bootstrap: int,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pred = cohort.copy()
    for name, values in base_predictions.items():
        pred[f"{name}_prob"] = values
    pred["meta_lr_prob"] = meta_prob
    pred["true_label"] = labels
    pred["pred_label"] = (meta_prob >= 0.5).astype(int)
    pred.to_csv(OUT_DIR / "yonsei_oof_predictions.csv", index=False, encoding="utf-8-sig")
    cohort.to_csv(OUT_DIR / "yonsei_cohort_manifest.csv", index=False, encoding="utf-8-sig")
    fold_table.to_csv(OUT_DIR / "cv_folds.csv", index=False, encoding="utf-8-sig")

    metric_ci = bootstrap_ci(labels, meta_prob, bootstrap)
    metric_ci.to_csv(OUT_DIR / "cancer_screening_metrics_ci.csv", index=False, encoding="utf-8-sig")
    base_rows = [
        {"model": name, **binary_metrics(labels, values)}
        for name, values in base_predictions.items()
    ]
    pd.DataFrame(base_rows).to_csv(
        OUT_DIR / "base_model_metrics.csv", index=False, encoding="utf-8-sig"
    )

    fpr, tpr, thresholds = roc_curve(labels, meta_prob)
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds}).to_csv(
        OUT_DIR / "roc_curve.csv", index=False, encoding="utf-8-sig"
    )
    plt.figure(figsize=(4.2, 3.8))
    plt.plot(fpr, tpr, label=f"AUROC {roc_auc_score(labels, meta_prob):.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="#888888", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Yonsei prospective Cancer Screening")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "roc_curve.png", dpi=300)
    plt.close()

    summary = {
        "input_spectra": str(INPUT_SPECTRA),
        "input_manifest": str(INPUT_MANIFEST),
        "reference_current_model_run": str(REFERENCE_RUN),
        "n_subjects": int(len(labels)),
        "n_cancer": int(labels.sum()),
        "n_control": int((labels == 0).sum()),
        "n_splits": int(fold_table["fold"].nunique()),
        "primary_model": "STK-V2 Cancer Screening binary, current base models + LR meta learner",
        "cancer_type_id": "not_applicable_single_cancer_type_PAN",
    }
    (OUT_DIR / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_runtime(n_jobs=-1, xgb_device="cpu")
    spectra, cohort, grid = aggregate_yonsei()
    cohort["label"] = (cohort["source_group"] == "YPAN").astype(int)
    labels = cohort["label"].to_numpy(dtype=int)
    subject_ids = cohort["subject_id"].to_numpy(dtype=object)
    fold_table, base_predictions, meta_prob = oof_base_predictions(
        spectra, labels, subject_ids, grid, args.n_splits
    )
    write_outputs(cohort, fold_table, base_predictions, meta_prob, labels, args.bootstrap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
