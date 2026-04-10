#!/usr/bin/env python3
"""
Phase STK-V2 — Full re-evaluation + peak interpretation.

Inputs (existing):
  results/weekend_experiments/stacking_optimization/oof_predictions.npz
  results/weekend_experiments/stacking_optimization/best_ensemble_config.json
  results/weekend_experiments/stacking_optimization/base_model_contribution.csv
  models/production_stacking/{meta_s1,meta_s2,base_*}.joblib
  models/production_stacking/common_grid.npy
  models/production_stacking/peak_config.json (KNOWN_PEAKS layout)

Outputs (overwritten):
  results/training/stacking_optimization_v2/experiment_summary.json
  results/training/stacking_optimization_v2/peak_interpretation.json
"""
from __future__ import annotations
import json, sys, math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, f1_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[2]
OOF_DIR = ROOT / "results" / "weekend_experiments" / "stacking_optimization"
PROD_DIR = ROOT / "models" / "production_stacking"
OUT_DIR = ROOT / "results" / "training" / "stacking_optimization_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CANCER_TYPES = ("PRO", "LUN", "CRC", "PAN", "OVA", "BRE", "BLC")
N_CLASSES = len(CANCER_TYPES)
BASE_NAMES = [
    "lr_raw", "lr_d1", "lr_d2", "lr_concat", "lr_peak",
    "xgb_raw", "xgb_d1", "rf_raw", "rf_d1", "ridge_concat",
]

# Mirror stacking_optimization.py KNOWN_PEAKS so lr_peak coefficients can be
# mapped to peak names. Order is critical (it determines feature index).
KNOWN_PEAKS = [
    (448.1, "ring_deform"), (538.7, "SS_stretch"), (617.7, "CS_stretch"),
    (683.3, "creatinine"), (723.8, "adenine"), (795.1, "hippuric"),
    (849.1, "tyrosine"), (895.4, "uric_acid"), (933.9, "creatinine2"),
    (999.5, "phe_urea"), (1147.9, "uric_CN"), (1230.8, "amide_III"),
    (1292.5, "CH2_twist"), (1352.3, "trp_fermi"), (1448.7, "CH2_deform"),
    (1597.1, "purine_CC"), (1651.1, "amide_I"),
]
PEAK_RATIOS = [
    ("phe_urea", "adenine"), ("phe_urea", "creatinine"),
    ("hippuric", "creatinine"), ("CS_stretch", "creatinine"),
    ("amide_I", "CH2_deform"), ("adenine", "purine_CC"),
    ("tyrosine", "phe_urea"),
]


# ----------------------------------------------------------------------------
# 1. Compute full val + train metrics from OOF predictions
# ----------------------------------------------------------------------------

def make_meta(meta_type: str, task: str):
    if meta_type == "lr":
        clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs",
                                 multi_class="multinomial" if task == "multiclass" else "auto")
    elif meta_type == "elasticnet":
        clf = LogisticRegression(C=0.5, penalty="elasticnet", l1_ratio=0.5,
                                 max_iter=2000, solver="saga",
                                 multi_class="multinomial" if task == "multiclass" else "auto")
    else:
        raise ValueError(meta_type)
    return make_pipeline(StandardScaler(), clf)


def evaluate_stacking(meta_type: str, oof: dict, y_bin: np.ndarray,
                      y_type: np.ndarray, sample_ids: np.ndarray,
                      n_splits: int = 5, seed: int = 42) -> dict:
    """5-fold StratifiedGroupKFold on stacked OOF features."""
    meta_s1 = np.column_stack([oof[f"{m}_s1"] for m in BASE_NAMES])  # (N, 10)
    meta_s2 = np.hstack([oof[f"{m}_s2"] for m in BASE_NAMES])         # (N, 70)

    val_s1_pred = np.full(len(y_bin), np.nan)
    val_s2_pred_proba = np.full((len(y_bin), N_CLASSES), np.nan)

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in sgkf.split(meta_s1, y_bin, sample_ids):
        # Stage 1
        m1 = make_meta(meta_type, "binary")
        m1.fit(meta_s1[tr], y_bin[tr])
        val_s1_pred[te] = m1.predict_proba(meta_s1[te])[:, 1]

        # Stage 2 — cancer-only
        ctr = y_bin[tr] == 1
        cte = y_bin[te] == 1
        if ctr.sum() > 5 and cte.sum() > 0:
            m2 = make_meta(meta_type, "multiclass")
            m2.fit(meta_s2[tr][ctr], y_type[tr][ctr])
            proba = m2.predict_proba(meta_s2[te][cte])
            te_idx = np.where(te)[0] if te.dtype == bool else te
            cte_idx = te_idx[cte]
            full = np.zeros((len(te_idx), N_CLASSES))
            for i, cls in enumerate(m2.classes_):
                full[cte, cls] = proba[:, i]
            val_s2_pred_proba[te_idx] = full

    # ---- Stage 1 metrics on full val OOF ----
    bp = val_s1_pred
    bt = y_bin
    pred_label = (bp >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(bt, pred_label).ravel()
    val_s1 = {
        "val_s1_auc": float(roc_auc_score(bt, bp)),
        "val_s1_pr_auc": float(average_precision_score(bt, bp)),
        "val_s1_accuracy": float(accuracy_score(bt, pred_label)),
        "val_s1_sensitivity": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "val_s1_specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "val_s1_f1": float(f1_score(bt, pred_label)),
    }

    # ---- Stage 2 metrics ----
    cancer_mask = (y_bin == 1)
    s2_pred_label = val_s2_pred_proba[cancer_mask].argmax(axis=1)
    s2_true = y_type[cancer_mask]
    val_s2 = {
        "val_s2_accuracy": float(accuracy_score(s2_true, s2_pred_label)),
        "val_s2_f1_macro": float(f1_score(s2_true, s2_pred_label, average="macro", zero_division=0)),
    }
    try:
        val_s2["val_s2_auc"] = float(
            roc_auc_score(s2_true, val_s2_pred_proba[cancer_mask],
                          multi_class="ovr", average="macro")
        )
    except ValueError:
        val_s2["val_s2_auc"] = float("nan")

    # Per-fold std (Stage 1 AUC and Stage 2 F1) — re-run lightweight per-fold
    s1_aucs, s2_f1s = [], []
    for tr, te in sgkf.split(meta_s1, y_bin, sample_ids):
        m1 = make_meta(meta_type, "binary")
        m1.fit(meta_s1[tr], y_bin[tr])
        p = m1.predict_proba(meta_s1[te])[:, 1]
        s1_aucs.append(roc_auc_score(y_bin[te], p))
        ctr = y_bin[tr] == 1; cte = y_bin[te] == 1
        if ctr.sum() > 5 and cte.sum() > 0:
            m2 = make_meta(meta_type, "multiclass")
            m2.fit(meta_s2[tr][ctr], y_type[tr][ctr])
            pp = m2.predict(meta_s2[te][cte])
            s2_f1s.append(f1_score(y_type[te][cte], pp, average="macro", zero_division=0))

    val_s1["val_s1_auc_std"] = float(np.std(s1_aucs))
    val_s2["val_s2_f1_macro_std"] = float(np.std(s2_f1s))

    # ---- Train (resub) metrics — fit on all OOF, predict on all OOF ----
    m1_full = make_meta(meta_type, "binary")
    m1_full.fit(meta_s1, y_bin)
    tr_s1 = m1_full.predict_proba(meta_s1)[:, 1]
    train_s1_auc = float(roc_auc_score(y_bin, tr_s1))

    m2_full = make_meta(meta_type, "multiclass")
    cm = y_bin == 1
    m2_full.fit(meta_s2[cm], y_type[cm])
    tr_s2 = m2_full.predict(meta_s2[cm])
    tr_s2_proba = m2_full.predict_proba(meta_s2[cm])
    train_s2_f1 = float(f1_score(y_type[cm], tr_s2, average="macro", zero_division=0))
    try:
        train_s2_auc = float(roc_auc_score(y_type[cm], tr_s2_proba,
                                           multi_class="ovr", average="macro"))
    except ValueError:
        train_s2_auc = float("nan")

    return {
        **val_s1, **val_s2,
        "train_s1_auc": train_s1_auc,
        "train_s2_f1_macro": train_s2_f1,
        "train_s2_auc": train_s2_auc,
    }


# ----------------------------------------------------------------------------
# 2. Peak interpretation
# ----------------------------------------------------------------------------

def get_estimator(model):
    if hasattr(model, "named_steps"):
        return list(model.named_steps.values())[-1]
    return model


def importance_vector(model_name: str) -> tuple[str, np.ndarray]:
    """Return (feature_kind, importance_vector) for a base model's S1 head."""
    m = joblib.load(PROD_DIR / f"base_{model_name}_s1.joblib")
    est = get_estimator(m)
    if hasattr(est, "coef_"):
        v = np.abs(est.coef_).ravel()
    elif hasattr(est, "feature_importances_"):
        v = np.asarray(est.feature_importances_).ravel()
    else:
        raise RuntimeError(f"no importance for {model_name}")

    if model_name in ("lr_raw", "xgb_raw", "rf_raw"):
        return "raw_935", v
    if model_name in ("lr_d1", "xgb_d1", "rf_d1"):
        return "d1_935", v
    if model_name == "lr_d2":
        return "d2_935", v
    if model_name in ("lr_concat", "ridge_concat"):
        return "concat_2805", v
    if model_name == "lr_peak":
        return "peak_75", v
    raise ValueError(model_name)


def top_wavenumbers(grid: np.ndarray, imp: np.ndarray, k: int = 10) -> list[dict]:
    idx = np.argsort(imp)[::-1][:k]
    return [{"wavenumber": float(round(grid[i], 1)),
             "importance": float(round(imp[i], 6))} for i in idx]


def top_concat_features(grid: np.ndarray, imp: np.ndarray, k: int = 10) -> list[dict]:
    """concat = [raw|d1|d2] each 935 features."""
    n = len(grid)
    idx = np.argsort(imp)[::-1][:k]
    out = []
    for i in idx:
        view = ["raw", "d1", "d2"][i // n]
        wn = float(round(grid[i % n], 1))
        out.append({"wavenumber": wn, "view": view,
                    "importance": float(round(imp[i], 6))})
    return out


def peak_feature_names() -> list[str]:
    """Match extract_peak_features layout: [areas|heights|fwhms|shifts|ratios]."""
    n = len(KNOWN_PEAKS)
    names = []
    for kind in ("area", "height", "fwhm", "shift"):
        names.extend([f"{p[1]}_{kind}" for p in KNOWN_PEAKS])
    for a, b in PEAK_RATIOS:
        names.append(f"{a}/{b}_ratio")
    return names  # length = 75


def top_peak_features(imp: np.ndarray, k: int = 10) -> list[dict]:
    names = peak_feature_names()
    idx = np.argsort(imp)[::-1][:k]
    return [{"feature": names[i], "importance": float(round(imp[i], 6))} for i in idx]


def stacking_aggregate_top_peaks(grid: np.ndarray, per_base_imp: dict,
                                 meta_weights: np.ndarray, contributions: dict,
                                 k: int = 15) -> list[dict]:
    """
    Aggregate stacking-level wavenumber importance:
        score(w) = sum_b |meta_w[b]| * contribution[b] * imp_b(w)
    where imp_b(w) is mapped onto the 935-wavenumber axis. Concat models
    contribute to (raw + d1 + d2) views; lr_peak excluded (different axis).
    """
    n = len(grid)
    score = np.zeros(n)
    for bi, name in enumerate(BASE_NAMES):
        if name == "lr_peak":
            continue
        kind, imp = per_base_imp[name]
        # normalize each base's importance vector to unit-sum so models are comparable
        if imp.sum() > 0:
            imp_n = imp / imp.sum()
        else:
            imp_n = imp
        weight = abs(float(meta_weights[bi])) * float(contributions.get(name, 0.0))
        if weight <= 0:
            continue
        if kind == "raw_935" or kind == "d1_935" or kind == "d2_935":
            score += weight * imp_n
        elif kind == "concat_2805":
            # average the three 935 chunks
            chunks = imp_n.reshape(3, n).mean(axis=0)
            score += weight * chunks

    idx = np.argsort(score)[::-1][:k]
    total = float(score.sum()) if score.sum() > 0 else 1.0
    return [{"wavenumber": float(round(grid[i], 1)),
             "score": float(round(score[i], 6)),
             "score_pct": float(round(100 * score[i] / total, 3))} for i in idx]


# ----------------------------------------------------------------------------
# 3. Main
# ----------------------------------------------------------------------------

def main():
    print(f"[STK-V2 full eval] {datetime.now().isoformat()}")
    oof_npz = np.load(OOF_DIR / "oof_predictions.npz", allow_pickle=True)
    oof = {k: oof_npz[k] for k in oof_npz.files}
    y_bin = oof["binary_labels"].astype(int)
    y_type = oof["cancer_type_labels"].astype(int)
    sample_ids = oof["sample_ids"]

    n = len(y_bin)
    n_cancer = int(y_bin.sum())
    print(f"  n={n}, cancer={n_cancer}, non-cancer={n-n_cancer}")

    # ---- Metrics ----
    print("  computing metrics: stacking_elasticnet_meta ...")
    en_metrics = evaluate_stacking("elasticnet", oof, y_bin, y_type, sample_ids)
    print("  computing metrics: stacking_lr_meta ...")
    lr_metrics = evaluate_stacking("lr", oof, y_bin, y_type, sample_ids)

    # ---- Peak interpretation ----
    print("  loading production stacking artifacts ...")
    grid = np.load(PROD_DIR / "common_grid.npy")
    if len(grid) != 935:
        # 933 vs 935 — recreate using model coef shape
        print(f"  WARN: grid has {len(grid)} points, expected 935; padding via linspace")
        grid = np.linspace(grid.min(), grid.max(), 935)

    per_base_imp = {name: importance_vector(name) for name in BASE_NAMES}

    per_base_top = {}
    for name in BASE_NAMES:
        kind, imp = per_base_imp[name]
        if kind in ("raw_935", "d1_935", "d2_935"):
            top = top_wavenumbers(grid, imp, k=10)
            per_base_top[name] = {"feature_space": kind, "top_features": top}
        elif kind == "concat_2805":
            top = top_concat_features(grid, imp, k=10)
            per_base_top[name] = {"feature_space": kind, "top_features": top}
        elif kind == "peak_75":
            top = top_peak_features(imp, k=10)
            per_base_top[name] = {"feature_space": kind, "top_features": top}

    # Meta weights (S1 LogisticRegression coef in production stacking)
    meta_s1_model = joblib.load(PROD_DIR / "meta_s1.joblib")
    meta_w = get_estimator(meta_s1_model).coef_.ravel()  # length = 10
    meta_w_dict = {BASE_NAMES[i]: float(round(meta_w[i], 6)) for i in range(len(BASE_NAMES))}

    # Contributions from existing CSV
    contrib_df = pd.read_csv(OOF_DIR / "base_model_contribution.csv")
    contributions = dict(zip(contrib_df["base_model"], contrib_df["mean_auc_drop"]))

    stacking_top = stacking_aggregate_top_peaks(grid, per_base_imp, meta_w, contributions, k=15)

    peak_interp = {
        "phase": "STK-V2",
        "generated_at": datetime.now().isoformat(),
        "method": {
            "per_base_model": "abs(coef_) for LR/Ridge S1 head; feature_importances_ for XGB/RF",
            "stacking_aggregate": "sum_b |meta_w[b]| * contribution[b] * normalized_imp_b(w); concat averaged across 3 views; lr_peak excluded (different feature space)",
            "wavenumber_grid_n": int(len(grid)),
        },
        "meta_weights_s1": meta_w_dict,
        "base_model_contributions": {k: float(round(v, 6)) for k, v in contributions.items()},
        "per_base_model": per_base_top,
        "stacking_aggregate_top_peaks": stacking_top,
    }

    with open(OUT_DIR / "peak_interpretation.json", "w") as f:
        json.dump(peak_interp, f, indent=2)
    print(f"  wrote {OUT_DIR / 'peak_interpretation.json'}")

    # ---- Headline numbers from original nested CV (best_ensemble_config.json) ----
    with open(OOF_DIR / "best_ensemble_config.json") as f:
        nested = json.load(f)
    nested_by_meta = {row["meta_learner"]: row for row in nested["meta_learner_summary"]}

    common = {
        "version": "v002",
        "n_samples": n,
        "n_features": 935,
        "n_cancer": n_cancer,
        "n_non_cancer": n - n_cancer,
    }

    def model_entry(name, metrics, meta_key):
        nrow = nested_by_meta[meta_key]
        e = {
            "model_name": name,
            **common,
            # Headline (nested CV — original Phase STK-V2 protocol)
            "val_s1_auc": float(round(nrow["mean_s1_auc"], 6)),
            "val_s1_auc_std": float(round(nrow["std_s1_auc"], 6)),
            "val_s2_f1_macro": float(round(nrow["mean_s2_f1"], 6)),
            "val_s2_f1_macro_std": float(round(nrow["std_s2_f1"], 6)),
            # Extended metrics (5-fold StratifiedGroupKFold on stacked OOF features)
            "val_s1_pr_auc": metrics["val_s1_pr_auc"],
            "val_s1_accuracy": metrics["val_s1_accuracy"],
            "val_s1_sensitivity": metrics["val_s1_sensitivity"],
            "val_s1_specificity": metrics["val_s1_specificity"],
            "val_s1_f1": metrics["val_s1_f1"],
            "val_s2_accuracy": metrics["val_s2_accuracy"],
            "val_s2_auc": metrics["val_s2_auc"],
            "train_s1_auc": metrics["train_s1_auc"],
            "train_s2_f1_macro": metrics["train_s2_f1_macro"],
            "train_s2_auc": metrics["train_s2_auc"],
            "metric_sources": {
                "val_s1_auc / val_s2_f1_macro (+std)": "nested CV (outer 5-fold × inner 5-fold OOF + inner 3-fold meta selection)",
                "val_s1_{pr_auc,accuracy,sensitivity,specificity,f1}, val_s2_{accuracy,auc}, train_*": "5-fold StratifiedGroupKFold refit on stacked OOF features (re-evaluation, no base retraining)",
            },
        }
        return e

    summary = {
        "phase": "STK-V2",
        "timestamp": datetime.now().isoformat(),
        "regenerated_from": "oof_predictions.npz (no base model retraining required)",
        "aggregate": "mean",
        "n_splits": 5,
        "cv_strategy": "StratifiedGroupKFold(n_splits=5, seed=42) on stacked OOF features",
        "cancer_types": list(CANCER_TYPES),
        "base_models": BASE_NAMES,
        "models": [
            model_entry("stacking_elasticnet_meta", en_metrics, "elasticnet"),
            model_entry("stacking_lr_meta", lr_metrics, "lr"),
        ],
        "peak_interpretation_file": "peak_interpretation.json",
    }
    with open(OUT_DIR / "experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  wrote {OUT_DIR / 'experiment_summary.json'}")
    print("done.")


if __name__ == "__main__":
    main()
