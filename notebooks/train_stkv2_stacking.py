"""
train_stkv2_stacking.py
=======================
STK-V2 stacking 재현 (Cancer Screening, subject-level nested CV)

구조 (원본 STK-V2 스펙)
-----------------------
  10 base model:
    lr_raw      : ch0 935  → StandardScaler + LogisticRegression
    lr_d1       : d1  935  → StandardScaler + LogisticRegression
    lr_d2       : d2  935  → StandardScaler + LogisticRegression
    lr_concat   : raw+d1+d2 (2805) → StandardScaler + LogisticRegression
    lr_peak     : 75 peak  → StandardScaler + LogisticRegression
    xgb_raw     : ch0 935  → XGBoost
    xgb_d1      : d1  935  → XGBoost
    rf_raw      : ch0 935  → RandomForest
    rf_d1       : d1  935  → RandomForest
    ridge_concat: raw+d1+d2 → strong-L2 LogisticRegression
  meta learner:
    ElasticNet logistic regression (10 base cancer-prob → 최종)

평가
----
  - leakage-free: subject-level GroupKFold (outer 5)
  - meta 학습: outer train 안에서 inner GroupKFold OOF prediction
  - 지표: ROC-AUC, balanced accuracy, confusion matrix
  - canonical comparison: threshold selected inside outer-train folds
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

DECISION_THRESHOLD = 0.4   # STK-V2 artifact cancer threshold


def _select_optimal_threshold(y_true, probability):
    labels = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probability, dtype=float)
    unique_probabilities = np.unique(probabilities)
    midpoints = (
        (unique_probabilities[:-1] + unique_probabilities[1:]) / 2.0
        if len(unique_probabilities) > 1
        else np.asarray([], dtype=float)
    )
    candidates = np.unique(
        np.concatenate([np.asarray([0.0, 1.0]), midpoints])
    )
    scores = np.asarray([
        balanced_accuracy_score(labels, probabilities >= threshold)
        for threshold in candidates
    ])
    best_score = float(np.max(scores))
    tied = candidates[np.isclose(scores, best_score, rtol=0.0, atol=1e-12)]
    selected = tied[int(np.argmin(np.abs(tied - 0.5)))]
    return float(selected), best_score


def _make_meta_model():
    return LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.5,
        C=1.0,
        max_iter=5000,
        random_state=0,
    )


# ---------------------------------------------------------------------------
# base model 정의
# ---------------------------------------------------------------------------
def make_base_models():
    def lr():
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, C=1.0))
    def ridge():
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, C=0.1))  # strong L2
    def xgb():
        return XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            n_jobs=-1, random_state=0, verbosity=0,
        )
    def rf():
        return RandomForestClassifier(
            n_estimators=400, max_depth=None, n_jobs=-1, random_state=0)

    # (이름, view키, factory)
    return [
        ("lr_raw",       "raw",    lr),
        ("lr_d1",        "d1",     lr),
        ("lr_d2",        "d2",     lr),
        ("lr_concat",    "concat", lr),
        ("lr_peak",      "peak",   lr),
        ("xgb_raw",      "raw",    xgb),
        ("xgb_d1",       "d1",     xgb),
        ("rf_raw",       "raw",    rf),
        ("rf_d1",        "d1",     rf),
        ("ridge_concat", "concat", ridge),
    ]


def build_views(d):
    """view 키 → feature matrix"""
    raw, d1, d2, peak = d["raw"], d["d1"], d["d2"], d["peak"]
    concat = np.concatenate([raw, d1, d2], axis=1)
    return {"raw": raw, "d1": d1, "d2": d2, "concat": concat, "peak": peak}


# ---------------------------------------------------------------------------
# nested subject-level CV
# ---------------------------------------------------------------------------
def run_nested_cv(
    views,
    y,
    groups,
    n_outer=5,
    n_inner=5,
    seed=0,
    tune_threshold=False,
    return_details=False,
):
    base_specs = make_base_models()
    n = len(y)
    outer = GroupKFold(n_splits=n_outer)

    oof_final = np.full(n, np.nan)   # 최종 meta OOF prediction
    oof_prediction = np.full(n, -1, dtype=int)
    oof_threshold = np.full(n, np.nan)
    fold_metrics = []

    for fold, (tr, te) in enumerate(outer.split(views["raw"], y, groups), start=1):
        y_tr = y[tr]
        g_tr = groups[tr]

        # --- inner OOF: base model → meta 학습용 feature ---
        inner = GroupKFold(n_splits=n_inner)
        meta_tr = np.zeros((len(tr), len(base_specs)))
        for j, (name, vkey, factory) in enumerate(base_specs):
            Xv = views[vkey]
            oof_col = np.zeros(len(tr))
            for itr, ite in inner.split(Xv[tr], y_tr, g_tr):
                mdl = factory()
                mdl.fit(Xv[tr][itr], y_tr[itr])
                oof_col[ite] = mdl.predict_proba(Xv[tr][ite])[:, 1]
            meta_tr[:, j] = oof_col

        if tune_threshold:
            threshold_probability = np.full(len(tr), np.nan)
            threshold_inner = GroupKFold(n_splits=n_inner)
            for threshold_train, threshold_validation in threshold_inner.split(
                meta_tr,
                y_tr,
                g_tr,
            ):
                threshold_model = _make_meta_model()
                threshold_model.fit(
                    meta_tr[threshold_train],
                    y_tr[threshold_train],
                )
                threshold_probability[threshold_validation] = (
                    threshold_model.predict_proba(
                        meta_tr[threshold_validation]
                    )[:, 1]
                )
            threshold, threshold_selection_bacc = _select_optimal_threshold(
                y_tr,
                threshold_probability,
            )
        else:
            threshold = DECISION_THRESHOLD
            threshold_selection_bacc = float("nan")

        # --- meta learner (ElasticNet logistic) ---
        meta = _make_meta_model()
        meta.fit(meta_tr, y_tr)

        # --- base model 을 outer-train 전체로 재학습 후 test 예측 ---
        meta_te = np.zeros((len(te), len(base_specs)))
        for j, (name, vkey, factory) in enumerate(base_specs):
            Xv = views[vkey]
            mdl = factory()
            mdl.fit(Xv[tr], y_tr)
            meta_te[:, j] = mdl.predict_proba(Xv[te])[:, 1]

        proba_te = meta.predict_proba(meta_te)[:, 1]
        oof_final[te] = proba_te

        auc = roc_auc_score(y[te], proba_te)
        pred = (proba_te >= threshold).astype(int)
        oof_prediction[te] = pred
        oof_threshold[te] = threshold
        bacc = balanced_accuracy_score(y[te], pred)
        fold_metrics.append({
            "fold": int(fold),
            "auc": float(auc),
            "balanced_accuracy": float(bacc),
            "threshold": float(threshold),
            "threshold_selection_balanced_accuracy": float(
                threshold_selection_bacc
            ),
        })
        print(f"  [outer {fold}] AUC={auc:.4f}  balanced_acc={bacc:.4f}")

    if return_details:
        return {
            "oof_probability": oof_final,
            "oof_prediction": oof_prediction,
            "oof_threshold": oof_threshold,
            "fold_metrics": fold_metrics,
        }
    return oof_final, [
        (row["auc"], row["balanced_accuracy"])
        for row in fold_metrics
    ]


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./stkv2_dataset")
    args = ap.parse_args()
    d_dir = Path(args.data)

    data = {
        "raw":  np.load(d_dir / "X_ch0.npy"),
        "d1":   np.load(d_dir / "X_d1.npy"),
        "d2":   np.load(d_dir / "X_d2.npy"),
        "peak": np.load(d_dir / "X_peak.npy"),
    }
    y = np.load(d_dir / "y_screen.npy")
    groups = np.load(d_dir / "groups.npy")
    views = build_views(data)

    print(f"[data] n={len(y)}  cancer={int(y.sum())}  "
          f"raw{data['raw'].shape} peak{data['peak'].shape}")
    print("[nested subject-level CV] Cancer Screening (STK-V2 stacking)")

    oof, fold_metrics = run_nested_cv(views, y, groups)

    aucs = np.array([m[0] for m in fold_metrics])
    baccs = np.array([m[1] for m in fold_metrics])
    pred = (oof >= DECISION_THRESHOLD).astype(int)

    print("\n===== STK-V2 Cancer Screening 결과 =====")
    print(f"AUC           : {aucs.mean():.4f} ± {aucs.std():.4f}")
    print(f"Balanced Acc  : {baccs.mean():.4f} ± {baccs.std():.4f}")
    print(f"Overall AUC   : {roc_auc_score(y, oof):.4f}")
    print("Confusion matrix (row=true, col=pred, thr=0.6):")
    print(confusion_matrix(y, pred))


if __name__ == "__main__":
    main()
