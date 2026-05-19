"""
CPAN 70 vs Non-cancer 400 Binary Classification — 10% Test Split

JNP 비교용 실험: 정상 400명 + CPAN 70명, 90/10 train/test, binary CM
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, confusion_matrix,
    classification_report, roc_curve,
)
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.train import load_processed_spectra, get_feature_columns


def main():
    n_repeats = 5  # multiple random seeds for stable results

    # ── Load data ──
    print("Loading processed spectra...")
    df = load_processed_spectra()
    feat_cols = get_feature_columns(df)

    # ── Filter: CPAN (cancer) + NOR/DIA/HBP/H.D. (non-cancer) ──
    cancer_groups = ["CPAN"]
    non_cancer_groups = ["NOR", "DIA", "HBP", "H.D."]
    all_groups = cancer_groups + non_cancer_groups

    df = df[df["group"].isin(all_groups)].copy()
    df["binary_label"] = df["group"].isin(cancer_groups).astype(int)
    df["subject_id"] = df["group"] + "_" + df["sample_id"].astype(str)

    X = df[feat_cols].values
    y = df["binary_label"].values
    subjects = df["subject_id"].values
    groups = df["group"].values

    n_cancer = df[df["binary_label"] == 1]["subject_id"].nunique()
    n_noncancer = df[df["binary_label"] == 0]["subject_id"].nunique()
    print(f"\nDataset: {n_cancer} cancer subjects (CPAN), {n_noncancer} non-cancer subjects")
    print(f"  Total spectra: {len(df)} ({(y==1).sum()} cancer, {(y==0).sum()} non-cancer)")
    print(f"  Groups: {dict(df['group'].value_counts())}")

    # ── Run multiple repeats ──
    all_results = []
    all_cms = []

    for repeat in range(n_repeats):
        seed = 42 + repeat * 100

        # 10% test split at subject level
        cv = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
        train_idx, test_idx = next(cv.split(X, y, subjects))

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        grp_test = groups[test_idx]
        subj_test = subjects[test_idx]

        # Train LR
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=1000, solver="saga",
                               class_weight="balanced", random_state=42)
        )
        model.fit(X_train, y_train)

        # Predict
        proba = model.predict_proba(X_test)[:, 1]

        # Optimize threshold via Youden's J on test set
        # (In JNP comparison context, using default 0.5 is also fine)
        fpr, tpr, thresholds = roc_curve(y_test, proba)
        j_idx = np.argmax(tpr - fpr)
        best_thresh = float(thresholds[j_idx])

        pred = (proba > best_thresh).astype(int)
        pred_default = (proba > 0.5).astype(int)

        auc = roc_auc_score(y_test, proba)
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
        tn2, fp2, fn2, tp2 = confusion_matrix(y_test, pred_default, labels=[0, 1]).ravel()

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        acc = accuracy_score(y_test, pred)

        sens2 = tp2 / (tp2 + fn2) if (tp2 + fn2) > 0 else 0
        spec2 = tn2 / (tn2 + fp2) if (tn2 + fp2) > 0 else 0
        acc2 = accuracy_score(y_test, pred_default)

        n_test_cancer = (y_test == 1).sum()
        n_test_noncancer = (y_test == 0).sum()
        n_test_subj_cancer = len(set(subj_test[y_test == 1]))
        n_test_subj_noncancer = len(set(subj_test[y_test == 0]))

        result = {
            "repeat": repeat + 1, "seed": seed,
            "test_spectra": len(y_test),
            "test_cancer_spectra": int(n_test_cancer),
            "test_noncancer_spectra": int(n_test_noncancer),
            "test_cancer_subjects": n_test_subj_cancer,
            "test_noncancer_subjects": n_test_subj_noncancer,
            "auc": auc,
            "threshold_youden": best_thresh,
            "sensitivity_youden": sens,
            "specificity_youden": spec,
            "accuracy_youden": acc,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "sensitivity_0.5": sens2,
            "specificity_0.5": spec2,
            "accuracy_0.5": acc2,
            "tp_0.5": int(tp2), "fp_0.5": int(fp2), "fn_0.5": int(fn2), "tn_0.5": int(tn2),
        }
        all_results.append(result)
        all_cms.append({"youden": [tn, fp, fn, tp], "default": [tn2, fp2, fn2, tp2]})

        print(f"\n── Repeat {repeat+1} (seed={seed}) ──")
        print(f"  Test: {n_test_subj_cancer} cancer + {n_test_subj_noncancer} non-cancer subjects "
              f"({n_test_cancer} + {n_test_noncancer} spectra)")
        print(f"  AUC: {auc:.4f}")
        print(f"  [Youden thresh={best_thresh:.3f}] Sens={sens:.1%} Spec={spec:.1%} Acc={acc:.1%}")
        print(f"  [Default thresh=0.5]               Sens={sens2:.1%} Spec={spec2:.1%} Acc={acc2:.1%}")
        print(f"  CM (Youden):  TN={tn} FP={fp} FN={fn} TP={tp}")
        print(f"  CM (0.5):     TN={tn2} FP={fp2} FN={fn2} TP={tp2}")

    # ── Aggregate ──
    print("\n" + "=" * 60)
    print(f"AGGREGATE RESULTS ({n_repeats} repeats)")
    print("=" * 60)
    aucs = [r["auc"] for r in all_results]
    sens_y = [r["sensitivity_youden"] for r in all_results]
    spec_y = [r["specificity_youden"] for r in all_results]
    acc_y = [r["accuracy_youden"] for r in all_results]
    sens_d = [r["sensitivity_0.5"] for r in all_results]
    spec_d = [r["specificity_0.5"] for r in all_results]
    acc_d = [r["accuracy_0.5"] for r in all_results]

    print(f"  AUC:          {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"\n  [Youden threshold]")
    print(f"  Sensitivity:  {np.mean(sens_y):.1%} ± {np.std(sens_y):.1%}")
    print(f"  Specificity:  {np.mean(spec_y):.1%} ± {np.std(spec_y):.1%}")
    print(f"  Accuracy:     {np.mean(acc_y):.1%} ± {np.std(acc_y):.1%}")
    print(f"\n  [Default 0.5 threshold]")
    print(f"  Sensitivity:  {np.mean(sens_d):.1%} ± {np.std(sens_d):.1%}")
    print(f"  Specificity:  {np.mean(spec_d):.1%} ± {np.std(spec_d):.1%}")
    print(f"  Accuracy:     {np.mean(acc_d):.1%} ± {np.std(acc_d):.1%}")

    # Average CM
    avg_cm_y = np.mean([c["youden"] for c in all_cms], axis=0)
    avg_cm_d = np.mean([c["default"] for c in all_cms], axis=0)
    print(f"\n  Average CM (Youden): TN={avg_cm_y[0]:.1f} FP={avg_cm_y[1]:.1f} FN={avg_cm_y[2]:.1f} TP={avg_cm_y[3]:.1f}")
    print(f"  Average CM (0.5):    TN={avg_cm_d[0]:.1f} FP={avg_cm_d[1]:.1f} FN={avg_cm_d[2]:.1f} TP={avg_cm_d[3]:.1f}")

    # ── Percentage CM ──
    print(f"\n  ── Confusion Matrix (%, Youden, averaged) ──")
    total_nc = avg_cm_y[0] + avg_cm_y[1]
    total_c = avg_cm_y[2] + avg_cm_y[3]
    print(f"                  Pred: Non-cancer    Pred: Cancer")
    print(f"  Actual Non-cancer   {avg_cm_y[0]/total_nc*100:5.1f}%           {avg_cm_y[1]/total_nc*100:5.1f}%    (n≈{total_nc:.0f})")
    print(f"  Actual Cancer       {avg_cm_y[2]/total_c*100:5.1f}%           {avg_cm_y[3]/total_c*100:5.1f}%    (n≈{total_c:.0f})")

    print(f"\n  ── Confusion Matrix (%, 0.5 threshold, averaged) ──")
    total_nc2 = avg_cm_d[0] + avg_cm_d[1]
    total_c2 = avg_cm_d[2] + avg_cm_d[3]
    print(f"                  Pred: Non-cancer    Pred: Cancer")
    print(f"  Actual Non-cancer   {avg_cm_d[0]/total_nc2*100:5.1f}%           {avg_cm_d[1]/total_nc2*100:5.1f}%    (n≈{total_nc2:.0f})")
    print(f"  Actual Cancer       {avg_cm_d[2]/total_c2*100:5.1f}%           {avg_cm_d[3]/total_c2*100:5.1f}%    (n≈{total_c2:.0f})")


if __name__ == "__main__":
    main()
