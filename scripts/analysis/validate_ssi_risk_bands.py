"""Validate the 5-tier SSI risk-stratification bands (src/sers/scoring.py RISK_LEVELS)
against nested OOF cross-validated predictions from the STK-V2 model.

Checks whether observed cancer rate increases monotonically across the
equal-interval SSI bins (0-2/2-4/4-6/6-8/8-10) anchored at the fixed
decision threshold (0.60, "balanced"/Youden's J).

Input: results/training/stacking_v2/oof_predictions.npz (10 base model OOF
probabilities, n=1628, from the STK-V2 nested-CV workflow — see Claude
memory reference_stk_v2_per_cancer_calc.md for provenance).

Usage: python scripts/analysis/validate_ssi_risk_bands.py
"""

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

MODELS = [
    "lr_raw",
    "lr_d1",
    "lr_d2",
    "lr_concat",
    "lr_peak",
    "xgb_raw",
    "xgb_d1",
    "rf_raw",
    "rf_d1",
    "ridge_concat",
]

THRESHOLD = 0.60  # locked "balanced" decision threshold (Youden's J)
CUTOFF = 4.0
MAX_SCORE = 10.0
BANDS = [
    (0, 2, "LOW"),
    (2, 4, "LOW_MODERATE"),
    (4, 6, "MODERATE"),
    (6, 8, "HIGH"),
    (8, 10.0001, "VERY_HIGH"),
]


def prob_to_ssi(p, theta=THRESHOLD, cutoff=CUTOFF, max_score=MAX_SCORE):
    p = np.clip(p, 0.0, 1.0)
    if p <= theta:
        return cutoff * p / theta
    return cutoff + (max_score - cutoff) * (p - theta) / (1.0 - theta)


def main():
    oof = np.load("results/training/stacking_v2/oof_predictions.npz")
    y = oof["binary_labels"]
    meta_s1 = np.column_stack([oof[f"{m}_s1"] for m in MODELS])

    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    final_prob = np.zeros(len(y))
    for tr, te in skf.split(meta_s1, y):
        clf = LogisticRegression(C=1.0, max_iter=2000)
        clf.fit(meta_s1[tr], y[tr])
        final_prob[te] = clf.predict_proba(meta_s1[te])[:, 1]

    ssi = np.array([prob_to_ssi(p) for p in final_prob])

    print(f"Nested 5-fold meta CV AUC: {roc_auc_score(y, final_prob):.4f}")
    print(f"{'Band':14s} {'SSI range':>10s} {'N':>6s} {'Cancer':>7s} {'Rate':>8s} {'95% CI':>18s}")
    for lo, hi, name in BANDS:
        mask = (ssi >= lo) & (ssi < hi)
        n = int(mask.sum())
        if n == 0:
            print(f"{name:14s} [{lo:.0f}-{hi:.0f})  {n:6d}       -        -                 -")
            continue
        k = int(y[mask].sum())
        rate = k / n
        ci_lo, ci_hi = stats.beta.ppf([0.025, 0.975], k + 0.5, n - k + 0.5)
        print(
            f"{name:14s} [{lo:.0f}-{hi:.0f})  {n:6d}  {k:6d}  {rate * 100:6.1f}%   "
            f"[{ci_lo * 100:5.1f}%, {ci_hi * 100:5.1f}%]"
        )


if __name__ == "__main__":
    main()
