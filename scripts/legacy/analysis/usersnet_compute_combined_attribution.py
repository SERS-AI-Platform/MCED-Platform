"""uSERS-Net combined attribution = α·LR_coef·(x-baseline) + (1-α)·IG_ResNet18.

Per (sample, cancer_class, wavenumber) attribution. LR contribution is reconstructed
from fold-saved LR coefficients (StandardScaler + LogisticRegression). ResNet18 IG is
loaded from ig_full.npz (already computed).

Output:
    combined_attribution.npz  — (n, 7, L) signed combined attribution
    combined_top_peaks.csv    — per-cancer top-15 peaks with attribution magnitude
"""
from __future__ import annotations
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("combined_attr")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = PROJECT_ROOT / "results" / "runs" / "2026-05-11_usersnet_alpha0.8_ovr_roc"
ALPHA = 0.8
CANCER_TYPES = ["PRO", "BRE", "OVA", "LUN", "CRC", "PAN", "BLC"]
TARGET_CANCERS = ["BLC", "BRE", "CRC", "LUN", "OVA", "PAN", "PRO"]


def main():
    # Inputs
    inputs = np.load(RUN_DIR / "resnet18_ckpts" / "fold_inputs.npz", allow_pickle=True)
    X = inputs["X"].astype(np.float32)
    groups_arr = inputs["groups_arr"]
    ctl = inputs["ctl"]
    wavenumbers = inputs["wavenumbers"]
    n, L = X.shape
    nor_mask = (groups_arr == "NOR")
    baseline = X[nor_mask].mean(axis=0) if nor_mask.sum() > 0 else X.mean(axis=0)

    # ResNet18 IG (already computed)
    ig = np.load(RUN_DIR / "ig_per_cancer" / "ig_full.npz", allow_pickle=True)
    ig_attr = ig["attributions"].astype(np.float32)  # (n, 7, L)
    covered = ig["covered"]

    # LR coefficients per fold
    lr = np.load(RUN_DIR / "extra_models" / "logistic_regression_oof.npz", allow_pickle=True)
    s2_coef = lr["stage2_coef"]            # (n_folds, 7, L) in standardized feature space
    s2_int = lr["stage2_intercept"]        # (n_folds, 7)
    s2_sm = lr["stage2_scaler_mean"]       # (n_folds, L)
    s2_ss = lr["stage2_scaler_scale"]      # (n_folds, L)

    # Need fold assignment per sample. Recompute the same fold split.
    from sklearn.model_selection import StratifiedGroupKFold
    bl = inputs["bl"].astype(int)
    sample_ids = inputs["sample_ids"]
    composite = bl * 100 + np.clip(ctl, 0, 99)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    fold_of_sample = np.full(n, -1, dtype=int)
    for fi, (_, va) in enumerate(sgkf.split(X, composite, groups=sample_ids)):
        fold_of_sample[va] = fi
    logger.info(f"Fold coverage: {(fold_of_sample>=0).sum()}/{n}")

    # LR contribution per sample per cancer class.
    # In original feature space, for a standardized LR (sklearn pipeline StandardScaler+LR):
    #   logit_c(x) = beta_c · (x - mu) / sigma + intercept_c
    # Per-feature contribution to logit:
    #   contrib_c[j] = beta_c[j] / sigma[j] * (x[j] - mu[j])
    # We use this as signed attribution in original-x space.
    lr_attr = np.zeros((n, len(CANCER_TYPES), L), dtype=np.float32)
    for fi in range(5):
        sel = (fold_of_sample == fi)
        if sel.sum() == 0:
            continue
        xs = X[sel]                              # (k, L)
        coef = s2_coef[fi]                       # (7, L)
        mu = s2_sm[fi]; sg = s2_ss[fi]           # (L,)
        eff = coef / sg                          # (7, L)
        # contribution: (xs - mu) * eff[c]   summed over feature gives logit-intercept
        cent = xs - mu[None, :]                  # (k, L)
        # broadcasting → (k, 7, L)
        contrib = cent[:, None, :] * eff[None, :, :]
        lr_attr[sel] = contrib.astype(np.float32)

    combined = ALPHA * lr_attr + (1.0 - ALPHA) * ig_attr
    out_path = RUN_DIR / "combined_attribution.npz"
    np.savez_compressed(out_path,
                        combined=combined, lr_attr=lr_attr, ig_attr=ig_attr,
                        X=X, groups_arr=groups_arr, ctl=ctl,
                        wavenumbers=wavenumbers, baseline=baseline,
                        cancer_types=np.array(CANCER_TYPES), alpha=ALPHA)
    logger.info(f"Saved {out_path}  shapes combined={combined.shape}")

    # Top peaks per cancer
    rows = []
    for cancer in TARGET_CANCERS:
        ci = CANCER_TYPES.index(cancer)
        sample_mask = (groups_arr == cancer) & covered
        if sample_mask.sum() == 0:
            continue
        attr = combined[sample_mask, ci]      # (k, L)
        mean_abs = np.mean(np.abs(attr), axis=0)
        mean_signed = np.mean(attr, axis=0)
        cancer_mean = X[sample_mask].mean(axis=0)
        peaks, _ = find_peaks(mean_abs, distance=8, prominence=mean_abs.std() * 0.5)
        top_idx = peaks[np.argsort(-mean_abs[peaks])][:15]
        for rank, pi in enumerate(top_idx, 1):
            rows.append({"cancer": cancer, "rank": rank,
                         "wavenumber": float(wavenumbers[pi]),
                         "mean_abs_attr": float(mean_abs[pi]),
                         "mean_signed_attr": float(mean_signed[pi]),
                         "cancer_intensity": float(cancer_mean[pi]),
                         "nor_intensity": float(baseline[pi]),
                         "delta_vs_nor": float(cancer_mean[pi] - baseline[pi])})
    df = pd.DataFrame(rows)
    df.to_csv(RUN_DIR / "combined_top_peaks_per_cancer.csv", index=False, encoding="utf-8-sig")
    logger.info(f"Saved {RUN_DIR/'combined_top_peaks_per_cancer.csv'} ({len(df)} rows)")


if __name__ == "__main__":
    main()
