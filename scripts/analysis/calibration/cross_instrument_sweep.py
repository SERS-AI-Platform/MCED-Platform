#!/usr/bin/env python3
"""
Cross-Instrument Calibration Transfer — full sweep & proper self-CV.

Loads cached sample-level dataframes from cross_instrument_calibration.py and
runs every comparable test we can think of:

  1. Proper self-CV reference (5-fold StratifiedGroupKFold) for both domains.
     This is the *honest* ceiling that transfer should approach.
  2. PDS hyperparameter sweep over (half_window × ridge).
  3. OSC component sweep (k = 1, 2, 3, 5).
  4. Method comparison (raw / affine / best-PDS / best-OSC / chain) under
     multi-seed paired-fit splits → mean ± std for stability.
  5. Per-group transfer breakdown for the best method, both directions.

Run AFTER cross_instrument_calibration.py has populated the parquet cache.

Output → results/cross_instrument/calibration/sweep/
  ├ self_cv_reference.json
  ├ pds_sweep.csv
  ├ osc_sweep.csv
  ├ method_comparison_multiseed.csv
  ├ per_group_best.csv
  └ sweep_report.json
"""
from __future__ import annotations
import json
import logging
import sys
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sers.calibration_transfer import (
    AffineTransfer, ChainTransfer, IdentityTransfer, OSCTransfer, PDSTransfer,
    paired_correlation, paired_rmse,
)

# Reuse constants from main script
from scripts.analysis.calibration.cross_instrument_calibration import (
    CANCER_TYPES, GRID, GROUP_ALIASES, NON_CANCER_GROUPS, OUT_DIR, apply_aliases,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

SWEEP_DIR = OUT_DIR / "sweep"
SWEEP_DIR.mkdir(parents=True, exist_ok=True)

FEAT_COLS = [f"x_{wn:.2f}" for wn in GRID]
SEEDS = [42, 7, 123, 2024, 9999]
PAIRED_TRAIN_FRACTION = 0.5


# =============================================================================
# Cached data load
# =============================================================================
def load_cached():
    df_t = pd.read_parquet(OUT_DIR / "thermo_sample_level.parquet")
    df_m = pd.read_parquet(OUT_DIR / "medical_sample_level.parquet")
    logger.info(f"Loaded cache — Thermo {len(df_t)}, Medical {len(df_m)}")
    return df_t, df_m


def build_paired(df_t: pd.DataFrame, df_m: pd.DataFrame):
    key_t = set(zip(df_t["group"], df_t["sample_id"]))
    key_m = set(zip(df_m["group"], df_m["sample_id"]))
    paired_keys = sorted(key_t & key_m)
    df_t_idx = df_t.set_index(["group", "sample_id"])
    df_m_idx = df_m.set_index(["group", "sample_id"])
    X_t = df_t_idx.loc[paired_keys, FEAT_COLS].values
    X_m = df_m_idx.loc[paired_keys, FEAT_COLS].values
    paired_df = pd.DataFrame(paired_keys, columns=["group", "sample_id"])
    return paired_df, X_t, X_m


# =============================================================================
# Two-stage LR evaluator (same as main)
# =============================================================================
def _two_stage_eval(df_train: pd.DataFrame, df_test: pd.DataFrame) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    valid = set(CANCER_TYPES) | set(NON_CANCER_GROUPS)
    df_train = df_train[df_train["group"].isin(valid)]
    df_test = df_test[df_test["group"].isin(valid)]

    Xtr, Xte = df_train[FEAT_COLS].values, df_test[FEAT_COLS].values
    ytr = df_train["group"].isin(CANCER_TYPES).astype(int).values
    yte = df_test["group"].isin(CANCER_TYPES).astype(int).values
    ct_map = {ct: i for i, ct in enumerate(CANCER_TYPES)}
    ctl_tr = np.array([ct_map.get(g, -1) for g in df_train["group"]])
    ctl_te = np.array([ct_map.get(g, -1) for g in df_test["group"]])

    s1 = make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, max_iter=2000, solver="lbfgs", class_weight="balanced"))
    s1.fit(Xtr, ytr)
    p1 = s1.predict_proba(Xte)[:, 1]
    pred1 = (p1 > 0.5).astype(int)
    try:
        auc = roc_auc_score(yte, p1)
    except ValueError:
        auc = float("nan")
    f1 = f1_score(yte, pred1, average="macro")
    acc = accuracy_score(yte, pred1)

    mtr, mte = ctl_tr >= 0, ctl_te >= 0
    s2_f1 = float("nan")
    if mtr.sum() and mte.sum() and len(np.unique(ctl_tr[mtr])) >= 2:
        s2 = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, max_iter=2000, solver="lbfgs", class_weight="balanced"))
        s2.fit(Xtr[mtr], ctl_tr[mtr])
        pred2 = s2.predict(Xte[mte])
        s2_f1 = f1_score(ctl_te[mte], pred2, average="macro", zero_division=0)

    return dict(s1_auc=float(auc), s1_f1=float(f1), s1_acc=float(acc),
                s2_f1=float(s2_f1), n_train=len(df_train), n_test=len(df_test),
                p1=p1, yte=yte, groups=df_test["group"].values)


# =============================================================================
# Self-CV reference (proper)
# =============================================================================
def self_cv(df: pd.DataFrame, label: str, n_splits: int = 5) -> dict:
    """Real 5-fold StratifiedGroupKFold within a single domain."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    valid = set(CANCER_TYPES) | set(NON_CANCER_GROUPS)
    df = df[df["group"].isin(valid)].reset_index(drop=True)
    X = df[FEAT_COLS].values
    y = df["group"].isin(CANCER_TYPES).astype(int).values
    ct_map = {ct: i for i, ct in enumerate(CANCER_TYPES)}
    ctl = np.array([ct_map.get(g, -1) for g in df["group"]])
    sample_ids = (df["group"] + "_" + df["sample_id"].astype(str)).values

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_aucs, fold_s2f1 = [], []
    for tr, va in sgkf.split(X, y, sample_ids):
        s1 = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, max_iter=2000, solver="lbfgs", class_weight="balanced"))
        s1.fit(X[tr], y[tr])
        p = s1.predict_proba(X[va])[:, 1]
        try:
            fold_aucs.append(roc_auc_score(y[va], p))
        except ValueError:
            fold_aucs.append(float("nan"))

        mtr, mva = ctl[tr] >= 0, ctl[va] >= 0
        if mtr.sum() and mva.sum() and len(np.unique(ctl[tr][mtr])) >= 2:
            s2 = make_pipeline(StandardScaler(), LogisticRegression(
                C=1.0, max_iter=2000, solver="lbfgs", class_weight="balanced"))
            s2.fit(X[tr][mtr], ctl[tr][mtr])
            pred2 = s2.predict(X[va][mva])
            fold_s2f1.append(f1_score(ctl[va][mva], pred2, average="macro",
                                       zero_division=0))
        else:
            fold_s2f1.append(float("nan"))

    res = {
        "label": label,
        "n_samples": int(len(df)),
        "mean_auc": float(np.nanmean(fold_aucs)),
        "std_auc": float(np.nanstd(fold_aucs)),
        "mean_s2_f1": float(np.nanmean(fold_s2f1)),
        "std_s2_f1": float(np.nanstd(fold_s2f1)),
        "fold_aucs": [float(a) for a in fold_aucs],
        "fold_s2_f1": [float(a) for a in fold_s2f1],
    }
    logger.info(f"  {label}: AUC={res['mean_auc']:.4f}±{res['std_auc']:.4f} "
                f"S2F1={res['mean_s2_f1']:.4f}±{res['std_s2_f1']:.4f}")
    return res


# =============================================================================
# Helper: transform full df
# =============================================================================
def transform_full(df: pd.DataFrame, tr) -> pd.DataFrame:
    if isinstance(tr, IdentityTransfer):
        return df
    out = df.copy()
    out[FEAT_COLS] = tr.transform(df[FEAT_COLS].values)
    return out


def split_pairs(n: int, seed: int):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_fit = int(round(n * PAIRED_TRAIN_FRACTION))
    return perm[:n_fit], perm[n_fit:]


# =============================================================================
# Sweeps
# =============================================================================
def pds_sweep(X_t_fit, X_t_held, X_m_fit, X_m_held, df_t_a, df_m_a) -> pd.DataFrame:
    rows = []
    windows = [3, 5, 7, 11, 15, 21, 31]
    ridges = [0.01, 0.1, 1.0, 10.0]
    logger.info(f"\n[PDS sweep] {len(windows)*len(ridges)} configs × 2 directions")
    for w, r in product(windows, ridges):
        # M→T
        tr_m2t = PDSTransfer(half_window=w, ridge=r).fit(X_m_fit, X_t_fit)
        held_rmse = paired_rmse(tr_m2t.transform(X_m_held), X_t_held)
        held_corr = paired_correlation(tr_m2t.transform(X_m_held), X_t_held)
        df_train = transform_full(df_m_a, tr_m2t)
        res = _two_stage_eval(df_train, df_t_a)
        rows.append(dict(direction="Medical→Thermo", half_window=w, ridge=r,
                         held_rmse=held_rmse, held_corr=held_corr,
                         s1_auc=res["s1_auc"], s1_f1=res["s1_f1"],
                         s2_f1=res["s2_f1"]))
        # T→M
        tr_t2m = PDSTransfer(half_window=w, ridge=r).fit(X_t_fit, X_m_fit)
        held_rmse = paired_rmse(tr_t2m.transform(X_t_held), X_m_held)
        held_corr = paired_correlation(tr_t2m.transform(X_t_held), X_m_held)
        df_train = transform_full(df_t_a, tr_t2m)
        res = _two_stage_eval(df_train, df_m_a)
        rows.append(dict(direction="Thermo→Medical", half_window=w, ridge=r,
                         held_rmse=held_rmse, held_corr=held_corr,
                         s1_auc=res["s1_auc"], s1_f1=res["s1_f1"],
                         s2_f1=res["s2_f1"]))
        logger.info(f"  w={w:2d} ridge={r:<5g} M→T AUC={rows[-2]['s1_auc']:.3f}  "
                    f"T→M AUC={rows[-1]['s1_auc']:.3f}")
    return pd.DataFrame(rows)


def osc_sweep(X_t_fit, X_t_held, X_m_fit, X_m_held, df_t_a, df_m_a) -> pd.DataFrame:
    rows = []
    ks = [1, 2, 3, 5, 8]
    logger.info(f"\n[OSC sweep] k = {ks}")
    for k in ks:
        for direction, Xs_fit, Xt_fit, df_src, df_tgt in [
            ("Medical→Thermo", X_m_fit, X_t_fit, df_m_a, df_t_a),
            ("Thermo→Medical", X_t_fit, X_m_fit, df_t_a, df_m_a),
        ]:
            tr = OSCTransfer(n_components=k).fit(Xs_fit, Xt_fit)
            df_train = transform_full(df_src, tr)
            res = _two_stage_eval(df_train, df_tgt)
            rows.append(dict(direction=direction, n_components=k,
                             s1_auc=res["s1_auc"], s1_f1=res["s1_f1"],
                             s2_f1=res["s2_f1"]))
            logger.info(f"  {direction} k={k}  AUC={res['s1_auc']:.3f} "
                        f"S2F1={res['s2_f1']:.3f}")
    return pd.DataFrame(rows)


def method_comparison_multiseed(X_t_paired, X_m_paired, df_t_a, df_m_a,
                                best_pds_cfg: dict) -> pd.DataFrame:
    """Multi-seed paired-fit splits to estimate stability of each method."""
    n_pair = X_t_paired.shape[0]
    rows = []
    logger.info(f"\n[Multi-seed] {len(SEEDS)} seeds × 5 methods × 2 directions")
    for seed in SEEDS:
        fit_idx, held_idx = split_pairs(n_pair, seed)
        X_t_fit, X_t_held = X_t_paired[fit_idx], X_t_paired[held_idx]
        X_m_fit, X_m_held = X_m_paired[fit_idx], X_m_paired[held_idx]

        def make_methods(Xs, Xt):
            return [
                ("raw",      IdentityTransfer()),
                ("affine",   AffineTransfer().fit(Xs, Xt)),
                ("pds",      PDSTransfer(half_window=best_pds_cfg["half_window"],
                                         ridge=best_pds_cfg["ridge"]).fit(Xs, Xt)),
                ("osc",      OSCTransfer(n_components=2).fit(Xs, Xt)),
                ("pds+osc",  ChainTransfer(steps=[
                    PDSTransfer(half_window=best_pds_cfg["half_window"],
                                ridge=best_pds_cfg["ridge"]),
                    OSCTransfer(n_components=2),
                ]).fit(Xs, Xt)),
            ]

        for direction, src_X, tgt_X, df_src, df_tgt in [
            ("Medical→Thermo", X_m_fit, X_t_fit, df_m_a, df_t_a),
            ("Thermo→Medical", X_t_fit, X_m_fit, df_t_a, df_m_a),
        ]:
            for name, tr in make_methods(src_X, tgt_X):
                df_train = transform_full(df_src, tr)
                res = _two_stage_eval(df_train, df_tgt)
                rows.append(dict(seed=seed, direction=direction, method=name,
                                 s1_auc=res["s1_auc"], s1_f1=res["s1_f1"],
                                 s2_f1=res["s2_f1"]))
    return pd.DataFrame(rows)


def per_group_breakdown(df_train: pd.DataFrame, df_test: pd.DataFrame,
                        label: str) -> pd.DataFrame:
    res = _two_stage_eval(df_train, df_test)
    rows = []
    for g in sorted(set(res["groups"])):
        m = res["groups"] == g
        rows.append(dict(label=label, group=g, n=int(m.sum()),
                         mean_prob=float(res["p1"][m].mean()),
                         std_prob=float(res["p1"][m].std()),
                         is_cancer=g in CANCER_TYPES))
    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================
def main():
    t0 = datetime.now()
    logger.info("=" * 70)
    logger.info("  Cross-Instrument Calibration Sweep")
    logger.info("=" * 70)

    df_t, df_m = load_cached()
    df_t_a = apply_aliases(df_t)
    df_m_a = apply_aliases(df_m)
    paired_df, X_t_paired, X_m_paired = build_paired(df_t, df_m)
    n_pair = X_t_paired.shape[0]
    logger.info(f"Paired matrices: {X_t_paired.shape}")

    # ---- Self-CV reference --------------------------------------------------
    logger.info("\n[1] Proper self-CV reference (5-fold StratifiedGroupKFold)")
    self_t = self_cv(df_t_a, "Thermo")
    self_m = self_cv(df_m_a, "Medical")
    with open(SWEEP_DIR / "self_cv_reference.json", "w") as f:
        json.dump({"thermo": self_t, "medical": self_m}, f, indent=2)

    # ---- Default split for sweeps -------------------------------------------
    fit_idx, held_idx = split_pairs(n_pair, seed=42)
    X_t_fit, X_t_held = X_t_paired[fit_idx], X_t_paired[held_idx]
    X_m_fit, X_m_held = X_m_paired[fit_idx], X_m_paired[held_idx]

    # ---- PDS sweep ----------------------------------------------------------
    pds_df = pds_sweep(X_t_fit, X_t_held, X_m_fit, X_m_held, df_t_a, df_m_a)
    pds_df.to_csv(SWEEP_DIR / "pds_sweep.csv", index=False)

    # Pick best PDS by mean AUC across both directions
    pds_pivot = (pds_df.groupby(["half_window", "ridge"])["s1_auc"].mean()
                 .reset_index().sort_values("s1_auc", ascending=False))
    best = pds_pivot.iloc[0]
    best_pds = {"half_window": int(best["half_window"]),
                "ridge": float(best["ridge"]),
                "mean_auc": float(best["s1_auc"])}
    logger.info(f"\n  best PDS: w={best_pds['half_window']} "
                f"ridge={best_pds['ridge']} mean AUC={best_pds['mean_auc']:.4f}")

    # ---- OSC sweep ----------------------------------------------------------
    osc_df = osc_sweep(X_t_fit, X_t_held, X_m_fit, X_m_held, df_t_a, df_m_a)
    osc_df.to_csv(SWEEP_DIR / "osc_sweep.csv", index=False)

    # ---- Multi-seed method comparison --------------------------------------
    multi_df = method_comparison_multiseed(X_t_paired, X_m_paired, df_t_a, df_m_a,
                                           best_pds)
    multi_df.to_csv(SWEEP_DIR / "method_comparison_multiseed.csv", index=False)

    multi_summary = (multi_df.groupby(["direction", "method"])
                     .agg(s1_auc_mean=("s1_auc", "mean"),
                          s1_auc_std=("s1_auc", "std"),
                          s1_f1_mean=("s1_f1", "mean"),
                          s2_f1_mean=("s2_f1", "mean"),
                          s2_f1_std=("s2_f1", "std"))
                     .reset_index())
    multi_summary.to_csv(SWEEP_DIR / "method_comparison_summary.csv", index=False)
    logger.info("\nMethod comparison (multi-seed mean ± std):\n"
                + multi_summary.to_string(index=False))

    # ---- Per-group breakdown for best method --------------------------------
    logger.info("\n[Per-group] best PDS, both directions")
    tr_m2t = PDSTransfer(half_window=best_pds["half_window"],
                         ridge=best_pds["ridge"]).fit(X_m_fit, X_t_fit)
    tr_t2m = PDSTransfer(half_window=best_pds["half_window"],
                         ridge=best_pds["ridge"]).fit(X_t_fit, X_m_fit)
    pg_m2t = per_group_breakdown(transform_full(df_m_a, tr_m2t), df_t_a,
                                 "Medical→Thermo (PDS-best)")
    pg_t2m = per_group_breakdown(transform_full(df_t_a, tr_t2m), df_m_a,
                                 "Thermo→Medical (PDS-best)")
    pg_all = pd.concat([pg_m2t, pg_t2m], ignore_index=True)
    pg_all.to_csv(SWEEP_DIR / "per_group_best.csv", index=False)
    logger.info("\n" + pg_all.to_string(index=False))

    # ---- Final report -------------------------------------------------------
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "duration_seconds": (datetime.now() - t0).total_seconds(),
        "n_paired": int(n_pair),
        "self_cv": {"thermo": self_t, "medical": self_m},
        "best_pds": best_pds,
        "best_pds_per_direction": {
            d: pds_df[pds_df["direction"] == d]
                .sort_values("s1_auc", ascending=False).iloc[0].to_dict()
            for d in pds_df["direction"].unique()
        },
        "best_osc_per_direction": {
            d: osc_df[osc_df["direction"] == d]
                .sort_values("s1_auc", ascending=False).iloc[0].to_dict()
            for d in osc_df["direction"].unique()
        },
        "method_comparison_summary": multi_summary.to_dict(orient="records"),
    }
    with open(SWEEP_DIR / "sweep_report.json", "w") as f:
        json.dump(report, f, indent=2, default=float)

    logger.info(f"\n[Done] {(datetime.now() - t0).total_seconds():.1f}s — "
                f"outputs in {SWEEP_DIR}")


if __name__ == "__main__":
    main()
