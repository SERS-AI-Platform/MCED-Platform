#!/usr/bin/env python3
"""
Cross-Instrument Calibration Transfer (Thermo ↔ Medical, bidirectional)
========================================================================

Pipeline
--------
1. Load Thermo + Medical raw spectra and preprocess both onto a SHARED grid
   (402–2198 cm⁻¹, 933 points). Medical gets the -28 cm⁻¹ shift before
   resampling so the urea peak aligns.
2. Aggregate to sample-level mean (one row per (group, sample_id)).
3. Identify paired samples — same (group, sample_id) present in both domains.
4. Step 0  Diagnostics: paired RMSE / correlation, overlay plots, PCA of the
            paired-difference matrix.
5. Steps 1–3  Fit four calibration transformers on a paired-train split:
              Affine, PDS, OSC, PDS+OSC. Identity acts as raw baseline.
6. Step 4  Evaluate every (transformer × direction) with the same LR two-stage
            classifier used by cross_instrument_analysis.py: train on the
            (mapped) source domain, test on the target domain.
7. Save results, plots, and a leaderboard CSV.

Outputs
-------
results/cross_instrument/calibration/
  ├ diagnosis/
  │   ├ paired_overlay_<group>.png
  │   ├ paired_difference_pca.png
  │   └ paired_stats.csv
  ├ paired_samples.csv
  ├ calibration_report.json
  └ leaderboard.csv
"""

from __future__ import annotations
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sers.io import find_spectra, parse_filename, read_spectrum
from src.sers.preprocessing import preprocess_single_spectrum
from src.sers.calibration_transfer import (
    AffineTransfer, ChainTransfer, IdentityTransfer, OSCTransfer, PDSTransfer,
    paired_correlation, paired_rmse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================
THERMO_DIR = PROJECT_ROOT / "data" / "raw_data"
MEDICAL_DIR = PROJECT_ROOT / "data" / "raw_data_medical"
OUT_DIR = PROJECT_ROOT / "results" / "cross_instrument" / "calibration"
DIAG_DIR = OUT_DIR / "diagnosis"

# Folder name → group code  (SPAN excluded — Thermo-only post-op)
THERMO_FOLDER_MAP = {
    "1. Prostate cancer (100개)": "PRO",
    "2. Breast cancer (30개)":    "BRE",
    "3. Ovarian cancer (70개)":   "OVA",
    "4. Lung cancer (300개)":     "LUN",
    "5. Normal (100개)":          "NOR",
    "6. Diabetes (100개)":        "DIA",
    "7. High blood pressure (100개)":              "HBP",
    "8. High blood pressure + Diabetes (100개)":   "H.D.",
    "9. Colorectal cancer (300개)":                "CRC",
    "10-1. C-Pancreatic cancer (70개)":            "CPAN",
    "11 BLC (299개)":                              "BLC",
}
MEDICAL_FOLDER_MAP = {
    "1. Prostate cancer (100개)": "PRO",
    "2. Breast cancer (30개)":    "BRE",
    "3. Ovarian cancer (70개)":   "OVA",
    "4. Lung cancer (300개)":     "LUN",
    "5. Normal (100개)":          "NOR",
    "6. Diabetes (100개)":        "DIA",
    "7. High blood pressure (100개)":              "HBP",
    "8. High blood pressure + Diabetes (100개)":   "H.D.",
    "9. Colorectal cancer (300개)":                "CRC",
    "10-1. C-Pancreatic cancer (70개)":            "CPAN",
    "10-3. Y-Pancreatic cancer (30개)":            "YPAN",
    "11. Bladdder Cancer (299개)":                 "BLC",
    "12. Y-Normal (29개)":                         "YNOR",
}

# Shared wavenumber grid (matches production preprocessing)
GRID = np.linspace(402.0, 2198.0, 933)

# Medical wavenumber shift (urea peak alignment)
MEDICAL_SHIFT = -28.0

# NOR replicates 7-20 are machine QC, drop them
NOR_MAX_REPLICATE = 6

# Two-stage classifier groups
CANCER_TYPES = ("PRO", "LUN", "CRC", "PAN", "OVA", "BRE", "BLC")
NON_CANCER_GROUPS = ("NOR", "DIA", "HBP", "H.D.")

# Group aliases (CPAN/YPAN → PAN, YNOR → NOR)
GROUP_ALIASES = {"CPAN": "PAN", "YPAN": "PAN", "YNOR": "NOR"}

PAIRED_TRAIN_FRACTION = 0.5      # half of paired samples used for fitting transformers
RANDOM_STATE = 42


# =============================================================================
# Loading + preprocessing onto shared grid
# =============================================================================
def _preprocess_thermo(x, y):
    return preprocess_single_spectrum(
        x, y, GRID,
        do_trim=True, trim_region=(400, 2200),
        do_smooth=True, smooth_window=11, smooth_poly=3,
        do_baseline=True, baseline_window=101,
        normalization="snv",
    )


def _preprocess_medical(x, y):
    # Medical BG-removed spectra are already smoothed; skip SG smoothing.
    return preprocess_single_spectrum(
        x, y, GRID,
        do_trim=True, trim_region=(400, 2200),
        do_smooth=False,
        do_baseline=True, baseline_window=101,
        normalization="snv",
    )


def load_domain(
    data_dir: Path,
    folder_map: Dict[str, str],
    pattern: str,
    preprocess_fn,
    wavenumber_shift: float = 0.0,
    read_from_subdir: str | None = None,
) -> pd.DataFrame:
    """Load every spectrum from a domain and preprocess onto the shared grid.

    Returns one row per replicate with columns: group, sample_id, replicate, x_*
    """
    rows = []
    n_failed = 0
    feature_cols = [f"x_{wn:.2f}" for wn in GRID]

    for folder_name, group in folder_map.items():
        folder = data_dir / folder_name
        source_dir = folder / read_from_subdir if read_from_subdir else folder
        if not source_dir.is_dir():
            logger.warning(f"Missing folder: {source_dir}")
            continue

        files = find_spectra(source_dir, pattern=pattern, recursive=False)
        # Match Thermo .CSV uppercase too if .csv pattern returns nothing
        if not files and pattern == "*.csv":
            files = find_spectra(source_dir, pattern="*.CSV", recursive=False)
        files = [
            f for f in files
            if "_ave" not in f.stem.lower()
            and "zone.identifier" not in f.name.lower()
            and "multidata" not in f.stem.lower()
        ]

        for fp in files:
            try:
                sid = parse_filename(fp, fallback_group=group)
                if sid.group == "NOR" and sid.replicate > NOR_MAX_REPLICATE:
                    continue
                x, y = read_spectrum(fp)
                if wavenumber_shift != 0.0:
                    x = x + wavenumber_shift
                y_proc = preprocess_fn(x, y)
                row = {
                    "group": sid.group,
                    "sample_id": sid.sample_id,
                    "replicate": sid.replicate,
                }
                row.update(dict(zip(feature_cols, y_proc)))
                rows.append(row)
            except Exception as e:
                n_failed += 1
                logger.debug(f"Failed {fp.name}: {e}")

    df = pd.DataFrame(rows)
    logger.info(
        f"  {data_dir.name}: {len(df)} spectra, "
        f"{df['group'].nunique()} groups, {n_failed} failed"
    )
    return df


def aggregate_to_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Mean over replicates → one row per (group, sample_id)."""
    feat_cols = [c for c in df.columns if c.startswith("x_")]
    sample = df.groupby(["group", "sample_id"], as_index=False)[feat_cols].mean()
    return sample


def apply_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Disambiguate sample_ids before group merge, then apply aliases."""
    df = df.copy()
    needs_prefix = df["group"].isin(["YPAN", "YNOR"])
    df.loc[needs_prefix, "sample_id"] = (
        df.loc[needs_prefix, "group"].astype(str) + "_"
        + df.loc[needs_prefix, "sample_id"].astype(str)
    )
    df["group"] = df["group"].replace(GROUP_ALIASES)
    return df


# =============================================================================
# Diagnostics (Step 0)
# =============================================================================
def compute_paired_stats(
    Xs: np.ndarray, Xt: np.ndarray, label_s: str, label_t: str,
    transformer=None,
) -> dict:
    if transformer is not None:
        Xs = transformer.transform(Xs)
    return {
        "rmse": paired_rmse(Xs, Xt),
        "mean_correlation": paired_correlation(Xs, Xt),
        "n_pairs": int(Xs.shape[0]),
        "label": f"{label_s}->{label_t}",
    }


def plot_paired_overlays(
    paired: pd.DataFrame, X_t: np.ndarray, X_m: np.ndarray, out_dir: Path,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = sorted(paired["group"].unique())
    for g in groups:
        idx = paired.index[paired["group"] == g][:6]
        if len(idx) == 0:
            continue
        n = len(idx)
        fig, axes = plt.subplots(n, 1, figsize=(10, 1.6 * n), sharex=True)
        if n == 1:
            axes = [axes]
        for ax, i in zip(axes, idx):
            ax.plot(GRID, X_t[i], color="#2196F3", lw=1, label="Thermo")
            ax.plot(GRID, X_m[i], color="#E91E63", lw=1, label="Medical", alpha=0.85)
            ax.set_ylabel(f"{paired.loc[i, 'sample_id']}", fontsize=8)
            ax.legend(loc="upper right", fontsize=7)
            ax.grid(alpha=0.3)
        axes[-1].set_xlabel("Raman shift (cm⁻¹)")
        fig.suptitle(f"Paired overlay — {g}", fontweight="bold")
        plt.tight_layout()
        fig.savefig(out_dir / f"paired_overlay_{g}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)


def plot_difference_pca(X_t: np.ndarray, X_m: np.ndarray, paired: pd.DataFrame,
                        out_path: Path):
    D = X_t - X_m
    Dc = D - D.mean(axis=0)
    U, s, Vt = np.linalg.svd(Dc, full_matrices=False)
    var_ratio = (s ** 2) / (s ** 2).sum()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(np.arange(1, len(var_ratio) + 1), var_ratio.cumsum(), "o-")
    axes[0].set_xlim(0, 20)
    axes[0].set_xlabel("Component")
    axes[0].set_ylabel("Cumulative variance")
    axes[0].set_title("Paired-difference PCA — instrument bias subspace")
    axes[0].grid(alpha=0.3)
    axes[0].axhline(0.9, color="gray", ls="--", lw=0.8)

    for k in range(min(3, Vt.shape[0])):
        axes[1].plot(GRID, Vt[k], lw=1, label=f"PC{k+1} ({var_ratio[k]*100:.1f}%)")
    axes[1].set_xlabel("Raman shift (cm⁻¹)")
    axes[1].set_ylabel("Loading")
    axes[1].set_title("Top 3 bias directions")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return var_ratio[:10].tolist()


# =============================================================================
# LR two-stage evaluation
# =============================================================================
def _two_stage_eval(
    df_train: pd.DataFrame, df_test: pd.DataFrame, feat_cols: List[str],
) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    valid = set(CANCER_TYPES) | set(NON_CANCER_GROUPS)
    df_train = df_train[df_train["group"].isin(valid)].copy()
    df_test = df_test[df_test["group"].isin(valid)].copy()

    Xtr = df_train[feat_cols].values
    Xte = df_test[feat_cols].values
    ytr = (df_train["group"].isin(CANCER_TYPES)).astype(int).values
    yte = (df_test["group"].isin(CANCER_TYPES)).astype(int).values
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
    acc = accuracy_score(yte, pred1)
    f1 = f1_score(yte, pred1, average="macro")

    # Stage 2
    mtr = ctl_tr >= 0
    mte = ctl_te >= 0
    s2_f1 = float("nan")
    if mtr.sum() > 0 and mte.sum() > 0 and len(np.unique(ctl_tr[mtr])) >= 2:
        s2 = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, max_iter=2000, solver="lbfgs", class_weight="balanced",
            multi_class="multinomial"))
        s2.fit(Xtr[mtr], ctl_tr[mtr])
        pred2 = s2.predict(Xte[mte])
        s2_f1 = f1_score(ctl_te[mte], pred2, average="macro", zero_division=0)

    per_group = {}
    for g in sorted(df_test["group"].unique()):
        m = df_test["group"].values == g
        per_group[g] = {
            "n": int(m.sum()),
            "mean_prob": float(p1[m].mean()),
        }

    return {
        "n_train": int(len(df_train)),
        "n_test": int(len(df_test)),
        "s1_auc": float(auc),
        "s1_acc": float(acc),
        "s1_f1": float(f1),
        "s2_f1": float(s2_f1),
        "per_group": per_group,
    }


# =============================================================================
# Main
# =============================================================================
def main():
    t0 = datetime.now()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("  Cross-Instrument Calibration Transfer (bidirectional)")
    logger.info("=" * 70)

    # ----- 1. Load + preprocess both domains onto shared grid ----------------
    logger.info("\n[1] Loading Thermo …")
    df_t_rep = load_domain(THERMO_DIR, THERMO_FOLDER_MAP, "*.CSV",
                            _preprocess_thermo)
    logger.info("\n[1] Loading Medical …")
    df_m_rep = load_domain(MEDICAL_DIR, MEDICAL_FOLDER_MAP, "*.txt",
                            _preprocess_medical,
                            wavenumber_shift=MEDICAL_SHIFT,
                            read_from_subdir="Background")

    if df_t_rep.empty or df_m_rep.empty:
        logger.error("One of the domains is empty — aborting.")
        return

    # ----- 2. Aggregate to sample-level mean ---------------------------------
    df_t = aggregate_to_sample(df_t_rep)
    df_m = aggregate_to_sample(df_m_rep)
    logger.info(f"  Thermo sample-level rows : {len(df_t)}")
    logger.info(f"  Medical sample-level rows: {len(df_m)}")

    # Apply group aliases AFTER computing pairing key (which uses raw groups)
    feat_cols = [f"x_{wn:.2f}" for wn in GRID]

    # ----- 3. Identify paired samples ----------------------------------------
    key_t = set(zip(df_t["group"], df_t["sample_id"]))
    key_m = set(zip(df_m["group"], df_m["sample_id"]))
    paired_keys = sorted(key_t & key_m)
    logger.info(f"\n[2] Paired samples (intersection): {len(paired_keys)}")
    if len(paired_keys) < 50:
        logger.error("Too few paired samples; aborting.")
        return

    paired_df = pd.DataFrame(paired_keys, columns=["group", "sample_id"])
    paired_df.to_csv(OUT_DIR / "paired_samples.csv", index=False)

    # Cache sample-level dataframes for downstream sweeps
    df_t.to_parquet(OUT_DIR / "thermo_sample_level.parquet", index=False)
    df_m.to_parquet(OUT_DIR / "medical_sample_level.parquet", index=False)

    # Build paired matrices in matched order
    df_t_idx = df_t.set_index(["group", "sample_id"])
    df_m_idx = df_m.set_index(["group", "sample_id"])
    X_t_paired = df_t_idx.loc[paired_keys, feat_cols].values
    X_m_paired = df_m_idx.loc[paired_keys, feat_cols].values
    paired_df_full = paired_df.copy()

    logger.info(
        f"  Paired matrix shapes: Thermo {X_t_paired.shape}, "
        f"Medical {X_m_paired.shape}"
    )

    # ----- 4. Step 0 diagnostics ---------------------------------------------
    logger.info("\n[3] Step 0  —  diagnostics")
    raw_rmse = paired_rmse(X_t_paired, X_m_paired)
    raw_corr = paired_correlation(X_t_paired, X_m_paired)
    logger.info(f"  raw paired RMSE: {raw_rmse:.4f}   mean corr: {raw_corr:.4f}")

    plot_paired_overlays(paired_df_full, X_t_paired, X_m_paired, DIAG_DIR)
    var_ratio = plot_difference_pca(
        X_t_paired, X_m_paired, paired_df_full,
        DIAG_DIR / "paired_difference_pca.png",
    )
    pd.DataFrame({
        "metric": ["rmse_raw", "corr_raw"],
        "value": [raw_rmse, raw_corr],
    }).to_csv(DIAG_DIR / "paired_stats.csv", index=False)

    # ----- 5. Train/test split for paired calibration ------------------------
    rng = np.random.default_rng(RANDOM_STATE)
    n_pair = len(paired_keys)
    perm = rng.permutation(n_pair)
    n_fit = int(round(n_pair * PAIRED_TRAIN_FRACTION))
    fit_idx = perm[:n_fit]
    held_idx = perm[n_fit:]
    logger.info(f"  Calibration fit pairs: {len(fit_idx)} | held-out: {len(held_idx)}")

    X_t_fit, X_t_held = X_t_paired[fit_idx], X_t_paired[held_idx]
    X_m_fit, X_m_held = X_m_paired[fit_idx], X_m_paired[held_idx]

    # ----- 6. Fit transformers (BOTH directions) -----------------------------
    logger.info("\n[4] Fitting calibration transformers …")

    def _build_transformers(direction: str, Xs_fit, Xt_fit):
        affine = AffineTransfer().fit(Xs_fit, Xt_fit)
        pds = PDSTransfer(half_window=7, ridge=1.0).fit(Xs_fit, Xt_fit)
        osc = OSCTransfer(n_components=2).fit(Xs_fit, Xt_fit)
        chain = ChainTransfer(steps=[
            PDSTransfer(half_window=7, ridge=1.0),
            OSCTransfer(n_components=2),
        ]).fit(Xs_fit, Xt_fit)
        return [
            ("raw",       IdentityTransfer()),
            ("affine",    affine),
            ("pds",       pds),
            ("osc",       osc),
            ("pds+osc",   chain),
        ]

    transformers_m2t = _build_transformers("m2t", X_m_fit, X_t_fit)  # Medical→Thermo
    transformers_t2m = _build_transformers("t2m", X_t_fit, X_m_fit)  # Thermo→Medical

    # Held-out paired RMSE / correlation per method
    held_stats = []
    for name, tr in transformers_m2t:
        Xm2t = tr.transform(X_m_held)
        held_stats.append({
            "direction": "Medical→Thermo",
            "method": name,
            "held_rmse": paired_rmse(Xm2t, X_t_held),
            "held_corr": paired_correlation(Xm2t, X_t_held),
        })
    for name, tr in transformers_t2m:
        Xt2m = tr.transform(X_t_held)
        held_stats.append({
            "direction": "Thermo→Medical",
            "method": name,
            "held_rmse": paired_rmse(Xt2m, X_m_held),
            "held_corr": paired_correlation(Xt2m, X_m_held),
        })
    held_df = pd.DataFrame(held_stats)
    held_df.to_csv(OUT_DIR / "paired_held_metrics.csv", index=False)
    logger.info("  paired held-out metrics:\n" + held_df.to_string(index=False))

    # ----- 7. Build mapped training sets and evaluate ------------------------
    # We re-use the FULL Thermo / Medical datasets (not just paired) for training
    # the downstream classifier — calibration only changes the feature mapping.
    df_t_aliased = apply_aliases(df_t)
    df_m_aliased = apply_aliases(df_m)

    def transform_full(df: pd.DataFrame, tr) -> pd.DataFrame:
        if isinstance(tr, IdentityTransfer):
            return df
        out = df.copy()
        out[feat_cols] = tr.transform(df[feat_cols].values)
        return out

    leaderboard = []
    detail = {"Medical→Thermo": {}, "Thermo→Medical": {}}

    logger.info("\n[5] Bidirectional LR two-stage evaluation")

    # Direction A: Medical→Thermo  →  train a Medical-mapped-to-Thermo model and
    # evaluate on the Thermo dataset (the target domain).
    logger.info("\n  ── Medical → Thermo (test = Thermo full set) ──")
    for name, tr in transformers_m2t:
        df_train = transform_full(df_m_aliased, tr)
        df_test = df_t_aliased
        res = _two_stage_eval(df_train, df_test, feat_cols)
        res["method"] = name
        res["direction"] = "Medical→Thermo"
        detail["Medical→Thermo"][name] = res
        leaderboard.append({
            "direction": "Medical→Thermo",
            "method": name,
            "n_train": res["n_train"],
            "n_test": res["n_test"],
            "s1_auc": res["s1_auc"],
            "s1_f1": res["s1_f1"],
            "s2_f1": res["s2_f1"],
        })
        logger.info(
            f"    {name:9s}  AUC={res['s1_auc']:.4f}  S1F1={res['s1_f1']:.4f}  "
            f"S2F1={res['s2_f1']:.4f}"
        )

    # Direction B: Thermo→Medical
    logger.info("\n  ── Thermo → Medical (test = Medical full set) ──")
    for name, tr in transformers_t2m:
        df_train = transform_full(df_t_aliased, tr)
        df_test = df_m_aliased
        res = _two_stage_eval(df_train, df_test, feat_cols)
        res["method"] = name
        res["direction"] = "Thermo→Medical"
        detail["Thermo→Medical"][name] = res
        leaderboard.append({
            "direction": "Thermo→Medical",
            "method": name,
            "n_train": res["n_train"],
            "n_test": res["n_test"],
            "s1_auc": res["s1_auc"],
            "s1_f1": res["s1_f1"],
            "s2_f1": res["s2_f1"],
        })
        logger.info(
            f"    {name:9s}  AUC={res['s1_auc']:.4f}  S1F1={res['s1_f1']:.4f}  "
            f"S2F1={res['s2_f1']:.4f}"
        )

    # ----- 8. Self-CV reference (no calibration) -----------------------------
    logger.info("\n[6] Self-instrument reference (train=test domain, sanity)")
    self_thermo = _two_stage_eval(df_t_aliased, df_t_aliased, feat_cols)
    self_medical = _two_stage_eval(df_m_aliased, df_m_aliased, feat_cols)
    logger.info(f"  Thermo  self : AUC={self_thermo['s1_auc']:.4f} S2F1={self_thermo['s2_f1']:.4f}")
    logger.info(f"  Medical self : AUC={self_medical['s1_auc']:.4f} S2F1={self_medical['s2_f1']:.4f}")

    # ----- 9. Save reports ---------------------------------------------------
    leaderboard_df = pd.DataFrame(leaderboard)
    leaderboard_df.to_csv(OUT_DIR / "leaderboard.csv", index=False)

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "duration_seconds": (datetime.now() - t0).total_seconds(),
        "config": {
            "grid": {"x_min": float(GRID[0]), "x_max": float(GRID[-1]),
                     "n_points": int(GRID.size)},
            "medical_wavenumber_shift": MEDICAL_SHIFT,
            "paired_train_fraction": PAIRED_TRAIN_FRACTION,
            "random_state": RANDOM_STATE,
            "cancer_types": list(CANCER_TYPES),
            "non_cancer_groups": list(NON_CANCER_GROUPS),
        },
        "n_paired_samples": len(paired_keys),
        "n_calibration_fit": int(len(fit_idx)),
        "n_calibration_held": int(len(held_idx)),
        "raw_paired_rmse": raw_rmse,
        "raw_paired_correlation": raw_corr,
        "difference_pca_top10_var_ratio": var_ratio,
        "self_reference": {
            "thermo": self_thermo,
            "medical": self_medical,
        },
        "results": detail,
    }
    with open(OUT_DIR / "calibration_report.json", "w") as f:
        json.dump(report, f, indent=2, default=float)

    logger.info(f"\n[7] Done in {(datetime.now() - t0).total_seconds():.1f}s")
    logger.info(f"  Outputs → {OUT_DIR}")
    logger.info("\nLeaderboard:\n" + leaderboard_df.to_string(index=False))


if __name__ == "__main__":
    main()
