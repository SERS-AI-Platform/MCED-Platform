#!/usr/bin/env python3
"""Experiment 2: Leave-One-Hospital-Out (LOHO) Validation.

7 hospital sites -> for each core hospital, train on remaining hospitals,
test on held-out -> external validation performance + 1,000 permutation
tests for significance.

Phase 1: LOHO on 5 core hospitals (Chungbuk, SNUH, StMarys, Yangsan, Inje)
Phase 2: External validation on Boramae (BPRO/BNOR) and Yonsei (YPAN/YNOR)
Phase 3: Standard 5-fold CV baseline for generalization gap analysis

Usage:
    python scripts/analysis/weekend_experiments/loho_validation.py
    python scripts/analysis/weekend_experiments/loho_validation.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.sers.config import RESULTS_DIR, FIG_DIR  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CANCER_TYPES = ["PRO", "BRE", "OVA", "LUN", "CRC", "PAN", "BLC"]
NON_CANCER = ["NOR", "DIA", "HBP", "H.D."]
ALL_STANDARD_GROUPS = CANCER_TYPES + NON_CANCER

# Aliases: raw group -> canonical group
GROUP_ALIASES = {"CPAN": "PAN", "YPAN": "PAN", "YNOR": "NOR"}

# Core hospitals for LOHO (standard cohort)
HOSPITAL_GROUPS = {
    "Chungbuk": {
        "groups": {"PRO": (1, 100), "CRC": (1, 300), "CPAN": (1, 70), "BLC": (1, 299)},
        "label": "Chungbuk National University Hospital",
    },
    "SNUH": {
        "groups": {"OVA": (31, 70), "LUN": (1, 130)},
        "label": "Seoul National University Hospital",
    },
    "StMarys": {
        "groups": {"LUN": (131, 300)},
        "label": "Seoul St. Mary's Hospital",
    },
    "Yangsan": {
        "groups": {"NOR": (1, 100), "DIA": (1, 100), "HBP": (1, 100), "H.D.": (1, 100)},
        "label": "Yangsan Pusan National University Hospital",
    },
    "Inje": {
        "groups": {"BRE": (1, 30), "OVA": (1, 30)},
        "label": "Inje University Busan Paik Hospital",
    },
}

# Prospective validation cohorts
EXTERNAL_COHORTS = {
    "Boramae": {
        "groups": ["BPRO", "BNOR"],
        "label": "Seoul National University Boramae Hospital",
    },
    "Yonsei": {
        "groups": ["YPAN", "YNOR"],
        "label": "Yonsei Severance Hospital",
    },
}

N_PERMUTATIONS = 1_000
RANDOM_STATE = 42

logger = logging.getLogger("loho_validation")


# ===========================================================================
# Helpers
# ===========================================================================
def setup_logging(out_dir: Path) -> None:
    """Configure logging to file and console."""
    log_file = out_dir / "experiment.log"
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    handlers = [
        logging.FileHandler(log_file, mode="w"),
        logging.StreamHandler(sys.stdout),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)


def atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON atomically via temp-file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def elapsed_str(t0: float) -> str:
    e = time.time() - t0
    m, s = divmod(int(e), 60)
    return f"{m:02d}:{s:02d}"


def eta_str(t0: float, done: int, total: int) -> str:
    if done == 0:
        return "??:??"
    e = time.time() - t0
    remaining = e / done * (total - done)
    m, s = divmod(int(remaining), 60)
    return f"{m:02d}:{s:02d}"


# ===========================================================================
# Feature extraction
# ===========================================================================
def extract_1st_derivative(X: np.ndarray) -> np.ndarray:
    """Savitzky-Golay 1st derivative (window=11, poly=3)."""
    return np.apply_along_axis(lambda y: savgol_filter(y, 11, 3, deriv=1), 1, X)


# ===========================================================================
# Data loading
# ===========================================================================
def load_data() -> Tuple[pd.DataFrame, List[str], np.ndarray]:
    """Load processed spectra and return aggregated df, feat cols, wavenumbers."""
    spec_path = RESULTS_DIR / "processed_spectra.csv"
    logger.info(f"Loading spectra from {spec_path}")
    spec_df = pd.read_csv(spec_path)

    feat_cols = [c for c in spec_df.columns if c.startswith("x_")]
    wavenumbers = np.array([float(c.replace("x_", "")) for c in feat_cols])

    # Mean aggregation per sample
    agg_dict = {c: "mean" for c in feat_cols}
    agg_dict["replicate"] = "count"
    agg_df = spec_df.groupby(["group", "sample_id"], as_index=False).agg(agg_dict)

    logger.info(f"Loaded {len(spec_df)} spectra -> {len(agg_df)} aggregated samples")
    logger.info(f"Groups: {sorted(agg_df['group'].unique())}")
    return agg_df, feat_cols, wavenumbers


# ===========================================================================
# Hospital assignment
# ===========================================================================
def assign_hospitals(agg_df: pd.DataFrame) -> np.ndarray:
    """Assign each sample to a hospital based on group + sample_id ranges.

    Returns array of hospital names (same length as agg_df).
    Samples not matching any hospital get 'Unknown'.
    """
    hospitals = np.full(len(agg_df), "Unknown", dtype=object)

    for idx, row in agg_df.iterrows():
        grp = row["group"]
        try:
            sid = int(row["sample_id"])
        except (ValueError, TypeError):
            continue

        # Check core hospitals
        for hosp_name, hosp_info in HOSPITAL_GROUPS.items():
            for grp_code, (lo, hi) in hosp_info["groups"].items():
                if grp == grp_code and lo <= sid <= hi:
                    hospitals[idx] = hosp_name
                    break
            if hospitals[idx] != "Unknown":
                break

        # Check external cohorts (by raw group name)
        if hospitals[idx] == "Unknown":
            for cohort_name, cohort_info in EXTERNAL_COHORTS.items():
                if grp in cohort_info["groups"]:
                    hospitals[idx] = cohort_name
                    break

    return hospitals


def prepare_labels(
    agg_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create binary labels and cancer-type labels.

    Returns
    -------
    binary_labels : 1=cancer, 0=non-cancer
    cancer_type_labels : integer label for cancer type (>=0) or -1 for non-cancer
    """
    # Map raw groups to canonical
    canonical = agg_df["group"].map(lambda g: GROUP_ALIASES.get(g, g)).values

    binary = np.array([1 if g in CANCER_TYPES else 0 for g in canonical])

    # Cancer type encoding
    ct_map = {c: i for i, c in enumerate(CANCER_TYPES)}
    ctl = np.array([ct_map.get(g, -1) for g in canonical])

    return binary, ctl


# ===========================================================================
# Model training / evaluation
# ===========================================================================
def make_s1_model():
    """Stage 1: Binary cancer vs non-cancer."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0, max_iter=1000, solver="saga",
            class_weight="balanced", random_state=RANDOM_STATE,
        ),
    )


def make_s2_model():
    """Stage 2: Cancer type classification (multinomial)."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0, max_iter=1000, solver="saga",
            class_weight="balanced", multi_class="multinomial",
            random_state=RANDOM_STATE,
        ),
    )


def evaluate_fold(
    X_train: np.ndarray, X_test: np.ndarray,
    bl_train: np.ndarray, bl_test: np.ndarray,
    ctl_train: np.ndarray, ctl_test: np.ndarray,
) -> Dict[str, Any]:
    """Train two-stage model and evaluate. Returns dict of metrics."""
    X_train_d1 = extract_1st_derivative(X_train)
    X_test_d1 = extract_1st_derivative(X_test)

    result: Dict[str, Any] = {}

    # ---- Stage 1: Binary ----
    m1 = make_s1_model()
    m1.fit(X_train_d1, bl_train)
    s1_prob = m1.predict_proba(X_test_d1)[:, 1]
    s1_pred = m1.predict(X_test_d1)

    result["s1_auc"] = roc_auc_score(bl_test, s1_prob)
    result["s1_sens"] = recall_score(bl_test, s1_pred, pos_label=1, zero_division=0)
    result["s1_spec"] = recall_score(bl_test, s1_pred, pos_label=0, zero_division=0)

    # ---- Stage 2: Cancer type (cancer-only subset) ----
    cancer_tr = ctl_train >= 0
    cancer_te = ctl_test >= 0

    if cancer_tr.sum() > 10 and cancer_te.sum() > 0:
        m2 = make_s2_model()
        m2.fit(X_train_d1[cancer_tr], ctl_train[cancer_tr])
        s2_pred = m2.predict(X_test_d1[cancer_te])
        s2_true = ctl_test[cancer_te]

        result["s2_f1"] = f1_score(s2_true, s2_pred, average="macro", zero_division=0)

        # Per-cancer sensitivity
        per_cancer = {}
        for ci, cname in enumerate(CANCER_TYPES):
            mask = s2_true == ci
            if mask.sum() > 0:
                per_cancer[cname] = float(recall_score(
                    s2_true == ci, s2_pred == ci, pos_label=True, zero_division=0,
                ))
        result["per_cancer_sens"] = per_cancer
        result["cancer_types_in_test"] = list(per_cancer.keys())
    else:
        result["s2_f1"] = np.nan
        result["per_cancer_sens"] = {}
        result["cancer_types_in_test"] = []

    return result


# ===========================================================================
# Permutation test
# ===========================================================================
def permutation_test(
    X_train: np.ndarray, X_test: np.ndarray,
    bl_train: np.ndarray, bl_test: np.ndarray,
    ctl_train: np.ndarray, ctl_test: np.ndarray,
    observed: Dict[str, float],
    n_perm: int = N_PERMUTATIONS,
    checkpoint_path: Optional[Path] = None,
) -> Dict[str, float]:
    """Run permutation test for S1 AUC and S2 F1.

    Returns dict of p-values.
    """
    rng = np.random.RandomState(RANDOM_STATE)
    X_train_d1 = extract_1st_derivative(X_train)
    X_test_d1 = extract_1st_derivative(X_test)

    # Load checkpoint if exists
    start_perm = 0
    perm_s1_aucs: List[float] = []
    perm_s2_f1s: List[float] = []
    if checkpoint_path and checkpoint_path.exists():
        ckpt = json.loads(checkpoint_path.read_text())
        start_perm = ckpt.get("completed", 0)
        perm_s1_aucs = ckpt.get("perm_s1_aucs", [])
        perm_s2_f1s = ckpt.get("perm_s2_f1s", [])
        logger.info(f"  Resumed permutation from {start_perm}/{n_perm}")

    # Pre-train models once (we only permute test labels)
    m1 = make_s1_model()
    m1.fit(X_train_d1, bl_train)
    s1_prob = m1.predict_proba(X_test_d1)[:, 1]

    cancer_tr = ctl_train >= 0
    cancer_te = ctl_test >= 0
    has_s2 = cancer_tr.sum() > 10 and cancer_te.sum() > 0
    if has_s2:
        m2 = make_s2_model()
        m2.fit(X_train_d1[cancer_tr], ctl_train[cancer_tr])
        s2_pred = m2.predict(X_test_d1[cancer_te])

    # Advance RNG to correct position
    for _ in range(start_perm):
        rng.permutation(len(bl_test))
        if has_s2:
            rng.permutation(int(cancer_te.sum()))

    for i in range(start_perm, n_perm):
        # S1: permute binary labels
        perm_bl = bl_test[rng.permutation(len(bl_test))]
        try:
            perm_auc = roc_auc_score(perm_bl, s1_prob)
        except ValueError:
            perm_auc = 0.5
        perm_s1_aucs.append(perm_auc)

        # S2: permute cancer type labels
        if has_s2:
            perm_ctl = ctl_test[cancer_te][rng.permutation(int(cancer_te.sum()))]
            perm_f1 = f1_score(perm_ctl, s2_pred, average="macro", zero_division=0)
            perm_s2_f1s.append(perm_f1)

        # Checkpoint every 200 iterations
        if checkpoint_path and (i + 1) % 200 == 0:
            atomic_write_json(checkpoint_path, {
                "completed": i + 1,
                "perm_s1_aucs": perm_s1_aucs,
                "perm_s2_f1s": perm_s2_f1s,
            })

    # Save final checkpoint
    if checkpoint_path:
        atomic_write_json(checkpoint_path, {
            "completed": n_perm,
            "perm_s1_aucs": perm_s1_aucs,
            "perm_s2_f1s": perm_s2_f1s,
        })

    # Compute p-values
    obs_auc = observed.get("s1_auc", 0)
    p_s1 = (np.sum(np.array(perm_s1_aucs) >= obs_auc) + 1) / (n_perm + 1)

    p_s2 = np.nan
    if has_s2 and not np.isnan(observed.get("s2_f1", np.nan)):
        obs_f1 = observed["s2_f1"]
        p_s2 = (np.sum(np.array(perm_s2_f1s) >= obs_f1) + 1) / (n_perm + 1)

    return {"p_s1_auc": float(p_s1), "p_s2_f1": float(p_s2)}


# ===========================================================================
# Standard 5-fold CV baseline
# ===========================================================================
def run_cv_baseline(
    X: np.ndarray, bl: np.ndarray, ctl: np.ndarray,
) -> Dict[str, float]:
    """Run stratified 5-fold CV and return mean metrics."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    aucs, senss, specs, f1s = [], [], [], []

    for train_idx, test_idx in skf.split(X, bl):
        res = evaluate_fold(
            X[train_idx], X[test_idx],
            bl[train_idx], bl[test_idx],
            ctl[train_idx], ctl[test_idx],
        )
        aucs.append(res["s1_auc"])
        senss.append(res["s1_sens"])
        specs.append(res["s1_spec"])
        if not np.isnan(res["s2_f1"]):
            f1s.append(res["s2_f1"])

    return {
        "cv_s1_auc": float(np.mean(aucs)),
        "cv_s1_auc_std": float(np.std(aucs)),
        "cv_s1_sens": float(np.mean(senss)),
        "cv_s1_spec": float(np.mean(specs)),
        "cv_s2_f1": float(np.mean(f1s)) if f1s else np.nan,
        "cv_s2_f1_std": float(np.std(f1s)) if f1s else np.nan,
    }


# ===========================================================================
# Visualization
# ===========================================================================
def plot_hospital_performance(results_df: pd.DataFrame, out_dir: Path) -> None:
    """Grouped bar chart: S1 AUC + S2 F1 per hospital."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    hospitals = results_df["hospital"].values
    x = np.arange(len(hospitals))
    w = 0.35

    # S1 AUC
    ax = axes[0]
    ax.bar(x, results_df["s1_auc"].values, w, label="AUC", color="#4C72B0")
    ax.bar(x + w, results_df["s1_sens"].values, w, label="Sensitivity", color="#55A868")
    ax.set_xticks(x + w / 2)
    ax.set_xticklabels(hospitals, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Stage 1: Cancer Detection (Binary)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.axhline(0.5, ls="--", color="gray", alpha=0.5)

    # S2 F1
    ax = axes[1]
    f1_vals = results_df["s2_f1"].values
    valid = ~np.isnan(f1_vals)
    colors = ["#DD8452" if v else "#CCCCCC" for v in valid]
    ax.bar(x, np.where(valid, f1_vals, 0), 0.6, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(hospitals, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Macro F1")
    ax.set_title("Stage 2: Cancer Type Classification")
    ax.set_ylim(0, 1.05)

    fig.suptitle("LOHO Validation: Per-Hospital Performance", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "hospital_performance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved hospital_performance.png")


def plot_generalization_gap(gap_df: pd.DataFrame, out_dir: Path) -> None:
    """Heatmap of LOHO vs CV metrics."""
    metrics = ["s1_auc", "s1_sens", "s1_spec", "s2_f1"]
    hospitals = list(gap_df["hospital"].values) + ["5-Fold CV"]

    data = np.zeros((len(hospitals), len(metrics)))
    for i, row in gap_df.iterrows():
        for j, m in enumerate(metrics):
            data[i, j] = row[m] if not np.isnan(row[m]) else 0

    # Last row = CV baseline
    for j, m in enumerate(metrics):
        cv_col = f"cv_{m}"
        if cv_col in gap_df.columns:
            vals = gap_df[cv_col].dropna()
            data[-1, j] = vals.iloc[0] if len(vals) > 0 else 0

    fig, ax = plt.subplots(figsize=(8, max(4, len(hospitals) * 0.6 + 1)))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(["S1 AUC", "S1 Sens", "S1 Spec", "S2 F1"], fontsize=10)
    ax.set_yticks(range(len(hospitals)))
    ax.set_yticklabels(hospitals, fontsize=10)

    for i in range(len(hospitals)):
        for j in range(len(metrics)):
            val = data[i, j]
            txt = f"{val:.3f}" if val > 0 else "N/A"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color="white" if val < 0.5 else "black")

    fig.colorbar(im, ax=ax, shrink=0.7)
    ax.set_title("Generalization Gap: LOHO vs 5-Fold CV", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "generalization_gap_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved generalization_gap_heatmap.png")


# ===========================================================================
# Main experiment
# ===========================================================================
def run_loho(
    agg_df: pd.DataFrame,
    feat_cols: List[str],
    hospitals: np.ndarray,
    bl: np.ndarray,
    ctl: np.ndarray,
    out_dir: Path,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Phase 1: Leave-One-Hospital-Out on 5 core hospitals."""
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    X = agg_df[feat_cols].values
    core_hospitals = list(HOSPITAL_GROUPS.keys())

    # Filter to standard cohort samples that belong to a core hospital
    core_mask = np.isin(hospitals, core_hospitals)
    logger.info(f"Core hospital samples: {core_mask.sum()} / {len(hospitals)}")

    if dry_run:
        core_hospitals = core_hospitals[:1]
        logger.info(f"DRY RUN: only processing {core_hospitals[0]}")

    results = []
    t0 = time.time()

    for hi, hosp in enumerate(core_hospitals):
        logger.info(f"\n{'='*60}")
        logger.info(f"LOHO fold {hi+1}/{len(core_hospitals)}: hold out {hosp}")
        logger.info(f"  Elapsed: {elapsed_str(t0)} | ETA: {eta_str(t0, hi, len(core_hospitals))}")

        # Check fold checkpoint
        fold_ckpt = ckpt_dir / f"loho_{hosp}_lr.json"
        if fold_ckpt.exists():
            cached = json.loads(fold_ckpt.read_text())
            if cached.get("complete"):
                logger.info(f"  Loaded from checkpoint")
                results.append(cached["result"])

                # Still need to run permutation test if not done
                perm_ckpt = ckpt_dir / f"perm_{hosp}_lr.json"
                if not perm_ckpt.exists() or json.loads(perm_ckpt.read_text()).get("completed", 0) < N_PERMUTATIONS:
                    test_mask = core_mask & (hospitals == hosp)
                    train_mask = core_mask & (hospitals != hosp)
                    n_perm = 10 if dry_run else N_PERMUTATIONS
                    logger.info(f"  Running {n_perm} permutations...")
                    permutation_test(
                        X[train_mask], X[test_mask],
                        bl[train_mask], bl[test_mask],
                        ctl[train_mask], ctl[test_mask],
                        cached["result"], n_perm, perm_ckpt,
                    )
                continue

        test_mask = core_mask & (hospitals == hosp)
        train_mask = core_mask & (hospitals != hosp)

        n_train, n_test = train_mask.sum(), test_mask.sum()
        logger.info(f"  Train: {n_train}, Test: {n_test}")
        logger.info(f"  Test groups: {sorted(agg_df.loc[test_mask, 'group'].unique())}")

        if n_test == 0:
            logger.warning(f"  No test samples for {hosp}, skipping")
            continue

        res = evaluate_fold(
            X[train_mask], X[test_mask],
            bl[train_mask], bl[test_mask],
            ctl[train_mask], ctl[test_mask],
        )
        res["hospital"] = hosp
        res["model"] = "LR_d1"
        res["n_train"] = int(n_train)
        res["n_test"] = int(n_test)

        logger.info(f"  S1 AUC={res['s1_auc']:.4f}  Sens={res['s1_sens']:.4f}  Spec={res['s1_spec']:.4f}")
        if not np.isnan(res.get("s2_f1", np.nan)):
            logger.info(f"  S2 F1={res['s2_f1']:.4f}")

        # Save fold checkpoint
        atomic_write_json(fold_ckpt, {"complete": True, "result": res})
        results.append(res)

        # Permutation test
        n_perm = 10 if dry_run else N_PERMUTATIONS
        perm_ckpt = ckpt_dir / f"perm_{hosp}_lr.json"
        logger.info(f"  Running {n_perm} permutations...")
        perm_pvals = permutation_test(
            X[train_mask], X[test_mask],
            bl[train_mask], bl[test_mask],
            ctl[train_mask], ctl[test_mask],
            res, n_perm, perm_ckpt,
        )
        logger.info(f"  p(S1 AUC)={perm_pvals['p_s1_auc']:.4f}  p(S2 F1)={perm_pvals['p_s2_f1']:.4f}")

    return pd.DataFrame(results)


def run_external_validation(
    agg_df: pd.DataFrame,
    feat_cols: List[str],
    hospitals: np.ndarray,
    bl: np.ndarray,
    ctl: np.ndarray,
    out_dir: Path,
) -> pd.DataFrame:
    """Phase 2: Train on all standard data, test on Boramae/Yonsei."""
    core_hospitals = list(HOSPITAL_GROUPS.keys())
    core_mask = np.isin(hospitals, core_hospitals)
    X = agg_df[feat_cols].values

    results = []
    for cohort_name, cohort_info in EXTERNAL_COHORTS.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"External validation: {cohort_name} ({cohort_info['label']})")

        ext_mask = hospitals == cohort_name
        n_ext = ext_mask.sum()
        if n_ext == 0:
            logger.warning(f"  No samples for {cohort_name}, skipping")
            continue

        logger.info(f"  Train (all standard): {core_mask.sum()}, Test: {n_ext}")
        ext_groups = agg_df.loc[ext_mask, "group"].unique()
        logger.info(f"  External groups: {sorted(ext_groups)}")

        # Build external labels
        # Boramae: BPRO -> cancer (maps to prostate), BNOR -> non-cancer
        # Yonsei: YPAN -> cancer (maps to pancreatic), YNOR -> non-cancer
        ext_bl = np.zeros(n_ext, dtype=int)
        ext_ctl = np.full(n_ext, -1, dtype=int)
        ext_grps = agg_df.loc[ext_mask, "group"].values

        ct_map = {c: i for i, c in enumerate(CANCER_TYPES)}
        ext_cancer_map = {
            "BPRO": "PRO",  # Boramae prostate -> PRO
            "YPAN": "PAN",  # Yonsei pancreatic -> PAN
        }
        ext_noncancer = {"BNOR", "YNOR"}

        for i, g in enumerate(ext_grps):
            if g in ext_cancer_map:
                ext_bl[i] = 1
                canonical = ext_cancer_map[g]
                ext_ctl[i] = ct_map.get(canonical, -1)
            elif g in ext_noncancer:
                ext_bl[i] = 0
                ext_ctl[i] = -1

        X_train = X[core_mask]
        X_test = X[ext_mask]

        res = evaluate_fold(
            X_train, X_test,
            bl[core_mask], ext_bl,
            ctl[core_mask], ext_ctl,
        )
        res["hospital"] = cohort_name
        res["model"] = "LR_d1"
        res["n_train"] = int(core_mask.sum())
        res["n_test"] = int(n_ext)
        res["cohort_type"] = "prospective_external"

        logger.info(f"  S1 AUC={res['s1_auc']:.4f}  Sens={res['s1_sens']:.4f}  Spec={res['s1_spec']:.4f}")
        if not np.isnan(res.get("s2_f1", np.nan)):
            logger.info(f"  S2 F1={res['s2_f1']:.4f}")

        results.append(res)

    return pd.DataFrame(results) if results else pd.DataFrame()


# ===========================================================================
# Main
# ===========================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="LOHO Validation Experiment")
    parser.add_argument("--dry-run", action="store_true", help="Run 1 hospital + 10 permutations only")
    args = parser.parse_args()

    # Output directory
    out_dir = RESULTS_DIR / "weekend_experiments" / "loho_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_out_dir = FIG_DIR / "weekend" / "loho_validation"
    fig_out_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(out_dir)
    logger.info("=" * 60)
    logger.info("Experiment 2: Leave-One-Hospital-Out (LOHO) Validation")
    logger.info(f"Output: {out_dir}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info("=" * 60)

    # Save config
    config_out = {
        "experiment": "LOHO Validation",
        "timestamp": datetime.now().isoformat(),
        "dry_run": args.dry_run,
        "n_permutations": 10 if args.dry_run else N_PERMUTATIONS,
        "random_state": RANDOM_STATE,
        "cancer_types": CANCER_TYPES,
        "non_cancer": NON_CANCER,
        "hospitals": {k: v["label"] for k, v in HOSPITAL_GROUPS.items()},
        "external_cohorts": {k: v["label"] for k, v in EXTERNAL_COHORTS.items()},
    }
    atomic_write_json(out_dir / "config.json", config_out)

    # ---- Load data ----
    agg_df, feat_cols, wavenumbers = load_data()

    # ---- Assign hospitals ----
    hospitals = assign_hospitals(agg_df)
    bl, ctl = prepare_labels(agg_df)

    # Log hospital distribution
    for h in sorted(set(hospitals)):
        mask = hospitals == h
        n = mask.sum()
        grps = sorted(agg_df.loc[mask, "group"].unique())
        logger.info(f"  {h}: {n} samples — {grps}")

    unknown_mask = hospitals == "Unknown"
    if unknown_mask.sum() > 0:
        logger.warning(f"  {unknown_mask.sum()} samples not assigned to any hospital")
        unk_grps = agg_df.loc[unknown_mask, "group"].unique()
        logger.warning(f"  Unknown groups: {sorted(unk_grps)}")

    # ---- Phase 1: LOHO on 5 core hospitals ----
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1: Leave-One-Hospital-Out (5 core hospitals)")
    logger.info("=" * 60)
    t_start = time.time()

    loho_df = run_loho(agg_df, feat_cols, hospitals, bl, ctl, out_dir, dry_run=args.dry_run)

    # Flatten per_cancer_sens for CSV
    loho_csv = loho_df.copy()
    loho_csv["per_cancer_sens"] = loho_csv["per_cancer_sens"].apply(
        lambda d: json.dumps(d) if isinstance(d, dict) else str(d)
    )
    loho_csv["cancer_types_in_test"] = loho_csv["cancer_types_in_test"].apply(
        lambda x: ",".join(x) if isinstance(x, list) else str(x)
    )
    loho_csv.to_csv(out_dir / "loho_results.csv", index=False)
    logger.info(f"\nSaved loho_results.csv ({len(loho_df)} rows)")

    # ---- Phase 2: External validation ----
    if not args.dry_run:
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 2: External validation (Boramae, Yonsei)")
        logger.info("=" * 60)

        ext_df = run_external_validation(agg_df, feat_cols, hospitals, bl, ctl, out_dir)
        if not ext_df.empty:
            ext_csv = ext_df.copy()
            for col in ["per_cancer_sens", "cancer_types_in_test"]:
                if col in ext_csv.columns:
                    ext_csv[col] = ext_csv[col].apply(
                        lambda x: json.dumps(x) if isinstance(x, (dict, list)) else str(x)
                    )
            ext_csv.to_csv(out_dir / "external_validation.csv", index=False)
            logger.info(f"Saved external_validation.csv ({len(ext_df)} rows)")
    else:
        ext_df = pd.DataFrame()

    # ---- Phase 3: 5-Fold CV baseline ----
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3: Standard 5-Fold CV baseline")
    logger.info("=" * 60)

    core_hospitals = list(HOSPITAL_GROUPS.keys())
    core_mask = np.isin(hospitals, core_hospitals)
    X_core = agg_df.loc[core_mask, feat_cols].values

    cv_baseline = run_cv_baseline(X_core, bl[core_mask], ctl[core_mask])
    logger.info(f"  CV S1 AUC: {cv_baseline['cv_s1_auc']:.4f} +/- {cv_baseline['cv_s1_auc_std']:.4f}")
    logger.info(f"  CV S1 Sens: {cv_baseline['cv_s1_sens']:.4f}")
    logger.info(f"  CV S1 Spec: {cv_baseline['cv_s1_spec']:.4f}")
    if not np.isnan(cv_baseline["cv_s2_f1"]):
        logger.info(f"  CV S2 F1: {cv_baseline['cv_s2_f1']:.4f} +/- {cv_baseline['cv_s2_f1_std']:.4f}")

    # ---- Generalization gap ----
    gap_rows = []
    for _, row in loho_df.iterrows():
        gap_row = {
            "hospital": row["hospital"],
            "s1_auc": row["s1_auc"],
            "s1_sens": row["s1_sens"],
            "s1_spec": row["s1_spec"],
            "s2_f1": row.get("s2_f1", np.nan),
            "cv_s1_auc": cv_baseline["cv_s1_auc"],
            "cv_s1_sens": cv_baseline["cv_s1_sens"],
            "cv_s1_spec": cv_baseline["cv_s1_spec"],
            "cv_s2_f1": cv_baseline["cv_s2_f1"],
            "gap_s1_auc": cv_baseline["cv_s1_auc"] - row["s1_auc"],
            "gap_s1_sens": cv_baseline["cv_s1_sens"] - row["s1_sens"],
            "gap_s1_spec": cv_baseline["cv_s1_spec"] - row["s1_spec"],
        }
        if not np.isnan(row.get("s2_f1", np.nan)) and not np.isnan(cv_baseline["cv_s2_f1"]):
            gap_row["gap_s2_f1"] = cv_baseline["cv_s2_f1"] - row["s2_f1"]
        else:
            gap_row["gap_s2_f1"] = np.nan
        gap_rows.append(gap_row)

    gap_df = pd.DataFrame(gap_rows)
    gap_df.to_csv(out_dir / "generalization_gap.csv", index=False)
    logger.info(f"Saved generalization_gap.csv")

    # ---- Permutation p-values summary ----
    perm_rows = []
    ckpt_dir = out_dir / "checkpoints"
    for hosp in (list(HOSPITAL_GROUPS.keys()) if not args.dry_run else list(HOSPITAL_GROUPS.keys())[:1]):
        perm_path = ckpt_dir / f"perm_{hosp}_lr.json"
        if perm_path.exists():
            perm_data = json.loads(perm_path.read_text())
            # Find observed values from loho_df
            obs_row = loho_df[loho_df["hospital"] == hosp]
            if len(obs_row) > 0:
                obs_auc = obs_row.iloc[0]["s1_auc"]
                obs_f1 = obs_row.iloc[0].get("s2_f1", np.nan)
                n_perm = perm_data.get("completed", 0)

                perm_aucs = np.array(perm_data.get("perm_s1_aucs", []))
                p_auc = (np.sum(perm_aucs >= obs_auc) + 1) / (n_perm + 1) if n_perm > 0 else np.nan

                perm_f1s = np.array(perm_data.get("perm_s2_f1s", []))
                p_f1 = np.nan
                if len(perm_f1s) > 0 and not np.isnan(obs_f1):
                    p_f1 = (np.sum(perm_f1s >= obs_f1) + 1) / (n_perm + 1)

                perm_rows.append({
                    "hospital": hosp,
                    "n_permutations": n_perm,
                    "observed_s1_auc": obs_auc,
                    "p_s1_auc": p_auc,
                    "observed_s2_f1": obs_f1,
                    "p_s2_f1": p_f1,
                })

    if perm_rows:
        perm_df = pd.DataFrame(perm_rows)
        perm_df.to_csv(out_dir / "permutation_pvalues.csv", index=False)
        logger.info(f"Saved permutation_pvalues.csv")

    # ---- Plots ----
    logger.info("\nGenerating figures...")
    if len(loho_df) > 0:
        plot_hospital_performance(loho_df, fig_out_dir)
        if len(gap_df) > 0:
            plot_generalization_gap(gap_df, fig_out_dir)

    # ---- Summary ----
    total_time = time.time() - t_start
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info(f"Total time: {total_time / 60:.1f} min")
    logger.info(f"Output directory: {out_dir}")
    logger.info("=" * 60)

    # Print summary table
    if len(loho_df) > 0:
        logger.info("\nLOHO Results Summary:")
        logger.info(f"{'Hospital':<12} {'S1 AUC':>8} {'Sens':>8} {'Spec':>8} {'S2 F1':>8}")
        logger.info("-" * 48)
        for _, row in loho_df.iterrows():
            f1_str = f"{row['s2_f1']:.4f}" if not np.isnan(row.get('s2_f1', np.nan)) else "  N/A"
            logger.info(
                f"{row['hospital']:<12} {row['s1_auc']:>8.4f} {row['s1_sens']:>8.4f} "
                f"{row['s1_spec']:>8.4f} {f1_str:>8}"
            )

        logger.info(f"\n5-Fold CV baseline: AUC={cv_baseline['cv_s1_auc']:.4f}, "
                     f"Sens={cv_baseline['cv_s1_sens']:.4f}, "
                     f"Spec={cv_baseline['cv_s1_spec']:.4f}")


if __name__ == "__main__":
    main()
