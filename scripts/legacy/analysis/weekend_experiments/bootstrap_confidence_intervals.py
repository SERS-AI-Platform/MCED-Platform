#!/usr/bin/env python3
"""
Bootstrap Statistical Validation — Experiment 1

4 model configs × 2,000 bootstrap resamples × 5-fold CV
→ Publication-ready confidence intervals for AUC, F1, Sensitivity, Specificity,
  and per-cancer metrics.

Model configs:
  (a) lr_d1        — LR on 1st derivative features (935 dims)
  (b) lr_raw       — LR on raw spectrum (935 dims)
  (c) clinical_3view — LR on 3-view spectral (raw+d1+d2) + peak features + clinical
  (d) lr_d1d2      — LR on d1+d2 concat (1870 dims)

Usage:
    python scripts/analysis/weekend_experiments/bootstrap_confidence_intervals.py
    python scripts/analysis/weekend_experiments/bootstrap_confidence_intervals.py --dry-run
    python scripts/analysis/weekend_experiments/bootstrap_confidence_intervals.py --n-resamples 500
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import warnings

from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from scipy.special import voigt_profile
from scipy.integrate import trapezoid
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.sers.config import RESULTS_DIR, FIG_DIR  # noqa: E402
from models.clinical_utils import load_clinical, merge_clinical_features  # noqa: E402

logger = logging.getLogger("bootstrap_ci")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CANCER_TYPES = ["PRO", "BRE", "OVA", "LUN", "CRC", "PAN", "BLC"]
NON_CANCER = ["NOR", "DIA", "HBP", "H.D."]
GROUP_ALIASES = {"PAN": ["CPAN", "YPAN"], "NOR": ["YNOR"]}

KNOWN_PEAKS = [
    (448.1, "ring_deform", 15), (538.7, "SS_stretch", 15),
    (617.7, "CS_stretch", 15), (683.3, "creatinine", 15),
    (723.8, "adenine", 15), (795.1, "hippuric", 15),
    (849.1, "tyrosine", 15), (895.4, "uric_acid", 15),
    (933.9, "creatinine2", 15), (999.5, "phe_urea", 15),
    (1147.9, "uric_CN", 15), (1230.8, "amide_III", 20),
    (1292.5, "CH2_twist", 15), (1352.3, "trp_fermi", 15),
    (1448.7, "CH2_deform", 20), (1597.1, "purine_CC", 20),
    (1651.1, "amide_I", 20),
]

MODEL_CONFIGS = ["lr_d1", "lr_raw", "clinical_3view", "lr_d1d2"]

N_SPLITS = 5
DEFAULT_N_RESAMPLES = 2000
CI_ALPHA = 0.05  # 95 % CI


# ============================================================================
# Feature extraction
# ============================================================================

def extract_1st_derivative(X: np.ndarray) -> np.ndarray:
    return np.apply_along_axis(
        lambda y: savgol_filter(y, window_length=11, polyorder=3, deriv=1),
        axis=1, arr=X,
    )


def extract_2nd_derivative(X: np.ndarray) -> np.ndarray:
    return np.apply_along_axis(
        lambda y: savgol_filter(y, window_length=11, polyorder=3, deriv=2),
        axis=1, arr=X,
    )


def voigt_func(x, amplitude, center, sigma, gamma):
    return amplitude * voigt_profile(x - center, sigma, gamma)


def extract_peak_features(X: np.ndarray, wavenumbers: np.ndarray) -> np.ndarray:
    """Extract peak parameters via Voigt fitting for each KNOWN_PEAK."""
    n_samples, n_peaks = len(X), len(KNOWN_PEAKS)
    peak_areas = np.zeros((n_samples, n_peaks))
    peak_heights = np.zeros((n_samples, n_peaks))
    peak_fwhms = np.zeros((n_samples, n_peaks))
    peak_shifts = np.zeros((n_samples, n_peaks))

    for pi, (center, name, half_w) in enumerate(KNOWN_PEAKS):
        mask = (wavenumbers >= center - half_w * 1.5) & (
            wavenumbers <= center + half_w * 1.5
        )
        if mask.sum() < 5:
            continue
        x_region = wavenumbers[mask]
        for si in range(n_samples):
            y_shifted = X[si, mask] - np.min(X[si, mask])
            try:
                amp_guess = max(np.max(y_shifted), 1e-6)
                popt, _ = curve_fit(
                    voigt_func, x_region, y_shifted,
                    p0=[amp_guess, center, half_w / 3, half_w / 3],
                    bounds=(
                        [0, center - half_w, 0.1, 0.1],
                        [amp_guess * 10, center + half_w, half_w * 2, half_w * 2],
                    ),
                    maxfev=2000,
                )
                amp, ctr, sigma, gamma = popt
                y_fit = voigt_func(x_region, *popt)
                area = trapezoid(y_fit, x_region)
                fL = 2 * gamma
                fG = 2 * sigma * np.sqrt(2 * np.log(2))
                fwhm = 0.5346 * fL + np.sqrt(0.2166 * fL**2 + fG**2)
            except (RuntimeError, ValueError):
                amp, ctr, area, fwhm = 0.0, center, 0.0, 0.0
            peak_areas[si, pi] = area
            peak_heights[si, pi] = amp
            peak_fwhms[si, pi] = fwhm
            peak_shifts[si, pi] = ctr - center

    features = [peak_areas, peak_heights, peak_fwhms, peak_shifts]
    pn2i = {p[1]: i for i, p in enumerate(KNOWN_PEAKS)}
    ratios = [
        ("phe_urea", "adenine"),
        ("phe_urea", "creatinine"),
        ("hippuric", "creatinine"),
        ("CS_stretch", "creatinine"),
        ("amide_I", "CH2_deform"),
        ("adenine", "purine_CC"),
        ("tyrosine", "phe_urea"),
    ]
    for na, nb in ratios:
        features.append(
            peak_areas[:, pn2i[na] : pn2i[na] + 1]
            / (peak_areas[:, pn2i[nb] : pn2i[nb] + 1] + 1e-10)
        )
    return np.nan_to_num(np.hstack(features), nan=0.0, posinf=0.0, neginf=0.0)


# ============================================================================
# Data loading
# ============================================================================

def load_data():
    """Load and aggregate spectra, return X_raw, wavenumbers, groups, sample_ids, agg_df."""
    spec_df = pd.read_csv(RESULTS_DIR / "processed_spectra.csv")
    feat_cols = [c for c in spec_df.columns if c.startswith("x_")]
    wavenumbers = np.array([float(c.replace("x_", "")) for c in feat_cols])

    # Mean aggregation per sample
    agg_df = spec_df.groupby(["group", "sample_id"], as_index=False).agg(
        {**{c: "mean" for c in feat_cols}, "replicate": "count"}
    )

    # Resolve aliases
    def resolve(g):
        for alias, members in GROUP_ALIASES.items():
            if g in members:
                return alias
        return g

    agg_df["group"] = agg_df["group"].map(resolve)

    # Filter to valid groups
    valid = set(CANCER_TYPES) | set(NON_CANCER)
    agg_df = agg_df[agg_df["group"].isin(valid)].reset_index(drop=True)

    X_raw = agg_df[feat_cols].values
    groups = agg_df["group"].values
    sample_ids = agg_df["sample_id"].values

    logger.info(
        f"Data loaded: {len(agg_df)} subjects, {len(feat_cols)} features, "
        f"groups={np.unique(groups).tolist()}"
    )
    return X_raw, wavenumbers, groups, sample_ids, agg_df, feat_cols


def build_feature_matrices(X_raw, wavenumbers, agg_df, feat_cols):
    """Build feature matrices for all 4 model configs.

    Returns dict: config_name -> X matrix.
    """
    logger.info("Building feature matrices...")

    X_d1 = extract_1st_derivative(X_raw)
    X_d2 = extract_2nd_derivative(X_raw)

    # lr_raw
    features = {"lr_raw": X_raw.copy()}

    # lr_d1
    features["lr_d1"] = X_d1.copy()

    # lr_d1d2
    features["lr_d1d2"] = np.hstack([X_d1, X_d2])

    # clinical_3view: raw+d1+d2 + peak features + clinical
    X_3view = np.hstack([X_raw, X_d1, X_d2])
    peak_feat = extract_peak_features(X_raw, wavenumbers)
    clin = load_clinical(PROJECT_ROOT)
    clinical_cols = ["age", "sex_numeric", "bmi"]
    clinical_data = merge_clinical_features(agg_df, clin, clinical_cols, PROJECT_ROOT)
    X_clinical_3view = np.hstack([X_3view, peak_feat, clinical_data])
    features["clinical_3view"] = X_clinical_3view

    for name, X in features.items():
        logger.info(f"  {name}: {X.shape}")

    return features


# ============================================================================
# Two-stage CV evaluation
# ============================================================================

def two_stage_cv(
    X: np.ndarray,
    groups: np.ndarray,
    sample_ids: np.ndarray,
    n_splits: int = N_SPLITS,
    random_state: int = 42,
) -> dict:
    """Run 2-stage (binary + multiclass) CV and return metrics dict."""
    cancer_set = set(CANCER_TYPES)
    bl = np.array([1 if g in cancer_set else 0 for g in groups])
    ctl = np.array(
        [CANCER_TYPES.index(g) if g in cancer_set else -1 for g in groups]
    )

    skf = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )

    all_s1_true, all_s1_prob = [], []
    all_s2_true, all_s2_pred = [], []

    for tr, val in skf.split(X, bl, groups=sample_ids):
        # Stage 1: Binary cancer vs. non-cancer
        m1 = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0, max_iter=1000, solver="saga",
                class_weight="balanced", random_state=42,
            ),
        )
        m1.fit(X[tr], bl[tr])
        prob = m1.predict_proba(X[val])[:, 1]
        all_s1_true.extend(bl[val])
        all_s1_prob.extend(prob)

        # Stage 2: Cancer-type classification
        cm_tr = ctl[tr] >= 0
        cm_val = ctl[val] >= 0
        if cm_tr.sum() > 10 and cm_val.sum() > 0:
            m2 = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=1.0, max_iter=1000, solver="saga",
                    class_weight="balanced", multi_class="multinomial",
                    random_state=42,
                ),
            )
            m2.fit(X[tr][cm_tr], ctl[tr][cm_tr])
            pred = m2.predict(X[val][cm_val])
            all_s2_true.extend(ctl[val][cm_val])
            all_s2_pred.extend(pred)

    # Compute metrics
    s1_true = np.array(all_s1_true)
    s1_prob = np.array(all_s1_prob)
    s1_pred = (s1_prob > 0.5).astype(int)
    s2_true = np.array(all_s2_true)
    s2_pred = np.array(all_s2_pred)

    tn, fp, fn, tp = confusion_matrix(s1_true, s1_pred).ravel()

    per_cancer = {}
    for i, ct in enumerate(CANCER_TYPES):
        n_true = np.sum(s2_true == i)
        if n_true > 0:
            per_cancer[ct] = float(
                np.sum((s2_true == i) & (s2_pred == i)) / n_true
            )
        else:
            per_cancer[ct] = np.nan

    return {
        "s1_auc": float(roc_auc_score(s1_true, s1_prob)),
        "s1_sens": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "s1_spec": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "s1_f1": float(f1_score(s1_true, s1_pred, zero_division=0)),
        "s2_f1": float(
            f1_score(s2_true, s2_pred, average="macro", zero_division=0)
        ),
        "per_cancer": per_cancer,
    }


# ============================================================================
# Bootstrap helpers
# ============================================================================

def resample_by_sample_id(
    X: np.ndarray,
    groups: np.ndarray,
    sample_ids: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample unique sample_ids with replacement, rebuild dataset."""
    unique_ids = np.unique(sample_ids)
    chosen = rng.choice(unique_ids, size=len(unique_ids), replace=True)

    # Map each original sample_id to its indices
    id_to_idx = {}
    for i, sid in enumerate(sample_ids):
        id_to_idx.setdefault(sid, []).append(i)

    indices = []
    new_sample_ids = []
    for boot_idx, sid in enumerate(chosen):
        orig_indices = id_to_idx[sid]
        indices.extend(orig_indices)
        # Give resampled duplicates unique IDs to avoid grouping issues in CV
        new_sample_ids.extend([f"{sid}__boot{boot_idx}"] * len(orig_indices))

    indices = np.array(indices)
    return X[indices], groups[indices], np.array(new_sample_ids)


def save_checkpoint(path: Path, data: dict):
    """Atomic write: write to temp then rename."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_completed_indices(ckpt_dir: Path, config_name: str) -> dict[int, dict]:
    """Scan checkpoint directory for completed resamples."""
    completed = {}
    if not ckpt_dir.exists():
        return completed
    prefix = f"{config_name}_"
    for f in ckpt_dir.iterdir():
        if f.name.startswith(prefix) and f.suffix == ".json":
            try:
                idx = int(f.stem.split("_")[-1])
                data = json.loads(f.read_text())
                completed[idx] = data
            except (ValueError, json.JSONDecodeError):
                continue
    return completed


def bca_ci(
    stat_samples: np.ndarray,
    stat_original: float,
    jackknife_stats: np.ndarray,
    alpha: float = CI_ALPHA,
) -> tuple[float, float]:
    """Bias-corrected and accelerated bootstrap CI.

    Parameters
    ----------
    stat_samples : array of bootstrap statistics (B,)
    stat_original : the statistic computed on the full dataset
    jackknife_stats : leave-one-out statistics (N,)
    alpha : significance level (default 0.05 for 95% CI)
    """
    from scipy.stats import norm

    B = len(stat_samples)
    if B == 0:
        return np.nan, np.nan

    # Bias correction factor z0
    z0 = norm.ppf(np.mean(stat_samples < stat_original))
    if np.isinf(z0):
        # Fallback to percentile CI
        lo = np.percentile(stat_samples, 100 * alpha / 2)
        hi = np.percentile(stat_samples, 100 * (1 - alpha / 2))
        return float(lo), float(hi)

    # Acceleration factor a
    theta_bar = np.mean(jackknife_stats)
    diffs = theta_bar - jackknife_stats
    num = np.sum(diffs**3)
    den = 6.0 * (np.sum(diffs**2)) ** 1.5
    a = num / den if den != 0 else 0.0

    z_lo = norm.ppf(alpha / 2)
    z_hi = norm.ppf(1 - alpha / 2)

    def adj(z):
        numer = z0 + z
        return norm.cdf(z0 + numer / (1 - a * numer))

    p_lo = adj(z_lo)
    p_hi = adj(z_hi)

    # Clamp to valid range
    p_lo = np.clip(p_lo, 0.0, 1.0)
    p_hi = np.clip(p_hi, 0.0, 1.0)

    lo = np.percentile(stat_samples, 100 * p_lo)
    hi = np.percentile(stat_samples, 100 * p_hi)
    return float(lo), float(hi)


def compute_jackknife_stats(
    X: np.ndarray,
    groups: np.ndarray,
    sample_ids: np.ndarray,
    metric_key: str,
    cancer_type: str | None = None,
) -> np.ndarray:
    """Leave-one-sample-out jackknife estimates for a given metric."""
    unique_ids = np.unique(sample_ids)
    n = len(unique_ids)
    stats = np.full(n, np.nan)

    for i, drop_id in enumerate(unique_ids):
        mask = sample_ids != drop_id
        if mask.sum() < 20:
            continue
        try:
            result = two_stage_cv(X[mask], groups[mask], sample_ids[mask])
            if cancer_type is not None:
                stats[i] = result["per_cancer"].get(cancer_type, np.nan)
            else:
                stats[i] = result.get(metric_key, np.nan)
        except Exception:
            continue

    # Drop NaN entries
    return stats[~np.isnan(stats)]


# ============================================================================
# Single bootstrap resample worker
# ============================================================================

def run_single_resample(
    idx: int,
    X: np.ndarray,
    groups: np.ndarray,
    sample_ids: np.ndarray,
    seed: int,
) -> dict:
    """Run one bootstrap resample: resample + full CV."""
    rng = np.random.default_rng(seed)
    X_b, g_b, sid_b = resample_by_sample_id(X, groups, sample_ids, rng)

    # Use a different random_state per resample for CV splitting
    result = two_stage_cv(X_b, g_b, sid_b, random_state=seed % (2**31))
    result["resample_idx"] = idx
    result["seed"] = seed
    return result


# ============================================================================
# Main bootstrap loop for one config
# ============================================================================

def run_bootstrap_config(
    config_name: str,
    X: np.ndarray,
    groups: np.ndarray,
    sample_ids: np.ndarray,
    n_resamples: int,
    output_dir: Path,
    n_jobs: int = 2,
    base_seed: int = 12345,
):
    """Run bootstrap for one model config with checkpointing."""
    from joblib import Parallel, delayed

    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Load completed resamples
    completed = load_completed_indices(ckpt_dir, config_name)
    logger.info(
        f"[{config_name}] {len(completed)}/{n_resamples} resamples already completed"
    )

    remaining = [i for i in range(n_resamples) if i not in completed]
    if not remaining:
        logger.info(f"[{config_name}] All resamples completed, loading results")
        all_results = [completed[i] for i in range(n_resamples)]
        return all_results

    t0 = time.time()
    batch_size = max(1, min(50, len(remaining)))

    for batch_start in range(0, len(remaining), batch_size):
        batch = remaining[batch_start : batch_start + batch_size]

        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(run_single_resample)(
                idx, X, groups, sample_ids, base_seed + idx
            )
            for idx in batch
        )

        # Save checkpoints
        for res in results:
            idx = res["resample_idx"]
            ckpt_path = ckpt_dir / f"{config_name}_{idx:04d}.json"
            save_checkpoint(ckpt_path, res)
            completed[idx] = res

        done = len(completed)
        elapsed = time.time() - t0
        rate = (done - (len(completed) - len(results))) / max(elapsed, 0.01)
        remaining_count = n_resamples - done
        eta = remaining_count / rate if rate > 0 else 0

        logger.info(
            f"[{config_name}] {done}/{n_resamples} done | "
            f"elapsed {elapsed:.0f}s | ETA {eta:.0f}s"
        )

    all_results = [completed[i] for i in range(n_resamples)]
    logger.info(
        f"[{config_name}] Completed all {n_resamples} resamples in "
        f"{time.time() - t0:.1f}s"
    )
    return all_results


# ============================================================================
# Aggregate results and compute CIs
# ============================================================================

def aggregate_results(
    config_name: str,
    results: list[dict],
    X: np.ndarray,
    groups: np.ndarray,
    sample_ids: np.ndarray,
    output_dir: Path,
    compute_bca: bool = True,
) -> pd.DataFrame:
    """Compute CIs for all metrics of one model config.

    Returns a DataFrame with columns:
        config, metric, mean, median, ci_lower, ci_upper, std
    """
    # Original (non-bootstrapped) metrics
    logger.info(f"[{config_name}] Computing original metrics on full dataset...")
    orig = two_stage_cv(X, groups, sample_ids)

    scalar_metrics = ["s1_auc", "s1_sens", "s1_spec", "s1_f1", "s2_f1"]
    rows = []

    for metric in scalar_metrics:
        vals = np.array([r[metric] for r in results])
        stat_orig = orig[metric]

        if compute_bca and len(vals) >= 100:
            logger.info(f"[{config_name}] Computing jackknife for BCa CI: {metric}")
            jk = compute_jackknife_stats(X, groups, sample_ids, metric)
            ci_lo, ci_hi = bca_ci(vals, stat_orig, jk)
        else:
            ci_lo = float(np.percentile(vals, 100 * CI_ALPHA / 2))
            ci_hi = float(np.percentile(vals, 100 * (1 - CI_ALPHA / 2)))

        rows.append({
            "config": config_name,
            "metric": metric,
            "original": stat_orig,
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "std": float(np.std(vals)),
        })

    # Per-cancer sensitivity
    for ct in CANCER_TYPES:
        vals = np.array([r["per_cancer"].get(ct, np.nan) for r in results])
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            continue
        stat_orig = orig["per_cancer"].get(ct, np.nan)

        if compute_bca and len(vals) >= 100 and not np.isnan(stat_orig):
            logger.info(
                f"[{config_name}] Computing jackknife for BCa CI: {ct}_sens"
            )
            jk = compute_jackknife_stats(
                X, groups, sample_ids, "", cancer_type=ct
            )
            ci_lo, ci_hi = bca_ci(vals, stat_orig, jk)
        else:
            ci_lo = float(np.percentile(vals, 100 * CI_ALPHA / 2))
            ci_hi = float(np.percentile(vals, 100 * (1 - CI_ALPHA / 2)))

        rows.append({
            "config": config_name,
            "metric": f"{ct}_sens",
            "original": stat_orig,
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "std": float(np.std(vals)),
        })

    # Save raw bootstrap results
    raw_rows = []
    for r in results:
        row = {
            "resample_idx": r["resample_idx"],
            "s1_auc": r["s1_auc"],
            "s1_sens": r["s1_sens"],
            "s1_spec": r["s1_spec"],
            "s1_f1": r["s1_f1"],
            "s2_f1": r["s2_f1"],
        }
        for ct in CANCER_TYPES:
            row[f"{ct}_sens"] = r["per_cancer"].get(ct, np.nan)
        raw_rows.append(row)

    raw_df = pd.DataFrame(raw_rows)
    raw_path = output_dir / f"bootstrap_results_{config_name}.csv"
    raw_df.to_csv(raw_path, index=False)
    logger.info(f"  Saved raw results → {raw_path}")

    return pd.DataFrame(rows)


# ============================================================================
# Visualization
# ============================================================================

def plot_violin(ci_df: pd.DataFrame, all_raw: dict, output_dir: Path):
    """Violin plot comparing all configs across main metrics."""
    metrics = ["s1_auc", "s1_sens", "s1_spec", "s1_f1", "s2_f1"]
    metric_labels = {
        "s1_auc": "Binary AUC",
        "s1_sens": "Binary Sensitivity",
        "s1_spec": "Binary Specificity",
        "s1_f1": "Binary F1",
        "s2_f1": "Multiclass Macro F1",
    }

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63"]

    for ax, metric in zip(axes, metrics):
        data = []
        labels = []
        for ci, config in enumerate(MODEL_CONFIGS):
            if config in all_raw:
                vals = [r[metric] for r in all_raw[config]]
                data.append(vals)
                labels.append(config)

        if not data:
            continue

        parts = ax.violinplot(data, showmeans=True, showmedians=True)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(colors[i % len(colors)])
            pc.set_alpha(0.7)
        parts["cmeans"].set_color("black")
        parts["cmedians"].set_color("red")

        # Add CI markers from ci_df
        for i, config in enumerate(labels):
            row = ci_df[(ci_df["config"] == config) & (ci_df["metric"] == metric)]
            if len(row) == 1:
                lo = row["ci_lower"].values[0]
                hi = row["ci_upper"].values[0]
                ax.plot([i + 1, i + 1], [lo, hi], "k-", linewidth=2, zorder=5)
                ax.plot(i + 1, lo, "k_", markersize=8, zorder=5)
                ax.plot(i + 1, hi, "k_", markersize=8, zorder=5)

        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_title(metric_labels.get(metric, metric), fontsize=11, fontweight="bold")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    fig.suptitle(
        "Bootstrap 95% CI — Model Comparison", fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    path = output_dir / "violin_plots.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved violin plot → {path}")


def plot_per_cancer_ci(ci_df: pd.DataFrame, output_dir: Path):
    """Per-cancer sensitivity CI plot for all configs."""
    cancer_metrics = [f"{ct}_sens" for ct in CANCER_TYPES]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63"]
    offsets = np.linspace(-0.15, 0.15, len(MODEL_CONFIGS))

    for ci, config in enumerate(MODEL_CONFIGS):
        subset = ci_df[
            (ci_df["config"] == config) & (ci_df["metric"].isin(cancer_metrics))
        ]
        if subset.empty:
            continue

        x_pos = []
        means = []
        ci_lo = []
        ci_hi = []

        for j, ct in enumerate(CANCER_TYPES):
            row = subset[subset["metric"] == f"{ct}_sens"]
            if len(row) == 1:
                x_pos.append(j + offsets[ci])
                means.append(row["mean"].values[0])
                ci_lo.append(row["mean"].values[0] - row["ci_lower"].values[0])
                ci_hi.append(row["ci_upper"].values[0] - row["mean"].values[0])

        if x_pos:
            ax.errorbar(
                x_pos, means,
                yerr=[ci_lo, ci_hi],
                fmt="o", capsize=4, capthick=1.5, linewidth=1.5,
                color=colors[ci], label=config, markersize=6,
            )

    ax.set_xticks(range(len(CANCER_TYPES)))
    ax.set_xticklabels(CANCER_TYPES, fontsize=11)
    ax.set_ylabel("Sensitivity", fontsize=12)
    ax.set_title(
        "Per-Cancer Sensitivity — 95% Bootstrap CI", fontsize=14, fontweight="bold"
    )
    ax.legend(loc="lower left", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = output_dir / "per_cancer_ci.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved per-cancer CI plot → {path}")


# ============================================================================
# LaTeX table generation
# ============================================================================

def generate_latex_table(ci_df: pd.DataFrame, output_path: Path):
    """Generate a publication-ready LaTeX table."""
    scalar_metrics = ["s1_auc", "s1_sens", "s1_spec", "s1_f1", "s2_f1"]
    metric_labels = {
        "s1_auc": "Binary AUC",
        "s1_sens": "Sensitivity",
        "s1_spec": "Specificity",
        "s1_f1": "Binary F1",
        "s2_f1": "Macro F1 (7-class)",
    }

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Bootstrap 95\% confidence intervals (2{,}000 resamples, 5-fold CV)}")
    lines.append(r"\label{tab:bootstrap_ci}")
    n_configs = len(MODEL_CONFIGS)
    col_spec = "l" + "c" * n_configs
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    header = "Metric & " + " & ".join(
        [r"\textbf{" + c.replace("_", r"\_") + "}" for c in MODEL_CONFIGS]
    ) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for metric in scalar_metrics:
        label = metric_labels.get(metric, metric)
        cells = [label]
        for config in MODEL_CONFIGS:
            row = ci_df[(ci_df["config"] == config) & (ci_df["metric"] == metric)]
            if len(row) == 1:
                m = row["mean"].values[0]
                lo = row["ci_lower"].values[0]
                hi = row["ci_upper"].values[0]
                cells.append(f"{m:.3f} [{lo:.3f}, {hi:.3f}]")
            else:
                cells.append("---")
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\midrule")

    # Per-cancer rows
    for ct in CANCER_TYPES:
        metric = f"{ct}_sens"
        cells = [f"{ct} Sens."]
        for config in MODEL_CONFIGS:
            row = ci_df[(ci_df["config"] == config) & (ci_df["metric"] == metric)]
            if len(row) == 1:
                m = row["mean"].values[0]
                lo = row["ci_lower"].values[0]
                hi = row["ci_upper"].values[0]
                cells.append(f"{m:.3f} [{lo:.3f}, {hi:.3f}]")
            else:
                cells.append("---")
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    output_path.write_text("\n".join(lines))
    logger.info(f"Saved LaTeX table → {output_path}")


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Bootstrap Statistical Validation — Experiment 1"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run 2 resamples per config only, then exit",
    )
    parser.add_argument(
        "--n-resamples", type=int, default=DEFAULT_N_RESAMPLES,
        help=f"Number of bootstrap resamples (default {DEFAULT_N_RESAMPLES})",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=2,
        help="joblib parallelism for resamples (default 2)",
    )
    parser.add_argument(
        "--configs", nargs="+", default=MODEL_CONFIGS,
        choices=MODEL_CONFIGS,
        help="Model configs to run (default: all 4)",
    )
    parser.add_argument(
        "--skip-bca", action="store_true",
        help="Skip BCa (use percentile CI instead, much faster)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: results/bootstrap_ci/YYYYMMDD_HHMMSS)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = RESULTS_DIR / "bootstrap_ci" / ts
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_output_dir = FIG_DIR / "weekend" / "bootstrap_ci"
    fig_output_dir.mkdir(parents=True, exist_ok=True)

    # Logging setup
    log_fmt = "%(asctime)s | %(levelname)-7s | %(message)s"
    log_datefmt = "%Y-%m-%d %H:%M:%S"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(output_dir / "experiment.log"),
    ]
    logging.basicConfig(level=logging.INFO, format=log_fmt, datefmt=log_datefmt, handlers=handlers)

    n_resamples = 2 if args.dry_run else args.n_resamples
    compute_bca = not args.skip_bca and not args.dry_run

    logger.info("=" * 70)
    logger.info("Bootstrap Statistical Validation — Experiment 1")
    logger.info("=" * 70)
    logger.info(f"  Configs:     {args.configs}")
    logger.info(f"  Resamples:   {n_resamples}")
    logger.info(f"  CV folds:    {N_SPLITS}")
    logger.info(f"  BCa CI:      {compute_bca}")
    logger.info(f"  n_jobs:      {args.n_jobs}")
    logger.info(f"  Output:      {output_dir}")
    logger.info(f"  Dry run:     {args.dry_run}")
    logger.info("")

    # Save experiment config
    exp_config = {
        "experiment": "bootstrap_confidence_intervals",
        "date": datetime.now().isoformat(),
        "n_resamples": n_resamples,
        "n_splits": N_SPLITS,
        "ci_alpha": CI_ALPHA,
        "compute_bca": compute_bca,
        "n_jobs": args.n_jobs,
        "configs": args.configs,
        "dry_run": args.dry_run,
        "cancer_types": CANCER_TYPES,
        "non_cancer": NON_CANCER,
        "base_seed": 12345,
    }
    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(exp_config, indent=2))

    # Load data
    t_start = time.time()
    X_raw, wavenumbers, groups, sample_ids, agg_df, feat_cols = load_data()
    feature_matrices = build_feature_matrices(X_raw, wavenumbers, agg_df, feat_cols)

    logger.info(f"Data loading + feature extraction: {time.time() - t_start:.1f}s")
    logger.info("")

    # Run bootstrap for each config
    all_raw_results = {}
    all_ci_dfs = []

    for config_name in args.configs:
        logger.info(f"{'=' * 50}")
        logger.info(f"Config: {config_name}")
        logger.info(f"{'=' * 50}")

        X = feature_matrices[config_name]

        results = run_bootstrap_config(
            config_name=config_name,
            X=X,
            groups=groups,
            sample_ids=sample_ids,
            n_resamples=n_resamples,
            output_dir=output_dir,
            n_jobs=args.n_jobs,
        )
        all_raw_results[config_name] = results

        ci_df = aggregate_results(
            config_name=config_name,
            results=results,
            X=X,
            groups=groups,
            sample_ids=sample_ids,
            output_dir=output_dir,
            compute_bca=compute_bca,
        )
        all_ci_dfs.append(ci_df)
        logger.info("")

    # Combine all CI results
    ci_all = pd.concat(all_ci_dfs, ignore_index=True)
    ci_path = output_dir / "confidence_intervals.csv"
    ci_all.to_csv(ci_path, index=False)
    logger.info(f"Saved combined CI → {ci_path}")

    # LaTeX table
    generate_latex_table(ci_all, output_dir / "confidence_intervals.tex")

    # Plots
    plot_violin(ci_all, all_raw_results, fig_output_dir)
    plot_per_cancer_ci(ci_all, fig_output_dir)

    # Summary
    elapsed = time.time() - t_start
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"Experiment complete in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    logger.info("=" * 70)
    logger.info(f"Output directory: {output_dir}")
    logger.info("Files:")
    for f in sorted(output_dir.iterdir()):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            logger.info(f"  {f.name:45s} {size_kb:8.1f} KB")

    # Print summary table
    logger.info("")
    logger.info("Summary of 95% CIs:")
    logger.info(f"{'Config':<18s} {'Metric':<14s} {'Mean':>7s} {'CI Lower':>9s} {'CI Upper':>9s}")
    logger.info("-" * 60)
    for _, row in ci_all.iterrows():
        logger.info(
            f"{row['config']:<18s} {row['metric']:<14s} "
            f"{row['mean']:7.4f} {row['ci_lower']:9.4f} {row['ci_upper']:9.4f}"
        )


if __name__ == "__main__":
    main()
