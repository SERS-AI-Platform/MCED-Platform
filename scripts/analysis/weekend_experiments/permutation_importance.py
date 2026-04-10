"""
Experiment 4: Permutation Feature Importance for SERS Cancer Detection

935 derivative features x 1,000 permutations each -> importance ranking
with FDR-corrected significance + wavenumber mapping.

Approach:
    Standard permutation importance: train model once per CV fold, then for
    each feature, shuffle that feature's column in validation set and re-score.
    Importance = baseline_metric - permuted_metric.

Two-stage evaluation:
    S1 (binary): cancer vs non-cancer -> AUC drop
    S2 (multiclass): cancer-type classification -> macro-F1 drop

Usage:
    python scripts/analysis/weekend_experiments/permutation_importance.py
    python scripts/analysis/weekend_experiments/permutation_importance.py --dry-run
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
from scipy.signal import savgol_filter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
from src.sers.config import RESULTS_DIR

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CANCER_TYPES = ["PRO", "BRE", "OVA", "LUN", "CRC", "PAN", "BLC"]
NON_CANCER = ["NOR", "DIA", "HBP", "H.D."]
GROUP_ALIASES = {"PAN": ["CPAN", "YPAN"], "NOR": ["YNOR"]}

N_SPLITS = 5
N_PERM_FULL = 1000
N_PERM_DRY = 10
N_FEATURES_DRY = 5
BATCH_SIZE = 50
N_JOBS = 4
N_NULL_FEATURES = 100
FDR_ALPHA = 0.05


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def setup_output_dir(dry_run: bool) -> Path:
    tag = "dry_run" if dry_run else datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / "weekend_experiments" / f"exp4_permutation_importance_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)
    return out_dir


def setup_logging(out_dir: Path) -> logging.Logger:
    logger = logging.getLogger("perm_importance")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(out_dir / "experiment.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data():
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

    valid = set(CANCER_TYPES) | set(NON_CANCER)
    agg_df = agg_df[agg_df["group"].isin(valid)].reset_index(drop=True)

    X_raw = agg_df[feat_cols].values
    groups = agg_df["group"].values
    sample_ids = agg_df["sample_id"].values

    return X_raw, groups, sample_ids, wavenumbers, feat_cols


def extract_1st_derivative(X: np.ndarray) -> np.ndarray:
    return np.apply_along_axis(lambda y: savgol_filter(y, 11, 3, deriv=1), 1, X)


def make_labels(groups: np.ndarray):
    cancer_set = set(CANCER_TYPES)
    bl = np.array([1 if g in cancer_set else 0 for g in groups])
    ctl = np.array([CANCER_TYPES.index(g) if g in cancer_set else -1 for g in groups])
    return bl, ctl


# ---------------------------------------------------------------------------
# CV fold training
# ---------------------------------------------------------------------------
def train_fold_models(X_d1: np.ndarray, bl: np.ndarray, ctl: np.ndarray,
                      sample_ids: np.ndarray) -> list[dict]:
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    composite = bl * 100 + np.clip(ctl, 0, 99)

    fold_data = []
    for fold_i, (tr_idx, val_idx) in enumerate(sgkf.split(X_d1, composite, groups=sample_ids)):
        # S1: binary cancer vs non-cancer
        m1 = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=1000, solver="saga",
                               class_weight="balanced", random_state=42),
        )
        m1.fit(X_d1[tr_idx], bl[tr_idx])

        # S2: multiclass cancer type (cancer samples only)
        cancer_tr = ctl[tr_idx] >= 0
        m2 = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=1000, solver="saga",
                               class_weight="balanced", multi_class="multinomial",
                               random_state=42),
        )
        m2.fit(X_d1[tr_idx][cancer_tr], ctl[tr_idx][cancer_tr])

        fold_data.append({
            "m1": m1, "m2": m2,
            "val_idx": val_idx, "tr_idx": tr_idx,
            "bl_val": bl[val_idx], "ctl_val": ctl[val_idx],
        })
    return fold_data


# ---------------------------------------------------------------------------
# Baseline metrics
# ---------------------------------------------------------------------------
def compute_baseline_metrics(fold_data: list[dict], X_d1: np.ndarray) -> list[dict]:
    baselines = []
    for fd in fold_data:
        val_idx = fd["val_idx"]
        s1_prob = fd["m1"].predict_proba(X_d1[val_idx])[:, 1]
        s1_auc = roc_auc_score(fd["bl_val"], s1_prob)

        cancer_val = fd["ctl_val"] >= 0
        if cancer_val.sum() > 0:
            s2_pred = fd["m2"].predict(X_d1[val_idx][cancer_val])
            s2_f1 = f1_score(fd["ctl_val"][cancer_val], s2_pred,
                             average="macro", zero_division=0)
        else:
            s2_f1 = np.nan
        baselines.append({"s1_auc": s1_auc, "s2_f1": s2_f1})
    return baselines


# ---------------------------------------------------------------------------
# Permutation (single feature)
# ---------------------------------------------------------------------------
def permute_feature(feature_idx: int, fold_data: list[dict], X_d1: np.ndarray,
                    baselines: list[dict], wavenumbers: np.ndarray,
                    n_perm: int = 1000, rng_seed: int = 42) -> dict:
    rng = np.random.RandomState(rng_seed + feature_idx)
    auc_drops = []
    f1_drops = []

    for _perm_i in range(n_perm):
        fold_auc_drops = []
        fold_f1_drops = []
        for fi, fd in enumerate(fold_data):
            val_idx = fd["val_idx"]
            X_perm = X_d1[val_idx].copy()
            X_perm[:, feature_idx] = rng.permutation(X_perm[:, feature_idx])

            # S1
            s1_prob = fd["m1"].predict_proba(X_perm)[:, 1]
            s1_auc = roc_auc_score(fd["bl_val"], s1_prob)
            fold_auc_drops.append(baselines[fi]["s1_auc"] - s1_auc)

            # S2
            cancer_val = fd["ctl_val"] >= 0
            if cancer_val.sum() > 0:
                s2_pred = fd["m2"].predict(X_perm[cancer_val])
                s2_f1 = f1_score(fd["ctl_val"][cancer_val], s2_pred,
                                 average="macro", zero_division=0)
                fold_f1_drops.append(baselines[fi]["s2_f1"] - s2_f1)

        auc_drops.append(np.mean(fold_auc_drops))
        f1_drops.append(np.mean(fold_f1_drops) if fold_f1_drops else np.nan)

    return {
        "feature_idx": int(feature_idx),
        "wavenumber": float(wavenumbers[feature_idx]),
        "mean_auc_drop": float(np.mean(auc_drops)),
        "std_auc_drop": float(np.std(auc_drops)),
        "mean_f1_drop": float(np.nanmean(f1_drops)),
        "std_f1_drop": float(np.nanstd(f1_drops)),
        "all_auc_drops": [float(v) for v in auc_drops],
        "all_f1_drops": [float(v) if not np.isnan(v) else None for v in f1_drops],
    }


# ---------------------------------------------------------------------------
# Null distribution (shuffle labels, not features)
# ---------------------------------------------------------------------------
def compute_null_distribution(fold_data: list[dict], X_d1: np.ndarray,
                              bl: np.ndarray, ctl: np.ndarray,
                              sample_ids: np.ndarray, wavenumbers: np.ndarray,
                              n_perm: int, n_null_features: int,
                              logger: logging.Logger) -> list[dict]:
    """Compute null distribution by shuffling labels for a subset of features."""
    rng = np.random.RandomState(999)
    n_features = X_d1.shape[1]
    null_feature_indices = rng.choice(n_features, size=min(n_null_features, n_features),
                                      replace=False)
    null_feature_indices.sort()

    logger.info(f"Computing null distribution: {len(null_feature_indices)} features x "
                f"{n_perm} permutations (shuffled labels)")

    null_results = []
    for count, fidx in enumerate(null_feature_indices):
        seed = 100000 + fidx
        feat_rng = np.random.RandomState(seed)
        auc_drops = []

        for _ in range(n_perm):
            fold_auc_drops = []
            for fi, fd in enumerate(fold_data):
                val_idx = fd["val_idx"]
                # Shuffle the binary labels
                bl_shuffled = feat_rng.permutation(fd["bl_val"])
                s1_prob = fd["m1"].predict_proba(X_d1[val_idx])[:, 1]
                try:
                    s1_auc_shuffled = roc_auc_score(bl_shuffled, s1_prob)
                except ValueError:
                    s1_auc_shuffled = 0.5
                # Permute the feature
                X_perm = X_d1[val_idx].copy()
                X_perm[:, fidx] = feat_rng.permutation(X_perm[:, fidx])
                s1_prob_perm = fd["m1"].predict_proba(X_perm)[:, 1]
                try:
                    s1_auc_perm = roc_auc_score(bl_shuffled, s1_prob_perm)
                except ValueError:
                    s1_auc_perm = 0.5
                fold_auc_drops.append(s1_auc_shuffled - s1_auc_perm)
            auc_drops.append(np.mean(fold_auc_drops))

        null_results.append({
            "feature_idx": int(fidx),
            "wavenumber": float(wavenumbers[fidx]),
            "null_auc_drops": [float(v) for v in auc_drops],
            "mean_null_auc_drop": float(np.mean(auc_drops)),
            "std_null_auc_drop": float(np.std(auc_drops)),
        })
        if (count + 1) % 20 == 0:
            logger.info(f"  Null distribution: {count + 1}/{len(null_feature_indices)}")

    return null_results


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def save_checkpoint(results: list[dict], start: int, end: int, ckpt_dir: Path):
    fname = f"batch_{start:04d}_{end:04d}.json"
    tmp_path = ckpt_dir / f".tmp_{fname}"
    final_path = ckpt_dir / fname
    with open(tmp_path, "w") as f:
        json.dump(results, f)
    os.replace(tmp_path, final_path)


def load_completed_batches(ckpt_dir: Path, logger: logging.Logger) -> dict[int, dict]:
    """Load all completed checkpoint batches. Returns {feature_idx: result}."""
    completed = {}
    for p in sorted(ckpt_dir.glob("batch_*.json")):
        try:
            with open(p) as f:
                batch = json.load(f)
            for r in batch:
                completed[r["feature_idx"]] = r
            logger.info(f"  Loaded checkpoint: {p.name} ({len(batch)} features)")
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"  Skipping corrupt checkpoint {p.name}: {e}")
    return completed


# ---------------------------------------------------------------------------
# FDR correction
# ---------------------------------------------------------------------------
def compute_pvalues_and_fdr(results: list[dict], null_results: list[dict]):
    """Compute p-values from null distribution and apply BH-FDR correction."""
    from statsmodels.stats.multitest import multipletests

    # Pool all null importance values
    all_null_auc = []
    for nr in null_results:
        all_null_auc.extend(nr["null_auc_drops"])
    all_null_auc = np.array(all_null_auc)

    # p-value for each feature: fraction of null >= observed
    p_values_auc = []
    p_values_f1 = []
    for r in results:
        obs_auc = r["mean_auc_drop"]
        p_auc = (np.sum(all_null_auc >= obs_auc) + 1) / (len(all_null_auc) + 1)
        p_values_auc.append(p_auc)

        obs_f1 = r["mean_f1_drop"]
        # Use same null for F1 (conservative approximation)
        p_f1 = (np.sum(all_null_auc >= obs_f1) + 1) / (len(all_null_auc) + 1)
        p_values_f1.append(p_f1)

    p_values_auc = np.array(p_values_auc)
    p_values_f1 = np.array(p_values_f1)

    # Benjamini-Hochberg FDR correction
    _, fdr_sig_auc, _, _ = multipletests(p_values_auc, alpha=FDR_ALPHA, method="fdr_bh")
    _, fdr_sig_f1, _, _ = multipletests(p_values_f1, alpha=FDR_ALPHA, method="fdr_bh")

    for i, r in enumerate(results):
        r["p_value_auc"] = float(p_values_auc[i])
        r["p_value_f1"] = float(p_values_f1[i])
        r["fdr_significant_auc"] = bool(fdr_sig_auc[i])
        r["fdr_significant_f1"] = bool(fdr_sig_f1[i])

    return results


# ---------------------------------------------------------------------------
# Output CSVs
# ---------------------------------------------------------------------------
def save_csvs(results: list[dict], null_results: list[dict], out_dir: Path):
    # Importance ranking (sorted by mean_auc_drop descending)
    df = pd.DataFrame([{
        "wavenumber": r["wavenumber"],
        "feature_idx": r["feature_idx"],
        "mean_auc_drop": r["mean_auc_drop"],
        "std_auc_drop": r["std_auc_drop"],
        "mean_f1_drop": r["mean_f1_drop"],
        "std_f1_drop": r["std_f1_drop"],
        "p_value_auc": r["p_value_auc"],
        "p_value_f1": r["p_value_f1"],
        "fdr_significant_auc": r["fdr_significant_auc"],
        "fdr_significant_f1": r["fdr_significant_f1"],
    } for r in results])

    df_ranked = df.sort_values("mean_auc_drop", ascending=False).reset_index(drop=True)
    df_ranked.to_csv(out_dir / "importance_ranking.csv", index=False)

    # Significant only
    df_sig = df_ranked[df_ranked["fdr_significant_auc"] | df_ranked["fdr_significant_f1"]]
    df_sig.to_csv(out_dir / "importance_significant.csv", index=False)

    # By wavenumber
    df_wn = df.sort_values("wavenumber").reset_index(drop=True)
    df_wn.to_csv(out_dir / "wavenumber_importance.csv", index=False)

    # Null distribution
    null_df = pd.DataFrame([{
        "feature_idx": nr["feature_idx"],
        "wavenumber": nr["wavenumber"],
        "mean_null_auc_drop": nr["mean_null_auc_drop"],
        "std_null_auc_drop": nr["std_null_auc_drop"],
    } for nr in null_results])
    null_df.to_csv(out_dir / "null_distribution.csv", index=False)

    return df_ranked


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_top50(df_ranked: pd.DataFrame, out_dir: Path):
    top = df_ranked.head(50).copy()
    top = top.sort_values("mean_auc_drop", ascending=True)  # for horizontal bar

    fig, ax = plt.subplots(figsize=(10, 14))
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in top["mean_auc_drop"]]
    bars = ax.barh(range(len(top)), top["mean_auc_drop"], xerr=top["std_auc_drop"],
                   color=colors, edgecolor="none", capsize=2, height=0.7)

    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([f"{wn:.0f} cm$^{{-1}}$" for wn in top["wavenumber"]], fontsize=8)
    ax.set_xlabel("Mean AUC Drop (Importance)", fontsize=11)
    ax.set_title("Top 50 Features by Permutation Importance (S1 AUC)", fontsize=13, pad=12)
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")

    # Mark FDR-significant
    for i, (_, row) in enumerate(top.iterrows()):
        if row["fdr_significant_auc"]:
            ax.text(row["mean_auc_drop"] + row["std_auc_drop"] + 0.0002, i,
                    "*", fontsize=10, color="red", va="center")

    ax.text(0.98, 0.02, "* FDR-significant (q < 0.05)", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color="red")
    ax.text(0.98, 0.05, "Red = positive importance (feature helps model)\n"
            "Blue = negative importance (feature hurts model)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="gray")

    plt.tight_layout()
    fig.savefig(out_dir / "top50_importance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_wavenumber_map(df_wn_sorted: pd.DataFrame, X_raw: np.ndarray,
                        wavenumbers: np.ndarray, out_dir: Path):
    mean_spectrum = X_raw.mean(axis=0)

    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax1.plot(wavenumbers, mean_spectrum, color="#333333", linewidth=0.8, alpha=0.7,
             label="Mean Spectrum")
    ax1.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=11)
    ax1.set_ylabel("Mean Intensity", fontsize=11, color="#333333")
    ax1.tick_params(axis="y", labelcolor="#333333")
    ax1.invert_xaxis()

    ax2 = ax1.twinx()
    importance = df_wn_sorted["mean_auc_drop"].values
    wn_sorted = df_wn_sorted["wavenumber"].values
    sig_mask = df_wn_sorted["fdr_significant_auc"].values

    ax2.fill_between(wn_sorted, 0, importance, alpha=0.3, color="#d62728",
                     label="AUC Importance")
    ax2.plot(wn_sorted, importance, color="#d62728", linewidth=0.8)

    # Highlight significant regions
    if sig_mask.any():
        sig_wn = wn_sorted[sig_mask]
        sig_imp = importance[sig_mask]
        ax2.scatter(sig_wn, sig_imp, color="#d62728", s=8, zorder=5,
                    label="FDR-significant")

    ax2.set_ylabel("Permutation Importance (AUC Drop)", fontsize=11, color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.axhline(0, color="gray", linewidth=0.3, linestyle="--")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    ax1.set_title("Permutation Importance Overlaid on Mean SERS Spectrum", fontsize=13, pad=12)
    plt.tight_layout()
    fig.savefig(out_dir / "wavenumber_map.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_volcano(results: list[dict], out_dir: Path):
    auc_drops = np.array([r["mean_auc_drop"] for r in results])
    p_vals = np.array([r["p_value_auc"] for r in results])
    fdr_sig = np.array([r["fdr_significant_auc"] for r in results])
    neg_log_p = -np.log10(np.clip(p_vals, 1e-300, 1.0))

    fig, ax = plt.subplots(figsize=(10, 7))

    # Non-significant
    ns_mask = ~fdr_sig
    ax.scatter(auc_drops[ns_mask], neg_log_p[ns_mask], s=10, alpha=0.4,
               color="#888888", label="Not significant", edgecolors="none")

    # Significant
    if fdr_sig.any():
        ax.scatter(auc_drops[fdr_sig], neg_log_p[fdr_sig], s=20, alpha=0.8,
                   color="#d62728", label="FDR-significant", edgecolors="none")

    # Threshold line
    if fdr_sig.any():
        min_sig_p = p_vals[fdr_sig].max()
        ax.axhline(-np.log10(min_sig_p), color="#d62728", linewidth=0.8,
                    linestyle="--", alpha=0.5, label=f"FDR threshold (p={min_sig_p:.4f})")

    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Permutation Importance (Mean AUC Drop)", fontsize=11)
    ax.set_ylabel("$-\\log_{10}(p)$", fontsize=11)
    ax.set_title("Volcano Plot: Permutation Importance Significance", fontsize=13, pad=12)
    ax.legend(fontsize=9)

    # Label top features
    top_indices = np.argsort(auc_drops)[-5:]
    for idx in top_indices:
        wn = results[idx]["wavenumber"]
        ax.annotate(f"{wn:.0f}", (auc_drops[idx], neg_log_p[idx]),
                    fontsize=7, ha="left", va="bottom",
                    xytext=(5, 5), textcoords="offset points")

    plt.tight_layout()
    fig.savefig(out_dir / "significance_volcano.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Permutation Feature Importance (Exp 4)")
    parser.add_argument("--dry-run", action="store_true",
                        help=f"Quick test: {N_FEATURES_DRY} features x {N_PERM_DRY} permutations")
    parser.add_argument("--n-jobs", type=int, default=N_JOBS,
                        help=f"Parallel jobs (default: {N_JOBS})")
    parser.add_argument("--n-perm", type=int, default=None,
                        help=f"Override number of permutations (default: {N_PERM_FULL})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Features per checkpoint batch (default: {BATCH_SIZE})")
    args = parser.parse_args()

    dry_run = args.dry_run
    n_perm = args.n_perm or (N_PERM_DRY if dry_run else N_PERM_FULL)
    n_jobs = args.n_jobs
    batch_size = args.batch_size

    out_dir = setup_output_dir(dry_run)
    logger = setup_logging(out_dir)
    ckpt_dir = out_dir / "checkpoints"

    logger.info("=" * 70)
    logger.info("Experiment 4: Permutation Feature Importance")
    logger.info(f"  dry_run={dry_run}, n_perm={n_perm}, n_jobs={n_jobs}, batch_size={batch_size}")
    logger.info(f"  output: {out_dir}")
    logger.info("=" * 70)

    # Save config
    config = {
        "experiment": "permutation_feature_importance",
        "dry_run": dry_run,
        "n_perm": n_perm,
        "n_jobs": n_jobs,
        "batch_size": batch_size,
        "n_splits": N_SPLITS,
        "fdr_alpha": FDR_ALPHA,
        "n_null_features": N_NULL_FEATURES if not dry_run else 5,
        "timestamp": datetime.now().isoformat(),
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    t0 = time.time()
    logger.info("Loading data...")
    X_raw, groups, sample_ids, wavenumbers, feat_cols = load_data()
    logger.info(f"  Samples: {X_raw.shape[0]}, Features: {X_raw.shape[1]}")
    logger.info(f"  Groups: {dict(zip(*np.unique(groups, return_counts=True)))}")

    # ------------------------------------------------------------------
    # 2. Feature extraction
    # ------------------------------------------------------------------
    logger.info("Extracting 1st derivative features...")
    X_d1 = extract_1st_derivative(X_raw)
    bl, ctl = make_labels(groups)
    logger.info(f"  Cancer: {bl.sum()}, Non-cancer: {(bl == 0).sum()}")

    # ------------------------------------------------------------------
    # 3. Train fold models
    # ------------------------------------------------------------------
    logger.info("Training CV fold models...")
    fold_data = train_fold_models(X_d1, bl, ctl, sample_ids)
    baselines = compute_baseline_metrics(fold_data, X_d1)
    for fi, b in enumerate(baselines):
        logger.info(f"  Fold {fi}: S1 AUC={b['s1_auc']:.4f}, S2 F1={b['s2_f1']:.4f}")
    logger.info(f"  Mean baseline: S1 AUC={np.mean([b['s1_auc'] for b in baselines]):.4f}, "
                f"S2 F1={np.nanmean([b['s2_f1'] for b in baselines]):.4f}")

    # ------------------------------------------------------------------
    # 4. Determine features to process
    # ------------------------------------------------------------------
    n_features = X_d1.shape[1]
    if dry_run:
        # Pick evenly spaced features for dry run
        feature_indices = np.linspace(0, n_features - 1, N_FEATURES_DRY, dtype=int).tolist()
        logger.info(f"DRY RUN: {len(feature_indices)} features, {n_perm} permutations each")
    else:
        feature_indices = list(range(n_features))

    # Load checkpoints
    completed = load_completed_batches(ckpt_dir, logger)
    remaining = [fi for fi in feature_indices if fi not in completed]
    logger.info(f"Features: {len(feature_indices)} total, {len(completed)} completed, "
                f"{len(remaining)} remaining")

    # ------------------------------------------------------------------
    # 5. Run permutation importance (parallel, batched)
    # ------------------------------------------------------------------
    if remaining:
        from joblib import Parallel, delayed

        # Create batches from remaining features
        batches = []
        for i in range(0, len(remaining), batch_size):
            batches.append(remaining[i:i + batch_size])

        total_done = len(completed)
        total_features = len(feature_indices)
        t_start = time.time()

        for batch_i, batch in enumerate(batches):
            batch_start = batch[0]
            batch_end = batch[-1]
            logger.info(f"Batch {batch_i + 1}/{len(batches)}: features {batch_start}-{batch_end} "
                        f"({len(batch)} features)")

            # Parallel over features within this batch
            batch_results = Parallel(n_jobs=n_jobs, verbose=0)(
                delayed(permute_feature)(
                    fidx, fold_data, X_d1, baselines, wavenumbers, n_perm, 42
                )
                for fidx in batch
            )

            # Save checkpoint (atomic)
            save_checkpoint(batch_results, batch_start, batch_end, ckpt_dir)
            for r in batch_results:
                completed[r["feature_idx"]] = r

            total_done += len(batch)
            elapsed = time.time() - t_start
            rate = total_done - len(completed) + len(batch)  # features done this run
            features_this_run = total_done - (len(completed) - len(batch_results))
            if features_this_run > 0:
                per_feature = elapsed / (total_done - len(feature_indices) + len(remaining) -
                                         (len(remaining) - sum(len(b) for b in batches[:batch_i + 1])))
            else:
                per_feature = 0

            features_processed_this_session = sum(len(b) for b in batches[:batch_i + 1])
            if features_processed_this_session > 0:
                per_feature = elapsed / features_processed_this_session
                remaining_features = len(remaining) - features_processed_this_session
                eta = per_feature * remaining_features
                logger.info(f"  Progress: {total_done}/{total_features} features | "
                            f"Elapsed: {elapsed:.0f}s | "
                            f"ETA: {eta:.0f}s ({eta / 60:.1f}min)")
            else:
                logger.info(f"  Progress: {total_done}/{total_features} features | "
                            f"Elapsed: {elapsed:.0f}s")

    # ------------------------------------------------------------------
    # 6. Collect all results in original feature order
    # ------------------------------------------------------------------
    results = [completed[fi] for fi in feature_indices]
    logger.info(f"All {len(results)} features complete.")

    # ------------------------------------------------------------------
    # 7. Null distribution
    # ------------------------------------------------------------------
    n_null = 5 if dry_run else N_NULL_FEATURES
    null_results = compute_null_distribution(
        fold_data, X_d1, bl, ctl, sample_ids, wavenumbers,
        n_perm=min(n_perm, 100),  # Cap null permutations for speed
        n_null_features=n_null,
        logger=logger,
    )

    # ------------------------------------------------------------------
    # 8. FDR correction
    # ------------------------------------------------------------------
    logger.info("Computing p-values and FDR correction...")
    results = compute_pvalues_and_fdr(results, null_results)
    n_sig_auc = sum(1 for r in results if r["fdr_significant_auc"])
    n_sig_f1 = sum(1 for r in results if r["fdr_significant_f1"])
    logger.info(f"  FDR-significant features: AUC={n_sig_auc}, F1={n_sig_f1}")

    # ------------------------------------------------------------------
    # 9. Save CSVs
    # ------------------------------------------------------------------
    logger.info("Saving CSV outputs...")
    df_ranked = save_csvs(results, null_results, out_dir)

    # ------------------------------------------------------------------
    # 10. Figures
    # ------------------------------------------------------------------
    logger.info("Generating figures...")

    # Wavenumber-sorted df for wavenumber map
    df_wn = pd.DataFrame([{
        "wavenumber": r["wavenumber"],
        "mean_auc_drop": r["mean_auc_drop"],
        "fdr_significant_auc": r["fdr_significant_auc"],
    } for r in results]).sort_values("wavenumber").reset_index(drop=True)

    plot_top50(df_ranked, out_dir)
    plot_wavenumber_map(df_wn, X_raw, wavenumbers, out_dir)
    plot_volcano(results, out_dir)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed_total = time.time() - t0
    logger.info("=" * 70)
    logger.info("EXPERIMENT COMPLETE")
    logger.info(f"  Total time: {elapsed_total:.0f}s ({elapsed_total / 60:.1f}min)")
    logger.info(f"  Features: {len(results)}, Permutations/feature: {n_perm}")
    logger.info(f"  FDR-significant (AUC): {n_sig_auc}/{len(results)}")
    logger.info(f"  FDR-significant (F1):  {n_sig_f1}/{len(results)}")
    if len(df_ranked) > 0:
        top5 = df_ranked.head(5)
        logger.info("  Top 5 features by AUC importance:")
        for _, row in top5.iterrows():
            sig = " *" if row["fdr_significant_auc"] else ""
            logger.info(f"    {row['wavenumber']:.0f} cm-1: "
                        f"AUC drop={row['mean_auc_drop']:.6f} +/- {row['std_auc_drop']:.6f}{sig}")
    logger.info(f"  Output directory: {out_dir}")
    logger.info("=" * 70)

    print(f"\nDone. Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
