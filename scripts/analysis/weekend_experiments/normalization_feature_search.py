#!/usr/bin/env python3
"""
Experiment 6: Normalization x Feature Transform Full Search
============================================================

5 normalizations x 6 feature transforms x 3 models x 20 seeds = 1,800 conditions.
Each condition evaluated via 5-fold StratifiedGroupKFold CV.

Optimization: raw spectra (trimmed + smoothed + baseline-corrected, but NOT
normalized) are loaded once and cached. Each normalization is then applied
in-memory, avoiding repeated I/O.

Usage:
    python scripts/analysis/weekend_experiments/normalization_feature_search.py
    python scripts/analysis/weekend_experiments/normalization_feature_search.py --dry-run
    python scripts/analysis/weekend_experiments/normalization_feature_search.py --n-seeds 5 --n-jobs 4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import time
import warnings
from datetime import datetime, timedelta
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.signal import savgol_filter

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "analysis"))

from src.sers.preprocessing import (
    baseline_correction,
    normalize_spectrum,
    resample,
    trim_spectrum,
)
from src.sers.io import find_spectra, read_spectrum, parse_filename
from src.sers.config import RESULTS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CANCER_TYPES = ("PRO", "LUN", "CRC", "PAN", "OVA", "BRE", "BLC")
NON_CANCER_GROUPS = ("NOR", "DIA", "HBP", "H.D.")

THERMO_MAP = {
    "1. Prostate cancer (100개)": "PRO",
    "2. Breast cancer (30개)": "BRE",
    "3. Ovarian cancer (70개)": "OVA",
    "4. Lung cancer (300개)": "LUN",
    "5. Normal (100개)": "NOR",
    "6. Diabetes (100개)": "DIA",
    "7. High blood pressure (100개)": "HBP",
    "8. High blood pressure + Diabetes (100개)": "H.D.",
    "9. Colorectal cancer (300개)": "CRC",
    "10-1. C-Pancreatic cancer (70개)": "CPAN",
    "10-3. Y-Pancreatic cancer (YPAN)": "YPAN",
    "11 BLC (299개)": "BLC",
    "12. Y-Normal (YNOR)": "YNOR",
}

NORMALIZATIONS = ["snv", "minmax", "l2", "area", "none"]
MODEL_NAMES = ["LR", "XGB", "RF"]
SG_WINDOW, SG_POLY = 11, 3
GRID = np.linspace(400, 2200, 901)
SEEDS_FULL = list(range(42, 62))  # 42..61 = 20 seeds

# ---------------------------------------------------------------------------
# Feature transforms (applied AFTER normalization + mean-aggregation)
# ---------------------------------------------------------------------------
FEATURE_TRANSFORMS = {
    "raw": lambda X: X,
    "d1": lambda X: np.apply_along_axis(
        lambda y: savgol_filter(y, 11, 3, deriv=1), 1, X
    ),
    "d2": lambda X: np.apply_along_axis(
        lambda y: savgol_filter(y, 11, 3, deriv=2), 1, X
    ),
    "raw_d1": lambda X: np.hstack(
        [
            X,
            np.apply_along_axis(lambda y: savgol_filter(y, 11, 3, deriv=1), 1, X),
        ]
    ),
    "raw_d1_d2": lambda X: np.hstack(
        [
            X,
            np.apply_along_axis(lambda y: savgol_filter(y, 11, 3, deriv=1), 1, X),
            np.apply_along_axis(lambda y: savgol_filter(y, 11, 3, deriv=2), 1, X),
        ]
    ),
    "d1_d2": lambda X: np.hstack(
        [
            np.apply_along_axis(lambda y: savgol_filter(y, 11, 3, deriv=1), 1, X),
            np.apply_along_axis(lambda y: savgol_filter(y, 11, 3, deriv=2), 1, X),
        ]
    ),
}


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------
def preprocess_channel_no_norm(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Trim + smooth + baseline correct (NO normalization). For caching."""
    y_sg = savgol_filter(y, SG_WINDOW, SG_POLY, deriv=0)
    x_tr, y_tr = trim_spectrum(x.copy(), y_sg, region=(400, 2200))
    y_tr = baseline_correction(y_tr, window=101)
    return resample(x_tr, y_tr, grid)


def apply_normalization(X: np.ndarray, method: str) -> np.ndarray:
    """Apply normalization to each row of X."""
    if method == "none":
        return X.copy()
    out = np.empty_like(X)
    for i in range(X.shape[0]):
        out[i] = normalize_spectrum(X[i], method=method)
    return out


# ---------------------------------------------------------------------------
# Data loading (raw cache — no normalization)
# ---------------------------------------------------------------------------
def load_raw_cache(data_dir: Path, folder_map: dict, grid: np.ndarray, pattern: str = "*.CSV"):
    """Load raw spectra, preprocess WITHOUT normalization. Returns (X_raw, meta_df)."""
    all_X, meta_rows, failed = [], [], 0
    for folder_name, group in folder_map.items():
        folder = data_dir / folder_name
        if not folder.is_dir():
            continue
        files = find_spectra(folder, pattern=pattern, recursive=False)
        if not files:
            for p in ["*.txt", "*.csv", "*.CSV"]:
                files = find_spectra(folder, pattern=p, recursive=False)
                if files:
                    break
        files = [
            f
            for f in files
            if "_ave" not in f.stem.lower() and "zone.identifier" not in f.name.lower()
        ]
        for fp in files:
            try:
                sid = parse_filename(fp, fallback_group=group)
                x, y = read_spectrum(fp)
                ch0 = preprocess_channel_no_norm(x, y, grid)
                all_X.append(ch0)
                meta_rows.append(
                    {
                        "group": sid.group,
                        "sample_id": sid.sample_id,
                        "replicate": sid.replicate,
                    }
                )
            except Exception:
                failed += 1
    X = np.stack(all_X).astype(np.float32)
    df = pd.DataFrame(meta_rows)
    logger.info(f"  Raw cache: {len(X)} spectra ({failed} failed), shape {X.shape}")
    return X, df


# ---------------------------------------------------------------------------
# Mean aggregation per sample
# ---------------------------------------------------------------------------
def mean_aggregate(X: np.ndarray, meta: pd.DataFrame):
    """Mean aggregate spectra per (group, sample_id). Returns (X_agg, groups, sample_ids)."""
    meta = meta.copy()
    meta["_idx"] = np.arange(len(meta))
    grouped = meta.groupby(["group", "sample_id"])["_idx"].apply(list)
    X_agg, groups, sample_ids = [], [], []
    for (grp, sid), idxs in grouped.items():
        X_agg.append(X[idxs].mean(axis=0))
        groups.append(grp)
        sample_ids.append(sid)
    return np.array(X_agg, dtype=np.float32), np.array(groups), np.array(sample_ids)


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------
def build_model(model_name: str):
    if model_name == "LR":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0, max_iter=1000, solver="saga",
                class_weight="balanced", random_state=42,
            ),
        )
    elif model_name == "XGB":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, n_jobs=2,
            tree_method="hist", verbosity=0,
            objective="binary:logistic", eval_metric="logloss",
        )
    elif model_name == "RF":
        return RandomForestClassifier(
            n_estimators=500, class_weight="balanced_subsample",
            n_jobs=2, random_state=42,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ---------------------------------------------------------------------------
# CV evaluation
# ---------------------------------------------------------------------------
def run_cv(X: np.ndarray, groups: np.ndarray, sample_ids: np.ndarray, model_name: str, seed: int = 42):
    cancer_set = set(CANCER_TYPES)
    bl = np.array([1 if g in cancer_set else 0 for g in groups])
    ctl = np.array([CANCER_TYPES.index(g) if g in cancer_set else -1 for g in groups])

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    composite = bl * 100 + np.clip(ctl, 0, 99)

    aucs, f1s = [], []
    for tr, val in sgkf.split(X, composite, groups=sample_ids):
        # Stage 1: binary cancer vs. non-cancer
        m1 = build_model(model_name)
        m1.fit(X[tr], bl[tr])
        prob = m1.predict_proba(X[val])[:, 1]
        aucs.append(roc_auc_score(bl[val], prob))

        # Stage 2: multiclass cancer-type classification
        cm_tr = ctl[tr] >= 0
        cm_val = ctl[val] >= 0
        if cm_tr.sum() > 10 and cm_val.sum() > 0:
            m2 = build_model(model_name)
            if model_name == "XGB":
                m2.set_params(
                    objective="multi:softprob",
                    num_class=len(CANCER_TYPES),
                    eval_metric="mlogloss",
                )
            m2.fit(X[tr][cm_tr], ctl[tr][cm_tr])
            pred = m2.predict(X[val][cm_val])
            f1s.append(f1_score(ctl[val][cm_val], pred, average="macro", zero_division=0))

    return {
        "s1_auc": float(np.mean(aucs)),
        "s2_f1": float(np.mean(f1s)) if f1s else float("nan"),
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def ckpt_path(ckpt_dir: Path, norm: str, transform: str, model: str, seed: int) -> Path:
    return ckpt_dir / f"{norm}_{transform}_{model}_seed{seed}.json"


def load_checkpoint(path: Path) -> dict | None:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return None
    return None


def save_checkpoint(path: Path, data: dict):
    """Atomic write: temp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with open(fd, "w") as f:
            json.dump(data, f)
        Path(tmp).replace(path)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_heatmap(agg_df: pd.DataFrame, model_name: str, metric: str, out_path: Path):
    """Plot norm x transform heatmap for one model."""
    sub = agg_df[agg_df["model"] == model_name].copy()
    if sub.empty:
        return

    col = f"mean_{metric}"
    std_col = f"std_{metric}"
    pivot_mean = sub.pivot(index="norm", columns="transform", values=col)
    pivot_std = sub.pivot(index="norm", columns="transform", values=std_col)

    # Reorder
    norm_order = [n for n in NORMALIZATIONS if n in pivot_mean.index]
    tf_order = [t for t in FEATURE_TRANSFORMS if t in pivot_mean.columns]
    pivot_mean = pivot_mean.reindex(index=norm_order, columns=tf_order)
    pivot_std = pivot_std.reindex(index=norm_order, columns=tf_order)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(pivot_mean.values, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(tf_order)))
    ax.set_xticklabels(tf_order, rotation=45, ha="right")
    ax.set_yticks(range(len(norm_order)))
    ax.set_yticklabels(norm_order)

    # Annotate cells
    for i in range(len(norm_order)):
        for j in range(len(tf_order)):
            val = pivot_mean.values[i, j]
            std = pivot_std.values[i, j]
            if np.isnan(val):
                continue
            txt = f"{val:.3f}\n+/-{std:.3f}"
            color = "white" if val > (pivot_mean.values[~np.isnan(pivot_mean.values)].max() * 0.8) else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=color)

    metric_label = "S1 AUC" if metric == "s1_auc" else "S2 Macro-F1"
    ax.set_title(f"{model_name} — {metric_label} (mean +/- std over seeds)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Feature Transform")
    ax.set_ylabel("Normalization")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(metric_label)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved heatmap: {out_path.name}")


def plot_best_per_model(agg_df: pd.DataFrame, metric: str, out_path: Path):
    """Bar chart: best combo per model."""
    col = f"mean_{metric}"
    std_col = f"std_{metric}"
    rows = []
    for m in MODEL_NAMES:
        sub = agg_df[agg_df["model"] == m]
        if sub.empty:
            continue
        best = sub.loc[sub[col].idxmax()]
        rows.append(
            {
                "model": m,
                "norm": best["norm"],
                "transform": best["transform"],
                col: best[col],
                std_col: best[std_col],
            }
        )
    if not rows:
        return

    bdf = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(bdf))
    bars = ax.bar(x, bdf[col], yerr=bdf[std_col], capsize=5, color=["#4C72B0", "#DD8452", "#55A868"])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['model']}\n{r['norm']}+{r['transform']}" for _, r in bdf.iterrows()], fontsize=9)

    for i, bar in enumerate(bars):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + bdf[std_col].iloc[i] + 0.002,
            f"{bdf[col].iloc[i]:.4f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    metric_label = "S1 AUC" if metric == "s1_auc" else "S2 Macro-F1"
    ax.set_title(f"Best {metric_label} per Model (20 seeds)", fontsize=12, fontweight="bold")
    ax.set_ylabel(metric_label)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Exp6: Normalization x Feature Transform Full Search")
    parser.add_argument("--dry-run", action="store_true", help="Quick test: 2 norms x 2 transforms x 1 model x 2 seeds = 8")
    parser.add_argument("--n-seeds", type=int, default=20, help="Number of seeds (default 20)")
    parser.add_argument("--n-jobs", type=int, default=1, help="(reserved for future parallel use)")
    parser.add_argument("--data-dir", type=str, default=None, help="Override data directory")
    args = parser.parse_args()

    # ---- Setup ----
    out_dir = RESULTS_DIR / "weekend_experiments" / "exp6_norm_feature_search"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # File logger
    fh = logging.FileHandler(out_dir / "experiment.log", mode="a")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)

    # ---- Determine search space ----
    if args.dry_run:
        norms = ["snv", "l2"]
        transforms = ["raw", "d1"]
        models = ["LR"]
        seeds = [42, 43]
        logger.info("=== DRY RUN MODE (8 conditions) ===")
    else:
        norms = NORMALIZATIONS
        transforms = list(FEATURE_TRANSFORMS.keys())
        models = MODEL_NAMES
        seeds = SEEDS_FULL[: args.n_seeds]

    total = len(norms) * len(transforms) * len(models) * len(seeds)
    logger.info(f"Search space: {len(norms)} norms x {len(transforms)} transforms x {len(models)} models x {len(seeds)} seeds = {total}")

    # ---- Save config ----
    config = {
        "experiment": "exp6_norm_feature_search",
        "timestamp": datetime.now().isoformat(),
        "dry_run": args.dry_run,
        "normalizations": norms,
        "feature_transforms": transforms,
        "models": models,
        "seeds": seeds,
        "n_folds": 5,
        "total_conditions": total,
        "grid_points": len(GRID),
        "grid_range": [float(GRID[0]), float(GRID[-1])],
        "sg_window": SG_WINDOW,
        "sg_poly": SG_POLY,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    # ---- Locate data ----
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = PROJECT_ROOT / "data" / "raw_data"
    if not data_dir.is_dir():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)

    # Filter folder map to existing directories
    folder_map = {k: v for k, v in THERMO_MAP.items() if (data_dir / k).is_dir()}
    if not folder_map:
        logger.error(f"No data folders found in {data_dir}")
        sys.exit(1)
    logger.info(f"Found {len(folder_map)} data folders in {data_dir}")

    # ---- Step 1: Load raw cache (no normalization) ----
    logger.info("Loading raw spectra (trim + smooth + baseline, NO normalization)...")
    t0 = time.time()
    X_raw, meta_raw = load_raw_cache(data_dir, folder_map, GRID)
    logger.info(f"  Raw cache loaded in {time.time() - t0:.1f}s")

    # ---- Step 2: Pre-compute normalized + aggregated data per normalization ----
    logger.info("Pre-computing normalized + aggregated data for each normalization...")
    norm_data = {}
    for nm in norms:
        t1 = time.time()
        X_norm = apply_normalization(X_raw, nm)
        X_agg, groups, sample_ids = mean_aggregate(X_norm, meta_raw)
        # Map hospital-specific codes to disease names
        alias_map = {"CPAN": "PAN", "YPAN": "PAN", "YNOR": "NOR"}
        groups = np.array([alias_map.get(g, g) for g in groups])
        # Filter to known groups
        valid = set(CANCER_TYPES) | set(NON_CANCER_GROUPS)
        mask = np.array([g in valid for g in groups])
        norm_data[nm] = {
            "X": X_agg[mask],
            "groups": groups[mask],
            "sample_ids": sample_ids[mask],
        }
        logger.info(f"  {nm}: {mask.sum()} samples, {time.time() - t1:.1f}s")

    # ---- Step 3: Run grid search ----
    logger.info("=" * 60)
    logger.info("Starting grid search...")
    results = []
    done_count = 0
    t_start = time.time()

    conditions = list(product(norms, transforms, models, seeds))

    for idx, (nm, tf, mdl, seed) in enumerate(conditions):
        cp = ckpt_path(ckpt_dir, nm, tf, mdl, seed)
        cached = load_checkpoint(cp)
        if cached is not None:
            results.append(cached)
            done_count += 1
            continue

        # Apply feature transform
        nd = norm_data[nm]
        X_tf = FEATURE_TRANSFORMS[tf](nd["X"])

        # Run CV
        metrics = run_cv(X_tf, nd["groups"], nd["sample_ids"], mdl, seed)

        record = {
            "norm": nm,
            "transform": tf,
            "model": mdl,
            "seed": seed,
            "s1_auc": metrics["s1_auc"],
            "s2_f1": metrics["s2_f1"],
        }
        save_checkpoint(cp, record)
        results.append(record)
        done_count += 1

        # Progress
        if done_count % 10 == 0 or done_count == total:
            elapsed = time.time() - t_start
            rate = done_count / max(elapsed, 1e-6)
            remaining = (total - done_count) / max(rate, 1e-6)
            eta_str = str(timedelta(seconds=int(remaining)))
            logger.info(
                f"  [{done_count:>5}/{total}] "
                f"{nm}/{tf}/{mdl}/s{seed} "
                f"AUC={metrics['s1_auc']:.4f}  F1={metrics['s2_f1']:.4f}  "
                f"elapsed={timedelta(seconds=int(elapsed))}  ETA={eta_str}"
            )

    elapsed_total = time.time() - t_start
    logger.info(f"Grid search complete in {timedelta(seconds=int(elapsed_total))}")

    # ---- Step 4: Aggregate results ----
    logger.info("Aggregating results...")
    full_df = pd.DataFrame(results)
    full_df.to_csv(out_dir / "full_search_results.csv", index=False)
    logger.info(f"  Saved full_search_results.csv ({len(full_df)} rows)")

    agg = (
        full_df.groupby(["norm", "transform", "model"])
        .agg(
            mean_s1_auc=("s1_auc", "mean"),
            std_s1_auc=("s1_auc", "std"),
            mean_s2_f1=("s2_f1", "mean"),
            std_s2_f1=("s2_f1", "std"),
            n_seeds=("seed", "count"),
        )
        .reset_index()
    )
    agg.to_csv(out_dir / "aggregated_results.csv", index=False)
    logger.info(f"  Saved aggregated_results.csv ({len(agg)} rows)")

    # ---- Best combos ----
    best = {}
    for metric in ["s1_auc", "s2_f1"]:
        col = f"mean_{metric}"
        valid_agg = agg.dropna(subset=[col])
        if valid_agg.empty:
            continue
        best_row = valid_agg.loc[valid_agg[col].idxmax()]
        best[metric] = {
            "norm": best_row["norm"],
            "transform": best_row["transform"],
            "model": best_row["model"],
            f"mean_{metric}": float(best_row[col]),
            f"std_{metric}": float(best_row[f"std_{metric}"]),
        }
    (out_dir / "best_combo.json").write_text(json.dumps(best, indent=2))
    logger.info(f"  Best S1 AUC: {best.get('s1_auc', {})}")
    logger.info(f"  Best S2 F1:  {best.get('s2_f1', {})}")

    # ---- Step 5: Generate plots ----
    logger.info("Generating plots...")

    # Heatmaps per model (S1 AUC)
    for mdl in models:
        plot_heatmap(agg, mdl, "s1_auc", out_dir / f"heatmap_norm_x_transform_{mdl}.png")

    # Best per model bar chart (S1 AUC)
    plot_best_per_model(agg, "s1_auc", out_dir / "best_per_model.png")

    logger.info("=" * 60)
    logger.info("Experiment 6 complete.")
    logger.info(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
