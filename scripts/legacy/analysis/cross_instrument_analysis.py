#!/usr/bin/env python3
"""
Cross-Instrument SERS Analysis
===============================

1. Preprocess medical-instrument data (raw_data_medical/)
2. Compare Thermo vs Medical spectra in 400-2200 cm⁻¹
3. Train model on medical data (6-spot medoid, same architecture)
4. Cross-instrument inference (Thermo→Medical, Medical→Thermo)

Usage:
    python scripts/analysis/cross_instrument_analysis.py
"""

from __future__ import annotations
import sys, os, json, logging, re
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sers.io import read_spectrum, parse_filename, find_spectra, make_common_grid
from src.sers.preprocessing import (
    preprocess_single_spectrum, preprocess_spectra, save_processed_spectra,
)
from src.sers.config import load_config, RESULTS_DIR

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
OUTPUT_DIR = PROJECT_ROOT / "results" / "cross_instrument"

# Medical folder → group code (S-Pan excluded)
MEDICAL_FOLDER_MAP = {
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
    # "10-2. S-Pancreatic cancer (72개)": EXCLUDED — post-op, no screening value
    "10-3. Y-Pancreatic cancer (30개)": "YPAN",
    "11. Bladdder Cancer (299개)": "BLC",
    "12. Y-Normal (29개)": "YNOR",
}

# NOR: only replicates 1-6 (reps 7-20 were machine performance testing)
NOR_MAX_REPLICATE = 6

# Model class definitions — full set including BLC, BRE, YPAN, YNOR
CANCER_TYPES = ("PRO", "LUN", "CRC", "PAN", "OVA", "BRE", "BLC")
NON_CANCER_GROUPS = ("NOR", "DIA", "HBP", "H.D.")

PREP_PARAMS = dict(
    do_trim=True, trim_region=(400, 2200),
    do_smooth=True, smooth_window=11, smooth_poly=3,
    do_baseline=True, baseline_window=101,
    normalization="snv",
)


# =============================================================================
# Data Loading
# =============================================================================
def load_instrument_data(data_dir: Path, folder_map: dict, pattern: str = "*.txt",
                         n_expected_reps: int = 6) -> dict:
    """Load raw spectra from an instrument directory.

    Returns dict: (group, sample_id, replicate) → (x, y)
    Skips _ave files and Background folders.
    """
    spectra = {}
    meta_rows = []
    failed = []

    for folder_name, group in folder_map.items():
        folder = data_dir / folder_name
        if not folder.is_dir():
            logger.warning(f"Folder not found: {folder}")
            continue

        # Find spectrum files, try both patterns
        files = find_spectra(folder, pattern=pattern, recursive=False)
        if not files:
            files = find_spectra(folder, pattern="*.CSV", recursive=False)
        if not files:
            files = find_spectra(folder, pattern="*.csv", recursive=False)

        # Filter out _ave and Zone.Identifier
        files = [f for f in files
                 if "_ave" not in f.stem.lower()
                 and "zone.identifier" not in f.name.lower()]

        for fp in files:
            try:
                sid = parse_filename(fp, fallback_group=group)

                # NOR: only keep replicates 1-6 (7-20 are machine QC)
                if sid.group == "NOR" and sid.replicate > NOR_MAX_REPLICATE:
                    continue

                x, y = read_spectrum(fp)
                key = (sid.group, sid.sample_id, sid.replicate)
                spectra[key] = (x, y)
                meta_rows.append({
                    "group": sid.group, "sample_id": sid.sample_id,
                    "replicate": sid.replicate, "n_points": len(x),
                    "x_min": float(x.min()), "x_max": float(x.max()),
                    "y_mean": float(y.mean()), "file": fp.name,
                })
            except Exception as e:
                failed.append((fp.name, str(e)))

    meta = pd.DataFrame(meta_rows)
    logger.info(f"Loaded {len(spectra)} spectra from {data_dir.name} "
                f"({len(failed)} failed)")
    if not meta.empty:
        logger.info(f"  Groups: {sorted(meta['group'].unique())}")
        logger.info(f"  Samples/group: {meta.groupby('group')['sample_id'].nunique().to_dict()}")
    return spectra, meta, failed


# =============================================================================
# Preprocessing
# =============================================================================
def build_grid(spectra: dict) -> np.ndarray:
    """Build common wavenumber grid from spectra, trimmed to 400-2200."""
    x_arrays = [x for x, y in spectra.values()]
    # After trim, find intersection
    x_min = max(x[x >= 400].min() for x, _ in spectra.values())
    x_max = min(x[x <= 2200].max() for x, _ in spectra.values())
    # Use same n_points as production model (933) for compatibility
    return np.linspace(max(x_min, 402.0), min(x_max, 2198.0), 933)


def preprocess_all(spectra: dict, grid: np.ndarray) -> dict:
    """Preprocess all spectra with production params."""
    processed = {}
    for key, (x, y) in spectra.items():
        try:
            y_proc = preprocess_single_spectrum(x, y, grid, **PREP_PARAMS)
            processed[key] = y_proc
        except Exception as e:
            logger.warning(f"Preprocessing failed for {key}: {e}")
    return processed


def build_dataframe(processed: dict, grid: np.ndarray) -> pd.DataFrame:
    """Convert processed spectra dict to DataFrame matching train.py format."""
    rows = []
    for (group, sid, rep), y_proc in processed.items():
        row = {"group": group, "sample_id": sid, "replicate": rep}
        for i, wn in enumerate(grid):
            row[f"x_{wn:.2f}"] = y_proc[i]
        rows.append(row)
    return pd.DataFrame(rows)


# No patient-level aggregation — train on individual spectra (experiment protocol)


# =============================================================================
# Spectral Comparison Plot
# =============================================================================
def plot_spectral_comparison(thermo_processed, medical_processed,
                             grid_thermo, grid_medical, output_dir):
    """Compare mean spectra per group between instruments."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build mean spectra per group per instrument
    def group_means(processed, grid):
        groups = {}
        for (g, sid, rep), y in processed.items():
            groups.setdefault(g, []).append(y)
        return {g: np.mean(ys, axis=0) for g, ys in groups.items()}, grid

    thermo_means, tg = group_means(thermo_processed, grid_thermo)
    medical_means, mg = group_means(medical_processed, grid_medical)

    common_groups = sorted(set(thermo_means) & set(medical_means))
    if not common_groups:
        logger.warning("No common groups for comparison plot")
        return

    n = len(common_groups)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    colors = {"Thermo": "#2196F3", "Medical": "#E91E63"}

    for ax, group in zip(axes, common_groups):
        if group in thermo_means:
            ax.plot(tg, thermo_means[group], color=colors["Thermo"],
                    label="Thermo", alpha=0.8, linewidth=1)
        if group in medical_means:
            ax.plot(mg, medical_means[group], color=colors["Medical"],
                    label="Medical", alpha=0.8, linewidth=1)
        ax.set_ylabel("SNV Intensity")
        ax.set_title(f"{group}", fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Raman Shift (cm⁻¹)")
    fig.suptitle("Thermo vs Medical Instrument — Mean Preprocessed Spectra (400-2200 cm⁻¹)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(output_dir / "spectral_comparison_by_group.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved spectral comparison plot")

    # Correlation heatmap between instruments
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    corr_matrix = np.zeros((len(common_groups), len(common_groups)))
    labels_t = [f"T-{g}" for g in common_groups]
    labels_m = [f"M-{g}" for g in common_groups]

    all_means = []
    all_labels = []
    for g in common_groups:
        if g in thermo_means:
            all_means.append(thermo_means[g])
            all_labels.append(f"Thermo-{g}")
        if g in medical_means:
            all_means.append(medical_means[g])
            all_labels.append(f"Medical-{g}")

    # Interpolate to same grid if different
    if len(tg) != len(mg) or not np.allclose(tg, mg):
        from scipy.interpolate import interp1d
        common = np.linspace(max(tg.min(), mg.min()), min(tg.max(), mg.max()), 933)
        interp_means = []
        for i, m in enumerate(all_means):
            g = tg if "Thermo" in all_labels[i] else mg
            f = interp1d(g, m, kind="linear", fill_value="extrapolate")
            interp_means.append(f(common))
        all_means = interp_means

    corr = np.corrcoef(all_means)
    im = ax2.imshow(corr, vmin=0.5, vmax=1.0, cmap="RdYlGn", aspect="auto")
    ax2.set_xticks(range(len(all_labels)))
    ax2.set_yticks(range(len(all_labels)))
    ax2.set_xticklabels(all_labels, rotation=45, ha="right", fontsize=8)
    ax2.set_yticklabels(all_labels, fontsize=8)
    for i in range(len(all_labels)):
        for j in range(len(all_labels)):
            ax2.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center", fontsize=7)
    plt.colorbar(im, ax=ax2, label="Pearson Correlation")
    ax2.set_title("Cross-Instrument Spectral Correlation", fontweight="bold")
    plt.tight_layout()
    fig2.savefig(output_dir / "cross_instrument_correlation.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    logger.info("Saved cross-instrument correlation heatmap")


# =============================================================================
# Model Training (reusing train.py logic)
# =============================================================================
def train_model_on_data(df_agg, feat_cols, cancer_types, non_cancer_groups,
                        n_splits=5, label=""):
    """Train two-stage model (LR for production model) using CV. Returns results dict."""
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        roc_auc_score, f1_score, accuracy_score,
        confusion_matrix, classification_report,
    )

    valid_groups = set(cancer_types) | set(non_cancer_groups)
    df = df_agg[df_agg["group"].isin(valid_groups)].copy()

    if len(df) == 0:
        logger.error(f"No samples with valid groups for training ({label})")
        return None

    X = df[feat_cols].values
    groups_arr = df["group"].values
    # Group by patient (group+sample_id) for StratifiedGroupKFold
    sample_ids = (df["group"] + "_" + df["sample_id"].astype(str)).values

    binary_labels = np.array([1 if g in cancer_types else 0 for g in groups_arr])
    cancer_type_map = {ct: i for i, ct in enumerate(cancer_types)}
    cancer_type_labels = np.array([cancer_type_map.get(g, -1) for g in groups_arr])

    logger.info(f"\n{'='*60}")
    logger.info(f"  Training: {label}")
    logger.info(f"{'='*60}")
    logger.info(f"  Samples: {len(df)}, Features: {len(feat_cols)}")
    logger.info(f"  Cancer: {(binary_labels==1).sum()}, Non-cancer: {(binary_labels==0).sum()}")
    for ct in cancer_types:
        n = (groups_arr == ct).sum()
        if n > 0:
            logger.info(f"    {ct}: {n}")

    # Stratified Group K-Fold
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Stage 1: Binary
    s1_probs_all = np.full(len(df), np.nan)
    s1_labels_all = binary_labels.copy()

    # Stage 2: Cancer type
    s2_probs_all = np.full((len(df), len(cancer_types)), np.nan)
    s2_labels_all = cancer_type_labels.copy()

    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, binary_labels, sample_ids)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = binary_labels[train_idx], binary_labels[val_idx]

        # Stage 1
        s1 = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, max_iter=1000, solver="saga", class_weight="balanced"))
        s1.fit(X_tr, y_tr)
        s1_prob = s1.predict_proba(X_val)[:, 1]
        s1_probs_all[val_idx] = s1_prob

        try:
            auc = roc_auc_score(y_val, s1_prob)
        except ValueError:
            auc = float("nan")

        # Stage 2
        cancer_mask_tr = binary_labels[train_idx] == 1
        cancer_mask_val = binary_labels[val_idx] == 1

        if cancer_mask_tr.sum() > 0 and cancer_mask_val.sum() > 0:
            s2 = make_pipeline(StandardScaler(), LogisticRegression(
                C=1.0, max_iter=1000, solver="saga", class_weight="balanced",
                multi_class="multinomial"))

            ctl_tr = cancer_type_labels[train_idx][cancer_mask_tr]
            ctl_val = cancer_type_labels[val_idx][cancer_mask_val]

            # Only train on cancer types that exist in training fold
            present = np.unique(ctl_tr)
            if len(present) >= 2:
                s2.fit(X_tr[cancer_mask_tr], ctl_tr)
                s2_prob = s2.predict_proba(X_val[cancer_mask_val])
                # Map back to full cancer_types indices
                for ci, cls in enumerate(s2.classes_):
                    full_idx = cls
                    val_cancer_idx = val_idx[cancer_mask_val]
                    s2_probs_all[val_cancer_idx, full_idx] = s2_prob[:, ci]

                s2_pred = s2.predict(X_val[cancer_mask_val])
                s2_f1 = f1_score(ctl_val, s2_pred, average="macro", zero_division=0)
            else:
                s2_f1 = float("nan")
        else:
            s2_f1 = float("nan")

        fold_metrics.append({"fold": fold, "s1_auc": auc, "s2_f1_macro": s2_f1})
        logger.info(f"  Fold {fold}: AUC={auc:.4f}, S2-F1={s2_f1:.4f}")

    # Overall metrics
    valid_s1 = ~np.isnan(s1_probs_all)
    overall_auc = roc_auc_score(s1_labels_all[valid_s1], s1_probs_all[valid_s1])
    s1_pred = (s1_probs_all > 0.5).astype(int)
    s1_pred[np.isnan(s1_probs_all)] = -1
    valid_mask = s1_pred >= 0
    overall_acc = accuracy_score(s1_labels_all[valid_mask], s1_pred[valid_mask])
    overall_f1 = f1_score(s1_labels_all[valid_mask], s1_pred[valid_mask], average="macro")

    mean_auc = np.nanmean([m["s1_auc"] for m in fold_metrics])
    mean_s2_f1 = np.nanmean([m["s2_f1_macro"] for m in fold_metrics])

    logger.info(f"\n  Overall {label}:")
    logger.info(f"    S1 AUC:  {overall_auc:.4f} (mean fold: {mean_auc:.4f})")
    logger.info(f"    S1 Acc:  {overall_acc:.4f}")
    logger.info(f"    S1 F1:   {overall_f1:.4f}")
    logger.info(f"    S2 F1:   {mean_s2_f1:.4f}")

    return {
        "label": label,
        "n_samples": len(df),
        "n_features": len(feat_cols),
        "overall_auc": overall_auc,
        "overall_acc": overall_acc,
        "overall_f1": overall_f1,
        "mean_fold_auc": mean_auc,
        "mean_fold_s2_f1": mean_s2_f1,
        "fold_metrics": fold_metrics,
        "groups_arr": groups_arr,
        "binary_labels": binary_labels,
        "cancer_type_labels": cancer_type_labels,
        "s1_probs": s1_probs_all,
        "X": X,
        "feat_cols": feat_cols,
        "sample_ids": sample_ids,
    }


# =============================================================================
# Cross-Instrument Inference
# =============================================================================
def cross_instrument_inference(train_result, test_df, test_feat_cols,
                               cancer_types, non_cancer_groups,
                               label="Cross"):
    """Train on full source data, predict on target instrument."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        roc_auc_score, f1_score, accuracy_score,
        confusion_matrix, classification_report,
    )

    valid_groups = set(cancer_types) | set(non_cancer_groups)
    test_df_valid = test_df[test_df["group"].isin(valid_groups)].copy()

    X_train = train_result["X"]
    y_train = train_result["binary_labels"]
    ctl_train = train_result["cancer_type_labels"]

    X_test = test_df_valid[test_feat_cols].values
    groups_test = test_df_valid["group"].values
    y_test = np.array([1 if g in cancer_types else 0 for g in groups_test])
    ctl_test = np.array([{ct: i for i, ct in enumerate(cancer_types)}.get(g, -1)
                         for g in groups_test])

    logger.info(f"\n{'='*60}")
    logger.info(f"  Cross-Instrument: {label}")
    logger.info(f"{'='*60}")
    logger.info(f"  Train: {len(X_train)} samples")
    logger.info(f"  Test:  {len(X_test)} samples")

    # Stage 1
    s1 = make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, max_iter=1000, solver="saga", class_weight="balanced"))
    s1.fit(X_train, y_train)
    s1_prob = s1.predict_proba(X_test)[:, 1]
    s1_pred = (s1_prob > 0.5).astype(int)

    try:
        auc = roc_auc_score(y_test, s1_prob)
    except ValueError:
        auc = float("nan")
    acc = accuracy_score(y_test, s1_pred)
    f1 = f1_score(y_test, s1_pred, average="macro")
    cm = confusion_matrix(y_test, s1_pred)

    logger.info(f"  S1 AUC:  {auc:.4f}")
    logger.info(f"  S1 Acc:  {acc:.4f}")
    logger.info(f"  S1 F1:   {f1:.4f}")
    logger.info(f"  Confusion matrix (binary):\n{cm}")

    # Stage 2
    cancer_train = ctl_train >= 0
    cancer_test = ctl_test >= 0
    s2_f1 = float("nan")
    s2_report = ""

    if cancer_train.sum() > 0 and cancer_test.sum() > 0:
        s2 = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, max_iter=1000, solver="saga", class_weight="balanced",
            multi_class="multinomial"))
        present_train = np.unique(ctl_train[cancer_train])
        if len(present_train) >= 2:
            s2.fit(X_train[cancer_train], ctl_train[cancer_train])
            s2_pred = s2.predict(X_test[cancer_test])
            s2_f1 = f1_score(ctl_test[cancer_test], s2_pred, average="macro", zero_division=0)

            target_names = [cancer_types[i] for i in sorted(np.unique(
                np.concatenate([ctl_test[cancer_test], s2_pred]))
            ) if i < len(cancer_types)]
            s2_report = classification_report(
                ctl_test[cancer_test], s2_pred,
                target_names=target_names, zero_division=0,
            )
            logger.info(f"  S2 F1 macro: {s2_f1:.4f}")
            logger.info(f"  S2 Classification Report:\n{s2_report}")

    # Per-group analysis
    logger.info(f"\n  Per-group cancer probability (mean ± std):")
    for g in sorted(set(groups_test)):
        mask = groups_test == g
        probs = s1_prob[mask]
        is_cancer = g in cancer_types
        logger.info(f"    {g:6s} (n={mask.sum():4d}): "
                    f"prob={probs.mean():.3f}±{probs.std():.3f}  "
                    f"{'[CANCER]' if is_cancer else '[NON-CA]'}")

    return {
        "label": label,
        "s1_auc": auc, "s1_acc": acc, "s1_f1": f1,
        "s2_f1": s2_f1,
        "confusion_matrix": cm.tolist(),
        "s2_report": s2_report,
        "per_group": {g: {"n": int((groups_test==g).sum()),
                          "mean_prob": float(s1_prob[groups_test==g].mean()),
                          "std_prob": float(s1_prob[groups_test==g].std())}
                      for g in sorted(set(groups_test))},
    }


# =============================================================================
# Summary Report
# =============================================================================
def generate_summary_figure(thermo_result, medical_result,
                            cross_t2m, cross_m2t, output_dir):
    """Generate a summary comparison figure."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: CV performance comparison
    ax = axes[0]
    labels = ["Thermo\n(self-CV)", "Medical\n(self-CV)"]
    aucs = [thermo_result["mean_fold_auc"], medical_result["mean_fold_auc"]]
    f1s = [thermo_result["mean_fold_s2_f1"], medical_result["mean_fold_s2_f1"]]

    x_pos = np.arange(len(labels))
    w = 0.35
    ax.bar(x_pos - w/2, aucs, w, label="S1 AUC", color="#2196F3", alpha=0.85)
    ax.bar(x_pos + w/2, f1s, w, label="S2 F1", color="#FF9800", alpha=0.85)
    for i, (a, f) in enumerate(zip(aucs, f1s)):
        ax.text(i - w/2, a + 0.01, f"{a:.3f}", ha="center", fontsize=9)
        ax.text(i + w/2, f + 0.01, f"{f:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title("Self-Instrument CV", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: Cross-instrument performance
    ax = axes[1]
    labels = ["Thermo→Medical", "Medical→Thermo"]
    aucs = [cross_t2m["s1_auc"], cross_m2t["s1_auc"]]
    f1s = [cross_t2m["s1_f1"], cross_m2t["s1_f1"]]

    x_pos = np.arange(len(labels))
    ax.bar(x_pos - w/2, aucs, w, label="S1 AUC", color="#E91E63", alpha=0.85)
    ax.bar(x_pos + w/2, f1s, w, label="S1 F1", color="#9C27B0", alpha=0.85)
    for i, (a, f) in enumerate(zip(aucs, f1s)):
        ax.text(i - w/2, a + 0.01, f"{a:.3f}", ha="center", fontsize=9)
        ax.text(i + w/2, f + 0.01, f"{f:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title("Cross-Instrument Inference", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Panel 3: Per-group cancer probability (cross T→M)
    ax = axes[2]
    groups = sorted(cross_t2m["per_group"].keys())
    means = [cross_t2m["per_group"][g]["mean_prob"] for g in groups]
    is_cancer = [g in CANCER_TYPES for g in groups]
    colors = ["#E53935" if c else "#43A047" for c in is_cancer]
    ax.barh(range(len(groups)), means, color=colors, alpha=0.8)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups)
    ax.set_xlabel("Mean Cancer Probability")
    ax.set_title("Thermo→Medical\nPer-Group Predictions", fontweight="bold")
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlim(0, 1)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "cross_instrument_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved summary figure")


# =============================================================================
# Main
# =============================================================================
def main():
    t0 = datetime.now()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load Thermo data (existing processed spectra)
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 64)
    logger.info("  Cross-Instrument SERS Analysis")
    logger.info("=" * 64)

    thermo_csv = RESULTS_DIR / "processed_spectra.csv"
    if thermo_csv.exists():
        logger.info("\n[Step 1a] Loading pre-processed Thermo spectra...")
        df_thermo = pd.read_csv(thermo_csv)
        thermo_feat_cols = [c for c in df_thermo.columns if c.startswith("x_")]
        thermo_grid = np.array([float(c.split("_")[1]) for c in thermo_feat_cols])
        logger.info(f"  Thermo: {len(df_thermo)} spectra, {len(thermo_feat_cols)} features")
        logger.info(f"  Grid: {thermo_grid[0]:.1f} - {thermo_grid[-1]:.1f} cm⁻¹")
    else:
        logger.info("\n[Step 1a] Processing Thermo raw data...")
        config = load_config("config/config.yaml")
        thermo_spectra, thermo_meta, _ = load_instrument_data(
            THERMO_DIR, config.folder_to_group, pattern="*.CSV", n_expected_reps=5)
        thermo_grid = build_grid(thermo_spectra)
        thermo_processed = preprocess_all(thermo_spectra, thermo_grid)
        df_thermo = build_dataframe(thermo_processed, thermo_grid)
        thermo_feat_cols = [c for c in df_thermo.columns if c.startswith("x_")]
        logger.info(f"  Thermo: {len(df_thermo)} spectra")

    # ------------------------------------------------------------------
    # 2. Load & Preprocess Medical data
    # ------------------------------------------------------------------
    logger.info("\n[Step 1b] Loading & preprocessing Medical instrument data...")
    medical_spectra, medical_meta, medical_failed = load_instrument_data(
        MEDICAL_DIR, MEDICAL_FOLDER_MAP, pattern="*.txt", n_expected_reps=6)

    medical_grid = build_grid(medical_spectra)
    logger.info(f"  Medical grid: {medical_grid[0]:.1f} - {medical_grid[-1]:.1f} cm⁻¹ ({len(medical_grid)} pts)")

    medical_processed = preprocess_all(medical_spectra, medical_grid)
    df_medical = build_dataframe(medical_processed, medical_grid)
    medical_feat_cols = [c for c in df_medical.columns if c.startswith("x_")]
    logger.info(f"  Medical: {len(df_medical)} spectra, {len(medical_feat_cols)} features")

    # Save processed medical spectra
    df_medical.to_csv(OUTPUT_DIR / "processed_spectra_medical.csv", index=False)
    medical_meta.to_csv(OUTPUT_DIR / "metadata_medical.csv", index=False)

    # ------------------------------------------------------------------
    # 3. Spectral Comparison (400-2200)
    # ------------------------------------------------------------------
    logger.info("\n[Step 2] Comparing spectra between instruments...")

    # Build processed dicts for comparison plot
    thermo_proc_dict = {}
    for _, row in df_thermo.iterrows():
        key = (row["group"], str(row["sample_id"]), int(row["replicate"]))
        thermo_proc_dict[key] = row[thermo_feat_cols].values.astype(float)

    plot_spectral_comparison(
        thermo_proc_dict, medical_processed,
        thermo_grid, medical_grid, OUTPUT_DIR)

    # ------------------------------------------------------------------
    # 4. Train on each instrument (self-CV) — individual spectra, no aggregation
    # ------------------------------------------------------------------
    logger.info("\n[Step 3] Training models (5-fold CV, individual spectra)...")

    # Disambiguate sample_ids before group merge (CPAN_4 ≠ YPAN_4)
    df_medical_mapped = df_medical.copy()
    needs_prefix = df_medical_mapped["group"].isin(["YPAN", "YNOR"])
    df_medical_mapped.loc[needs_prefix, "sample_id"] = (
        df_medical_mapped.loc[needs_prefix, "group"] + "_" + df_medical_mapped.loc[needs_prefix, "sample_id"].astype(str)
    )
    df_medical_mapped["group"] = df_medical_mapped["group"].replace({
        "CPAN": "PAN", "YPAN": "PAN", "YNOR": "NOR",
    })

    # Thermo also has CPAN/YPAN/YNOR/SPAN — map consistently
    df_thermo_mapped = df_thermo.copy()
    needs_prefix = df_thermo_mapped["group"].isin(["YPAN", "YNOR"])
    df_thermo_mapped.loc[needs_prefix, "sample_id"] = (
        df_thermo_mapped.loc[needs_prefix, "group"] + "_" + df_thermo_mapped.loc[needs_prefix, "sample_id"].astype(str)
    )
    df_thermo_mapped["group"] = df_thermo_mapped["group"].replace({
        "CPAN": "PAN", "YPAN": "PAN", "YNOR": "NOR",
    })

    thermo_result = train_model_on_data(
        df_thermo_mapped, thermo_feat_cols,
        CANCER_TYPES, NON_CANCER_GROUPS,
        n_splits=5, label="Thermo (self-CV)")

    medical_result = train_model_on_data(
        df_medical_mapped, medical_feat_cols,
        CANCER_TYPES, NON_CANCER_GROUPS,
        n_splits=5, label="Medical (self-CV)")

    # ------------------------------------------------------------------
    # 5. Cross-instrument inference
    # ------------------------------------------------------------------
    logger.info("\n[Step 4] Cross-instrument inference...")

    # Need common feature space — interpolate to shared grid
    shared_grid = np.linspace(
        max(thermo_grid[0], medical_grid[0]),
        min(thermo_grid[-1], medical_grid[-1]),
        933)

    def reindex_to_grid(df, old_feat_cols, old_grid, new_grid):
        """Interpolate features to new grid (vectorized)."""
        new_feat_cols = [f"x_{wn:.2f}" for wn in new_grid]
        X_old = df[old_feat_cols].values
        # Vectorized interpolation using numpy
        X_new = np.zeros((len(X_old), len(new_grid)))
        for i in range(len(X_old)):
            X_new[i] = np.interp(new_grid, old_grid, X_old[i])
        # Build DataFrame at once (avoid fragmentation)
        meta_cols = ["group", "sample_id"]
        if "replicate" in df.columns:
            meta_cols.append("replicate")
        df_new = pd.concat([
            df[meta_cols].reset_index(drop=True),
            pd.DataFrame(X_new, columns=new_feat_cols),
        ], axis=1)
        return df_new, new_feat_cols

    df_thermo_shared, shared_feat = reindex_to_grid(
        df_thermo_mapped, thermo_feat_cols, thermo_grid, shared_grid)
    df_medical_shared, _ = reindex_to_grid(
        df_medical_mapped, medical_feat_cols, medical_grid, shared_grid)

    # Re-train on shared grid for fair comparison
    thermo_shared_result = train_model_on_data(
        df_thermo_shared, shared_feat,
        CANCER_TYPES, NON_CANCER_GROUPS,
        n_splits=5, label="Thermo (shared grid)")

    medical_shared_result = train_model_on_data(
        df_medical_shared, shared_feat,
        CANCER_TYPES, NON_CANCER_GROUPS,
        n_splits=5, label="Medical (shared grid)")

    # Thermo → Medical
    cross_t2m = cross_instrument_inference(
        thermo_shared_result, df_medical_shared, shared_feat,
        CANCER_TYPES, NON_CANCER_GROUPS,
        label="Thermo → Medical")

    # Medical → Thermo
    cross_m2t = cross_instrument_inference(
        medical_shared_result, df_thermo_shared, shared_feat,
        CANCER_TYPES, NON_CANCER_GROUPS,
        label="Medical → Thermo")

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    logger.info("\n[Step 5] Generating summary...")
    generate_summary_figure(
        thermo_shared_result, medical_shared_result,
        cross_t2m, cross_m2t, OUTPUT_DIR)

    # Save JSON report
    report = {
        "timestamp": datetime.now().isoformat(),
        "duration_sec": (datetime.now() - t0).total_seconds(),
        "thermo_self_cv": {
            "n_samples": thermo_shared_result["n_samples"],
            "s1_auc": thermo_shared_result["mean_fold_auc"],
            "s2_f1": thermo_shared_result["mean_fold_s2_f1"],
        },
        "medical_self_cv": {
            "n_samples": medical_shared_result["n_samples"],
            "s1_auc": medical_shared_result["mean_fold_auc"],
            "s2_f1": medical_shared_result["mean_fold_s2_f1"],
        },
        "cross_thermo_to_medical": {
            "s1_auc": cross_t2m["s1_auc"],
            "s1_acc": cross_t2m["s1_acc"],
            "s1_f1": cross_t2m["s1_f1"],
            "s2_f1": cross_t2m["s2_f1"],
            "per_group": cross_t2m["per_group"],
        },
        "cross_medical_to_thermo": {
            "s1_auc": cross_m2t["s1_auc"],
            "s1_acc": cross_m2t["s1_acc"],
            "s1_f1": cross_m2t["s1_f1"],
            "s2_f1": cross_m2t["s2_f1"],
            "per_group": cross_m2t["per_group"],
        },
        "cancer_types": list(CANCER_TYPES),
        "non_cancer_groups": list(NON_CANCER_GROUPS),
        "preprocessing": PREP_PARAMS,
    }
    report["preprocessing"]["trim_region"] = list(report["preprocessing"]["trim_region"])

    with open(OUTPUT_DIR / "cross_instrument_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Print final summary table
    logger.info("\n" + "=" * 64)
    logger.info("  FINAL RESULTS SUMMARY")
    logger.info("=" * 64)
    logger.info(f"  {'Experiment':<25} {'S1 AUC':>8} {'S1 F1':>8} {'S2 F1':>8}")
    logger.info(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}")
    logger.info(f"  {'Thermo self-CV':<25} {thermo_shared_result['mean_fold_auc']:>8.4f} "
                f"{thermo_shared_result['overall_f1']:>8.4f} {thermo_shared_result['mean_fold_s2_f1']:>8.4f}")
    logger.info(f"  {'Medical self-CV':<25} {medical_shared_result['mean_fold_auc']:>8.4f} "
                f"{medical_shared_result['overall_f1']:>8.4f} {medical_shared_result['mean_fold_s2_f1']:>8.4f}")
    logger.info(f"  {'Thermo → Medical':<25} {cross_t2m['s1_auc']:>8.4f} "
                f"{cross_t2m['s1_f1']:>8.4f} {cross_t2m['s2_f1']:>8.4f}")
    logger.info(f"  {'Medical → Thermo':<25} {cross_m2t['s1_auc']:>8.4f} "
                f"{cross_m2t['s1_f1']:>8.4f} {cross_m2t['s2_f1']:>8.4f}")
    logger.info("=" * 64)
    logger.info(f"\n  Output: {OUTPUT_DIR}")
    logger.info(f"  Duration: {(datetime.now() - t0).total_seconds():.1f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
