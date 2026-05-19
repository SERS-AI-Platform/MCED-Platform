#!/usr/bin/env python3
"""
Peak Ratio Feature Experiment
==============================

Tests whether biologically-motivated peak ratio features improve
cancer detection and type identification beyond full-spectrum LR.

Ablation conditions:
  1. Full spectrum only (935 features) — baseline
  2. Peak intensities only (~15 features from known peaks)
  3. Peak ratios only (~10 ratio features)
  4. Full spectrum + peak ratios (935 + 10 = hybrid)

Evaluation: 5-fold StratifiedGroupKFold, LR(C=1.0, saga, balanced)
Metrics: Stage 1 AUC/F1 (cancer vs non-cancer), Stage 2 macro F1 (type ID)

Usage:
    python scripts/analysis/peak_ratio_feature_experiment.py
"""

from __future__ import annotations

import sys
import logging
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.integrate import trapezoid
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, f1_score

import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sers.config import RESULTS_DIR

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

CANCER_TYPES = ["PRO", "BRE", "OVA", "LUN", "CRC", "PAN", "BLC"]
NON_CANCER = ["NOR", "DIA", "HBP", "H.D."]

# Aliases: map raw group names to canonical names
GROUP_ALIASES = {
    "CPAN": "PAN",
    "YPAN": "PAN",
    "YNOR": "NOR",
}
# Groups to remove entirely
EXCLUDE_GROUPS = {"SPAN", "TBN"}

KNOWN_PEAKS = {
    618: "C-S stretch (Cysteine/Adenine)",
    683: "Creatinine/ring",
    724: "Adenine ring breathing",
    795: "Hippuric acid",
    849: "Tyr/Trp Fermi resonance",
    895: "C-C stretch",
    934: "C-C protein",
    999: "Phenylalanine ring breathing",
    1148: "C-N/C-O-C stretch",
    1231: "Amide III",
    1293: "Amide III/CH2 twist",
    1352: "CH deformation/Trp",
    1449: "CH2 deformation",
    1597: "C=C/Purine ring",
    1651: "Amide I",
}

# Biologically meaningful peak ratios
PEAK_RATIOS = [
    (999, 683, "Phe/Creatinine"),        # Amino acid vs kidney baseline
    (849, 999, "Tyr/Phe"),               # Tyrosine vs Phenylalanine
    (724, 683, "Adenine/Creatinine"),     # Nucleobase vs kidney
    (1352, 1449, "Trp_CH/CH2"),          # Tryptophan vs lipid
    (1597, 1651, "Purine/AmideI"),       # Nucleic acid vs protein
    (618, 683, "Cysteine/Creatinine"),   # Thiol vs kidney
    (795, 683, "Hippuric/Creatinine"),   # Gut metabolite vs kidney
    (1148, 999, "Glycogen/Phe"),         # Sugar vs amino acid
    (895, 683, "CC_stretch/Creatinine"), # General
    (1231, 1651, "AmideIII/AmideI"),     # Protein secondary structure
]

SEED = 42
N_SPLITS = 5
PEAK_WINDOW = 10  # ±10 cm⁻¹ window around each known peak


# =============================================================================
# 1. Data Loading
# =============================================================================

def load_data(csv_path=None):
    """Load processed spectra and return df, feature columns, wavenumber array."""
    if csv_path is None:
        csv_path = RESULTS_DIR / "processed_spectra.csv"
    df = pd.read_csv(csv_path)
    feat_cols = [c for c in df.columns if c.startswith("x_")]
    wavenumbers = np.array([float(c.replace("x_", "")) for c in feat_cols])
    return df, feat_cols, wavenumbers


def preprocess_groups(df):
    """Apply group aliases, remove excluded groups."""
    df = df.copy()
    df["group"] = df["group"].replace(GROUP_ALIASES)
    df = df[~df["group"].isin(EXCLUDE_GROUPS)].reset_index(drop=True)
    return df


def aggregate_mean(df, feat_cols):
    """Mean aggregation per sample_id (average replicates)."""
    meta_cols = ["group", "sample_id"]
    return df.groupby(meta_cols, as_index=False).agg(
        {**{c: "mean" for c in feat_cols}, "replicate": "count"}
    )


# =============================================================================
# 2. Peak Feature Engineering
# =============================================================================

def find_nearest_idx(wavenumbers, target_cm):
    """Find the index of the wavenumber closest to target_cm."""
    return np.argmin(np.abs(wavenumbers - target_cm))


def get_peak_window(wavenumbers, center_cm, half_width_cm=PEAK_WINDOW):
    """Return (lo_idx, hi_idx) for the ±half_width_cm window around center_cm."""
    center_idx = find_nearest_idx(wavenumbers, center_cm)
    lo_wn = center_cm - half_width_cm
    hi_wn = center_cm + half_width_cm
    lo_idx = find_nearest_idx(wavenumbers, lo_wn)
    hi_idx = find_nearest_idx(wavenumbers, hi_wn) + 1  # inclusive
    lo_idx = max(0, lo_idx)
    hi_idx = min(len(wavenumbers), hi_idx)
    return lo_idx, hi_idx


def extract_peak_intensities(X, wavenumbers):
    """
    For each known peak position, extract from ±10 cm⁻¹ window:
      1. Max intensity in window
      2. Area (trapz) in window

    Returns: feature matrix (n_samples, n_peaks*2), feature names
    """
    features = []
    names = []

    for peak_cm, label in KNOWN_PEAKS.items():
        lo, hi = get_peak_window(wavenumbers, peak_cm)
        window_wn = wavenumbers[lo:hi]

        # Max intensity in window
        max_intensity = X[:, lo:hi].max(axis=1)
        features.append(max_intensity)
        names.append(f"peak_{peak_cm}_max")

        # Area under window (trapezoidal)
        areas = np.array([
            trapezoid(X[s, lo:hi], window_wn) if len(window_wn) > 1 else X[s, lo]
            for s in range(len(X))
        ])
        features.append(areas)
        names.append(f"peak_{peak_cm}_area")

    X_peaks = np.column_stack(features)
    return X_peaks, names


def extract_peak_ratios(X, wavenumbers):
    """
    Compute biologically meaningful peak ratios.
    Each ratio = max_intensity(peak_A window) / max_intensity(peak_B window).

    Returns: feature matrix (n_samples, n_ratios), feature names
    """
    features = []
    names = []

    for peak_a, peak_b, label in PEAK_RATIOS:
        lo_a, hi_a = get_peak_window(wavenumbers, peak_a)
        lo_b, hi_b = get_peak_window(wavenumbers, peak_b)

        intensity_a = X[:, lo_a:hi_a].max(axis=1)
        intensity_b = X[:, lo_b:hi_b].max(axis=1)

        ratio = intensity_a / (intensity_b + 1e-10)
        features.append(ratio)
        names.append(f"ratio_{label}")

    X_ratios = np.column_stack(features)
    return X_ratios, names


def detect_data_driven_peaks(X, wavenumbers):
    """
    Apply scipy find_peaks on mean spectrum for exploratory reporting.
    Not used for features, but logged for comparison with known peaks.
    """
    mean_spectrum = X.mean(axis=0)
    peaks, props = find_peaks(
        mean_spectrum,
        prominence=0.02,
        distance=15,
        width=3,
    )
    detected = [(wavenumbers[p], mean_spectrum[p], props["prominences"][i])
                for i, p in enumerate(peaks)]
    return detected


# =============================================================================
# 3. Model & Evaluation
# =============================================================================

def make_lr_pipeline():
    """Create a fresh LR pipeline (StandardScaler + LR)."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0, max_iter=1000, solver="saga",
            class_weight="balanced", random_state=SEED,
        ),
    )


def run_two_stage_cv(X, groups, sample_ids, condition_name, n_splits=N_SPLITS):
    """
    Run 5-fold StratifiedGroupKFold with two-stage evaluation.

    Returns dict with per-fold and summary metrics.
    """
    cancer_set = set(CANCER_TYPES)
    non_cancer_set = set(NON_CANCER)
    valid_groups = cancer_set | non_cancer_set

    # Filter to valid groups only
    valid_mask = np.isin(groups, list(valid_groups))
    X = X[valid_mask]
    groups = groups[valid_mask]
    sample_ids = sample_ids[valid_mask]

    binary_labels = np.array([1 if g in cancer_set else 0 for g in groups])
    cancer_type_labels = np.array([
        CANCER_TYPES.index(g) if g in cancer_set else -1 for g in groups
    ])

    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(
        skf.split(X, binary_labels, groups=sample_ids)
    ):
        X_train, X_test = X[train_idx], X[test_idx]
        yb_train, yb_test = binary_labels[train_idx], binary_labels[test_idx]
        yc_train, yc_test = cancer_type_labels[train_idx], cancer_type_labels[test_idx]

        # --- Stage 1: Cancer Detection (binary) ---
        model_s1 = make_lr_pipeline()
        model_s1.fit(X_train, yb_train)

        s1_prob = model_s1.predict_proba(X_test)[:, 1]
        s1_pred = model_s1.predict(X_test)

        try:
            det_auc = roc_auc_score(yb_test, s1_prob)
        except ValueError:
            det_auc = np.nan

        det_f1 = f1_score(yb_test, s1_pred, average="binary", zero_division=0)

        # --- Stage 2: Cancer Type ID (cancer samples only) ---
        cancer_train_mask = yc_train >= 0
        cancer_test_mask = yc_test >= 0

        type_f1 = np.nan
        if cancer_train_mask.sum() > 0 and cancer_test_mask.sum() > 0:
            present_classes = sorted(set(yc_train[cancer_train_mask]))
            if len(present_classes) >= 2:
                label_map = {c: i for i, c in enumerate(present_classes)}
                inv_map = {i: c for c, i in label_map.items()}

                yc_train_local = np.array([label_map[c] for c in yc_train[cancer_train_mask]])
                yc_test_local = np.array([label_map.get(c, -1) for c in yc_test[cancer_test_mask]])

                seen_mask = yc_test_local >= 0
                if seen_mask.sum() > 0:
                    model_s2 = make_lr_pipeline()
                    model_s2.fit(X_train[cancer_train_mask], yc_train_local)
                    yc_pred = model_s2.predict(X_test[cancer_test_mask][seen_mask])

                    type_f1 = f1_score(
                        yc_test_local[seen_mask], yc_pred,
                        average="macro", zero_division=0,
                    )

        fold_results.append({
            "fold": fold_idx,
            "det_auc": det_auc,
            "det_f1": det_f1,
            "type_f1": type_f1,
        })

    fold_df = pd.DataFrame(fold_results)

    return {
        "condition": condition_name,
        "n_features": X.shape[1],
        "n_samples": len(X),
        "det_auc_mean": fold_df["det_auc"].mean(),
        "det_auc_std": fold_df["det_auc"].std(),
        "det_f1_mean": fold_df["det_f1"].mean(),
        "det_f1_std": fold_df["det_f1"].std(),
        "type_f1_mean": fold_df["type_f1"].mean(),
        "type_f1_std": fold_df["type_f1"].std(),
        "fold_details": fold_df,
    }


# =============================================================================
# 4. Feature Importance Analysis
# =============================================================================

def get_feature_importance(X, y_binary, y_type, groups, sample_ids, feature_names):
    """
    Train on all data and return LR coefficients as importance proxy.
    Returns importance for both Stage 1 and Stage 2.
    """
    cancer_set = set(CANCER_TYPES)
    non_cancer_set = set(NON_CANCER)
    valid_mask = np.isin(groups, list(cancer_set | non_cancer_set))
    X = X[valid_mask]
    y_binary = y_binary[valid_mask]
    y_type = y_type[valid_mask]

    # Stage 1 importance
    model_s1 = make_lr_pipeline()
    model_s1.fit(X, y_binary)
    s1_coef = np.abs(model_s1.named_steps["logisticregression"].coef_[0])

    # Stage 2 importance (cancer samples only, mean absolute coef across classes)
    cancer_mask = y_type >= 0
    if cancer_mask.sum() > 0:
        model_s2 = make_lr_pipeline()
        model_s2.fit(X[cancer_mask], y_type[cancer_mask])
        s2_coef = np.abs(model_s2.named_steps["logisticregression"].coef_).mean(axis=0)
    else:
        s2_coef = np.zeros(X.shape[1])

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "s1_importance": s1_coef,
        "s2_importance": s2_coef,
    }).sort_values("s2_importance", ascending=False)

    return importance_df


# =============================================================================
# 5. Visualization
# =============================================================================

def plot_comparison_bar(results_list, out_path):
    """Bar chart comparing 4 conditions across 3 metrics."""
    conditions = [r["condition"] for r in results_list]
    metrics = ["det_auc", "det_f1", "type_f1"]
    metric_labels = ["Detection AUC", "Detection F1", "Type ID F1 (macro)"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    x = np.arange(len(conditions))
    bar_width = 0.6
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

    for ax, metric, label in zip(axes, metrics, metric_labels):
        means = [r[f"{metric}_mean"] for r in results_list]
        stds = [r[f"{metric}_std"] for r in results_list]

        bars = ax.bar(x, means, bar_width, yerr=stds, capsize=4,
                      color=colors[:len(conditions)], edgecolor="black", linewidth=0.5,
                      alpha=0.85)
        ax.set_ylabel(label, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=25, ha="right", fontsize=9)
        ax.set_title(label, fontweight="bold", fontsize=12)

        # Add value labels on bars
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.005,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        # Set y-axis to reasonable range
        ymin = max(0, min(means) - 0.1)
        ymax = min(1.0, max(means) + max(stds) + 0.05)
        ax.set_ylim(ymin, ymax)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Peak Ratio Feature Experiment — 4-Condition Ablation",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  Saved comparison chart: {out_path}")


def plot_detected_peaks(detected_peaks, wavenumbers, mean_spectrum, out_path):
    """Plot mean spectrum with detected peaks annotated."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(wavenumbers, mean_spectrum, "k-", linewidth=0.8, alpha=0.8)

    # Mark known peaks
    for peak_cm, label in KNOWN_PEAKS.items():
        idx = find_nearest_idx(wavenumbers, peak_cm)
        ax.axvline(wavenumbers[idx], color="blue", alpha=0.2, linewidth=0.8)
        ax.annotate(f"{peak_cm}", (wavenumbers[idx], mean_spectrum[idx]),
                    textcoords="offset points", xytext=(0, 10),
                    fontsize=6, ha="center", color="blue", rotation=45)

    # Mark data-driven peaks
    for wn, intensity, prom in detected_peaks:
        ax.plot(wn, intensity, "rv", markersize=5, alpha=0.6)

    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.set_title("Mean Spectrum with Known Peak Positions (blue) and Detected Peaks (red)")
    ax.invert_xaxis()
    ax.grid(alpha=0.2)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  Saved peak detection plot: {out_path}")


# =============================================================================
# 6. Main
# =============================================================================

def main():
    t0 = datetime.now()
    out_dir = RESULTS_DIR / "training" / "peak_ratio_experiment"
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(out_dir / "experiment.log", mode="w", encoding="utf-8"),
        ],
    )

    logger.info("=" * 64)
    logger.info("  Peak Ratio Feature Experiment")
    logger.info("=" * 64)

    # ------------------------------------------------------------------
    # Step 1: Load & preprocess data
    # ------------------------------------------------------------------
    logger.info("\n[1] Loading data...")
    df, feat_cols, wavenumbers = load_data()
    df = preprocess_groups(df)

    logger.info(f"  Groups after alias resolution: {sorted(df['group'].unique())}")
    logger.info(f"  Group counts:\n{df.groupby('group')['sample_id'].nunique().to_string()}")

    logger.info("\n[2] Aggregating replicates (mean)...")
    df_agg = aggregate_mean(df, feat_cols)

    X_full = df_agg[feat_cols].values
    groups = df_agg["group"].values
    sample_ids = df_agg["sample_id"].values

    logger.info(f"  Samples: {len(X_full)}")
    logger.info(f"  Wavenumber range: {wavenumbers[0]:.1f} - {wavenumbers[-1]:.1f} cm^-1")
    logger.info(f"  Full spectrum features: {X_full.shape[1]}")

    # ------------------------------------------------------------------
    # Step 2: Peak detection (exploratory)
    # ------------------------------------------------------------------
    logger.info("\n[3] Peak detection on mean spectrum...")
    mean_spectrum = X_full.mean(axis=0)
    detected_peaks = detect_data_driven_peaks(X_full, wavenumbers)
    logger.info(f"  Detected {len(detected_peaks)} data-driven peaks:")
    for wn, intensity, prom in detected_peaks:
        # Check overlap with known peaks
        nearest_known = min(KNOWN_PEAKS.keys(), key=lambda k: abs(k - wn))
        overlap = "*" if abs(nearest_known - wn) < 15 else " "
        logger.info(f"    {overlap} {wn:.1f} cm^-1 (I={intensity:.4f}, prom={prom:.4f})"
                     f"  [nearest known: {nearest_known} cm^-1]")

    plot_detected_peaks(detected_peaks, wavenumbers, mean_spectrum,
                        out_dir / "peak_detection.png")

    # ------------------------------------------------------------------
    # Step 3: Feature extraction
    # ------------------------------------------------------------------
    logger.info("\n[4] Extracting peak features...")

    # Peak intensities (max + area per known peak)
    X_peak_int, peak_int_names = extract_peak_intensities(X_full, wavenumbers)
    logger.info(f"  Peak intensity features: {X_peak_int.shape[1]} ({len(KNOWN_PEAKS)} peaks x 2)")

    # Peak ratios
    X_ratios, ratio_names = extract_peak_ratios(X_full, wavenumbers)
    logger.info(f"  Peak ratio features: {X_ratios.shape[1]}")
    for name in ratio_names:
        logger.info(f"    {name}")

    # Hybrid: full spectrum + peak ratios
    X_hybrid = np.hstack([X_full, X_ratios])
    hybrid_names = feat_cols + ratio_names
    logger.info(f"  Hybrid features: {X_hybrid.shape[1]} (full {X_full.shape[1]} + ratios {X_ratios.shape[1]})")

    # Check for NaN/Inf in all feature sets
    for label, X_check in [("peak_int", X_peak_int), ("ratios", X_ratios), ("hybrid", X_hybrid)]:
        n_nan = np.isnan(X_check).sum()
        n_inf = np.isinf(X_check).sum()
        if n_nan > 0 or n_inf > 0:
            logger.warning(f"  {label}: NaN={n_nan}, Inf={n_inf} -- replacing with 0")
            X_check = np.nan_to_num(X_check, nan=0.0, posinf=0.0, neginf=0.0)

    # ------------------------------------------------------------------
    # Step 4: Run 4-condition ablation
    # ------------------------------------------------------------------
    conditions = [
        ("Full Spectrum", X_full),
        ("Peak Intensities", X_peak_int),
        ("Peak Ratios", X_ratios),
        ("Full + Ratios", X_hybrid),
    ]

    logger.info("\n[5] Running 4-condition ablation (5-fold CV)...")
    all_results = []

    for cond_name, X_cond in conditions:
        logger.info(f"\n  --- {cond_name} ({X_cond.shape[1]} features) ---")
        result = run_two_stage_cv(X_cond, groups, sample_ids, cond_name)
        all_results.append(result)

        logger.info(f"    Det AUC:   {result['det_auc_mean']:.4f} +/- {result['det_auc_std']:.4f}")
        logger.info(f"    Det F1:    {result['det_f1_mean']:.4f} +/- {result['det_f1_std']:.4f}")
        logger.info(f"    Type F1:   {result['type_f1_mean']:.4f} +/- {result['type_f1_std']:.4f}")

        fold_df = result["fold_details"]
        for _, row in fold_df.iterrows():
            logger.info(
                f"      Fold {int(row['fold'])}: AUC={row['det_auc']:.4f}  "
                f"F1={row['det_f1']:.4f}  TypeF1={row['type_f1']:.4f}"
            )

    # ------------------------------------------------------------------
    # Step 5: Summary table
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 64)
    logger.info("  RESULTS SUMMARY")
    logger.info("=" * 64)

    summary_rows = []
    for r in all_results:
        summary_rows.append({
            "Condition": r["condition"],
            "N_Features": r["n_features"],
            "Det_AUC": f"{r['det_auc_mean']:.4f}+/-{r['det_auc_std']:.4f}",
            "Det_F1": f"{r['det_f1_mean']:.4f}+/-{r['det_f1_std']:.4f}",
            "Type_F1": f"{r['type_f1_mean']:.4f}+/-{r['type_f1_std']:.4f}",
        })

    summary_df = pd.DataFrame(summary_rows)
    logger.info(f"\n{summary_df.to_string(index=False)}")

    # Save summary CSV
    csv_path = out_dir / "results_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    logger.info(f"\n  Saved summary CSV: {csv_path}")

    # Save per-fold details CSV
    fold_rows = []
    for r in all_results:
        for _, row in r["fold_details"].iterrows():
            fold_rows.append({
                "condition": r["condition"],
                "n_features": r["n_features"],
                "fold": int(row["fold"]),
                "det_auc": row["det_auc"],
                "det_f1": row["det_f1"],
                "type_f1": row["type_f1"],
            })
    fold_csv = out_dir / "fold_details.csv"
    pd.DataFrame(fold_rows).to_csv(fold_csv, index=False)
    logger.info(f"  Saved fold details CSV: {fold_csv}")

    # ------------------------------------------------------------------
    # Step 6: Comparison chart
    # ------------------------------------------------------------------
    logger.info("\n[6] Generating comparison chart...")
    plot_comparison_bar(all_results, out_dir / "comparison_chart.png")

    # ------------------------------------------------------------------
    # Step 7: Feature importance (if hybrid >= baseline)
    # ------------------------------------------------------------------
    baseline_type_f1 = all_results[0]["type_f1_mean"]
    hybrid_type_f1 = all_results[3]["type_f1_mean"]

    if hybrid_type_f1 >= baseline_type_f1:
        logger.info("\n[7] Hybrid >= Baseline -- analyzing peak ratio importance...")

        cancer_set = set(CANCER_TYPES)
        non_cancer_set = set(NON_CANCER)
        valid_mask = np.isin(groups, list(cancer_set | non_cancer_set))

        y_binary = np.array([1 if g in cancer_set else 0 for g in groups])
        y_type = np.array([
            CANCER_TYPES.index(g) if g in cancer_set else -1 for g in groups
        ])

        importance_df = get_feature_importance(
            X_hybrid, y_binary, y_type, groups, sample_ids, hybrid_names,
        )

        # Show top-ranked ratio features
        ratio_importance = importance_df[importance_df["feature"].str.startswith("ratio_")]
        logger.info(f"\n  Peak ratio feature importance (sorted by Type ID):")
        for _, row in ratio_importance.head(len(PEAK_RATIOS)).iterrows():
            logger.info(
                f"    {row['feature']:30s}  S1={row['s1_importance']:.4f}  S2={row['s2_importance']:.4f}"
            )

        # Save importance
        imp_path = out_dir / "feature_importance.csv"
        importance_df.to_csv(imp_path, index=False)
        logger.info(f"  Saved feature importance: {imp_path}")
    else:
        logger.info(f"\n[7] Hybrid ({hybrid_type_f1:.4f}) < Baseline ({baseline_type_f1:.4f})")
        logger.info("  Peak ratios did not improve Type ID F1. Skipping importance analysis.")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    elapsed = datetime.now() - t0
    logger.info(f"\nDone in {elapsed.total_seconds():.1f}s")
    logger.info(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
