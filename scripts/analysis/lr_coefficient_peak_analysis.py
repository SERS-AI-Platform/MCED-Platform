"""
LR Coefficient Visualization with Peak Assignment Overlay

Four analyses:
  1. Stage 1 Binary Coefficient Spectrum — cancer vs non-cancer
  2. Stage 2 Per-Cancer-Type Coefficient Spectra — OvR subplots
  3. Coefficient × Peak Assignment Cross-Reference Table (CSV)
  4. Cancer-Discriminative Peak Heatmap — metabolite × cancer matrix

Usage:
    python scripts/analysis/lr_coefficient_peak_analysis.py
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold

import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sers.visualization._common import (
    apply_publication_style, save_figure, ensure_output_dir,
    extract_feature_columns, feature_axis_from_names,
    PAPER_TITLE_SIZE, PAPER_LABEL_SIZE, PAPER_TICK_SIZE, PAPER_LEGEND_SIZE,
    PAPER_ANNOTATION_SIZE, PAPER_LINEWIDTH, DEFAULT_DPI,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
CANCER_TYPES = ["PRO", "BRE", "OVA", "LUN", "CRC", "PAN", "BLC"]
NON_CANCER = ["NOR", "DIA", "HBP", "H.D."]
GROUP_ALIASES = {"PAN": ["CPAN", "YPAN"], "NOR": ["NOR", "YNOR"]}
SEED = 42
TOP_N_BINARY = 20
TOP_N_PER_CANCER = 15
TOLERANCE_CM1 = 15

OUTPUT_DIR = PROJECT_ROOT / "results" / "training" / "peak_analysis"

CANCER_COLORS = {
    "PRO": "#1976D2",  # blue
    "BRE": "#E91E63",  # pink
    "OVA": "#9C27B0",  # purple
    "LUN": "#4CAF50",  # green
    "CRC": "#EF5350",  # red
    "PAN": "#FFA726",  # orange
    "BLC": "#00ACC1",  # cyan
}

# ── Band Assignments ──────────────────────────────────────────────────────────
# Combined from cross_reference_peaks.py and literature review
BAND_ASSIGNMENTS = {
    "~442 (Uric acid)": {"center": 442, "range": (432, 452), "metabolites": ["Uric acid"], "vibration": "NH2 deformation"},
    "~527 (Urea)": {"center": 527, "range": (520, 535), "metabolites": ["Urea"], "vibration": "Ring deformation"},
    "~560 (Hypoxanthine)": {"center": 560, "range": (550, 570), "metabolites": ["Hypoxanthine"], "vibration": "Ring vibration"},
    "~605 (Creatinine)": {"center": 605, "range": (595, 615), "metabolites": ["Creatinine"], "vibration": "Ring deformation"},
    "~618 (C-S stretch)": {"center": 618, "range": (610, 630), "metabolites": ["Cysteine", "Adenine", "Betaine", "Cholesterol", "Arginine", "Phenylalanine", "Hippuric acid"], "vibration": "C-S stretch"},
    "~640 (Uric acid)": {"center": 640, "range": (630, 650), "metabolites": ["Uric acid"], "vibration": "Skeletal ring deformation"},
    "~683 (Creatinine)": {"center": 683, "range": (670, 695), "metabolites": ["Creatinine", "Guanine", "Purine", "Hippuric acid", "Tyrosine"], "vibration": "C-S stretch / ring breathing"},
    "~724 (Adenine)": {"center": 724, "range": (715, 735), "metabolites": ["Adenine", "Hypoxanthine", "Hippuric acid", "Benzoic acid"], "vibration": "Ring breathing"},
    "~754 (Tryptophan)": {"center": 754, "range": (745, 765), "metabolites": ["Tryptophan"], "vibration": "Ring breathing"},
    "~795 (Hippuric)": {"center": 795, "range": (785, 805), "metabolites": ["Hippuric acid", "Kynurenine", "Uric acid", "Creatinine"], "vibration": "CH2 rocking"},
    "~849 (Tyr/Trp)": {"center": 849, "range": (838, 860), "metabolites": ["Tyrosine", "Tryptophan", "Creatinine", "Cysteine", "Hypoxanthine"], "vibration": "Ring vibration"},
    "~895 (C-C stretch)": {"center": 895, "range": (885, 910), "metabolites": ["Hippuric acid", "TMAO", "Uric acid", "Cysteine", "Creatinine"], "vibration": "C-C stretch"},
    "~934 (C-C protein)": {"center": 934, "range": (925, 945), "metabolites": ["Multiple (non-specific)"], "vibration": "C-C stretch"},
    "~999 (Phe ring)": {"center": 1004, "range": (990, 1015), "metabolites": ["Phenylalanine", "Uric acid", "Urea"], "vibration": "Symmetric ring breathing"},
    "~1050 (Glycogen)": {"center": 1050, "range": (1040, 1060), "metabolites": ["Glycogen", "Creatinine"], "vibration": "C-O stretch"},
    "~1086 (Uric acid)": {"center": 1086, "range": (1075, 1095), "metabolites": ["Uric acid", "Ethanolamine"], "vibration": "NH3, CH2"},
    "~1130 (Uric acid)": {"center": 1130, "range": (1120, 1140), "metabolites": ["Uric acid", "Protein"], "vibration": "C-N stretch"},
    "~1148 (C-N/C-O-C)": {"center": 1148, "range": (1138, 1165), "metabolites": ["Glycogen", "Glucose", "Xylose", "Cysteine", "Urea"], "vibration": "C-N stretch"},
    "~1210 (Nucleic acid)": {"center": 1210, "range": (1200, 1225), "metabolites": ["Nucleic acid", "Tryptophan", "Amide III"], "vibration": "C-C6H5 stretch"},
    "~1231 (Amide III)": {"center": 1231, "range": (1220, 1250), "metabolites": ["Tryptophan", "Taurine", "Kynurenine", "Threonine", "Isoleucine"], "vibration": "Amide III"},
    "~1293 (CH2)": {"center": 1293, "range": (1283, 1305), "metabolites": ["O-Acetylcarnitine", "Cholesterol", "Fatty acids"], "vibration": "CH2 twist"},
    "~1340 (Nucleic acid)": {"center": 1340, "range": (1330, 1365), "metabolites": ["Adenine", "Guanine", "Tryptophan", "Uric acid"], "vibration": "CH3CH2 wagging / ring vibration"},
    "~1397 (Creatine)": {"center": 1397, "range": (1388, 1408), "metabolites": ["Creatine"], "vibration": "COO- stretch"},
    "~1420 (Creatinine)": {"center": 1420, "range": (1410, 1440), "metabolites": ["Creatinine", "Carotenoids", "Uric acid"], "vibration": "C=C stretch"},
    "~1449 (CH2 def)": {"center": 1449, "range": (1435, 1465), "metabolites": ["Lipids", "Fatty acids", "Hydroxybutyrate"], "vibration": "CH2 deformation"},
    "~1518 (Carotenoids)": {"center": 1518, "range": (1508, 1530), "metabolites": ["Carotenoids"], "vibration": "C=C stretch"},
    "~1540 (Neopterin)": {"center": 1540, "range": (1530, 1555), "metabolites": ["Neopterin"], "vibration": "Ring vibration"},
    "~1597 (C=C/Purine)": {"center": 1597, "range": (1580, 1615), "metabolites": ["Adenine", "Kynurenine", "Tyrosine", "Phenylalanine", "Urea"], "vibration": "C=N, C=C ring stretch"},
    "~1651 (Amide I)": {"center": 1651, "range": (1640, 1665), "metabolites": ["Maleic acid", "Glycogen", "Kynurenine", "Stearic acid", "Protein"], "vibration": "C=O stretch (Amide I)"},
    "~1680 (C=O)": {"center": 1680, "range": (1670, 1695), "metabolites": ["Guanine", "Uracil", "Maleic acid", "NADH"], "vibration": "C=O stretch"},
    "~1706 (Creatinine)": {"center": 1706, "range": (1696, 1716), "metabolites": ["Creatinine"], "vibration": "C=O stretch"},
    "~2098 (S-H)": {"center": 2098, "range": (2085, 2115), "metabolites": ["Unknown / S-H stretch"], "vibration": "S-H stretch or instrumental"},
}


# ── Data Loading ───────────────────────────────────────────────────────────────
def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray]:
    """Load processed spectra, apply group aliases, filter to relevant groups."""
    df = pd.read_csv(PROJECT_ROOT / "results" / "processed_spectra.csv")
    # Apply aliases
    for alias, originals in GROUP_ALIASES.items():
        df.loc[df["group"].isin(originals), "group"] = alias
    # Remove SPAN (post-op)
    df = df[df["group"] != "SPAN"].copy()

    all_groups = CANCER_TYPES + NON_CANCER
    df = df[df["group"].isin(all_groups)].copy()

    feat_cols = extract_feature_columns(df)
    wavenumbers = feature_axis_from_names(feat_cols)

    X = df[feat_cols].values.astype(np.float32)
    groups_arr = df["group"].values
    sample_ids = (df["group"] + " " + df["sample_id"].astype(str)).values

    # Binary labels: cancer=1, non-cancer=0
    binary_labels = np.array([1 if g in CANCER_TYPES else 0 for g in groups_arr])
    # Cancer type labels: 0-6 for cancer, -1 for non-cancer
    cancer_type_labels = np.array([
        CANCER_TYPES.index(g) if g in CANCER_TYPES else -1 for g in groups_arr
    ])

    logger.info(f"Data loaded: {X.shape[0]} spectra, {X.shape[1]} features")
    logger.info(f"Groups: {pd.Series(groups_arr).value_counts().to_dict()}")

    return X, binary_labels, cancer_type_labels, groups_arr, sample_ids, feat_cols, wavenumbers


# ── Model Loading / Training ──────────────────────────────────────────────────
def load_or_train_stage1(X, y, sample_ids, feat_cols):
    """Train stage1 binary model on current data."""
    logger.info(f"Training stage1 binary model on {X.shape[0]} samples, {X.shape[1]} features...")
    model = make_pipeline(StandardScaler(), LogisticRegression(
        max_iter=5000, C=1.0, solver="lbfgs", random_state=SEED,
    ))
    model.fit(X, y)

    lr = model.named_steps["logisticregression"]
    scaler = model.named_steps["standardscaler"]
    return model, lr, scaler


def load_or_train_stage2(X, y_type, groups_arr, sample_ids, feat_cols):
    """Train stage2 OvR model on cancer-only data."""
    cancer_mask = y_type >= 0
    X_cancer = X[cancer_mask]
    y_cancer = y_type[cancer_mask]

    logger.info(f"Training stage2 OvR model on {X_cancer.shape[0]} cancer samples...")
    model = make_pipeline(StandardScaler(), LogisticRegression(
        max_iter=5000, C=1.0, solver="lbfgs", multi_class="ovr",
        random_state=SEED,
    ))
    model.fit(X_cancer, y_cancer)

    lr = model.named_steps["logisticregression"]
    scaler = model.named_steps["standardscaler"]
    return model, lr, scaler, cancer_mask


# ── Helpers ────────────────────────────────────────────────────────────────────
def coef_to_original_scale(coef: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Scale LR coefficients back to original feature space."""
    return coef / scaler.scale_


def find_nearest_band(wavenumber: float, tolerance: float = TOLERANCE_CM1) -> Tuple[Optional[str], Optional[str], Optional[str], float]:
    """Find nearest BAND_ASSIGNMENTS entry within tolerance.

    Returns (band_name, primary_metabolite, vibration_mode, distance_cm1).
    """
    best_dist = float("inf")
    best_band = None
    for band_name, info in BAND_ASSIGNMENTS.items():
        center = info["center"]
        dist = abs(wavenumber - center)
        if dist < best_dist:
            best_dist = dist
            best_band = band_name
            best_info = info

    if best_dist <= tolerance and best_band is not None:
        metabolites_str = ", ".join(best_info["metabolites"][:3])
        return best_band, metabolites_str, best_info["vibration"], best_dist
    return None, None, None, best_dist


def get_top_features(coef: np.ndarray, wavenumbers: np.ndarray, n: int = 15) -> pd.DataFrame:
    """Get top N features by |coefficient|."""
    abs_coef = np.abs(coef)
    top_idx = np.argsort(abs_coef)[::-1][:n]
    rows = []
    for i in top_idx:
        rows.append({
            "wavenumber": wavenumbers[i],
            "coefficient": coef[i],
            "abs_coefficient": abs_coef[i],
            "direction": "+" if coef[i] > 0 else "-",
            "feature_idx": i,
        })
    return pd.DataFrame(rows)


# ── Analysis 1: Stage 1 Binary Coefficient Spectrum ──────────────────────────
def analysis_1_binary_coefficient(coef_orig: np.ndarray, wavenumbers: np.ndarray):
    """Plot LR coefficients for cancer vs non-cancer with peak overlay."""
    logger.info("\n" + "=" * 80)
    logger.info("Analysis 1: Stage 1 Binary Coefficient Spectrum")
    logger.info("=" * 80)

    fig, ax = plt.subplots(figsize=(16, 6))

    # Plot coefficient spectrum
    ax.plot(wavenumbers, coef_orig, color="#333333", linewidth=1.2, alpha=0.9, zorder=3)
    ax.fill_between(wavenumbers, 0, coef_orig,
                     where=(coef_orig > 0), color="#EF5350", alpha=0.25, label="Pro-cancer (+)")
    ax.fill_between(wavenumbers, 0, coef_orig,
                     where=(coef_orig < 0), color="#42A5F5", alpha=0.25, label="Anti-cancer (-)")
    ax.axhline(y=0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

    # Highlight top N wavenumbers
    top_df = get_top_features(coef_orig, wavenumbers, TOP_N_BINARY)
    for _, row in top_df.iterrows():
        color = "#D32F2F" if row["direction"] == "+" else "#1565C0"
        ax.scatter(row["wavenumber"], row["coefficient"], s=40, color=color,
                   zorder=5, edgecolors="white", linewidth=0.5)

    # Overlay metabolite band assignments as vertical shaded regions
    band_colors = plt.cm.Set3(np.linspace(0, 1, len(BAND_ASSIGNMENTS)))
    y_min, y_max = ax.get_ylim()
    label_y_positions = []

    for i, (band_name, info) in enumerate(BAND_ASSIGNMENTS.items()):
        lo, hi = info["range"]
        if lo < wavenumbers.min() or hi > wavenumbers.max():
            continue
        ax.axvspan(lo, hi, alpha=0.08, color=band_colors[i], zorder=1)

        # Short label (just metabolites)
        short_label = info["metabolites"][0] if len(info["metabolites"]) > 0 else ""
        center = info["center"]

        # Stagger vertical position to avoid overlap
        y_pos = y_max + (y_max - y_min) * 0.02
        if i % 2 == 1:
            y_pos = y_max + (y_max - y_min) * 0.08

        label_y_positions.append(y_pos)
        ax.annotate(
            f"{center:.0f}",
            xy=(center, y_max), xytext=(center, y_pos),
            fontsize=7, ha="center", va="bottom", color=band_colors[i] * 0.6,
            rotation=90,
            arrowprops=dict(arrowstyle="-", color=band_colors[i] * 0.6, lw=0.5),
        )

    # Annotate top 10 with wavenumber labels
    for _, row in top_df.head(10).iterrows():
        band_name, metabolite, vibration, dist = find_nearest_band(row["wavenumber"])
        label = f"{row['wavenumber']:.0f}"
        if metabolite:
            primary = metabolite.split(",")[0].strip()
            label += f"\n({primary})"

        y_offset = 20 if row["direction"] == "+" else -25
        ax.annotate(
            label,
            xy=(row["wavenumber"], row["coefficient"]),
            xytext=(0, y_offset),
            textcoords="offset points",
            ha="center", va="bottom" if row["direction"] == "+" else "top",
            fontsize=PAPER_ANNOTATION_SIZE - 2,
            color="#333333",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
            arrowprops=dict(arrowstyle="-", color="#999999", lw=0.5),
        )

    apply_publication_style(
        ax,
        xlabel="Raman Shift (cm$^{-1}$)",
        ylabel="LR Coefficient (original scale)",
        title="Stage 1: Cancer Detection — LR Coefficient Spectrum with Peak Assignments",
    )
    ax.legend(loc="upper right", fontsize=PAPER_LEGEND_SIZE, framealpha=0.9)

    out_path = OUTPUT_DIR / "stage1_binary_coefficient_spectrum.png"
    save_figure(fig, out_path)
    logger.info(f"Saved: {out_path}")

    # Print summary
    logger.info(f"\nTop {TOP_N_BINARY} most influential wavenumbers (Stage 1):")
    for _, row in top_df.iterrows():
        band_name, metabolite, vibration, dist = find_nearest_band(row["wavenumber"])
        met_str = f"  -> {metabolite} ({vibration}), d={dist:.0f} cm-1" if metabolite else "  -> No match"
        logger.info(f"  {row['wavenumber']:.1f} cm-1  coef={row['coefficient']:+.6f}  {met_str}")


# ── Analysis 2: Stage 2 Per-Cancer-Type Coefficient Spectra ──────────────────
def analysis_2_per_cancer_coefficients(coef_matrix: np.ndarray, wavenumbers: np.ndarray, classes: List[str]):
    """Plot OvR coefficient spectra for each cancer type in subplots."""
    logger.info("\n" + "=" * 80)
    logger.info("Analysis 2: Stage 2 Per-Cancer-Type Coefficient Spectra")
    logger.info("=" * 80)

    n_types = len(classes)
    fig, axes = plt.subplots(n_types, 1, figsize=(16, 3 * n_types), sharex=True)
    if n_types == 1:
        axes = [axes]

    # Compute specificity: for each wavenumber, how uniquely important is it for each type?
    abs_coef = np.abs(coef_matrix)
    # Ratio of max to second-max |coef| across cancer types for each feature
    sorted_abs = np.sort(abs_coef, axis=0)[::-1]  # shape: (n_types, n_features)
    specificity = np.zeros_like(abs_coef)
    for ct in range(n_types):
        # Specificity = this type's |coef| / mean of all types' |coef|
        mean_abs = abs_coef.mean(axis=0)
        mean_abs[mean_abs == 0] = 1e-10
        specificity[ct] = abs_coef[ct] / mean_abs

    for ct_idx, cancer_type in enumerate(classes):
        ax = axes[ct_idx]
        coef = coef_matrix[ct_idx]
        color = CANCER_COLORS.get(cancer_type, "#666666")

        ax.plot(wavenumbers, coef, color=color, linewidth=1.2, alpha=0.9)
        ax.fill_between(wavenumbers, 0, coef, where=(coef > 0), color=color, alpha=0.2)
        ax.fill_between(wavenumbers, 0, coef, where=(coef < 0), color=color, alpha=0.1)
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)

        # Mark uniquely important peaks (specificity > 2.0 and in top 20 by |coef|)
        top_idx = np.argsort(np.abs(coef))[::-1][:20]
        unique_peaks = [i for i in top_idx if specificity[ct_idx, i] > 2.0]

        for i in unique_peaks[:8]:  # Limit annotations
            marker_color = "#D32F2F" if coef[i] > 0 else "#1565C0"
            ax.scatter(wavenumbers[i], coef[i], s=50, color=marker_color,
                       zorder=5, edgecolors="white", linewidth=0.8, marker="D")
            band_name, metabolite, _, _ = find_nearest_band(wavenumbers[i])
            label = f"{wavenumbers[i]:.0f}"
            if metabolite:
                primary = metabolite.split(",")[0].strip()
                label += f" ({primary})"
            y_offset = 12 if coef[i] > 0 else -15
            ax.annotate(
                label, xy=(wavenumbers[i], coef[i]),
                xytext=(0, y_offset), textcoords="offset points",
                ha="center", fontsize=7, color="#333333",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8),
            )

        apply_publication_style(ax, ylabel="Coefficient")
        ax.set_title(f"{cancer_type}", fontsize=PAPER_LABEL_SIZE, fontweight="bold",
                     loc="left", color=color, pad=2)

        # Light band overlays
        band_colors_arr = plt.cm.Pastel1(np.linspace(0, 1, len(BAND_ASSIGNMENTS)))
        for bi, (_, info) in enumerate(BAND_ASSIGNMENTS.items()):
            lo, hi = info["range"]
            if lo >= wavenumbers.min() and hi <= wavenumbers.max():
                ax.axvspan(lo, hi, alpha=0.05, color=band_colors_arr[bi], zorder=0)

    axes[-1].set_xlabel("Raman Shift (cm$^{-1}$)", fontsize=PAPER_LABEL_SIZE, labelpad=8)

    fig.suptitle(
        "Stage 2: Per-Cancer-Type OvR LR Coefficients (diamonds = uniquely discriminative)",
        fontsize=PAPER_TITLE_SIZE, fontweight="bold", y=1.01,
    )

    out_path = OUTPUT_DIR / "stage2_per_cancer_coefficient_spectra.png"
    save_figure(fig, out_path)
    logger.info(f"Saved: {out_path}")

    # Print unique peaks per cancer type
    for ct_idx, cancer_type in enumerate(classes):
        coef = coef_matrix[ct_idx]
        top_idx = np.argsort(np.abs(coef))[::-1][:10]
        unique = [i for i in top_idx if specificity[ct_idx, i] > 2.0]
        logger.info(f"\n{cancer_type} — uniquely discriminative peaks ({len(unique)}):")
        for i in unique[:8]:
            band_name, metabolite, vibration, dist = find_nearest_band(wavenumbers[i])
            met_str = f"{metabolite}" if metabolite else "No match"
            logger.info(f"  {wavenumbers[i]:.1f} cm-1  coef={coef[i]:+.6f}  spec={specificity[ct_idx, i]:.1f}x  {met_str}")


# ── Analysis 3: Cross-Reference Table ────────────────────────────────────────
def analysis_3_crossref_table(
    coef_s1: np.ndarray,
    coef_s2: np.ndarray,
    wavenumbers: np.ndarray,
    classes: List[str],
):
    """Create coefficient × peak assignment cross-reference CSV."""
    logger.info("\n" + "=" * 80)
    logger.info("Analysis 3: Coefficient x Peak Assignment Cross-Reference Table")
    logger.info("=" * 80)

    rows = []

    # Stage 1 binary
    top_s1 = get_top_features(coef_s1, wavenumbers, TOP_N_PER_CANCER)
    for _, row in top_s1.iterrows():
        band_name, metabolite, vibration, dist = find_nearest_band(row["wavenumber"])
        rows.append({
            "cancer_type": "Binary (Cancer vs Non-cancer)",
            "wavenumber": row["wavenumber"],
            "coefficient": row["coefficient"],
            "abs_coefficient": row["abs_coefficient"],
            "direction": row["direction"],
            "nearest_band": band_name or "",
            "nearest_metabolite": metabolite or "",
            "vibration_mode": vibration or "",
            "distance_cm1": dist,
        })

    # Stage 2 per cancer type
    for ct_idx, cancer_type in enumerate(classes):
        coef = coef_s2[ct_idx]
        top_ct = get_top_features(coef, wavenumbers, TOP_N_PER_CANCER)
        for _, row in top_ct.iterrows():
            band_name, metabolite, vibration, dist = find_nearest_band(row["wavenumber"])
            rows.append({
                "cancer_type": cancer_type,
                "wavenumber": row["wavenumber"],
                "coefficient": row["coefficient"],
                "abs_coefficient": row["abs_coefficient"],
                "direction": row["direction"],
                "nearest_band": band_name or "",
                "nearest_metabolite": metabolite or "",
                "vibration_mode": vibration or "",
                "distance_cm1": dist,
            })

    df_out = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "coefficient_peak_crossref.csv"
    ensure_output_dir(out_path)
    df_out.to_csv(out_path, index=False)
    logger.info(f"Saved: {out_path}")
    logger.info(f"Total entries: {len(df_out)}")

    # Print summary: match rate
    matched = df_out[df_out["nearest_metabolite"] != ""]
    logger.info(f"Matched to metabolite band: {len(matched)}/{len(df_out)} ({100*len(matched)/len(df_out):.0f}%)")

    # Per cancer type summary
    for ct in df_out["cancer_type"].unique():
        sub = df_out[df_out["cancer_type"] == ct]
        n_matched = (sub["nearest_metabolite"] != "").sum()
        logger.info(f"  {ct}: {n_matched}/{len(sub)} matched")

    return df_out


# ── Analysis 4: Cancer-Discriminative Peak Heatmap ───────────────────────────
def analysis_4_peak_heatmap(coef_s2: np.ndarray, wavenumbers: np.ndarray, classes: List[str]):
    """Create heatmap: metabolite peaks × cancer types with mean |coefficient| in window."""
    logger.info("\n" + "=" * 80)
    logger.info("Analysis 4: Cancer-Discriminative Peak Heatmap")
    logger.info("=" * 80)

    band_names = []
    band_short_names = []
    matrix_data = []

    for band_name, info in BAND_ASSIGNMENTS.items():
        lo, hi = info["range"]
        # Find wavenumber indices in this band window
        mask = (wavenumbers >= lo) & (wavenumbers <= hi)
        if mask.sum() == 0:
            continue

        band_names.append(band_name)
        # Short name for display
        primary_met = info["metabolites"][0] if info["metabolites"] else "?"
        center = info["center"]
        band_short_names.append(f"{center} ({primary_met})")

        # Mean |coefficient| per cancer type in this window
        row_vals = []
        for ct_idx in range(len(classes)):
            mean_abs = np.mean(np.abs(coef_s2[ct_idx, mask]))
            row_vals.append(mean_abs)
        matrix_data.append(row_vals)

    matrix = np.array(matrix_data)  # (n_bands, n_cancer_types)

    # Row-wise normalization (per peak, show relative discrimination)
    row_max = matrix.max(axis=1, keepdims=True)
    row_max[row_max == 0] = 1e-10
    matrix_norm = matrix / row_max

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, max(8, len(band_short_names) * 0.4)))

    im = ax.imshow(matrix_norm, aspect="auto", cmap="YlOrRd", interpolation="nearest")

    # Axes
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, fontsize=PAPER_TICK_SIZE, fontweight="bold")
    ax.set_yticks(range(len(band_short_names)))
    ax.set_yticklabels(band_short_names, fontsize=PAPER_TICK_SIZE - 1)

    # Annotate cells with normalized values
    for i in range(matrix_norm.shape[0]):
        for j in range(matrix_norm.shape[1]):
            val = matrix_norm[i, j]
            text_color = "white" if val > 0.65 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=text_color, fontweight="bold" if val > 0.8 else "normal")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="Relative |coefficient| (row-normalized)")

    apply_publication_style(
        ax,
        xlabel="Cancer Type",
        ylabel="Metabolite Peak Region",
        title="Cancer-Discriminative Peak Heatmap (OvR LR Coefficients)",
    )

    out_path = OUTPUT_DIR / "cancer_peak_discrimination_heatmap.png"
    save_figure(fig, out_path)
    logger.info(f"Saved: {out_path}")

    # Also save raw matrix as CSV
    heatmap_df = pd.DataFrame(matrix, columns=classes, index=band_short_names)
    heatmap_df.index.name = "peak_region"
    csv_path = OUTPUT_DIR / "peak_discrimination_matrix.csv"
    heatmap_df.to_csv(csv_path)
    logger.info(f"Saved: {csv_path}")

    # Print top discriminative peaks per cancer
    logger.info("\nTop 3 discriminative peaks per cancer type:")
    for ct_idx, cancer_type in enumerate(classes):
        col_vals = matrix_norm[:, ct_idx]
        top3 = np.argsort(col_vals)[::-1][:3]
        peaks_str = ", ".join([f"{band_short_names[i]} ({col_vals[i]:.2f})" for i in top3])
        logger.info(f"  {cancer_type}: {peaks_str}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    logger.info("=" * 80)
    logger.info("LR Coefficient Visualization with Peak Assignment Overlay")
    logger.info("=" * 80)

    # Load data
    X, binary_labels, cancer_type_labels, groups_arr, sample_ids, feat_cols, wavenumbers = load_data()

    # --- Stage 1: Binary ---
    model_s1, lr_s1, scaler_s1 = load_or_train_stage1(X, binary_labels, sample_ids, feat_cols)
    coef_s1_raw = lr_s1.coef_[0]  # binary: shape (n_features,)
    coef_s1_orig = coef_to_original_scale(coef_s1_raw, scaler_s1)
    logger.info(f"Stage 1 coef shape: {coef_s1_raw.shape}, classes: {lr_s1.classes_}")

    # --- Stage 2: Multi-class OvR ---
    model_s2, lr_s2, scaler_s2, cancer_mask = load_or_train_stage2(
        X, cancer_type_labels, groups_arr, sample_ids, feat_cols
    )
    coef_s2_raw = lr_s2.coef_  # OvR: shape (n_classes, n_features)
    coef_s2_orig = np.zeros_like(coef_s2_raw)
    for i in range(coef_s2_raw.shape[0]):
        coef_s2_orig[i] = coef_to_original_scale(coef_s2_raw[i], scaler_s2)

    # Map class indices back to cancer type names
    s2_classes = [CANCER_TYPES[c] for c in lr_s2.classes_]
    logger.info(f"Stage 2 coef shape: {coef_s2_raw.shape}, classes: {s2_classes}")

    # Ensure output directory
    ensure_output_dir(OUTPUT_DIR / "dummy")

    # Run analyses
    analysis_1_binary_coefficient(coef_s1_orig, wavenumbers)
    analysis_2_per_cancer_coefficients(coef_s2_orig, wavenumbers, s2_classes)
    crossref_df = analysis_3_crossref_table(coef_s1_orig, coef_s2_orig, wavenumbers, s2_classes)
    analysis_4_peak_heatmap(coef_s2_orig, wavenumbers, s2_classes)

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Figures saved:")
    logger.info(f"  1. stage1_binary_coefficient_spectrum.png")
    logger.info(f"  2. stage2_per_cancer_coefficient_spectra.png")
    logger.info(f"  3. coefficient_peak_crossref.csv")
    logger.info(f"  4. cancer_peak_discrimination_heatmap.png")
    logger.info(f"  5. peak_discrimination_matrix.csv")
    logger.info(f"\nTotal wavenumber features: {len(wavenumbers)}")
    logger.info(f"  Range: {wavenumbers.min():.1f} - {wavenumbers.max():.1f} cm-1")
    logger.info(f"Band assignments loaded: {len(BAND_ASSIGNMENTS)}")


if __name__ == "__main__":
    main()
