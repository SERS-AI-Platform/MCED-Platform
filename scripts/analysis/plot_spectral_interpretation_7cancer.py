"""
Spectral Interpretation Analysis — 7-Cancer SERS Model (Phase W)

Generates publication-quality plots for LR coefficient-based spectral interpretation:
1. Feature importance (Stage 1 binary + Stage 2 multiclass)
2. Mean spectra by cancer type overlay
3. Difference spectra (cancer - non-cancer)
4. Top discriminating wavenumber regions with SERS peak annotations
5. Per-cancer coefficient heatmap
6. BRE-focused comparison analysis

Output: results/figures/spectral_interpretation_7cancer/
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import LinearSegmentedColormap
import yaml
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from models.train import (
    load_processed_spectra, get_feature_columns,
    resolve_aliases, create_labels,
)
from sers.models._legacy.resnet_v1.model import ModelConfig

# =============================================================================
# Constants
# =============================================================================
CANCER_TYPES = ("PRO", "BRE", "OVA", "LUN", "CRC", "PAN", "BLC")
NON_CANCER = ("NOR", "DIA", "HBP", "H.D.")

GROUP_COLORS = {
    "PRO": "#E91E63", "BRE": "#FF69B4", "OVA": "#AB47BC",
    "LUN": "#42A5F5", "CRC": "#EF5350", "PAN": "#FFA726",
    "BLC": "#7E57C2", "NOR": "#8D6E63", "DIA": "#66BB6A",
    "HBP": "#26A69A", "H.D.": "#78909C",
}

GROUP_LABELS = {
    "PRO": "Prostate", "BRE": "Breast", "OVA": "Ovarian",
    "LUN": "Lung", "CRC": "Colorectal", "PAN": "Pancreatic",
    "BLC": "Bladder", "NOR": "Normal", "DIA": "Diabetes",
    "HBP": "High BP", "H.D.": "High BP+DIA",
}

SERS_PEAKS = {
    618:  "C-S stretch\n(Cysteine)",
    683:  "C-S/ring\n(Creatinine)",
    724:  "Adenine\nring",
    849:  "Tyr Fermi\ndoublet",
    999:  "Phe ring\n(dominant)",
    1148: "C-N/\nC-O-C",
    1231: "Amide III",
    1352: "CH\ndeformation",
    1449: "CH\u2082\ndeformation",
    1597: "C=C/\nPurine",
    1651: "Amide I",
}

BG_COLOR = "#0a0e1a"
TEXT_COLOR = "#e0e0e0"
GRID_COLOR = "#1a2035"
DPI = 200

OUTPUT_DIR = PROJECT_ROOT / "results" / "figures" / "spectral_interpretation_7cancer"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": BG_COLOR,
        "axes.facecolor": BG_COLOR,
        "axes.edgecolor": "#2a3050",
        "axes.labelcolor": TEXT_COLOR,
        "text.color": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "grid.color": GRID_COLOR,
        "grid.alpha": 0.4,
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 8,
        "legend.facecolor": "#101528",
        "legend.edgecolor": "#2a3050",
    })


def load_data():
    """Load spectra and build config for 7-cancer model."""
    with open(PROJECT_ROOT / "config" / "config.yaml") as f:
        cfg_dict = yaml.safe_load(f)

    df = load_processed_spectra()
    feat_cols = get_feature_columns(df)

    config = ModelConfig.from_pipeline_config(
        cfg_dict, n_spectral_features=len(feat_cols),
        cancer_types=CANCER_TYPES,
        non_cancer_groups=NON_CANCER,
    )
    # Expand cancer_groups_raw to include alias members
    raw_groups = []
    for ct in CANCER_TYPES:
        if ct in config.group_aliases:
            raw_groups.extend(config.group_aliases[ct])
        else:
            raw_groups.append(ct)
    config = ModelConfig(
        **{**config.__dict__,
           "cancer_groups_raw": tuple(raw_groups)}
    )

    df = resolve_aliases(df, config)
    return df, feat_cols, config


def extract_wavenumbers(feat_cols):
    """Convert x_402.00 style column names to float wavenumbers."""
    return np.array([float(c.replace("x_", "")) for c in feat_cols])


def train_lr_models(X, binary_labels, cancer_type_labels, config):
    """Train Stage 1 (binary) and Stage 2 (7-class) LR models."""
    # Stage 1: binary
    s1 = make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, max_iter=1000, solver="saga", class_weight="balanced",
        random_state=42))
    s1.fit(X, binary_labels)

    # Stage 2: 7-class (cancer samples only)
    cancer_mask = binary_labels == 1
    X_cancer = X[cancer_mask]
    y_cancer = cancer_type_labels[cancer_mask]
    s2 = make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, max_iter=1000, solver="saga", class_weight="balanced",
        multi_class="multinomial", random_state=42))
    s2.fit(X_cancer, y_cancer)

    return s1, s2


# =============================================================================
# Plot 1: Feature Importance (LR Coefficients)
# =============================================================================
def plot_feature_importance(wn, s1, s2, config):
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    # Stage 1: binary coefficients (1D)
    coef_s1 = np.abs(s1.named_steps["logisticregression"].coef_[0])
    ax = axes[0]
    ax.fill_between(wn, coef_s1, alpha=0.4, color="#42A5F5")
    ax.plot(wn, coef_s1, color="#42A5F5", linewidth=0.8)
    ax.set_ylabel("|Coefficient|")
    ax.set_title("Stage 1 — Cancer Screening (Binary LR): Feature Importance", fontweight="bold")
    ax.grid(True, alpha=0.3)
    # annotate top peaks
    _annotate_sers_peaks(ax, wn, coef_s1)

    # Stage 2: sum of absolute coefficients across 7 classes
    coef_s2 = np.abs(s2.named_steps["logisticregression"].coef_)  # (7, n_features)
    coef_s2_sum = coef_s2.sum(axis=0)
    ax = axes[1]
    ax.fill_between(wn, coef_s2_sum, alpha=0.4, color="#FFA726")
    ax.plot(wn, coef_s2_sum, color="#FFA726", linewidth=0.8)
    ax.set_ylabel("Sum |Coefficient|")
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_title("Stage 2 — Cancer Type ID (7-class LR): Summed Feature Importance", fontweight="bold")
    ax.grid(True, alpha=0.3)
    _annotate_sers_peaks(ax, wn, coef_s2_sum)

    plt.tight_layout()
    path = OUTPUT_DIR / "01_feature_importance_lr_coefficients.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  Saved: {path}")
    return coef_s1, coef_s2


def _annotate_sers_peaks(ax, wn, values, n_top=6):
    """Mark known SERS peaks on the plot."""
    for peak_wn, label in SERS_PEAKS.items():
        idx = np.argmin(np.abs(wn - peak_wn))
        val = values[idx]
        ax.axvline(wn[idx], color="#ffffff", alpha=0.15, linewidth=0.5)


# =============================================================================
# Plot 2: Mean Spectra Overlay
# =============================================================================
def plot_mean_spectra(df, feat_cols, wn):
    fig, ax = plt.subplots(figsize=(16, 6))

    all_groups = list(CANCER_TYPES) + list(NON_CANCER)
    for g in all_groups:
        sub = df[df["group"] == g]
        if len(sub) == 0:
            continue
        mean_spec = sub[feat_cols].mean().values
        lw = 1.5 if g in CANCER_TYPES else 1.0
        alpha = 0.9 if g in CANCER_TYPES else 0.6
        ls = "-" if g in CANCER_TYPES else "--"
        ax.plot(wn, mean_spec, color=GROUP_COLORS[g], linewidth=lw,
                alpha=alpha, linestyle=ls, label=f"{g} ({GROUP_LABELS[g]}, n={sub['sample_id'].nunique()})")

    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Preprocessed Intensity (SNV)")
    ax.set_title("Mean SERS Spectra by Cancer Type & Control Groups", fontweight="bold")
    ax.legend(ncol=3, loc="upper right", framealpha=0.8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = OUTPUT_DIR / "02_mean_spectra_overlay.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# Plot 3: Difference Spectra
# =============================================================================
def plot_difference_spectra(df, feat_cols, wn):
    fig, axes = plt.subplots(len(CANCER_TYPES), 1, figsize=(16, 2.2 * len(CANCER_TYPES)),
                              sharex=True)

    # mean non-cancer spectrum
    nc_mask = df["group"].isin(NON_CANCER)
    mean_nc = df.loc[nc_mask, feat_cols].mean().values

    for i, ct in enumerate(CANCER_TYPES):
        ax = axes[i]
        sub = df[df["group"] == ct]
        if len(sub) == 0:
            continue
        mean_cancer = sub[feat_cols].mean().values
        diff = mean_cancer - mean_nc

        ax.fill_between(wn, diff, where=diff > 0, alpha=0.4,
                         color=GROUP_COLORS[ct], interpolate=True)
        ax.fill_between(wn, diff, where=diff < 0, alpha=0.3,
                         color="#555555", interpolate=True)
        ax.plot(wn, diff, color=GROUP_COLORS[ct], linewidth=0.8)
        ax.axhline(0, color="#ffffff", linewidth=0.3, alpha=0.5)
        ax.set_ylabel(f"{ct}", fontweight="bold", fontsize=10)
        ax.grid(True, alpha=0.2)

        # SERS peak markers
        for pk in SERS_PEAKS:
            ax.axvline(pk, color="#ffffff", alpha=0.1, linewidth=0.5)

    axes[-1].set_xlabel("Wavenumber (cm$^{-1}$)")
    fig.suptitle("Difference Spectra: Cancer Type Mean - Non-Cancer Mean",
                 fontweight="bold", fontsize=14, y=1.01)
    plt.tight_layout()
    path = OUTPUT_DIR / "03_difference_spectra.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# Plot 4: Top Discriminating Wavenumber Regions
# =============================================================================
def plot_top_wavenumbers(wn, coef_s1, coef_s2, config):
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    for ax_idx, (coef, title_label, color) in enumerate([
        (coef_s1, "Stage 1 (Binary)", "#42A5F5"),
        (coef_s2.sum(axis=0), "Stage 2 (7-class Sum)", "#FFA726"),
    ]):
        ax = axes[ax_idx]
        ax.plot(wn, coef, color=color, linewidth=0.8, alpha=0.8)
        ax.fill_between(wn, coef, alpha=0.2, color=color)

        # Find closest peaks and rank by coefficient
        peak_vals = {}
        for pk_wn, pk_label in SERS_PEAKS.items():
            idx = np.argmin(np.abs(wn - pk_wn))
            peak_vals[pk_wn] = (coef[idx], pk_label, idx)

        # Sort by importance and take top 10
        sorted_peaks = sorted(peak_vals.items(), key=lambda x: x[1][0], reverse=True)[:10]

        y_max = coef.max()
        for rank, (pk_wn, (val, label, idx)) in enumerate(sorted_peaks):
            ax.axvline(wn[idx], color="#ffffff", alpha=0.3, linewidth=0.7, linestyle=":")
            # Stagger vertical offsets to reduce overlap
            offset = 0.55 + 0.35 * (rank % 2)
            ax.annotate(
                f"#{rank+1}\n{int(pk_wn)} cm$^{{-1}}$\n{label}",
                xy=(wn[idx], val), xytext=(wn[idx], y_max * offset),
                fontsize=7, ha="center", va="bottom", color="#ffffff",
                arrowprops=dict(arrowstyle="-", color="#ffffff", alpha=0.4, lw=0.5),
                bbox=dict(boxstyle="round,pad=0.2", fc="#1a2035", ec="#2a3050", alpha=0.8),
            )

        ax.set_ylabel("|Coefficient|")
        ax.set_title(f"Top Discriminating Wavenumber Regions — {title_label}", fontweight="bold")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Wavenumber (cm$^{-1}$)")
    plt.tight_layout()
    path = OUTPUT_DIR / "04_top_discriminating_wavenumbers.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# Plot 5: Per-Cancer Coefficient Heatmap
# =============================================================================
def plot_coefficient_heatmap(wn, coef_s2, config):
    fig, ax = plt.subplots(figsize=(18, 5))

    # Downsample for readability: take every 5th point
    step = 5
    wn_ds = wn[::step]
    coef_ds = coef_s2[:, ::step]

    # Custom diverging colormap
    cmap = LinearSegmentedColormap.from_list(
        "custom_div", ["#1a237e", "#0a0e1a", "#b71c1c"], N=256)

    vmax = np.percentile(np.abs(coef_ds), 98)
    im = ax.imshow(coef_ds, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax,
                    extent=[wn_ds[0], wn_ds[-1], len(CANCER_TYPES) - 0.5, -0.5])

    ax.set_yticks(range(len(CANCER_TYPES)))
    ax.set_yticklabels([f"{ct} ({GROUP_LABELS[ct]})" for ct in CANCER_TYPES], fontsize=10)

    # Annotate SERS peaks at top
    for pk_wn, pk_label in SERS_PEAKS.items():
        if wn_ds[0] <= pk_wn <= wn_ds[-1]:
            ax.axvline(pk_wn, color="#ffffff", alpha=0.3, linewidth=0.5, linestyle=":")
            ax.text(pk_wn, -0.7, f"{int(pk_wn)}\n{pk_label.split(chr(10))[0]}",
                    fontsize=6, ha="center", va="bottom", color="#cccccc", rotation=0)

    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_title("Stage 2 LR Coefficients per Cancer Type (7-class)", fontweight="bold", pad=30)

    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("LR Coefficient", color=TEXT_COLOR)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COLOR)

    plt.tight_layout()
    path = OUTPUT_DIR / "05_coefficient_heatmap_7class.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# Plot 6: BRE-Focused Analysis
# =============================================================================
def plot_bre_analysis(df, feat_cols, wn):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 6a: BRE mean vs non-cancer mean
    ax = axes[0, 0]
    nc_mask = df["group"].isin(NON_CANCER)
    mean_nc = df.loc[nc_mask, feat_cols].mean().values
    bre_data = df[df["group"] == "BRE"]
    mean_bre = bre_data[feat_cols].mean().values
    ax.plot(wn, mean_nc, color="#78909C", linewidth=1.0, alpha=0.8, label="Non-Cancer Mean")
    ax.plot(wn, mean_bre, color=GROUP_COLORS["BRE"], linewidth=1.5, label=f"BRE (n={bre_data['sample_id'].nunique()})")
    ax.fill_between(wn, mean_bre, mean_nc, alpha=0.15, color=GROUP_COLORS["BRE"])
    ax.set_title("BRE vs Non-Cancer Mean", fontweight="bold")
    ax.set_ylabel("Intensity (SNV)")
    ax.legend(framealpha=0.8)
    ax.grid(True, alpha=0.3)

    # 6b: BRE difference from non-cancer
    ax = axes[0, 1]
    diff_bre = mean_bre - mean_nc
    ax.fill_between(wn, diff_bre, where=diff_bre > 0, alpha=0.5, color=GROUP_COLORS["BRE"])
    ax.fill_between(wn, diff_bre, where=diff_bre < 0, alpha=0.3, color="#555555")
    ax.plot(wn, diff_bre, color=GROUP_COLORS["BRE"], linewidth=0.8)
    ax.axhline(0, color="#ffffff", linewidth=0.3, alpha=0.5)
    ax.set_title("BRE Difference Spectrum (BRE - NonCancer)", fontweight="bold")
    ax.set_ylabel("\u0394 Intensity")
    ax.grid(True, alpha=0.3)
    for pk in SERS_PEAKS:
        ax.axvline(pk, color="#ffffff", alpha=0.12, linewidth=0.5)

    # 6c: BRE vs each other cancer type
    ax = axes[1, 0]
    for ct in CANCER_TYPES:
        if ct == "BRE":
            continue
        sub = df[df["group"] == ct]
        if len(sub) == 0:
            continue
        mean_ct = sub[feat_cols].mean().values
        diff = mean_bre - mean_ct
        ax.plot(wn, diff, color=GROUP_COLORS[ct], linewidth=0.8, alpha=0.7,
                label=f"BRE - {ct}")
    ax.axhline(0, color="#ffffff", linewidth=0.3, alpha=0.5)
    ax.set_title("BRE vs Other Cancer Types (Difference)", fontweight="bold")
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("\u0394 Intensity")
    ax.legend(fontsize=7, ncol=2, framealpha=0.8)
    ax.grid(True, alpha=0.3)

    # 6d: BRE replicate variability (std band)
    ax = axes[1, 1]
    if len(bre_data) > 1:
        bre_std = bre_data[feat_cols].std().values
        ax.fill_between(wn, mean_bre - bre_std, mean_bre + bre_std,
                         alpha=0.3, color=GROUP_COLORS["BRE"], label="\u00b1 1 SD")
    ax.plot(wn, mean_bre, color=GROUP_COLORS["BRE"], linewidth=1.2, label="BRE Mean")
    # overlay all-cancer mean for reference
    cancer_mask = df["group"].isin(CANCER_TYPES)
    mean_all_cancer = df.loc[cancer_mask, feat_cols].mean().values
    ax.plot(wn, mean_all_cancer, color="#ffffff", linewidth=0.8, alpha=0.5,
            linestyle="--", label="All Cancer Mean")
    ax.set_title("BRE Spectral Variability (n=30, small cohort)", fontweight="bold")
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Intensity (SNV)")
    ax.legend(framealpha=0.8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = OUTPUT_DIR / "06_bre_focused_analysis.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# Combined Multi-Panel Figure
# =============================================================================
def plot_combined(wn, coef_s1, coef_s2, df, feat_cols, config):
    fig = plt.figure(figsize=(22, 18))
    gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.25)

    # Panel A: Stage 1 coefficients
    ax = fig.add_subplot(gs[0, 0])
    ax.fill_between(wn, coef_s1, alpha=0.4, color="#42A5F5")
    ax.plot(wn, coef_s1, color="#42A5F5", linewidth=0.8)
    ax.set_title("A. Stage 1 (Binary) — Feature Importance", fontweight="bold")
    ax.set_ylabel("|Coefficient|")
    ax.grid(True, alpha=0.3)
    for pk in SERS_PEAKS:
        ax.axvline(pk, color="#ffffff", alpha=0.12, linewidth=0.5)

    # Panel B: Stage 2 summed coefficients
    ax = fig.add_subplot(gs[0, 1])
    coef_s2_sum = coef_s2.sum(axis=0)
    ax.fill_between(wn, coef_s2_sum, alpha=0.4, color="#FFA726")
    ax.plot(wn, coef_s2_sum, color="#FFA726", linewidth=0.8)
    ax.set_title("B. Stage 2 (7-class) — Summed Feature Importance", fontweight="bold")
    ax.set_ylabel("Sum |Coefficient|")
    ax.grid(True, alpha=0.3)
    for pk in SERS_PEAKS:
        ax.axvline(pk, color="#ffffff", alpha=0.12, linewidth=0.5)

    # Panel C: Mean spectra overlay
    ax = fig.add_subplot(gs[1, :])
    all_groups = list(CANCER_TYPES) + list(NON_CANCER)
    for g in all_groups:
        sub = df[df["group"] == g]
        if len(sub) == 0:
            continue
        mean_spec = sub[feat_cols].mean().values
        lw = 1.2 if g in CANCER_TYPES else 0.8
        alpha = 0.9 if g in CANCER_TYPES else 0.5
        ls = "-" if g in CANCER_TYPES else "--"
        ax.plot(wn, mean_spec, color=GROUP_COLORS[g], linewidth=lw,
                alpha=alpha, linestyle=ls, label=f"{g}")
    ax.set_title("C. Mean Preprocessed Spectra by Group", fontweight="bold")
    ax.set_ylabel("Intensity (SNV)")
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.legend(ncol=6, loc="upper right", framealpha=0.8, fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel D: Heatmap
    ax = fig.add_subplot(gs[2, 0])
    step = 5
    wn_ds = wn[::step]
    coef_ds = coef_s2[:, ::step]
    cmap = LinearSegmentedColormap.from_list("div", ["#1a237e", "#0a0e1a", "#b71c1c"], N=256)
    vmax = np.percentile(np.abs(coef_ds), 98)
    im = ax.imshow(coef_ds, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax,
                    extent=[wn_ds[0], wn_ds[-1], len(CANCER_TYPES) - 0.5, -0.5])
    ax.set_yticks(range(len(CANCER_TYPES)))
    ax.set_yticklabels(list(CANCER_TYPES), fontsize=9)
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_title("D. Stage 2 Coefficient Heatmap", fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.7)

    # Panel E: BRE vs non-cancer
    ax = fig.add_subplot(gs[2, 1])
    nc_mask = df["group"].isin(NON_CANCER)
    mean_nc = df.loc[nc_mask, feat_cols].mean().values
    mean_bre = df[df["group"] == "BRE"][feat_cols].mean().values
    diff_bre = mean_bre - mean_nc
    ax.fill_between(wn, diff_bre, where=diff_bre > 0, alpha=0.5, color=GROUP_COLORS["BRE"])
    ax.fill_between(wn, diff_bre, where=diff_bre < 0, alpha=0.3, color="#555555")
    ax.plot(wn, diff_bre, color=GROUP_COLORS["BRE"], linewidth=0.8)
    ax.axhline(0, color="#ffffff", linewidth=0.3, alpha=0.5)
    ax.set_title("E. BRE Difference (BRE - NonCancer)", fontweight="bold")
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("\u0394 Intensity")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Spectral Interpretation Analysis — 7-Cancer SERS Model (Phase W)",
                 fontsize=16, fontweight="bold", y=0.995, color="#ffffff")
    path = OUTPUT_DIR / "00_combined_spectral_interpretation.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 70)
    print("Spectral Interpretation Analysis — 7-Cancer Model (Phase W)")
    print("=" * 70)

    setup_style()

    print("\n[1/7] Loading data...")
    df, feat_cols, config = load_data()
    wn = extract_wavenumbers(feat_cols)
    print(f"  Wavenumber range: {wn[0]:.1f} — {wn[-1]:.1f} cm-1 ({len(wn)} points)")
    print(f"  Groups in data: {sorted(df['group'].unique())}")

    print("\n[2/7] Training LR models...")
    X, binary_labels, cancer_type_labels, sample_ids, groups_arr = create_labels(df, config)
    s1, s2 = train_lr_models(X, binary_labels, cancer_type_labels, config)
    print(f"  Stage 1 accuracy: {s1.score(X, binary_labels):.4f}")
    cancer_mask = binary_labels == 1
    print(f"  Stage 2 accuracy: {s2.score(X[cancer_mask], cancer_type_labels[cancer_mask]):.4f}")

    print("\n[3/7] Plot 1 — Feature importance (LR coefficients)...")
    coef_s1, coef_s2 = plot_feature_importance(wn, s1, s2, config)

    print("\n[4/7] Plot 2 — Mean spectra overlay...")
    plot_mean_spectra(df, feat_cols, wn)

    print("\n[5/7] Plot 3 — Difference spectra...")
    plot_difference_spectra(df, feat_cols, wn)

    print("\n[6/7] Plot 4 — Top discriminating wavenumbers...")
    plot_top_wavenumbers(wn, coef_s1, coef_s2, config)

    print("\n[7/7] Plot 5 — Coefficient heatmap...")
    plot_coefficient_heatmap(wn, coef_s2, config)

    print("\n[Bonus] Plot 6 — BRE-focused analysis...")
    plot_bre_analysis(df, feat_cols, wn)

    print("\n[Combined] Multi-panel figure...")
    plot_combined(wn, coef_s1, coef_s2, df, feat_cols, config)

    print(f"\n{'=' * 70}")
    print(f"All plots saved to: {OUTPUT_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
