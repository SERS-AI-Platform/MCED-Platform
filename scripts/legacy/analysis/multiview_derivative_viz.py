#!/usr/bin/env python3
"""
Multi-View Derivative Spectrum — Preprocessing Validation & Visualization

Step 1 of uSERS-Net v2 experiment:
  - Generate raw / d1 / d2 channels via SG filter
  - Validate zero-crossings (d1=peak center, d2=inflection)
  - Visualize per-group mean spectra across all 3 channels
  - Compare channel statistics

Usage:
    python scripts/analysis/multiview_derivative_viz.py
"""

from __future__ import annotations
import sys, logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sers.config import RESULTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(levelname)-7s │ %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

OUTPUT_DIR = RESULTS_DIR / "multiview_derivative"


# =============================================================================
# Core: derivative channel computation
# =============================================================================
def compute_derivative_channels(
    spectrum: np.ndarray,
    window_length: int = 11,
    polyorder: int = 3,
    normalize: bool = True,
) -> np.ndarray:
    """
    SERS spectrum → 3-channel multi-view array.

    Args:
        spectrum: (N,) preprocessed SERS spectrum (already SNV-normalized)
        window_length: SG filter window (odd)
        polyorder: SG polynomial order
        normalize: if True, per-channel min-max normalization

    Returns:
        (3, N) array [raw_smooth, d1, d2]
    """
    x_smooth = savgol_filter(spectrum, window_length, polyorder, deriv=0)
    x_d1 = savgol_filter(spectrum, window_length, polyorder, deriv=1)
    x_d2 = savgol_filter(spectrum, window_length, polyorder, deriv=2)

    if normalize:
        def _norm(x):
            mn, mx = x.min(), x.max()
            return (x - mn) / (mx - mn + 1e-8)
        x_smooth, x_d1, x_d2 = _norm(x_smooth), _norm(x_d1), _norm(x_d2)

    return np.stack([x_smooth, x_d1, x_d2], axis=0)


def find_zero_crossings(signal: np.ndarray) -> np.ndarray:
    """Find indices where signal crosses zero (sign change)."""
    signs = np.sign(signal)
    crossings = np.where(np.diff(signs) != 0)[0]
    return crossings


# =============================================================================
# Visualization
# =============================================================================
def plot_single_sample_3ch(wavenumbers, spectrum, group, sid, output_dir):
    """Show raw, d1, d2 for a single sample with zero-crossings marked."""
    channels = compute_derivative_channels(spectrum, normalize=False)

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    titles = ["Raw (SG smoothed)", "1st Derivative (d1)", "2nd Derivative (d2)"]
    colors = ["#2196F3", "#E91E63", "#FF9800"]

    for i, (ax, title, color) in enumerate(zip(axes, titles, colors)):
        ax.plot(wavenumbers, channels[i], color=color, linewidth=0.8)
        ax.set_ylabel(title, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)

        if i > 0:  # mark zero-crossings on derivatives
            zc = find_zero_crossings(channels[i])
            ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
            for idx in zc:
                ax.axvline(wavenumbers[idx], color="gray", alpha=0.15, linewidth=0.5)
            ax.set_title(f"{len(zc)} zero-crossings", fontsize=8, loc="right", color="gray")

    axes[-1].set_xlabel("Raman Shift (cm⁻¹)")
    fig.suptitle(f"{group} #{sid} — Multi-View Derivative Channels", fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / f"sample_{group}_{sid}_3ch.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_group_mean_3ch(df, feat_cols, wavenumbers, output_dir):
    """Plot group-mean spectra for all 3 channels, all groups on one figure."""
    groups = sorted(df["group"].unique())
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    ch_names = ["Raw (SG smoothed)", "1st Derivative", "2nd Derivative"]

    # Color palette
    cmap = plt.cm.get_cmap("tab10", len(groups))
    group_colors = {g: cmap(i) for i, g in enumerate(groups)}

    for ch_idx, (ax, ch_name) in enumerate(zip(axes, ch_names)):
        for g in groups:
            mask = df["group"] == g
            spectra = df.loc[mask, feat_cols].values
            # Compute derivative channels for each spectrum, take mean
            ch_data = []
            for s in spectra:
                channels = compute_derivative_channels(s, normalize=False)
                ch_data.append(channels[ch_idx])
            mean_ch = np.mean(ch_data, axis=0)
            ax.plot(wavenumbers, mean_ch, color=group_colors[g],
                    linewidth=1, alpha=0.8, label=g)

        ax.set_ylabel(ch_name, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)
        if ch_idx > 0:
            ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    axes[0].legend(loc="upper right", fontsize=8, ncol=3)
    axes[-1].set_xlabel("Raman Shift (cm⁻¹)")
    fig.suptitle("Group Mean Spectra — Multi-View Derivative Channels", fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / "group_mean_3ch.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved group_mean_3ch.png")


def plot_zero_crossing_analysis(df, feat_cols, wavenumbers, output_dir):
    """Analyze d1 zero-crossings: where are peaks across groups?"""
    groups = sorted(df["group"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: histogram of zero-crossing positions for cancer vs non-cancer
    cancer_groups = {"PRO", "BRE", "OVA", "LUN", "CRC", "CPAN", "BLC"}
    zc_cancer, zc_noncancer = [], []

    for _, row in df.iterrows():
        spectrum = row[feat_cols].values.astype(float)
        d1 = savgol_filter(spectrum, 11, 3, deriv=1)
        zc_idx = find_zero_crossings(d1)
        wn_crossings = wavenumbers[zc_idx]
        if row["group"] in cancer_groups:
            zc_cancer.extend(wn_crossings)
        else:
            zc_noncancer.extend(wn_crossings)

    ax = axes[0]
    ax.hist(zc_cancer, bins=80, alpha=0.6, color="#E53935", label=f"Cancer (n={len(zc_cancer)})", density=True)
    ax.hist(zc_noncancer, bins=80, alpha=0.6, color="#43A047", label=f"Non-cancer (n={len(zc_noncancer)})", density=True)
    ax.set_xlabel("Raman Shift (cm⁻¹)")
    ax.set_ylabel("Density")
    ax.set_title("d1 Zero-Crossing Distribution (= Peak Positions)", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: mean number of zero-crossings per group
    ax = axes[1]
    group_zc_counts = {}
    for g in groups:
        mask = df["group"] == g
        counts = []
        for _, row in df[mask].iterrows():
            spectrum = row[feat_cols].values.astype(float)
            d1 = savgol_filter(spectrum, 11, 3, deriv=1)
            counts.append(len(find_zero_crossings(d1)))
        group_zc_counts[g] = counts

    means = [np.mean(group_zc_counts[g]) for g in groups]
    stds = [np.std(group_zc_counts[g]) for g in groups]
    colors = ["#E53935" if g in cancer_groups else "#43A047" for g in groups]
    ax.barh(range(len(groups)), means, xerr=stds, color=colors, alpha=0.8, capsize=3)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups)
    ax.set_xlabel("Mean Zero-Crossings (d1)")
    ax.set_title("Peak Count by Group", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "zero_crossing_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved zero_crossing_analysis.png")


def plot_channel_correlation(df, feat_cols, output_dir):
    """Show inter-channel correlation distribution."""
    correlations = {"raw-d1": [], "raw-d2": [], "d1-d2": []}
    for _, row in df.sample(min(500, len(df)), random_state=42).iterrows():
        spectrum = row[feat_cols].values.astype(float)
        ch = compute_derivative_channels(spectrum, normalize=True)
        correlations["raw-d1"].append(np.corrcoef(ch[0], ch[1])[0, 1])
        correlations["raw-d2"].append(np.corrcoef(ch[0], ch[2])[0, 1])
        correlations["d1-d2"].append(np.corrcoef(ch[1], ch[2])[0, 1])

    fig, ax = plt.subplots(figsize=(8, 4))
    labels = list(correlations.keys())
    data = [correlations[k] for k in labels]
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    colors = ["#2196F3", "#E91E63", "#FF9800"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Pearson Correlation")
    ax.set_title("Inter-Channel Correlation (n=500 samples)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for i, k in enumerate(labels):
        mean_corr = np.mean(correlations[k])
        ax.text(i + 1, mean_corr + 0.02, f"μ={mean_corr:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(output_dir / "channel_correlation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved channel_correlation.png")


# =============================================================================
# Main
# =============================================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  Multi-View Derivative — Preprocessing Validation")
    logger.info("=" * 60)

    # Load Thermo processed spectra
    csv_path = RESULTS_DIR / "processed_spectra.csv"
    logger.info(f"\nLoading {csv_path}...")
    df = pd.read_csv(csv_path)
    feat_cols = [c for c in df.columns if c.startswith("x_")]
    wavenumbers = np.array([float(c.split("_")[1]) for c in feat_cols])
    logger.info(f"  {len(df)} spectra, {len(feat_cols)} features, "
                f"range {wavenumbers[0]:.0f}-{wavenumbers[-1]:.0f} cm⁻¹")

    # 1. Single sample visualization (one per group)
    logger.info("\n[1] Single sample 3-channel visualization...")
    for g in sorted(df["group"].unique()):
        sub = df[df["group"] == g]
        row = sub.iloc[0]
        spectrum = row[feat_cols].values.astype(float)
        plot_single_sample_3ch(wavenumbers, spectrum, g, row["sample_id"], OUTPUT_DIR)
    logger.info(f"  Saved {df['group'].nunique()} sample plots")

    # 2. Group mean 3-channel comparison
    logger.info("\n[2] Group mean 3-channel plot...")
    plot_group_mean_3ch(df, feat_cols, wavenumbers, OUTPUT_DIR)

    # 3. Zero-crossing analysis
    logger.info("\n[3] Zero-crossing analysis...")
    plot_zero_crossing_analysis(df, feat_cols, wavenumbers, OUTPUT_DIR)

    # 4. Channel correlation
    logger.info("\n[4] Inter-channel correlation...")
    plot_channel_correlation(df, feat_cols, OUTPUT_DIR)

    # 5. Statistics summary
    logger.info("\n[5] Channel statistics summary...")
    sample = df.sample(min(500, len(df)), random_state=42)
    stats = {"channel": [], "mean_range": [], "mean_std": [], "mean_n_zero_crossings": []}
    ch_names = ["raw_smooth", "d1", "d2"]

    for ch_idx, ch_name in enumerate(ch_names):
        ranges, stds, zcs = [], [], []
        for _, row in sample.iterrows():
            spectrum = row[feat_cols].values.astype(float)
            ch = compute_derivative_channels(spectrum, normalize=False)
            ranges.append(ch[ch_idx].max() - ch[ch_idx].min())
            stds.append(ch[ch_idx].std())
            if ch_idx > 0:
                zcs.append(len(find_zero_crossings(ch[ch_idx])))
            else:
                zcs.append(0)
        stats["channel"].append(ch_name)
        stats["mean_range"].append(np.mean(ranges))
        stats["mean_std"].append(np.mean(stds))
        stats["mean_n_zero_crossings"].append(np.mean(zcs) if ch_idx > 0 else "-")

    stats_df = pd.DataFrame(stats)
    logger.info(f"\n{stats_df.to_string(index=False)}")
    stats_df.to_csv(OUTPUT_DIR / "channel_statistics.csv", index=False)

    logger.info(f"\n  Output: {OUTPUT_DIR}")
    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
