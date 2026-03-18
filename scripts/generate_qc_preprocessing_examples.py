"""
Generate example plots for each QC and Preprocessing step using real SERS data.

Outputs:
    results/figures/qc_preprocessing_examples.png  -- Combined figure
    results/figures/step_*.png                     -- Individual step plots

Each panel shows BEFORE/AFTER with real spectra so the pipeline can be explained visually.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.sers import read_spectrum
from src.sers.io import find_spectra, make_common_grid
from src.sers.preprocessing import (
    trim_spectrum, smooth, baseline_correction, snv, minmax_scale,
    FINGERPRINT_REGION,
)
from src.sers.qc.qc import (
    calculate_intensity_gate,
    calculate_replicate_qc,
    _interpolate_to_grid,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Style ──
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "#FAFAFA",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

COLORS = {
    "raw": "#1976D2",
    "after": "#E53935",
    "good": "#43A047",
    "bad": "#E53935",
    "threshold": "#FF9800",
    "highlight": "#9C27B0",
}


def load_sample_spectra(data_dir, group_folder, n_samples=3, n_reps=5):
    """Load a few samples with all replicates from a specific group."""
    folder = data_dir / group_folder
    files = sorted(find_spectra(folder, pattern="*.CSV", recursive=False))
    if not files:
        files = sorted(find_spectra(folder, pattern="*.csv", recursive=False))

    # Group by sample
    samples = {}
    for fp in files:
        name = fp.stem  # e.g. "NOR 1_1"
        parts = name.rsplit("_", 1)
        if len(parts) == 2:
            sample_name, rep = parts[0], parts[1]
            samples.setdefault(sample_name, []).append(fp)

    # Pick samples that have all replicates
    good_samples = {k: v for k, v in samples.items() if len(v) >= n_reps}
    picked = list(good_samples.items())[:n_samples]

    result = {}
    for sample_name, fps in picked:
        spectra = []
        for fp in sorted(fps)[:n_reps]:
            x, y = read_spectrum(fp)
            spectra.append((x, y))
        result[sample_name] = spectra
    return result


def main():
    data_dir = PROJECT_ROOT / "data" / "raw_data"
    fig_dir = PROJECT_ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("Loading real spectra...")

    # Load Normal and Cancer samples
    normal_data = load_sample_spectra(data_dir, "5. Normal (100\uac1c)", n_samples=2)
    cancer_data = load_sample_spectra(data_dir, "1. Prostate cancer (100\uac1c)", n_samples=2)

    # Pick one sample for step-by-step demo
    demo_name = list(normal_data.keys())[0]
    demo_spectra = normal_data[demo_name]
    x_demo, y_demo = demo_spectra[0]  # First replicate

    print(f"Demo sample: {demo_name} ({len(demo_spectra)} replicates)")
    print(f"  Wavenumber range: {x_demo.min():.1f} - {x_demo.max():.1f} cm-1")
    print(f"  Points: {len(x_demo)}")

    # ================================================================
    # Figure 1: Raw Spectrum Overview
    # ================================================================
    fig1, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(x_demo, y_demo, color=COLORS["raw"], linewidth=0.8)
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.set_title(f"Raw SERS Spectrum: {demo_name}")
    ax.axvspan(400, 2200, alpha=0.08, color="green", label="Fingerprint region (400-2200)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig1.savefig(fig_dir / "step0_raw_spectrum.png", dpi=150)
    plt.close(fig1)
    print("Saved: step0_raw_spectrum.png")

    # ================================================================
    # Figure 2: QC Level 0 - Intensity Gate
    # ================================================================
    # Load many spectra to show distribution
    print("\nLoading spectra for Intensity Gate demo...")
    all_spectra = {}
    folders = [
        ("5. Normal (100\uac1c)", "NOR"),
        ("1. Prostate cancer (100\uac1c)", "PRO"),
        ("6. Diabetes (100\uac1c)", "DIA"),
    ]
    for folder, grp in folders:
        folder_path = data_dir / folder
        files = sorted(find_spectra(folder_path, pattern="*.CSV", recursive=False))[:50]
        for fp in files:
            name = fp.stem
            parts = name.rsplit("_", 1)
            if len(parts) == 2:
                x, y = read_spectrum(fp)
                key = (grp, parts[0], parts[1])
                all_spectra[key] = (x, y)

    gate_df = calculate_intensity_gate(all_spectra)
    threshold = gate_df["gate_threshold"].iloc[0]
    median_fp = gate_df["fp_mean"].median()

    fig2, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: histogram of fp_mean
    ax = axes[0]
    for grp, color in [("NOR", "#1976D2"), ("PRO", "#E53935"), ("DIA", "#43A047")]:
        vals = gate_df[gate_df["group"] == grp]["fp_mean"]
        ax.hist(vals, bins=20, alpha=0.5, color=color, label=grp, edgecolor="white")
    ax.axvline(threshold, color=COLORS["threshold"], linestyle="--", linewidth=2,
               label=f"Threshold = {threshold:.1f}\n(median x 0.1)")
    ax.axvline(median_fp, color="gray", linestyle=":", linewidth=1.5,
               label=f"Median = {median_fp:.1f}")
    ax.set_xlabel("Fingerprint Mean Intensity (fp_mean)")
    ax.set_ylabel("Count")
    ax.set_title("QC Level 0: Intensity Gate")
    ax.legend(fontsize=8)

    # Right: example spectra (high vs low intensity)
    ax = axes[1]
    high_row = gate_df.nlargest(1, "fp_mean").iloc[0]
    low_row = gate_df.nsmallest(1, "fp_mean").iloc[0]
    high_key = (high_row["group"], high_row["sample_id"], high_row["replicate"])
    low_key = (low_row["group"], low_row["sample_id"], low_row["replicate"])

    x_h, y_h = all_spectra[high_key]
    x_l, y_l = all_spectra[low_key]
    ax.plot(x_h, y_h, color=COLORS["good"], linewidth=0.8, alpha=0.8,
            label=f"High intensity ({high_key[0]} {high_key[1]})\nfp_mean={high_row['fp_mean']:.1f}")
    ax.plot(x_l, y_l, color=COLORS["bad"], linewidth=0.8, alpha=0.8,
            label=f"Low intensity ({low_key[0]} {low_key[1]})\nfp_mean={low_row['fp_mean']:.1f}")
    ax.axvspan(400, 2200, alpha=0.05, color="green")
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Intensity")
    ax.set_title("Intensity Gate: High vs Low Enhancement")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig2.savefig(fig_dir / "step1_qc_intensity_gate.png", dpi=150)
    plt.close(fig2)
    print("Saved: step1_qc_intensity_gate.png")

    # ================================================================
    # Figure 3: QC Level 1 - Replicate RSD & Correlation
    # ================================================================
    print("\nCalculating Replicate QC...")
    x_arrays = [x for x, y in all_spectra.values()]
    common_grid = make_common_grid(x_arrays)
    qc_stats = calculate_replicate_qc(all_spectra, common_grid)

    fig3, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Left: RSD distribution
    ax = axes[0]
    for grp, color in [("NOR", "#1976D2"), ("PRO", "#E53935"), ("DIA", "#43A047")]:
        vals = qc_stats[qc_stats["group"] == grp]["mean_rsd"]
        if len(vals) > 0:
            ax.hist(vals, bins=15, alpha=0.5, color=color, label=grp, edgecolor="white")
    ax.axvline(5.0, color=COLORS["threshold"], linestyle="--", linewidth=2,
               label="Threshold (RSD < 5%)")
    ax.set_xlabel("Mean RSD (%)")
    ax.set_ylabel("Count")
    ax.set_title("QC Level 1: Replicate RSD")
    ax.legend(fontsize=8)

    # Middle: Correlation distribution
    ax = axes[1]
    for grp, color in [("NOR", "#1976D2"), ("PRO", "#E53935"), ("DIA", "#43A047")]:
        vals = qc_stats[qc_stats["group"] == grp]["mean_corr"]
        if len(vals) > 0:
            ax.hist(vals, bins=15, alpha=0.5, color=color, label=grp, edgecolor="white")
    ax.axvline(0.95, color=COLORS["threshold"], linestyle="--", linewidth=2,
               label="Threshold (Corr > 0.95)")
    ax.set_xlabel("Mean Pairwise Correlation")
    ax.set_ylabel("Count")
    ax.set_title("QC Level 1: Replicate Correlation")
    ax.legend(fontsize=8)

    # Right: Example replicates overlay (good sample)
    ax = axes[2]
    # Pick a sample with good RSD and show its 5 replicates
    good_sample = qc_stats.nsmallest(1, "mean_rsd").iloc[0]
    good_key = (good_sample["group"], good_sample["sample_id"])
    rep_colors = ["#1976D2", "#E53935", "#43A047", "#FF9800", "#9C27B0"]
    n_plotted = 0
    for (grp, sid, rep), (x, y) in sorted(all_spectra.items()):
        if (grp, sid) == good_key:
            y_interp = _interpolate_to_grid(x, y, common_grid)
            ax.plot(common_grid, y_interp, color=rep_colors[n_plotted % 5],
                    linewidth=0.7, alpha=0.7, label=f"Rep {rep}")
            n_plotted += 1
            if n_plotted >= 5:
                break
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Intensity")
    ax.set_title(f"5 Replicates: {good_key[0]} {good_key[1]}\n"
                 f"RSD={good_sample['mean_rsd']:.2f}%, Corr={good_sample['mean_corr']:.4f}")
    ax.legend(fontsize=8, ncol=2)

    plt.tight_layout()
    fig3.savefig(fig_dir / "step2_qc_replicate.png", dpi=150)
    plt.close(fig3)
    print("Saved: step2_qc_replicate.png")

    # ================================================================
    # Figure 4: Preprocessing Step-by-Step
    # ================================================================
    print("\nGenerating preprocessing step-by-step...")

    # Use demo spectrum
    x_raw, y_raw = x_demo.copy(), y_demo.copy()

    # Step 1: Trim
    x_trim, y_trim = trim_spectrum(x_raw, y_raw, region=FINGERPRINT_REGION)

    # Step 2: Smooth
    y_smooth = smooth(y_trim, window_length=11, polyorder=3)

    # Step 3: Baseline correction
    y_baseline = baseline_correction(y_smooth, window=101)

    # Step 4a: SNV normalization
    y_snv = snv(y_baseline)

    # Step 4b: MinMax normalization (for comparison)
    y_minmax = minmax_scale(y_baseline)

    fig4, axes = plt.subplots(2, 3, figsize=(16, 9))

    # (0,0) Raw spectrum
    ax = axes[0, 0]
    ax.plot(x_raw, y_raw, color=COLORS["raw"], linewidth=0.8)
    ax.axvspan(400, 2200, alpha=0.08, color="green")
    ax.axvline(400, color="green", linestyle="--", linewidth=1, alpha=0.5)
    ax.axvline(2200, color="green", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_title("(1) Raw Spectrum")
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Intensity")
    ax.text(0.02, 0.95, f"Full range: {x_raw.min():.0f}-{x_raw.max():.0f} cm$^{{-1}}$\n"
                         f"Points: {len(x_raw)}",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # (0,1) After trim
    ax = axes[0, 1]
    ax.plot(x_raw, y_raw, color="#BBBBBB", linewidth=0.5, label="Before (full)")
    ax.plot(x_trim, y_trim, color=COLORS["after"], linewidth=0.8, label="After trim")
    ax.set_title("(2) Trim to Fingerprint Region (400-2200 cm$^{-1}$)")
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Intensity")
    ax.legend(fontsize=8)
    ax.text(0.02, 0.95, f"Trimmed: {len(x_raw)} -> {len(x_trim)} points\n"
                         f"Removed: {len(x_raw)-len(x_trim)} points",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # (0,2) After smooth
    ax = axes[0, 2]
    ax.plot(x_trim, y_trim, color="#BBBBBB", linewidth=0.5, label="Before (trimmed)")
    ax.plot(x_trim, y_smooth, color=COLORS["after"], linewidth=0.8, label="After smooth")
    ax.set_title("(3) Savitzky-Golay Smoothing (window=11, poly=3)")
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Intensity")
    ax.legend(fontsize=8)

    # Zoom inset to show smoothing effect
    # Find a noisy region and zoom in
    mid = len(x_trim) // 2
    zoom_slice = slice(mid - 30, mid + 30)
    axins = ax.inset_axes([0.55, 0.5, 0.4, 0.4])
    axins.plot(x_trim[zoom_slice], y_trim[zoom_slice], color="#BBBBBB", linewidth=0.8)
    axins.plot(x_trim[zoom_slice], y_smooth[zoom_slice], color=COLORS["after"], linewidth=1.2)
    axins.set_title("Zoom", fontsize=7)
    axins.tick_params(labelsize=6)

    # (1,0) Baseline correction
    ax = axes[1, 0]
    # Show rolling minimum baseline
    baseline_est = pd.Series(y_smooth).rolling(101, center=True, min_periods=1).min().to_numpy()
    ax.plot(x_trim, y_smooth, color="#BBBBBB", linewidth=0.5, label="Smoothed")
    ax.fill_between(x_trim, 0, baseline_est, alpha=0.2, color=COLORS["threshold"],
                    label="Estimated baseline")
    ax.plot(x_trim, baseline_est, color=COLORS["threshold"], linewidth=1, linestyle="--")
    ax.plot(x_trim, y_baseline, color=COLORS["after"], linewidth=0.8, label="After correction")
    ax.set_title("(4) Baseline Correction (rolling min, window=101)")
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Intensity")
    ax.legend(fontsize=8)

    # (1,1) SNV normalization
    ax = axes[1, 1]
    ax.plot(x_trim, y_baseline, color="#BBBBBB", linewidth=0.5, label="Before (baseline-corrected)")
    ax.plot(x_trim, y_snv, color=COLORS["after"], linewidth=0.8, label="After SNV")
    ax.set_title("(5a) SNV Normalization: (y - mean) / std")
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Normalized Intensity")
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.5)
    ax.legend(fontsize=8)
    ax.text(0.02, 0.95, f"mean={y_snv.mean():.4f}\nstd={y_snv.std():.4f}",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # (1,2) MinMax normalization (for comparison)
    ax = axes[1, 2]
    ax.plot(x_trim, y_baseline, color="#BBBBBB", linewidth=0.5, label="Before (baseline-corrected)")
    ax.plot(x_trim, y_minmax, color="#9C27B0", linewidth=0.8, label="After MinMax")
    ax.set_title("(5b) MinMax Normalization: (y - min) / (max - min)")
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Normalized Intensity [0, 1]")
    ax.legend(fontsize=8)
    ax.text(0.02, 0.95, f"range=[{y_minmax.min():.2f}, {y_minmax.max():.2f}]",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.suptitle(f"Preprocessing Pipeline: {demo_name}", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig4.savefig(fig_dir / "step3_preprocessing_pipeline.png", dpi=150, bbox_inches="tight")
    plt.close(fig4)
    print("Saved: step3_preprocessing_pipeline.png")

    # ================================================================
    # Figure 5: SNV Effect - Why it makes data linearly separable
    # ================================================================
    print("\nGenerating SNV comparison (Cancer vs Normal)...")

    fig5, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Load and preprocess Normal vs Cancer
    common_grid_trim = common_grid[(common_grid >= 400) & (common_grid <= 2200)]

    def preprocess_full(x, y, grid):
        """Apply full pipeline."""
        xt, yt = trim_spectrum(x, y, FINGERPRINT_REGION)
        ys = smooth(yt)
        yb = baseline_correction(ys)
        y_snv_out = snv(yb)
        y_mm_out = minmax_scale(yb)
        y_interp_snv = np.interp(grid, xt, y_snv_out)
        y_interp_mm = np.interp(grid, xt, y_mm_out)
        y_interp_raw = np.interp(grid, xt, yb)
        return y_interp_raw, y_interp_snv, y_interp_mm

    # Process all loaded spectra
    nor_raw, nor_snv_list, nor_mm = [], [], []
    pro_raw, pro_snv_list, pro_mm = [], [], []

    for sample_name, spectra in normal_data.items():
        for x, y in spectra:
            yr, ys, ym = preprocess_full(x, y, common_grid_trim)
            nor_raw.append(yr)
            nor_snv_list.append(ys)
            nor_mm.append(ym)

    for sample_name, spectra in cancer_data.items():
        for x, y in spectra:
            yr, ys, ym = preprocess_full(x, y, common_grid_trim)
            pro_raw.append(yr)
            pro_snv_list.append(ys)
            pro_mm.append(ym)

    # (0,0) Raw baseline-corrected: Normal
    ax = axes[0, 0]
    for y in nor_raw[:5]:
        ax.plot(common_grid_trim, y, color="#1976D2", linewidth=0.5, alpha=0.5)
    for y in pro_raw[:5]:
        ax.plot(common_grid_trim, y, color="#E53935", linewidth=0.5, alpha=0.5)
    ax.plot([], [], color="#1976D2", label="Normal (NOR)")
    ax.plot([], [], color="#E53935", label="Prostate Cancer (PRO)")
    ax.set_title("Before Normalization\n(Baseline-corrected only)")
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Intensity")
    ax.legend(fontsize=8)
    ax.text(0.02, 0.95, "Intensity scale varies\nbetween measurements\n(SERS hot-spot effect)",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))

    # (0,1) After SNV: Normal vs Cancer
    ax = axes[0, 1]
    for y in nor_snv_list[:5]:
        ax.plot(common_grid_trim, y, color="#1976D2", linewidth=0.5, alpha=0.5)
    for y in pro_snv_list[:5]:
        ax.plot(common_grid_trim, y, color="#E53935", linewidth=0.5, alpha=0.5)
    ax.plot([], [], color="#1976D2", label="Normal (NOR)")
    ax.plot([], [], color="#E53935", label="Prostate Cancer (PRO)")
    ax.set_title("After SNV Normalization")
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Normalized Intensity")
    ax.legend(fontsize=8)
    ax.text(0.02, 0.95, "Scale normalized\nPeak RATIOS preserved\n-> Linearly separable",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.9))

    # (1,0) Mean spectra comparison
    ax = axes[1, 0]
    nor_mean = np.mean(nor_snv_list, axis=0)
    pro_mean = np.mean(pro_snv_list, axis=0)
    nor_std = np.std(nor_snv_list, axis=0)
    pro_std = np.std(pro_snv_list, axis=0)

    ax.plot(common_grid_trim, nor_mean, color="#1976D2", linewidth=1.5, label="Normal mean")
    ax.fill_between(common_grid_trim, nor_mean - nor_std, nor_mean + nor_std,
                    color="#1976D2", alpha=0.15)
    ax.plot(common_grid_trim, pro_mean, color="#E53935", linewidth=1.5, label="Cancer mean")
    ax.fill_between(common_grid_trim, pro_mean - pro_std, pro_mean + pro_std,
                    color="#E53935", alpha=0.15)
    ax.set_title("Mean SNV Spectra (with std band)")
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Normalized Intensity")
    ax.legend(fontsize=8)

    # (1,1) Difference spectrum
    ax = axes[1, 1]
    diff = pro_mean - nor_mean
    ax.plot(common_grid_trim, diff, color="#9C27B0", linewidth=1)
    ax.fill_between(common_grid_trim, 0, diff,
                    where=(diff > 0), color="#E53935", alpha=0.3, label="Cancer > Normal")
    ax.fill_between(common_grid_trim, 0, diff,
                    where=(diff < 0), color="#1976D2", alpha=0.3, label="Normal > Cancer")
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.5)
    ax.set_title("Difference: Cancer - Normal (SNV)")
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Intensity Difference")
    ax.legend(fontsize=8)

    # Mark key peaks
    peak_regions = [
        (630, "630\nC-S stretch"),
        (725, "725\nAdenine"),
        (1003, "1003\nPhe"),
        (1330, "1330\nCH def"),
        (1450, "1450\nCH$_2$ bend"),
        (1590, "1590\nAmide II"),
    ]
    for wn, label in peak_regions:
        idx = np.argmin(np.abs(common_grid_trim - wn))
        if idx < len(diff):
            ax.annotate(label, xy=(wn, diff[idx]),
                       fontsize=6, ha="center", va="bottom" if diff[idx] > 0 else "top",
                       color="#333333")

    plt.suptitle("SNV Normalization Effect: Cancer vs Normal Spectra", fontsize=14,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    fig5.savefig(fig_dir / "step4_snv_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig5)
    print("Saved: step4_snv_comparison.png")

    # ================================================================
    # Figure 6: Complete Pipeline Summary (single-page overview)
    # ================================================================
    print("\nGenerating pipeline summary...")

    fig6, axes = plt.subplots(2, 4, figsize=(20, 8))

    # Row 1: QC steps
    # (0,0) Raw data loading
    ax = axes[0, 0]
    for i, (x, y) in enumerate(demo_spectra[:5]):
        ax.plot(x, y, linewidth=0.6, alpha=0.6, color=rep_colors[i], label=f"Rep {i+1}")
    ax.set_title("Step 1: Load Raw Data\n(5 replicates per sample)")
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Intensity")
    ax.legend(fontsize=7, ncol=2)

    # (0,1) Intensity gate
    ax = axes[0, 1]
    fp_means = gate_df["fp_mean"].values
    passed = gate_df[gate_df["gate_pass"]]["fp_mean"]
    failed = gate_df[~gate_df["gate_pass"]]["fp_mean"]
    ax.hist(passed, bins=20, color=COLORS["good"], alpha=0.7, label=f"Pass ({len(passed)})")
    if len(failed) > 0:
        ax.hist(failed, bins=5, color=COLORS["bad"], alpha=0.7, label=f"Fail ({len(failed)})")
    ax.axvline(threshold, color=COLORS["threshold"], linestyle="--", linewidth=2,
               label=f"Gate = {threshold:.1f}")
    ax.set_title("Step 2: QC Level 0\nIntensity Gate")
    ax.set_xlabel("fp_mean")
    ax.legend(fontsize=7)

    # (0,2) RSD check
    ax = axes[0, 2]
    rsd_vals = qc_stats["mean_rsd"]
    ax.hist(rsd_vals, bins=20, color=COLORS["good"], alpha=0.7, edgecolor="white")
    ax.axvline(5.0, color=COLORS["threshold"], linestyle="--", linewidth=2,
               label="Threshold = 5%")
    ax.set_title("Step 3: QC Level 1\nReplicate RSD (%)")
    ax.set_xlabel("Mean RSD (%)")
    ax.legend(fontsize=7)

    # (0,3) Correlation check
    ax = axes[0, 3]
    corr_vals = qc_stats["mean_corr"]
    ax.hist(corr_vals, bins=20, color=COLORS["good"], alpha=0.7, edgecolor="white")
    ax.axvline(0.95, color=COLORS["threshold"], linestyle="--", linewidth=2,
               label="Threshold = 0.95")
    ax.set_title("Step 4: QC Level 1\nReplicate Correlation")
    ax.set_xlabel("Mean Pairwise Correlation")
    ax.legend(fontsize=7)

    # Row 2: Preprocessing steps
    # (1,0) Trim
    ax = axes[1, 0]
    ax.plot(x_raw, y_raw, color="#BBBBBB", linewidth=0.4)
    ax.plot(x_trim, y_trim, color=COLORS["after"], linewidth=0.7)
    ax.set_title("Step 5: Trim\n(400-2200 cm$^{-1}$)")
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")

    # (1,1) Smooth
    ax = axes[1, 1]
    ax.plot(x_trim, y_trim, color="#BBBBBB", linewidth=0.4)
    ax.plot(x_trim, y_smooth, color=COLORS["after"], linewidth=0.7)
    ax.set_title("Step 6: Smooth\n(Savitzky-Golay)")
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")

    # (1,2) Baseline
    ax = axes[1, 2]
    ax.plot(x_trim, y_smooth, color="#BBBBBB", linewidth=0.4)
    ax.plot(x_trim, baseline_est, color=COLORS["threshold"], linewidth=0.7, linestyle="--")
    ax.plot(x_trim, y_baseline, color=COLORS["after"], linewidth=0.7)
    ax.set_title("Step 7: Baseline Correction\n(Rolling minimum)")
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")

    # (1,3) Normalize
    ax = axes[1, 3]
    ax.plot(x_trim, y_snv, color=COLORS["after"], linewidth=0.7, label="SNV")
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.5)
    ax.set_title("Step 8: Normalize\n(SNV: zero mean, unit std)")
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.legend(fontsize=7)

    # Add stage labels
    fig6.text(0.5, 0.98, "QC Pipeline (Raw Data)", ha="center", fontsize=13,
              fontweight="bold", color="#1976D2",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="#E3F2FD", alpha=0.8))
    fig6.text(0.5, 0.48, "Preprocessing Pipeline (QC-Passed Only)", ha="center", fontsize=13,
              fontweight="bold", color="#E53935",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFEBEE", alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig6.savefig(fig_dir / "pipeline_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig6)
    print("Saved: pipeline_overview.png")

    print(f"\nAll figures saved to: {fig_dir}/")
    print("Files:")
    for f in sorted(fig_dir.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
