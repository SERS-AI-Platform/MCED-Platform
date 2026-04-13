"""
Export per-group mean SERS spectra as individual PNG images.

Generates white-background, publication-quality spectrum plots for each
disease group, plus a combined overlay plot.

Output: results/figures/group_spectra/
  - SERS_spectrum_{GROUP}.png (per group, with ±1SD band)
  - SERS_spectrum_all_overlay.png (all groups on one plot)
  - SERS_spectrum_cancer_overlay.png (cancer groups only)
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT)

SERS_CSV = os.path.join(PROJECT, "results", "processed_spectra.csv")
OUT_DIR = "/home/user/workspace/solum-dashboard/exported_spectra"
os.makedirs(OUT_DIR, exist_ok=True)

WMIN, WMAX = 400, 1800

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

CANCER_TYPES = ["PRO", "BRE", "OVA", "LUN", "CRC", "PAN", "BLC"]
# Aliases and related groups
CANCER_ALIASES = {"CPAN", "YPAN", "SPAN"}  # SPAN=post-op, but still cancer origin
NON_CANCER = {"NOR", "DIA", "HBP", "H.D.", "YNOR"}
# Note: SPAN is post-operative → excluded from model but biologically cancer

SERS_PEAKS = {
    618: "C-S", 683: "Creatinine", 724: "Adenine",
    849: "Tyr", 999: "Phe", 1148: "C-N",
    1231: "Amide III", 1352: "CH def", 1449: "CH₂",
    1597: "C=C/Purine", 1651: "Amide I",
}

DPI = 200


def load_data():
    df = pd.read_csv(SERS_CSV)
    wn_cols = [c for c in df.columns if c.startswith("x_")]
    wn = np.array([float(c.replace("x_", "")) for c in wn_cols])
    mask = (wn >= WMIN) & (wn <= WMAX)
    wn_fp = wn[mask]
    fp_cols = [c for c, m in zip(wn_cols, mask) if m]
    return df, wn_fp, fp_cols


def plot_single_group(wn, mean, std, group, count, path):
    """Plot a single group spectrum with ±1SD band and full legend."""
    fig, ax = plt.subplots(figsize=(14, 5.5))
    color = GROUP_COLORS.get(group, "#333")
    label_name = GROUP_LABELS.get(group, group)

    ax.fill_between(wn, mean - std, mean + std, alpha=0.15, color=color,
                     label=f"±1 SD")
    ax.plot(wn, mean, color=color, linewidth=1.8,
            label=f"{group} ({label_name}) Mean, n={count}")

    # Peak markers
    for pk_wn, pk_label in SERS_PEAKS.items():
        if WMIN <= pk_wn <= WMAX:
            idx = np.argmin(np.abs(wn - pk_wn))
            ax.axvline(pk_wn, color="#bbbbbb", linewidth=0.6, alpha=0.5)
            ax.annotate(f"{pk_wn}\n{pk_label}", (wn[idx], mean[idx]),
                        textcoords="offset points", xytext=(0, 14),
                        fontsize=6.5, ha="center", color="#555", alpha=0.9)

    ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=12)
    ax.set_ylabel("Intensity (SNV)", fontsize=12)
    ax.set_title(f"SERS Mean Spectrum — {group} ({label_name}, n={count})",
                 fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9,
              edgecolor="#ddd", fancybox=True)
    ax.grid(True, alpha=0.15)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    plt.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_overlay(wn, group_data, groups, title, filename):
    """Plot multiple groups overlaid on one chart with full legend."""
    fig, ax = plt.subplots(figsize=(16, 6.5))

    for g in groups:
        if g not in group_data:
            continue
        mean, std, count = group_data[g]
        color = GROUP_COLORS.get(g, "#333")
        label_name = GROUP_LABELS.get(g, g)
        ax.plot(wn, mean, color=color, linewidth=1.5, alpha=0.9,
                label=f"{g} ({label_name}, n={count})")

    for pk_wn, pk_label in SERS_PEAKS.items():
        if WMIN <= pk_wn <= WMAX:
            ax.axvline(pk_wn, color="#cccccc", linewidth=0.5, alpha=0.4)

    ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=12)
    ax.set_ylabel("Intensity (SNV)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="upper right", fontsize=9, ncol=2, framealpha=0.9,
              edgecolor="#ddd", fancybox=True)
    ax.grid(True, alpha=0.15)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    print("=" * 60)
    print("Export Group Spectra as PNG")
    print("=" * 60)

    df, wn, fp_cols = load_data()
    groups = sorted(df["group"].unique())
    print(f"Groups: {groups}")
    print(f"Wavenumber: {wn[0]:.0f}-{wn[-1]:.0f} cm⁻¹, {len(wn)} points")

    group_data = {}

    for g in groups:
        sub = df[df["group"] == g]
        mean = sub[fp_cols].mean().values
        std = sub[fp_cols].std().values
        count = len(sub)

        group_data[g] = (mean, std, count)

        path = os.path.join(OUT_DIR, f"SERS_spectrum_{g.replace('.', '_')}.png")
        plot_single_group(wn, mean, std, g, count, path)
        print(f"  {g}: {count} spectra → {path}")

    # Overlay plots
    print("\nGenerating overlay plots...")
    plot_overlay(wn, group_data, groups,
                 "SERS Mean Spectra — All Groups",
                 "SERS_spectrum_all_overlay.png")

    plot_overlay(wn, group_data, CANCER_TYPES,
                 "SERS Mean Spectra — Cancer Groups Only",
                 "SERS_spectrum_cancer_overlay.png")

    cancer_all = [g for g in groups if g in CANCER_TYPES or g in CANCER_ALIASES]
    plot_overlay(wn, group_data, cancer_all,
                 "SERS Mean Spectra — All Cancer Groups (incl. aliases)",
                 "SERS_spectrum_cancer_all_overlay.png")

    non_cancer = [g for g in groups if g in NON_CANCER]
    plot_overlay(wn, group_data, non_cancer,
                 "SERS Mean Spectra — Non-Cancer Controls",
                 "SERS_spectrum_noncancer_overlay.png")

    print(f"\nAll images saved to: {OUT_DIR}")
    print(f"Total: {len(groups)} individual + 3 overlay = {len(groups) + 3} PNG files")


if __name__ == "__main__":
    main()
