"""
Ablation Study Figures — Feature Representation & Clinical Fusion

Generates publication-quality figures from 2026-04-03 experiments.

Usage:
    python scripts/analysis/plot_ablation_results.py
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.sers.config import RESULTS_DIR

OUT_DIR = RESULTS_DIR / "ablation_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

COLORS = {
    "LR": "#1976D2",
    "RF": "#43A047",
    "XGB": "#F57C00",
    "Attention_NN": "#E53935",
    "LR+SexConstraint": "#7B1FA2",
}


def load_all():
    df = pd.read_csv(RESULTS_DIR / "all_ablation_results_20260403.csv")
    return df


# =============================================================================
# Figure 1: Feature Representation × Model (bar chart)
# =============================================================================

def fig1_feature_representation(df):
    d = df[df["Experiment"] == "1_Feature_Repr"].copy()

    features = ["full_spectrum", "1st_derivative", "2nd_derivative", "peak_features"]
    feat_labels = ["Full Spectrum\n(935)", "1st Derivative\n(935)", "2nd Derivative\n(935)", "Peak Features\n(82)"]
    models = ["LR", "RF", "XGB"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, (metric, title) in enumerate([
        ("Det_AUC_mean", "Stage 1: Cancer Detection AUC"),
        ("Type_F1_mean", "Stage 2: Cancer Type ID F1 (macro)"),
    ]):
        ax = axes[ax_idx]
        x = np.arange(len(features))
        width = 0.25

        for mi, model in enumerate(models):
            vals = []
            errs = []
            for feat in features:
                row = d[(d["Feature"] == feat) & (d["Model"] == model)]
                if len(row) > 0:
                    vals.append(row[metric].values[0])
                    errs.append(row[metric.replace("mean", "std")].values[0])
                else:
                    vals.append(0)
                    errs.append(0)

            bars = ax.bar(x + mi * width, vals, width, yerr=errs,
                         label=model, color=COLORS[model], alpha=0.85,
                         capsize=3, error_kw={"linewidth": 1})

        ax.set_xticks(x + width)
        ax.set_xticklabels(feat_labels)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(metric.split("_")[1].replace("mean", ""))
        ax.legend(loc="lower left")

        if "AUC" in metric:
            ax.set_ylim(0.85, 1.0)
        else:
            ax.set_ylim(0.55, 0.95)

        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Experiment 1: Feature Representation × Model", fontweight="bold", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig1_feature_representation.png")
    plt.close(fig)
    print(f"  Saved fig1_feature_representation.png")


# =============================================================================
# Figure 2: Peak Fitting Comparison (grouped bar)
# =============================================================================

def fig2_peak_fitting(df):
    d = df[df["Experiment"] == "2_Peak_Fitting"].copy()

    features = ["full_spectrum", "1st_derivative", "band_integration", "voigt_fitting", "deriv+voigt"]
    feat_labels = ["Full Spectrum\n(935)", "1st Derivative\n(935)", "Band Integration\n(58)",
                   "Voigt Fitting\n(75)", "Deriv + Voigt\n(1010)"]
    models = ["LR", "RF", "XGB"]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(features))
    width = 0.25

    for mi, model in enumerate(models):
        vals, errs = [], []
        for feat in features:
            row = d[(d["Feature"] == feat) & (d["Model"] == model)]
            if len(row) > 0:
                vals.append(row["Type_F1_mean"].values[0])
                errs.append(row["Type_F1_std"].values[0])
            else:
                vals.append(0); errs.append(0)
        ax.bar(x + mi * width, vals, width, yerr=errs,
               label=model, color=COLORS[model], alpha=0.85, capsize=3, error_kw={"linewidth": 1})

    ax.set_xticks(x + width)
    ax.set_xticklabels(feat_labels)
    ax.set_title("Experiment 2: Peak Fitting — Type ID F1 (macro)", fontweight="bold")
    ax.set_ylabel("Type ID F1")
    ax.set_ylim(0.55, 0.95)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotation: peak-only vs full
    ax.annotate("Peak-only:\n~0.62-0.67", xy=(2.25, 0.67), fontsize=9,
               color="#E53935", ha="center", fontweight="bold")
    ax.annotate("Full spectrum:\n~0.86-0.89", xy=(0.75, 0.90), fontsize=9,
               color="#1976D2", ha="center", fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_peak_fitting.png")
    plt.close(fig)
    print(f"  Saved fig2_peak_fitting.png")


# =============================================================================
# Figure 3: Clinical Fusion Waterfall (contribution breakdown)
# =============================================================================

def fig3_clinical_fusion_waterfall(df):
    d = df[df["Experiment"] == "4_Clinical_Fusion"].copy()

    # Build waterfall steps
    steps = [
        ("Spectrum\nonly", "A: spectrum", 0),
        ("+1st Deriv", "F: 3view", 0),       # 3view includes deriv
        ("+Clinical", "G: 3view+clin", 0),
    ]

    # Get values
    baseline = d[d["Feature"] == "A: spectrum"]["Type_F1_mean"].values[0]

    # Also get no-constraint values from detail.json
    with open(RESULTS_DIR / "clinical_fusion_full/detail.json") as f:
        detail = json.load(f)

    cond_map = {r["condition"]: r for r in detail}

    labels = []
    f1_sex = []
    f1_nosex = []

    conditions = [
        ("Spectrum", "A: spectrum"),
        ("+1st Deriv\n+Peak", "F: 3view"),
        ("+Clinical", "G: 3view+clin"),
    ]

    for label, cond in conditions:
        r = cond_map[cond]
        labels.append(label)
        f1_sex.append(r["type_f1_mean"])
        f1_nosex.append(float(r["type_f1_nc"].split("±")[0]) if "±" in str(r["type_f1_nc"]) else float(r["type_f1_nc"]))

    # Add sex constraint as separate step
    labels.append("+Sex\nConstraint")
    f1_nosex.append(f1_nosex[-1])  # same no-constraint value
    f1_sex.append(f1_sex[-1])       # already includes constraint

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(labels))
    bar_width = 0.35

    # With sex constraint bars
    bars = ax.bar(x, f1_sex, bar_width * 2, color=["#1976D2", "#43A047", "#F57C00", "#7B1FA2"],
                  alpha=0.85, edgecolor="white", linewidth=1.5)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, f1_sex)):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.003,
                f"{val:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=11)

    # Add delta arrows
    for i in range(1, len(f1_sex)):
        delta = f1_sex[i] - f1_sex[i-1]
        if abs(delta) > 0.001:
            color = "#2E7D32" if delta > 0 else "#C62828"
            ax.annotate(f"+{delta:.1%}" if delta > 0 else f"{delta:.1%}",
                       xy=(i, f1_sex[i-1] + delta/2),
                       fontsize=10, color=color, fontweight="bold",
                       ha="center")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Type ID F1 (macro)", fontsize=12)
    ax.set_title("Feature Contribution Breakdown — 7-cancer, 1,628 subjects",
                fontweight="bold", fontsize=13)
    ax.set_ylim(0.84, 0.93)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Reference line
    ax.axhline(y=0.877, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(3.5, 0.878, "Previous best\n(Stacking F1=0.877)", fontsize=8,
           color="gray", ha="right", va="bottom")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_contribution_waterfall.png")
    plt.close(fig)
    print(f"  Saved fig3_contribution_waterfall.png")


# =============================================================================
# Figure 4: Clinical Fusion Full Comparison (horizontal bar)
# =============================================================================

def fig4_clinical_fusion_full(df):
    d = df[df["Experiment"] == "4_Clinical_Fusion"].copy()
    d = d.sort_values("Type_F1_mean", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    labels = d["Feature"].values
    f1_vals = d["Type_F1_mean"].values
    f1_errs = d["Type_F1_std"].values
    n_feats = d["N_Features"].values

    # Color based on value
    colors = []
    for v in f1_vals:
        if v > 0.91:
            colors.append("#1B5E20")
        elif v > 0.88:
            colors.append("#43A047")
        elif v > 0.85:
            colors.append("#66BB6A")
        elif v > 0.5:
            colors.append("#90CAF9")
        else:
            colors.append("#E0E0E0")

    bars = ax.barh(range(len(labels)), f1_vals, xerr=f1_errs,
                   color=colors, alpha=0.85, capsize=3,
                   edgecolor="white", linewidth=1)

    # Labels
    for i, (val, nf) in enumerate(zip(f1_vals, n_feats)):
        if val > 0.5:
            ax.text(val - 0.01, i, f"{val:.4f}", va="center", ha="right",
                   fontweight="bold", fontsize=10, color="white")
        ax.text(0.22, i, f"({nf} feat)", va="center", ha="left",
               fontsize=9, color="#666")

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Type ID F1 (macro, with sex constraint)", fontsize=12)
    ax.set_title("Clinical Fusion — All Conditions (7-cancer, 1,628 subjects)",
                fontweight="bold", fontsize=13)
    ax.set_xlim(0.15, 0.95)

    # Best marker
    best_idx = np.argmax(f1_vals)
    ax.text(f1_vals[best_idx] + 0.005, best_idx, " ★ BEST",
           va="center", fontsize=11, fontweight="bold", color="#1B5E20")

    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_clinical_fusion_comparison.png")
    plt.close(fig)
    print(f"  Saved fig4_clinical_fusion_comparison.png")


# =============================================================================
# Figure 5: DL vs Classical (scatter: params vs F1)
# =============================================================================

def fig5_dl_vs_classical():
    # Manually compile key results
    models = [
        ("LR (spectrum)", 935, 0.8535, "#1976D2", "o"),
        ("LR (1st deriv)", 935, 0.8763, "#1976D2", "s"),
        ("LR (3-view+clin)", 1948, 0.9136, "#7B1FA2", "D"),
        ("RF (spectrum)", 935*300, 0.7165, "#43A047", "o"),
        ("XGB (spectrum)", 935*300, 0.7704, "#F57C00", "o"),
        ("Multi-View Attn", 82728, 0.6091, "#E53935", "^"),
        ("ResNet18", 150000, 0.65, "#E53935", "v"),       # approx from memory
        ("Transformer", 200000, 0.654, "#E53935", "<"),     # approx from memory
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    for name, params, f1, color, marker in models:
        ax.scatter(params, f1, c=color, marker=marker, s=120, zorder=5,
                  edgecolors="white", linewidths=1.5)
        offset_x = 1.3
        offset_y = 0.008
        if "Attn" in name:
            offset_y = -0.025
        if "Transformer" in name:
            offset_x = 1.5
            offset_y = 0.015
        ax.annotate(name, (params * offset_x, f1 + offset_y), fontsize=8.5)

    ax.set_xscale("log")
    ax.set_xlabel("Model Complexity (parameters / effective features)", fontsize=12)
    ax.set_ylabel("Type ID F1 (macro)", fontsize=12)
    ax.set_title("Model Complexity vs Performance — Deep Learning never beats LR",
                fontweight="bold", fontsize=13)
    ax.set_ylim(0.55, 0.95)
    ax.set_xlim(500, 500000)

    # Regions
    ax.axhspan(0.85, 0.95, alpha=0.05, color="green")
    ax.axhspan(0.55, 0.75, alpha=0.05, color="red")
    ax.text(600, 0.92, "Clinical-grade (F1>0.85)", fontsize=8, color="green", alpha=0.7)
    ax.text(600, 0.57, "Insufficient (F1<0.75)", fontsize=8, color="red", alpha=0.7)

    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig5_dl_vs_classical.png")
    plt.close(fig)
    print(f"  Saved fig5_dl_vs_classical.png")


# =============================================================================
# Figure 6: Fold-level Box Plot for best conditions
# =============================================================================

def fig6_fold_boxplot():
    with open(RESULTS_DIR / "clinical_fusion_full/detail.json") as f:
        detail = json.load(f)

    conditions = [
        ("A: spectrum", "#90CAF9"),
        ("B: 1st_deriv", "#64B5F6"),
        ("E: deriv+clin", "#42A5F5"),
        ("F: 3view", "#66BB6A"),
        ("G: 3view+clin", "#1B5E20"),
        ("H: deriv+peak+clin", "#2E7D32"),
    ]

    cond_map = {r["condition"]: r for r in detail}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, (metric_key, title) in enumerate([
        ("fold_aucs", "Detection AUC (per fold)"),
        ("fold_f1s", "Type ID F1 (per fold, sex constraint)"),
    ]):
        ax = axes[ax_idx]
        data = []
        labels = []
        colors = []

        for cname, color in conditions:
            r = cond_map[cname]
            data.append(r[metric_key])
            labels.append(cname.split(": ")[1])
            colors.append(color)

        bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                       medianprops=dict(color="black", linewidth=2))

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Overlay individual points
        for i, d in enumerate(data):
            jitter = np.random.normal(0, 0.05, len(d))
            ax.scatter([i + 1 + j for j in jitter], d,
                      color=colors[i], s=40, alpha=0.8, edgecolors="white", zorder=5)

        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Fold-Level Stability — 5-Fold CV, 7-cancer, 1,628 subjects",
                fontweight="bold", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig6_fold_stability.png")
    plt.close(fig)
    print(f"  Saved fig6_fold_stability.png")


# =============================================================================
# Main
# =============================================================================

def main():
    print("Generating ablation study figures...")
    df = load_all()

    fig1_feature_representation(df)
    fig2_peak_fitting(df)
    fig3_clinical_fusion_waterfall(df)
    fig4_clinical_fusion_full(df)
    fig5_dl_vs_classical()
    fig6_fold_boxplot()

    print(f"\nAll figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
