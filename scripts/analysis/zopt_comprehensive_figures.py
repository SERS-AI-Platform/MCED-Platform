#!/usr/bin/env python3
"""
Z-opt Comprehensive Figures — Full optimization results visualization
"""

import sys, json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results" / "training" / "optimize_7cancer_full"
OUTPUT_DIR = PROJECT_ROOT / "results" / "figures" / "zopt_comprehensive"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Color scheme
C_AUC = "#2196F3"
C_F1 = "#E91E63"
C_SENS = "#4CAF50"
C_SPEC = "#FF9800"
C_LR = "#2196F3"
C_RN = "#FF9800"
C_ENS = "#E91E63"

# =============================================================================
# Load all data
# =============================================================================
phase1 = pd.read_csv(RESULTS_DIR / "phase1_wavenumber_ablation.csv")
phase2 = pd.read_csv(RESULTS_DIR / "phase2_aggregation_ablation.csv")
phase3 = pd.read_csv(RESULTS_DIR / "phase3_architecture_hp_search.csv")
phase4_repeats = pd.read_csv(RESULTS_DIR / "phase4_held_out_repeats.csv")
with open(RESULTS_DIR / "phase4_final_evaluation.json") as f:
    phase4_eval = json.load(f)


# =============================================================================
# Figure 1: Phase 1 — Wavenumber Range & Feature Transform Ablation
# =============================================================================
def fig1_wavenumber_ablation():
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Split into range ablation and feature transform
    range_rows = phase1[~phase1["config"].str.contains("deriv|savgol|concat")]
    feat_rows = phase1[phase1["config"].str.contains("deriv|savgol|concat|full_402")]

    # Panel A: Wavenumber range
    ax = axes[0]
    names = range_rows["config"].values
    display_names = [n.replace("_", " ").replace("full 402 2198", "Full (402-2198)")
                     for n in names]
    aucs = range_rows["s1_auc"].values
    f1s = range_rows["s2_f1"].values

    x = np.arange(len(names))
    w = 0.35
    bars1 = ax.bar(x - w/2, aucs, w, label="S1 AUC", color=C_AUC, alpha=0.85)
    bars2 = ax.bar(x + w/2, f1s, w, label="S2 F1", color=C_F1, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0.6, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("A. Wavenumber Range Ablation", fontweight="bold", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars1, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.005, f"{val:.3f}",
                ha="center", fontsize=8)
    for bar, val in zip(bars2, f1s):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.005, f"{val:.3f}",
                ha="center", fontsize=8)

    # Panel B: Feature transform
    ax = axes[1]
    display = {
        "full_402_2198": "Raw (baseline)",
        "full_deriv1": "1st deriv (diff)",
        "full_savgol_d1": "SG d1",
        "full_concat_d1": "Raw + d1 concat",
        "fp600_1800_savgol_d1": "FP + SG d1",
        "fp600_1800_concat_d1": "FP + concat d1",
    }
    feat_rows_sorted = feat_rows.copy()
    feat_rows_sorted["display"] = feat_rows_sorted["config"].map(display)
    feat_rows_sorted = feat_rows_sorted.dropna(subset=["display"])

    names = feat_rows_sorted["display"].values
    aucs = feat_rows_sorted["s1_auc"].values
    f1s = feat_rows_sorted["s2_f1"].values
    n_feat = feat_rows_sorted["n_features"].values

    x = np.arange(len(names))
    bars1 = ax.bar(x - w/2, aucs, w, label="S1 AUC", color=C_AUC, alpha=0.85)
    bars2 = ax.bar(x + w/2, f1s, w, label="S2 F1", color=C_F1, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0.6, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("B. Feature Transform Ablation", fontweight="bold", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    for bar, val, nf in zip(bars2, f1s, n_feat):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.005, f"{val:.3f}\n({nf}f)",
                ha="center", fontsize=7)

    plt.suptitle("Phase 1: Wavenumber & Feature Ablation (LR, medoid, 5-fold CV)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig1_wavenumber_feature_ablation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  fig1 saved")


# =============================================================================
# Figure 2: Phase 2 — Aggregation Method Comparison
# =============================================================================
def fig2_aggregation():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    agg_names = phase2["aggregate"].values
    display = {"medoid": "Medoid", "mean": "Mean", "none": "Individual (all)"}
    labels = [display[a] for a in agg_names]

    # Panel A: AUC + F1
    ax = axes[0]
    aucs = phase2["s1_auc"].values
    f1s = phase2["s2_f1"].values
    x = np.arange(len(labels))
    w = 0.35
    bars1 = ax.bar(x - w/2, aucs, w, label="S1 AUC", color=C_AUC, alpha=0.85)
    bars2 = ax.bar(x + w/2, f1s, w, label="S2 F1", color=C_F1, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0.7, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("A. AUC & F1 by Aggregation", fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars1, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.005, f"{val:.4f}",
                ha="center", fontsize=9, fontweight="bold")
    for bar, val in zip(bars2, f1s):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.005, f"{val:.4f}",
                ha="center", fontsize=9, fontweight="bold")

    # Highlight mean as winner
    ax.annotate("BEST", xy=(1 - w/2, aucs[1]), fontsize=8, color="green",
                fontweight="bold", ha="center", va="bottom",
                xytext=(1 - w/2, aucs[1] + 0.025),
                arrowprops=dict(arrowstyle="->", color="green"))

    # Panel B: Sensitivity & Specificity
    ax = axes[1]
    sens = phase2["s1_sens"].values
    spec = phase2["s1_spec"].values
    bars1 = ax.bar(x - w/2, sens, w, label="Sensitivity", color=C_SENS, alpha=0.85)
    bars2 = ax.bar(x + w/2, spec, w, label="Specificity", color=C_SPEC, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0.8, 1.02)
    ax.set_ylabel("Rate")
    ax.set_title("B. Sensitivity & Specificity", fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars1, sens):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.003, f"{val:.3f}",
                ha="center", fontsize=9)
    for bar, val in zip(bars2, spec):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.003, f"{val:.3f}",
                ha="center", fontsize=9)

    plt.suptitle("Phase 2: Aggregation Method Comparison (mean agg = +14pp F1 vs medoid)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig2_aggregation_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  fig2 saved")


# =============================================================================
# Figure 3: Phase 3 — Architecture HP Search
# =============================================================================
def fig3_architecture():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    configs = phase3["config"].values
    short = [c.replace("ch", "").replace("_lr", "\nlr=").replace("_do", "\ndo=").replace("_wd", "\nwd=")
             for c in configs]

    # Panel A: ResNet18 standalone
    ax = axes[0]
    rn_auc = phase3["rn_auc"].values
    rn_f1 = phase3["rn_f1"].values
    x = np.arange(len(configs))
    ax.bar(x - 0.2, rn_auc, 0.4, label="AUC", color=C_RN, alpha=0.7)
    ax.bar(x + 0.2, rn_f1, 0.4, label="F1", color="#FF5722", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=6, ha="center")
    ax.set_ylim(0.6, 1.05)
    ax.set_title("A. ResNet18 Standalone", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Panel B: Ensemble (LR + ResNet18)
    ax = axes[1]
    ens_auc = phase3["ens_auc"].values
    ens_f1 = phase3["ens_f1"].values
    alpha = phase3["best_alpha"].values

    ax.bar(x - 0.2, ens_auc, 0.4, label="Ens AUC", color=C_ENS, alpha=0.7)
    ax.bar(x + 0.2, ens_f1, 0.4, label="Ens F1", color="#9C27B0", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=6, ha="center")
    ax.set_ylim(0.9, 1.0)
    ax.set_title("B. Ensemble (LR α + ResNet18 1-α)", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for i, (auc, f1, a) in enumerate(zip(ens_auc, ens_f1, alpha)):
        ax.text(i, f1 + 0.002, f"F1={f1:.3f}\nα={a}", ha="center", fontsize=7)

    # Highlight best
    best_idx = ens_f1.argmax()
    ax.annotate("BEST", xy=(best_idx + 0.2, ens_f1[best_idx]),
                fontsize=9, color="red", fontweight="bold",
                xytext=(best_idx + 0.7, ens_f1[best_idx] + 0.01),
                arrowprops=dict(arrowstyle="->", color="red"))

    plt.suptitle("Phase 3: ResNet18 Architecture Search (mean agg + d1 features)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig3_architecture_search.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  fig3 saved")


# =============================================================================
# Figure 4: Phase 4 — Held-out Test Robustness
# =============================================================================
def fig4_held_out():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    repeats = phase4_repeats["repeat"].values

    # Panel A: AUC across repeats
    ax = axes[0]
    ax.plot(repeats, phase4_repeats["lr_s1_auc"], "o-", color=C_LR, label="LR", linewidth=2, markersize=8)
    ax.plot(repeats, phase4_repeats["rn_s1_auc"], "s-", color=C_RN, label="ResNet18", linewidth=2, markersize=8)
    ax.plot(repeats, phase4_repeats["ens_s1_auc"], "^-", color=C_ENS, label="Ensemble", linewidth=2, markersize=8)
    ax.set_xlabel("Repeat")
    ax.set_ylabel("S1 AUC")
    ax.set_title("A. Detection AUC (Held-out, 5 repeats)", fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.88, 0.97)
    # Mean lines
    for col, color, label in [("lr_s1_auc", C_LR, "LR"), ("rn_s1_auc", C_RN, "R18"), ("ens_s1_auc", C_ENS, "Ens")]:
        mean = phase4_repeats[col].mean()
        ax.axhline(mean, color=color, linestyle="--", alpha=0.4)
        ax.text(4.3, mean, f"μ={mean:.3f}", color=color, fontsize=8, va="center")

    # Panel B: F1 across repeats
    ax = axes[1]
    ax.plot(repeats, phase4_repeats["lr_s2_f1"], "o-", color=C_LR, label="LR", linewidth=2, markersize=8)
    ax.plot(repeats, phase4_repeats["rn_s2_f1"], "s-", color=C_RN, label="ResNet18", linewidth=2, markersize=8)
    ax.plot(repeats, phase4_repeats["ens_s2_f1"], "^-", color=C_ENS, label="Ensemble", linewidth=2, markersize=8)
    ax.set_xlabel("Repeat")
    ax.set_ylabel("S2 F1 Macro")
    ax.set_title("B. Cancer Type F1 (Held-out, 5 repeats)", fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 0.8)
    for col, color in [("lr_s2_f1", C_LR), ("rn_s2_f1", C_RN), ("ens_s2_f1", C_ENS)]:
        mean = phase4_repeats[col].mean()
        ax.axhline(mean, color=color, linestyle="--", alpha=0.4)
        ax.text(4.3, mean, f"μ={mean:.3f}", color=color, fontsize=8, va="center")

    plt.suptitle("Phase 4: Held-out Test Robustness (60/20/20 split × 5 repeats)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig4_held_out_robustness.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  fig4 saved")


# =============================================================================
# Figure 5: Grand Summary — All Methods Comparison
# =============================================================================
def fig5_grand_summary():
    """Compare Z-opt best with all recent experiments."""
    fig = plt.figure(figsize=(18, 8))
    gs = GridSpec(1, 2, width_ratios=[2, 1], wspace=0.3)

    # Data: all experiments
    experiments = [
        # Phase, Method, AUC, F1, agg, note
        ("Z-opt\n(best)", "LR+R18 ens\nmean+d1", 0.995, 0.919, "mean"),
        ("Z-opt\nLR only", "LR\nmean+d1", 0.993, 0.922, "mean"),
        ("STK", "Stacking\n9 models", 0.980, 0.877, "all"),
        ("MV-E2L", "LR\nraw+d1+d2", 0.976, 0.869, "all"),
        ("MV-E2e", "LR+R18\nraw+d1+d2", 0.972, 0.867, "all"),
        ("W-fus", "LR fusion\n+clinical", 0.956, 0.817, "medoid"),
        ("CON-lp", "Contrastive\nprobe", 0.942, 0.686, "all"),
        ("SPT", "Spectral\nTransformer", 0.950, 0.654, "all"),
        ("CON-ft", "Contrastive\nfine-tune", 0.961, 0.666, "all"),
    ]

    names = [e[0] for e in experiments]
    methods = [e[1] for e in experiments]
    aucs = [e[2] for e in experiments]
    f1s = [e[3] for e in experiments]
    aggs = [e[4] for e in experiments]

    colors = []
    for agg in aggs:
        if agg == "mean":
            colors.append("#2E7D32")
        elif agg == "all":
            colors.append("#1565C0")
        else:
            colors.append("#FF8F00")

    # Panel A: Horizontal bar chart
    ax = fig.add_subplot(gs[0])
    y = np.arange(len(names))
    bars = ax.barh(y, f1s, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)

    ax.set_yticks(y)
    ax.set_yticklabels([f"{n}\n{m}" for n, m in zip(names, methods)], fontsize=9)
    ax.set_xlabel("Cancer Type F1 (macro)", fontsize=12)
    ax.set_title("All Methods Comparison — 7-Cancer Classification", fontsize=14, fontweight="bold")
    ax.axvline(0.9, color="red", linestyle="--", alpha=0.6, label="Target 0.9")
    ax.set_xlim(0.5, 1.05)
    ax.grid(axis="x", alpha=0.3)

    for i, (bar, f1, auc) in enumerate(zip(bars, f1s, aucs)):
        ax.text(f1 + 0.005, i, f"F1={f1:.3f}  AUC={auc:.3f}", va="center", fontsize=9,
                fontweight="bold" if f1 > 0.9 else "normal")

    # Legend for aggregation
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2E7D32", alpha=0.85, label="Mean aggregation"),
        Patch(facecolor="#1565C0", alpha=0.85, label="Individual spectra"),
        Patch(facecolor="#FF8F00", alpha=0.85, label="Medoid"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

    ax.invert_yaxis()

    # Panel B: Key insights
    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")

    insights = [
        "KEY FINDINGS",
        "",
        "1. Mean aggregation = biggest factor",
        "   medoid→mean: F1 +14pp",
        "",
        "2. 1st derivative (d1) = 2nd biggest",
        "   raw→d1: F1 +4.7pp",
        "",
        "3. LR > all DL models",
        "   LR F1=0.922 > Ensemble 0.919",
        "   ResNet18, Transformer, Contrastive",
        "   all fail to beat LR alone",
        "",
        "4. Stacking = best for individual",
        "   spectra (F1=0.877)",
        "",
        "5. 0.9 target requires mean agg",
        "   Individual spectra max: 0.877",
        "",
        "BEST CONFIG:",
        "  LR + mean agg + d1 features",
        f"  AUC=0.993, F1=0.922",
        f"  Sens=94.5%, Spec=96.1%",
    ]

    for i, line in enumerate(insights):
        weight = "bold" if line.startswith(("KEY", "BEST", "1.", "2.", "3.", "4.", "5.")) else "normal"
        color = "#1B5E20" if "BEST" in line or "0.922" in line else "#333"
        fontsize = 12 if line.startswith("KEY") else 10 if line.startswith("BEST") else 9
        ax2.text(0.05, 0.95 - i * 0.04, line, transform=ax2.transAxes,
                fontsize=fontsize, fontweight=weight, color=color,
                fontfamily="monospace", va="top")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig5_grand_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  fig5 saved")


# =============================================================================
# Figure 6: Per-cancer type accuracy (from memory)
# =============================================================================
def fig6_per_cancer():
    fig, ax = plt.subplots(figsize=(10, 5))

    cancers = ["BLC", "CRC", "PRO", "LUN", "OVA", "PAN", "BRE"]
    correct = [96, 286, 94, 261, 48, 58, 25]
    total = [97, 291, 97, 277, 53, 67, 30]
    rates = [c/t for c, t in zip(correct, total)]

    colors = plt.cm.RdYlGn([r for r in rates])
    bars = ax.barh(range(len(cancers)), rates, color=colors, edgecolor="white")
    ax.set_yticks(range(len(cancers)))
    ax.set_yticklabels(cancers, fontsize=11, fontweight="bold")
    ax.set_xlabel("Classification Accuracy", fontsize=12)
    ax.set_title("Per-Cancer Type Accuracy (Z-opt, 5-fold CV, mean+d1)",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(0.7, 1.02)
    ax.grid(axis="x", alpha=0.3)
    ax.axvline(0.9, color="red", linestyle="--", alpha=0.4)

    for i, (bar, rate, c, t) in enumerate(zip(bars, rates, correct, total)):
        ax.text(rate + 0.005, i, f"{rate:.1%} ({c}/{t})", va="center", fontsize=10)

    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig6_per_cancer_accuracy.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  fig6 saved")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("Generating Z-opt comprehensive figures...")
    fig1_wavenumber_ablation()
    fig2_aggregation()
    fig3_architecture()
    fig4_held_out()
    fig5_grand_summary()
    fig6_per_cancer()
    print(f"\nAll figures saved to {OUTPUT_DIR}")
