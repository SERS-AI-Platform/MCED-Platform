"""
Step 4: Normalization Ablation — Does removing SNV narrow the LR-ResNet gap?

Hypothesis:
    SNV preprocessing makes spectra linearly separable, giving LR an inherent
    advantage. If we remove SNV, the gap between LR and ResNet18 should narrow
    because DL can learn its own normalization while LR cannot.

Experiment:
    1. Re-preprocess spectra with different normalizations (none, minmax, l2)
    2. Train LR + ResNet18 on each variant (same folds, same settings)
    3. Compare the LR-ResNet gap across normalization methods

Usage:
    python models/run_step4_normalization.py
    python models/run_step4_normalization.py --norms none minmax
    python models/run_step4_normalization.py --skip-preprocess  # if already preprocessed
"""

from __future__ import annotations

import sys
import subprocess
import argparse
import logging
import json
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# ── Configuration ──
NORM_CONFIGS = {
    "none":   {"preprocess_dir": "results/norm_none",   "experiment": "step4_none"},
    "minmax": {"preprocess_dir": "results/norm_minmax",  "experiment": "step4_minmax"},
    "l2":     {"preprocess_dir": "results/norm_l2",      "experiment": "step4_l2"},
}

# SNV baseline from Step 1 (no re-run needed)
SNV_BASELINE = {
    "logistic_regression": {"s1_auc": 0.981, "s2_acc": 0.902, "s2_f1_macro": 0.872, "s2_auc": 0.985},
    "resnet18":            {"s1_auc": 0.958, "s2_acc": 0.771, "s2_f1_macro": 0.710, "s2_auc": 0.930},
}

CANCER_TYPES = ["PRO", "LUN", "CRC", "CPAN", "OVA"]
NON_CANCER = ["NOR", "DIA", "HBP", "H.D."]


def run_cmd(cmd, description=""):
    """Run a shell command and stream output."""
    logger.info(f"  CMD: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        logger.error(f"  FAILED (exit {result.returncode}): {description}")
        return False
    return True


def preprocess(norm, config):
    """Re-run preprocessing with a specific normalization."""
    out_dir = config["preprocess_dir"]
    logger.info(f"\n{'='*60}")
    logger.info(f"  Preprocessing: normalization={norm}")
    logger.info(f"  Output: {out_dir}")
    logger.info(f"{'='*60}")

    cmd = [
        sys.executable, "run_qc_preprocess.py",
        "--normalization", norm,
        "--output-dir", out_dir,
    ]
    return run_cmd(cmd, f"preprocess {norm}")


def train_models(norm, config, epochs=150):
    """Train LR + ResNet18 on preprocessed data."""
    csv_path = f"{config['preprocess_dir']}/processed_spectra.csv"
    experiment = config["experiment"]

    logger.info(f"\n{'='*60}")
    logger.info(f"  Training: normalization={norm}")
    logger.info(f"  Input: {csv_path}")
    logger.info(f"  Experiment: {experiment}")
    logger.info(f"{'='*60}")

    cmd = [
        sys.executable, "models/train.py",
        "--input", csv_path,
        "--benchmark-models", "logistic_regression", "resnet18",
        "--aggregate", "none",
        "--experiment", experiment,
        "--epochs", str(epochs),
        "--cancer-types", *CANCER_TYPES,
        "--non-cancer-groups", *NON_CANCER,
    ]
    return run_cmd(cmd, f"train {norm}")


def collect_results(norm, config):
    """Parse training results for a normalization variant."""
    experiment = config["experiment"]

    results = {}
    for model in ["logistic_regression", "resnet18"]:
        # Find the latest version directory
        model_dir = PROJECT_ROOT / "results" / "training" / experiment / model
        if not model_dir.exists():
            logger.warning(f"  No results for {model} at {model_dir}")
            continue

        versions = sorted(model_dir.glob("v*"))
        if not versions:
            continue

        summary_path = versions[-1] / "training_summary.json"
        if not summary_path.exists():
            continue

        with open(summary_path) as f:
            summary = json.load(f)

        metrics = summary.get("metrics", summary.get("overall_metrics", {}))
        results[model] = {
            "s1_auc": metrics.get("val_s1_auc", 0),
            "s2_acc": metrics.get("val_s2_accuracy", 0),
            "s2_f1_macro": metrics.get("val_s2_f1_macro", 0),
            "s2_auc": metrics.get("val_s2_auc", 0),
        }

    return results


def build_comparison_table(all_results):
    """Build comparison DataFrame with gap calculations."""
    rows = []
    for norm, model_results in all_results.items():
        lr = model_results.get("logistic_regression", {})
        rn = model_results.get("resnet18", {})

        rows.append({
            "normalization": norm,
            "lr_s1_auc": lr.get("s1_auc", 0),
            "lr_s2_f1": lr.get("s2_f1_macro", 0),
            "lr_s2_auc": lr.get("s2_auc", 0),
            "rn_s1_auc": rn.get("s1_auc", 0),
            "rn_s2_f1": rn.get("s2_f1_macro", 0),
            "rn_s2_auc": rn.get("s2_auc", 0),
            "gap_s1_auc": lr.get("s1_auc", 0) - rn.get("s1_auc", 0),
            "gap_s2_f1": lr.get("s2_f1_macro", 0) - rn.get("s2_f1_macro", 0),
        })

    return pd.DataFrame(rows)


def plot_comparison(df, out_dir):
    """Generate comparison plots."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    norms = df["normalization"].tolist()
    x = np.arange(len(norms))
    w = 0.35

    # S1 AUC
    ax = axes[0]
    ax.bar(x - w/2, df["lr_s1_auc"], w, label="LR", color="#1976D2", alpha=0.8)
    ax.bar(x + w/2, df["rn_s1_auc"], w, label="ResNet18", color="#E53935", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(norms)
    ax.set_ylabel("Stage 1 AUC")
    ax.set_title("Stage 1: Cancer vs Non-cancer")
    ax.legend()
    ax.set_ylim(0.85, 1.0)
    ax.grid(True, alpha=0.3)

    # S2 F1
    ax = axes[1]
    ax.bar(x - w/2, df["lr_s2_f1"], w, label="LR", color="#1976D2", alpha=0.8)
    ax.bar(x + w/2, df["rn_s2_f1"], w, label="ResNet18", color="#E53935", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(norms)
    ax.set_ylabel("Stage 2 F1 Macro")
    ax.set_title("Stage 2: Cancer Type Classification")
    ax.legend()
    ax.set_ylim(0.4, 1.0)
    ax.grid(True, alpha=0.3)

    # Gap
    ax = axes[2]
    ax.bar(x, df["gap_s2_f1"], 0.5, color="#FF9800", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(norms)
    ax.set_ylabel("LR - ResNet18 (S2 F1)")
    ax.set_title("Gap: Does removing SNV narrow it?")
    ax.axhline(0, color="gray", linestyle=":")
    ax.grid(True, alpha=0.3)

    # Annotate gap values
    for i, gap in enumerate(df["gap_s2_f1"]):
        ax.text(i, gap + 0.005, f"{gap:.3f}", ha="center", fontsize=9, fontweight="bold")

    plt.suptitle("Step 4: Normalization Ablation — LR vs ResNet18 Gap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "normalization_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Plot saved: {out_dir / 'normalization_comparison.png'}")


def main():
    p = argparse.ArgumentParser(description="Step 4: Normalization Ablation")
    p.add_argument("--norms", nargs="+", default=["none", "minmax", "l2"],
                   choices=["none", "minmax", "l2"],
                   help="Normalization methods to test")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--skip-preprocess", action="store_true",
                   help="Skip preprocessing (use existing CSVs)")
    p.add_argument("--skip-train", action="store_true",
                   help="Skip training (only analyze existing results)")
    args = p.parse_args()

    out_dir = PROJECT_ROOT / "results" / "training" / "step4_normalization_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(out_dir / "step4.log", mode="w", encoding="utf-8")],
    )

    t0 = datetime.now()

    logger.info("=" * 64)
    logger.info("  Step 4: Normalization Ablation Experiment")
    logger.info("=" * 64)
    logger.info(f"  Normalizations to test: {args.norms}")
    logger.info(f"  SNV baseline (from Step 1): included as reference")
    logger.info(f"  Cancer types: {CANCER_TYPES}")
    logger.info(f"  Non-cancer: {NON_CANCER}")

    # Stage A: Preprocessing
    if not args.skip_preprocess:
        logger.info("\n" + "=" * 64)
        logger.info("  Stage A: Preprocessing")
        logger.info("=" * 64)
        for norm in args.norms:
            config = NORM_CONFIGS[norm]
            ok = preprocess(norm, config)
            if not ok:
                logger.error(f"  Preprocessing failed for {norm}, skipping")
                args.norms.remove(norm)

    # Stage B: Training
    if not args.skip_train:
        logger.info("\n" + "=" * 64)
        logger.info("  Stage B: Training LR + ResNet18")
        logger.info("=" * 64)
        for norm in args.norms:
            config = NORM_CONFIGS[norm]
            ok = train_models(norm, config, epochs=args.epochs)
            if not ok:
                logger.error(f"  Training failed for {norm}")

    # Stage C: Collect and Compare Results
    logger.info("\n" + "=" * 64)
    logger.info("  Stage C: Results Comparison")
    logger.info("=" * 64)

    all_results = {"snv": SNV_BASELINE}

    for norm in args.norms:
        config = NORM_CONFIGS[norm]
        results = collect_results(norm, config)
        if results:
            all_results[norm] = results
            logger.info(f"  {norm}: collected {len(results)} models")
        else:
            logger.warning(f"  {norm}: no results found")

    # Build comparison
    df = build_comparison_table(all_results)
    df.to_csv(out_dir / "comparison_table.csv", index=False)

    # Print table
    logger.info(f"\n{'='*80}")
    logger.info("  NORMALIZATION ABLATION RESULTS")
    logger.info(f"{'='*80}")
    logger.info(f"{'Norm':<10} {'LR S1':>8} {'RN S1':>8} {'Gap':>8} | {'LR S2F1':>8} {'RN S2F1':>8} {'Gap':>8}")
    logger.info("-" * 70)
    for _, row in df.iterrows():
        logger.info(
            f"{row['normalization']:<10} "
            f"{row['lr_s1_auc']:>8.4f} {row['rn_s1_auc']:>8.4f} {row['gap_s1_auc']:>8.4f} | "
            f"{row['lr_s2_f1']:>8.4f} {row['rn_s2_f1']:>8.4f} {row['gap_s2_f1']:>8.4f}"
        )

    # Interpretation
    snv_gap = df[df["normalization"] == "snv"]["gap_s2_f1"].values[0] if "snv" in df["normalization"].values else 0
    logger.info(f"\n  SNV baseline gap (S2 F1): {snv_gap:.4f}")

    for norm in args.norms:
        row = df[df["normalization"] == norm]
        if len(row) > 0:
            gap = row["gap_s2_f1"].values[0]
            delta = gap - snv_gap
            direction = "NARROWER" if delta < -0.02 else "WIDER" if delta > 0.02 else "SIMILAR"
            logger.info(f"  {norm} gap: {gap:.4f} (delta vs SNV: {delta:+.4f}) -> {direction}")

    # Plot
    plot_comparison(df, out_dir)

    # Save summary
    summary = {
        "experiment": "step4_normalization_ablation",
        "timestamp": datetime.now().isoformat(),
        "normalizations_tested": ["snv"] + args.norms,
        "cancer_types": CANCER_TYPES,
        "non_cancer_groups": NON_CANCER,
        "comparison_table": df.to_dict(orient="records"),
        "snv_gap_s2_f1": float(snv_gap),
        "elapsed": str(datetime.now() - t0),
    }
    with open(out_dir / "step4_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"\n  Elapsed: {datetime.now() - t0}")
    logger.info(f"  Output: {out_dir}/")
    logger.info("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
