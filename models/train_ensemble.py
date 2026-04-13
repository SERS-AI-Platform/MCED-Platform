"""
Step 5: Ensemble Experiment — LR + ResNet18

Combines Logistic Regression and ResNet18-1D predictions
to check if they capture complementary patterns.

Strategy:
    1. Run 5-fold CV for both LR and ResNet18 (same folds)
    2. For each val sample, combine predictions:
       - Stage 1: weighted average of cancer probabilities
       - Stage 2: weighted average of cancer-type logits
    3. Search optimal blend weight on val predictions

Usage:
    python models/train_ensemble.py
    python models/train_ensemble.py --epochs 150 --blend-weights 0.3 0.5 0.7
"""

from __future__ import annotations

import sys
import argparse
import logging
import json
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.model import (
    ModelConfig, TwoStageLoss, SERSDataset, SERSCancerDetector, model_summary,
)
from models.train import (
    load_processed_spectra, get_feature_columns,
    apply_class_selection, resolve_aliases, aggregate_replicates,
    create_labels, EarlyStopping, train_fold,
    _compute_overall_metrics, MODEL_DISPLAY_NAMES,
)
from src.sers.config import RESULTS_DIR

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def run_lr_fold(X_train, X_val, yb_train, yb_val, yc_train, yc_val, config, fold_i):
    """Run Logistic Regression for one fold, return predictions."""
    # Stage 1: binary
    binary_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=1000, solver="saga", class_weight="balanced",
                           random_state=config.random_state + fold_i)
    )
    binary_model.fit(X_train, yb_train)
    bp_val = binary_model.predict_proba(X_val)[:, 1]
    bp_train = binary_model.predict_proba(X_train)[:, 1]

    # Stage 2: cancer type
    cancer_mask = yc_train >= 0
    n_ct = config.n_cancer_types
    val_s2_prob = np.full((len(X_val), n_ct), 1.0 / n_ct)
    train_s2_prob = np.full((len(X_train), n_ct), 1.0 / n_ct)

    if cancer_mask.sum() > 0:
        present = np.unique(yc_train[cancer_mask])
        if len(present) > 1:
            local_labels = np.searchsorted(present, yc_train[cancer_mask])
            s2_model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1.0, max_iter=1000, solver="saga", class_weight="balanced",
                                   random_state=config.random_state + fold_i + 100)
            )
            s2_model.fit(X_train[cancer_mask], local_labels)
            local_val_prob = s2_model.predict_proba(X_val)
            local_train_prob = s2_model.predict_proba(X_train)
            # Expand local probabilities to full class space
            for j, cls in enumerate(present):
                val_s2_prob[:, cls] = local_val_prob[:, j]
                train_s2_prob[:, cls] = local_train_prob[:, j]

    return {
        "val_binary_prob": bp_val,
        "val_cancer_logits": val_s2_prob,  # Actually probabilities, but same interface
        "train_binary_prob": bp_train,
        "train_cancer_logits": train_s2_prob,
    }


def run_resnet_fold(X_train, X_val, bl_train, bl_val, ctl_train, ctl_val,
                    config, device, fold_i):
    """Run ResNet18 for one fold, return predictions."""
    train_ds = SERSDataset(X_train, bl_train, ctl_train, augment=True)
    val_ds = SERSDataset(X_val, bl_val, ctl_val, augment=False)

    runtime = {
        "num_workers": 4 if device.type == "cuda" else 0,
        "pin_memory": device.type == "cuda",
        "prefetch_factor": 2 if device.type == "cuda" else None,
        "use_amp": device.type == "cuda",
    }

    model = SERSCancerDetector(config).to(device)
    fold_result = train_fold(model, train_ds, val_ds, config, device, fold_i, runtime)

    train_m = fold_result["train_metrics"]
    val_m = fold_result["val_metrics"]

    return {
        "val_binary_prob": val_m["binary_prob"],
        "val_cancer_logits": val_m["cancer_logits"],
        "train_binary_prob": train_m["binary_prob"],
        "train_cancer_logits": train_m["cancer_logits"],
    }


def evaluate_blend(val_bp_lr, val_cl_lr, val_bp_rn, val_cl_rn, bl, ctl, alpha):
    """Evaluate blended predictions at weight alpha (alpha=LR weight)."""
    bp = alpha * val_bp_lr + (1 - alpha) * val_bp_rn

    # For cancer logits: LR produces probabilities, ResNet produces logits
    # Convert ResNet logits to probabilities first
    rn_probs = torch.softmax(torch.tensor(val_cl_rn), dim=-1).numpy()
    cl_blend = alpha * val_cl_lr + (1 - alpha) * rn_probs

    valid = ~np.isnan(bp)
    bp_v = bp[valid]
    bt_v = bl[valid]
    cl_v = cl_blend[valid]
    ct_v = ctl[valid]

    bpred = (bp_v > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(bt_v, bpred, labels=[0, 1]).ravel()

    result = {
        "alpha": alpha,
        "s1_auc": float(roc_auc_score(bt_v, bp_v)),
        "s1_accuracy": float(accuracy_score(bt_v, bpred)),
        "s1_sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0,
        "s1_specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0,
        "s1_f1": float(f1_score(bt_v, bpred)),
    }

    cancer_mask = ct_v >= 0
    if cancer_mask.sum() > 0:
        cpred = cl_v[cancer_mask].argmax(axis=1)
        result["s2_accuracy"] = float(accuracy_score(ct_v[cancer_mask], cpred))
        result["s2_f1_macro"] = float(f1_score(ct_v[cancer_mask], cpred, average="macro", zero_division=0))
        try:
            result["s2_auc"] = float(roc_auc_score(ct_v[cancer_mask], cl_v[cancer_mask], multi_class="ovr", average="macro"))
        except ValueError:
            result["s2_auc"] = float("nan")

    return result


def main():
    p = argparse.ArgumentParser(description="Ensemble: LR + ResNet18")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--device", default="auto")
    p.add_argument("--input", "-i", default=None, help="Path to processed_spectra.csv")
    p.add_argument("--aggregate", default="none")
    p.add_argument("--cancer-types", nargs="+", default=["PRO", "LUN", "CRC", "CPAN", "OVA"])
    p.add_argument("--non-cancer-groups", nargs="+", default=["NOR", "DIA", "HBP", "H.D."])
    p.add_argument("--blend-weights", nargs="+", type=float,
                    default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    args = p.parse_args()

    out_dir = Path("results/training/step5_ensemble")
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(out_dir / "train.log", mode="w", encoding="utf-8")],
    )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    logger.info("=" * 64)
    logger.info("  Step 5: Ensemble (LR + ResNet18-1D)")
    logger.info("=" * 64)
    logger.info(f"  Device: {device}")
    logger.info(f"  Blend weights to test: {args.blend_weights}")

    import yaml
    with open("config/config.yaml", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    df = load_processed_spectra(args.input)
    feat_cols = get_feature_columns(df)
    n_feat = len(feat_cols)

    mc = ModelConfig.from_pipeline_config(raw_cfg, n_spectral_features=n_feat)
    mc = apply_class_selection(mc, cancer_types=args.cancer_types, non_cancer_groups=args.non_cancer_groups)
    mc = ModelConfig(**{**mc.__dict__, "n_epochs": args.epochs})

    logger.info(f"  Cancer types ({mc.n_cancer_types}): {mc.cancer_types}")
    logger.info(f"  Non-cancer: {mc.non_cancer_groups}")

    df = resolve_aliases(df, mc)
    df_agg = aggregate_replicates(df, feat_cols, args.aggregate)
    X, bl, ctl, sample_ids, groups_arr = create_labels(df_agg, mc)

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=mc.random_state)
    n_samples = len(X)

    # Collect per-fold predictions for both models
    val_bp_lr = np.full(n_samples, np.nan)
    val_cl_lr = np.full((n_samples, mc.n_cancer_types), np.nan)
    val_bp_rn = np.full(n_samples, np.nan)
    val_cl_rn = np.full((n_samples, mc.n_cancer_types), np.nan)

    t0 = datetime.now()

    for fold_i, (train_idx, val_idx) in enumerate(cv.split(X, bl, sample_ids)):
        logger.info(f"\n{'='*50}")
        logger.info(f"  Fold {fold_i+1}/5 (train={len(train_idx)}, val={len(val_idx)})")
        logger.info(f"{'='*50}")

        X_tr, X_va = X[train_idx], X[val_idx]
        bl_tr, bl_va = bl[train_idx], bl[val_idx]
        ctl_tr, ctl_va = ctl[train_idx], ctl[val_idx]

        # LR
        logger.info(f"  [LR] Training...")
        lr_pred = run_lr_fold(X_tr, X_va, bl_tr, bl_va, ctl_tr, ctl_va, mc, fold_i)
        val_bp_lr[val_idx] = lr_pred["val_binary_prob"]
        val_cl_lr[val_idx] = lr_pred["val_cancer_logits"]
        lr_s1 = roc_auc_score(bl_va, lr_pred["val_binary_prob"])
        logger.info(f"  [LR] Val S1 AUC: {lr_s1:.3f}")

        # ResNet18
        logger.info(f"  [ResNet18] Training...")
        rn_pred = run_resnet_fold(X_tr, X_va, bl_tr, bl_va, ctl_tr, ctl_va, mc, device, fold_i)
        val_bp_rn[val_idx] = rn_pred["val_binary_prob"]
        val_cl_rn[val_idx] = rn_pred["val_cancer_logits"]
        rn_s1 = roc_auc_score(bl_va, rn_pred["val_binary_prob"])
        logger.info(f"  [ResNet18] Val S1 AUC: {rn_s1:.3f}")

    # Evaluate all blend weights
    logger.info(f"\n{'='*64}")
    logger.info("  Blend Weight Search")
    logger.info(f"{'='*64}")
    logger.info(f"  alpha = LR weight, (1-alpha) = ResNet18 weight")
    logger.info(f"  alpha=1.0 = pure LR, alpha=0.0 = pure ResNet18")

    blend_results = []
    for alpha in args.blend_weights:
        r = evaluate_blend(val_bp_lr, val_cl_lr, val_bp_rn, val_cl_rn, bl, ctl, alpha)
        blend_results.append(r)
        logger.info(
            f"  alpha={alpha:.1f} | S1_AUC={r['s1_auc']:.4f} "
            f"S2_Acc={r.get('s2_accuracy', 0):.4f} "
            f"S2_F1={r.get('s2_f1_macro', 0):.4f} "
            f"S2_AUC={r.get('s2_auc', 0):.4f}"
        )

    # Find best
    best = max(blend_results, key=lambda r: r.get("s2_f1_macro", 0))
    logger.info(f"\n  Best blend: alpha={best['alpha']:.1f}")
    logger.info(f"  S1 AUC: {best['s1_auc']:.4f}")
    logger.info(f"  S2 F1 macro: {best.get('s2_f1_macro', 0):.4f}")
    logger.info(f"  S2 AUC: {best.get('s2_auc', 0):.4f}")

    elapsed = datetime.now() - t0

    # Save results
    blend_df = pd.DataFrame(blend_results)
    blend_df.to_csv(out_dir / "blend_results.csv", index=False)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    alphas = [r["alpha"] for r in blend_results]

    axes[0].plot(alphas, [r["s1_auc"] for r in blend_results], "o-", color="#1976D2")
    axes[0].set_xlabel("alpha (LR weight)")
    axes[0].set_ylabel("Stage 1 AUC")
    axes[0].set_title("Stage 1: Cancer vs Non-cancer")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(alphas, [r.get("s2_f1_macro", 0) for r in blend_results], "o-", color="#E53935")
    axes[1].set_xlabel("alpha (LR weight)")
    axes[1].set_ylabel("Stage 2 F1 Macro")
    axes[1].set_title("Stage 2: Cancer Type")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(alphas, [r.get("s2_auc", 0) for r in blend_results], "o-", color="#43A047")
    axes[2].set_xlabel("alpha (LR weight)")
    axes[2].set_ylabel("Stage 2 AUC")
    axes[2].set_title("Stage 2: AUC")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "blend_sweep.png", dpi=150)
    plt.close()

    summary = {
        "experiment": "step5_ensemble",
        "timestamp": datetime.now().isoformat(),
        "models": ["logistic_regression", "resnet18"],
        "aggregate": args.aggregate,
        "n_samples": len(X),
        "cancer_types": list(mc.cancer_types),
        "non_cancer_groups": list(mc.non_cancer_groups),
        "best_alpha": best["alpha"],
        "best_metrics": {k: float(v) for k, v in best.items()},
        "all_blends": blend_results,
        "elapsed": str(elapsed),
    }
    with open(out_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    np.savez_compressed(
        out_dir / "fold_predictions.npz",
        val_bp_lr=val_bp_lr, val_cl_lr=val_cl_lr,
        val_bp_rn=val_bp_rn, val_cl_rn=val_cl_rn,
        binary_labels=bl, cancer_type_labels=ctl,
        groups=groups_arr, sample_ids=sample_ids,
    )

    logger.info(f"\n  Elapsed: {elapsed}")
    logger.info(f"  Output: {out_dir}/")
    logger.info("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
