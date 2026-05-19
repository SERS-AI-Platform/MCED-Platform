#!/usr/bin/env python3
"""
Multi-View Ensemble: LR + ResNet18-1D
======================================

Derivative channels (raw/d1/d2) from RAW spectra.
LR uses flattened multi-channel features, ResNet18 uses separate encoders.
Ensemble averages their probabilities.

Experiments:
  E0-LR:      LR on raw (935 features) — baseline
  E0-R18:     ResNet18 on raw (1ch) — baseline
  E0-ens:     LR + ResNet18 ensemble on raw
  E2-LR:      LR on raw+d1+d2 (2805 features)
  E2-R18:     ResNet18-separate on raw+d1+d2 (3ch)
  E2-ens:     LR + ResNet18-separate ensemble on raw+d1+d2

Usage:
    python scripts/analysis/multiview_ensemble.py --data thermo
    python scripts/analysis/multiview_ensemble.py --data medical
"""

from __future__ import annotations
import sys, os, json, argparse, logging, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, classification_report

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.model import (
    ModelConfig, SERSCancerDetector, SeparateEncoderDetector,
    TwoStageLoss, SERSDataset,
)
from src.sers.preprocessing import (
    trim_spectrum, baseline_correction, normalize_spectrum, resample,
)
from src.sers.config import RESULTS_DIR, FIG_DIR
from src.sers.io import find_spectra, read_spectrum, parse_filename

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s", datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = RESULTS_DIR / "multiview_ensemble"
FIG_OUTPUT_DIR = FIG_DIR / "analysis" / "multiview_ensemble"

CANCER_TYPES = ("PRO", "LUN", "CRC", "PAN", "OVA", "BRE", "BLC")
NON_CANCER_GROUPS = ("NOR", "DIA", "HBP", "H.D.")

SG_WINDOW = 11
SG_POLY = 3
TRIM_REGION = (400, 2200)


# =============================================================================
# Data loading — RAW spectra → multi-channel
# =============================================================================
def preprocess_channel(x, y, grid, deriv_order):
    """Raw spectrum → SG derivative → trim → baseline → SNV → resample."""
    y_sg = savgol_filter(y, SG_WINDOW, SG_POLY, deriv=deriv_order)
    x_tr, y_tr = trim_spectrum(x.copy(), y_sg, region=TRIM_REGION)
    if deriv_order == 0:
        y_tr = baseline_correction(y_tr, window=101)
    y_tr = normalize_spectrum(y_tr, method="snv")
    return resample(x_tr, y_tr, grid)


def load_raw_multichannel(data_dir, folder_map, grid, pattern="*.CSV", max_rep=None):
    """Load raw spectra → 3-channel (raw, d1, d2) preprocessed array."""
    all_X = []
    meta_rows = []
    failed = 0

    for folder_name, group in folder_map.items():
        folder = data_dir / folder_name
        if not folder.is_dir():
            continue

        files = find_spectra(folder, pattern=pattern, recursive=False)
        if not files:
            for p in ["*.txt", "*.csv", "*.CSV"]:
                files = find_spectra(folder, pattern=p, recursive=False)
                if files:
                    break

        files = [f for f in files
                 if "_ave" not in f.stem.lower()
                 and "zone.identifier" not in f.name.lower()]

        for fp in files:
            try:
                sid = parse_filename(fp, fallback_group=group)
                if max_rep and sid.group in max_rep and sid.replicate > max_rep[sid.group]:
                    continue

                x, y = read_spectrum(fp)
                ch0 = preprocess_channel(x, y, grid, 0)
                ch1 = preprocess_channel(x, y, grid, 1)
                ch2 = preprocess_channel(x, y, grid, 2)
                all_X.append(np.stack([ch0, ch1, ch2], axis=0))
                meta_rows.append({"group": sid.group, "sample_id": sid.sample_id,
                                  "replicate": sid.replicate})
            except Exception:
                failed += 1

    X = np.stack(all_X, axis=0).astype(np.float32)
    df = pd.DataFrame(meta_rows)
    logger.info(f"  Loaded {len(X)} spectra ({failed} failed), shape {X.shape}")
    return X, df


# =============================================================================
# LR Training
# =============================================================================
def train_lr(X_train, y_bin_train, y_type_train, X_val):
    """Train LR for both stages. Returns (s1_probs, s2_probs) on val set."""
    # Stage 1: binary
    s1 = make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, max_iter=2000, solver="saga", class_weight="balanced"))
    s1.fit(X_train, y_bin_train)
    s1_prob = s1.predict_proba(X_val)[:, 1]

    # Stage 2: cancer type (train on cancer samples only)
    cancer_mask = y_bin_train == 1
    s2_prob = np.zeros((len(X_val), len(CANCER_TYPES)))
    if cancer_mask.sum() > 10:
        s2 = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, max_iter=2000, solver="saga", class_weight="balanced",
            multi_class="multinomial"))
        s2.fit(X_train[cancer_mask], y_type_train[cancer_mask])
        s2_prob = s2.predict_proba(X_val)
        # Align columns to CANCER_TYPES order
        s2_prob_aligned = np.zeros((len(X_val), len(CANCER_TYPES)))
        for i, cls in enumerate(s2.classes_):
            s2_prob_aligned[:, cls] = s2_prob[:, i]
        s2_prob = s2_prob_aligned

    return s1_prob, s2_prob


# =============================================================================
# ResNet18 Training
# =============================================================================
def train_resnet(X_train, y_bin_train, y_type_train, X_val,
                 config, device, n_epochs=200, separate=False):
    """Train ResNet18 for both stages. Returns (s1_probs, s2_probs) on val set."""
    if separate and config.input_channels > 1:
        model = SeparateEncoderDetector(config).to(device)
    else:
        model = SERSCancerDetector(config).to(device)

    criterion = TwoStageLoss(stage1_weight=1.0, stage2_weight=1.0,
                             use_focal_loss=True, focal_gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    train_ds = SERSDataset(X_train, y_bin_train, y_type_train, augment=True)
    val_ds = SERSDataset(X_val, np.zeros(len(X_val)), np.zeros(len(X_val)), augment=False)

    # Class-balanced sampler
    class_counts = np.bincount(y_bin_train.astype(int), minlength=2).clip(1)
    weights = 1.0 / class_counts[y_bin_train.astype(int)]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_dl = DataLoader(train_ds, batch_size=32, sampler=sampler, num_workers=0, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

    best_loss, best_state, patience = float("inf"), None, 0
    for epoch in range(n_epochs):
        model.train()
        for batch in train_dl:
            spec = batch["spectra"].to(device)
            bl = batch["binary_label"].to(device)
            ctl = batch["cancer_type_label"].to(device)
            out = model(spec)
            losses = criterion(out, bl, ctl)
            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Quick val loss check every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            vl = []
            with torch.no_grad():
                for batch in DataLoader(
                    SERSDataset(X_val[:200], y_bin_train[:200] if len(y_bin_train) >= 200 else y_bin_train,
                                y_type_train[:200] if len(y_type_train) >= 200 else y_type_train),
                    batch_size=64
                ):
                    out = model(batch["spectra"].to(device))
                    l = criterion(out, batch["binary_label"].to(device),
                                  batch["cancer_type_label"].to(device))
                    vl.append(l["total"].item())
            mean_vl = np.mean(vl)
            if mean_vl < best_loss - 1e-4:
                best_loss = mean_vl
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= 8:  # 8 checks × 5 epochs = 40 epoch patience
                    break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    s1_probs, s2_probs = [], []
    with torch.no_grad():
        for batch in val_dl:
            out = model(batch["spectra"].to(device))
            s1_probs.append(out["binary_prob"].cpu().numpy().squeeze(-1))
            s2_probs.append(out["cancer_probs"].cpu().numpy())

    return np.concatenate(s1_probs), np.concatenate(s2_probs)


# =============================================================================
# Full CV experiment
# =============================================================================
def run_cv(X_3ch, df_meta, n_splits=5, n_epochs=200):
    """Run full CV for all 6 experiments simultaneously."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"  Device: {device}")

    valid = set(CANCER_TYPES) | set(NON_CANCER_GROUPS)
    mask = df_meta["group"].isin(valid).values
    X_3ch = X_3ch[mask]
    df_meta = df_meta[mask].reset_index(drop=True)

    groups_arr = df_meta["group"].values
    sample_ids = (df_meta["group"] + "_" + df_meta["sample_id"].astype(str)).values
    binary_labels = np.array([1.0 if g in CANCER_TYPES else 0.0 for g in groups_arr])
    ct_map = {ct: i for i, ct in enumerate(CANCER_TYPES)}
    cancer_type_labels = np.array([ct_map.get(g, -1) for g in groups_arr])

    n_feat = X_3ch.shape[-1]
    logger.info(f"  Data: {len(X_3ch)} spectra, cancer={int(binary_labels.sum())}, "
                f"non-cancer={int((binary_labels==0).sum())}")

    # Experiment configs
    experiments = {
        "E0-LR":  {"ch": [0], "model": "lr"},
        "E0-R18": {"ch": [0], "model": "r18"},
        "E0-ens": {"ch": [0], "model": "ensemble"},
        "E2-LR":  {"ch": [0, 1, 2], "model": "lr"},
        "E2-R18": {"ch": [0, 1, 2], "model": "r18_sep"},
        "E2-ens": {"ch": [0, 1, 2], "model": "ensemble_sep"},
    }

    # Storage for all predictions
    all_results = {name: {"s1_probs": np.full(len(X_3ch), np.nan),
                          "s2_probs": np.full((len(X_3ch), len(CANCER_TYPES)), np.nan)}
                   for name in experiments}

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(X_3ch, binary_labels, sample_ids)):
        logger.info(f"\n  ── Fold {fold} ──")
        y_bin_tr = binary_labels[tr_idx]
        y_type_tr = cancer_type_labels[tr_idx]
        y_bin_val = binary_labels[val_idx]
        y_type_val = cancer_type_labels[val_idx]

        for exp_name, exp in experiments.items():
            ch = exp["ch"]
            n_ch = len(ch)

            X_tr = X_3ch[tr_idx][:, ch, :]
            X_val = X_3ch[val_idx][:, ch, :]

            # Flatten for LR
            X_tr_flat = X_tr.reshape(len(X_tr), -1)
            X_val_flat = X_val.reshape(len(X_val), -1)

            model_type = exp["model"]

            if model_type == "lr":
                logger.info(f"    {exp_name}: LR on {n_ch}ch×{n_feat}={X_tr_flat.shape[1]} features")
                s1_p, s2_p = train_lr(X_tr_flat, y_bin_tr, y_type_tr, X_val_flat)

            elif model_type in ("r18", "r18_sep"):
                separate = "sep" in model_type
                cfg = ModelConfig(
                    n_spectral_features=n_feat, input_channels=n_ch,
                    cancer_types=CANCER_TYPES, non_cancer_groups=NON_CANCER_GROUPS,
                    resnet_channels=(32, 64, 128, 256), encoder_output_dim=256,
                    head_hidden_dim=128, dropout_rate=0.5,
                    learning_rate=5e-4, weight_decay=1e-3,
                    early_stopping_patience=40, scheduler_patience=15,
                )
                logger.info(f"    {exp_name}: ResNet18{'(sep)' if separate else ''} on {n_ch}ch")
                s1_p, s2_p = train_resnet(X_tr, y_bin_tr, y_type_tr, X_val,
                                          cfg, device, n_epochs, separate)

            elif model_type in ("ensemble", "ensemble_sep"):
                separate = "sep" in model_type
                # LR
                logger.info(f"    {exp_name}: LR + ResNet18{'(sep)' if separate else ''} ensemble")
                s1_lr, s2_lr = train_lr(X_tr_flat, y_bin_tr, y_type_tr, X_val_flat)
                # ResNet18
                cfg = ModelConfig(
                    n_spectral_features=n_feat, input_channels=n_ch,
                    cancer_types=CANCER_TYPES, non_cancer_groups=NON_CANCER_GROUPS,
                    resnet_channels=(32, 64, 128, 256), encoder_output_dim=256,
                    head_hidden_dim=128, dropout_rate=0.5,
                    learning_rate=5e-4, weight_decay=1e-3,
                    early_stopping_patience=40, scheduler_patience=15,
                )
                s1_r18, s2_r18 = train_resnet(X_tr, y_bin_tr, y_type_tr, X_val,
                                              cfg, device, n_epochs, separate)
                # Average
                s1_p = (s1_lr + s1_r18) / 2.0
                s2_p = (s2_lr + s2_r18) / 2.0

            all_results[exp_name]["s1_probs"][val_idx] = s1_p
            all_results[exp_name]["s2_probs"][val_idx] = s2_p

        # Report fold metrics
        for exp_name in experiments:
            s1_p = all_results[exp_name]["s1_probs"][val_idx]
            auc = roc_auc_score(y_bin_val, s1_p)
            cancer_mask = y_bin_val == 1
            if cancer_mask.sum() > 0:
                s2_p = all_results[exp_name]["s2_probs"][val_idx]
                f1_t = f1_score(y_type_val[cancer_mask],
                               s2_p[cancer_mask].argmax(axis=1),
                               average="macro", zero_division=0)
            else:
                f1_t = float("nan")
            logger.info(f"      {exp_name:<10} AUC={auc:.4f} F1-type={f1_t:.4f}")

    # Overall metrics
    results_summary = []
    for exp_name in experiments:
        s1_p = all_results[exp_name]["s1_probs"]
        s2_p = all_results[exp_name]["s2_probs"]

        valid_mask = ~np.isnan(s1_p)
        auc = roc_auc_score(binary_labels[valid_mask], s1_p[valid_mask])
        s1_pred = (s1_p > 0.5).astype(int)
        f1_bin = f1_score(binary_labels[valid_mask], s1_pred[valid_mask], average="macro")

        cancer_mask = binary_labels == 1
        both_mask = valid_mask & cancer_mask
        f1_type = f1_score(cancer_type_labels[both_mask],
                          s2_p[both_mask].argmax(axis=1),
                          average="macro", zero_division=0)

        # Per-fold AUC for std
        fold_aucs = []
        for _, val_idx in sgkf.split(X_3ch, binary_labels, sample_ids):
            fa = roc_auc_score(binary_labels[val_idx], s1_p[val_idx])
            fold_aucs.append(fa)

        results_summary.append({
            "experiment": exp_name,
            "n_channels": len(experiments[exp_name]["ch"]),
            "model": experiments[exp_name]["model"],
            "auc": auc,
            "auc_std": np.std(fold_aucs),
            "f1_binary": f1_bin,
            "f1_type": f1_type,
        })

    return results_summary


# =============================================================================
# Visualization
# =============================================================================
def plot_results(results, output_dir, data_name):
    """Summary bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    names = [r["experiment"] for r in results]
    aucs = [r["auc"] for r in results]
    f1s = [r["f1_type"] for r in results]

    colors = {"E0-LR": "#90CAF9", "E0-R18": "#42A5F5", "E0-ens": "#1565C0",
              "E2-LR": "#A5D6A7", "E2-R18": "#66BB6A", "E2-ens": "#2E7D32"}
    c = [colors.get(n, "#999") for n in names]

    ax = axes[0]
    bars = ax.bar(range(len(names)), aucs, color=c, alpha=0.9)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("AUC")
    ax.set_title(f"Stage 1: Detection AUC ({data_name})", fontweight="bold")
    ax.set_ylim(0.85, 1.0)
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(aucs):
        ax.text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=8)

    ax = axes[1]
    bars = ax.bar(range(len(names)), f1s, color=c, alpha=0.9)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("F1 Macro")
    ax.set_title(f"Stage 2: Cancer Type F1 ({data_name})", fontweight="bold")
    ax.set_ylim(0.4, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0.9, color="red", linestyle="--", alpha=0.5, label="Target 0.9")
    ax.legend()
    for i, v in enumerate(f1s):
        ax.text(i, v + 0.01, f"{v:.4f}", ha="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_dir / f"ensemble_results_{data_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", default="thermo", choices=["thermo", "medical"])
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-epochs", type=int, default=200)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = datetime.now()

    logger.info("=" * 64)
    logger.info("  Multi-View Ensemble: LR + ResNet18-1D")
    logger.info("=" * 64)

    THERMO_DIR = PROJECT_ROOT / "data" / "raw_data"
    MEDICAL_DIR = PROJECT_ROOT / "data" / "raw_data_medical"

    THERMO_MAP = {
        "1. Prostate cancer (100개)": "PRO", "2. Breast cancer (30개)": "BRE",
        "3. Ovarian cancer (70개)": "OVA", "4. Lung cancer (300개)": "LUN",
        "5. Normal (100개)": "NOR", "6. Diabetes (100개)": "DIA",
        "7. High blood pressure (100개)": "HBP",
        "8. High blood pressure + Diabetes (100개)": "H.D.",
        "9. Colorectal cancer (300개)": "CRC",
        "10-1. C-Pancreatic cancer (70개)": "CPAN",
        "10-3. Y-Pancreatic cancer (YPAN)": "YPAN",
        "11 BLC (299개)": "BLC", "12. Y-Normal (YNOR)": "YNOR",
    }
    MEDICAL_MAP = {
        "1. Prostate cancer (100개)": "PRO", "2. Breast cancer (30개)": "BRE",
        "3. Ovarian cancer (70개)": "OVA", "4. Lung cancer (300개)": "LUN",
        "5. Normal (100개)": "NOR", "6. Diabetes (100개)": "DIA",
        "7. High blood pressure (100개)": "HBP",
        "8. High blood pressure + Diabetes (100개)": "H.D.",
        "9. Colorectal cancer (300개)": "CRC",
        "10-1. C-Pancreatic cancer (70개)": "CPAN",
        "10-3. Y-Pancreatic cancer (30개)": "YPAN",
        "11. Bladdder Cancer (299개)": "BLC",
        "12. Y-Normal (29개)": "YNOR",
    }

    grid = np.linspace(402.0, 2198.0, 933)

    if args.data == "thermo":
        data_dir, folder_map, pattern, max_rep = THERMO_DIR, THERMO_MAP, "*.CSV", None
    else:
        data_dir, folder_map, pattern, max_rep = MEDICAL_DIR, MEDICAL_MAP, "*.txt", {"NOR": 6}

    logger.info(f"\n  Loading RAW spectra ({args.data})...")
    X_3ch, df_meta = load_raw_multichannel(data_dir, folder_map, grid, pattern, max_rep)
    # Disambiguate sample_ids before group merge (CPAN_4 ≠ YPAN_4)
    needs_prefix = df_meta["group"].isin(["YPAN", "YNOR"])
    df_meta.loc[needs_prefix, "sample_id"] = (
        df_meta.loc[needs_prefix, "group"] + "_" + df_meta.loc[needs_prefix, "sample_id"].astype(str)
    )
    df_meta["group"] = df_meta["group"].replace({"CPAN": "PAN", "YPAN": "PAN", "YNOR": "NOR"})

    logger.info(f"\n  Running 5-fold CV (6 experiments)...")
    results = run_cv(X_3ch, df_meta, n_splits=args.n_splits, n_epochs=args.n_epochs)

    # Output
    plot_results(results, FIG_OUTPUT_DIR, args.data)

    report = {
        "timestamp": datetime.now().isoformat(),
        "data": args.data,
        "duration_sec": (datetime.now() - t0).total_seconds(),
        "results": results,
    }
    with open(OUTPUT_DIR / f"ensemble_report_{args.data}.json", "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"\n{'='*64}")
    logger.info(f"  ENSEMBLE RESULTS ({args.data.upper()})")
    logger.info(f"{'='*64}")
    logger.info(f"  {'Exp':<10} {'AUC':>10} {'F1-bin':>8} {'F1-type':>8}")
    logger.info(f"  {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
    for r in results:
        logger.info(f"  {r['experiment']:<10} {r['auc']:>10.4f} {r['f1_binary']:>8.4f} {r['f1_type']:>8.4f}")
    logger.info(f"{'='*64}")
    logger.info(f"  Duration: {(datetime.now() - t0).total_seconds():.0f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
