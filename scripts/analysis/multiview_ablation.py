#!/usr/bin/env python3
"""
Multi-View Derivative Ablation Study
=====================================

Runs E0~E3 ablation + Shared vs Separate encoder comparison.
Uses existing uSERS-Net ResNet18-1D with input_channels={1,2,3}.

Experiments:
  E0: [raw]           — baseline (1ch)
  E1: [raw, d1]       — 2ch
  E2: [raw, d1, d2]   — 3ch (recommended)
  E3: [d1, d2]        — derivatives only (2ch)

Usage:
    python scripts/analysis/multiview_ablation.py
    python scripts/analysis/multiview_ablation.py --experiments E0 E2
    python scripts/analysis/multiview_ablation.py --data medical
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
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.model import (
    ModelConfig, SERSCancerDetector, SeparateEncoderDetector,
    TwoStageLoss, SERSDataset, build_model,
)
from src.sers.config import RESULTS_DIR, FIG_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s", datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = RESULTS_DIR / "multiview_ablation"
FIG_OUTPUT_DIR = FIG_DIR / "analysis" / "multiview_ablation"

# =============================================================================
# Experiment definitions
# =============================================================================
EXPERIMENTS = {
    "E0":     {"channels": [0],       "encoder": "shared",   "desc": "raw only (baseline)"},
    "E1":     {"channels": [0, 1],    "encoder": "shared",   "desc": "raw + d1 (shared)"},
    "E2":     {"channels": [0, 1, 2], "encoder": "shared",   "desc": "raw + d1 + d2 (shared)"},
    "E2-sep": {"channels": [0, 1, 2], "encoder": "separate", "desc": "raw + d1 + d2 (separate encoders)"},
    "E3":     {"channels": [1, 2],    "encoder": "shared",   "desc": "d1 + d2 only"},
}

CANCER_TYPES = ("PRO", "LUN", "CRC", "PAN", "OVA", "BRE", "BLC")
NON_CANCER_GROUPS = ("NOR", "DIA", "HBP", "H.D.")

SG_WINDOW = 11
SG_POLY = 3

PREP_PARAMS = dict(
    do_trim=True, trim_region=(400, 2200),
    do_baseline=True, baseline_window=101,
    normalization="snv",
)


# =============================================================================
# Multi-channel preprocessing — from RAW spectra
# =============================================================================
def preprocess_multichannel_raw(
    x: np.ndarray,
    y: np.ndarray,
    grid: np.ndarray,
    channel_indices: list,
    sg_window: int = SG_WINDOW,
    sg_poly: int = SG_POLY,
) -> np.ndarray:
    """Compute derivative channels from RAW spectrum, then preprocess each independently.

    Pipeline per channel:
        raw y → SG filter (deriv=0/1/2) → trim → baseline → SNV → resample

    Args:
        x: wavenumber array (raw)
        y: intensity array (raw)
        grid: target common grid (trimmed region)
        channel_indices: list of derivative orders [0], [0,1,2], etc.

    Returns: (n_channels, len(grid))
    """
    from src.sers.preprocessing import (
        trim_spectrum, baseline_correction, normalize_spectrum, resample,
    )

    channels = []
    for deriv_order in channel_indices:
        # Step 1: SG filter with derivative on RAW spectrum
        y_sg = savgol_filter(y, sg_window, sg_poly, deriv=deriv_order)

        # Step 2: Trim to fingerprint region
        x_tr, y_tr = trim_spectrum(x.copy(), y_sg, region=PREP_PARAMS["trim_region"])

        # Step 3: Baseline correction (skip for derivatives — they're already baseline-free)
        if deriv_order == 0 and PREP_PARAMS["do_baseline"]:
            y_tr = baseline_correction(y_tr, window=PREP_PARAMS["baseline_window"])

        # Step 4: SNV normalization (per-channel independent)
        y_tr = normalize_spectrum(y_tr, method=PREP_PARAMS["normalization"])

        # Step 5: Resample to common grid
        y_grid = resample(x_tr, y_tr, grid)
        channels.append(y_grid)

    return np.stack(channels, axis=0)  # (n_ch, n_grid)


def load_and_prepare_multichannel(
    data_dir: Path,
    folder_map: dict,
    grid: np.ndarray,
    channel_indices: list,
    pattern: str = "*.CSV",
    max_replicate: dict = None,
) -> tuple:
    """Load raw spectra and build multi-channel dataset.

    Returns: (X_multichannel, df_meta)
    """
    from src.sers.io import find_spectra, read_spectrum, parse_filename

    all_X = []
    meta_rows = []
    failed = 0

    for folder_name, group in folder_map.items():
        folder = data_dir / folder_name
        if not folder.is_dir():
            continue

        files = find_spectra(folder, pattern=pattern, recursive=False)
        if not files:
            files = find_spectra(folder, pattern="*.txt", recursive=False)
        if not files:
            files = find_spectra(folder, pattern="*.csv", recursive=False)

        files = [f for f in files
                 if "_ave" not in f.stem.lower()
                 and "zone.identifier" not in f.name.lower()]

        for fp in files:
            try:
                sid = parse_filename(fp, fallback_group=group)

                # NOR replicate limit
                if max_replicate and sid.group in max_replicate:
                    if sid.replicate > max_replicate[sid.group]:
                        continue

                x, y = read_spectrum(fp)
                ch = preprocess_multichannel_raw(x, y, grid, channel_indices)
                all_X.append(ch)
                meta_rows.append({
                    "group": sid.group, "sample_id": sid.sample_id,
                    "replicate": sid.replicate,
                })
            except Exception:
                failed += 1

    X = np.stack(all_X, axis=0).astype(np.float32)  # (N, n_ch, n_grid)
    df_meta = pd.DataFrame(meta_rows)
    logger.info(f"  Loaded {len(X)} spectra ({failed} failed), "
                f"{len(channel_indices)} channels, {grid.shape[0]} features")
    return X, df_meta


# =============================================================================
# Training loop
# =============================================================================
def train_one_fold(
    X_train, y_bin_train, y_type_train, X_val, y_bin_val, y_type_val,
    config: ModelConfig, device: torch.device, n_epochs: int = 100,
    encoder_type: str = "shared",
) -> dict:
    """Train ResNet18-1D for one fold, return validation metrics."""

    if encoder_type == "separate":
        model = SeparateEncoderDetector(config).to(device)
    else:
        model = SERSCancerDetector(config).to(device)
    criterion = TwoStageLoss(
        stage1_weight=config.stage1_loss_weight,
        stage2_weight=config.stage2_loss_weight,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=config.scheduler_patience, factor=0.5,
    )

    # Datasets
    train_ds = SERSDataset(X_train, y_bin_train, y_type_train, augment=True)
    val_ds = SERSDataset(X_val, y_bin_val, y_type_val, augment=False)

    # Class-balanced sampler
    class_counts = np.bincount(y_bin_train.astype(int), minlength=2)
    weights = 1.0 / class_counts[y_bin_train.astype(int)]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_dl = DataLoader(train_ds, batch_size=config.batch_size, sampler=sampler,
                          num_workers=0, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=config.batch_size * 2, shuffle=False,
                        num_workers=0, pin_memory=True)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(n_epochs):
        # Train
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

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_dl:
                spec = batch["spectra"].to(device)
                bl = batch["binary_label"].to(device)
                ctl = batch["cancer_type_label"].to(device)
                out = model(spec)
                losses = criterion(out, bl, ctl)
                val_losses.append(losses["total"].item())

        mean_val_loss = np.mean(val_losses)
        scheduler.step(mean_val_loss)

        if mean_val_loss < best_val_loss - 1e-4:
            best_val_loss = mean_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                break

    # Load best model and evaluate
    model.load_state_dict(best_state)
    model.eval()

    all_probs, all_labels = [], []
    all_type_probs, all_type_labels = [], []
    with torch.no_grad():
        for batch in val_dl:
            spec = batch["spectra"].to(device)
            out = model(spec)
            all_probs.append(out["binary_prob"].cpu().numpy().squeeze(-1))
            all_labels.append(batch["binary_label"].numpy().squeeze(-1))
            all_type_probs.append(out["cancer_probs"].cpu().numpy())
            all_type_labels.append(batch["cancer_type_label"].numpy())

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    type_probs = np.concatenate(all_type_probs)
    type_labels = np.concatenate(all_type_labels)

    auc = roc_auc_score(labels, probs)
    pred = (probs > 0.5).astype(int)
    f1_bin = f1_score(labels, pred, average="macro")

    # Stage 2: cancer type F1
    cancer_mask = labels == 1
    if cancer_mask.sum() > 0:
        type_pred = type_probs[cancer_mask].argmax(axis=1)
        f1_type = f1_score(type_labels[cancer_mask], type_pred, average="macro", zero_division=0)
    else:
        f1_type = float("nan")

    return {
        "auc": auc,
        "f1_binary": f1_bin,
        "f1_type": f1_type,
        "best_epoch": epoch - patience_counter,
        "n_epochs": epoch + 1,
    }


# =============================================================================
# Run experiment
# =============================================================================
def run_experiment(
    exp_name: str,
    X: np.ndarray,
    df_meta: pd.DataFrame,
    channel_indices: list,
    encoder_type: str = "shared",
    n_splits: int = 5,
    n_epochs: int = 100,
    head_hidden_dim: int = None,
) -> dict:
    """Run full CV for one experiment configuration.

    X: (N, n_all_channels, n_features) — pre-computed multi-channel data
    """

    logger.info(f"\n{'='*60}")
    logger.info(f"  Experiment: {exp_name} — channels {channel_indices}, encoder={encoder_type}")
    logger.info(f"{'='*60}")

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info(f"  Device: {device}")

    # Select channels from pre-computed 3-channel data
    X_exp = X[:, channel_indices, :]
    groups_arr = df_meta["group"].values
    sample_ids = (df_meta["group"] + "_" + df_meta["sample_id"].astype(str)).values

    binary_labels = np.array([1.0 if g in CANCER_TYPES else 0.0 for g in groups_arr])
    cancer_type_map = {ct: i for i, ct in enumerate(CANCER_TYPES)}
    cancer_type_labels = np.array([cancer_type_map.get(g, -1) for g in groups_arr])

    n_ch = len(channel_indices)
    n_feat = X_exp.shape[-1]

    logger.info(f"  Data: {len(X_exp)} spectra, {n_ch} channels, {n_feat} features")
    logger.info(f"  Cancer: {int(binary_labels.sum())}, Non-cancer: {int((binary_labels==0).sum())}")

    # Auto head_hidden_dim: scale with encoder output
    if head_hidden_dim is None:
        if encoder_type == "separate" and n_ch > 1:
            head_hidden_dim = 128  # larger head for 768-dim separate encoder
        else:
            head_hidden_dim = 64

    config = ModelConfig(
        n_spectral_features=n_feat,
        input_channels=n_ch,
        cancer_types=CANCER_TYPES,
        non_cancer_groups=NON_CANCER_GROUPS,
        resnet_channels=(32, 64, 128, 256),
        resnet_blocks=(2, 2, 2, 2),
        encoder_output_dim=256,
        head_hidden_dim=head_hidden_dim,
        dropout_rate=0.5,
        learning_rate=5e-4,
        weight_decay=1e-3,
        batch_size=32,
        n_epochs=n_epochs,
        early_stopping_patience=20,
        scheduler_patience=10,
    )

    # Stratified Group K-Fold
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_exp, binary_labels, sample_ids)):
        logger.info(f"  Fold {fold}...")
        X_tr, X_val = X_exp[train_idx], X_exp[val_idx]
        y_bin_tr, y_bin_val = binary_labels[train_idx], binary_labels[val_idx]
        y_type_tr, y_type_val = cancer_type_labels[train_idx], cancer_type_labels[val_idx]

        metrics = train_one_fold(
            X_tr, y_bin_tr, y_type_tr, X_val, y_bin_val, y_type_val,
            config, device, n_epochs=n_epochs, encoder_type=encoder_type,
        )
        fold_metrics.append(metrics)
        logger.info(f"    AUC={metrics['auc']:.4f}, F1-bin={metrics['f1_binary']:.4f}, "
                    f"F1-type={metrics['f1_type']:.4f} (epoch {metrics['best_epoch']})")

    # Summary
    mean_auc = np.mean([m["auc"] for m in fold_metrics])
    std_auc = np.std([m["auc"] for m in fold_metrics])
    mean_f1_type = np.mean([m["f1_type"] for m in fold_metrics])
    std_f1_type = np.std([m["f1_type"] for m in fold_metrics])
    mean_f1_bin = np.mean([m["f1_binary"] for m in fold_metrics])

    logger.info(f"\n  {exp_name} Summary:")
    logger.info(f"    S1 AUC:   {mean_auc:.4f} ± {std_auc:.4f}")
    logger.info(f"    S1 F1:    {mean_f1_bin:.4f}")
    logger.info(f"    S2 F1:    {mean_f1_type:.4f} ± {std_f1_type:.4f}")

    return {
        "experiment": exp_name,
        "channels": channel_indices,
        "n_channels": n_ch,
        "encoder_type": encoder_type,
        "n_samples": len(X),
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "mean_f1_binary": mean_f1_bin,
        "mean_f1_type": mean_f1_type,
        "std_f1_type": std_f1_type,
        "fold_metrics": fold_metrics,
    }


# =============================================================================
# Summary visualization
# =============================================================================
def plot_ablation_summary(results: list, output_dir: Path):
    """Bar chart comparing all experiments."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    names = [r["experiment"] for r in results]
    aucs = [r["mean_auc"] for r in results]
    auc_errs = [r["std_auc"] for r in results]
    f1s = [r["mean_f1_type"] for r in results]
    f1_errs = [r["std_f1_type"] for r in results]

    colors = ["#2196F3", "#4CAF50", "#E91E63", "#FF9800", "#9C27B0"]

    ax = axes[0]
    bars = ax.bar(range(len(names)), aucs, yerr=auc_errs, color=colors[:len(names)],
                  alpha=0.85, capsize=5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylabel("AUC")
    ax.set_title("Stage 1: Detection AUC", fontweight="bold")
    ax.set_ylim(0.8, 1.0)
    ax.grid(axis="y", alpha=0.3)
    for i, (b, v) in enumerate(zip(bars, aucs)):
        ax.text(i, v + auc_errs[i] + 0.003, f"{v:.4f}", ha="center", fontsize=9)

    ax = axes[1]
    bars = ax.bar(range(len(names)), f1s, yerr=f1_errs, color=colors[:len(names)],
                  alpha=0.85, capsize=5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylabel("F1 Macro")
    ax.set_title("Stage 2: Cancer Type F1", fontweight="bold")
    ax.set_ylim(0.4, 1.0)
    ax.grid(axis="y", alpha=0.3)
    for i, (b, v) in enumerate(zip(bars, f1s)):
        ax.text(i, v + f1_errs[i] + 0.01, f"{v:.4f}", ha="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_dir / "ablation_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved ablation_summary.png")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", "-e", nargs="+", default=None,
                        help="Experiments to run (default: all)")
    parser.add_argument("--data", "-d", default="thermo",
                        choices=["thermo", "medical"],
                        help="Dataset to use")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--head-dim", type=int, default=None,
                        help="Override head_hidden_dim (default: auto based on encoder)")
    parser.add_argument("--tag", type=str, default=None,
                        help="Tag for output filenames")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = datetime.now()

    logger.info("=" * 60)
    logger.info("  Multi-View Derivative Ablation Study (RAW spectra)")
    logger.info("=" * 60)

    # Data directories and folder maps
    THERMO_DIR = PROJECT_ROOT / "data" / "raw_data"
    MEDICAL_DIR = PROJECT_ROOT / "data" / "raw_data_medical"

    THERMO_FOLDER_MAP = {
        "1. Prostate cancer (100개)": "PRO",
        "2. Breast cancer (30개)": "BRE",
        "3. Ovarian cancer (70개)": "OVA",
        "4. Lung cancer (300개)": "LUN",
        "5. Normal (100개)": "NOR",
        "6. Diabetes (100개)": "DIA",
        "7. High blood pressure (100개)": "HBP",
        "8. High blood pressure + Diabetes (100개)": "H.D.",
        "9. Colorectal cancer (300개)": "CRC",
        "10-1. C-Pancreatic cancer (70개)": "CPAN",
        "10-3. Y-Pancreatic cancer (YPAN)": "YPAN",
        "11 BLC (299개)": "BLC",
        "12. Y-Normal (YNOR)": "YNOR",
    }

    MEDICAL_FOLDER_MAP = {
        "1. Prostate cancer (100개)": "PRO",
        "2. Breast cancer (30개)": "BRE",
        "3. Ovarian cancer (70개)": "OVA",
        "4. Lung cancer (300개)": "LUN",
        "5. Normal (100개)": "NOR",
        "6. Diabetes (100개)": "DIA",
        "7. High blood pressure (100개)": "HBP",
        "8. High blood pressure + Diabetes (100개)": "H.D.",
        "9. Colorectal cancer (300개)": "CRC",
        "10-1. C-Pancreatic cancer (70개)": "CPAN",
        "10-3. Y-Pancreatic cancer (30개)": "YPAN",
        "11. Bladdder Cancer (299개)": "BLC",
        "12. Y-Normal (29개)": "YNOR",
    }

    # Common grid (matching production model)
    grid = np.linspace(402.0, 2198.0, 933)

    # Always compute all 3 channels; experiments select subsets
    all_channel_indices = [0, 1, 2]

    if args.data == "thermo":
        data_dir = THERMO_DIR
        folder_map = THERMO_FOLDER_MAP
        pattern = "*.CSV"
        max_rep = None
    else:
        data_dir = MEDICAL_DIR
        folder_map = MEDICAL_FOLDER_MAP
        pattern = "*.txt"
        max_rep = {"NOR": 6}  # reps 7-20 are machine QC

    logger.info(f"\n  Loading RAW spectra from {data_dir.name}...")
    X_all, df_meta = load_and_prepare_multichannel(
        data_dir, folder_map, grid, all_channel_indices,
        pattern=pattern, max_replicate=max_rep,
    )

    # Disambiguate sample_ids before group merge (CPAN_4 ≠ YPAN_4)
    needs_prefix = df_meta["group"].isin(["YPAN", "YNOR"])
    df_meta.loc[needs_prefix, "sample_id"] = (
        df_meta.loc[needs_prefix, "group"] + "_" + df_meta.loc[needs_prefix, "sample_id"].astype(str)
    )
    # Map aliases
    df_meta["group"] = df_meta["group"].replace({"CPAN": "PAN", "YPAN": "PAN", "YNOR": "NOR"})

    # Filter valid groups
    valid_groups = set(CANCER_TYPES) | set(NON_CANCER_GROUPS)
    valid_mask = df_meta["group"].isin(valid_groups).values
    X_all = X_all[valid_mask]
    df_meta = df_meta[valid_mask].reset_index(drop=True)

    logger.info(f"  After filtering: {len(X_all)} spectra")
    logger.info(f"  Groups: {sorted(df_meta['group'].unique())}")
    for g in sorted(df_meta["group"].unique()):
        n = (df_meta["group"] == g).sum()
        logger.info(f"    {g}: {n}")

    # Select experiments
    exp_names = args.experiments or list(EXPERIMENTS.keys())
    logger.info(f"  Experiments: {exp_names}")

    results = []
    for exp_name in exp_names:
        exp = EXPERIMENTS[exp_name]
        result = run_experiment(
            exp_name, X_all, df_meta, exp["channels"],
            encoder_type=exp.get("encoder", "shared"),
            n_splits=args.n_splits, n_epochs=args.n_epochs,
            head_hidden_dim=args.head_dim,
        )
        results.append(result)

    # Summary
    plot_ablation_summary(results, FIG_OUTPUT_DIR)

    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "data": args.data,
        "duration_sec": (datetime.now() - t0).total_seconds(),
        "sg_params": {"window": SG_WINDOW, "polyorder": SG_POLY},
        "results": [{k: v for k, v in r.items() if k != "fold_metrics"} for r in results],
        "fold_details": {r["experiment"]: r["fold_metrics"] for r in results},
    }
    tag = f"_{args.tag}" if args.tag else ""
    with open(OUTPUT_DIR / f"ablation_report_{args.data}{tag}.json", "w") as f:
        json.dump(report, f, indent=2)

    # Print table
    logger.info(f"\n{'='*60}")
    logger.info(f"  ABLATION RESULTS ({args.data.upper()})")
    logger.info(f"{'='*60}")
    logger.info(f"  {'Exp':<6} {'Ch':>3} {'AUC':>12} {'F1-bin':>8} {'F1-type':>12}")
    logger.info(f"  {'-'*6} {'-'*3} {'-'*12} {'-'*8} {'-'*12}")
    for r in results:
        logger.info(f"  {r['experiment']:<6} {r['n_channels']:>3} "
                    f"{r['mean_auc']:.4f}±{r['std_auc']:.3f} "
                    f"{r['mean_f1_binary']:>8.4f} "
                    f"{r['mean_f1_type']:.4f}±{r['std_f1_type']:.3f}")
    logger.info(f"{'='*60}")
    logger.info(f"  Output: {OUTPUT_DIR}")
    logger.info(f"  Duration: {(datetime.now() - t0).total_seconds():.0f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
