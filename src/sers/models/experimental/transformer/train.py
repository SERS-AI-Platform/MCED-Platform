#!/usr/bin/env python3
"""
Spectral Patch Transformer — Training & Evaluation
====================================================

Usage:
    python models/transformer/train.py --data thermo
"""

from __future__ import annotations
import sys, json, argparse, logging, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, f1_score

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sers.models._legacy.resnet_v1.model import ModelConfig, TwoStageLoss, SERSDataset
from sers.models.experimental.transformer.model import SpectralTransformerDetector
from src.sers.config import RESULTS_DIR

from sers.models.usersnet.stacking import preprocess_channel, load_raw_multichannel

logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(levelname)-7s │ %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

OUTPUT_DIR = RESULTS_DIR / "transformer"
CANCER_TYPES = ("PRO", "LUN", "CRC", "CPAN", "OVA", "BRE", "BLC")
NON_CANCER_GROUPS = ("NOR", "DIA", "HBP", "H.D.")


def train_transformer_fold(X_train, y_bin_train, y_type_train,
                           X_val, y_bin_val, y_type_val,
                           config, transformer_config, device, n_epochs=200):
    """Train SpectralTransformer for one fold."""
    model = SpectralTransformerDetector(config, transformer_config).to(device)
    criterion = TwoStageLoss(use_focal_loss=True, focal_gamma=2.0, label_smoothing=0.1)

    # Warmup + cosine annealing
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    warmup_epochs = 10

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        progress = (epoch - warmup_epochs) / (n_epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    train_ds = SERSDataset(X_train, y_bin_train, y_type_train, augment=True,
                           noise_std=0.02, scale_range=(0.95, 1.05), shift_max=3)
    val_ds = SERSDataset(X_val, y_bin_val, y_type_val, augment=False)

    class_counts = np.bincount(y_bin_train.astype(int), minlength=2).clip(1)
    weights = 1.0 / class_counts[y_bin_train.astype(int)]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_dl = DataLoader(train_ds, batch_size=64, sampler=sampler, num_workers=0, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

    best_loss, best_state, patience = float("inf"), None, 0
    for epoch in range(n_epochs):
        model.train()
        for batch in train_dl:
            out = model(batch["spectra"].to(device))
            losses = criterion(out, batch["binary_label"].to(device), batch["cancer_type_label"].to(device))
            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        if (epoch + 1) % 5 == 0:
            model.eval()
            vl = []
            with torch.no_grad():
                for batch in val_dl:
                    out = model(batch["spectra"].to(device))
                    l = criterion(out, batch["binary_label"].to(device), batch["cancer_type_label"].to(device))
                    vl.append(l["total"].item())
            mean_vl = np.mean(vl)
            if mean_vl < best_loss - 1e-4:
                best_loss = mean_vl
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= 8:
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

    return np.concatenate(s1_probs), np.concatenate(s2_probs), epoch + 1


def run_transformer_cv(X_3ch, df_meta, n_splits=5, n_epochs=200):
    """Run full CV for Spectral Transformer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    valid = set(CANCER_TYPES) | set(NON_CANCER_GROUPS)
    mask = df_meta["group"].isin(valid).values
    X_3ch = X_3ch[mask]
    df_meta = df_meta[mask].reset_index(drop=True)

    groups_arr = df_meta["group"].values
    sample_ids = (df_meta["group"] + "_" + df_meta["sample_id"].astype(str)).values
    binary_labels = np.array([1.0 if g in CANCER_TYPES else 0.0 for g in groups_arr])
    ct_map = {ct: i for i, ct in enumerate(CANCER_TYPES)}
    cancer_type_labels = np.array([ct_map.get(g, -1) for g in groups_arr])

    n_feat = X_3ch.shape[2]
    n_ch = X_3ch.shape[1]

    config = ModelConfig(n_spectral_features=n_feat, input_channels=n_ch,
                         cancer_types=CANCER_TYPES, non_cancer_groups=NON_CANCER_GROUPS,
                         head_hidden_dim=128, dropout_rate=0.5)
    transformer_config = {
        "patch_size": 21, "d_model": 128, "n_heads": 4,
        "n_layers": 3, "ffn_dim": 256, "dropout": 0.2, "drop_path_rate": 0.1,
    }

    total_params = sum(p.numel() for p in SpectralTransformerDetector(config, transformer_config).parameters())
    logger.info(f"  Model: SpectralPatchTransformer, {total_params:,} params")
    logger.info(f"  Config: {transformer_config}")
    logger.info(f"  Data: {len(X_3ch)} spectra, {n_ch}ch, {n_feat} features")
    logger.info(f"  Device: {device}")

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    s1_all = np.full(len(X_3ch), np.nan)
    s2_all = np.full((len(X_3ch), len(CANCER_TYPES)), np.nan)
    fold_metrics = []

    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(X_3ch, binary_labels, sample_ids)):
        logger.info(f"\n  Fold {fold}...")
        s1_p, s2_p, last_epoch = train_transformer_fold(
            X_3ch[tr_idx], binary_labels[tr_idx], cancer_type_labels[tr_idx],
            X_3ch[val_idx], binary_labels[val_idx], cancer_type_labels[val_idx],
            config, transformer_config, device, n_epochs)

        s1_all[val_idx] = s1_p
        s2_all[val_idx] = s2_p

        auc = roc_auc_score(binary_labels[val_idx], s1_p)
        bm = binary_labels[val_idx] == 1
        f1t = f1_score(cancer_type_labels[val_idx][bm], s2_p[bm].argmax(axis=1),
                       average="macro", zero_division=0)
        fold_metrics.append({"fold": fold, "auc": auc, "f1_type": f1t, "last_epoch": last_epoch})
        logger.info(f"    AUC={auc:.4f}, F1-type={f1t:.4f} (epoch {last_epoch})")

    # Overall
    auc = roc_auc_score(binary_labels, s1_all)
    f1b = f1_score(binary_labels, (s1_all > 0.5).astype(int), average="macro")
    bm = binary_labels == 1
    f1t = f1_score(cancer_type_labels[bm], s2_all[bm].argmax(axis=1), average="macro", zero_division=0)

    return {"auc": auc, "f1_bin": f1b, "f1_type": f1t, "fold_metrics": fold_metrics,
            "params": total_params}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", default="thermo", choices=["thermo", "medical"])
    parser.add_argument("--n-epochs", type=int, default=200)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = datetime.now()

    logger.info("=" * 64)
    logger.info("  Spectral Patch Transformer — SERS Classification")
    logger.info("=" * 64)

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
    data_dir = PROJECT_ROOT / "data" / ("raw_data" if args.data == "thermo" else "raw_data_medical")
    folder_map = THERMO_MAP if args.data == "thermo" else MEDICAL_MAP
    pattern = "*.CSV" if args.data == "thermo" else "*.txt"
    max_rep = None if args.data == "thermo" else {"NOR": 6}

    logger.info(f"\n  Loading RAW spectra ({args.data})...")
    X_3ch, df_meta = load_raw_multichannel(data_dir, folder_map, grid, pattern, max_rep)
    df_meta["group"] = df_meta["group"].replace({"YPAN": "CPAN", "YNOR": "NOR"})

    result = run_transformer_cv(X_3ch, df_meta, n_epochs=args.n_epochs)

    report = {"timestamp": datetime.now().isoformat(), "data": args.data,
              "duration_sec": (datetime.now() - t0).total_seconds(), **result}
    with open(OUTPUT_DIR / f"transformer_report_{args.data}.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"\n{'='*64}")
    logger.info(f"  TRANSFORMER RESULTS ({args.data.upper()})")
    logger.info(f"{'='*64}")
    logger.info(f"  AUC:     {result['auc']:.4f}")
    logger.info(f"  F1-bin:  {result['f1_bin']:.4f}")
    logger.info(f"  F1-type: {result['f1_type']:.4f}")
    logger.info(f"  Params:  {result['params']:,}")
    logger.info(f"  Duration: {(datetime.now() - t0).total_seconds():.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
