"""
Step 3: Multi-Channel Input Experiment for ResNet18-1D

Instead of feeding SNV-preprocessed spectra (1 channel),
we create a 3-channel input:
  Ch0: SNV-preprocessed spectrum (baseline)
  Ch1: 1st derivative (spectral slope — peak positions)
  Ch2: 2nd derivative (curvature — peak shapes)

This gives the CNN raw structural information that SNV alone discards,
potentially allowing the encoder to learn features that LR cannot access.

Usage:
    python models/train_multichannel.py
    python models/train_multichannel.py --epochs 200 --lr 3e-4
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
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
)

import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.model import (
    ModelConfig, TwoStageLoss, ResNet1DEncoder, BasicBlock1D,
    BinaryHead, CancerTypeHead, model_summary,
)
from models.legacy.scripts.train import (
    load_processed_spectra, get_feature_columns,
    apply_class_selection, resolve_aliases, aggregate_replicates,
    create_labels, EarlyStopping, _compute_overall_metrics,
    compute_effective_num_weights, MODEL_DISPLAY_NAMES,
)
from src.sers.config import RESULTS_DIR

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


# =============================================================================
# Multi-Channel Feature Engineering
# =============================================================================
def make_multichannel(X: np.ndarray) -> np.ndarray:
    """Convert (N, L) SNV spectra to (N, 3, L) multi-channel.

    Channels:
        0: Original SNV spectrum
        1: 1st derivative (np.gradient)
        2: 2nd derivative (np.gradient of 1st)

    Each channel is independently standardized (zero mean, unit std).
    """
    N, L = X.shape
    X_mc = np.zeros((N, 3, L), dtype=np.float32)

    for i in range(N):
        ch0 = X[i]
        ch1 = np.gradient(ch0)
        ch2 = np.gradient(ch1)

        # Standardize each channel
        for j, ch in enumerate([ch0, ch1, ch2]):
            std = ch.std()
            if std > 1e-10:
                X_mc[i, j] = (ch - ch.mean()) / std
            else:
                X_mc[i, j] = ch - ch.mean()

    return X_mc


# =============================================================================
# Multi-Channel ResNet Encoder (3 input channels instead of 1)
# =============================================================================
class MultiChannelResNet1DEncoder(nn.Module):
    """ResNet18-1D encoder accepting 3-channel input."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        channels = config.resnet_channels
        blocks = config.resnet_blocks

        # Stem: 3 input channels
        self.conv1 = nn.Conv1d(3, channels[0], kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(channels[0], channels[0], blocks[0], stride=1)
        self.layer2 = self._make_layer(channels[0], channels[1], blocks[1], stride=2)
        self.layer3 = self._make_layer(channels[1], channels[2], blocks[2], stride=2)
        self.layer4 = self._make_layer(channels[2], channels[3], blocks[3], stride=2)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_dim = channels[-1]
        self._init_weights()

    def _make_layer(self, in_ch, out_ch, n_blocks, stride=1):
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        layers = [BasicBlock1D(in_ch, out_ch, stride, downsample)]
        for _ in range(1, n_blocks):
            layers.append(BasicBlock1D(out_ch, out_ch))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, 3, L)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x)
        return x.squeeze(-1)


class MultiChannelCancerDetector(nn.Module):
    """Two-stage cancer detector with 3-channel ResNet encoder."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = MultiChannelResNet1DEncoder(config)
        enc_dim = self.encoder.output_dim

        self.binary_head = BinaryHead(enc_dim, config.head_hidden_dim, config.dropout_rate)
        self.cancer_type_head = CancerTypeHead(
            enc_dim, config.n_cancer_types, config.head_hidden_dim, config.dropout_rate
        )
        self.register_buffer(
            "cancer_thresholds",
            torch.full((config.n_cancer_types,), config.default_threshold),
        )

    def forward(self, spectra):
        import torch.nn.functional as F
        emb = self.encoder(spectra)
        binary_logit = self.binary_head(emb)
        cancer_logits = self.cancer_type_head(emb)
        return {
            "binary_logit": binary_logit,
            "binary_prob": torch.sigmoid(binary_logit),
            "cancer_logits": cancer_logits,
            "cancer_probs": F.softmax(cancer_logits, dim=-1),
            "embedding": emb,
        }


# =============================================================================
# Dataset with multi-channel augmentation
# =============================================================================
class MultiChannelDataset(torch.utils.data.Dataset):
    def __init__(self, X_mc, binary_labels, cancer_type_labels, augment=False):
        self.spectra = torch.FloatTensor(X_mc)  # (N, 3, L)
        self.binary_labels = torch.FloatTensor(binary_labels).unsqueeze(-1)
        self.cancer_type_labels = torch.LongTensor(cancer_type_labels)
        self.augment = augment

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, idx):
        spec = self.spectra[idx]
        if self.augment:
            spec = spec + torch.randn_like(spec) * 0.02
            scale = torch.empty(1).uniform_(0.95, 1.05).item()
            spec = spec * scale
            shift = torch.randint(-3, 4, (1,)).item()
            if shift != 0:
                spec = torch.roll(spec, shifts=shift, dims=-1)
        return {
            "spectra": spec,
            "binary_label": self.binary_labels[idx],
            "cancer_type_label": self.cancer_type_labels[idx],
        }


# =============================================================================
# Training loop (reuses logic from train.py)
# =============================================================================
def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, use_amp=False):
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        spec = batch["spectra"].to(device)
        by = batch["binary_label"].to(device)
        cy = batch["cancer_type_label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            out = model(spec)
            losses = criterion(out, by, cy)
        if scaler is not None and use_amp:
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total += losses["total"].item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp=False):
    model.eval()
    total, n = 0.0, 0
    all_bp, all_bt, all_cl, all_ct, all_emb = [], [], [], [], []
    for batch in loader:
        spec = batch["spectra"].to(device)
        by = batch["binary_label"].to(device)
        cy = batch["cancer_type_label"].to(device)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            out = model(spec)
            losses = criterion(out, by, cy)
        total += losses["total"].item(); n += 1
        all_bp.append(out["binary_prob"].cpu().numpy())
        all_bt.append(by.cpu().numpy())
        all_cl.append(out["cancer_logits"].cpu().numpy())
        all_ct.append(cy.cpu().numpy())
        all_emb.append(out["embedding"].cpu().numpy())

    bp = np.concatenate(all_bp).squeeze()
    bt = np.concatenate(all_bt).squeeze()
    cl = np.concatenate(all_cl)
    ct = np.concatenate(all_ct)
    emb = np.concatenate(all_emb)

    auc_s1 = roc_auc_score(bt, bp) if len(np.unique(bt)) > 1 else float("nan")
    pred_s1 = (bp > 0.5).astype(int)
    acc_s1 = accuracy_score(bt, pred_s1)

    cm = ct >= 0
    f1_macro_s2 = float("nan")
    if cm.sum() > 0 and len(np.unique(ct[cm])) > 1:
        cp = torch.softmax(torch.tensor(cl[cm]), dim=-1).numpy()
        cpred = cp.argmax(axis=1)
        f1_macro_s2 = f1_score(ct[cm], cpred, average="macro", zero_division=0)

    return {
        "loss": total / max(n, 1),
        "auc_s1": auc_s1, "acc_s1": acc_s1,
        "f1_macro_s2": f1_macro_s2,
        "binary_prob": bp, "binary_true": bt,
        "cancer_logits": cl, "cancer_true": ct,
        "embedding": emb,
    }


def run_multichannel_cv(X_mc, bl, ctl, sample_ids, groups_arr, config, device, n_splits=5, epochs=150):
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=config.random_state)
    use_amp = device.type == "cuda"

    n_samples = len(X_mc)
    val_binary_prob = np.full(n_samples, np.nan)
    val_cancer_logits = np.full((n_samples, config.n_cancer_types), np.nan)
    val_embedding = np.full((n_samples, config.encoder_output_dim), np.nan)
    train_last = {}

    for fold_i, (train_idx, val_idx) in enumerate(cv.split(X_mc, bl, sample_ids)):
        logger.info(f"\n  -- Fold {fold_i+1}/{n_splits} (train={len(train_idx)}, val={len(val_idx)}) --")

        train_ds = MultiChannelDataset(X_mc[train_idx], bl[train_idx], ctl[train_idx], augment=True)
        val_ds = MultiChannelDataset(X_mc[val_idx], bl[val_idx], ctl[val_idx], augment=False)

        # Sampler
        bl_train = bl[train_idx]
        ctl_train = ctl[train_idx]
        sw = np.ones(len(train_ds), dtype=np.float64)
        n_pos, n_neg = (bl_train > 0.5).sum(), (bl_train <= 0.5).sum()
        for i in range(len(train_ds)):
            if bl_train[i] > 0.5:
                ct_count = (ctl_train[ctl_train >= 0] == ctl_train[i]).sum()
                sw[i] = 1.0 / max(ct_count, 1)
            else:
                sw[i] = 1.0 / max(n_neg, 1)
        sw = sw / sw.mean()
        sampler = WeightedRandomSampler(sw, len(train_ds), replacement=True)

        train_loader = DataLoader(train_ds, batch_size=config.batch_size, sampler=sampler, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        # Loss
        pw = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
        cm_mask = ctl_train >= 0
        cw_t = None
        if cm_mask.sum() > 0:
            cc = np.bincount(ctl_train[cm_mask], minlength=config.n_cancer_types).astype(float)
            cw = 1.0 / np.maximum(cc, 1)
            cw = cw / cw.sum() * config.n_cancer_types
            cw_t = torch.tensor(cw, dtype=torch.float32).to(device)

        criterion = TwoStageLoss(1.0, 1.0, pw, cw_t, label_smoothing=0.1)
        model = MultiChannelCancerDetector(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=config.scheduler_patience, factor=0.5, min_lr=1e-6)
        es = EarlyStopping(config.early_stopping_patience)
        scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)

        best_val_loss, best_state = float("inf"), None
        for epoch in range(epochs):
            tl = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, use_amp)
            vm = evaluate(model, val_loader, criterion, device, use_amp)
            scheduler.step(vm["loss"])
            lr = optimizer.param_groups[0]["lr"]

            if vm["loss"] < best_val_loss:
                best_val_loss = vm["loss"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"    Fold {fold_i+1} Ep {epoch+1:3d} | "
                    f"train={tl:.4f} val={vm['loss']:.4f} "
                    f"S1_AUC={vm['auc_s1']:.3f} S2_F1={vm['f1_macro_s2']:.3f} lr={lr:.1e}"
                )
            if es.step(vm["loss"]):
                logger.info(f"    Early stop at epoch {epoch+1}")
                break

        if best_state:
            model.load_state_dict(best_state)

        val_metrics = evaluate(model, val_loader, criterion, device, use_amp)
        train_metrics = evaluate(model, train_loader, criterion, device, use_amp)

        val_binary_prob[val_idx] = val_metrics["binary_prob"]
        val_cancer_logits[val_idx] = val_metrics["cancer_logits"]
        val_embedding[val_idx] = val_metrics["embedding"]
        train_last = {
            "binary_prob": train_metrics["binary_prob"],
            "binary_true": train_metrics["binary_true"],
            "cancer_logits": train_metrics["cancer_logits"],
            "cancer_true": train_metrics["cancer_true"],
        }

        logger.info(
            f"    Train S1={train_metrics['auc_s1']:.3f} "
            f"Val S1={val_metrics['auc_s1']:.3f} S2_F1={val_metrics['f1_macro_s2']:.3f}"
        )

    overall = _compute_overall_metrics(val_binary_prob, val_cancer_logits, bl, ctl, train_last)
    return overall, val_binary_prob, val_cancer_logits


# =============================================================================
# Main
# =============================================================================
def main():
    p = argparse.ArgumentParser(description="Multi-Channel ResNet18 Experiment")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="auto")
    p.add_argument("--aggregate", default="none", choices=["medoid", "mean", "none"])
    p.add_argument("--cancer-types", nargs="+", default=["PRO", "LUN", "CRC", "CPAN", "OVA"])
    p.add_argument("--non-cancer-groups", nargs="+", default=["NOR", "DIA", "HBP", "H.D."])
    args = p.parse_args()

    out_dir = Path("results/training/step3_multichannel")
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
    logger.info("  Step 3: Multi-Channel ResNet18-1D Experiment")
    logger.info("=" * 64)
    logger.info(f"  Device: {device}")
    logger.info(f"  Channels: [SNV spectrum, 1st derivative, 2nd derivative]")

    # Load data
    import yaml
    with open("config/config.yaml", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    df = load_processed_spectra()
    feat_cols = get_feature_columns(df)
    n_feat = len(feat_cols)

    mc = ModelConfig.from_pipeline_config(raw_cfg, n_spectral_features=n_feat)
    mc = apply_class_selection(mc, cancer_types=args.cancer_types, non_cancer_groups=args.non_cancer_groups)
    mc = ModelConfig(**{**mc.__dict__, "learning_rate": args.lr, "batch_size": args.batch_size, "n_epochs": args.epochs})

    logger.info(f"  Cancer types ({mc.n_cancer_types}): {mc.cancer_types}")
    logger.info(f"  Non-cancer: {mc.non_cancer_groups}")

    df = resolve_aliases(df, mc)
    df_agg = aggregate_replicates(df, feat_cols, args.aggregate)
    X, bl, ctl, sample_ids, groups_arr = create_labels(df_agg, mc)

    # Create multi-channel input
    logger.info(f"\n[Multi-Channel] Converting {X.shape} -> (N, 3, {X.shape[1]})")
    X_mc = make_multichannel(X)
    logger.info(f"  Result shape: {X_mc.shape}")

    # Run CV
    logger.info(f"\n[Training] 5-fold CV, {args.epochs} max epochs")
    t0 = datetime.now()
    overall, val_bp, val_cl = run_multichannel_cv(
        X_mc, bl, ctl, sample_ids, groups_arr, mc, device, n_splits=5, epochs=args.epochs,
    )

    elapsed = datetime.now() - t0
    logger.info("\n" + "=" * 64)
    logger.info("  Multi-Channel ResNet18-1D Results")
    logger.info("=" * 64)
    for k, v in overall.items():
        logger.info(f"  {k:28s}: {v:.4f}")
    logger.info(f"\n  Elapsed: {elapsed}")

    # Save
    summary = {
        "experiment": "step3_multichannel",
        "timestamp": datetime.now().isoformat(),
        "channels": ["SNV", "1st_derivative", "2nd_derivative"],
        "aggregate": args.aggregate,
        "n_samples": len(X),
        "n_features": n_feat,
        "cancer_types": list(mc.cancer_types),
        "non_cancer_groups": list(mc.non_cancer_groups),
        "metrics": {k: float(v) for k, v in overall.items()},
        "config": {
            "learning_rate": mc.learning_rate,
            "batch_size": mc.batch_size,
            "epochs": args.epochs,
            "resnet_channels": list(mc.resnet_channels),
            "dropout_rate": mc.dropout_rate,
        },
    }
    with open(out_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    np.savez_compressed(
        out_dir / "fold_predictions.npz",
        val_binary_prob=val_bp,
        val_cancer_logits=val_cl,
        binary_labels=bl,
        cancer_type_labels=ctl,
        groups=groups_arr,
        sample_ids=sample_ids,
    )

    logger.info(f"  Output: {out_dir}/")
    logger.info("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
