#!/usr/bin/env python3
"""
Experiment 3: Contrastive Pretraining Hyperparameter Sweep
==========================================================

Grid search over contrastive pretraining hyperparameters (36 configs),
followed by linear probe + fine-tune evaluation + supervised baseline.

Grid:  temperature × proj_dim × pretrain_epochs = 4 × 3 × 3 = 36

Usage:
    python scripts/analysis/weekend_experiments/contrastive_sweep.py
    python scripts/analysis/weekend_experiments/contrastive_sweep.py --dry-run
    python scripts/analysis/weekend_experiments/contrastive_sweep.py --data medical
"""

from __future__ import annotations

import sys
import os
import json
import argparse
import logging
import warnings
import tempfile
import shutil
import time
from pathlib import Path
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from sers.models.usersnet.stacking import preprocess_channel, load_raw_multichannel
from sers.models._legacy.resnet_v1.model import ModelConfig, SERSCancerDetector, TwoStageLoss, SERSDataset
from sers.models.experimental.contrastive.model import (
    ContrastiveEncoder, NTXentLoss,
    ContrastiveReplicateDataset, PatientBatchSampler,
)
from src.sers.config import RESULTS_DIR, FIG_DIR

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CANCER_TYPES = ("PRO", "LUN", "CRC", "PAN", "OVA", "BRE", "BLC")
NON_CANCER_GROUPS = ("NOR", "DIA", "HBP", "H.D.")

THERMO_MAP = {
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

MEDICAL_MAP = {
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

GRID = {
    "temperature": [0.05, 0.1, 0.2, 0.5],
    "proj_dim": [64, 128, 256],
    "pretrain_epochs": [100, 200, 300],
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Atomic write helper
# ============================================================================
def atomic_write_json(path: Path, data: dict):
    """Write JSON via temp file + rename for crash safety."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        shutil.move(tmp, str(path))
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def atomic_save_torch(path: Path, obj):
    """Save torch state_dict via temp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        torch.save(obj, tmp)
        shutil.move(tmp, str(path))
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


# ============================================================================
# Checkpoint naming
# ============================================================================
def _cfg_tag(temp: float, proj: int, epochs: int) -> str:
    return f"t{temp}_p{proj}_e{epochs}"


def pretrain_ckpt_path(ckpt_dir: Path, temp: float, proj: int, epochs: int) -> Path:
    return ckpt_dir / f"pretrain_{_cfg_tag(temp, proj, epochs)}.pt"


def eval_ckpt_path(ckpt_dir: Path, temp: float, proj: int, epochs: int) -> Path:
    return ckpt_dir / f"eval_{_cfg_tag(temp, proj, epochs)}.json"


# ============================================================================
# Pretraining
# ============================================================================
def pretrain_encoder(
    X_3ch: np.ndarray,
    df_meta: pd.DataFrame,
    n_epochs: int = 300,
    batch_size: int = 128,
    lr: float = 1e-3,
    temperature: float = 0.1,
    proj_dim: int = 128,
    device: str = "cuda",
) -> tuple:
    """Pretrain encoder with NT-Xent contrastive loss."""
    groups = df_meta["group"].values
    sample_ids = df_meta["sample_id"].values

    dataset = ContrastiveReplicateDataset(X_3ch, groups, sample_ids, augment=True)
    sampler = PatientBatchSampler(len(dataset), batch_size)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0, pin_memory=True)

    n_ch = X_3ch.shape[1]
    config = ModelConfig(
        n_spectral_features=X_3ch.shape[2],
        input_channels=n_ch,
        resnet_channels=(32, 64, 128, 256),
        encoder_output_dim=256,
    )
    model = ContrastiveEncoder(config, proj_dim=proj_dim).to(device)
    criterion = NTXentLoss(temperature)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    losses = []
    for epoch in range(n_epochs):
        model.train()
        epoch_loss, n_batches = 0, 0
        for batch in loader:
            anchor = batch["anchor"].to(device)
            positive = batch["positive"].to(device)
            _, z_i = model(anchor)
            _, z_j = model(positive)
            loss = criterion(z_i, z_j)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)
        if (epoch + 1) % 50 == 0:
            logger.info(f"    Pretrain epoch {epoch+1}/{n_epochs}: loss={avg_loss:.4f}")

    return model, losses


# ============================================================================
# Evaluation helpers
# ============================================================================
def extract_embeddings(
    model: ContrastiveEncoder,
    X: np.ndarray,
    device: str,
    batch_size: int = 256,
) -> np.ndarray:
    """Extract encoder representations (no projection head)."""
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            x = torch.FloatTensor(X[i : i + batch_size]).to(device)
            h = model.encode(x)
            embeddings.append(h.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def evaluate_linear_probe(
    embeddings: np.ndarray,
    binary_labels: np.ndarray,
    cancer_type_labels: np.ndarray,
    sample_ids: np.ndarray,
    n_splits: int = 5,
) -> dict:
    """Linear probe: LR on frozen encoder embeddings (5-fold CV)."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    s1_probs = np.full(len(embeddings), np.nan)
    s2_probs = np.full((len(embeddings), len(CANCER_TYPES)), np.nan)

    for fold, (tr_idx, val_idx) in enumerate(
        sgkf.split(embeddings, binary_labels, sample_ids)
    ):
        # Stage 1: binary
        s1 = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0, max_iter=2000, solver="saga", class_weight="balanced"
            ),
        )
        s1.fit(embeddings[tr_idx], binary_labels[tr_idx])
        s1_probs[val_idx] = s1.predict_proba(embeddings[val_idx])[:, 1]

        # Stage 2: cancer type
        cancer_mask = binary_labels[tr_idx] == 1
        if cancer_mask.sum() > 10:
            s2 = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=1.0,
                    max_iter=2000,
                    solver="saga",
                    class_weight="balanced",
                    multi_class="multinomial",
                ),
            )
            s2.fit(
                embeddings[tr_idx][cancer_mask],
                cancer_type_labels[tr_idx][cancer_mask],
            )
            raw = s2.predict_proba(embeddings[val_idx])
            for i, cls in enumerate(s2.classes_):
                if cls < len(CANCER_TYPES):
                    s2_probs[val_idx, cls] = raw[:, i]

    auc = roc_auc_score(binary_labels, s1_probs)
    bm = binary_labels == 1
    f1t = f1_score(
        cancer_type_labels[bm],
        s2_probs[bm].argmax(axis=1),
        average="macro",
        zero_division=0,
    )
    return {"auc": float(auc), "f1_type": float(f1t)}


def evaluate_finetune(
    pretrained_model: ContrastiveEncoder,
    X_3ch: np.ndarray,
    binary_labels: np.ndarray,
    cancer_type_labels: np.ndarray,
    sample_ids: np.ndarray,
    device: str,
    n_splits: int = 5,
    n_epochs: int = 100,
) -> dict:
    """Fine-tune pretrained encoder with differential LR and early stopping."""
    n_ch = X_3ch.shape[1]
    n_feat = X_3ch.shape[2]

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    s1_probs = np.full(len(X_3ch), np.nan)
    s2_probs = np.full((len(X_3ch), len(CANCER_TYPES)), np.nan)

    for fold, (tr_idx, val_idx) in enumerate(
        sgkf.split(X_3ch, binary_labels, sample_ids)
    ):
        config = ModelConfig(
            n_spectral_features=n_feat,
            input_channels=n_ch,
            cancer_types=CANCER_TYPES,
            non_cancer_groups=NON_CANCER_GROUPS,
            resnet_channels=(32, 64, 128, 256),
            encoder_output_dim=256,
            head_hidden_dim=128,
            dropout_rate=0.5,
            learning_rate=1e-4,
            weight_decay=1e-3,
        )
        model = SERSCancerDetector(config).to(device)

        # Load pretrained encoder weights
        pretrained_dict = {
            k.replace("encoder.", ""): v
            for k, v in pretrained_model.encoder.state_dict().items()
        }
        model.encoder.load_state_dict(pretrained_dict)

        criterion = TwoStageLoss(use_focal_loss=True, focal_gamma=2.0)

        # Differential LR: encoder slower, heads faster
        encoder_params = list(model.encoder.parameters())
        head_params = list(model.binary_head.parameters()) + list(
            model.cancer_type_head.parameters()
        )
        optimizer = torch.optim.AdamW(
            [
                {"params": encoder_params, "lr": 1e-4},
                {"params": head_params, "lr": 5e-4},
            ],
            weight_decay=1e-3,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs
        )

        train_ds = SERSDataset(
            X_3ch[tr_idx],
            binary_labels[tr_idx],
            cancer_type_labels[tr_idx],
            augment=True,
        )
        val_ds = SERSDataset(
            X_3ch[val_idx],
            binary_labels[val_idx],
            cancer_type_labels[val_idx],
            augment=False,
        )

        class_counts = np.bincount(
            binary_labels[tr_idx].astype(int), minlength=2
        ).clip(1)
        weights = 1.0 / class_counts[binary_labels[tr_idx].astype(int)]
        wrs = WeightedRandomSampler(weights, len(weights), replacement=True)

        train_dl = DataLoader(
            train_ds, batch_size=32, sampler=wrs, num_workers=0, pin_memory=True
        )
        val_dl = DataLoader(
            val_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True
        )

        best_loss, best_state, patience = float("inf"), None, 0
        for epoch in range(n_epochs):
            model.train()
            for batch in train_dl:
                out = model(batch["spectra"].to(device))
                losses = criterion(
                    out,
                    batch["binary_label"].to(device),
                    batch["cancer_type_label"].to(device),
                )
                optimizer.zero_grad()
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            # Validation check every 5 epochs
            if (epoch + 1) % 5 == 0:
                model.eval()
                vl = []
                with torch.no_grad():
                    for batch in val_dl:
                        out = model(batch["spectra"].to(device))
                        l = criterion(
                            out,
                            batch["binary_label"].to(device),
                            batch["cancer_type_label"].to(device),
                        )
                        vl.append(l["total"].item())
                mean_vl = np.mean(vl)
                if mean_vl < best_loss - 1e-4:
                    best_loss = mean_vl
                    best_state = {
                        k: v.cpu().clone() for k, v in model.state_dict().items()
                    }
                    patience = 0
                else:
                    patience += 1
                    if patience >= 8:
                        break

        if best_state:
            model.load_state_dict(best_state)
        model.eval()

        with torch.no_grad():
            all_s1, all_s2 = [], []
            for batch in val_dl:
                out = model(batch["spectra"].to(device))
                all_s1.append(out["binary_prob"].cpu().numpy().squeeze(-1))
                all_s2.append(out["cancer_probs"].cpu().numpy())
            s1_probs[val_idx] = np.concatenate(all_s1)
            s2_probs[val_idx] = np.concatenate(all_s2)

    auc = roc_auc_score(binary_labels, s1_probs)
    bm = binary_labels == 1
    f1t = f1_score(
        cancer_type_labels[bm],
        s2_probs[bm].argmax(axis=1),
        average="macro",
        zero_division=0,
    )
    return {"auc": float(auc), "f1_type": float(f1t)}


def evaluate_supervised_baseline(
    X_3ch: np.ndarray,
    binary_labels: np.ndarray,
    cancer_type_labels: np.ndarray,
    sample_ids: np.ndarray,
    device: str,
    n_epochs: int = 100,
    n_splits: int = 5,
) -> dict:
    """Supervised baseline: SERSCancerDetector from random init (no pretraining)."""
    n_ch = X_3ch.shape[1]
    n_feat = X_3ch.shape[2]

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    s1_probs = np.full(len(X_3ch), np.nan)
    s2_probs = np.full((len(X_3ch), len(CANCER_TYPES)), np.nan)

    for fold, (tr_idx, val_idx) in enumerate(
        sgkf.split(X_3ch, binary_labels, sample_ids)
    ):
        config = ModelConfig(
            n_spectral_features=n_feat,
            input_channels=n_ch,
            cancer_types=CANCER_TYPES,
            non_cancer_groups=NON_CANCER_GROUPS,
            resnet_channels=(32, 64, 128, 256),
            encoder_output_dim=256,
            head_hidden_dim=128,
            dropout_rate=0.5,
            learning_rate=5e-4,
            weight_decay=1e-3,
        )
        model = SERSCancerDetector(config).to(device)
        criterion = TwoStageLoss(use_focal_loss=True, focal_gamma=2.0)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=5e-4, weight_decay=1e-3
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs
        )

        train_ds = SERSDataset(
            X_3ch[tr_idx],
            binary_labels[tr_idx],
            cancer_type_labels[tr_idx],
            augment=True,
        )
        val_ds = SERSDataset(
            X_3ch[val_idx],
            binary_labels[val_idx],
            cancer_type_labels[val_idx],
            augment=False,
        )

        class_counts = np.bincount(
            binary_labels[tr_idx].astype(int), minlength=2
        ).clip(1)
        weights = 1.0 / class_counts[binary_labels[tr_idx].astype(int)]
        wrs = WeightedRandomSampler(weights, len(weights), replacement=True)

        train_dl = DataLoader(
            train_ds, batch_size=32, sampler=wrs, num_workers=0, pin_memory=True
        )
        val_dl = DataLoader(
            val_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True
        )

        best_loss, best_state, patience = float("inf"), None, 0
        for epoch in range(n_epochs):
            model.train()
            for batch in train_dl:
                out = model(batch["spectra"].to(device))
                losses = criterion(
                    out,
                    batch["binary_label"].to(device),
                    batch["cancer_type_label"].to(device),
                )
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
                        l = criterion(
                            out,
                            batch["binary_label"].to(device),
                            batch["cancer_type_label"].to(device),
                        )
                        vl.append(l["total"].item())
                mean_vl = np.mean(vl)
                if mean_vl < best_loss - 1e-4:
                    best_loss = mean_vl
                    best_state = {
                        k: v.cpu().clone() for k, v in model.state_dict().items()
                    }
                    patience = 0
                else:
                    patience += 1
                    if patience >= 8:
                        break

        if best_state:
            model.load_state_dict(best_state)
        model.eval()

        with torch.no_grad():
            all_s1, all_s2 = [], []
            for batch in val_dl:
                out = model(batch["spectra"].to(device))
                all_s1.append(out["binary_prob"].cpu().numpy().squeeze(-1))
                all_s2.append(out["cancer_probs"].cpu().numpy())
            s1_probs[val_idx] = np.concatenate(all_s1)
            s2_probs[val_idx] = np.concatenate(all_s2)

        logger.info(
            f"  Baseline fold {fold}: "
            f"AUC={roc_auc_score(binary_labels[val_idx], s1_probs[val_idx]):.4f}"
        )

    auc = roc_auc_score(binary_labels, s1_probs)
    bm = binary_labels == 1
    f1t = f1_score(
        cancer_type_labels[bm],
        s2_probs[bm].argmax(axis=1),
        average="macro",
        zero_division=0,
    )
    return {"auc": float(auc), "f1_type": float(f1t)}


# ============================================================================
# Plotting
# ============================================================================
def _fig_style():
    """Consistent figure style."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def plot_heatmap_temp_x_proj(df: pd.DataFrame, metric: str, out_dir: Path):
    """Heatmap: temperature (y) x proj_dim (x), best across epochs."""
    _fig_style()
    pivot = df.groupby(["temperature", "proj_dim"])[metric].max().reset_index()
    matrix = pivot.pivot(index="temperature", columns="proj_dim", values=metric)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        matrix, annot=True, fmt=".4f", cmap="YlOrRd", ax=ax,
        linewidths=0.5, linecolor="white",
    )
    ax.set_title(f"Best {metric} across epochs\n(temperature x proj_dim)", fontsize=13)
    ax.set_xlabel("Projection Dimension")
    ax.set_ylabel("Temperature")
    fig.tight_layout()
    fig.savefig(out_dir / "heatmap_temp_x_proj.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap_epochs_x_temp(df: pd.DataFrame, metric: str, out_dir: Path):
    """Heatmap: pretrain_epochs (y) x temperature (x), best across proj_dim."""
    _fig_style()
    pivot = df.groupby(["pretrain_epochs", "temperature"])[metric].max().reset_index()
    matrix = pivot.pivot(index="pretrain_epochs", columns="temperature", values=metric)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        matrix, annot=True, fmt=".4f", cmap="YlOrRd", ax=ax,
        linewidths=0.5, linecolor="white",
    )
    ax.set_title(f"Best {metric} across proj_dim\n(pretrain_epochs x temperature)", fontsize=13)
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Pretrain Epochs")
    fig.tight_layout()
    fig.savefig(out_dir / "heatmap_epochs_x_temp.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(all_losses: dict, df: pd.DataFrame, metric: str, out_dir: Path):
    """Loss curves for top-5 configs by metric."""
    _fig_style()
    top5 = df.nlargest(5, metric)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, 5))
    for idx, (_, row) in enumerate(top5.iterrows()):
        tag = _cfg_tag(row["temperature"], int(row["proj_dim"]), int(row["pretrain_epochs"]))
        if tag in all_losses:
            losses = all_losses[tag]
            ax.plot(
                range(1, len(losses) + 1), losses,
                color=colors[idx], linewidth=1.5,
                label=f"t={row['temperature']}, p={int(row['proj_dim'])}, "
                      f"e={int(row['pretrain_epochs'])} ({metric}={row[metric]:.4f})",
            )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NT-Xent Loss")
    ax.set_title(f"Pretraining Loss Curves (Top 5 by {metric})")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_probe_vs_finetune(df: pd.DataFrame, out_dir: Path):
    """Scatter: probe AUC vs fine-tune AUC, colored by temperature."""
    _fig_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # AUC
    ax = axes[0]
    for temp, grp in df.groupby("temperature"):
        ax.scatter(
            grp["probe_auc"], grp["finetune_auc"],
            s=60, alpha=0.8, label=f"temp={temp}",
        )
    mn = min(df["probe_auc"].min(), df["finetune_auc"].min()) - 0.01
    mx = max(df["probe_auc"].max(), df["finetune_auc"].max()) + 0.01
    ax.plot([mn, mx], [mn, mx], "k--", alpha=0.4, linewidth=1)
    ax.set_xlabel("Linear Probe AUC")
    ax.set_ylabel("Fine-tune AUC")
    ax.set_title("Probe vs Fine-tune: AUC")
    ax.legend(fontsize=9)

    # F1 type
    ax = axes[1]
    for temp, grp in df.groupby("temperature"):
        ax.scatter(
            grp["probe_f1_type"], grp["finetune_f1_type"],
            s=60, alpha=0.8, label=f"temp={temp}",
        )
    mn = min(df["probe_f1_type"].min(), df["finetune_f1_type"].min()) - 0.01
    mx = max(df["probe_f1_type"].max(), df["finetune_f1_type"].max()) + 0.01
    ax.plot([mn, mx], [mn, mx], "k--", alpha=0.4, linewidth=1)
    ax.set_xlabel("Linear Probe F1-type")
    ax.set_ylabel("Fine-tune F1-type")
    ax.set_title("Probe vs Fine-tune: Macro F1 (cancer type)")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_dir / "probe_vs_finetune.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Experiment 3: Contrastive Pretraining HP Sweep"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Quick test: 1 grid point, pretrain 5 epochs, finetune 5 epochs",
    )
    parser.add_argument(
        "--data", "-d", default="thermo", choices=["thermo", "medical"],
        help="Dataset to use (default: thermo)",
    )
    args = parser.parse_args()

    # Output directory
    out_dir = RESULTS_DIR / "contrastive_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_out_dir = FIG_DIR / "weekend" / "contrastive_sweep"
    fig_out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # File logging
    fh = logging.FileHandler(out_dir / "experiment.log", mode="a")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logging.getLogger().addHandler(fh)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    logger.info("=" * 70)
    logger.info("  Experiment 3: Contrastive Pretraining HP Sweep")
    logger.info("=" * 70)
    logger.info(f"  Device: {device}")
    logger.info(f"  Data:   {args.data}")
    logger.info(f"  Dry-run: {args.dry_run}")
    logger.info(f"  Output: {out_dir}")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    grid = np.linspace(402.0, 2198.0, 933)
    if args.data == "thermo":
        data_dir = PROJECT_ROOT / "data" / "raw_data"
        folder_map = THERMO_MAP
        pattern = "*.CSV"
        max_rep = None
    else:
        data_dir = PROJECT_ROOT / "data" / "raw_data_medical"
        folder_map = MEDICAL_MAP
        pattern = "*.txt"
        max_rep = {"NOR": 6}

    logger.info(f"\n  Loading RAW spectra ({args.data})...")
    X_3ch, df_meta = load_raw_multichannel(data_dir, folder_map, grid, pattern, max_rep)
    # Disambiguate sample_ids before group merge (CPAN_4 ≠ YPAN_4)
    needs_prefix = df_meta["group"].isin(["YPAN", "YNOR"])
    df_meta.loc[needs_prefix, "sample_id"] = (
        df_meta.loc[needs_prefix, "group"] + "_" + df_meta.loc[needs_prefix, "sample_id"].astype(str)
    )
    df_meta["group"] = df_meta["group"].replace({"CPAN": "PAN", "YPAN": "PAN", "YNOR": "NOR"})

    valid = set(CANCER_TYPES) | set(NON_CANCER_GROUPS)
    mask = df_meta["group"].isin(valid).values
    X_3ch = X_3ch[mask]
    df_meta = df_meta[mask].reset_index(drop=True)

    groups_arr = df_meta["group"].values
    sample_ids = (df_meta["group"] + "_" + df_meta["sample_id"].astype(str)).values
    binary_labels = np.array([1.0 if g in CANCER_TYPES else 0.0 for g in groups_arr])
    ct_map = {ct: i for i, ct in enumerate(CANCER_TYPES)}
    cancer_type_labels = np.array([ct_map.get(g, -1) for g in groups_arr])

    logger.info(
        f"  {len(X_3ch)} spectra, {int(binary_labels.sum())} cancer, "
        f"shape {X_3ch.shape}"
    )

    # ------------------------------------------------------------------
    # 2. Build grid
    # ------------------------------------------------------------------
    if args.dry_run:
        configs = [
            {
                "temperature": GRID["temperature"][0],
                "proj_dim": GRID["proj_dim"][0],
                "pretrain_epochs": 5,
            }
        ]
        finetune_epochs = 5
        logger.info(f"\n  DRY RUN: 1 config, pretrain=5, finetune=5")
    else:
        configs = [
            {"temperature": t, "proj_dim": p, "pretrain_epochs": e}
            for t, p, e in product(
                GRID["temperature"], GRID["proj_dim"], GRID["pretrain_epochs"]
            )
        ]
        finetune_epochs = 100
        logger.info(f"\n  Full sweep: {len(configs)} configs")

    # Save config
    atomic_write_json(out_dir / "config.json", {
        "grid": {k: [float(v) if isinstance(v, float) else v for v in vals]
                 for k, vals in GRID.items()},
        "n_configs": len(configs),
        "finetune_epochs": finetune_epochs,
        "data": args.data,
        "dry_run": args.dry_run,
        "device": str(device),
        "n_spectra": len(X_3ch),
        "n_cancer": int(binary_labels.sum()),
        "started": datetime.now().isoformat(),
    })

    # ------------------------------------------------------------------
    # 3. Sweep
    # ------------------------------------------------------------------
    results = []
    all_losses = {}  # tag -> loss list
    n_total = len(configs)

    for idx, cfg in enumerate(configs):
        temp = cfg["temperature"]
        proj = cfg["proj_dim"]
        epochs = cfg["pretrain_epochs"]
        tag = _cfg_tag(temp, proj, epochs)

        elapsed = time.time() - t0
        if idx > 0:
            eta = elapsed / idx * (n_total - idx)
            eta_str = f"{eta / 60:.1f}m"
        else:
            eta_str = "N/A"

        logger.info(
            f"\n{'='*60}"
            f"\n  [{idx+1}/{n_total}] temp={temp}, proj={proj}, epochs={epochs}"
            f"\n  Elapsed: {elapsed/60:.1f}m | ETA: {eta_str}"
            f"\n{'='*60}"
        )

        eval_path = eval_ckpt_path(ckpt_dir, temp, proj, epochs)
        pt_path = pretrain_ckpt_path(ckpt_dir, temp, proj, epochs)

        # --- Check eval checkpoint (skip entirely if done) ---
        if eval_path.exists():
            logger.info(f"  Eval checkpoint found, loading...")
            with open(eval_path) as f:
                cached = json.load(f)
            results.append(cached)
            # Try to load losses for plotting
            if pt_path.exists():
                try:
                    pt_data = torch.load(pt_path, map_location="cpu", weights_only=False)
                    if "losses" in pt_data:
                        all_losses[tag] = pt_data["losses"]
                except Exception:
                    pass
            continue

        # --- Pretrain (or load checkpoint) ---
        if pt_path.exists():
            logger.info(f"  Pretrain checkpoint found, loading encoder...")
            pt_data = torch.load(pt_path, map_location="cpu", weights_only=False)
            n_ch = X_3ch.shape[1]
            mc = ModelConfig(
                n_spectral_features=X_3ch.shape[2],
                input_channels=n_ch,
                resnet_channels=(32, 64, 128, 256),
                encoder_output_dim=256,
            )
            pretrained_model = ContrastiveEncoder(mc, proj_dim=proj).to(device)
            pretrained_model.load_state_dict(pt_data["state_dict"])
            losses = pt_data.get("losses", [])
        else:
            logger.info(f"  Pretraining...")
            pretrained_model, losses = pretrain_encoder(
                X_3ch, df_meta,
                n_epochs=epochs,
                temperature=temp,
                proj_dim=proj,
                device=str(device),
            )
            # Save pretrain checkpoint atomically
            atomic_save_torch(pt_path, {
                "state_dict": pretrained_model.state_dict(),
                "losses": losses,
                "config": cfg,
            })
            logger.info(f"  Pretrain saved: {pt_path.name}")

        all_losses[tag] = losses

        # --- Linear probe ---
        logger.info(f"  Evaluating linear probe...")
        embeddings = extract_embeddings(pretrained_model, X_3ch, str(device))
        probe_metrics = evaluate_linear_probe(
            embeddings, binary_labels, cancer_type_labels, sample_ids
        )
        logger.info(
            f"  Probe: AUC={probe_metrics['auc']:.4f}, "
            f"F1-type={probe_metrics['f1_type']:.4f}"
        )

        # --- Fine-tune ---
        logger.info(f"  Evaluating fine-tune ({finetune_epochs} epochs)...")
        ft_metrics = evaluate_finetune(
            pretrained_model, X_3ch, binary_labels, cancer_type_labels,
            sample_ids, str(device), n_epochs=finetune_epochs,
        )
        logger.info(
            f"  Fine-tune: AUC={ft_metrics['auc']:.4f}, "
            f"F1-type={ft_metrics['f1_type']:.4f}"
        )

        # --- Save eval checkpoint atomically ---
        row = {
            "temperature": temp,
            "proj_dim": proj,
            "pretrain_epochs": epochs,
            "probe_auc": probe_metrics["auc"],
            "probe_f1_type": probe_metrics["f1_type"],
            "finetune_auc": ft_metrics["auc"],
            "finetune_f1_type": ft_metrics["f1_type"],
        }
        atomic_write_json(eval_path, row)
        results.append(row)

    # ------------------------------------------------------------------
    # 4. Supervised baseline (random init)
    # ------------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info(f"  Supervised Baseline (random init, no pretraining)")
    logger.info(f"{'='*60}")

    baseline_path = ckpt_dir / "eval_supervised_baseline.json"
    if baseline_path.exists():
        logger.info("  Baseline checkpoint found, loading...")
        with open(baseline_path) as f:
            baseline_metrics = json.load(f)
    else:
        baseline_metrics = evaluate_supervised_baseline(
            X_3ch, binary_labels, cancer_type_labels, sample_ids,
            str(device), n_epochs=finetune_epochs,
        )
        atomic_write_json(baseline_path, baseline_metrics)

    logger.info(
        f"  Baseline: AUC={baseline_metrics['auc']:.4f}, "
        f"F1-type={baseline_metrics['f1_type']:.4f}"
    )

    # ------------------------------------------------------------------
    # 5. Compile results
    # ------------------------------------------------------------------
    df = pd.DataFrame(results)
    df.to_csv(out_dir / "sweep_results.csv", index=False)
    logger.info(f"\n  Results saved: sweep_results.csv ({len(df)} rows)")

    # Best config (by finetune AUC)
    if len(df) > 0:
        best_idx = df["finetune_auc"].idxmax()
        best_row = df.iloc[best_idx].to_dict()
    else:
        best_row = {}

    best_config = {
        "best_contrastive": best_row,
        "supervised_baseline": baseline_metrics,
        "improvement": {
            "auc_over_baseline": (
                best_row.get("finetune_auc", 0) - baseline_metrics["auc"]
            ),
            "f1_over_baseline": (
                best_row.get("finetune_f1_type", 0) - baseline_metrics["f1_type"]
            ),
        },
        "total_configs": len(df),
        "total_time_min": (time.time() - t0) / 60,
        "timestamp": datetime.now().isoformat(),
    }
    atomic_write_json(out_dir / "best_config.json", best_config)

    # ------------------------------------------------------------------
    # 6. Generate plots
    # ------------------------------------------------------------------
    if len(df) >= 2:
        logger.info("\n  Generating plots...")

        # Pick the primary metric for heatmaps
        primary_metric = "finetune_auc"

        plot_heatmap_temp_x_proj(df, primary_metric, fig_out_dir)
        logger.info("    heatmap_temp_x_proj.png")

        plot_heatmap_epochs_x_temp(df, primary_metric, fig_out_dir)
        logger.info("    heatmap_epochs_x_temp.png")

        plot_training_curves(all_losses, df, primary_metric, fig_out_dir)
        logger.info("    training_curves.png")

        plot_probe_vs_finetune(df, fig_out_dir)
        logger.info("    probe_vs_finetune.png")
    else:
        logger.info("  Skipping plots (not enough data points)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_min = (time.time() - t0) / 60
    logger.info(f"\n{'='*70}")
    logger.info(f"  SWEEP COMPLETE — {len(df)} configs in {total_min:.1f} min")
    logger.info(f"{'='*70}")
    logger.info(f"  {'Method':<35} {'AUC':>8} {'F1-type':>8}")
    logger.info(f"  {'-'*35} {'-'*8} {'-'*8}")

    if best_row:
        label = (
            f"Best contrastive (t={best_row['temperature']}, "
            f"p={int(best_row['proj_dim'])}, e={int(best_row['pretrain_epochs'])})"
        )
        logger.info(
            f"  {label:<35} {best_row['finetune_auc']:>8.4f} "
            f"{best_row['finetune_f1_type']:>8.4f}"
        )
        logger.info(
            f"  {'  -> probe':<35} {best_row['probe_auc']:>8.4f} "
            f"{best_row['probe_f1_type']:>8.4f}"
        )

    logger.info(
        f"  {'Supervised baseline':<35} {baseline_metrics['auc']:>8.4f} "
        f"{baseline_metrics['f1_type']:>8.4f}"
    )

    if best_row:
        delta_auc = best_row["finetune_auc"] - baseline_metrics["auc"]
        delta_f1 = best_row["finetune_f1_type"] - baseline_metrics["f1_type"]
        sign_a = "+" if delta_auc >= 0 else ""
        sign_f = "+" if delta_f1 >= 0 else ""
        logger.info(
            f"  {'Delta (best - baseline)':<35} {sign_a}{delta_auc:>7.4f} "
            f"{sign_f}{delta_f1:>7.4f}"
        )

    logger.info(f"\n  Output: {out_dir}")
    logger.info(f"{'='*70}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
