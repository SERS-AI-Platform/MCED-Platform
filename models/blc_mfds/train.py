"""
BLC MFDS — Bladder Cancer Single-Cancer Diagnostic Module

Binary classification: BLC vs Non-cancer (NOR, DIA, HBP, H.D.)
Full clinical features (26 variables with >95% completeness)
Experiments: SERS-only baseline, FiLM fusion, Cross-Attention fusion

Usage:
    python models/blc_mfds/train.py
    python models/blc_mfds/train.py --model film --epochs 200
    python models/blc_mfds/train.py --model xattn --n-attn-layers 1
"""

from __future__ import annotations

import sys
import argparse
import logging
import json
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve,
)

import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.model import ModelConfig, BasicBlock1D, BinaryHead
from models.clinical_utils import (
    load_clinical, build_id_mappings, compute_multichannel_spectra, TIER1_COLS,
)
from models.film.model import FiLMGenerator
from models.cross_attention.model import (
    ClinicalTokenizer, CrossAttentionBlock,
)
from models.train import (
    load_processed_spectra, get_feature_columns,
    resolve_aliases, aggregate_replicates,
    EarlyStopping, _build_loader,
    resolve_runtime_config, configure_torch_runtime,
)

logger = logging.getLogger(__name__)

# BLC clinical features (>95% completeness in both BLC and NC)
BLC_CLINICAL_COLS = [
    "age", "sex_numeric", "bmi",
    "smoking_status_numeric", "drinking_status_numeric",
    "bp_systolic", "bp_diastolic",
    "wbc", "rbc", "hb", "hct", "platelet",
    "neutrophil_pct", "lymphocyte_pct",
    "ast", "alt", "alp", "total_bilirubin",
    "bun", "creatinine", "uric_acid",
    "glucose", "calcium", "total_cholesterol",
    "ua_sg", "ua_ph",
]
N_CLINICAL = len(BLC_CLINICAL_COLS)  # 26


# =========================================================================
# Data Loading
# =========================================================================
def load_blc_data(aggregate="medoid"):
    """Load BLC + Non-cancer data with full clinical features."""
    import yaml

    df = load_processed_spectra()
    feat_cols = get_feature_columns(df)

    try:
        with open(PROJECT_ROOT / "config" / "config.yaml") as f:
            raw_cfg = yaml.safe_load(f)
        mc = ModelConfig.from_pipeline_config(raw_cfg, n_spectral_features=len(feat_cols))
    except FileNotFoundError:
        mc = ModelConfig(n_spectral_features=len(feat_cols))

    df = resolve_aliases(df, mc)

    # Filter to BLC + Non-cancer only
    target_groups = {"BLC", "NOR", "DIA", "HBP", "H.D."}
    df = df[df["group"].isin(target_groups)].reset_index(drop=True)
    logger.info(f"  BLC + NC filter: {len(df)} spectra")
    logger.info(f"  Groups: {df.groupby('group')['sample_id'].nunique().to_dict()}")

    # Aggregate
    df_agg = aggregate_replicates(df, feat_cols, aggregate)

    # Binary labels: BLC=1, Non-cancer=0
    binary_labels = (df_agg["group"] == "BLC").astype(int).values
    sample_ids = df_agg["sample_id"].values
    groups_arr = df_agg["group"].values

    X = df_agg[feat_cols].values.astype(np.float32)

    logger.info(f"  Binary: {binary_labels.sum()} BLC + {(1-binary_labels).sum()} Non-cancer")

    # Clinical features
    clin = load_clinical()
    # Add numeric columns for smoking/drinking
    clin["smoking_status_numeric"] = clin["smoking_status"].map(
        {"비흡연": 0, "흡연": 1, "과거흡연": 0.5, "현재흡연": 1}
    ).fillna(0).astype(float)
    clin["drinking_status_numeric"] = clin["drinking_status"].map(
        {"비음주": 0, "음주": 1, "과거음주": 0.5, "현재음주": 1}
    ).fillna(0).astype(float)

    # Build ID mappings for BLC (BLA)
    id_mappings = build_id_mappings(PROJECT_ROOT)

    # Merge clinical
    n = len(df_agg)
    clin_lookup = {}
    for _, row in clin.iterrows():
        pid = str(row["patient_id"])
        clin_lookup[pid] = {c: row.get(c, np.nan) for c in BLC_CLINICAL_COLS}

    clinical = np.full((n, N_CLINICAL), np.nan)
    matched = 0
    for i, (_, row) in enumerate(df_agg.iterrows()):
        key = f"{row['group']} {row['sample_id']}"

        # Direct match
        if key in clin_lookup:
            for j, c in enumerate(BLC_CLINICAL_COLS):
                clinical[i, j] = clin_lookup[key][c]
            matched += 1
            continue

        # ID mapping
        if key in id_mappings:
            mapped = id_mappings[key]
            if mapped in clin_lookup:
                for j, c in enumerate(BLC_CLINICAL_COLS):
                    clinical[i, j] = clin_lookup[mapped][c]
                matched += 1

    logger.info(f"  Clinical merge: {matched}/{n} matched ({matched/n*100:.1f}%)")

    # Impute missing with per-column median
    for col_idx in range(N_CLINICAL):
        col = clinical[:, col_idx]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            valid = col[~nan_mask]
            fill_val = np.median(valid) if len(valid) > 0 else 0.0
            clinical[nan_mask, col_idx] = fill_val
            n_fill = nan_mask.sum()
            if n_fill > 5:
                logger.info(f"    {BLC_CLINICAL_COLS[col_idx]}: {n_fill} imputed")

    clinical = clinical.astype(np.float32)

    # Multi-channel spectra
    X_3ch = compute_multichannel_spectra(X)

    return X_3ch, binary_labels, clinical, sample_ids, groups_arr


# =========================================================================
# Dataset
# =========================================================================
class BLCDataset(torch.utils.data.Dataset):
    def __init__(self, spectra, labels, clinical, augment=False):
        self.spectra = torch.FloatTensor(spectra)
        self.labels = torch.FloatTensor(labels).unsqueeze(-1)
        self.clinical = torch.FloatTensor(clinical)
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
        return {"spectra": spec, "label": self.labels[idx], "clinical": self.clinical[idx]}


# =========================================================================
# Models
# =========================================================================
class BLCResNetEncoder(nn.Module):
    """3-channel ResNet18-1D encoder for BLC."""
    def __init__(self, n_input_channels=3, channels=(32, 64, 128, 256)):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input_channels, channels[0], kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(channels[0], channels[0], 2, stride=1)
        self.layer2 = self._make_layer(channels[0], channels[1], 2, stride=2)
        self.layer3 = self._make_layer(channels[1], channels[2], 2, stride=2)
        self.layer4 = self._make_layer(channels[2], channels[3], 2, stride=2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_dim = channels[-1]
        self._init_weights()

    def _make_layer(self, in_ch, out_ch, n_blocks, stride=1):
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
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
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        return self.global_pool(x).squeeze(-1)


class BLCBaselineModel(nn.Module):
    """SERS-only baseline (no clinical)."""
    def __init__(self):
        super().__init__()
        self.encoder = BLCResNetEncoder()
        self.head = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(256, 64), nn.ReLU(),
            nn.Dropout(0.25), nn.Linear(64, 1),
        )

    def forward(self, spectra, clinical=None):
        emb = self.encoder(spectra)
        return {"logit": self.head(emb), "embedding": emb}


class BLCFiLMModel(nn.Module):
    """FiLM fusion: clinical conditions ResNet feature maps."""
    def __init__(self, n_clinical=N_CLINICAL, film_hidden=64):
        super().__init__()
        channels = (32, 64, 128, 256)
        self.conv1 = nn.Conv1d(3, channels[0], 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(3, stride=2, padding=1)

        self.layer1 = self._make_layer(channels[0], channels[0], 2, 1)
        self.layer2 = self._make_layer(channels[0], channels[1], 2, 2)
        self.layer3 = self._make_layer(channels[1], channels[2], 2, 2)
        self.layer4 = self._make_layer(channels[2], channels[3], 2, 2)

        self.film1 = FiLMGenerator(n_clinical, channels[0], film_hidden)
        self.film2 = FiLMGenerator(n_clinical, channels[1], film_hidden)
        self.film3 = FiLMGenerator(n_clinical, channels[2], film_hidden)
        self.film4 = FiLMGenerator(n_clinical, channels[3], film_hidden)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(256, 64), nn.ReLU(),
            nn.Dropout(0.25), nn.Linear(64, 1),
        )
        self._init_weights()

    def _make_layer(self, in_ch, out_ch, n_blocks, stride):
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
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
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, spectra, clinical):
        x = spectra if spectra.dim() == 3 else spectra.unsqueeze(1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x); g, b = self.film1(clinical); x = x * g.unsqueeze(-1) + b.unsqueeze(-1)
        x = self.layer2(x); g, b = self.film2(clinical); x = x * g.unsqueeze(-1) + b.unsqueeze(-1)
        x = self.layer3(x); g, b = self.film3(clinical); x = x * g.unsqueeze(-1) + b.unsqueeze(-1)
        x = self.layer4(x); g, b = self.film4(clinical); x = x * g.unsqueeze(-1) + b.unsqueeze(-1)

        emb = self.global_pool(x).squeeze(-1)
        return {"logit": self.head(emb), "embedding": emb}


class BLCCrossAttnModel(nn.Module):
    """Cross-Attention: SERS embedding attends to clinical tokens."""
    def __init__(self, n_clinical=N_CLINICAL, d_model=256, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.encoder = BLCResNetEncoder()
        self.sers_proj = nn.LayerNorm(d_model)
        self.clin_tokenizer = ClinicalTokenizer(n_clinical, d_model)
        self.attn_layers = nn.ModuleList([
            CrossAttentionBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(d_model, 64), nn.ReLU(),
            nn.Dropout(0.25), nn.Linear(64, 1),
        )

    def forward(self, spectra, clinical):
        sers_emb = self.encoder(spectra)
        q = self.sers_proj(sers_emb).unsqueeze(1)
        kv = self.clin_tokenizer(clinical)

        attn_weights_list = []
        for layer in self.attn_layers:
            q, aw = layer(q, kv)
            attn_weights_list.append(aw)

        emb = self.final_norm(q.squeeze(1))
        return {"logit": self.head(emb), "embedding": emb, "attn_weights": attn_weights_list}


# =========================================================================
# Training
# =========================================================================
def train_one_epoch(model, loader, optimizer, device, runtime, scaler=None):
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total, n = 0.0, 0
    for batch in loader:
        spec = batch["spectra"].to(device, non_blocking=True)
        clin = batch["clinical"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=runtime["use_amp"]):
            out = model(spec, clin)
            loss = criterion(out["logit"], label)
        if scaler and runtime["use_amp"]:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total += loss.item(); n += 1
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device, runtime):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total, n = 0.0, 0
    all_prob, all_true, all_emb = [], [], []

    for batch in loader:
        spec = batch["spectra"].to(device, non_blocking=True)
        clin = batch["clinical"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=runtime["use_amp"]):
            out = model(spec, clin)
            loss = criterion(out["logit"], label)
        total += loss.item(); n += 1
        all_prob.append(torch.sigmoid(out["logit"]).cpu().numpy())
        all_true.append(label.cpu().numpy())
        all_emb.append(out["embedding"].cpu().numpy())

    prob = np.concatenate(all_prob).squeeze()
    true = np.concatenate(all_true).squeeze()
    emb = np.concatenate(all_emb)

    pred = (prob > 0.5).astype(int)
    auc = roc_auc_score(true, prob) if len(np.unique(true)) > 1 else float("nan")
    acc = accuracy_score(true, pred)
    f1 = f1_score(true, pred)
    tn, fp, fn, tp = confusion_matrix(true, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        "loss": total / max(n, 1), "auc": auc, "acc": acc, "f1": f1,
        "sensitivity": sens, "specificity": spec,
        "prob": prob, "true": true, "embedding": emb,
    }


def run_fold(model, X, labels, clinical, train_idx, val_idx, config, device, fold_i, runtime):
    train_ds = BLCDataset(X[train_idx], labels[train_idx], clinical[train_idx], augment=True)
    val_ds = BLCDataset(X[val_idx], labels[val_idx], clinical[val_idx], augment=False)

    bl = labels[train_idx]
    n_pos, n_neg = bl.sum(), (1 - bl).sum()
    weights = np.where(bl > 0.5, 1.0 / max(n_pos, 1), 1.0 / max(n_neg, 1))
    weights = weights / weights.mean()
    sampler = WeightedRandomSampler(weights, len(train_ds), replacement=True)

    train_loader = _build_loader(train_ds, config["batch_size"], runtime, sampler=sampler)
    val_loader = _build_loader(val_ds, config["batch_size"], runtime, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["wd"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, min_lr=1e-6)
    es = EarlyStopping(config["patience"])
    scaler = torch.amp.GradScaler(device="cuda", enabled=runtime["use_amp"])

    best_val_loss, best_state = float("inf"), None
    history = {"train_loss": [], "val_loss": [], "val_auc": []}

    for epoch in range(config["epochs"]):
        tl = train_one_epoch(model, train_loader, optimizer, device, runtime, scaler)
        vm = evaluate(model, val_loader, device, runtime)
        scheduler.step(vm["loss"])
        lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(tl)
        history["val_loss"].append(vm["loss"])
        history["val_auc"].append(vm["auc"])

        if vm["loss"] < best_val_loss:
            best_val_loss = vm["loss"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"    Fold {fold_i+1} Ep {epoch+1:3d} | "
                f"train={tl:.4f} val={vm['loss']:.4f} "
                f"AUC={vm['auc']:.3f} F1={vm['f1']:.3f} "
                f"Sens={vm['sensitivity']:.3f} Spec={vm['specificity']:.3f} lr={lr:.1e}"
            )
        if es.step(vm["loss"]):
            logger.info(f"    Early stop at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)

    train_m = evaluate(model, train_loader, device, runtime)
    val_m = evaluate(model, val_loader, device, runtime)
    return {"history": history, "train": train_m, "val": val_m, "best_state": best_state, "epochs": epoch + 1}


# =========================================================================
# Main
# =========================================================================
def build_model(model_type, n_clinical, args):
    if model_type == "baseline":
        return BLCBaselineModel()
    elif model_type == "film":
        return BLCFiLMModel(n_clinical=n_clinical, film_hidden=args.film_hidden)
    elif model_type == "xattn":
        return BLCCrossAttnModel(
            n_clinical=n_clinical, d_model=args.d_model,
            n_heads=args.n_heads, n_layers=args.n_attn_layers, dropout=args.attn_dropout,
        )
    raise ValueError(f"Unknown model: {model_type}")


def parse_args():
    p = argparse.ArgumentParser(description="BLC MFDS — Bladder Cancer Diagnostic")
    p.add_argument("--model", choices=["baseline", "film", "xattn", "all"], default="all")
    p.add_argument("--aggregate", "-a", choices=["medoid", "mean", "none"], default="medoid")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--device", default="auto")
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--no-amp", action="store_true")
    # FiLM
    p.add_argument("--film-hidden", type=int, default=64)
    # CrossAttn
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-attn-layers", type=int, default=2)
    p.add_argument("--attn-dropout", type=float, default=0.1)
    return p.parse_args()


def main():
    args = parse_args()
    t0 = datetime.now()
    out_dir = Path("results/training/blc_mfds")
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(out_dir / "train.log", mode="w", encoding="utf-8"),
        ],
    )

    logger.info("=" * 64)
    logger.info("  BLC MFDS — Bladder Cancer Diagnostic Module")
    logger.info("=" * 64)

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = torch.device(args.device)
    configure_torch_runtime(device)
    logger.info(f"  Device: {device}")

    # Load data
    logger.info("\n[Step 1] Loading BLC + Non-cancer data...")
    X, labels, clinical, sample_ids, groups_arr = load_blc_data(args.aggregate)
    logger.info(f"  Spectra: {X.shape}, Clinical: {clinical.shape} ({N_CLINICAL} features)")

    config = {
        "batch_size": args.batch_size, "lr": args.lr,
        "wd": args.weight_decay, "epochs": args.epochs,
        "patience": args.patience,
    }
    runtime = resolve_runtime_config(args, device)

    # Models to run
    model_types = ["baseline", "film", "xattn"] if args.model == "all" else [args.model]
    all_results = {}

    for model_type in model_types:
        logger.info(f"\n{'='*64}")
        logger.info(f"  Model: {model_type.upper()}")
        logger.info(f"{'='*64}")

        model_dir = out_dir / model_type
        model_dir.mkdir(parents=True, exist_ok=True)

        cv = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
        val_probs = np.full(len(X), np.nan)
        fold_aucs = []

        for fold_i, (train_idx, val_idx) in enumerate(cv.split(X, labels, sample_ids)):
            logger.info(f"\n  -- Fold {fold_i+1}/{args.n_splits} (train={len(train_idx)}, val={len(val_idx)}) --")
            model = build_model(model_type, N_CLINICAL, args).to(device)
            result = run_fold(model, X, labels, clinical, train_idx, val_idx, config, device, fold_i, runtime)

            val_probs[val_idx] = result["val"]["prob"]
            fold_aucs.append(result["val"]["auc"])

            logger.info(
                f"    Train AUC={result['train']['auc']:.3f} "
                f"Val AUC={result['val']['auc']:.3f} F1={result['val']['f1']:.3f} "
                f"Sens={result['val']['sensitivity']:.3f} Spec={result['val']['specificity']:.3f}"
            )

            # Save checkpoint
            torch.save({
                "fold": fold_i, "model_state_dict": result["best_state"],
                "model_type": model_type, "n_clinical": N_CLINICAL,
            }, model_dir / f"fold_{fold_i}.pt")

        # Overall metrics
        valid = ~np.isnan(val_probs)
        pred = (val_probs[valid] > 0.5).astype(int)
        overall_auc = roc_auc_score(labels[valid], val_probs[valid])
        overall_f1 = f1_score(labels[valid], pred)
        tn, fp, fn, tp = confusion_matrix(labels[valid], pred, labels=[0, 1]).ravel()
        overall_sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        overall_spec = tn / (tn + fp) if (tn + fp) > 0 else 0

        logger.info(f"\n{'='*64}")
        logger.info(f"  {model_type.upper()} — Overall CV Results")
        logger.info(f"{'='*64}")
        logger.info(f"  AUC:         {overall_auc:.4f} (±{np.std(fold_aucs):.4f})")
        logger.info(f"  F1:          {overall_f1:.4f}")
        logger.info(f"  Sensitivity: {overall_sens:.4f}")
        logger.info(f"  Specificity: {overall_spec:.4f}")

        all_results[model_type] = {
            "auc": overall_auc, "auc_std": np.std(fold_aucs),
            "f1": overall_f1, "sensitivity": overall_sens, "specificity": overall_spec,
            "val_probs": val_probs.copy(), "labels": labels.copy(),
        }

        # Save
        summary = {
            "model": model_type, "n_clinical": N_CLINICAL,
            "clinical_features": BLC_CLINICAL_COLS,
            "n_samples": len(X), "n_blc": int(labels.sum()),
            "n_non_cancer": int((1 - labels).sum()),
            "metrics": {
                "auc": overall_auc, "auc_std": float(np.std(fold_aucs)),
                "f1": overall_f1, "sensitivity": overall_sens, "specificity": overall_spec,
            },
        }
        with open(model_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    # Comparison visualization
    if len(all_results) > 1:
        plot_comparison(all_results, out_dir)
        plot_roc_curves(all_results, out_dir)

    elapsed = datetime.now() - t0
    logger.info(f"\n  Done in {elapsed}. Output: {out_dir}")


def plot_comparison(results, out_dir):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("BLC MFDS: Baseline vs FiLM vs Cross-Attention (26 clinical features)",
                 fontsize=13, fontweight="bold")

    metrics = ["auc", "f1", "sensitivity", "specificity"]
    titles = ["AUC", "F1", "Sensitivity", "Specificity"]
    colors = {"baseline": "#95A5A6", "film": "#3498DB", "xattn": "#E74C3C"}

    for ax, metric, title in zip(axes, metrics, titles):
        names = list(results.keys())
        vals = [results[m][metric] for m in names]
        bars = ax.bar(names, vals, color=[colors.get(m, "gray") for m in names],
                      edgecolor="white", linewidth=2, width=0.6)
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(0.7, 1.0)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.tight_layout()
    fig.savefig(out_dir / "blc_mfds_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out_dir / 'blc_mfds_comparison.png'}")


def plot_roc_curves(results, out_dir):
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = {"baseline": "#95A5A6", "film": "#3498DB", "xattn": "#E74C3C"}

    for name, res in results.items():
        valid = ~np.isnan(res["val_probs"])
        fpr, tpr, _ = roc_curve(res["labels"][valid], res["val_probs"][valid])
        ax.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})",
                color=colors.get(name, "gray"), linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("BLC MFDS — ROC Curves (5-fold CV)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    plt.tight_layout()
    fig.savefig(out_dir / "blc_mfds_roc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out_dir / 'blc_mfds_roc.png'}")


if __name__ == "__main__":
    main()
