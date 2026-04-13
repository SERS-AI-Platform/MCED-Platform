"""
SERS Cancer Detection — Training Pipeline (ResNet18-1D)

Two-stage hierarchical: Binary + Cancer Type (8: incl CPAN, SPAN separate, BLC)

Usage:
    python train.py
    python train.py --epochs 200 --lr 3e-4
    python train.py --aggregate medoid --n-splits 5

Outputs (results/training/):
    fold_predictions.npz       — all predictions for test.py
    checkpoints/fold_*.pt      — per-fold model weights
    training_summary.json      — config + metrics
    fold_metrics.csv / reports / training_curves.png
"""

from __future__ import annotations

import sys
import argparse
import logging
import os
import json
import re
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from functools import singledispatch
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
    average_precision_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import WeightedRandomSampler
import joblib

import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from models.model import (
        ModelConfig, TwoStageLoss, SERSDataset, build_model, model_summary,
    )
except ImportError:
    from model import (
        ModelConfig, TwoStageLoss, SERSDataset, build_model, model_summary,
    )
from src.sers.config import RESULTS_DIR
from src.sers.visualization import plot_cancer_peak_difference

logger = logging.getLogger(__name__)

MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "resnet18": "ResNet18-1D",
    "cnn1d": "CNN1D-Shallow",
    "xgboost": "XGBoost-Hierarchical",
}

TORCH_MODEL_NAMES = {"resnet18", "cnn1d"}
CLASSICAL_MODEL_NAMES = {"logistic_regression", "random_forest", "xgboost"}

XGBOOST_DEFAULT_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "min_child_weight": 1,
    "n_jobs": 4,
    "tree_method": "hist",
    "verbosity": 0,
}

LOGREG_DEFAULT_PARAMS = {
    "C": 1.0,
    "max_iter": 1000,
    "solver": "saga",
    "class_weight": "balanced",
}

RANDOM_FOREST_DEFAULT_PARAMS = {
    "n_estimators": 500,
    "max_depth": None,
    "min_samples_leaf": 1,
    "class_weight": "balanced_subsample",
    "n_jobs": 1,
}


@dataclass(frozen=True)
class TorchRunnerRequest:
    X: np.ndarray
    binary_labels: np.ndarray
    cancer_type_labels: np.ndarray
    sample_ids: np.ndarray
    groups_arr: np.ndarray
    config: ModelConfig
    device: torch.device
    model_name: str
    runtime: Dict[str, Any]
    n_splits: int = 5
    ckpt_dir: Optional[Path] = None


@dataclass(frozen=True)
class ClassicalRunnerRequest:
    model_name: str
    X: np.ndarray
    binary_labels: np.ndarray
    cancer_type_labels: np.ndarray
    sample_ids: np.ndarray
    groups_arr: np.ndarray
    config: ModelConfig
    model_params: Dict[str, Any]
    n_splits: int = 5
    ckpt_dir: Optional[Path] = None


@dataclass
class FoldArtifacts:
    fold_result: Dict[str, Any]
    val_binary_prob: np.ndarray
    val_cancer_logits: np.ndarray
    val_embedding: np.ndarray
    train_last: Dict[str, Any]
    extra: Dict[str, Any] = field(default_factory=dict)


def apply_training_overrides(config, args):
    overrides = {}
    if args.epochs is not None:
        overrides["n_epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.lr is not None:
        overrides["learning_rate"] = args.lr
    if args.weight_decay is not None:
        overrides["weight_decay"] = args.weight_decay
    if args.dropout_rate is not None:
        overrides["dropout_rate"] = args.dropout_rate
    if args.stage2_loss_weight is not None:
        overrides["stage2_loss_weight"] = args.stage2_loss_weight
    if args.head_hidden_dim is not None:
        overrides["head_hidden_dim"] = args.head_hidden_dim
    if args.resnet_channels is not None:
        overrides["resnet_channels"] = tuple(args.resnet_channels)
        overrides["encoder_output_dim"] = int(args.resnet_channels[-1])
    if args.use_focal_loss:
        overrides["use_focal_loss"] = True
    if args.focal_gamma is not None:
        overrides["focal_gamma"] = args.focal_gamma
    if args.class_balance_beta is not None:
        overrides["class_balance_beta"] = args.class_balance_beta
    return ModelConfig(**{**config.__dict__, **overrides}) if overrides else config


def resolve_xgboost_params(args):
    params = dict(XGBOOST_DEFAULT_PARAMS)
    if args.xgb_n_estimators is not None:
        params["n_estimators"] = args.xgb_n_estimators
    if args.xgb_max_depth is not None:
        params["max_depth"] = args.xgb_max_depth
    if args.xgb_learning_rate is not None:
        params["learning_rate"] = args.xgb_learning_rate
    if args.xgb_subsample is not None:
        params["subsample"] = args.xgb_subsample
    if args.xgb_colsample_bytree is not None:
        params["colsample_bytree"] = args.xgb_colsample_bytree
    if args.xgb_reg_lambda is not None:
        params["reg_lambda"] = args.xgb_reg_lambda
    if args.xgb_min_child_weight is not None:
        params["min_child_weight"] = args.xgb_min_child_weight
    return params


def resolve_logreg_params(args):
    params = dict(LOGREG_DEFAULT_PARAMS)
    if args.logreg_c is not None:
        params["C"] = args.logreg_c
    if args.logreg_max_iter is not None:
        params["max_iter"] = args.logreg_max_iter
    return params


def resolve_random_forest_params(args):
    params = dict(RANDOM_FOREST_DEFAULT_PARAMS)
    if args.rf_n_estimators is not None:
        params["n_estimators"] = args.rf_n_estimators
    if args.rf_max_depth is not None:
        params["max_depth"] = args.rf_max_depth
    if args.rf_min_samples_leaf is not None:
        params["min_samples_leaf"] = args.rf_min_samples_leaf
    return params


def compute_effective_num_weights(labels, n_classes, beta):
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / np.maximum(effective_num, 1e-12)
    weights = weights / weights.sum() * n_classes
    return weights


def resolve_runtime_config(args, device):
    cpu_count = os.cpu_count() or 1
    num_workers = args.num_workers if args.num_workers is not None else min(4, cpu_count)
    num_workers = max(0, int(num_workers))
    if os.name == "nt" and num_workers > 0:
        logger.warning("  Windows environment detected; forcing num_workers=0 to avoid multiprocessing permission errors")
        num_workers = 0
    pin_memory = device.type == "cuda"
    prefetch_factor = None if num_workers == 0 else max(2, int(args.prefetch_factor))
    use_amp = device.type == "cuda" and not args.no_amp
    return {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "prefetch_factor": prefetch_factor,
        "use_amp": use_amp,
    }


def configure_torch_runtime(device):
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")


# =============================================================================
# 1. Data Loading
# =============================================================================
def load_processed_spectra(csv_path=None):
    if csv_path is None:
        csv_path = RESULTS_DIR / "processed_spectra.csv"
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} spectra from {csv_path}")
    logger.info(f"  Groups: {sorted(df['group'].unique())}")
    logger.info(f"  Samples/group: {df.groupby('group')['sample_id'].nunique().to_dict()}")
    return df


def get_feature_columns(df):
    return [c for c in df.columns if c.startswith("x_")]


def normalize_group_selection(values):
    if not values:
        return None
    return tuple(str(v).strip() for v in values if str(v).strip())


def apply_class_selection(config, cancer_types=None, non_cancer_groups=None):
    selected_cancer = normalize_group_selection(cancer_types) or tuple(config.cancer_types)
    selected_non_cancer = normalize_group_selection(non_cancer_groups) or tuple(config.non_cancer_groups)

    unknown_cancer = sorted(set(selected_cancer) - set(config.cancer_types))
    unknown_non_cancer = sorted(set(selected_non_cancer) - set(config.non_cancer_groups))
    if unknown_cancer:
        raise ValueError(f"Unknown cancer types: {unknown_cancer}")
    if unknown_non_cancer:
        raise ValueError(f"Unknown non-cancer groups: {unknown_non_cancer}")
    if not selected_cancer:
        raise ValueError("At least one cancer type must be selected")
    if not selected_non_cancer:
        raise ValueError("At least one non-cancer group must be selected")

    # Expand cancer_groups_raw: include alias members for selected types
    raw_groups = []
    for ct in selected_cancer:
        if ct in config.group_aliases:
            raw_groups.extend(config.group_aliases[ct])
        else:
            raw_groups.append(ct)

    return ModelConfig(
        **{
            **config.__dict__,
            "cancer_types": selected_cancer,
            "cancer_groups_raw": tuple(raw_groups),
            "non_cancer_groups": selected_non_cancer,
        }
    )


def resolve_aliases(df, config):
    """With empty aliases, this is effectively a no-op."""
    df = df.copy()
    df["group_raw"] = df["group"]
    df["group"] = df["group"].apply(config.resolve_group)
    changed = df["group_raw"] != df["group"]
    if changed.any():
        for raw, resolved in df[changed][["group_raw", "group"]].drop_duplicates().values:
            logger.info(f"    {raw} → {resolved}")
    else:
        logger.info("  No aliases (CPAN/SPAN kept separate)")
    return df


def apply_patient_exclusions(df, exclude_csv):
    """Drop specific patients listed in an exclusion CSV.

    The CSV must have 'group' and 'sample_id' columns matching the spectral data.
    """
    excl = pd.read_csv(exclude_csv)
    excl_keys = set(zip(excl["group"].astype(str), excl["sample_id"].astype(str)))
    mask = df.apply(
        lambda r: (str(r["group"]), str(r["sample_id"])) in excl_keys, axis=1
    )
    n_patients = len(set(zip(df[mask]["group"], df[mask]["sample_id"])))
    n_spectra = mask.sum()
    logger.info(f"  Excluded {n_patients} patients ({n_spectra} spectra) from {exclude_csv}")
    return df[~mask].reset_index(drop=True)


def resolve_version_name(model_root: Path, explicit_version: str | None = None) -> str:
    if explicit_version:
        return str(explicit_version).strip()

    existing = []
    for child in model_root.iterdir() if model_root.exists() else []:
        if not child.is_dir():
            continue
        name = child.name.lower()
        if len(name) >= 2 and name.startswith("v") and name[1:].isdigit():
            existing.append(int(name[1:]))
    next_idx = max(existing, default=0) + 1
    return f"v{next_idx:03d}"


def resolve_experiment_name(base_output: Path, explicit_experiment: str | None = None) -> str:
    if explicit_experiment:
        return str(explicit_experiment).strip()

    existing = []
    for child in base_output.iterdir() if base_output.exists() else []:
        if not child.is_dir():
            continue
        match = re.fullmatch(r"experiment_(\d+)", child.name.lower())
        if match:
            existing.append(int(match.group(1)))
    next_idx = max(existing, default=0) + 1
    return f"experiment_{next_idx:03d}"


def resolve_experiment_output_dir(base_output: Path, experiment: str | None = None) -> tuple[Path, str]:
    base_output.mkdir(parents=True, exist_ok=True)
    experiment_name = resolve_experiment_name(base_output, experiment)
    experiment_dir = base_output / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    with open(base_output / "latest_experiment.txt", "w", encoding="utf-8") as f:
        f.write(experiment_name)
    return experiment_dir, experiment_name


def resolve_run_output_dir(base_output: Path, model_name: str, version: str | None = None) -> tuple[Path, str]:
    model_root = base_output / model_name
    model_root.mkdir(parents=True, exist_ok=True)
    version_name = resolve_version_name(model_root, version)
    run_dir = model_root / version_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(model_root / "latest.txt", "w", encoding="utf-8") as f:
        f.write(version_name)
    return run_dir, version_name


# =============================================================================
# 2. Replicate Aggregation
# =============================================================================
def aggregate_medoid(df, feature_cols):
    rows = []
    for (group, sid), sub in df.groupby(["group", "sample_id"]):
        if len(sub) == 1:
            rows.append(sub.iloc[0]); continue
        spectra = sub[feature_cols].values
        medoid_idx = np.corrcoef(spectra).mean(axis=1).argmax()
        rows.append(sub.iloc[medoid_idx])
    result = pd.DataFrame(rows).reset_index(drop=True)
    logger.info(f"  Medoid: {len(df)} → {len(result)} samples")
    return result


def aggregate_mean(df, feature_cols):
    agg = df.groupby(["group", "sample_id"])[feature_cols].mean().reset_index()
    logger.info(f"  Mean: {len(df)} → {len(agg)} samples")
    return agg


def aggregate_replicates(df, feature_cols, method):
    logger.info(f"\n[Aggregation] method={method}")
    if method == "medoid": return aggregate_medoid(df, feature_cols)
    if method == "mean":   return aggregate_mean(df, feature_cols)
    if method == "none":   return df
    raise ValueError(f"Unknown: {method}")


# =============================================================================
# 3. Label Creation
# =============================================================================
def create_labels(df, config):
    feature_cols = get_feature_columns(df)
    valid_groups = set(config.cancer_types) | set(config.non_cancer_groups)
    df_valid = df[df["group"].isin(valid_groups)].copy()

    dropped = set(df["group"].unique()) - valid_groups
    if dropped:
        logger.warning(f"  Dropped groups: {dropped}")

    present_groups = set(df_valid["group"].unique())
    missing_groups = sorted(valid_groups - present_groups)
    if missing_groups:
        logger.warning(f"  Selected groups absent from data: {missing_groups}")

    X = df_valid[feature_cols].values
    groups_list = df_valid["group"].tolist()
    sample_ids = df_valid["sample_id"].values if "sample_id" in df_valid.columns else np.arange(len(df_valid))

    binary_labels = np.array([1 if g in config.cancer_types else 0 for g in groups_list])
    cancer_type_labels = np.array([config.cancer_type_index(g) for g in groups_list])

    if len(df_valid) == 0:
        raise ValueError("No samples remain after applying class selection")
    if len(np.unique(binary_labels)) < 2:
        raise ValueError("Selected training groups must include both cancer and non-cancer samples")

    logger.info(f"  Binary: {(binary_labels==1).sum()} cancer + {(binary_labels==0).sum()} non-cancer")
    for i, name in enumerate(config.cancer_types):
        logger.info(f"    [{i}] {name}: {(cancer_type_labels==i).sum()}")

    return X, binary_labels, cancer_type_labels, sample_ids, np.array(groups_list)


# =============================================================================
# 4. Training
# =============================================================================
class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-4):
        self.patience, self.min_delta = patience, min_delta
        self.counter, self.best_loss, self.should_stop = 0, float("inf"), False

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss, self.counter = val_loss, 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def _build_loader(dataset, batch_size, runtime, sampler=None, shuffle=False):
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "num_workers": runtime["num_workers"],
        "pin_memory": runtime["pin_memory"],
    }
    if sampler is not None:
        kwargs["sampler"] = sampler
    else:
        kwargs["shuffle"] = shuffle
    if runtime["num_workers"] > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = runtime["prefetch_factor"]
    return DataLoader(**kwargs)


def train_one_epoch(model, loader, criterion, optimizer, device, runtime, scaler=None):
    model.train()
    total, n = 0.0, 0
    use_amp = runtime["use_amp"]
    for batch in loader:
        spec = batch["spectra"].to(device, non_blocking=runtime["pin_memory"])
        by = batch["binary_label"].to(device, non_blocking=runtime["pin_memory"])
        cy = batch["cancer_type_label"].to(device, non_blocking=runtime["pin_memory"])
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
def evaluate(model, loader, criterion, device, runtime):
    model.eval()
    total, n = 0.0, 0
    all_bp, all_bt, all_cl, all_ct, all_emb = [], [], [], [], []
    use_amp = runtime["use_amp"]

    for batch in loader:
        spec = batch["spectra"].to(device, non_blocking=runtime["pin_memory"])
        by = batch["binary_label"].to(device, non_blocking=runtime["pin_memory"])
        cy = batch["cancer_type_label"].to(device, non_blocking=runtime["pin_memory"])
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
    pr_auc_s1 = average_precision_score(bt, bp) if len(np.unique(bt)) > 1 else float("nan")
    pred_s1 = (bp > 0.5).astype(int)
    acc_s1 = accuracy_score(bt, pred_s1)

    cm = ct >= 0
    auc_s2 = float("nan")
    acc_s2 = float("nan")
    f1_macro_s2 = float("nan")
    pr_auc_s2 = float("nan")
    if cm.sum() > 0 and len(np.unique(ct[cm])) > 1:
        cp = torch.softmax(torch.tensor(cl[cm]), dim=-1).numpy()
        cpred = cp.argmax(axis=1)
        acc_s2 = accuracy_score(ct[cm], cpred)
        f1_macro_s2 = f1_score(ct[cm], cpred, average="macro", zero_division=0)
        try:
            auc_s2 = roc_auc_score(ct[cm], cp, multi_class="ovr", average="macro")
        except ValueError:
            pass
        try:
            per_class_ap = []
            for cls_idx in np.unique(ct[cm]):
                cls_bg = (ct[cm] == cls_idx).astype(int)
                per_class_ap.append(average_precision_score(cls_bg, cp[:, int(cls_idx)]))
            pr_auc_s2 = float(np.mean(per_class_ap))
        except (ValueError, IndexError):
            pass

    return {
        "loss": total / max(n, 1),
        "auc_s1": auc_s1,
        "pr_auc_s1": pr_auc_s1,
        "auc_s2": auc_s2,
        "pr_auc_s2": pr_auc_s2,
        "acc_s1": acc_s1,
        "acc_s2": acc_s2,
        "f1_macro_s2": f1_macro_s2,
        "binary_prob": bp, "binary_true": bt,
        "cancer_logits": cl, "cancer_true": ct,
        "embedding": emb,
    }


def train_fold(model, train_ds, val_ds, config, device, fold_i, runtime):
    # WeightedRandomSampler: balance classes in each mini-batch
    bl = train_ds.binary_labels.squeeze().numpy()
    ctl = train_ds.cancer_type_labels.numpy()

    # Build per-sample weights combining binary + cancer type
    sample_weights = np.ones(len(train_ds), dtype=np.float64)
    n_pos, n_neg = (bl > 0.5).sum(), (bl <= 0.5).sum()
    for i in range(len(train_ds)):
        if bl[i] > 0.5:
            # Cancer sample: weight by inverse cancer-type frequency
            ct = ctl[i]
            ct_count = (ctl[ctl >= 0] == ct).sum()
            sample_weights[i] = 1.0 / max(ct_count, 1)
        else:
            # Non-cancer: weight to balance cancer vs non-cancer
            sample_weights[i] = 1.0 / max(n_neg, 1)
    # Normalize so mean = 1
    sample_weights = sample_weights / sample_weights.mean()

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_ds),
        replacement=True,
    )
    train_loader = _build_loader(train_ds, config.batch_size, runtime, sampler=sampler)
    val_loader = _build_loader(val_ds, config.batch_size, runtime, shuffle=False)

    # Class weights for loss
    pw = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)

    cm = ctl >= 0
    cw_t = None
    if cm.sum() > 0:
        if config.class_balance_beta is not None:
            cw = compute_effective_num_weights(
                ctl[cm],
                config.n_cancer_types,
                config.class_balance_beta,
            )
        else:
            cc = np.bincount(ctl[cm], minlength=config.n_cancer_types).astype(float)
            cw = 1.0 / np.maximum(cc, 1)
            cw = cw / cw.sum() * config.n_cancer_types
        cw_t = torch.tensor(cw, dtype=torch.float32).to(device)

    criterion = TwoStageLoss(
        config.stage1_loss_weight, config.stage2_loss_weight, pw, cw_t,
        label_smoothing=0.1,
        use_focal_loss=config.use_focal_loss,
        focal_gamma=config.focal_gamma,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=config.scheduler_patience, factor=0.5, min_lr=1e-6
    )
    es = EarlyStopping(config.early_stopping_patience)
    scaler = torch.amp.GradScaler(device="cuda", enabled=runtime["use_amp"])

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_auc_s1": [],
        "val_auc_s2": [],
        "val_f1_macro_s2": [],
        "val_error_s1": [],
        "val_error_s2": [],
        "lr": [],
    }
    best_val_loss, best_state = float("inf"), None

    for epoch in range(config.n_epochs):
        tl = train_one_epoch(model, train_loader, criterion, optimizer, device, runtime, scaler=scaler)
        vm = evaluate(model, val_loader, criterion, device, runtime)
        scheduler.step(vm["loss"])
        lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(tl)
        history["val_loss"].append(vm["loss"])
        history["val_auc_s1"].append(vm["auc_s1"])
        history["val_auc_s2"].append(vm["auc_s2"])
        history["val_f1_macro_s2"].append(vm["f1_macro_s2"])
        history["val_error_s1"].append(1.0 - vm["acc_s1"])
        history["val_error_s2"].append(np.nan if np.isnan(vm["acc_s2"]) else 1.0 - vm["acc_s2"])
        history["lr"].append(lr)

        if vm["loss"] < best_val_loss:
            best_val_loss = vm["loss"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"    Fold {fold_i+1} Ep {epoch+1:3d} │ "
                f"train={tl:.4f} val={vm['loss']:.4f} "
                f"S1_AUC={vm['auc_s1']:.3f} "
                f"S2_F1={vm['f1_macro_s2']:.3f} "
                f"S2_AUC={vm['auc_s2']:.3f} lr={lr:.1e}"
            )
        if es.step(vm["loss"]):
            logger.info(f"    Early stop at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)

    train_metrics = evaluate(model, train_loader, criterion, device, runtime)
    val_metrics = evaluate(model, val_loader, criterion, device, runtime)

    return {
        "history": history, "best_val_loss": best_val_loss,
        "train_metrics": train_metrics, "val_metrics": val_metrics,
        "epochs": epoch + 1, "best_state": best_state,
    }


def _compute_overall_metrics(
    val_binary_prob,
    val_cancer_logits,
    binary_labels,
    cancer_type_labels,
    train_last,
    include_train_stage2_f1=False,
):
    valid = ~np.isnan(val_binary_prob)
    bp_v = val_binary_prob[valid]
    bt_v = binary_labels[valid]
    cl_v = val_cancer_logits[valid]
    ct_v = cancer_type_labels[valid]

    bpred = (bp_v > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(bt_v, bpred, labels=[0, 1]).ravel()

    overall = {
        "val_s1_auc": float(roc_auc_score(bt_v, bp_v)),
        "val_s1_pr_auc": float(average_precision_score(bt_v, bp_v)),
        "val_s1_accuracy": float(accuracy_score(bt_v, bpred)),
        "val_s1_sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0,
        "val_s1_specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0,
        "val_s1_f1": float(f1_score(bt_v, bpred)),
    }

    cancer_mask = ct_v >= 0
    if cancer_mask.sum() > 0:
        cancer_prob = torch.softmax(torch.tensor(cl_v[cancer_mask]), dim=-1).numpy()
        cancer_pred = cancer_prob.argmax(axis=1)
        overall["val_s2_accuracy"] = float(accuracy_score(ct_v[cancer_mask], cancer_pred))
        overall["val_s2_f1_macro"] = float(
            f1_score(ct_v[cancer_mask], cancer_pred, average="macro", zero_division=0)
        )
        try:
            overall["val_s2_auc"] = float(
                roc_auc_score(ct_v[cancer_mask], cancer_prob, multi_class="ovr", average="macro")
            )
        except ValueError:
            overall["val_s2_auc"] = float("nan")
        try:
            per_class_ap = []
            for cls_idx in np.unique(ct_v[cancer_mask]):
                cls_bg = (ct_v[cancer_mask] == cls_idx).astype(int)
                per_class_ap.append(average_precision_score(cls_bg, cancer_prob[:, int(cls_idx)]))
            overall["val_s2_pr_auc"] = float(np.mean(per_class_ap))
        except (ValueError, IndexError):
            overall["val_s2_pr_auc"] = float("nan")

    if train_last:
        tbp = train_last["binary_prob"]
        tbt = train_last["binary_true"]
        overall["train_s1_auc"] = float(roc_auc_score(tbt, tbp))
        overall["train_s1_pr_auc"] = float(average_precision_score(tbt, tbp))
        train_cancer_mask = train_last["cancer_true"] >= 0
        if train_cancer_mask.sum() > 0:
            train_prob = torch.softmax(
                torch.tensor(train_last["cancer_logits"][train_cancer_mask]),
                dim=-1,
            ).numpy()
            if include_train_stage2_f1:
                overall["train_s2_f1_macro"] = float(
                    f1_score(
                        train_last["cancer_true"][train_cancer_mask],
                        train_prob.argmax(axis=1),
                        average="macro",
                        zero_division=0,
                    )
                )
            try:
                overall["train_s2_auc"] = float(
                    roc_auc_score(
                        train_last["cancer_true"][train_cancer_mask],
                        train_prob,
                        multi_class="ovr",
                        average="macro",
                    )
                )
            except ValueError:
                overall["train_s2_auc"] = float("nan")
            try:
                per_class_ap = []
                for cls_idx in np.unique(train_last["cancer_true"][train_cancer_mask]):
                    cls_bg = (train_last["cancer_true"][train_cancer_mask] == cls_idx).astype(int)
                    per_class_ap.append(average_precision_score(cls_bg, train_prob[:, int(cls_idx)]))
                overall["train_s2_pr_auc"] = float(np.mean(per_class_ap))
            except (ValueError, IndexError):
                overall["train_s2_pr_auc"] = float("nan")

    return overall


class BaseCrossValidator(ABC):
    include_train_stage2_f1 = False

    def __init__(
        self,
        X,
        binary_labels,
        cancer_type_labels,
        sample_ids,
        groups_arr,
        config,
        n_splits=5,
        ckpt_dir=None,
    ):
        self.X = X
        self.binary_labels = binary_labels
        self.cancer_type_labels = cancer_type_labels
        self.sample_ids = sample_ids
        self.groups_arr = groups_arr
        self.config = config
        self.n_splits = n_splits
        self.ckpt_dir = ckpt_dir
        self.cv = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=config.random_state,
        )

    @property
    @abstractmethod
    def model_display_name(self):
        raise NotImplementedError

    @abstractmethod
    def _make_embedding_buffer(self, n_samples):
        raise NotImplementedError

    @abstractmethod
    def _run_fold(self, fold_idx, train_idx, val_idx):
        raise NotImplementedError

    @abstractmethod
    def _log_fold_header(self, fold_idx, train_idx, val_idx):
        raise NotImplementedError

    @abstractmethod
    def _log_fold_summary(self, fold_idx, artifacts):
        raise NotImplementedError

    def _extract_history(self, fold_result):
        return fold_result.get("history")

    def _save_fold_artifacts(self, fold_idx, train_idx, val_idx, artifacts):
        return None

    def run(self):
        folds = []
        histories = []

        n_samples = len(self.X)
        val_binary_prob = np.full(n_samples, np.nan)
        val_cancer_logits = np.full((n_samples, self.config.n_cancer_types), np.nan)
        val_embedding = self._make_embedding_buffer(n_samples)
        fold_ids = np.full(n_samples, -1, dtype=int)
        train_last = {}

        for fold_idx, (train_idx, val_idx) in enumerate(
            self.cv.split(self.X, self.binary_labels, self.sample_ids)
        ):
            self._log_fold_header(fold_idx, train_idx, val_idx)
            artifacts = self._run_fold(fold_idx, train_idx, val_idx)
            folds.append(artifacts.fold_result)

            history = self._extract_history(artifacts.fold_result)
            if history is not None:
                histories.append(history)

            val_binary_prob[val_idx] = artifacts.val_binary_prob
            val_cancer_logits[val_idx] = artifacts.val_cancer_logits
            val_embedding[val_idx] = artifacts.val_embedding
            fold_ids[val_idx] = fold_idx
            train_last = artifacts.train_last

            self._log_fold_summary(fold_idx, artifacts)
            if self.ckpt_dir:
                os.makedirs(self.ckpt_dir, exist_ok=True)
                self._save_fold_artifacts(fold_idx, train_idx, val_idx, artifacts)

        overall = _compute_overall_metrics(
            val_binary_prob,
            val_cancer_logits,
            self.binary_labels,
            self.cancer_type_labels,
            train_last,
            include_train_stage2_f1=self.include_train_stage2_f1,
        )

        logger.info("\n" + "=" * 64)
        logger.info(f"  CV Results ({self.model_display_name})")
        logger.info("=" * 64)
        for key, value in overall.items():
            logger.info(f"  {key:28s}: {value:.4f}")

        return {
            "overall": overall,
            "folds": folds,
            "histories": histories,
            "val_binary_prob": val_binary_prob,
            "val_cancer_logits": val_cancer_logits,
            "val_embedding": val_embedding,
            "fold_ids": fold_ids,
            "train_last": train_last,
            "binary_labels": self.binary_labels,
            "cancer_type_labels": self.cancer_type_labels,
            "groups": self.groups_arr,
            "sample_ids": self.sample_ids,
            "X": self.X,
        }


class TorchCrossValidator(BaseCrossValidator):
    def __init__(self, request):
        super().__init__(
            request.X,
            request.binary_labels,
            request.cancer_type_labels,
            request.sample_ids,
            request.groups_arr,
            request.config,
            n_splits=request.n_splits,
            ckpt_dir=request.ckpt_dir,
        )
        self.device = request.device
        self.model_name = request.model_name
        self.runtime = request.runtime

    @property
    def model_display_name(self):
        return MODEL_DISPLAY_NAMES[self.model_name]

    def _make_embedding_buffer(self, n_samples):
        return np.full((n_samples, self.config.encoder_output_dim), np.nan)

    def _log_fold_header(self, fold_idx, train_idx, val_idx):
        logger.info(
            f"\n  -- Fold {fold_idx+1}/{self.n_splits} "
            f"(train={len(train_idx)}, val={len(val_idx)}) --"
        )

    def _run_fold(self, fold_idx, train_idx, val_idx):
        train_ds = SERSDataset(
            self.X[train_idx],
            self.binary_labels[train_idx],
            self.cancer_type_labels[train_idx],
            groups=self.groups_arr[train_idx].tolist(),
            sample_ids=[str(sample_id) for sample_id in self.sample_ids[train_idx]],
            augment=True,
        )
        val_ds = SERSDataset(
            self.X[val_idx],
            self.binary_labels[val_idx],
            self.cancer_type_labels[val_idx],
            groups=self.groups_arr[val_idx].tolist(),
            sample_ids=[str(sample_id) for sample_id in self.sample_ids[val_idx]],
            augment=False,
        )

        model = build_model(self.model_name, self.config).to(self.device)
        fold_result = train_fold(
            model,
            train_ds,
            val_ds,
            self.config,
            self.device,
            fold_idx,
            self.runtime,
        )
        train_metrics = fold_result["train_metrics"]
        val_metrics = fold_result["val_metrics"]

        return FoldArtifacts(
            fold_result=fold_result,
            val_binary_prob=val_metrics["binary_prob"],
            val_cancer_logits=val_metrics["cancer_logits"],
            val_embedding=val_metrics["embedding"],
            train_last={
                "binary_prob": train_metrics["binary_prob"],
                "binary_true": train_metrics["binary_true"],
                "cancer_logits": train_metrics["cancer_logits"],
                "cancer_true": train_metrics["cancer_true"],
                "embedding": train_metrics["embedding"],
                "idx": train_idx,
            },
        )

    def _log_fold_summary(self, fold_idx, artifacts):
        train_metrics = artifacts.fold_result["train_metrics"]
        val_metrics = artifacts.fold_result["val_metrics"]
        logger.info(
            f"    Train S1={train_metrics['auc_s1']:.3f} S2={train_metrics['auc_s2']:.3f} "
            f"Val S1={val_metrics['auc_s1']:.3f} S2={val_metrics['auc_s2']:.3f}"
        )

    def _save_fold_artifacts(self, fold_idx, train_idx, val_idx, artifacts):
        torch.save(
            {
                "fold": fold_idx,
                "model_state_dict": artifacts.fold_result["best_state"],
                "model_name": self.model_name,
                "config": self.config.__dict__,
                "train_idx": train_idx,
                "val_idx": val_idx,
            },
            self.ckpt_dir / f"fold_{fold_idx}.pt",
        )


class ClassicalCrossValidator(BaseCrossValidator):
    include_train_stage2_f1 = True

    def __init__(self, request):
        super().__init__(
            request.X,
            request.binary_labels,
            request.cancer_type_labels,
            request.sample_ids,
            request.groups_arr,
            request.config,
            n_splits=request.n_splits,
            ckpt_dir=request.ckpt_dir,
        )
        self.model_name = request.model_name
        self.model_params = request.model_params

    @property
    def model_display_name(self):
        return MODEL_DISPLAY_NAMES[self.model_name]

    def _make_embedding_buffer(self, n_samples):
        return np.full((n_samples, self.X.shape[1]), np.nan)

    def _extract_history(self, fold_result):
        return None

    def _log_fold_header(self, fold_idx, train_idx, val_idx):
        logger.info(
            f"\n  -- Fold {fold_idx+1}/{self.n_splits} "
            f"(train={len(train_idx)}, val={len(val_idx)}) --"
        )

    def _run_fold(self, fold_idx, train_idx, val_idx):
        X_train, X_val = self.X[train_idx], self.X[val_idx]
        yb_train, yb_val = self.binary_labels[train_idx], self.binary_labels[val_idx]
        yc_train, yc_val = self.cancer_type_labels[train_idx], self.cancer_type_labels[val_idx]

        binary_model = _build_classical_estimator(
            self.model_name,
            "binary:logistic",
            self.config.random_state + fold_idx,
            self.model_params,
        )
        binary_model = _fit_classical_estimator(
            self.model_name,
            binary_model,
            X_train,
            yb_train,
            sample_weight=_binary_sample_weights(yb_train),
        )

        bp_train = binary_model.predict_proba(X_train)[:, 1]
        bp_val = binary_model.predict_proba(X_val)[:, 1]

        cancer_train_mask = yc_train >= 0
        stage2_model = None
        present_classes = np.array([], dtype=int)
        train_stage2_prob = np.full(
            (len(X_train), self.config.n_cancer_types),
            1.0 / self.config.n_cancer_types,
        )
        val_stage2_prob = np.full(
            (len(X_val), self.config.n_cancer_types),
            1.0 / self.config.n_cancer_types,
        )

        if cancer_train_mask.sum() > 0:
            present_classes = np.unique(yc_train[cancer_train_mask])
            if len(present_classes) == 1:
                train_stage2_prob = np.zeros((len(X_train), self.config.n_cancer_types), dtype=np.float64)
                val_stage2_prob = np.zeros((len(X_val), self.config.n_cancer_types), dtype=np.float64)
                train_stage2_prob[:, present_classes[0]] = 1.0
                val_stage2_prob[:, present_classes[0]] = 1.0
            else:
                local_labels = np.searchsorted(present_classes, yc_train[cancer_train_mask])
                stage2_model = _build_classical_estimator(
                    self.model_name,
                    "multi:softprob",
                    self.config.random_state + fold_idx + 100,
                    self.model_params,
                    num_class=len(present_classes),
                )
                stage2_model = _fit_classical_estimator(
                    self.model_name,
                    stage2_model,
                    X_train[cancer_train_mask],
                    local_labels,
                    sample_weight=_multiclass_sample_weights(local_labels),
                )
                train_stage2_local = stage2_model.predict_proba(X_train)
                val_stage2_local = stage2_model.predict_proba(X_val)
                if train_stage2_local.ndim == 1:
                    train_stage2_local = np.column_stack([1 - train_stage2_local, train_stage2_local])
                    val_stage2_local = np.column_stack([1 - val_stage2_local, val_stage2_local])
                train_stage2_prob = _expand_stage2_probabilities(
                    np.asarray(train_stage2_local),
                    present_classes,
                    self.config.n_cancer_types,
                )
                val_stage2_prob = _expand_stage2_probabilities(
                    np.asarray(val_stage2_local),
                    present_classes,
                    self.config.n_cancer_types,
                )

        train_stage2_mask = yc_train >= 0
        if train_stage2_mask.sum() > 0:
            train_pred = train_stage2_prob[train_stage2_mask].argmax(axis=1)
            train_f1_s2 = f1_score(
                yc_train[train_stage2_mask],
                train_pred,
                average="macro",
                zero_division=0,
            )
            try:
                train_auc_s2 = roc_auc_score(
                    yc_train[train_stage2_mask],
                    train_stage2_prob[train_stage2_mask],
                    multi_class="ovr",
                    average="macro",
                )
            except ValueError:
                train_auc_s2 = float("nan")
        else:
            train_f1_s2 = float("nan")
            train_auc_s2 = float("nan")

        val_stage2_mask = yc_val >= 0
        if val_stage2_mask.sum() > 0:
            val_pred = val_stage2_prob[val_stage2_mask].argmax(axis=1)
            val_f1_s2 = f1_score(
                yc_val[val_stage2_mask],
                val_pred,
                average="macro",
                zero_division=0,
            )
            try:
                val_auc_s2 = roc_auc_score(
                    yc_val[val_stage2_mask],
                    val_stage2_prob[val_stage2_mask],
                    multi_class="ovr",
                    average="macro",
                )
            except ValueError:
                val_auc_s2 = float("nan")
        else:
            val_f1_s2 = float("nan")
            val_auc_s2 = float("nan")

        return FoldArtifacts(
            fold_result={
                "history": {},
                "best_val_loss": float("nan"),
                "train_metrics": {
                    "auc_s1": roc_auc_score(yb_train, bp_train),
                    "auc_s2": train_auc_s2,
                    "f1_macro_s2": train_f1_s2,
                },
                "val_metrics": {
                    "auc_s1": roc_auc_score(yb_val, bp_val),
                    "auc_s2": val_auc_s2,
                    "f1_macro_s2": val_f1_s2,
                },
                "epochs": 1,
                "best_state": None,
            },
            val_binary_prob=bp_val,
            val_cancer_logits=np.log(np.clip(val_stage2_prob, 1e-8, 1.0)),
            val_embedding=X_val,
            train_last={
                "binary_prob": bp_train,
                "binary_true": yb_train,
                "cancer_logits": np.log(np.clip(train_stage2_prob, 1e-8, 1.0)),
                "cancer_true": yc_train,
                "embedding": X_train,
                "idx": train_idx,
            },
            extra={
                "binary_model": binary_model,
                "stage2_model": stage2_model,
                "present_classes": present_classes,
            },
        )

    def _log_fold_summary(self, fold_idx, artifacts):
        val_metrics = artifacts.fold_result["val_metrics"]
        logger.info(
            f"    Fold {fold_idx+1} {self.model_display_name} "
            f"S1_AUC={val_metrics['auc_s1']:.3f} "
            f"S2_F1={val_metrics['f1_macro_s2']:.3f} "
            f"S2_AUC={val_metrics['auc_s2']:.3f}"
        )

    def _save_fold_artifacts(self, fold_idx, train_idx, val_idx, artifacts):
        _save_classical_estimator(
            artifacts.extra["binary_model"],
            self.ckpt_dir / f"fold_{fold_idx}_binary.joblib",
        )
        if artifacts.extra["stage2_model"] is not None:
            _save_classical_estimator(
                artifacts.extra["stage2_model"],
                self.ckpt_dir / f"fold_{fold_idx}_stage2.joblib",
            )
        with open(self.ckpt_dir / f"fold_{fold_idx}_meta.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "fold": fold_idx,
                    "model_name": self.model_name,
                    "present_classes": artifacts.extra["present_classes"].tolist(),
                    "train_idx": train_idx.tolist(),
                    "val_idx": val_idx.tolist(),
                },
                f,
                indent=2,
            )


@singledispatch
def build_cv_runner(request):
    raise TypeError(f"Unsupported CV runner request: {type(request)!r}")


@build_cv_runner.register
def _(request: TorchRunnerRequest):
    return TorchCrossValidator(request)


@build_cv_runner.register
def _(request: ClassicalRunnerRequest):
    return ClassicalCrossValidator(request)


# =============================================================================
# 5. Cross-Validation
# =============================================================================
def run_torch_cv(X, binary_labels, cancer_type_labels, sample_ids, groups_arr,
                 config, device, model_name, runtime, n_splits=5, ckpt_dir=None):
    return build_cv_runner(
        TorchRunnerRequest(
            X=X,
            binary_labels=binary_labels,
            cancer_type_labels=cancer_type_labels,
            sample_ids=sample_ids,
            groups_arr=groups_arr,
            config=config,
            device=device,
            model_name=model_name,
            runtime=runtime,
            n_splits=n_splits,
            ckpt_dir=ckpt_dir,
        )
    ).run()

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=config.random_state)
    folds, histories = [], []

    n = len(X)
    val_bp = np.full(n, np.nan)
    val_cl = np.full((n, config.n_cancer_types), np.nan)
    val_emb = np.full((n, config.encoder_output_dim), np.nan)
    fold_ids = np.full(n, -1, dtype=int)
    train_last = {}

    for i, (ti, vi) in enumerate(cv.split(X, binary_labels, sample_ids)):
        logger.info(f"\n  ── Fold {i+1}/{n_splits} (train={len(ti)}, val={len(vi)}) ──")

        train_ds = SERSDataset(X[ti], binary_labels[ti], cancer_type_labels[ti],
                               groups=groups_arr[ti].tolist(),
                               sample_ids=[str(s) for s in sample_ids[ti]],
                               augment=True)
        val_ds = SERSDataset(X[vi], binary_labels[vi], cancer_type_labels[vi],
                             groups=groups_arr[vi].tolist(),
                             sample_ids=[str(s) for s in sample_ids[vi]],
                             augment=False)

        model = build_model(model_name, config).to(device)
        fr = train_fold(model, train_ds, val_ds, config, device, i, runtime)
        folds.append(fr); histories.append(fr["history"])

        vm = fr["val_metrics"]
        val_bp[vi] = vm["binary_prob"]
        val_cl[vi] = vm["cancer_logits"]
        val_emb[vi] = vm["embedding"]
        fold_ids[vi] = i

        tm = fr["train_metrics"]
        train_last = {
            "binary_prob": tm["binary_prob"], "binary_true": tm["binary_true"],
            "cancer_logits": tm["cancer_logits"], "cancer_true": tm["cancer_true"],
            "embedding": tm["embedding"], "idx": ti,
        }

        logger.info(
            f"    → Train S1={tm['auc_s1']:.3f} S2={tm['auc_s2']:.3f} │ "
            f"Val S1={vm['auc_s1']:.3f} S2={vm['auc_s2']:.3f}"
        )

        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save({
                "fold": i, "model_state_dict": fr["best_state"],
                "model_name": model_name,
                "config": config.__dict__,
                "train_idx": ti, "val_idx": vi,
            }, ckpt_dir / f"fold_{i}.pt")

    # ── Aggregate ──
    valid = ~np.isnan(val_bp)
    bp_v, bt_v = val_bp[valid], binary_labels[valid]
    cl_v, ct_v = val_cl[valid], cancer_type_labels[valid]

    bpred = (bp_v > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(bt_v, bpred, labels=[0, 1]).ravel()

    overall = {
        "val_s1_auc": float(roc_auc_score(bt_v, bp_v)),
        "val_s1_pr_auc": float(average_precision_score(bt_v, bp_v)),
        "val_s1_accuracy": float(accuracy_score(bt_v, bpred)),
        "val_s1_sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0,
        "val_s1_specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0,
        "val_s1_f1": float(f1_score(bt_v, bpred)),
    }

    cm = ct_v >= 0
    if cm.sum() > 0:
        cp = torch.softmax(torch.tensor(cl_v[cm]), dim=-1).numpy()
        cpred = cp.argmax(axis=1)
        overall["val_s2_accuracy"] = float(accuracy_score(ct_v[cm], cpred))
        try:
            overall["val_s2_auc"] = float(roc_auc_score(ct_v[cm], cp, multi_class="ovr", average="macro"))
        except ValueError:
            overall["val_s2_auc"] = float("nan")
        overall["val_s2_f1_macro"] = float(f1_score(ct_v[cm], cpred, average="macro", zero_division=0))
        try:
            per_class_ap = []
            for cls_idx in np.unique(ct_v[cm]):
                cls_bg = (ct_v[cm] == cls_idx).astype(int)
                per_class_ap.append(average_precision_score(cls_bg, cp[:, int(cls_idx)]))
            overall["val_s2_pr_auc"] = float(np.mean(per_class_ap))
        except (ValueError, IndexError):
            overall["val_s2_pr_auc"] = float("nan")

    if train_last:
        tbp, tbt = train_last["binary_prob"], train_last["binary_true"]
        overall["train_s1_auc"] = float(roc_auc_score(tbt, tbp))
        overall["train_s1_pr_auc"] = float(average_precision_score(tbt, tbp))
        tcm = train_last["cancer_true"] >= 0
        if tcm.sum() > 0:
            tcp = torch.softmax(torch.tensor(train_last["cancer_logits"][tcm]), dim=-1).numpy()
            try:
                overall["train_s2_auc"] = float(roc_auc_score(train_last["cancer_true"][tcm], tcp, multi_class="ovr", average="macro"))
            except ValueError:
                overall["train_s2_auc"] = float("nan")
            try:
                per_class_ap = []
                for cls_idx in np.unique(train_last["cancer_true"][tcm]):
                    cls_bg = (train_last["cancer_true"][tcm] == cls_idx).astype(int)
                    per_class_ap.append(average_precision_score(cls_bg, tcp[:, int(cls_idx)]))
                overall["train_s2_pr_auc"] = float(np.mean(per_class_ap))
            except (ValueError, IndexError):
                overall["train_s2_pr_auc"] = float("nan")

    logger.info("\n" + "=" * 64)
    logger.info("  CV Results (ResNet18-1D)")
    logger.info("=" * 64)
    for k, v in overall.items():
        logger.info(f"  {k:28s}: {v:.4f}")

    return {
        "overall": overall, "folds": folds, "histories": histories,
        "val_binary_prob": val_bp, "val_cancer_logits": val_cl,
        "val_embedding": val_emb, "fold_ids": fold_ids,
        "train_last": train_last,
        "binary_labels": binary_labels, "cancer_type_labels": cancer_type_labels,
        "groups": groups_arr, "sample_ids": sample_ids, "X": X,
    }


def _make_xgb_classifier(objective, random_state, xgb_params, num_class=None):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "xgboost is required for --model xgboost. Install it in the active environment."
        ) from exc

    params = {
        **xgb_params,
        "objective": objective,
        "random_state": random_state,
        "eval_metric": "logloss" if objective == "binary:logistic" else "mlogloss",
    }
    if num_class is not None:
        params["num_class"] = num_class
    return XGBClassifier(**params)


def _make_logistic_regression_classifier(random_state, logreg_params):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            **logreg_params,
            random_state=random_state,
        ),
    )


def _make_random_forest_classifier(random_state, rf_params):
    return RandomForestClassifier(
        **rf_params,
        random_state=random_state,
    )


def _build_classical_estimator(model_name, objective, random_state, params, num_class=None):
    if model_name == "xgboost":
        return _make_xgb_classifier(objective, random_state, params, num_class=num_class)
    if model_name == "logistic_regression":
        return _make_logistic_regression_classifier(random_state, params)
    if model_name == "random_forest":
        return _make_random_forest_classifier(random_state, params)
    raise ValueError(f"Unsupported classical model: {model_name}")


def _fit_classical_estimator(model_name, estimator, X, y, sample_weight=None):
    if model_name == "logistic_regression":
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["logisticregression__sample_weight"] = sample_weight
        estimator.fit(X, y, **fit_kwargs)
        return estimator

    if sample_weight is not None:
        estimator.fit(X, y, sample_weight=sample_weight)
    else:
        estimator.fit(X, y)
    return estimator


def _save_classical_estimator(estimator, path):
    joblib.dump(estimator, path)


def _binary_sample_weights(labels):
    labels = np.asarray(labels)
    n_total = max(len(labels), 1)
    n_pos = max(int((labels == 1).sum()), 1)
    n_neg = max(int((labels == 0).sum()), 1)
    pos_weight = n_total / (2.0 * n_pos)
    neg_weight = n_total / (2.0 * n_neg)
    return np.where(labels == 1, pos_weight, neg_weight).astype(np.float64)


def _multiclass_sample_weights(labels):
    labels = np.asarray(labels)
    counts = np.bincount(labels)
    counts[counts == 0] = 1
    n_total = max(len(labels), 1)
    n_classes = max(len(counts), 1)
    return np.array([n_total / (n_classes * counts[label]) for label in labels], dtype=np.float64)


def _expand_stage2_probabilities(probabilities, present_classes, n_classes):
    full = np.full((probabilities.shape[0], n_classes), 1.0 / n_classes, dtype=np.float64)
    for local_idx, class_idx in enumerate(present_classes):
        full[:, class_idx] = probabilities[:, local_idx]
    full_sum = full.sum(axis=1, keepdims=True)
    full_sum[full_sum == 0] = 1.0
    return full / full_sum


def run_classical_cv(model_name, X, binary_labels, cancer_type_labels, sample_ids, groups_arr,
                     config, model_params, n_splits=5, ckpt_dir=None):
    return build_cv_runner(
        ClassicalRunnerRequest(
            model_name=model_name,
            X=X,
            binary_labels=binary_labels,
            cancer_type_labels=cancer_type_labels,
            sample_ids=sample_ids,
            groups_arr=groups_arr,
            config=config,
            model_params=model_params,
            n_splits=n_splits,
            ckpt_dir=ckpt_dir,
        )
    ).run()



def run_xgboost_cv(X, binary_labels, cancer_type_labels, sample_ids, groups_arr,
                   config, xgb_params, n_splits=5, ckpt_dir=None):
    return run_classical_cv(
        "xgboost",
        X,
        binary_labels,
        cancer_type_labels,
        sample_ids,
        groups_arr,
        config,
        xgb_params,
        n_splits=n_splits,
        ckpt_dir=ckpt_dir,
    )


# =============================================================================
# 6. MLflow
# =============================================================================
def log_mlflow(results, config, args, model_name, model_display_name):
    try:
        import mlflow
        mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
        mlflow.set_experiment("sers-cancer-detection")
        with mlflow.start_run(run_name=f"{model_name}_{args.aggregate}_{args.n_splits}fold"):
            mlflow.log_params({
                "model_name": model_name, "model": model_display_name, "aggregate": args.aggregate,
                "n_splits": args.n_splits, "batch_size": config.batch_size,
                "lr": config.learning_rate, "epochs": config.n_epochs,
                "encoder_dim": config.encoder_output_dim,
                "resnet_channels": str(config.resnet_channels),
            })
            for k, v in results["overall"].items():
                if isinstance(v, float) and not np.isnan(v):
                    mlflow.log_metric(k, v)
    except Exception as e:
        logger.warning(f"MLflow logging failed: {e}")


def save_epoch_histories(results, out_dir):
    columns = [
        "fold", "epoch", "train_loss", "val_loss", "val_auc_s1", "val_auc_s2",
        "val_f1_macro_s2", "val_error_s1", "val_error_s2", "lr",
    ]
    rows = []
    for fold_idx, history in enumerate(results["histories"], start=1):
        n_epochs = len(history["val_loss"])
        for epoch_idx in range(n_epochs):
            rows.append(
                {
                    "fold": fold_idx,
                    "epoch": epoch_idx + 1,
                    "train_loss": history["train_loss"][epoch_idx],
                    "val_loss": history["val_loss"][epoch_idx],
                    "val_auc_s1": history["val_auc_s1"][epoch_idx],
                    "val_auc_s2": history["val_auc_s2"][epoch_idx],
                    "val_f1_macro_s2": history["val_f1_macro_s2"][epoch_idx],
                    "val_error_s1": history["val_error_s1"][epoch_idx],
                    "val_error_s2": history["val_error_s2"][epoch_idx],
                    "lr": history["lr"][epoch_idx],
                }
            )
    pd.DataFrame(rows, columns=columns).to_csv(out_dir / "epoch_metrics.csv", index=False)


def save_training_visualizations(results, out_dir):
    if not results["histories"]:
        logger.info("  No epoch-level histories available for this model. Skipping training curves.")
        return

    fig, axes = plt.subplots(2, 4, figsize=(22, 9))
    for i, history in enumerate(results["histories"], start=1):
        label = f"F{i}"
        axes[0, 0].plot(history["train_loss"], alpha=0.8, label=label)
        axes[0, 1].plot(history["val_loss"], alpha=0.8, label=label)
        axes[0, 2].plot(history["val_auc_s1"], alpha=0.8, label=label)
        axes[0, 3].plot(history["val_f1_macro_s2"], alpha=0.8, label=label)
        axes[1, 0].plot(history["val_auc_s2"], alpha=0.8, label=label)
        axes[1, 1].plot(history["val_error_s1"], alpha=0.8, label=label)
        axes[1, 2].plot(history["val_error_s2"], alpha=0.8, label=label)
        axes[1, 3].plot(history["lr"], alpha=0.8, label=label)

    titles = [
        "Train Loss",
        "Validation Loss",
        "Stage 1 Validation AUC",
        "Stage 2 Validation Macro F1",
        "Stage 2 Validation AUC",
        "Stage 1 Validation Error Rate",
        "Stage 2 Validation Error Rate",
        "Learning Rate",
    ]
    for ax, title in zip(axes.flatten(), titles):
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=150)
    plt.close()


def save_stage1_difference_artifacts(df, out_dir):
    if "group" not in df.columns:
        return
    try:
        peaks = plot_cancer_peak_difference(
            df,
            out_dir / "stage1_cancer_vs_non_cancer.png",
            group_col="group",
            top_k=20,
            title="Stage 1: Cancer vs Non-cancer Peak Difference",
        )
        peaks.to_csv(out_dir / "stage1_cancer_vs_non_cancer_peaks.csv", index=False)
    except Exception as exc:
        logger.warning(f"Stage 1 cancer/non-cancer difference plot failed: {exc}")


def save_benchmark_visualizations(benchmark_df, out_dir):
    if benchmark_df is None or benchmark_df.empty:
        return

    df = benchmark_df.copy()
    df = df.sort_values("val_s2_f1_macro", ascending=False).reset_index(drop=True)
    labels = [f"{row['model']}\n{row['version']}" for _, row in df.iterrows()]
    y = np.arange(len(df))

    panels = [
        ("val_s1_auc", "Stage 1 AUROC", "#287271"),
        ("val_s2_f1_macro", "Stage 2 Macro F1", "#c8553d"),
        ("val_s2_auc", "Stage 2 AUROC", "#4d9de0"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 7), sharey=True)
    for ax, (metric, title, color) in zip(axes, panels):
        values = df[metric].astype(float).to_numpy()
        ax.barh(y, values, color=color, alpha=0.9)
        ax.set_title(title)
        ax.set_xlim(max(0.0, np.nanmin(values) - 0.08), 1.0)
        ax.set_xlabel("Score")
        ax.grid(True, axis="x", alpha=0.25)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        for yi, val in zip(y, values):
            ax.text(min(val + 0.01, 0.995), yi, f"{val:.3f}", va="center", fontsize=9)

    axes[0].invert_yaxis()
    fig.suptitle("Benchmark Comparison Across Models", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "benchmark_comparison.png", dpi=150)
    plt.close(fig)

    gap_cols = ["train_s1_auc", "train_s2_f1_macro", "train_s2_auc"]
    if all(col in df.columns for col in gap_cols):
        gap_df = pd.DataFrame(
            {
                "model": labels,
                "stage1_auc_gap": df["train_s1_auc"] - df["val_s1_auc"],
                "stage2_f1_gap": df["train_s2_f1_macro"] - df["val_s2_f1_macro"],
                "stage2_auc_gap": df["train_s2_auc"] - df["val_s2_auc"],
            }
        )
        gap_df.to_csv(out_dir / "benchmark_train_val_gaps.csv", index=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(gap_df))
        w = 0.25
        ax.bar(x - w, gap_df["stage1_auc_gap"], width=w, color="#287271", label="S1 AUROC gap")
        ax.bar(x, gap_df["stage2_f1_gap"], width=w, color="#c8553d", label="S2 Macro F1 gap")
        ax.bar(x + w, gap_df["stage2_auc_gap"], width=w, color="#4d9de0", label="S2 AUROC gap")
        ax.set_xticks(x)
        ax.set_xticklabels(gap_df["model"], rotation=20, ha="right")
        ax.set_ylabel("Train - Val")
        ax.set_title("Benchmark Train-Val Gap")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "benchmark_train_val_gap.png", dpi=150)
        plt.close(fig)


def write_experiment_logs(out_dir, model_name, args, results, summary, extra=None):
    record = {
        "timestamp": summary["timestamp"],
        "experiment": summary.get("experiment"),
        "model_name": model_name,
        "model_display_name": summary["model"],
        "version": summary.get("version"),
        "aggregate": args.aggregate,
        "input": str(args.input),
        "output": str(out_dir),
        "n_splits": args.n_splits,
        "n_samples": summary["n_samples"],
        "n_features": summary["n_features"],
        "cancer_types": summary["cancer_types"],
        "non_cancer_groups": summary["non_cancer_groups"],
        "metrics": results["overall"],
        "metadata": {
            "hypothesis": getattr(args, "hypothesis", None),
            "variable": getattr(args, "variable", None),
            "baseline": getattr(args, "baseline", None),
            "tags": getattr(args, "tags", None),
            "phase": getattr(args, "phase", None),
        },
    }
    if extra:
        record.update(extra)

    with open(out_dir / "experiment_log.json", "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, default=str)

    run_log_dir = PROJECT_ROOT / "logs"
    run_log_dir.mkdir(parents=True, exist_ok=True)
    with open(run_log_dir / "experiment_runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def update_experiment_registry(experiment_name, args, cancer_types, non_cancer_groups,
                               n_samples, model_names, metrics_summary, artifacts_dir):
    """Append or update experiment entry in logs/experiment_registry.json."""
    registry_path = PROJECT_ROOT / "logs" / "experiment_registry.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    if registry_path.exists():
        with open(registry_path, encoding="utf-8") as f:
            registry = json.load(f)
    else:
        registry = {"experiments": []}

    entry = {
        "name": experiment_name,
        "phase": getattr(args, "phase", None),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "hypothesis": getattr(args, "hypothesis", None),
        "variable": getattr(args, "variable", None),
        "baseline": getattr(args, "baseline", None),
        "cancer_types": list(cancer_types),
        "non_cancer_groups": list(non_cancer_groups),
        "aggregation": args.aggregate,
        "models": model_names,
        "n_samples": n_samples,
        "tags": getattr(args, "tags", None),
        "result_summary": metrics_summary,
        "artifacts_dir": str(artifacts_dir),
    }

    # Update existing or append
    existing_idx = next((i for i, e in enumerate(registry["experiments"]) if e["name"] == experiment_name), None)
    if existing_idx is not None:
        registry["experiments"][existing_idx] = entry
    else:
        registry["experiments"].append(entry)

    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False, default=str)


def save_experiment_manifest(experiment_dir, experiment_name, args, model_rows, started_at, elapsed):
    manifest = {
        "experiment": experiment_name,
        "timestamp": started_at,
        "elapsed": str(elapsed),
        "input": str(args.input),
        "output_root": str(Path(args.output)),
        "experiment_dir": str(experiment_dir),
        "aggregate": args.aggregate,
        "n_splits": args.n_splits,
        "models": model_rows,
    }
    with open(experiment_dir / "experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)


def save_fold_predictions(results, out_dir):
    save_dict = {
        "X": results["X"],
        "binary_labels": results["binary_labels"],
        "cancer_type_labels": results["cancer_type_labels"],
        "groups": results["groups"],
        "sample_ids": results["sample_ids"],
        "val_binary_prob": results["val_binary_prob"],
        "val_cancer_logits": results["val_cancer_logits"],
        "val_embedding": results["val_embedding"],
        "fold_ids": results["fold_ids"],
    }
    tl = results["train_last"]
    if tl:
        save_dict.update({
            "train_binary_prob": tl["binary_prob"],
            "train_binary_true": tl["binary_true"],
            "train_cancer_logits": tl["cancer_logits"],
            "train_cancer_true": tl["cancer_true"],
            "train_embedding": tl["embedding"],
            "train_idx": tl["idx"],
        })
    np.savez_compressed(out_dir / "fold_predictions.npz", **save_dict)


def build_model_params_summary(model_name, config, args):
    if model_name == "xgboost":
        return resolve_xgboost_params(args)
    if model_name == "logistic_regression":
        return resolve_logreg_params(args)
    if model_name == "random_forest":
        return resolve_random_forest_params(args)
    return {
        "resnet_channels": list(config.resnet_channels),
        "resnet_blocks": list(config.resnet_blocks),
        "encoder_output_dim": config.encoder_output_dim,
        "head_hidden_dim": config.head_hidden_dim,
        "dropout_rate": config.dropout_rate,
        "use_focal_loss": config.use_focal_loss,
        "focal_gamma": config.focal_gamma,
        "class_balance_beta": config.class_balance_beta,
    }


def build_training_summary(results, config, args, model_name, model_display_name, n_samples, n_features, timestamp,
                           version_name, output_dir, experiment_name=None, model_params=None):
    return {
        "timestamp": timestamp,
        "experiment": experiment_name,
        "model_name": model_name,
        "model": model_display_name,
        "version": version_name,
        "output_dir": str(output_dir),
        "aggregate": args.aggregate,
        "n_splits": args.n_splits,
        "n_samples": n_samples,
        "n_features": n_features,
        "cancer_types": list(config.cancer_types),
        "non_cancer_groups": list(config.non_cancer_groups),
        "group_aliases": config.group_aliases,
        "metrics": results["overall"],
        "config": {
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "n_epochs": config.n_epochs,
            "weight_decay": config.weight_decay,
            "dropout_rate": config.dropout_rate,
            "resnet_channels": list(config.resnet_channels),
            "resnet_blocks": list(config.resnet_blocks),
            "encoder_output_dim": config.encoder_output_dim,
            "head_hidden_dim": config.head_hidden_dim,
            "stage1_loss_weight": config.stage1_loss_weight,
            "stage2_loss_weight": config.stage2_loss_weight,
            "use_focal_loss": config.use_focal_loss,
            "focal_gamma": config.focal_gamma,
            "class_balance_beta": config.class_balance_beta,
            "early_stopping_patience": config.early_stopping_patience,
            "scheduler_patience": config.scheduler_patience,
            "random_state": config.random_state,
            "num_workers": args.num_workers,
            "prefetch_factor": args.prefetch_factor,
            "use_amp": not args.no_amp,
            "selected_cancer_types": list(config.cancer_types),
            "selected_non_cancer_groups": list(config.non_cancer_groups),
        },
        "model_params": model_params or build_model_params_summary(model_name, config, args),
        "metadata": {
            "hypothesis": getattr(args, "hypothesis", None),
            "variable": getattr(args, "variable", None),
            "baseline": getattr(args, "baseline", None),
            "tags": getattr(args, "tags", None),
            "phase": getattr(args, "phase", None),
        },
    }


def save_training_summary(summary, out_dir):
    with open(out_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)


def save_fold_metrics(results, out_dir):
    rows = []
    for i, fr in enumerate(results["folds"]):
        tm, vm = fr["train_metrics"], fr["val_metrics"]
        rows.append({
            "fold": i + 1,
            "epochs": fr["epochs"],
            "best_loss": fr["best_val_loss"],
            "train_auc_s1": tm.get("auc_s1", np.nan),
            "train_pr_auc_s1": tm.get("pr_auc_s1", np.nan),
            "train_auc_s2": tm.get("auc_s2", np.nan),
            "train_pr_auc_s2": tm.get("pr_auc_s2", np.nan),
            "train_f1_macro_s2": tm.get("f1_macro_s2", np.nan),
            "val_auc_s1": vm.get("auc_s1", np.nan),
            "val_pr_auc_s1": vm.get("pr_auc_s1", np.nan),
            "val_auc_s2": vm.get("auc_s2", np.nan),
            "val_pr_auc_s2": vm.get("pr_auc_s2", np.nan),
            "val_f1_macro_s2": vm.get("f1_macro_s2", np.nan),
        })
    pd.DataFrame(rows).to_csv(out_dir / "fold_metrics.csv", index=False)


def save_classification_reports(results, binary_labels, cancer_type_labels, config, out_dir):
    valid = ~np.isnan(results["val_binary_prob"])
    bpred = (results["val_binary_prob"][valid] > 0.5).astype(int)
    with open(out_dir / "report_stage1.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(binary_labels[valid], bpred, target_names=["Non-cancer", "Cancer"]))

    ct_v, cl_v = cancer_type_labels[valid], results["val_cancer_logits"][valid]
    cancer_mask = ct_v >= 0
    if cancer_mask.sum() == 0:
        return

    cp = torch.softmax(torch.tensor(cl_v[cancer_mask]), dim=-1).numpy()
    cpred = cp.argmax(axis=1)
    present = sorted(set(ct_v[cancer_mask]))
    names = [config.cancer_types[j] for j in present]
    with open(out_dir / "report_stage2.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(ct_v[cancer_mask], cpred, labels=present, target_names=names))


def save_run_artifacts(results, summary, config, args, out_dir, df_valid, binary_labels, cancer_type_labels):
    logger.info("\n[Step 6] Saving...")
    save_fold_predictions(results, out_dir)
    save_training_summary(summary, out_dir)
    save_fold_metrics(results, out_dir)
    save_epoch_histories(results, out_dir)
    save_classification_reports(results, binary_labels, cancer_type_labels, config, out_dir)
    # Save figures to centralized directory
    from src.sers.config import FIG_DIR, training_dir_to_figure_slug
    fig_dir = FIG_DIR / "training" / training_dir_to_figure_slug(out_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    save_training_visualizations(results, fig_dir)
    save_stage1_difference_artifacts(df_valid, fig_dir)
    write_experiment_logs(
        out_dir,
        summary["model_name"],
        args,
        results,
        summary,
        extra={"has_epoch_history": bool(results["histories"])},
    )


def run_single_model(model_name, out_dir, version_name, experiment_name, X, binary_labels, cancer_type_labels, sample_ids, groups_arr,
                     df_valid, config, device, args, timestamp):
    model_display_name = MODEL_DISPLAY_NAMES[model_name]
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model_params = build_model_params_summary(model_name, config, args)

    logger.info("\n" + "=" * 64)
    logger.info(f"  Model: {model_display_name}")
    logger.info("=" * 64)
    runtime = resolve_runtime_config(args, device)
    if model_name in TORCH_MODEL_NAMES:
        logger.info(
            f"  Runtime: amp={runtime['use_amp']} num_workers={runtime['num_workers']} "
            f"pin_memory={runtime['pin_memory']}"
        )

    if model_name in CLASSICAL_MODEL_NAMES:
        results = run_classical_cv(
            model_name,
            X,
            binary_labels,
            cancer_type_labels,
            sample_ids,
            groups_arr,
            config,
            model_params,
            n_splits=args.n_splits,
            ckpt_dir=ckpt_dir,
        )
    else:
        summary_text = model_summary(build_model(model_name, config)).replace("\u2014", "-")
        logger.info(f"\n{summary_text}")
        results = run_torch_cv(
            X,
            binary_labels,
            cancer_type_labels,
            sample_ids,
            groups_arr,
            config,
            device,
            model_name=model_name,
            runtime=runtime,
            n_splits=args.n_splits,
            ckpt_dir=ckpt_dir,
        )

    summary = build_training_summary(
        results,
        config,
        args,
        model_name=model_name,
        model_display_name=model_display_name,
        n_samples=len(X),
        n_features=X.shape[1],
        timestamp=timestamp,
        version_name=version_name,
        output_dir=out_dir,
        experiment_name=experiment_name,
        model_params=model_params,
    )
    save_run_artifacts(results, summary, config, args, out_dir, df_valid, binary_labels, cancer_type_labels)

    if not args.no_mlflow:
        log_mlflow(results, config, args, model_name, model_display_name)

    return results, summary


# =============================================================================
# 7. Main
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="SERS ResNet18-1D Training")
    p.add_argument("--input", "-i", default="results/processed_spectra.csv")
    p.add_argument("--output", "-o", default="results/training")
    p.add_argument("--config", "-c", default="config/config.yaml")
    model_choices = ["logistic_regression", "random_forest", "xgboost", "cnn1d", "resnet18"]
    p.add_argument("--model", choices=model_choices, default="resnet18")
    p.add_argument("--benchmark-models", nargs="+", choices=model_choices, default=None)
    p.add_argument("--experiment", default=None,
                   help="Experiment label under the output root. Defaults to auto-increment like experiment_001.")
    p.add_argument("--version", default=None,
                   help="Version label to store under each model directory. Defaults to auto-increment like v001.")
    p.add_argument("--cancer-types", nargs="+", default=None,
                   help="Subset of cancer classes to include, e.g. PRO BRE CRC")
    p.add_argument("--non-cancer-groups", nargs="+", default=None,
                   help="Subset of non-cancer groups to include, e.g. NOR DIA")
    p.add_argument("--aggregate", "-a", choices=["medoid", "mean", "none"], default="medoid")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--dropout-rate", type=float, default=None)
    p.add_argument("--stage2-loss-weight", type=float, default=None)
    p.add_argument("--head-hidden-dim", type=int, default=None)
    p.add_argument("--resnet-channels", nargs="+", type=int, default=None,
                   help="Override ResNet channels, e.g. --resnet-channels 16 32 64 128")
    p.add_argument("--use-focal-loss", action="store_true")
    p.add_argument("--focal-gamma", type=float, default=None)
    p.add_argument("--class-balance-beta", type=float, default=None,
                   help="Effective-number beta for stage2 class-balanced loss, e.g. 0.999")
    p.add_argument("--logreg-c", type=float, default=None)
    p.add_argument("--logreg-max-iter", type=int, default=None)
    p.add_argument("--rf-n-estimators", type=int, default=None)
    p.add_argument("--rf-max-depth", type=int, default=None)
    p.add_argument("--rf-min-samples-leaf", type=int, default=None)
    p.add_argument("--xgb-n-estimators", type=int, default=None)
    p.add_argument("--xgb-max-depth", type=int, default=None)
    p.add_argument("--xgb-learning-rate", type=float, default=None)
    p.add_argument("--xgb-subsample", type=float, default=None)
    p.add_argument("--xgb-colsample-bytree", type=float, default=None)
    p.add_argument("--xgb-reg-lambda", type=float, default=None)
    p.add_argument("--xgb-min-child-weight", type=float, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-mlflow", action="store_true")
    # ── Experiment metadata ──
    p.add_argument("--hypothesis", default=None,
                   help="What this experiment tests, e.g. 'Adding BRE maintains detection AUC'")
    p.add_argument("--variable", default=None,
                   help="Independent variable, e.g. 'cancer_set', 'aggregation', 'model_type'")
    p.add_argument("--baseline", default=None,
                   help="Reference experiment to compare against, e.g. 'V-6cancer'")
    p.add_argument("--tags", nargs="+", default=None,
                   help="Searchable tags, e.g. 'ablation', 'production', 'exploratory'")
    p.add_argument("--phase", default=None,
                   help="Phase letter for milestone experiments, e.g. 'U'")
    p.add_argument("--exclude-patients", default=None,
                   help="CSV file with (group, sample_id) columns to exclude from training")
    return p.parse_args()


def main():
    args = parse_args()
    t0 = datetime.now()
    out_root = Path(args.output)
    experiment_dir, experiment_name = resolve_experiment_output_dir(out_root, args.experiment)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-7s │ %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(experiment_dir / "train.log", mode="w", encoding="utf-8")],
    )

    logger.info("=" * 64)
    logger.info("  SERS Two-Stage Training")
    logger.info("=" * 64)
    logger.info(f"  Experiment: {experiment_name}")
    logger.info(f"  Output dir: {experiment_dir}")

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

    try:
        import yaml
        with open(args.config, encoding="utf-8") as f:
            raw_cfg = yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"  {args.config} not found, using defaults")
        raw_cfg = {}

    logger.info("\n[Step 1] Loading data...")
    df = load_processed_spectra(args.input)
    feat_cols = get_feature_columns(df)
    n_feat = len(feat_cols)

    mc = ModelConfig.from_pipeline_config(raw_cfg, n_spectral_features=n_feat) if raw_cfg else ModelConfig(n_spectral_features=n_feat)
    mc = apply_training_overrides(mc, args)
    mc = apply_class_selection(
        mc,
        cancer_types=args.cancer_types,
        non_cancer_groups=args.non_cancer_groups,
    )

    logger.info(f"  Cancer types ({mc.n_cancer_types}): {mc.cancer_types}")
    logger.info(f"  Non-cancer: {mc.non_cancer_groups}")
    logger.info(f"  ResNet channels: {mc.resnet_channels}, blocks: {mc.resnet_blocks}")

    logger.info("\n[Step 2] Resolving aliases...")
    df = resolve_aliases(df, mc)

    if args.exclude_patients:
        logger.info("\n[Step 2b] Applying patient exclusions...")
        df = apply_patient_exclusions(df, args.exclude_patients)

    logger.info("\n[Step 3] Aggregating replicates...")
    df_agg = aggregate_replicates(df, feat_cols, args.aggregate)

    logger.info("\n[Step 4] Creating labels...")
    X, bl, ctl, sample_ids, groups_arr = create_labels(df_agg, mc)

    valid_groups = set(mc.cancer_types) | set(mc.non_cancer_groups)
    df_valid = df_agg[df_agg["group"].isin(valid_groups)].copy()

    model_names = args.benchmark_models if args.benchmark_models else [args.model]
    benchmark_rows = []
    benchmark_dir = experiment_dir / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    for model_name in model_names:
        run_out_dir, version_name = resolve_run_output_dir(experiment_dir, model_name, args.version)
        logger.info(f"\n[Step 5] {args.n_splits}-fold CV for {MODEL_DISPLAY_NAMES[model_name]} ({version_name})...")
        try:
            results, summary = run_single_model(
                model_name,
                run_out_dir,
                version_name,
                experiment_name,
                X,
                bl,
                ctl,
                sample_ids,
                groups_arr,
                df_valid,
                mc,
                device,
                args,
                t0.isoformat(),
            )
            row = {
                "experiment": experiment_name,
                "status": "completed",
                "model_name": model_name,
                "model": summary["model"],
                "version": version_name,
                "output": str(run_out_dir),
                "aggregate": args.aggregate,
                "n_samples": summary["n_samples"],
                "n_features": summary["n_features"],
            }
            row.update(summary["metrics"])
            benchmark_rows.append(row)
        except Exception as exc:
            logger.exception(f"  Model run failed for {MODEL_DISPLAY_NAMES[model_name]}: {exc}")
            benchmark_rows.append(
                {
                    "experiment": experiment_name,
                    "status": "failed",
                    "model_name": model_name,
                    "model": MODEL_DISPLAY_NAMES[model_name],
                    "version": version_name,
                    "output": str(run_out_dir),
                    "aggregate": args.aggregate,
                    "error": str(exc),
                }
            )
            if len(model_names) == 1:
                raise

    if args.benchmark_models and benchmark_rows:
        benchmark_df = pd.DataFrame(benchmark_rows)
        benchmark_df.to_csv(benchmark_dir / "benchmark_summary.csv", index=False)
        completed_df = benchmark_df[benchmark_df["status"] == "completed"].copy()
        if not completed_df.empty:
            save_benchmark_visualizations(completed_df, benchmark_dir)

    elapsed = datetime.now() - t0
    save_experiment_manifest(experiment_dir, experiment_name, args, benchmark_rows, t0.isoformat(), elapsed)

    # Update experiment registry
    metrics_summary = ""
    if benchmark_rows:
        best = benchmark_rows[0]
        metrics_summary = f"Det AUC {best.get('val_s1_auc', 'N/A')}, Id F1 {best.get('val_s2_f1_macro', 'N/A')}"
    update_experiment_registry(
        experiment_name=experiment_name, args=args,
        cancer_types=mc.cancer_types, non_cancer_groups=mc.non_cancer_groups,
        n_samples=len(X), model_names=model_names,
        metrics_summary=metrics_summary, artifacts_dir=experiment_dir,
    )

    logger.info(f"\n{'=' * 64}")
    logger.info(f"  Training complete! ({elapsed})")
    logger.info(f"  Output: {experiment_dir}/")
    if args.benchmark_models:
        logger.info(f"  Benchmark summary: {benchmark_dir / 'benchmark_summary.csv'}")
        logger.info(f"  Next: python models/test.py -i {experiment_dir / model_names[0]}")
    else:
        logger.info(f"  Next: python models/test.py -i {experiment_dir / model_names[0]}")
    logger.info(f"{'=' * 64}")
    return 0

    logger.info(f"\n{model_summary(SERSCancerDetector(mc))}")

    logger.info(f"\n[Step 5] {args.n_splits}-fold CV...")
    results = run_cv(X, bl, ctl, sample_ids, groups_arr, mc, device,
                     args.n_splits, ckpt_dir=ckpt_dir)

    # ── Save ──
    logger.info("\n[Step 6] Saving...")

    save_dict = {
        "X": results["X"],
        "binary_labels": results["binary_labels"],
        "cancer_type_labels": results["cancer_type_labels"],
        "groups": results["groups"], "sample_ids": results["sample_ids"],
        "val_binary_prob": results["val_binary_prob"],
        "val_cancer_logits": results["val_cancer_logits"],
        "val_embedding": results["val_embedding"],
        "fold_ids": results["fold_ids"],
    }
    tl = results["train_last"]
    if tl:
        save_dict.update({
            "train_binary_prob": tl["binary_prob"], "train_binary_true": tl["binary_true"],
            "train_cancer_logits": tl["cancer_logits"], "train_cancer_true": tl["cancer_true"],
            "train_embedding": tl["embedding"], "train_idx": tl["idx"],
        })
    np.savez_compressed(out_dir / "fold_predictions.npz", **save_dict)

    summary = {
        "timestamp": t0.isoformat(), "model": "ResNet18-1D",
        "aggregate": args.aggregate, "n_splits": args.n_splits,
        "n_samples": len(X), "n_features": n_feat,
        "cancer_types": list(mc.cancer_types),
        "non_cancer_groups": list(mc.non_cancer_groups),
        "group_aliases": mc.group_aliases,
        "metrics": results["overall"],
        "config": {
            "learning_rate": mc.learning_rate, "batch_size": mc.batch_size,
            "n_epochs": mc.n_epochs, "dropout_rate": mc.dropout_rate,
            "resnet_channels": list(mc.resnet_channels),
            "resnet_blocks": list(mc.resnet_blocks),
            "encoder_output_dim": mc.encoder_output_dim,
            "head_hidden_dim": mc.head_hidden_dim,
            "stage1_loss_weight": mc.stage1_loss_weight,
            "stage2_loss_weight": mc.stage2_loss_weight,
        },
    }
    with open(out_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    fold_data = []
    for i, fr in enumerate(results["folds"]):
        tm, vm = fr["train_metrics"], fr["val_metrics"]
        fold_data.append({
            "fold": i+1, "epochs": fr["epochs"], "best_loss": fr["best_val_loss"],
            "train_auc_s1": tm["auc_s1"], "train_auc_s2": tm["auc_s2"],
            "train_f1_macro_s2": tm["f1_macro_s2"],
            "val_auc_s1": vm["auc_s1"], "val_auc_s2": vm["auc_s2"],
            "val_f1_macro_s2": vm["f1_macro_s2"],
        })
    pd.DataFrame(fold_data).to_csv(out_dir / "fold_metrics.csv", index=False)
    save_epoch_histories(results, out_dir)

    # Classification reports
    valid = ~np.isnan(results["val_binary_prob"])
    bpred = (results["val_binary_prob"][valid] > 0.5).astype(int)
    with open(out_dir / "report_stage1.txt", "w") as f:
        f.write(classification_report(bl[valid], bpred, target_names=["Non-cancer", "Cancer"]))

    ct_v, cl_v = ctl[valid], results["val_cancer_logits"][valid]
    cm = ct_v >= 0
    if cm.sum() > 0:
        cp = torch.softmax(torch.tensor(cl_v[cm]), dim=-1).numpy()
        cpred = cp.argmax(axis=1)
        present = sorted(set(ct_v[cm]))
        names = [mc.cancer_types[j] for j in present]
        with open(out_dir / "report_stage2.txt", "w") as f:
            f.write(classification_report(ct_v[cm], cpred, labels=present, target_names=names))

    from src.sers.config import FIG_DIR, training_dir_to_figure_slug
    fig_dir = FIG_DIR / "training" / training_dir_to_figure_slug(out_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    save_training_visualizations(results, fig_dir)

    if not args.no_mlflow:
        log_mlflow(results, mc, args)

    elapsed = datetime.now() - t0
    logger.info(f"\n{'=' * 64}")
    logger.info(f"  Training complete! ({elapsed})")
    logger.info(f"  Output: {out_dir}/")
    logger.info(f"  Next: python test.py -i {out_dir}")
    logger.info(f"{'=' * 64}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
