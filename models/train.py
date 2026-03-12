"""
SERS Cancer Detection — Training Pipeline (ResNet18-1D)

Two-stage hierarchical: Binary + Cancer Type (7: incl CPAN, SPAN separate)

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
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
)
from torch.utils.data import WeightedRandomSampler

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
    "resnet18": "ResNet18-1D",
    "cnn1d": "CNN1D-Shallow",
    "xgboost": "XGBoost-Hierarchical",
}

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

    return ModelConfig(
        **{
            **config.__dict__,
            "cancer_types": selected_cancer,
            "cancer_groups_raw": selected_cancer,
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


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        spec = batch["spectra"].to(device)
        by = batch["binary_label"].to(device)
        cy = batch["cancer_type_label"].to(device)
        optimizer.zero_grad()
        out = model(spec)
        losses = criterion(out, by, cy)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += losses["total"].item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total, n = 0.0, 0
    all_bp, all_bt, all_cl, all_ct, all_emb = [], [], [], [], []

    for batch in loader:
        spec = batch["spectra"].to(device)
        by = batch["binary_label"].to(device)
        cy = batch["cancer_type_label"].to(device)
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
    auc_s2 = float("nan")
    acc_s2 = float("nan")
    f1_macro_s2 = float("nan")
    if cm.sum() > 0 and len(np.unique(ct[cm])) > 1:
        cp = torch.softmax(torch.tensor(cl[cm]), dim=-1).numpy()
        cpred = cp.argmax(axis=1)
        acc_s2 = accuracy_score(ct[cm], cpred)
        f1_macro_s2 = f1_score(ct[cm], cpred, average="macro", zero_division=0)
        try:
            auc_s2 = roc_auc_score(ct[cm], cp, multi_class="ovr", average="macro")
        except ValueError:
            pass

    return {
        "loss": total / max(n, 1),
        "auc_s1": auc_s1,
        "auc_s2": auc_s2,
        "acc_s1": acc_s1,
        "acc_s2": acc_s2,
        "f1_macro_s2": f1_macro_s2,
        "binary_prob": bp, "binary_true": bt,
        "cancer_logits": cl, "cancer_true": ct,
        "embedding": emb,
    }


def train_fold(model, train_ds, val_ds, config, device, fold_i):
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
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)

    # Class weights for loss
    pw = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)

    cm = ctl >= 0
    cw_t = None
    if cm.sum() > 0:
        cc = np.bincount(ctl[cm], minlength=config.n_cancer_types).astype(float)
        cw = 1.0 / np.maximum(cc, 1)
        cw = cw / cw.sum() * config.n_cancer_types
        cw_t = torch.tensor(cw, dtype=torch.float32).to(device)

    criterion = TwoStageLoss(
        config.stage1_loss_weight, config.stage2_loss_weight, pw, cw_t,
        label_smoothing=0.1,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=config.scheduler_patience, factor=0.5, min_lr=1e-6
    )
    es = EarlyStopping(config.early_stopping_patience)

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
        tl = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vm = evaluate(model, val_loader, criterion, device)
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

    train_metrics = evaluate(model, train_loader, criterion, device)
    val_metrics = evaluate(model, val_loader, criterion, device)

    return {
        "history": history, "best_val_loss": best_val_loss,
        "train_metrics": train_metrics, "val_metrics": val_metrics,
        "epochs": epoch + 1, "best_state": best_state,
    }


# =============================================================================
# 5. Cross-Validation
# =============================================================================
def run_torch_cv(X, binary_labels, cancer_type_labels, sample_ids, groups_arr,
                 config, device, model_name, n_splits=5, ckpt_dir=None):

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
        fr = train_fold(model, train_ds, val_ds, config, device, i)
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

    if train_last:
        tbp, tbt = train_last["binary_prob"], train_last["binary_true"]
        overall["train_s1_auc"] = float(roc_auc_score(tbt, tbp))
        tcm = train_last["cancer_true"] >= 0
        if tcm.sum() > 0:
            tcp = torch.softmax(torch.tensor(train_last["cancer_logits"][tcm]), dim=-1).numpy()
            try:
                overall["train_s2_auc"] = float(roc_auc_score(train_last["cancer_true"][tcm], tcp, multi_class="ovr", average="macro"))
            except ValueError:
                overall["train_s2_auc"] = float("nan")

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


def _make_xgb_classifier(objective, random_state, num_class=None):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "xgboost is required for --model xgboost. Install it in the active environment."
        ) from exc

    params = {
        **XGBOOST_DEFAULT_PARAMS,
        "objective": objective,
        "random_state": random_state,
        "eval_metric": "logloss" if objective == "binary:logistic" else "mlogloss",
    }
    if num_class is not None:
        params["num_class"] = num_class
    return XGBClassifier(**params)


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


def run_xgboost_cv(X, binary_labels, cancer_type_labels, sample_ids, groups_arr,
                   config, n_splits=5, ckpt_dir=None):
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=config.random_state)
    folds, histories = [], []

    n = len(X)
    val_bp = np.full(n, np.nan)
    val_cl = np.full((n, config.n_cancer_types), np.nan)
    val_emb = np.full((n, X.shape[1]), np.nan)
    fold_ids = np.full(n, -1, dtype=int)
    train_last = {}

    for i, (ti, vi) in enumerate(cv.split(X, binary_labels, sample_ids)):
        logger.info(f"\n  -- Fold {i+1}/{n_splits} (train={len(ti)}, val={len(vi)}) --")

        X_train, X_val = X[ti], X[vi]
        yb_train, yb_val = binary_labels[ti], binary_labels[vi]
        yc_train, yc_val = cancer_type_labels[ti], cancer_type_labels[vi]

        binary_model = _make_xgb_classifier("binary:logistic", config.random_state + i)
        binary_model.fit(X_train, yb_train, sample_weight=_binary_sample_weights(yb_train))

        bp_train = binary_model.predict_proba(X_train)[:, 1]
        bp_val = binary_model.predict_proba(X_val)[:, 1]

        cancer_train_mask = yc_train >= 0
        stage2_model = None
        present_classes = np.array([], dtype=int)
        train_stage2_prob = np.full((len(X_train), config.n_cancer_types), 1.0 / config.n_cancer_types)
        val_stage2_prob = np.full((len(X_val), config.n_cancer_types), 1.0 / config.n_cancer_types)

        if cancer_train_mask.sum() > 0:
            present_classes = np.unique(yc_train[cancer_train_mask])
            if len(present_classes) == 1:
                train_stage2_prob = np.zeros((len(X_train), config.n_cancer_types), dtype=np.float64)
                val_stage2_prob = np.zeros((len(X_val), config.n_cancer_types), dtype=np.float64)
                train_stage2_prob[:, present_classes[0]] = 1.0
                val_stage2_prob[:, present_classes[0]] = 1.0
            else:
                local_labels = np.searchsorted(present_classes, yc_train[cancer_train_mask])
                stage2_model = _make_xgb_classifier(
                    "multi:softprob",
                    config.random_state + i + 100,
                    num_class=len(present_classes),
                )
                stage2_model.fit(
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
                    config.n_cancer_types,
                )
                val_stage2_prob = _expand_stage2_probabilities(
                    np.asarray(val_stage2_local),
                    present_classes,
                    config.n_cancer_types,
                )

        val_bp[vi] = bp_val
        val_cl[vi] = np.log(np.clip(val_stage2_prob, 1e-8, 1.0))
        val_emb[vi] = X_val
        fold_ids[vi] = i

        train_last = {
            "binary_prob": bp_train,
            "binary_true": yb_train,
            "cancer_logits": np.log(np.clip(train_stage2_prob, 1e-8, 1.0)),
            "cancer_true": yc_train,
            "embedding": X_train,
            "idx": ti,
        }

        tm_stage2_mask = yc_train >= 0
        if tm_stage2_mask.sum() > 0:
            train_cpred = train_stage2_prob[tm_stage2_mask].argmax(axis=1)
            train_f1_s2 = f1_score(yc_train[tm_stage2_mask], train_cpred, average="macro", zero_division=0)
            try:
                train_auc_s2 = roc_auc_score(
                    yc_train[tm_stage2_mask],
                    train_stage2_prob[tm_stage2_mask],
                    multi_class="ovr",
                    average="macro",
                )
            except ValueError:
                train_auc_s2 = float("nan")
        else:
            train_f1_s2 = float("nan")
            train_auc_s2 = float("nan")

        vm_stage2_mask = yc_val >= 0
        if vm_stage2_mask.sum() > 0:
            cpred = val_stage2_prob[vm_stage2_mask].argmax(axis=1)
            val_f1_s2 = f1_score(yc_val[vm_stage2_mask], cpred, average="macro", zero_division=0)
            try:
                val_auc_s2 = roc_auc_score(
                    yc_val[vm_stage2_mask],
                    val_stage2_prob[vm_stage2_mask],
                    multi_class="ovr",
                    average="macro",
                )
            except ValueError:
                val_auc_s2 = float("nan")
        else:
            val_f1_s2 = float("nan")
            val_auc_s2 = float("nan")

        logger.info(
            f"    Fold {i+1} XGB "
            f"S1_AUC={roc_auc_score(yb_val, bp_val):.3f} "
            f"S2_F1={val_f1_s2:.3f} "
            f"S2_AUC={val_auc_s2:.3f}"
        )

        folds.append(
            {
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
            }
        )

        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)
            binary_model.get_booster().save_model(str(ckpt_dir / f"fold_{i}_binary.json"))
            if stage2_model is not None:
                stage2_model.get_booster().save_model(str(ckpt_dir / f"fold_{i}_stage2.json"))
            with open(ckpt_dir / f"fold_{i}_meta.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "fold": i,
                        "model_name": "xgboost",
                        "present_classes": present_classes.tolist(),
                        "train_idx": ti.tolist(),
                        "val_idx": vi.tolist(),
                    },
                    f,
                    indent=2,
                )

    valid = ~np.isnan(val_bp)
    bp_v, bt_v = val_bp[valid], binary_labels[valid]
    cl_v, ct_v = val_cl[valid], cancer_type_labels[valid]

    bpred = (bp_v > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(bt_v, bpred, labels=[0, 1]).ravel()
    overall = {
        "val_s1_auc": float(roc_auc_score(bt_v, bp_v)),
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
        overall["val_s2_f1_macro"] = float(f1_score(ct_v[cm], cpred, average="macro", zero_division=0))
        try:
            overall["val_s2_auc"] = float(roc_auc_score(ct_v[cm], cp, multi_class="ovr", average="macro"))
        except ValueError:
            overall["val_s2_auc"] = float("nan")

    if train_last:
        tbp, tbt = train_last["binary_prob"], train_last["binary_true"]
        overall["train_s1_auc"] = float(roc_auc_score(tbt, tbp))
        tcm = train_last["cancer_true"] >= 0
        if tcm.sum() > 0:
            tcp = torch.softmax(torch.tensor(train_last["cancer_logits"][tcm]), dim=-1).numpy()
            overall["train_s2_f1_macro"] = float(
                f1_score(train_last["cancer_true"][tcm], tcp.argmax(axis=1), average="macro", zero_division=0)
            )
            try:
                overall["train_s2_auc"] = float(
                    roc_auc_score(train_last["cancer_true"][tcm], tcp, multi_class="ovr", average="macro")
                )
            except ValueError:
                overall["train_s2_auc"] = float("nan")

    logger.info("\n" + "=" * 64)
    logger.info("  CV Results (XGBoost-Hierarchical)")
    logger.info("=" * 64)
    for k, v in overall.items():
        logger.info(f"  {k:28s}: {v:.4f}")

    return {
        "overall": overall,
        "folds": folds,
        "histories": histories,
        "val_binary_prob": val_bp,
        "val_cancer_logits": val_cl,
        "val_embedding": val_emb,
        "fold_ids": fold_ids,
        "train_last": train_last,
        "binary_labels": binary_labels,
        "cancer_type_labels": cancer_type_labels,
        "groups": groups_arr,
        "sample_ids": sample_ids,
        "X": X,
    }


# =============================================================================
# 6. MLflow
# =============================================================================
def log_mlflow(results, config, args, model_name, model_display_name):
    try:
        import mlflow
        mlflow.set_tracking_uri((PROJECT_ROOT / "mlruns").resolve().as_uri())
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


def write_experiment_logs(out_dir, model_name, args, results, summary, extra=None):
    record = {
        "timestamp": summary["timestamp"],
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
    }
    if extra:
        record.update(extra)

    with open(out_dir / "experiment_log.json", "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, default=str)

    run_log_dir = PROJECT_ROOT / "logs"
    run_log_dir.mkdir(parents=True, exist_ok=True)
    with open(run_log_dir / "experiment_runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


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


def build_training_summary(results, config, args, model_name, model_display_name, n_samples, n_features, timestamp,
                           version_name, output_dir):
    return {
        "timestamp": timestamp,
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
            "early_stopping_patience": config.early_stopping_patience,
            "scheduler_patience": config.scheduler_patience,
            "random_state": config.random_state,
            "selected_cancer_types": list(config.cancer_types),
            "selected_non_cancer_groups": list(config.non_cancer_groups),
        },
        "model_params": XGBOOST_DEFAULT_PARAMS if model_name == "xgboost" else {
            "resnet_channels": list(config.resnet_channels),
            "resnet_blocks": list(config.resnet_blocks),
            "encoder_output_dim": config.encoder_output_dim,
            "head_hidden_dim": config.head_hidden_dim,
            "dropout_rate": config.dropout_rate,
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
            "train_auc_s2": tm.get("auc_s2", np.nan),
            "train_f1_macro_s2": tm.get("f1_macro_s2", np.nan),
            "val_auc_s1": vm.get("auc_s1", np.nan),
            "val_auc_s2": vm.get("auc_s2", np.nan),
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
    save_training_visualizations(results, out_dir)
    save_stage1_difference_artifacts(df_valid, out_dir)
    write_experiment_logs(
        out_dir,
        summary["model_name"],
        args,
        results,
        summary,
        extra={"has_epoch_history": bool(results["histories"])},
    )


def run_single_model(model_name, out_dir, version_name, X, binary_labels, cancer_type_labels, sample_ids, groups_arr,
                     df_valid, config, device, args, timestamp):
    model_display_name = MODEL_DISPLAY_NAMES[model_name]
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger.info("\n" + "=" * 64)
    logger.info(f"  Model: {model_display_name}")
    logger.info("=" * 64)

    if model_name == "xgboost":
        results = run_xgboost_cv(
            X,
            binary_labels,
            cancer_type_labels,
            sample_ids,
            groups_arr,
            config,
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
    p.add_argument("--config", "-c", default="config.yaml")
    p.add_argument("--model", choices=["resnet18", "cnn1d", "xgboost"], default="resnet18")
    p.add_argument("--benchmark-models", nargs="+", choices=["resnet18", "cnn1d", "xgboost"], default=None)
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
    p.add_argument("--device", default="auto")
    p.add_argument("--no-mlflow", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    t0 = datetime.now()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-7s │ %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler("train.log", mode="w", encoding="utf-8")],
    )

    logger.info("=" * 64)
    logger.info("  SERS Two-Stage Training")
    logger.info("=" * 64)

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = torch.device(args.device)
    logger.info(f"  Device: {device}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    if args.epochs:
        mc = ModelConfig(**{**mc.__dict__, "n_epochs": args.epochs})
    if args.batch_size:
        mc = ModelConfig(**{**mc.__dict__, "batch_size": args.batch_size})
    if args.lr:
        mc = ModelConfig(**{**mc.__dict__, "learning_rate": args.lr})
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

    logger.info("\n[Step 3] Aggregating replicates...")
    df_agg = aggregate_replicates(df, feat_cols, args.aggregate)

    logger.info("\n[Step 4] Creating labels...")
    X, bl, ctl, sample_ids, groups_arr = create_labels(df_agg, mc)

    valid_groups = set(mc.cancer_types) | set(mc.non_cancer_groups)
    df_valid = df_agg[df_agg["group"].isin(valid_groups)].copy()

    model_names = args.benchmark_models if args.benchmark_models else [args.model]
    benchmark_rows = []

    for model_name in model_names:
        run_out_dir, version_name = resolve_run_output_dir(out_dir, model_name, args.version)
        logger.info(f"\n[Step 5] {args.n_splits}-fold CV for {MODEL_DISPLAY_NAMES[model_name]} ({version_name})...")
        results, summary = run_single_model(
            model_name,
            run_out_dir,
            version_name,
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

    if args.benchmark_models and benchmark_rows:
        pd.DataFrame(benchmark_rows).to_csv(out_dir / "benchmark_summary.csv", index=False)

    elapsed = datetime.now() - t0
    logger.info(f"\n{'=' * 64}")
    logger.info(f"  Training complete! ({elapsed})")
    logger.info(f"  Output: {out_dir}/")
    if args.benchmark_models:
        logger.info(f"  Benchmark summary: {out_dir / 'benchmark_summary.csv'}")
        logger.info(f"  Next: python models/test.py -i {out_dir / model_names[0]}")
    else:
        logger.info(f"  Next: python models/test.py -i {out_dir / model_names[0]}")
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

    save_training_visualizations(results, out_dir)

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
