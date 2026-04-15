"""
Cross-Attention Fusion Training — SERS + Clinical via Transformer

Usage:
    python models/cross_attention/train.py
    python models/cross_attention/train.py --aggregate medoid --n-splits 5
    python models/cross_attention/train.py --n-attn-layers 1 --d-model 128

Outputs (results/training/<experiment>/xattn_resnet18/):
    fold_predictions.npz, checkpoints/, training_summary.json, etc.
"""

from __future__ import annotations

import sys
import argparse
import logging
import os
import json
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sers.models._legacy.resnet_v1.model import ModelConfig, TwoStageLoss
from sers.models.usersnet.clinical_fusion import (
    load_clinical, merge_clinical_features, ClinicalSERSDataset,
    compute_multichannel_spectra, TIER1_COLS,
)
from sers.models.experimental.cross_attention.model import (
    CrossAttentionConfig, CrossAttentionCancerDetector,
    build_cross_attention_model, cross_attention_model_summary,
)

from models.train import (
    load_processed_spectra, get_feature_columns, apply_class_selection,
    resolve_aliases, apply_patient_exclusions, apply_training_overrides,
    aggregate_replicates, create_labels,
    EarlyStopping, _build_loader, _compute_overall_metrics,
    resolve_experiment_output_dir, resolve_run_output_dir,
    resolve_runtime_config, configure_torch_runtime,
    save_fold_predictions, save_training_summary, save_training_visualizations,
    save_fold_metrics, save_classification_reports, save_epoch_histories,
    build_training_summary, FoldArtifacts, BaseCrossValidator,
)
from src.sers.config import RESULTS_DIR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training loop (modified for clinical input)
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device, runtime, scaler=None):
    model.train()
    total, n = 0.0, 0
    use_amp = runtime["use_amp"]
    for batch in loader:
        spec = batch["spectra"].to(device, non_blocking=runtime["pin_memory"])
        clin = batch["clinical"].to(device, non_blocking=runtime["pin_memory"])
        by = batch["binary_label"].to(device, non_blocking=runtime["pin_memory"])
        cy = batch["cancer_type_label"].to(device, non_blocking=runtime["pin_memory"])
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            out = model(spec, clin)
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
    all_attn = []
    use_amp = runtime["use_amp"]

    for batch in loader:
        spec = batch["spectra"].to(device, non_blocking=runtime["pin_memory"])
        clin = batch["clinical"].to(device, non_blocking=runtime["pin_memory"])
        by = batch["binary_label"].to(device, non_blocking=runtime["pin_memory"])
        cy = batch["cancer_type_label"].to(device, non_blocking=runtime["pin_memory"])
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            out = model(spec, clin)
            losses = criterion(out, by, cy)
        total += losses["total"].item(); n += 1

        all_bp.append(out["binary_prob"].cpu().numpy())
        all_bt.append(by.cpu().numpy())
        all_cl.append(out["cancer_logits"].cpu().numpy())
        all_ct.append(cy.cpu().numpy())
        all_emb.append(out["embedding"].cpu().numpy())
        # Collect last layer attention weights for interpretability
        if "attn_weights" in out and out["attn_weights"]:
            all_attn.append(out["attn_weights"][-1].cpu().numpy())

    bp = np.concatenate(all_bp).squeeze()
    bt = np.concatenate(all_bt).squeeze()
    cl = np.concatenate(all_cl)
    ct = np.concatenate(all_ct)
    emb = np.concatenate(all_emb)

    auc_s1 = roc_auc_score(bt, bp) if len(np.unique(bt)) > 1 else float("nan")
    pred_s1 = (bp > 0.5).astype(int)
    acc_s1 = accuracy_score(bt, pred_s1)

    cm = ct >= 0
    auc_s2, acc_s2, f1_macro_s2 = float("nan"), float("nan"), float("nan")
    if cm.sum() > 0 and len(np.unique(ct[cm])) > 1:
        cp = torch.softmax(torch.tensor(cl[cm]), dim=-1).numpy()
        cpred = cp.argmax(axis=1)
        acc_s2 = accuracy_score(ct[cm], cpred)
        f1_macro_s2 = f1_score(ct[cm], cpred, average="macro", zero_division=0)
        try:
            auc_s2 = roc_auc_score(ct[cm], cp, multi_class="ovr", average="macro")
        except ValueError:
            pass

    result = {
        "loss": total / max(n, 1),
        "auc_s1": auc_s1, "auc_s2": auc_s2,
        "acc_s1": acc_s1, "acc_s2": acc_s2,
        "f1_macro_s2": f1_macro_s2,
        "binary_prob": bp, "binary_true": bt,
        "cancer_logits": cl, "cancer_true": ct,
        "embedding": emb,
    }
    if all_attn:
        result["attn_weights"] = np.concatenate(all_attn, axis=0)
    return result


def train_fold(model, train_ds, val_ds, config, device, fold_i, runtime):
    bl = train_ds.binary_labels.squeeze().numpy()
    ctl = train_ds.cancer_type_labels.numpy()

    sample_weights = np.ones(len(train_ds), dtype=np.float64)
    n_pos, n_neg = (bl > 0.5).sum(), (bl <= 0.5).sum()
    for i in range(len(train_ds)):
        if bl[i] > 0.5:
            ct = ctl[i]
            ct_count = (ctl[ctl >= 0] == ct).sum()
            sample_weights[i] = 1.0 / max(ct_count, 1)
        else:
            sample_weights[i] = 1.0 / max(n_neg, 1)
    sample_weights = sample_weights / sample_weights.mean()

    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_ds), replacement=True)
    train_loader = _build_loader(train_ds, config.batch_size, runtime, sampler=sampler)
    val_loader = _build_loader(val_ds, config.batch_size, runtime, shuffle=False)

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
        use_focal_loss=config.use_focal_loss,
        focal_gamma=config.focal_gamma,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=config.scheduler_patience, factor=0.5, min_lr=1e-6,
    )
    es = EarlyStopping(config.early_stopping_patience)
    scaler = torch.amp.GradScaler(device="cuda", enabled=runtime["use_amp"])

    history = {
        "train_loss": [], "val_loss": [], "val_auc_s1": [], "val_auc_s2": [],
        "val_f1_macro_s2": [], "val_error_s1": [], "val_error_s2": [], "lr": [],
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
                f"    Fold {fold_i+1} Ep {epoch+1:3d} | "
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


# ---------------------------------------------------------------------------
# Cross-Validator
# ---------------------------------------------------------------------------
class CrossAttentionCrossValidator(BaseCrossValidator):

    def __init__(self, X, binary_labels, cancer_type_labels, sample_ids, groups_arr,
                 clinical_features, config, xattn_config, device, runtime,
                 n_splits=5, ckpt_dir=None):
        super().__init__(X, binary_labels, cancer_type_labels, sample_ids, groups_arr,
                         config, n_splits=n_splits, ckpt_dir=ckpt_dir)
        self.clinical_features = clinical_features
        self.xattn_config = xattn_config
        self.device = device
        self.runtime = runtime

    @property
    def model_display_name(self):
        return "CrossAttn-ResNet18-1D"

    def _make_embedding_buffer(self, n_samples):
        return np.full((n_samples, self.xattn_config.d_model), np.nan)

    def _log_fold_header(self, fold_idx, train_idx, val_idx):
        logger.info(
            f"\n  -- Fold {fold_idx+1}/{self.n_splits} "
            f"(train={len(train_idx)}, val={len(val_idx)}) --"
        )

    def _run_fold(self, fold_idx, train_idx, val_idx):
        train_ds = ClinicalSERSDataset(
            self.X[train_idx],
            self.binary_labels[train_idx],
            self.cancer_type_labels[train_idx],
            self.clinical_features[train_idx],
            groups=self.groups_arr[train_idx].tolist(),
            sample_ids=[str(sid) for sid in self.sample_ids[train_idx]],
            augment=True,
        )
        val_ds = ClinicalSERSDataset(
            self.X[val_idx],
            self.binary_labels[val_idx],
            self.cancer_type_labels[val_idx],
            self.clinical_features[val_idx],
            groups=self.groups_arr[val_idx].tolist(),
            sample_ids=[str(sid) for sid in self.sample_ids[val_idx]],
            augment=False,
        )

        model = build_cross_attention_model(self.config, self.xattn_config).to(self.device)
        fold_result = train_fold(model, train_ds, val_ds, self.config, self.device, fold_idx, self.runtime)
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
            extra={"val_attn_weights": val_metrics.get("attn_weights")},
        )

    def _log_fold_summary(self, fold_idx, artifacts):
        train_metrics = artifacts.fold_result["train_metrics"]
        val_metrics = artifacts.fold_result["val_metrics"]
        logger.info(
            f"    Train S1={train_metrics['auc_s1']:.3f} S2={train_metrics['auc_s2']:.3f} "
            f"Val S1={val_metrics['auc_s1']:.3f} S2={val_metrics['auc_s2']:.3f}"
        )

    def _save_fold_artifacts(self, fold_idx, train_idx, val_idx, artifacts):
        save_dict = {
            "fold": fold_idx,
            "model_state_dict": artifacts.fold_result["best_state"],
            "model_name": "xattn_resnet18",
            "config": self.config.__dict__,
            "xattn_config": self.xattn_config.__dict__,
            "train_idx": train_idx,
            "val_idx": val_idx,
        }
        # Save attention weights for interpretability
        attn_w = artifacts.extra.get("val_attn_weights")
        if attn_w is not None:
            save_dict["val_attn_weights"] = attn_w
        torch.save(save_dict, self.ckpt_dir / f"fold_{fold_idx}.pt")


# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------
def log_mlflow_xattn(results, config, xattn_config, args):
    try:
        import mlflow
        mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
        mlflow.set_experiment("sers-fusion-cross-attention")
        with mlflow.start_run(run_name=f"xattn_{args.aggregate}_{args.n_splits}fold"):
            mlflow.log_params({
                "fusion_type": "cross_attention",
                "model": "CrossAttn-ResNet18-1D",
                "aggregate": args.aggregate,
                "n_splits": args.n_splits,
                "batch_size": config.batch_size,
                "lr": config.learning_rate,
                "epochs": config.n_epochs,
                "encoder_dim": config.encoder_output_dim,
                "resnet_channels": str(config.resnet_channels),
                "n_clinical": xattn_config.n_clinical,
                "d_model": xattn_config.d_model,
                "n_heads": xattn_config.n_heads,
                "n_layers": xattn_config.n_layers,
                "attn_dropout": xattn_config.attn_dropout,
            })
            for k, v in results["overall"].items():
                if isinstance(v, float) and not np.isnan(v):
                    mlflow.log_metric(k, v)
    except Exception as e:
        logger.warning(f"MLflow logging failed: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Cross-Attention Fusion — SERS + Clinical Training")
    p.add_argument("--input", "-i", default="results/processed_spectra.csv")
    p.add_argument("--output", "-o", default="results/training")
    p.add_argument("--config", "-c", default="config/config.yaml")
    p.add_argument("--cancer-types", nargs="+", default=None)
    p.add_argument("--non-cancer-groups", nargs="+", default=None)
    p.add_argument("--aggregate", "-a", choices=["medoid", "mean", "none"], default="medoid")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--dropout-rate", type=float, default=None)
    p.add_argument("--stage2-loss-weight", type=float, default=None)
    p.add_argument("--head-hidden-dim", type=int, default=None)
    p.add_argument("--resnet-channels", nargs="+", type=int, default=None)
    p.add_argument("--use-focal-loss", action="store_true")
    p.add_argument("--focal-gamma", type=float, default=None)
    p.add_argument("--class-balance-beta", type=float, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument("--experiment", default=None)
    p.add_argument("--version", default=None)
    p.add_argument("--exclude-patients", default=None)
    # Experiment metadata
    p.add_argument("--hypothesis", default=None)
    p.add_argument("--variable", default=None)
    p.add_argument("--baseline", default=None)
    p.add_argument("--tags", nargs="+", default=None)
    p.add_argument("--phase", default=None)
    # Cross-Attention specific
    p.add_argument("--n-clinical", type=int, default=3)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-attn-layers", type=int, default=2)
    p.add_argument("--attn-dropout", type=float, default=0.1)
    return p.parse_args()


def main():
    args = parse_args()
    t0 = datetime.now()
    out_root = Path(args.output)
    experiment_dir, experiment_name = resolve_experiment_output_dir(out_root, args.experiment)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(experiment_dir / "train_xattn.log", mode="w", encoding="utf-8"),
        ],
    )

    logger.info("=" * 64)
    logger.info("  Cross-Attention Fusion — SERS + Clinical Training")
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

    # Step 1: Load spectral data
    logger.info("\n[Step 1] Loading spectral data...")
    df = load_processed_spectra(args.input)
    feat_cols = get_feature_columns(df)
    n_feat = len(feat_cols)

    mc = ModelConfig.from_pipeline_config(raw_cfg, n_spectral_features=n_feat) if raw_cfg else ModelConfig(n_spectral_features=n_feat)
    mc = apply_training_overrides(mc, args)
    mc = apply_class_selection(mc, cancer_types=args.cancer_types, non_cancer_groups=args.non_cancer_groups)

    logger.info(f"  Cancer types ({mc.n_cancer_types}): {mc.cancer_types}")
    logger.info(f"  Non-cancer: {mc.non_cancer_groups}")

    # Step 2: Resolve aliases
    logger.info("\n[Step 2] Resolving aliases...")
    df = resolve_aliases(df, mc)

    if args.exclude_patients:
        logger.info("\n[Step 2b] Applying patient exclusions...")
        df = apply_patient_exclusions(df, args.exclude_patients)

    # Step 3: Load clinical data
    logger.info("\n[Step 3] Loading clinical data...")
    clin = load_clinical(PROJECT_ROOT)

    # Step 4: Aggregate replicates
    logger.info("\n[Step 4] Aggregating replicates...")
    df_agg = aggregate_replicates(df, feat_cols, args.aggregate)

    # Step 5: Create labels
    logger.info("\n[Step 5] Creating labels...")
    X, bl, ctl, sample_ids, groups_arr = create_labels(df_agg, mc)

    valid_groups = set(mc.cancer_types) | set(mc.non_cancer_groups)
    df_valid = df_agg[df_agg["group"].isin(valid_groups)].copy()

    # Step 6: Merge clinical features
    logger.info("\n[Step 6] Merging clinical features...")
    clinical_features = merge_clinical_features(df_valid, clin, TIER1_COLS)
    logger.info(f"  Clinical shape: {clinical_features.shape}")

    # Step 6b: Compute multi-channel spectra (raw + derivatives)
    logger.info("\n[Step 6b] Computing multi-channel spectra...")
    X = compute_multichannel_spectra(X)  # (N, 3, L)

    # Cross-Attention config
    xattn_config = CrossAttentionConfig(
        n_clinical=args.n_clinical,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_attn_layers,
        attn_dropout=args.attn_dropout,
    )

    # Model summary
    model = build_cross_attention_model(mc, xattn_config)
    logger.info(f"\n{cross_attention_model_summary(model)}")
    del model

    # Step 7: Cross-validation
    run_out_dir, version_name = resolve_run_output_dir(experiment_dir, "xattn_resnet18", args.version)
    ckpt_dir = run_out_dir / "checkpoints"
    run_out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    runtime = resolve_runtime_config(args, device)
    logger.info(
        f"  Runtime: amp={runtime['use_amp']} num_workers={runtime['num_workers']} "
        f"pin_memory={runtime['pin_memory']}"
    )

    logger.info(f"\n[Step 7] {args.n_splits}-fold CV for CrossAttn-ResNet18-1D ({version_name})...")
    cv_runner = CrossAttentionCrossValidator(
        X, bl, ctl, sample_ids, groups_arr,
        clinical_features, mc, xattn_config, device, runtime,
        n_splits=args.n_splits, ckpt_dir=ckpt_dir,
    )
    results = cv_runner.run()

    # Step 8: Save artifacts
    logger.info("\n[Step 8] Saving results...")
    model_display_name = "CrossAttn-ResNet18-1D"
    model_params = {
        "resnet_channels": list(mc.resnet_channels),
        "encoder_output_dim": mc.encoder_output_dim,
        "head_hidden_dim": mc.head_hidden_dim,
        "dropout_rate": mc.dropout_rate,
        "n_clinical": xattn_config.n_clinical,
        "d_model": xattn_config.d_model,
        "n_heads": xattn_config.n_heads,
        "n_layers": xattn_config.n_layers,
        "attn_dropout": xattn_config.attn_dropout,
        "fusion_type": "cross_attention",
    }

    summary = build_training_summary(
        results, mc, args,
        model_name="xattn_resnet18",
        model_display_name=model_display_name,
        n_samples=len(X),
        n_features=X.shape[-1],
        timestamp=t0.isoformat(),
        version_name=version_name,
        output_dir=run_out_dir,
        experiment_name=experiment_name,
        model_params=model_params,
    )

    save_training_summary(summary, run_out_dir)
    save_fold_predictions(results, run_out_dir)
    save_fold_metrics(results, run_out_dir)
    save_classification_reports(results, bl, ctl, mc, run_out_dir)
    if results.get("histories"):
        save_epoch_histories(results, run_out_dir)
        save_training_visualizations(results, run_out_dir)

    if not args.no_mlflow:
        log_mlflow_xattn(results, mc, xattn_config, args)

    elapsed = datetime.now() - t0
    logger.info(f"\n  Done in {elapsed}. Output: {run_out_dir}")
    logger.info(f"  S1 AUC: {results['overall'].get('val_s1_auc', 'N/A'):.4f}")
    logger.info(f"  S2 F1:  {results['overall'].get('val_s2_f1_macro', 'N/A'):.4f}")


if __name__ == "__main__":
    main()
