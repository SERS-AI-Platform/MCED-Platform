"""
7-Cancer Full Optimization — uSERS-Net
Comprehensive optimization: preprocessing, QC, wavenumber grid, architecture, HP

Phase 1: Wavenumber grid ablation (LR fast scan)
Phase 2: Aggregation ablation
Phase 3: Architecture + HP grid search (ResNet18 + Ensemble)
Phase 4: Final evaluation (5-fold CV + held-out test)

Usage:
    python scripts/analysis/optimize_7cancer_full.py --phase 1
    python scripts/analysis/optimize_7cancer_full.py --phase all
"""

from __future__ import annotations

import sys, os, json, time, argparse, warnings, logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pandas as pd
import torch
import yaml

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from sers.models._legacy.resnet_v1.model import ModelConfig, SERSDataset, SERSCancerDetector
from models.train import (
    load_processed_spectra, get_feature_columns,
    apply_class_selection, resolve_aliases, aggregate_replicates,
    create_labels, train_fold, EarlyStopping,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, f1_score, roc_curve, accuracy_score
from scipy.signal import savgol_filter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ──
CANCER_TYPES = ["PRO", "BRE", "OVA", "LUN", "CRC", "PAN", "BLC"]
NON_CANCER = ["NOR", "DIA", "HBP", "H.D."]
N_SPLITS = 5
RANDOM_STATE = 42
OUT_DIR = PROJECT_ROOT / "results" / "training" / "optimize_7cancer_full"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_config():
    with open(PROJECT_ROOT / "config" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_and_prepare(aggregate="medoid", wavenumber_range=None, feat_transform=None):
    """Load data, apply aliases, aggregate, create labels.

    wavenumber_range: (min_wn, max_wn) or None for full range
    feat_transform: None, 'derivative1', 'derivative2', 'pca30'
    """
    raw_cfg = load_config()
    df = load_processed_spectra()
    feat_cols = get_feature_columns(df)

    # Wavenumber range filtering
    if wavenumber_range is not None:
        wn_min, wn_max = wavenumber_range
        selected_cols = [c for c in feat_cols if wn_min <= float(c.split("_")[1]) <= wn_max]
        feat_cols = selected_cols
        logger.info(f"  Wavenumber range [{wn_min}, {wn_max}]: {len(feat_cols)} features")

    n_feat = len(feat_cols)
    mc = ModelConfig.from_pipeline_config(raw_cfg, n_spectral_features=n_feat)
    mc = apply_class_selection(mc, cancer_types=CANCER_TYPES, non_cancer_groups=NON_CANCER)

    df = resolve_aliases(df, mc)
    df_agg = aggregate_replicates(df, feat_cols, aggregate)

    # Create labels using the selected feature columns
    valid_groups = set(mc.cancer_types) | set(mc.non_cancer_groups)
    df_valid = df_agg[df_agg["group"].isin(valid_groups)].copy()

    X = df_valid[feat_cols].values
    groups_list = df_valid["group"].tolist()
    sample_ids = df_valid["sample_id"].values
    groups_arr = np.array(groups_list)

    binary_labels = np.array([1 if g in mc.cancer_types else 0 for g in groups_list])
    cancer_type_labels = np.array([mc.cancer_type_index(g) for g in groups_list])

    # Feature transforms
    if feat_transform == "derivative1":
        X = np.gradient(X, axis=1)
    elif feat_transform == "derivative2":
        X = np.gradient(np.gradient(X, axis=1), axis=1)
    elif feat_transform == "savgol_d1":
        X = savgol_filter(X, window_length=11, polyorder=3, deriv=1, axis=1)
    elif feat_transform == "concat_d1":
        d1 = savgol_filter(X, window_length=11, polyorder=3, deriv=1, axis=1)
        X = np.hstack([X, d1])

    logger.info(f"  Data: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"  Cancer: {(binary_labels==1).sum()}, Non-cancer: {(binary_labels==0).sum()}")

    return X, binary_labels, cancer_type_labels, sample_ids, groups_arr, mc


def run_lr_cv(X, bl, ctl, sample_ids, groups_arr, mc, n_splits=N_SPLITS,
              include_clinical=False, sex_constraint=False):
    """Fast Logistic Regression 5-fold CV. Returns metrics dict."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    # Composite label for stratification
    composite = bl * 100 + np.clip(ctl, 0, 99)

    all_bp = np.full(len(X), np.nan)
    all_cl = np.full((len(X), mc.n_cancer_types), np.nan)
    fold_aucs = []

    for fold_i, (train_idx, val_idx) in enumerate(sgkf.split(X, composite, groups=sample_ids)):
        X_tr, X_val = X[train_idx], X[val_idx]
        bl_tr, bl_val = bl[train_idx], bl[val_idx]
        ctl_tr, ctl_val = ctl[train_idx], ctl[val_idx]

        # Stage 1: Binary
        pipe_s1 = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, max_iter=1000, solver="saga", class_weight="balanced", random_state=RANDOM_STATE))
        pipe_s1.fit(X_tr, bl_tr)
        bp = pipe_s1.predict_proba(X_val)[:, 1]
        all_bp[val_idx] = bp

        # Stage 2: Cancer type (cancer samples only)
        cancer_tr = ctl_tr >= 0
        if cancer_tr.sum() > 0:
            pipe_s2 = make_pipeline(StandardScaler(), LogisticRegression(
                C=1.0, max_iter=1000, solver="saga", class_weight="balanced",
                multi_class="multinomial", random_state=RANDOM_STATE))
            pipe_s2.fit(X_tr[cancer_tr], ctl_tr[cancer_tr])
            cl = pipe_s2.predict_proba(X_val)
            all_cl[val_idx] = cl

        auc = roc_auc_score(bl_val, bp)
        fold_aucs.append(auc)

    # Overall metrics
    valid = ~np.isnan(all_bp)
    s1_auc = roc_auc_score(bl[valid], all_bp[valid])

    fpr, tpr, thresholds = roc_curve(bl[valid], all_bp[valid])
    thr = thresholds[np.argmax(tpr - fpr)]
    pred_bin = (all_bp[valid] > thr).astype(int)
    s1_sens = (pred_bin[bl[valid] == 1] == 1).mean()
    s1_spec = (pred_bin[bl[valid] == 0] == 0).mean()

    cancer_mask = ctl[valid] >= 0
    if cancer_mask.sum() > 0:
        pred_type = all_cl[valid][cancer_mask].argmax(axis=1)
        s2_f1 = f1_score(ctl[valid][cancer_mask], pred_type, average="macro", zero_division=0)
        try:
            s2_auc = roc_auc_score(ctl[valid][cancer_mask], all_cl[valid][cancer_mask],
                                    multi_class="ovr", average="macro")
        except:
            s2_auc = float("nan")
    else:
        s2_f1 = float("nan")
        s2_auc = float("nan")

    return {
        "s1_auc": s1_auc,
        "s1_auc_std": np.std(fold_aucs),
        "s1_sens": s1_sens,
        "s1_spec": s1_spec,
        "s2_f1": s2_f1,
        "s2_auc": s2_auc,
        "threshold": thr,
        "n_samples": int(valid.sum()),
    }


def run_resnet_cv(X, bl, ctl, sample_ids, groups_arr, mc,
                  lr=3e-4, dropout=0.3, wd=1e-3, bs=32, channels=(64, 128, 256, 512),
                  n_epochs=150, n_splits=N_SPLITS):
    """ResNet18 5-fold CV. Returns metrics dict."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mc_run = ModelConfig(**{**mc.__dict__,
        "n_spectral_features": X.shape[1],
        "learning_rate": lr,
        "dropout_rate": dropout,
        "weight_decay": wd,
        "batch_size": bs,
        "n_epochs": n_epochs,
        "resnet_channels": channels,
        "encoder_output_dim": channels[-1],
        "random_state": RANDOM_STATE,
    })

    runtime = {
        "num_workers": 4 if device.type == "cuda" else 0,
        "pin_memory": device.type == "cuda",
        "prefetch_factor": 2 if device.type == "cuda" else None,
        "use_amp": device.type == "cuda",
    }

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    composite = bl * 100 + np.clip(ctl, 0, 99)

    all_bp = np.full(len(X), np.nan)
    all_cl = np.full((len(X), mc_run.n_cancer_types), np.nan)
    fold_aucs = []

    for fold_i, (train_idx, val_idx) in enumerate(sgkf.split(X, composite, groups=sample_ids)):
        train_ds = SERSDataset(X[train_idx], bl[train_idx], ctl[train_idx], augment=True)
        val_ds = SERSDataset(X[val_idx], bl[val_idx], ctl[val_idx], augment=False)

        model = SERSCancerDetector(mc_run).to(device)
        fold_result = train_fold(model, train_ds, val_ds, mc_run, device, fold_i, runtime)

        val_m = fold_result["val_metrics"]
        all_bp[val_idx] = val_m["binary_prob"]
        all_cl[val_idx] = val_m["cancer_logits"]

        auc = roc_auc_score(bl[val_idx], val_m["binary_prob"])
        fold_aucs.append(auc)
        logger.info(f"  Fold {fold_i+1}: AUC={auc:.4f}")

    # Compute metrics
    valid = ~np.isnan(all_bp)
    s1_auc = roc_auc_score(bl[valid], all_bp[valid])

    fpr, tpr, thresholds = roc_curve(bl[valid], all_bp[valid])
    thr = thresholds[np.argmax(tpr - fpr)]
    pred_bin = (all_bp[valid] > thr).astype(int)
    s1_sens = (pred_bin[bl[valid] == 1] == 1).mean()
    s1_spec = (pred_bin[bl[valid] == 0] == 0).mean()

    cancer_mask = ctl[valid] >= 0
    cl_probs = torch.softmax(torch.tensor(all_cl[valid][cancer_mask]), dim=-1).numpy()
    pred_type = cl_probs.argmax(axis=1)
    s2_f1 = f1_score(ctl[valid][cancer_mask], pred_type, average="macro", zero_division=0)
    try:
        s2_auc = roc_auc_score(ctl[valid][cancer_mask], cl_probs,
                                multi_class="ovr", average="macro")
    except:
        s2_auc = float("nan")

    return {
        "s1_auc": s1_auc,
        "s1_auc_std": np.std(fold_aucs),
        "s1_sens": s1_sens,
        "s1_spec": s1_spec,
        "s2_f1": s2_f1,
        "s2_auc": s2_auc,
        "threshold": thr,
        "n_samples": int(valid.sum()),
        "all_bp": all_bp,
        "all_cl": all_cl,
    }


def run_ensemble(bp_lr, cl_lr, bp_rn, cl_rn, bl, ctl, alpha=0.8):
    """Blend LR + ResNet18 predictions."""
    valid = ~np.isnan(bp_lr) & ~np.isnan(bp_rn)
    bp = alpha * bp_lr[valid] + (1 - alpha) * bp_rn[valid]

    rn_probs = torch.softmax(torch.tensor(cl_rn[valid]), dim=-1).numpy()
    cl = alpha * cl_lr[valid] + (1 - alpha) * rn_probs

    s1_auc = roc_auc_score(bl[valid], bp)
    fpr, tpr, thresholds = roc_curve(bl[valid], bp)
    thr = thresholds[np.argmax(tpr - fpr)]
    pred_bin = (bp > thr).astype(int)
    s1_sens = (pred_bin[bl[valid] == 1] == 1).mean()
    s1_spec = (pred_bin[bl[valid] == 0] == 0).mean()

    cancer_mask = ctl[valid] >= 0
    pred_type = cl[cancer_mask].argmax(axis=1)
    s2_f1 = f1_score(ctl[valid][cancer_mask], pred_type, average="macro", zero_division=0)

    return {
        "s1_auc": s1_auc, "s1_sens": s1_sens, "s1_spec": s1_spec,
        "s2_f1": s2_f1, "threshold": thr, "alpha": alpha,
    }


def run_held_out_test(X, bl, ctl, sample_ids, groups_arr, mc,
                      lr_config=None, rn_config=None, alpha=0.8,
                      n_repeats=5, test_ratio=0.2, val_ratio=0.2):
    """Train/val/test split evaluation (repeated)."""
    from sklearn.model_selection import StratifiedGroupKFold as SGKF

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    for rep in range(n_repeats):
        seed = RANDOM_STATE + rep
        np.random.seed(seed)

        # Use composite keys (group + sample_id) for patient-level splitting
        patient_keys = np.array([f"{g}_{s}" for g, s in zip(groups_arr, sample_ids)])
        unique_pkeys = np.unique(patient_keys)

        # Stratification by group (cancer type) not just binary
        pkey_to_group = {}
        pkey_to_bl = {}
        for pk, g, b in zip(patient_keys, groups_arr, bl):
            pkey_to_group[pk] = g
            pkey_to_bl[pk] = b

        pkey_labels = np.array([pkey_to_group[pk] for pk in unique_pkeys])

        # Stratified split by group
        from sklearn.model_selection import StratifiedShuffleSplit
        sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
        remain_idx, test_pk_idx = next(sss_test.split(unique_pkeys, pkey_labels))

        remain_pkeys = unique_pkeys[remain_idx]
        test_pkeys = set(unique_pkeys[test_pk_idx])

        # Split remaining into train/val
        remain_labels = np.array([pkey_to_group[pk] for pk in remain_pkeys])
        val_frac = val_ratio / (1 - test_ratio)
        sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
        train_pk_idx, val_pk_idx = next(sss_val.split(remain_pkeys, remain_labels))

        train_pkeys = set(remain_pkeys[train_pk_idx])
        val_pkeys = set(remain_pkeys[val_pk_idx])

        train_mask = np.array([pk in train_pkeys for pk in patient_keys])
        val_mask = np.array([pk in val_pkeys for pk in patient_keys])
        test_mask = np.array([pk in test_pkeys for pk in patient_keys])

        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]
        test_idx = np.where(test_mask)[0]

        X_tr, X_val, X_te = X[train_idx], X[val_idx], X[test_idx]
        bl_tr, bl_val, bl_te = bl[train_idx], bl[val_idx], bl[test_idx]
        ctl_tr, ctl_val, ctl_te = ctl[train_idx], ctl[val_idx], ctl[test_idx]

        # --- LR ---
        pipe_s1 = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, max_iter=1000, solver="saga", class_weight="balanced", random_state=seed))
        pipe_s1.fit(X_tr, bl_tr)
        bp_lr_te = pipe_s1.predict_proba(X_te)[:, 1]

        cancer_tr = ctl_tr >= 0
        pipe_s2 = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, max_iter=1000, solver="saga", class_weight="balanced",
            multi_class="multinomial", random_state=seed))
        pipe_s2.fit(X_tr[cancer_tr], ctl_tr[cancer_tr])
        cl_lr_te = pipe_s2.predict_proba(X_te)

        # --- ResNet18 ---
        rn_cfg = rn_config or {}
        mc_run = ModelConfig(**{**mc.__dict__,
            "n_spectral_features": X.shape[1],
            "learning_rate": rn_cfg.get("lr", 3e-4),
            "dropout_rate": rn_cfg.get("dropout", 0.3),
            "weight_decay": rn_cfg.get("wd", 1e-3),
            "batch_size": rn_cfg.get("bs", 32),
            "n_epochs": rn_cfg.get("n_epochs", 150),
            "resnet_channels": rn_cfg.get("channels", (64, 128, 256, 512)),
            "encoder_output_dim": rn_cfg.get("channels", (64, 128, 256, 512))[-1],
            "random_state": seed,
        })

        runtime = {
            "num_workers": 4 if device.type == "cuda" else 0,
            "pin_memory": device.type == "cuda",
            "prefetch_factor": 2 if device.type == "cuda" else None,
            "use_amp": device.type == "cuda",
        }

        # Train on train, validate on val, test on test
        train_ds = SERSDataset(X_tr, bl_tr, ctl_tr, augment=True)
        val_ds = SERSDataset(X_val, bl_val, ctl_val, augment=False)

        model = SERSCancerDetector(mc_run).to(device)
        fold_result = train_fold(model, train_ds, val_ds, mc_run, device, rep, runtime)

        # Evaluate on test set
        test_ds = SERSDataset(X_te, bl_te, ctl_te, augment=False)
        from torch.utils.data import DataLoader
        test_loader = DataLoader(test_ds, batch_size=mc_run.batch_size, shuffle=False,
                                  num_workers=runtime["num_workers"],
                                  pin_memory=runtime["pin_memory"])

        model.eval()
        all_bp_rn, all_cl_rn = [], []
        with torch.no_grad():
            for batch in test_loader:
                spec = batch["spectra"].to(device)
                out = model(spec)
                all_bp_rn.append(out["binary_prob"].cpu().numpy())
                all_cl_rn.append(out["cancer_logits"].cpu().numpy())

        bp_rn_te = np.concatenate(all_bp_rn).squeeze()
        cl_rn_te = np.concatenate(all_cl_rn)

        # --- Ensemble ---
        rn_probs = torch.softmax(torch.tensor(cl_rn_te), dim=-1).numpy()
        bp_ens = alpha * bp_lr_te + (1 - alpha) * bp_rn_te
        cl_ens = alpha * cl_lr_te + (1 - alpha) * rn_probs

        # Metrics
        s1_auc_lr = roc_auc_score(bl_te, bp_lr_te)
        s1_auc_rn = roc_auc_score(bl_te, bp_rn_te)
        s1_auc_ens = roc_auc_score(bl_te, bp_ens)

        cancer_te = ctl_te >= 0
        if cancer_te.sum() > 0:
            s2_f1_lr = f1_score(ctl_te[cancer_te], cl_lr_te[cancer_te].argmax(1),
                                 average="macro", zero_division=0)
            s2_f1_rn = f1_score(ctl_te[cancer_te], rn_probs[cancer_te].argmax(1),
                                 average="macro", zero_division=0)
            s2_f1_ens = f1_score(ctl_te[cancer_te], cl_ens[cancer_te].argmax(1),
                                  average="macro", zero_division=0)
        else:
            s2_f1_lr = s2_f1_rn = s2_f1_ens = float("nan")

        results.append({
            "repeat": rep,
            "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
            "lr_s1_auc": s1_auc_lr, "lr_s2_f1": s2_f1_lr,
            "rn_s1_auc": s1_auc_rn, "rn_s2_f1": s2_f1_rn,
            "ens_s1_auc": s1_auc_ens, "ens_s2_f1": s2_f1_ens,
        })
        logger.info(f"  Repeat {rep+1}: LR AUC={s1_auc_lr:.4f}, RN AUC={s1_auc_rn:.4f}, "
                     f"Ens AUC={s1_auc_ens:.4f}, Ens F1={s2_f1_ens:.4f}")

    df_res = pd.DataFrame(results)
    return {
        "lr_s1_auc": f"{df_res['lr_s1_auc'].mean():.4f}±{df_res['lr_s1_auc'].std():.4f}",
        "rn_s1_auc": f"{df_res['rn_s1_auc'].mean():.4f}±{df_res['rn_s1_auc'].std():.4f}",
        "ens_s1_auc": f"{df_res['ens_s1_auc'].mean():.4f}±{df_res['ens_s1_auc'].std():.4f}",
        "lr_s2_f1": f"{df_res['lr_s2_f1'].mean():.4f}±{df_res['lr_s2_f1'].std():.4f}",
        "rn_s2_f1": f"{df_res['rn_s2_f1'].mean():.4f}±{df_res['rn_s2_f1'].std():.4f}",
        "ens_s2_f1": f"{df_res['ens_s2_f1'].mean():.4f}±{df_res['ens_s2_f1'].std():.4f}",
        "raw": df_res,
    }


# =============================================================================
# Phase 1: Wavenumber Grid & Preprocessing Ablation
# =============================================================================
def phase1_wavenumber_ablation():
    """Test different wavenumber ranges and feature transforms with LR."""
    logger.info("\n" + "="*64)
    logger.info("  PHASE 1: Wavenumber Grid & Preprocessing Ablation")
    logger.info("="*64)

    configs = [
        # (name, wavenumber_range, feat_transform, aggregate)
        ("full_402_2198", None, None, "medoid"),
        ("fingerprint_600_1800", (600, 1800), None, "medoid"),
        ("core_800_1600", (800, 1600), None, "medoid"),
        ("trimmed_450_2100", (450, 2100), None, "medoid"),
        ("high_region_1800_2198", (1800, 2198), None, "medoid"),
        ("low_region_402_800", (402, 800), None, "medoid"),
        # Feature transforms on full range
        ("full_deriv1", None, "derivative1", "medoid"),
        ("full_savgol_d1", None, "savgol_d1", "medoid"),
        ("full_concat_d1", None, "concat_d1", "medoid"),
        # Combined: best region + transform
        ("fp600_1800_savgol_d1", (600, 1800), "savgol_d1", "medoid"),
        ("fp600_1800_concat_d1", (600, 1800), "concat_d1", "medoid"),
    ]

    results = []
    for name, wn_range, feat_tf, agg in configs:
        logger.info(f"\n--- {name} ---")
        t0 = time.time()
        try:
            X, bl, ctl, sids, groups, mc = load_and_prepare(
                aggregate=agg, wavenumber_range=wn_range, feat_transform=feat_tf)
            metrics = run_lr_cv(X, bl, ctl, sids, groups, mc)
            elapsed = time.time() - t0
            row = {"config": name, "wn_range": str(wn_range), "feat_transform": feat_tf or "none",
                   "n_features": X.shape[1], "elapsed_s": elapsed, **metrics}
            results.append(row)
            logger.info(f"  S1 AUC: {metrics['s1_auc']:.4f}, S2 F1: {metrics['s2_f1']:.4f} ({elapsed:.1f}s)")
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            results.append({"config": name, "error": str(e)})

    df = pd.DataFrame(results)
    out_path = OUT_DIR / "phase1_wavenumber_ablation.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"\n  Phase 1 results saved to {out_path}")

    # Print summary
    logger.info("\n  Phase 1 Summary:")
    logger.info(f"  {'Config':<30} {'Features':>8} {'S1 AUC':>8} {'S2 F1':>8}")
    logger.info("  " + "-"*60)
    for _, r in df.iterrows():
        if "error" not in r or pd.isna(r.get("error")):
            logger.info(f"  {r['config']:<30} {r.get('n_features','?'):>8} "
                        f"{r.get('s1_auc',0):>8.4f} {r.get('s2_f1',0):>8.4f}")

    return df


# =============================================================================
# Phase 2: Aggregation Ablation
# =============================================================================
def phase2_aggregation_ablation(best_wn_range=None, best_feat_transform=None):
    """Test different aggregation strategies."""
    logger.info("\n" + "="*64)
    logger.info("  PHASE 2: Aggregation Ablation")
    logger.info("="*64)

    agg_methods = ["medoid", "mean", "none"]
    results = []

    for agg in agg_methods:
        logger.info(f"\n--- aggregate={agg} ---")
        t0 = time.time()
        X, bl, ctl, sids, groups, mc = load_and_prepare(
            aggregate=agg, wavenumber_range=best_wn_range, feat_transform=best_feat_transform)
        metrics = run_lr_cv(X, bl, ctl, sids, groups, mc)
        elapsed = time.time() - t0
        row = {"aggregate": agg, "n_samples": X.shape[0], "elapsed_s": elapsed, **metrics}
        results.append(row)
        logger.info(f"  S1 AUC: {metrics['s1_auc']:.4f}, S2 F1: {metrics['s2_f1']:.4f} ({elapsed:.1f}s)")

    df = pd.DataFrame(results)
    out_path = OUT_DIR / "phase2_aggregation_ablation.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"\n  Phase 2 results saved to {out_path}")
    return df


# =============================================================================
# Phase 3: Architecture + HP Grid Search
# =============================================================================
def phase3_architecture_hp_search(best_wn_range=None, best_feat_transform=None,
                                   best_aggregate="medoid"):
    """Grid search: ResNet channels + HP + ensemble alpha."""
    logger.info("\n" + "="*64)
    logger.info("  PHASE 3: Architecture + HP Grid Search")
    logger.info("="*64)

    X, bl, ctl, sids, groups, mc = load_and_prepare(
        aggregate=best_aggregate, wavenumber_range=best_wn_range,
        feat_transform=best_feat_transform)

    # First run LR baseline (needed for ensemble)
    logger.info("\n--- LR Baseline ---")
    lr_metrics = run_lr_cv(X, bl, ctl, sids, groups, mc)
    logger.info(f"  LR: S1 AUC={lr_metrics['s1_auc']:.4f}, S2 F1={lr_metrics['s2_f1']:.4f}")

    # Also get LR fold predictions for ensemble
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    composite = bl * 100 + np.clip(ctl, 0, 99)
    bp_lr = np.full(len(X), np.nan)
    cl_lr = np.full((len(X), mc.n_cancer_types), np.nan)

    for fold_i, (train_idx, val_idx) in enumerate(sgkf.split(X, composite, groups=sids)):
        pipe_s1 = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, max_iter=1000, solver="saga", class_weight="balanced", random_state=RANDOM_STATE))
        pipe_s1.fit(X[train_idx], bl[train_idx])
        bp_lr[val_idx] = pipe_s1.predict_proba(X[val_idx])[:, 1]

        cancer_tr = ctl[train_idx] >= 0
        pipe_s2 = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, max_iter=1000, solver="saga", class_weight="balanced",
            multi_class="multinomial", random_state=RANDOM_STATE))
        pipe_s2.fit(X[train_idx][cancer_tr], ctl[train_idx][cancer_tr])
        cl_lr[val_idx] = pipe_s2.predict_proba(X[val_idx])

    # ResNet grid search
    channel_configs = [
        (16, 32, 64, 128),
        (32, 64, 128, 256),
        (64, 128, 256, 512),
    ]
    lr_rates = [1e-4, 3e-4, 5e-4]
    dropouts = [0.3, 0.5]
    weight_decays = [1e-3, 1e-2]

    # Prioritized configs (not full grid — too expensive)
    configs = [
        # Baseline configs
        {"channels": (64, 128, 256, 512), "lr": 3e-4, "dropout": 0.3, "wd": 1e-3},
        {"channels": (32, 64, 128, 256), "lr": 3e-4, "dropout": 0.5, "wd": 1e-3},
        {"channels": (32, 64, 128, 256), "lr": 5e-4, "dropout": 0.5, "wd": 1e-3},
        # Wider
        {"channels": (64, 128, 256, 512), "lr": 3e-4, "dropout": 0.5, "wd": 1e-3},
        {"channels": (64, 128, 256, 512), "lr": 1e-4, "dropout": 0.3, "wd": 1e-3},
        {"channels": (64, 128, 256, 512), "lr": 5e-4, "dropout": 0.3, "wd": 1e-2},
        # Smaller
        {"channels": (16, 32, 64, 128), "lr": 3e-4, "dropout": 0.3, "wd": 1e-3},
        {"channels": (16, 32, 64, 128), "lr": 5e-4, "dropout": 0.5, "wd": 1e-3},
        # Deeper regularization
        {"channels": (64, 128, 256, 512), "lr": 3e-4, "dropout": 0.3, "wd": 1e-2},
        {"channels": (32, 64, 128, 256), "lr": 3e-4, "dropout": 0.3, "wd": 1e-2},
    ]

    results = []

    for i, cfg in enumerate(configs):
        name = f"ch{'_'.join(map(str, cfg['channels']))}_lr{cfg['lr']}_do{cfg['dropout']}_wd{cfg['wd']}"
        logger.info(f"\n--- Config {i+1}/{len(configs)}: {name} ---")
        t0 = time.time()

        try:
            rn_metrics = run_resnet_cv(
                X, bl, ctl, sids, groups, mc,
                lr=cfg["lr"], dropout=cfg["dropout"], wd=cfg["wd"],
                channels=cfg["channels"],
            )
            elapsed = time.time() - t0

            # Test ensemble with different alphas
            best_ens = None
            for alpha in [0.5, 0.6, 0.7, 0.8, 0.9]:
                ens = run_ensemble(bp_lr, cl_lr, rn_metrics["all_bp"], rn_metrics["all_cl"],
                                   bl, ctl, alpha=alpha)
                if best_ens is None or ens["s1_auc"] > best_ens["s1_auc"]:
                    best_ens = ens

            row = {
                "config": name,
                **{f"rn_{k}": v for k, v in rn_metrics.items() if k not in ("all_bp", "all_cl")},
                **{f"ens_{k}": v for k, v in best_ens.items()},
                "elapsed_s": elapsed,
            }
            results.append(row)
            logger.info(f"  RN: AUC={rn_metrics['s1_auc']:.4f}, F1={rn_metrics['s2_f1']:.4f}")
            logger.info(f"  Ensemble (α={best_ens['alpha']}): AUC={best_ens['s1_auc']:.4f}, F1={best_ens['s2_f1']:.4f}")
            logger.info(f"  ({elapsed:.0f}s)")
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            results.append({"config": name, "error": str(e)})

    df = pd.DataFrame(results)
    out_path = OUT_DIR / "phase3_architecture_hp_search.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"\n  Phase 3 results saved to {out_path}")

    # Print summary
    logger.info("\n  Phase 3 Summary:")
    logger.info(f"  {'Config':<50} {'RN AUC':>8} {'Ens AUC':>8} {'Ens F1':>8} {'Alpha':>6}")
    logger.info("  " + "-"*84)
    for _, r in df.iterrows():
        if "error" not in r or pd.isna(r.get("error")):
            logger.info(f"  {r['config']:<50} {r.get('rn_s1_auc',0):>8.4f} "
                        f"{r.get('ens_s1_auc',0):>8.4f} {r.get('ens_s2_f1',0):>8.4f} "
                        f"{r.get('ens_alpha',0):>6.1f}")

    return df


# =============================================================================
# Phase 4: Final Evaluation
# =============================================================================
def phase4_final_evaluation(best_wn_range=None, best_feat_transform=None,
                            best_aggregate="medoid", best_rn_config=None, best_alpha=0.8):
    """Final eval: 5-fold CV + held-out test (5 repeats)."""
    logger.info("\n" + "="*64)
    logger.info("  PHASE 4: Final Evaluation")
    logger.info("="*64)

    X, bl, ctl, sids, groups, mc = load_and_prepare(
        aggregate=best_aggregate, wavenumber_range=best_wn_range,
        feat_transform=best_feat_transform)

    rn_config = best_rn_config or {
        "lr": 3e-4, "dropout": 0.3, "wd": 1e-3,
        "channels": (64, 128, 256, 512), "n_epochs": 150,
    }

    # --- 5-fold CV ---
    logger.info("\n--- 5-Fold CV ---")
    lr_metrics = run_lr_cv(X, bl, ctl, sids, groups, mc)
    rn_metrics = run_resnet_cv(X, bl, ctl, sids, groups, mc, **{
        k: v for k, v in rn_config.items() if k in ("lr", "dropout", "wd", "bs", "channels", "n_epochs")
    })

    # Get LR fold predictions for ensemble
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    composite = bl * 100 + np.clip(ctl, 0, 99)
    bp_lr = np.full(len(X), np.nan)
    cl_lr = np.full((len(X), mc.n_cancer_types), np.nan)
    for fold_i, (train_idx, val_idx) in enumerate(sgkf.split(X, composite, groups=sids)):
        pipe_s1 = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, max_iter=1000, solver="saga", class_weight="balanced", random_state=RANDOM_STATE))
        pipe_s1.fit(X[train_idx], bl[train_idx])
        bp_lr[val_idx] = pipe_s1.predict_proba(X[val_idx])[:, 1]
        cancer_tr = ctl[train_idx] >= 0
        pipe_s2 = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, max_iter=1000, solver="saga", class_weight="balanced",
            multi_class="multinomial", random_state=RANDOM_STATE))
        pipe_s2.fit(X[train_idx][cancer_tr], ctl[train_idx][cancer_tr])
        cl_lr[val_idx] = pipe_s2.predict_proba(X[val_idx])

    ens_cv = run_ensemble(bp_lr, cl_lr, rn_metrics["all_bp"], rn_metrics["all_cl"],
                           bl, ctl, alpha=best_alpha)

    cv_results = {
        "method": "5-fold CV",
        "lr_s1_auc": lr_metrics["s1_auc"], "lr_s2_f1": lr_metrics["s2_f1"],
        "rn_s1_auc": rn_metrics["s1_auc"], "rn_s2_f1": rn_metrics["s2_f1"],
        "ens_s1_auc": ens_cv["s1_auc"], "ens_s2_f1": ens_cv["s2_f1"],
        "ens_s1_sens": ens_cv["s1_sens"], "ens_s1_spec": ens_cv["s1_spec"],
    }

    logger.info(f"  5-fold CV:")
    logger.info(f"    LR:       AUC={lr_metrics['s1_auc']:.4f}, F1={lr_metrics['s2_f1']:.4f}")
    logger.info(f"    ResNet18: AUC={rn_metrics['s1_auc']:.4f}, F1={rn_metrics['s2_f1']:.4f}")
    logger.info(f"    Ensemble: AUC={ens_cv['s1_auc']:.4f}, F1={ens_cv['s2_f1']:.4f}")

    # --- Held-out Test ---
    logger.info("\n--- Held-out Test (5 repeats) ---")
    hot_results = run_held_out_test(
        X, bl, ctl, sids, groups, mc,
        rn_config=rn_config, alpha=best_alpha, n_repeats=5)

    logger.info(f"\n  Held-out Test (mean±std):")
    logger.info(f"    LR:       AUC={hot_results['lr_s1_auc']}, F1={hot_results['lr_s2_f1']}")
    logger.info(f"    ResNet18: AUC={hot_results['rn_s1_auc']}, F1={hot_results['rn_s2_f1']}")
    logger.info(f"    Ensemble: AUC={hot_results['ens_s1_auc']}, F1={hot_results['ens_s2_f1']}")

    # Save
    summary = {
        "cv_results": cv_results,
        "held_out_results": {k: v for k, v in hot_results.items() if k != "raw"},
        "config": {
            "cancer_types": CANCER_TYPES,
            "non_cancer": NON_CANCER,
            "wn_range": str(best_wn_range),
            "feat_transform": best_feat_transform,
            "aggregate": best_aggregate,
            "rn_config": rn_config,
            "alpha": best_alpha,
        },
    }
    out_path = OUT_DIR / "phase4_final_evaluation.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    hot_results["raw"].to_csv(OUT_DIR / "phase4_held_out_repeats.csv", index=False)

    logger.info(f"\n  Phase 4 results saved to {out_path}")
    return summary


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="all", help="1, 2, 3, 4, or all")
    args = parser.parse_args()

    t0 = datetime.now()
    logger.info("="*64)
    logger.info("  7-Cancer Full Optimization — uSERS-Net")
    logger.info(f"  Started: {t0.isoformat()}")
    logger.info(f"  Output: {OUT_DIR}")
    logger.info("="*64)

    phases = args.phase.split(",") if args.phase != "all" else ["1", "2", "3", "4"]

    # Carry-over best configs between phases
    best_wn_range = None
    best_feat_transform = None
    best_aggregate = "medoid"
    best_rn_config = None
    best_alpha = 0.8

    if "1" in phases:
        df1 = phase1_wavenumber_ablation()
        # Pick best by S1 AUC
        valid = df1.dropna(subset=["s1_auc"])
        if len(valid) > 0:
            best_row = valid.loc[valid["s1_auc"].idxmax()]
            best_config_name = best_row["config"]
            logger.info(f"\n  ★ Best wavenumber config: {best_config_name} "
                        f"(AUC={best_row['s1_auc']:.4f}, F1={best_row['s2_f1']:.4f})")

            # Parse best config
            wn_str = best_row.get("wn_range", "None")
            if wn_str != "None" and wn_str != "nan":
                parts = wn_str.strip("()").split(",")
                best_wn_range = (float(parts[0]), float(parts[1]))
            ft = best_row.get("feat_transform", "none")
            best_feat_transform = None if ft == "none" else ft

    if "2" in phases:
        df2 = phase2_aggregation_ablation(best_wn_range, best_feat_transform)
        valid = df2.dropna(subset=["s1_auc"])
        if len(valid) > 0:
            best_row = valid.loc[valid["s1_auc"].idxmax()]
            best_aggregate = best_row["aggregate"]
            logger.info(f"\n  ★ Best aggregation: {best_aggregate} "
                        f"(AUC={best_row['s1_auc']:.4f}, F1={best_row['s2_f1']:.4f})")

    if "3" in phases:
        df3 = phase3_architecture_hp_search(best_wn_range, best_feat_transform, best_aggregate)
        valid = df3.dropna(subset=["ens_s1_auc"])
        if len(valid) > 0:
            best_row = valid.loc[valid["ens_s1_auc"].idxmax()]
            cfg_name = best_row["config"]
            # Parse config from name
            parts = cfg_name.split("_")
            # Extract channels, lr, dropout, wd from config name
            ch_parts = []
            lr_val = 3e-4
            do_val = 0.3
            wd_val = 1e-3

            i = 0
            while i < len(parts):
                if parts[i] == "ch":
                    # Collect channel numbers
                    i += 1
                    while i < len(parts) and parts[i].isdigit():
                        ch_parts.append(int(parts[i]))
                        i += 1
                elif parts[i].startswith("lr"):
                    lr_val = float(parts[i][2:])
                    i += 1
                elif parts[i].startswith("do"):
                    do_val = float(parts[i][2:])
                    i += 1
                elif parts[i].startswith("wd"):
                    wd_val = float(parts[i][2:])
                    i += 1
                else:
                    i += 1

            best_rn_config = {
                "lr": lr_val, "dropout": do_val, "wd": wd_val,
                "channels": tuple(ch_parts) if ch_parts else (64, 128, 256, 512),
                "n_epochs": 150,
            }
            best_alpha = best_row.get("ens_alpha", 0.8)
            logger.info(f"\n  ★ Best architecture: {cfg_name}")
            logger.info(f"    Ens AUC={best_row['ens_s1_auc']:.4f}, F1={best_row['ens_s2_f1']:.4f}, α={best_alpha}")

    if "4" in phases:
        phase4_final_evaluation(
            best_wn_range, best_feat_transform, best_aggregate,
            best_rn_config, best_alpha)

    elapsed = datetime.now() - t0
    logger.info(f"\n{'='*64}")
    logger.info(f"  All phases complete! ({elapsed})")
    logger.info(f"  Results: {OUT_DIR}/")
    logger.info(f"{'='*64}")


if __name__ == "__main__":
    main()
