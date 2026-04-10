#!/usr/bin/env python3
"""
STK-V2: Stacking / Meta-Learner Optimization
=============================================

Extended base models (10) x meta-learners (5) x nested CV (outer 5 x inner 3)
for unbiased ensemble performance estimation.

Usage:
    python models/train_stacking.py
    python models/train_stacking.py --dry-run
    python models/train_stacking.py --val-group SPAN --meta-learner elasticnet
"""

from __future__ import annotations

import sys
import os
import json
import argparse
import logging
import tempfile
import shutil
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from scipy.special import voigt_profile
from scipy.integrate import trapezoid

from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, f1_score

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.stacking_utils import preprocess_channel, load_raw_multichannel, build_classifier, train_base_model  # noqa: E402
from src.sers.config import RESULTS_DIR, FIG_DIR  # noqa: E402

OUTPUT_DIR = RESULTS_DIR / "training" / "stacking_v2"
FIG_OUTPUT_DIR = FIG_DIR / "training" / "stacking_v2"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

# Legacy path — load existing OOF/checkpoints if new dir is empty
_LEGACY_OUTPUT_DIR = RESULTS_DIR / "weekend_experiments" / "stacking_optimization"
_LEGACY_CHECKPOINT_DIR = _LEGACY_OUTPUT_DIR / "checkpoints"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CANCER_TYPES = ("PRO", "LUN", "CRC", "PAN", "OVA", "BRE", "BLC")
NON_CANCER_GROUPS = ("NOR", "DIA", "HBP", "H.D.")

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

KNOWN_PEAKS = [
    (448.1, "ring_deform", 15), (538.7, "SS_stretch", 15),
    (617.7, "CS_stretch", 15), (683.3, "creatinine", 15),
    (723.8, "adenine", 15), (795.1, "hippuric", 15),
    (849.1, "tyrosine", 15), (895.4, "uric_acid", 15),
    (933.9, "creatinine2", 15), (999.5, "phe_urea", 15),
    (1147.9, "uric_CN", 15), (1230.8, "amide_III", 20),
    (1292.5, "CH2_twist", 15), (1352.3, "trp_fermi", 15),
    (1448.7, "CH2_deform", 20), (1597.1, "purine_CC", 20),
    (1651.1, "amide_I", 20),
]

# ---------------------------------------------------------------------------
# Extended base models
# ---------------------------------------------------------------------------
EXTENDED_BASE_MODELS = {
    "lr_raw":       {"channels": [0],       "model": "lr",    "pca": None},
    "lr_d1":        {"channels": [1],       "model": "lr",    "pca": None},
    "lr_d2":        {"channels": [2],       "model": "lr",    "pca": None},
    "lr_concat":    {"channels": [0, 1, 2], "model": "lr",    "pca": None},
    "lr_peak":      {"channels": "peak",    "model": "lr",    "pca": None},
    "xgb_raw":      {"channels": [0],       "model": "xgb",   "pca": None},
    "xgb_d1":       {"channels": [1],       "model": "xgb",   "pca": None},
    "rf_raw":       {"channels": [0],       "model": "rf",    "pca": None},
    "rf_d1":        {"channels": [1],       "model": "rf",    "pca": None},
    "ridge_concat": {"channels": [0, 1, 2], "model": "ridge", "pca": None},
}

# ---------------------------------------------------------------------------
# Peak feature extraction (Voigt fitting)
# ---------------------------------------------------------------------------

def voigt_func(x, amplitude, center, sigma, gamma):
    return amplitude * voigt_profile(x - center, sigma, gamma)


def extract_peak_features(X, wavenumbers):
    """Extract peak features from raw spectra using Voigt fitting."""
    n_samples, n_peaks = len(X), len(KNOWN_PEAKS)
    peak_areas = np.zeros((n_samples, n_peaks))
    peak_heights = np.zeros((n_samples, n_peaks))
    peak_fwhms = np.zeros((n_samples, n_peaks))
    peak_shifts = np.zeros((n_samples, n_peaks))

    for pi, (center, name, half_w) in enumerate(KNOWN_PEAKS):
        mask = (wavenumbers >= center - half_w * 1.5) & (wavenumbers <= center + half_w * 1.5)
        if mask.sum() < 5:
            continue
        x_region = wavenumbers[mask]
        for si in range(n_samples):
            y_shifted = X[si, mask] - np.min(X[si, mask])
            try:
                amp_guess = max(np.max(y_shifted), 1e-6)
                popt, _ = curve_fit(
                    voigt_func, x_region, y_shifted,
                    p0=[amp_guess, center, half_w / 3, half_w / 3],
                    bounds=(
                        [0, center - half_w, 0.1, 0.1],
                        [amp_guess * 10, center + half_w, half_w * 2, half_w * 2],
                    ),
                    maxfev=2000,
                )
                amp, ctr, sigma, gamma = popt
                y_fit = voigt_func(x_region, *popt)
                area = trapezoid(y_fit, x_region)
                fL = 2 * gamma
                fG = 2 * sigma * np.sqrt(2 * np.log(2))
                fwhm = 0.5346 * fL + np.sqrt(0.2166 * fL ** 2 + fG ** 2)
            except (RuntimeError, ValueError):
                amp, ctr, area, fwhm = 0.0, center, 0.0, 0.0
            peak_areas[si, pi] = area
            peak_heights[si, pi] = amp
            peak_fwhms[si, pi] = fwhm
            peak_shifts[si, pi] = ctr - center

    features = [peak_areas, peak_heights, peak_fwhms, peak_shifts]
    pn2i = {p[1]: i for i, p in enumerate(KNOWN_PEAKS)}
    for na, nb in [
        ("phe_urea", "adenine"), ("phe_urea", "creatinine"),
        ("hippuric", "creatinine"), ("CS_stretch", "creatinine"),
        ("amide_I", "CH2_deform"), ("adenine", "purine_CC"),
        ("tyrosine", "phe_urea"),
    ]:
        features.append(
            peak_areas[:, pn2i[na]:pn2i[na] + 1]
            / (peak_areas[:, pn2i[nb]:pn2i[nb] + 1] + 1e-10)
        )
    return np.nan_to_num(np.hstack(features), nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# Extended train_base_model (handles "peak" channel type)
# ---------------------------------------------------------------------------

def train_base_model_ext(spec, X_train, y_bin_train, y_type_train, X_val,
                         n_classes, X_peak_train=None, X_peak_val=None):
    """Train a base model for both stages. Supports peak features."""
    ch = spec["channels"]
    pca_n = spec["pca"]
    model_type = spec["model"]

    if ch == "peak":
        X_tr = X_peak_train
        X_va = X_peak_val
    else:
        X_tr = X_train[:, ch, :].reshape(len(X_train), -1)
        X_va = X_val[:, ch, :].reshape(len(X_val), -1)

    if pca_n:
        pca = PCA(n_components=pca_n, random_state=42)
        scaler = StandardScaler()
        X_tr = pca.fit_transform(scaler.fit_transform(X_tr))
        X_va = pca.transform(scaler.transform(X_va))

    # Stage 1: binary
    s1 = build_classifier(model_type, "binary")
    s1.fit(X_tr, y_bin_train)
    s1_prob = s1.predict_proba(X_va)[:, 1]

    # Stage 2: cancer type (cancer only)
    cancer_mask = y_bin_train == 1
    s2_prob = np.zeros((len(X_va), n_classes))
    if cancer_mask.sum() > 10:
        s2 = build_classifier(model_type, "multiclass", n_classes)
        s2.fit(X_tr[cancer_mask], y_type_train[cancer_mask])
        raw_prob = s2.predict_proba(X_va)
        for i, cls in enumerate(s2.classes_):
            if cls < n_classes:
                s2_prob[:, cls] = raw_prob[:, i]

    return s1_prob, s2_prob


# ---------------------------------------------------------------------------
# Meta-learner factory
# ---------------------------------------------------------------------------

def make_meta_learners(n_classes):
    """Return dict of meta-learner factories keyed by name."""
    try:
        from xgboost import XGBClassifier
        xgb_available = True
    except ImportError:
        xgb_available = False

    def _lr(task):
        return LogisticRegression(
            C=1.0, max_iter=2000, solver="lbfgs",
            multi_class="multinomial" if task == "multiclass" else "auto",
        )

    def _xgb(task):
        if not xgb_available:
            return _lr(task)
        kw = dict(n_estimators=200, max_depth=4, learning_rate=0.05,
                  n_jobs=2, verbosity=0)
        if task == "multiclass":
            kw.update(objective="multi:softprob", num_class=n_classes)
        else:
            kw.update(objective="binary:logistic")
        return XGBClassifier(**kw)

    def _rf(task):
        return RandomForestClassifier(
            n_estimators=300, max_depth=5,
            class_weight="balanced_subsample", n_jobs=2, random_state=42,
        )

    def _elasticnet(task):
        return LogisticRegression(
            C=0.5, penalty="elasticnet", l1_ratio=0.5,
            max_iter=2000, solver="saga",
            multi_class="multinomial" if task == "multiclass" else "auto",
        )

    def _mlp(task):
        return MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=500,
            early_stopping=True, validation_fraction=0.15, random_state=42,
        )

    return {"lr": _lr, "xgb": _xgb, "rf": _rf, "elasticnet": _elasticnet, "mlp": _mlp}


# ---------------------------------------------------------------------------
# Atomic file write helper
# ---------------------------------------------------------------------------

def atomic_save_npz(path, **arrays):
    """Save .npz atomically via temp + rename."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(tmp, **arrays)
        shutil.move(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def atomic_save_json(path, data):
    """Save JSON atomically via temp + rename."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(suffix=".json", dir=path.parent)
    os.close(fd)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass
    try:
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        shutil.move(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def oof_checkpoint_path(model_name, outer_fold):
    return CHECKPOINT_DIR / f"oof_{model_name}_outerfold{outer_fold}.npz"


def meta_checkpoint_path(meta_type, outer_fold):
    return CHECKPOINT_DIR / f"meta_{meta_type}_outerfold{outer_fold}.json"


def load_oof_checkpoint(model_name, outer_fold):
    p = oof_checkpoint_path(model_name, outer_fold)
    if not p.exists():
        p = _LEGACY_CHECKPOINT_DIR / p.name
    if p.exists():
        d = np.load(p)
        return d["s1_prob"], d["s2_prob"]
    return None


def load_meta_checkpoint(meta_type, outer_fold):
    p = meta_checkpoint_path(meta_type, outer_fold)
    if not p.exists():
        p = _LEGACY_CHECKPOINT_DIR / p.name
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    """Load and prepare multichannel SERS data with mean aggregation."""
    grid = np.linspace(402.0, 2198.0, 933)
    data_dir = PROJECT_ROOT / "data" / "raw_data"

    logger.info("Loading raw multichannel spectra ...")
    X_3ch, df_meta = load_raw_multichannel(data_dir, THERMO_MAP, grid, "*.CSV")
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

    # Mean aggregation per sample
    agg_groups = df_meta.groupby(["group", "sample_id"]).first().reset_index()
    X_agg = []
    for _, row in agg_groups.iterrows():
        mask_s = (df_meta["group"] == row["group"]) & (df_meta["sample_id"] == row["sample_id"])
        X_agg.append(X_3ch[mask_s.values].mean(axis=0))
    X_3ch_agg = np.stack(X_agg)
    df_meta_agg = agg_groups.reset_index(drop=True)

    groups_arr = df_meta_agg["group"].values
    sample_ids = (df_meta_agg["group"] + "_" + df_meta_agg["sample_id"].astype(str)).values
    binary_labels = np.array([1.0 if g in CANCER_TYPES else 0.0 for g in groups_arr])
    ct_map = {ct: i for i, ct in enumerate(CANCER_TYPES)}
    cancer_type_labels = np.array([ct_map.get(g, -1) for g in groups_arr])

    logger.info(
        f"  Aggregated: {len(X_3ch_agg)} samples, "
        f"{int(binary_labels.sum())} cancer, {int((binary_labels == 0).sum())} non-cancer"
    )
    return X_3ch_agg, df_meta_agg, grid, groups_arr, sample_ids, binary_labels, cancer_type_labels


# ---------------------------------------------------------------------------
# Generate OOF predictions for one base model across inner CV
# ---------------------------------------------------------------------------

def generate_oof_inner(
    model_name, spec, X_outer_train, y_bin_outer, y_type_outer,
    sample_ids_outer, n_classes, grid, X_peak_outer, inner_splits=5,
):
    """
    Inner CV on outer-train set to produce OOF predictions for one base model.
    Returns arrays (s1_oof, s2_oof) of shape (n_outer_train,) and (n_outer_train, n_classes).
    """
    n = len(X_outer_train)
    s1_oof = np.full(n, np.nan)
    s2_oof = np.full((n, n_classes), np.nan)

    inner_sgkf = StratifiedGroupKFold(n_splits=inner_splits, shuffle=True, random_state=123)
    for inner_fold, (itr, ival) in enumerate(inner_sgkf.split(
        X_outer_train, y_bin_outer, sample_ids_outer
    )):
        if spec["channels"] == "peak":
            pk_tr = X_peak_outer[itr]
            pk_va = X_peak_outer[ival]
        else:
            pk_tr, pk_va = None, None

        s1_p, s2_p = train_base_model_ext(
            spec, X_outer_train[itr], y_bin_outer[itr],
            y_type_outer[itr], X_outer_train[ival], n_classes,
            X_peak_train=pk_tr, X_peak_val=pk_va,
        )
        s1_oof[ival] = s1_p
        s2_oof[ival] = s2_p

    return s1_oof, s2_oof


# ---------------------------------------------------------------------------
# Evaluate a single meta-learner on outer-train meta-features via inner CV
# ---------------------------------------------------------------------------

def evaluate_meta_learner_inner(
    meta_factory, meta_s1, meta_s2,
    y_bin, y_type, sample_ids, n_classes, inner_splits=3,
):
    """
    Inner 3-fold CV to evaluate a meta-learner on the outer-train set.
    Returns mean (s1_auc, s2_f1).
    """
    inner_sgkf = StratifiedGroupKFold(n_splits=inner_splits, shuffle=True, random_state=456)
    aucs, f1s = [], []

    for itr, ival in inner_sgkf.split(meta_s1, y_bin, sample_ids):
        # S1 meta-learner
        ml_s1 = meta_factory("binary")
        ml_s1.fit(meta_s1[itr], y_bin[itr])
        try:
            s1_pred = ml_s1.predict_proba(meta_s1[ival])[:, 1]
        except AttributeError:
            s1_pred = ml_s1.decision_function(meta_s1[ival])
        auc = roc_auc_score(y_bin[ival], s1_pred)
        aucs.append(auc)

        # S2 meta-learner (cancer only)
        cancer_tr = y_bin[itr] == 1
        cancer_va = y_bin[ival] == 1
        if cancer_tr.sum() > 5 and cancer_va.sum() > 0:
            ml_s2 = meta_factory("multiclass")
            ml_s2.fit(meta_s2[itr][cancer_tr], y_type[itr][cancer_tr])
            s2_pred = ml_s2.predict(meta_s2[ival][cancer_va])
            f1 = f1_score(y_type[ival][cancer_va], s2_pred, average="macro", zero_division=0)
            f1s.append(f1)
        else:
            f1s.append(0.0)

    return float(np.mean(aucs)), float(np.mean(f1s))


# ---------------------------------------------------------------------------
# Main nested CV
# ---------------------------------------------------------------------------

def run_nested_cv(
    X_3ch_agg, grid, sample_ids, binary_labels, cancer_type_labels,
    n_classes, base_models, meta_learner_factories, dry_run=False,
):
    """
    Nested CV: outer 5-fold, inner 5-fold for OOF, inner 3-fold for meta selection.
    """
    n = len(X_3ch_agg)
    outer_splits = 5 if not dry_run else 1
    inner_oof_splits = 5
    inner_meta_splits = 3

    if dry_run:
        # Subset base models and meta-learners
        bm_names = list(base_models.keys())[:3]
        base_models = {k: base_models[k] for k in bm_names}
        ml_names = list(meta_learner_factories.keys())[:2]
        meta_learner_factories = {k: meta_learner_factories[k] for k in ml_names}
        logger.info(f"  [DRY-RUN] {outer_splits} outer fold, {len(base_models)} base models, {len(meta_learner_factories)} meta-learners")

    model_names = list(base_models.keys())
    meta_names = list(meta_learner_factories.keys())
    n_models = len(model_names)
    n_meta = len(meta_names)

    # Precompute peak features for full dataset (channel 0)
    logger.info("  Extracting peak features (Voigt fitting) ...")
    X_peak_all = extract_peak_features(X_3ch_agg[:, 0, :], grid)
    logger.info(f"  Peak features shape: {X_peak_all.shape}")

    outer_sgkf = StratifiedGroupKFold(n_splits=max(outer_splits, 2), shuffle=True, random_state=42)
    all_splits = list(outer_sgkf.split(X_3ch_agg, binary_labels, sample_ids))
    if dry_run:
        all_splits = all_splits[:1]

    # Storage for results
    all_results = []  # list of dicts
    outer_oof_s1 = {m: np.full(n, np.nan) for m in model_names}
    outer_oof_s2 = {m: np.full((n, n_classes), np.nan) for m in model_names}
    best_meta_per_fold = {}

    for outer_fold, (outer_tr, outer_te) in enumerate(all_splits):
        logger.info(f"\n{'='*60}")
        logger.info(f"  OUTER FOLD {outer_fold} ({len(outer_tr)} train, {len(outer_te)} test)")
        logger.info(f"{'='*60}")

        X_outer_train = X_3ch_agg[outer_tr]
        X_outer_test = X_3ch_agg[outer_te]
        y_bin_outer = binary_labels[outer_tr]
        y_type_outer = cancer_type_labels[outer_tr]
        sids_outer = sample_ids[outer_tr]
        X_peak_outer = X_peak_all[outer_tr]
        X_peak_test = X_peak_all[outer_te]

        # ===== Level 0: Generate OOF predictions for each base model =====
        logger.info(f"\n  -- Level 0: OOF for {n_models} base models (inner {inner_oof_splits}-fold) --")
        oof_s1_train = {}
        oof_s2_train = {}
        # Also generate predictions on outer-test for final evaluation
        test_s1 = {}
        test_s2 = {}

        for mi, (m_name, spec) in enumerate(base_models.items()):
            # Check checkpoint
            cached = load_oof_checkpoint(m_name, outer_fold)
            if cached is not None:
                logger.info(f"    [{mi+1}/{n_models}] {m_name} -- loaded from checkpoint")
                oof_s1_train[m_name] = cached[0]
                oof_s2_train[m_name] = cached[1]
            else:
                logger.info(f"    [{mi+1}/{n_models}] {m_name} -- training inner CV ...")
                s1_oof, s2_oof = generate_oof_inner(
                    m_name, spec, X_outer_train, y_bin_outer, y_type_outer,
                    sids_outer, n_classes, grid, X_peak_outer,
                    inner_splits=inner_oof_splits,
                )
                oof_s1_train[m_name] = s1_oof
                oof_s2_train[m_name] = s2_oof
                # Save checkpoint
                atomic_save_npz(
                    oof_checkpoint_path(m_name, outer_fold),
                    s1_prob=s1_oof, s2_prob=s2_oof,
                )

            # Train on full outer-train, predict on outer-test
            if spec["channels"] == "peak":
                pk_tr, pk_te = X_peak_outer, X_peak_test
            else:
                pk_tr, pk_te = None, None

            s1_te, s2_te = train_base_model_ext(
                spec, X_outer_train, y_bin_outer, y_type_outer,
                X_outer_test, n_classes,
                X_peak_train=pk_tr, X_peak_val=pk_te,
            )
            test_s1[m_name] = s1_te
            test_s2[m_name] = s2_te

            # Store for global OOF
            outer_oof_s1[m_name][outer_te] = s1_te
            outer_oof_s2[m_name][outer_te] = s2_te

            # Log OOF quality
            valid_mask = ~np.isnan(oof_s1_train[m_name])
            if valid_mask.sum() > 10:
                auc_inner = roc_auc_score(y_bin_outer[valid_mask], oof_s1_train[m_name][valid_mask])
                logger.info(f"      OOF AUC (inner): {auc_inner:.4f}")

        # ===== Level 1: Stack meta-features & evaluate meta-learners =====
        logger.info(f"\n  -- Level 1: Evaluating {n_meta} meta-learners (inner {inner_meta_splits}-fold) --")

        # Build meta-feature matrices from OOF predictions on outer-train
        meta_s1_train = np.column_stack([oof_s1_train[m] for m in model_names])
        meta_s2_train = np.hstack([oof_s2_train[m] for m in model_names])

        # Build meta-feature matrices for outer-test
        meta_s1_test = np.column_stack([test_s1[m] for m in model_names])
        meta_s2_test = np.hstack([test_s2[m] for m in model_names])

        # Replace NaNs in meta-features with 0 (from inner CV)
        meta_s1_train = np.nan_to_num(meta_s1_train, nan=0.0)
        meta_s2_train = np.nan_to_num(meta_s2_train, nan=0.0)

        best_meta_name = None
        best_meta_score = -1.0
        meta_fold_results = {}

        for ml_name, ml_factory in meta_learner_factories.items():
            cached_meta = load_meta_checkpoint(ml_name, outer_fold)
            if cached_meta is not None:
                inner_auc = cached_meta["inner_s1_auc"]
                inner_f1 = cached_meta["inner_s2_f1"]
                logger.info(f"    {ml_name:<12} (cached) inner AUC={inner_auc:.4f}  F1={inner_f1:.4f}")
            else:
                inner_auc, inner_f1 = evaluate_meta_learner_inner(
                    ml_factory, meta_s1_train, meta_s2_train,
                    y_bin_outer, y_type_outer, sids_outer, n_classes,
                    inner_splits=inner_meta_splits,
                )
                atomic_save_json(
                    meta_checkpoint_path(ml_name, outer_fold),
                    {"inner_s1_auc": inner_auc, "inner_s2_f1": inner_f1},
                )
                logger.info(f"    {ml_name:<12} inner AUC={inner_auc:.4f}  F1={inner_f1:.4f}")

            meta_fold_results[ml_name] = {"inner_s1_auc": inner_auc, "inner_s2_f1": inner_f1}
            combined = inner_auc * 0.6 + inner_f1 * 0.4
            if combined > best_meta_score:
                best_meta_score = combined
                best_meta_name = ml_name

        best_meta_per_fold[outer_fold] = best_meta_name
        logger.info(f"  => Best meta-learner for fold {outer_fold}: {best_meta_name}")

        # ===== Train ALL meta-learners on full outer-train, predict outer-test =====
        for ml_name, ml_factory in meta_learner_factories.items():
            # S1
            ml_s1 = ml_factory("binary")
            ml_s1.fit(meta_s1_train, y_bin_outer)
            try:
                s1_pred = ml_s1.predict_proba(meta_s1_test)[:, 1]
            except AttributeError:
                s1_pred = ml_s1.decision_function(meta_s1_test)

            s1_auc = roc_auc_score(binary_labels[outer_te], s1_pred)

            # S2
            cancer_tr = y_bin_outer == 1
            cancer_te = binary_labels[outer_te] == 1
            s2_f1 = 0.0
            if cancer_tr.sum() > 5 and cancer_te.sum() > 0:
                ml_s2 = ml_factory("multiclass")
                ml_s2.fit(meta_s2_train[cancer_tr], y_type_outer[cancer_tr])
                s2_pred = ml_s2.predict(meta_s2_test[cancer_te])
                s2_f1 = f1_score(
                    cancer_type_labels[outer_te][cancer_te], s2_pred,
                    average="macro", zero_division=0,
                )

            all_results.append({
                "outer_fold": outer_fold,
                "meta_learner": ml_name,
                "s1_auc": round(s1_auc, 6),
                "s2_f1": round(s2_f1, 6),
                "is_best_inner": ml_name == best_meta_name,
            })
            logger.info(f"    {ml_name:<12} outer-test AUC={s1_auc:.4f}  F1={s2_f1:.4f}")

    return all_results, outer_oof_s1, outer_oof_s2, model_names, meta_names, best_meta_per_fold, X_peak_all


# ---------------------------------------------------------------------------
# Permutation importance for base model contribution
# ---------------------------------------------------------------------------

def compute_base_model_contribution(
    outer_oof_s1, model_names, binary_labels, cancer_type_labels, n_classes,
    n_repeats=10,
):
    """
    Compute contribution of each base model via permutation importance.
    Shuffle its OOF predictions and measure performance drop.
    """
    # Build meta-features from global OOF
    meta_s1 = np.column_stack([outer_oof_s1[m] for m in model_names])
    valid = ~np.isnan(meta_s1).any(axis=1)
    meta_s1_v = meta_s1[valid]
    y_bin_v = binary_labels[valid]

    baseline_auc = roc_auc_score(y_bin_v, meta_s1_v.mean(axis=1))

    contributions = []
    rng = np.random.RandomState(42)

    for mi, m_name in enumerate(model_names):
        drops = []
        for _ in range(n_repeats):
            meta_shuffled = meta_s1_v.copy()
            rng.shuffle(meta_shuffled[:, mi])
            shuffled_auc = roc_auc_score(y_bin_v, meta_shuffled.mean(axis=1))
            drops.append(baseline_auc - shuffled_auc)
        contributions.append({
            "base_model": m_name,
            "mean_auc_drop": round(float(np.mean(drops)), 6),
            "std_auc_drop": round(float(np.std(drops)), 6),
            "baseline_auc": round(baseline_auc, 6),
        })

    return sorted(contributions, key=lambda x: x["mean_auc_drop"], reverse=True)


# ---------------------------------------------------------------------------
# Single model vs ensemble comparison
# ---------------------------------------------------------------------------

def compare_single_vs_ensemble(outer_oof_s1, outer_oof_s2, model_names,
                               binary_labels, cancer_type_labels, n_classes):
    """Compare best single model performance vs ensemble."""
    rows = []
    for m_name in model_names:
        valid = ~np.isnan(outer_oof_s1[m_name])
        if valid.sum() < 20:
            continue
        auc = roc_auc_score(binary_labels[valid], outer_oof_s1[m_name][valid])
        cancer_mask = (binary_labels == 1) & valid
        if cancer_mask.sum() > 0:
            f1 = f1_score(
                cancer_type_labels[cancer_mask],
                outer_oof_s2[m_name][cancer_mask].argmax(axis=1),
                average="macro", zero_division=0,
            )
        else:
            f1 = 0.0
        rows.append({"model": m_name, "type": "single", "auc": round(auc, 6), "f1_type": round(f1, 6)})

    # Simple average ensemble
    all_s1 = np.column_stack([outer_oof_s1[m] for m in model_names])
    valid = ~np.isnan(all_s1).any(axis=1)
    avg_s1 = all_s1[valid].mean(axis=1)
    auc_avg = roc_auc_score(binary_labels[valid], avg_s1)

    all_s2 = np.stack([outer_oof_s2[m] for m in model_names], axis=0)
    avg_s2 = np.nanmean(all_s2[:, valid, :], axis=0)
    cancer_mask = (binary_labels[valid] == 1)
    f1_avg = 0.0
    if cancer_mask.sum() > 0:
        f1_avg = f1_score(
            cancer_type_labels[valid][cancer_mask],
            avg_s2[cancer_mask].argmax(axis=1),
            average="macro", zero_division=0,
        )
    rows.append({"model": "ensemble_avg", "type": "ensemble", "auc": round(auc_avg, 6), "f1_type": round(f1_avg, 6)})

    return rows


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_meta_learner_comparison(results_df, output_dir):
    """Boxplot of meta-learner performance across outer folds."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    meta_names = results_df["meta_learner"].unique()
    colors = plt.cm.Set2(np.linspace(0, 1, len(meta_names)))

    # S1 AUC boxplot
    ax = axes[0]
    data_auc = [results_df[results_df["meta_learner"] == m]["s1_auc"].values for m in meta_names]
    bp = ax.boxplot(data_auc, labels=meta_names, patch_artist=True, widths=0.6)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
    ax.set_ylabel("Stage 1 AUC")
    ax.set_title("Meta-Learner Comparison: Detection AUC", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for i, d in enumerate(data_auc):
        ax.text(i + 1, np.median(d) + 0.001, f"{np.median(d):.4f}", ha="center", fontsize=8)

    # S2 F1 boxplot
    ax = axes[1]
    data_f1 = [results_df[results_df["meta_learner"] == m]["s2_f1"].values for m in meta_names]
    bp = ax.boxplot(data_f1, labels=meta_names, patch_artist=True, widths=0.6)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
    ax.set_ylabel("Stage 2 F1 Macro")
    ax.set_title("Meta-Learner Comparison: Cancer Type F1", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for i, d in enumerate(data_f1):
        ax.text(i + 1, np.median(d) + 0.005, f"{np.median(d):.4f}", ha="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_dir / "meta_learner_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved meta_learner_comparison.png")


def plot_contribution_analysis(contributions, output_dir):
    """Bar chart of base model permutation importance."""
    fig, ax = plt.subplots(figsize=(10, 6))

    names = [c["base_model"] for c in contributions]
    drops = [c["mean_auc_drop"] for c in contributions]
    stds = [c["std_auc_drop"] for c in contributions]

    colors = ["#E91E63" if d > 0 else "#90CAF9" for d in drops]
    bars = ax.barh(range(len(names)), drops, xerr=stds, color=colors, alpha=0.85,
                   capsize=3, edgecolor="gray", linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Mean AUC Drop (higher = more important)")
    ax.set_title("Base Model Contribution (Permutation Importance)", fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.grid(axis="x", alpha=0.3)

    for i, (d, s) in enumerate(zip(drops, stds)):
        ax.text(d + s + 0.0005, i, f"{d:.4f}", va="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_dir / "contribution_analysis.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved contribution_analysis.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_val_group_data(group_name, grid):
    """Load a specific group's data as 3-channel multichannel for val-group inference."""
    data_dir = PROJECT_ROOT / "data" / "raw_data"

    # Full folder mapping (includes groups not in training, e.g. SPAN)
    import yaml
    with open(PROJECT_ROOT / "config" / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    full_map = cfg["dataset"]["folder_to_group"]

    # Support alias expansion (e.g. PAN → CPAN + YPAN)
    GROUP_ALIASES = {"PAN": ["CPAN", "YPAN"], "NOR": ["NOR", "YNOR"]}
    raw_groups = GROUP_ALIASES.get(group_name.upper(), [group_name.upper()])

    # Reverse lookup from full config mapping
    target_map = {k: v for k, v in full_map.items() if v in raw_groups}
    if not target_map:
        raise ValueError(f"No folder mapping found for group(s) {raw_groups}")

    logger.info(f"  Loading {raw_groups} from {len(target_map)} folders ...")
    X_3ch, df_meta = load_raw_multichannel(data_dir, target_map, grid, "*.CSV")
    logger.info(f"  Loaded {len(X_3ch)} spectra for {raw_groups}")

    # Mean aggregation per sample
    agg_groups = df_meta.groupby(["group", "sample_id"]).first().reset_index()
    X_agg = []
    for _, row in agg_groups.iterrows():
        mask_s = (df_meta["group"] == row["group"]) & (df_meta["sample_id"] == row["sample_id"])
        X_agg.append(X_3ch[mask_s.values].mean(axis=0))
    X_3ch_agg = np.stack(X_agg)
    df_meta_agg = agg_groups.reset_index(drop=True)

    logger.info(f"  Aggregated: {len(X_3ch_agg)} samples")
    return X_3ch_agg, df_meta_agg


def run_val_group_inference(args):
    """Run STK-V2 stacking inference on an out-of-training group (e.g. SPAN)."""
    import torch
    t0 = datetime.now()

    group_name = args.val_group.upper()
    meta_type = args.meta_learner

    logger.info("=" * 64)
    logger.info(f"  STK-V2 Val-Group Inference: {group_name}")
    logger.info(f"  Meta-learner: {meta_type}")
    logger.info("=" * 64)

    grid = np.linspace(402.0, 2198.0, 933)
    n_classes = len(CANCER_TYPES)

    # ── 1. Load training data (same as STK-V2 training) ──
    logger.info("\n[Step 1] Loading training data ...")
    X_3ch_train, df_meta_train, _, groups_train, sample_ids_train, \
        binary_labels_train, cancer_type_labels_train = load_data()

    # ── 2. Load val-group data ──
    logger.info(f"\n[Step 2] Loading {group_name} data ...")
    X_3ch_val, df_meta_val = load_val_group_data(group_name, grid)
    n_val = len(X_3ch_val)
    val_sample_ids = df_meta_val["sample_id"].values
    val_groups = df_meta_val["group"].values

    # ── 3. Extract peak features ──
    logger.info("\n[Step 3] Extracting peak features ...")
    X_peak_train = extract_peak_features(X_3ch_train[:, 0, :], grid)
    X_peak_val = extract_peak_features(X_3ch_val[:, 0, :], grid)
    logger.info(f"  Peak features: train={X_peak_train.shape}, val={X_peak_val.shape}")

    # ── 4. Train base models on full training data, predict val-group ──
    logger.info(f"\n[Step 4] Training {len(EXTENDED_BASE_MODELS)} base models → predicting {group_name} ...")
    base_s1 = {}
    base_s2 = {}

    for mi, (m_name, spec) in enumerate(EXTENDED_BASE_MODELS.items()):
        pk_tr = X_peak_train if spec["channels"] == "peak" else None
        pk_va = X_peak_val if spec["channels"] == "peak" else None

        s1_prob, s2_prob = train_base_model_ext(
            spec, X_3ch_train, binary_labels_train, cancer_type_labels_train,
            X_3ch_val, n_classes,
            X_peak_train=pk_tr, X_peak_val=pk_va,
        )
        base_s1[m_name] = s1_prob
        base_s2[m_name] = s2_prob
        logger.info(f"    [{mi+1}/{len(EXTENDED_BASE_MODELS)}] {m_name}: "
                    f"mean_cancer_prob={s1_prob.mean():.3f}")

    # ── 5. Build meta-features ──
    model_names = list(EXTENDED_BASE_MODELS.keys())
    meta_s1_val = np.column_stack([base_s1[m] for m in model_names])
    meta_s2_val = np.hstack([base_s2[m] for m in model_names])

    # Also need OOF meta-features for training the meta-learner
    # Load from saved oof_predictions.npz (check new path, fallback to legacy)
    oof_path = OUTPUT_DIR / "oof_predictions.npz"
    if not oof_path.exists():
        oof_path = _LEGACY_OUTPUT_DIR / "oof_predictions.npz"
    if not oof_path.exists():
        logger.error(f"  OOF predictions not found in {OUTPUT_DIR} or {_LEGACY_OUTPUT_DIR}")
        logger.error("  Run: python models/train_stacking.py  (without --val-group)")
        return 1

    logger.info(f"\n[Step 5] Loading OOF predictions for meta-learner training ...")
    oof = np.load(oof_path, allow_pickle=True)
    meta_s1_train = np.column_stack([oof[f"{m}_s1"] for m in model_names])
    meta_s2_train = np.hstack([oof[f"{m}_s2"] for m in model_names])
    meta_s1_train = np.nan_to_num(meta_s1_train, nan=0.0)
    meta_s2_train = np.nan_to_num(meta_s2_train, nan=0.0)

    # ── 6. Train meta-learner on full OOF → predict val-group ──
    logger.info(f"\n[Step 6] Training {meta_type} meta-learner → predicting {group_name} ...")
    meta_factories = make_meta_learners(n_classes)
    if meta_type not in meta_factories:
        logger.error(f"  Unknown meta-learner: {meta_type}. Available: {list(meta_factories.keys())}")
        return 1

    factory = meta_factories[meta_type]

    # Stage 1
    ml_s1 = factory("binary")
    ml_s1.fit(meta_s1_train, binary_labels_train)
    try:
        s1_final = ml_s1.predict_proba(meta_s1_val)[:, 1]
    except AttributeError:
        s1_final = ml_s1.decision_function(meta_s1_val)

    # Stage 2
    cancer_mask_train = binary_labels_train == 1
    ml_s2 = factory("multiclass")
    ml_s2.fit(meta_s2_train[cancer_mask_train], cancer_type_labels_train[cancer_mask_train])
    s2_prob = ml_s2.predict_proba(meta_s2_val)
    # Map local classes back to full class space
    s2_full = np.zeros((n_val, n_classes))
    for j, cls in enumerate(ml_s2.classes_):
        if cls < n_classes:
            s2_full[:, cls] = s2_prob[:, j]
    s2_pred = s2_full.argmax(axis=1)
    s2_pred_names = [CANCER_TYPES[i] if i < len(CANCER_TYPES) else f"cls_{i}" for i in s2_pred]

    # ── 7. Report ──
    logger.info(f"\n{'=' * 64}")
    logger.info(f"  Results: {group_name} ({n_val} samples) through STK-V2 ({meta_type})")
    logger.info(f"{'=' * 64}")

    logger.info(f"\n[Stage 1] Cancer Screening Probability")
    logger.info(f"  Mean cancer prob: {s1_final.mean():.4f} ± {s1_final.std():.4f}")
    for thresh in [0.3, 0.5, 0.7]:
        n_pos = (s1_final > thresh).sum()
        logger.info(f"  Threshold {thresh:.1f}: {n_pos}/{n_val} classified as cancer "
                    f"({n_pos/n_val*100:.1f}%)")

    logger.info(f"\n[Stage 2] Cancer Type Classification")
    pred_counts = pd.Series(s2_pred_names).value_counts()
    for ct, count in pred_counts.items():
        logger.info(f"  {ct}: {count}/{n_val} ({count/n_val*100:.1f}%)")

    logger.info(f"\n  Mean class probabilities:")
    for j, ct in enumerate(CANCER_TYPES):
        if j < s2_full.shape[1]:
            logger.info(f"    {ct}: {s2_full[:, j].mean():.4f} ± {s2_full[:, j].std():.4f}")

    # ── 8. Per-base-model breakdown ──
    logger.info(f"\n[Base Model Breakdown]")
    for m_name in model_names:
        bp = base_s1[m_name]
        s2p = base_s2[m_name].argmax(axis=1)
        s2names = [CANCER_TYPES[i] if i < len(CANCER_TYPES) else f"?" for i in s2p]
        top_type = pd.Series(s2names).value_counts().index[0]
        logger.info(f"  {m_name:<14} cancer_prob={bp.mean():.3f}±{bp.std():.3f}  "
                    f"top_type={top_type}")

    # ── 9. Save results ──
    out_dir = OUTPUT_DIR / f"val_group_{group_name.lower()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(n_val):
        row = {
            "group": val_groups[i],
            "sample_id": val_sample_ids[i],
            "cancer_prob": round(float(s1_final[i]), 6),
            "predicted_type": s2_pred_names[i],
        }
        for j, ct in enumerate(CANCER_TYPES):
            if j < s2_full.shape[1]:
                row[f"prob_{ct}"] = round(float(s2_full[i, j]), 6)
        # Base model probs
        for m_name in model_names:
            row[f"base_{m_name}_s1"] = round(float(base_s1[m_name][i]), 6)
        rows.append(row)
    df_result = pd.DataFrame(rows)
    df_result.to_csv(out_dir / "predictions.csv", index=False)

    # Visualizations
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Cancer prob distribution
    axes[0].hist(s1_final, bins=30, color="#E53935", alpha=0.7, edgecolor="black")
    axes[0].axvline(0.5, color="black", ls="--", lw=1.5, label="Threshold 0.5")
    axes[0].set(xlabel="Cancer Probability", ylabel="Count",
                title=f"{group_name} — S1 Cancer Prob (n={n_val})")
    axes[0].legend()

    # Predicted type pie
    colors_map = {
        "PRO": "#E91E63", "BRE": "#F06292", "OVA": "#AB47BC",
        "LUN": "#42A5F5", "CRC": "#EF5350", "PAN": "#FFA726", "BLC": "#7E57C2",
    }
    colors = [colors_map.get(ct, "#999999") for ct in pred_counts.index]
    axes[1].pie(pred_counts.values,
                labels=[f"{ct}\n({c})" for ct, c in pred_counts.items()],
                colors=colors, autopct="%1.1f%%", startangle=90)
    axes[1].set_title(f"{group_name} — Predicted Cancer Types")

    # Base model agreement heatmap
    base_preds = np.column_stack([base_s1[m] for m in model_names])
    im = axes[2].imshow(base_preds.T, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    axes[2].set_yticks(range(len(model_names)))
    axes[2].set_yticklabels(model_names, fontsize=8)
    axes[2].set_xlabel("Sample")
    axes[2].set_title(f"{group_name} — Base Model Cancer Probs")
    fig.colorbar(im, ax=axes[2], label="P(cancer)")

    plt.tight_layout()
    fig.savefig(out_dir / "val_group_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ── 10. Confusion Matrix ──
    if not args.no_cm:
        logger.info(f"\n[Step 10] Generating confusion matrices ...")
        from sklearn.metrics import confusion_matrix as sk_confusion_matrix
        import seaborn as sns

        # Ground truth: SPAN → determine expected label
        # Map val group to cancer type index (SPAN is pancreatic → PAN)
        GROUP_TO_EXPECTED = {
            "SPAN": "PAN", "CPAN": "PAN", "YPAN": "PAN",
        }
        expected_type = GROUP_TO_EXPECTED.get(group_name, group_name)

        # Stage 1 CM: all should be cancer
        gt_binary = np.ones(n_val, dtype=int)
        pred_binary = (s1_final > 0.5).astype(int)
        cm_s1 = sk_confusion_matrix(gt_binary, pred_binary, labels=[0, 1])
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        sns.heatmap(cm_s1, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Non-cancer", "Cancer"],
                    yticklabels=["Non-cancer", "Cancer"], ax=axes[0])
        axes[0].set(xlabel="Predicted", ylabel="Ground Truth",
                    title=f"{group_name} — Stage 1 CM (GT=Cancer, t=0.5)")

        # Stage 2 CM: GT = expected_type for all, predicted = s2_pred_names
        gt_type_idx = CANCER_TYPES.index(expected_type) if expected_type in CANCER_TYPES else -1
        if gt_type_idx >= 0:
            gt_s2 = np.full(n_val, gt_type_idx, dtype=int)
            cm_s2 = sk_confusion_matrix(gt_s2, s2_pred, labels=list(range(n_classes)))
            sns.heatmap(cm_s2, annot=True, fmt="d", cmap="OrRd",
                        xticklabels=list(CANCER_TYPES),
                        yticklabels=list(CANCER_TYPES), ax=axes[1])
            axes[1].set(xlabel="Predicted", ylabel="Ground Truth",
                        title=f"{group_name} — Stage 2 CM (GT={expected_type})")

            # Log accuracy
            correct = (s2_pred == gt_type_idx).sum()
            logger.info(f"  Stage 2 accuracy (GT={expected_type}): "
                        f"{correct}/{n_val} ({correct/n_val*100:.1f}%)")
        else:
            axes[1].text(0.5, 0.5, f"No GT mapping for {group_name}",
                         ha="center", va="center", transform=axes[1].transAxes)
            axes[1].set_title("Stage 2 CM — N/A")

        plt.tight_layout()
        fig.savefig(out_dir / "confusion_matrices.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Saved confusion_matrices.png")

    # ── 11. SHAP Analysis ──
    if not args.no_shap:
        try:
            import shap
        except ImportError:
            logger.warning("  shap not installed, skipping SHAP analysis")
            shap = None

        if shap is not None:
            shap_dir = out_dir / "shap"
            shap_dir.mkdir(exist_ok=True)

            # ── 11a. Meta-learner SHAP (which base models drive the decision) ──
            logger.info(f"\n[Step 11a] Meta-learner SHAP ...")
            try:
                meta_s1_bg = np.nan_to_num(
                    np.column_stack([oof[f"{m}_s1"] for m in model_names]), nan=0.0
                )
                n_bg = min(args.shap_samples, len(meta_s1_bg))
                bg_idx = np.random.RandomState(42).choice(len(meta_s1_bg), n_bg, replace=False)

                explainer_s1 = shap.LinearExplainer(ml_s1, meta_s1_bg[bg_idx])
                shap_vals_s1 = explainer_s1.shap_values(meta_s1_val)

                fig, ax = plt.subplots(figsize=(10, 6))
                mean_abs = np.abs(shap_vals_s1).mean(axis=0)
                order = np.argsort(mean_abs)[::-1]
                ax.barh(range(len(model_names)),
                        mean_abs[order],
                        color="#1976D2", alpha=0.85)
                ax.set_yticks(range(len(model_names)))
                ax.set_yticklabels([model_names[i] for i in order])
                ax.set_xlabel("Mean |SHAP value|")
                ax.set_title(f"Meta-Learner SHAP — Stage 1 (base model importance for {group_name})")
                ax.invert_yaxis()
                plt.tight_layout()
                fig.savefig(shap_dir / "meta_shap_s1.png", dpi=200)
                plt.close(fig)

                # Stage 2 meta SHAP
                meta_s2_bg = np.nan_to_num(
                    np.hstack([oof[f"{m}_s2"] for m in model_names]), nan=0.0
                )
                cancer_mask_bg = oof["binary_labels"] == 1
                meta_s2_bg_cancer = meta_s2_bg[cancer_mask_bg]
                n_bg_s2 = min(args.shap_samples, len(meta_s2_bg_cancer))
                bg_idx_s2 = np.random.RandomState(42).choice(
                    len(meta_s2_bg_cancer), n_bg_s2, replace=False
                )

                explainer_s2 = shap.LinearExplainer(ml_s2, meta_s2_bg_cancer[bg_idx_s2])
                shap_vals_s2 = explainer_s2.shap_values(meta_s2_val)

                # Aggregate SHAP by base model (each has n_classes columns)
                if isinstance(shap_vals_s2, list):
                    shap_s2_all = np.abs(np.stack(shap_vals_s2)).mean(axis=0)  # avg over classes
                else:
                    shap_s2_all = np.abs(shap_vals_s2)
                per_model_shap = []
                for mi, m_name in enumerate(model_names):
                    start = mi * n_classes
                    end = start + n_classes
                    per_model_shap.append(shap_s2_all[:, start:end].mean())
                per_model_shap = np.array(per_model_shap)
                order_s2 = np.argsort(per_model_shap)[::-1]

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(range(len(model_names)), per_model_shap[order_s2],
                        color="#E53935", alpha=0.85)
                ax.set_yticks(range(len(model_names)))
                ax.set_yticklabels([model_names[i] for i in order_s2])
                ax.set_xlabel("Mean |SHAP value| (aggregated over classes)")
                ax.set_title(f"Meta-Learner SHAP — Stage 2 (base model importance for {group_name})")
                ax.invert_yaxis()
                plt.tight_layout()
                fig.savefig(shap_dir / "meta_shap_s2.png", dpi=200)
                plt.close(fig)
                logger.info(f"  Saved meta_shap_s1.png, meta_shap_s2.png")

                # Save CSV
                pd.DataFrame({
                    "base_model": model_names,
                    "s1_mean_abs_shap": np.abs(shap_vals_s1).mean(axis=0),
                    "s2_mean_abs_shap": per_model_shap,
                }).to_csv(shap_dir / "meta_shap_summary.csv", index=False)

            except Exception as e:
                logger.warning(f"  Meta-learner SHAP failed: {e}")

            # ── 11b. Base model spectral SHAP (wavenumber importance) ──
            logger.info(f"\n[Step 11b] Base model spectral SHAP ...")
            # Use top-2 contributing base models for spectral SHAP
            top_base = ["lr_raw", "lr_d1"]
            wavenumbers = grid

            for bm_name in top_base:
                spec = EXTENDED_BASE_MODELS[bm_name]
                ch = spec["channels"]
                model_type = spec["model"]

                # Prepare features
                X_tr_bm = X_3ch_train[:, ch, :].reshape(len(X_3ch_train), -1)
                X_va_bm = X_3ch_val[:, ch, :].reshape(len(X_3ch_val), -1)

                # Train S1 model (Pipeline: StandardScaler + LR)
                s1_bm = build_classifier(model_type, "binary")
                s1_bm.fit(X_tr_bm, binary_labels_train)

                try:
                    n_bg_bm = min(args.shap_samples, len(X_tr_bm))
                    bg_idx_bm = np.random.RandomState(42).choice(
                        len(X_tr_bm), n_bg_bm, replace=False
                    )
                    # Extract the linear model from the pipeline and use scaled data
                    from sklearn.pipeline import Pipeline
                    if isinstance(s1_bm, Pipeline):
                        scaler_bm = s1_bm[:-1]  # all steps except last
                        lr_model = s1_bm[-1]     # the estimator
                        X_tr_scaled = scaler_bm.transform(X_tr_bm)
                        X_va_scaled = scaler_bm.transform(X_va_bm)
                        explainer_bm = shap.LinearExplainer(lr_model, X_tr_scaled[bg_idx_bm])
                        shap_vals_bm = explainer_bm.shap_values(X_va_scaled)
                    else:
                        explainer_bm = shap.LinearExplainer(s1_bm, X_tr_bm[bg_idx_bm])
                        shap_vals_bm = explainer_bm.shap_values(X_va_bm)

                    mean_shap_bm = np.abs(shap_vals_bm).mean(axis=0)

                    # Feature names
                    if len(ch) == 1:
                        feat_wn = wavenumbers
                    else:
                        feat_wn = np.tile(wavenumbers, len(ch))

                    fig, ax = plt.subplots(figsize=(14, 5))
                    if len(ch) == 1:
                        ax.plot(feat_wn, mean_shap_bm, color="#1976D2", lw=1.2)
                        ax.fill_between(feat_wn, mean_shap_bm, alpha=0.3, color="#1976D2")
                        ax.set_xlabel("Wavenumber (cm⁻¹)")
                    else:
                        for ci, c_idx in enumerate(ch):
                            ch_name = ["raw", "d1", "d2"][c_idx]
                            start = ci * len(wavenumbers)
                            end = start + len(wavenumbers)
                            ax.plot(wavenumbers, mean_shap_bm[start:end],
                                    lw=1.2, label=ch_name)
                        ax.legend()
                        ax.set_xlabel("Wavenumber (cm⁻¹)")

                    ax.set_ylabel("Mean |SHAP value|")
                    ax.set_title(f"{bm_name} — Spectral SHAP for {group_name} (Stage 1)")

                    # Annotate top peaks
                    if len(ch) == 1:
                        top_k = 10
                        top_idx = np.argsort(mean_shap_bm)[-top_k:]
                        for idx in top_idx:
                            if mean_shap_bm[idx] > mean_shap_bm.mean() * 2:
                                ax.annotate(f"{feat_wn[idx]:.0f}",
                                            xy=(feat_wn[idx], mean_shap_bm[idx]),
                                            fontsize=7, ha="center", va="bottom")

                    plt.tight_layout()
                    fig.savefig(shap_dir / f"spectral_shap_{bm_name}.png", dpi=200)
                    plt.close(fig)
                    logger.info(f"  Saved spectral_shap_{bm_name}.png")

                    # Save CSV
                    if len(ch) == 1:
                        pd.DataFrame({
                            "wavenumber": feat_wn,
                            "mean_abs_shap": mean_shap_bm,
                        }).to_csv(shap_dir / f"spectral_shap_{bm_name}.csv", index=False)

                except Exception as e:
                    logger.warning(f"  Spectral SHAP for {bm_name} failed: {e}")

    elapsed = datetime.now() - t0
    logger.info(f"\n{'=' * 64}")
    logger.info(f"  Val-group inference complete! ({elapsed})")
    logger.info(f"  Output: {out_dir}/")
    logger.info(f"{'=' * 64}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 5: Full Stacking / Meta-Learner Optimization"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Quick test: 1 outer fold, 3 base models, 2 meta-learners")
    parser.add_argument("--val-group", type=str, default=None,
                        help="Run inference on an out-of-training group (e.g. SPAN)")
    parser.add_argument("--meta-learner", type=str, default="elasticnet",
                        help="Meta-learner for val-group inference (default: elasticnet)")
    parser.add_argument("--no-shap", action="store_true",
                        help="Skip SHAP analysis for val-group inference")
    parser.add_argument("--no-cm", action="store_true",
                        help="Skip confusion matrix for val-group inference")
    parser.add_argument("--shap-samples", type=int, default=100,
                        help="Background samples for SHAP (default: 100)")
    args = parser.parse_args()

    if args.val_group:
        return run_val_group_inference(args)

    t0 = datetime.now()

    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Set up file logging
    fh = logging.FileHandler(OUTPUT_DIR / "experiment.log", mode="a")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(fh)

    logger.info("=" * 64)
    logger.info("  Experiment 5: Full Stacking / Meta-Learner Optimization")
    logger.info(f"  Started: {t0.isoformat()}")
    logger.info(f"  Dry-run: {args.dry_run}")
    logger.info("=" * 64)

    # Load data
    X_3ch_agg, df_meta_agg, grid, groups_arr, sample_ids, binary_labels, cancer_type_labels = load_data()
    n_classes = len(CANCER_TYPES)

    # Meta-learner factories
    meta_learner_factories = make_meta_learners(n_classes)

    # Save config
    config = {
        "experiment": "stacking_optimization",
        "timestamp": t0.isoformat(),
        "dry_run": args.dry_run,
        "n_samples": len(X_3ch_agg),
        "n_cancer": int(binary_labels.sum()),
        "n_non_cancer": int((binary_labels == 0).sum()),
        "n_classes": n_classes,
        "cancer_types": list(CANCER_TYPES),
        "base_models": list(EXTENDED_BASE_MODELS.keys()),
        "meta_learners": list(meta_learner_factories.keys()),
        "outer_folds": 5 if not args.dry_run else 1,
        "inner_oof_folds": 5,
        "inner_meta_folds": 3,
    }
    atomic_save_json(OUTPUT_DIR / "config.json", config)

    # Run nested CV
    (all_results, outer_oof_s1, outer_oof_s2,
     model_names, meta_names, best_meta_per_fold, X_peak_all) = run_nested_cv(
        X_3ch_agg, grid, sample_ids, binary_labels, cancer_type_labels,
        n_classes, EXTENDED_BASE_MODELS, meta_learner_factories,
        dry_run=args.dry_run,
    )

    # ===== Save nested_cv_results.csv =====
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "nested_cv_results.csv", index=False)
    logger.info(f"\n  Saved nested_cv_results.csv ({len(results_df)} rows)")

    # ===== Save oof_predictions.npz =====
    oof_arrays = {}
    for m_name in model_names:
        oof_arrays[f"{m_name}_s1"] = outer_oof_s1[m_name]
        oof_arrays[f"{m_name}_s2"] = outer_oof_s2[m_name]
    oof_arrays["binary_labels"] = binary_labels
    oof_arrays["cancer_type_labels"] = cancer_type_labels
    oof_arrays["sample_ids"] = sample_ids
    atomic_save_npz(OUTPUT_DIR / "oof_predictions.npz", **oof_arrays)
    logger.info("  Saved oof_predictions.npz")

    # ===== Determine best ensemble config =====
    # Average across folds for each meta-learner
    summary = results_df.groupby("meta_learner").agg(
        mean_s1_auc=("s1_auc", "mean"),
        std_s1_auc=("s1_auc", "std"),
        mean_s2_f1=("s2_f1", "mean"),
        std_s2_f1=("s2_f1", "std"),
    ).reset_index()
    summary["combined"] = summary["mean_s1_auc"] * 0.6 + summary["mean_s2_f1"] * 0.4
    best_row = summary.loc[summary["combined"].idxmax()]
    best_meta = best_row["meta_learner"]

    best_config = {
        "best_meta_learner": best_meta,
        "mean_s1_auc": round(float(best_row["mean_s1_auc"]), 6),
        "std_s1_auc": round(float(best_row["std_s1_auc"]), 6),
        "mean_s2_f1": round(float(best_row["mean_s2_f1"]), 6),
        "std_s2_f1": round(float(best_row["std_s2_f1"]), 6),
        "base_models_used": model_names,
        "best_per_fold": {str(k): v for k, v in best_meta_per_fold.items()},
        "meta_learner_summary": summary.to_dict(orient="records"),
    }
    atomic_save_json(OUTPUT_DIR / "best_ensemble_config.json", best_config)
    logger.info(f"  Saved best_ensemble_config.json (best: {best_meta})")

    # ===== Base model contribution =====
    logger.info("\n  Computing base model contribution (permutation importance) ...")
    contributions = compute_base_model_contribution(
        outer_oof_s1, model_names, binary_labels, cancer_type_labels, n_classes,
    )
    pd.DataFrame(contributions).to_csv(OUTPUT_DIR / "base_model_contribution.csv", index=False)
    logger.info("  Saved base_model_contribution.csv")

    # ===== Single vs ensemble comparison =====
    single_vs = compare_single_vs_ensemble(
        outer_oof_s1, outer_oof_s2, model_names,
        binary_labels, cancer_type_labels, n_classes,
    )
    pd.DataFrame(single_vs).to_csv(OUTPUT_DIR / "single_vs_ensemble.csv", index=False)
    logger.info("  Saved single_vs_ensemble.csv")

    # ===== Plots =====
    logger.info("\n  Generating plots ...")
    plot_meta_learner_comparison(results_df, FIG_OUTPUT_DIR)
    plot_contribution_analysis(contributions, FIG_OUTPUT_DIR)

    # ===== Final summary =====
    duration = (datetime.now() - t0).total_seconds()
    logger.info(f"\n{'='*64}")
    logger.info(f"  EXPERIMENT 5 SUMMARY")
    logger.info(f"{'='*64}")
    logger.info(f"  Duration: {duration:.0f}s ({duration/60:.1f} min)")
    logger.info(f"  Samples: {len(X_3ch_agg)} ({int(binary_labels.sum())} cancer)")
    logger.info(f"  Base models: {len(model_names)}")
    logger.info(f"  Meta-learners: {len(meta_names)}")
    logger.info(f"  Best meta-learner: {best_meta}")
    logger.info(f"    S1 AUC: {best_row['mean_s1_auc']:.4f} +/- {best_row['std_s1_auc']:.4f}")
    logger.info(f"    S2 F1:  {best_row['mean_s2_f1']:.4f} +/- {best_row['std_s2_f1']:.4f}")
    logger.info(f"\n  Meta-learner ranking:")
    for _, row in summary.sort_values("combined", ascending=False).iterrows():
        flag = " *" if row["meta_learner"] == best_meta else ""
        logger.info(
            f"    {row['meta_learner']:<12} AUC={row['mean_s1_auc']:.4f}+/-{row['std_s1_auc']:.4f}  "
            f"F1={row['mean_s2_f1']:.4f}+/-{row['std_s2_f1']:.4f}{flag}"
        )
    logger.info(f"\n  Top 3 contributing base models:")
    for c in contributions[:3]:
        logger.info(f"    {c['base_model']:<14} AUC drop = {c['mean_auc_drop']:.4f}")
    logger.info(f"\n  Single vs Ensemble:")
    for sv in single_vs:
        logger.info(f"    {sv['model']:<14} {sv['type']:<10} AUC={sv['auc']:.4f}  F1={sv['f1_type']:.4f}")
    logger.info(f"{'='*64}")
    logger.info(f"  Output: {OUTPUT_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
