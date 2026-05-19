#!/usr/bin/env python3
"""
STK-V2: Stacking / Meta-Learner Optimization
=============================================

Extended base models (10) x meta-learners (5) x nested CV (outer 5 x inner 3)
for unbiased ensemble performance estimation.

Usage:
    python scripts/training/train_usersnet.py
    python scripts/training/train_usersnet.py --dry-run
    python scripts/training/train_usersnet.py --val-group SPAN --meta-learner elasticnet
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
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from scipy.special import voigt_profile
from scipy.integrate import trapezoid

from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve,
)
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths (will be set dynamically in main())
# ---------------------------------------------------------------------------
# Default: auto-detect from script location
_DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_DEFAULT_PROJECT_ROOT))

from sers.models.usersnet.stacking import preprocess_channel, load_raw_multichannel, load_processed_multichannel, build_classifier, train_base_model, configure_runtime, apply_sex_constraint, infer_sex_from_groups  # noqa: E402
from src.sers.config import RESULTS_DIR, FIG_DIR  # noqa: E402

# These will be set in main()
PROJECT_ROOT = _DEFAULT_PROJECT_ROOT
DATA_DIR = _DEFAULT_PROJECT_ROOT / "data" / "raw_data"
DATA_TYPE = "raw_spectrum"  # or "processed_csv"
OUTPUT_DIR = RESULTS_DIR / "training" / "stacking_v2"
FIG_OUTPUT_DIR = FIG_DIR / "training" / "stacking_v2"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
_LEGACY_OUTPUT_DIR = RESULTS_DIR / "weekend_experiments" / "stacking_optimization"
_LEGACY_CHECKPOINT_DIR = _LEGACY_OUTPUT_DIR / "checkpoints"
MODEL_N_JOBS = -1
XGB_DEVICE = "cpu"

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

GROUP_ALIASES = {"CPAN": "PAN", "YPAN": "PAN", "YNOR": "NOR"}
GROUP_EXPANSIONS = {"PAN": ["CPAN", "YPAN"], "NOR": ["NOR", "YNOR"]}
EXCLUDE_SOURCE_GROUPS: set[str] = set()

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
# Fixed train/val/test split
# ---------------------------------------------------------------------------
def create_fixed_split(
    sample_ids: np.ndarray,
    binary_labels: np.ndarray,
    stratify_labels: np.ndarray | None = None,
    test_ratio: float = 0.20,
    val_ratio: float = 0.20,
    force_test_prefix: str = "YPAN_",
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fixed train/val/test split with forced-test constraint.

    - YPAN samples → 무조건 test
    - 나머지 → stratified split (train 60 / val 20 / test 나머지)

    Returns: (train_idx, val_idx, test_idx)
    """
    n = len(sample_ids)
    split_labels = binary_labels if stratify_labels is None else np.asarray(stratify_labels)
    if len(split_labels) != n:
        raise ValueError("stratify_labels must have the same length as sample_ids")

    force_label = str(force_test_prefix).strip()
    force_token = force_label.rstrip("_")

    def _is_forced_test_sample(sid) -> bool:
        sid_str = str(sid)
        if not force_label:
            return False
        candidates = [force_label]
        if force_token and force_token != force_label:
            candidates.append(force_token)
        for token in candidates:
            if (
                sid_str == token
                or sid_str.startswith(f"{token}_")
                or sid_str.endswith(f"_{token}")
                or f"_{token}_" in sid_str
            ):
                return True
        return False

    def _valid_stratify(labels, test_count: int) -> bool:
        labels = np.asarray(labels)
        train_count = len(labels) - test_count
        if len(labels) == 0 or test_count <= 0 or train_count <= 0:
            return False
        _, counts = np.unique(labels, return_counts=True)
        n_classes = len(counts)
        return (
            n_classes > 1
            and counts.min() >= 2
            and test_count >= n_classes
            and train_count >= n_classes
        )

    def _choose_stratify(primary_labels, fallback_labels, test_count: int, split_name: str):
        if _valid_stratify(primary_labels, test_count):
            return primary_labels
        if _valid_stratify(fallback_labels, test_count):
            logger.warning(
                f"  {split_name}: group stratify is too granular; falling back to binary stratify"
            )
            return fallback_labels
        logger.warning(f"  {split_name}: stratify disabled due to insufficient class counts")
        return None

    # ── 1. YPAN 강제 test 분리 ──
    is_forced = np.array([_is_forced_test_sample(sid) for sid in sample_ids])
    forced_test_idx = np.where(is_forced)[0]
    remaining_idx = np.where(~is_forced)[0]

    logger.info(f"  Forced test ({force_token or force_label}): {len(forced_test_idx)} samples")
    logger.info(f"  Remaining for split: {len(remaining_idx)} samples")

    # ── 2. 나머지에서 test 추가 할당 ──
    n_target_test = int(n * test_ratio)
    n_extra_test = max(n_target_test - len(forced_test_idx), 0)
    remaining_labels = split_labels[remaining_idx]
    remaining_binary_labels = binary_labels[remaining_idx]

    if n_extra_test > 0:
        if n_extra_test >= len(remaining_idx):
            trainval_local = np.array([], dtype=int)
            extra_test_local = np.arange(len(remaining_idx))
        else:
            test_stratify = _choose_stratify(
                remaining_labels, remaining_binary_labels, n_extra_test, "test split"
            )
            trainval_local, extra_test_local = train_test_split(
                np.arange(len(remaining_idx)),
                test_size=n_extra_test,
                stratify=test_stratify,
                random_state=random_state,
            )
    else:
        trainval_local = np.arange(len(remaining_idx))
        extra_test_local = np.array([], dtype=int)

    # ── 3. train / val 분리 ──
    trainval_labels = remaining_labels[trainval_local]
    trainval_binary_labels = remaining_binary_labels[trainval_local]
    val_ratio_adj = val_ratio / (1.0 - test_ratio)  # trainval 내 비율 보정
    n_val = int(round(len(trainval_local) * val_ratio_adj))
    n_val = min(max(n_val, 1), max(len(trainval_local) - 1, 1))

    if len(trainval_local) > 1:
        val_stratify = _choose_stratify(
            trainval_labels, trainval_binary_labels, n_val, "validation split"
        )
        train_local, val_local = train_test_split(
            np.arange(len(trainval_local)),
            test_size=n_val,
            stratify=val_stratify,
            random_state=random_state,
        )
    else:
        train_local = np.arange(len(trainval_local))
        val_local = np.array([], dtype=int)

    # ── 4. global index로 변환 ──
    train_idx = remaining_idx[trainval_local[train_local]]
    val_idx   = remaining_idx[trainval_local[val_local]]
    test_idx  = np.concatenate([
        forced_test_idx,
        remaining_idx[extra_test_local],
    ])

    # ── 5. 검증 ──
    assert len(set(train_idx) & set(val_idx)) == 0, "train/val overlap!"
    assert len(set(train_idx) & set(test_idx)) == 0, "train/test overlap!"
    assert len(set(val_idx) & set(test_idx)) == 0, "val/test overlap!"
    assert len(train_idx) + len(val_idx) + len(test_idx) == n, "count mismatch!"

    # YPAN이 전부 test에 있는지 확인
    assert all(i in test_idx for i in forced_test_idx), "YPAN not all in test!"

    logger.info(f"  Split result:")
    logger.info(f"    Train: {len(train_idx)} ({len(train_idx)/n*100:.1f}%)")
    logger.info(f"    Val:   {len(val_idx)} ({len(val_idx)/n*100:.1f}%)")
    logger.info(f"    Test:  {len(test_idx)} ({len(test_idx)/n*100:.1f}%) "
                f"(YPAN={len(forced_test_idx)} + others={len(extra_test_local)})")

    return train_idx, val_idx, test_idx



# ---------------------------------------------------------------------------
# Extended train_base_model (handles "peak" channel type)
# ---------------------------------------------------------------------------

def _needs_contiguous_multiclass_labels(model) -> bool:
    """XGBoost sklearn multiclass requires local labels 0..K-1."""
    return model.__class__.__name__ == "XGBClassifier"


def fit_sparse_multiclass(model, X, y):
    """Fit multiclass model when global class ids may be non-contiguous."""
    y = np.asarray(y, dtype=int)
    classes = np.unique(y)
    if _needs_contiguous_multiclass_labels(model):
        class_to_local = {int(cls): i for i, cls in enumerate(classes)}
        y_local = np.array([class_to_local[int(v)] for v in y], dtype=int)
        model.fit(X, y_local)
        return model, classes, True

    model.fit(X, y)
    return model, np.asarray(getattr(model, "classes_", classes), dtype=int), False


def predict_sparse_multiclass_proba(fitted, X, n_classes: int) -> np.ndarray:
    """Return probability columns in global cancer-class order."""
    model, classes, _remapped = fitted
    raw_prob = model.predict_proba(X)
    out = np.zeros((len(X), n_classes))
    for i, cls in enumerate(classes):
        cls = int(cls)
        if cls < n_classes and i < raw_prob.shape[1]:
            out[:, cls] = raw_prob[:, i]
    return out


def predict_sparse_multiclass_labels(fitted, X) -> np.ndarray:
    """Return global class ids after optional local-label remapping."""
    model, classes, remapped = fitted
    pred = np.asarray(model.predict(X), dtype=int)
    if remapped:
        return classes[pred]
    return pred


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
        s2_fit = fit_sparse_multiclass(s2, X_tr[cancer_mask], y_type_train[cancer_mask])
        s2_prob = predict_sparse_multiclass_proba(s2_fit, X_va, n_classes)

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
            n_jobs=MODEL_N_JOBS,
        )

    def _xgb(task):
        if not xgb_available:
            return _lr(task)
        kw = dict(n_estimators=200, max_depth=4, learning_rate=0.05,
                  n_jobs=MODEL_N_JOBS, tree_method="hist",
                  device=XGB_DEVICE, verbosity=0)
        if task == "multiclass":
            kw.update(objective="multi:softprob", num_class=n_classes)
        else:
            kw.update(objective="binary:logistic")
        return XGBClassifier(**kw)

    def _rf(task):
        return RandomForestClassifier(
            n_estimators=300, max_depth=5,
            class_weight="balanced_subsample", n_jobs=MODEL_N_JOBS, random_state=42,
        )

    def _elasticnet(task):
        return LogisticRegression(
            C=0.5, penalty="elasticnet", l1_ratio=0.5,
            max_iter=2000, solver="saga",
            multi_class="multinomial" if task == "multiclass" else "auto",
            n_jobs=MODEL_N_JOBS,
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


def save_fixed_predictions(prefix, s1_prob, s2_prob, y_bin, y_type,
                           sample_ids, groups, output_dir):
    """Save fixed-split per-sample predictions for downstream figures."""
    atomic_save_npz(
        output_dir / f"{prefix}_predictions.npz",
        s1_prob=np.asarray(s1_prob, dtype=np.float64),
        s2_prob=np.asarray(s2_prob, dtype=np.float64),
        y_bin=np.asarray(y_bin, dtype=np.float64),
        y_type=np.asarray(y_type, dtype=np.int64),
        sample_ids=np.asarray(sample_ids, dtype=object),
        groups=np.asarray(groups, dtype=object) if groups is not None else np.asarray([], dtype=object),
    )
    logger.info(f"  Saved {prefix}_predictions.npz ({len(y_bin)} samples)")


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

def parse_group_arg_values(values) -> set[str]:
    """Parse comma-separated and repeatable group CLI values."""
    groups: set[str] = set()
    for value in values or []:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                groups.add(part.upper())
    return groups


def _filter_source_groups(
    X_3ch: np.ndarray,
    df_meta: pd.DataFrame,
    *,
    include: set[str] | None = None,
    exclude: set[str] | None = None,
    label: str = "dataset",
) -> tuple[np.ndarray, pd.DataFrame]:
    """Filter by raw/source group before CPAN/YPAN alias merging."""
    df_meta = df_meta.copy()
    if "source_group" not in df_meta.columns:
        df_meta["source_group"] = df_meta["group"].astype(str)

    source_upper = df_meta["source_group"].astype(str).str.upper()
    mask = np.ones(len(df_meta), dtype=bool)

    if include:
        include = {g.upper() for g in include}
        mask &= source_upper.isin(include).to_numpy()
    if exclude:
        exclude = {g.upper() for g in exclude}
        mask &= ~source_upper.isin(exclude).to_numpy()

    if include or exclude:
        kept = int(mask.sum())
        logger.info(
            f"  Source-group filter ({label}): kept {kept}/{len(df_meta)} spectra "
            f"(include={sorted(include or []) or 'all'}, exclude={sorted(exclude or []) or 'none'})"
        )

    if not mask.any():
        raise ValueError(
            f"No spectra left after source-group filtering "
            f"(include={sorted(include or [])}, exclude={sorted(exclude or [])})"
        )

    return X_3ch[mask], df_meta.loc[mask].reset_index(drop=True)


def _prepare_sample_ids_and_aliases(
    df_meta: pd.DataFrame,
    *,
    merge_aliases: bool,
) -> pd.DataFrame:
    """Preserve source_group, disambiguate Y* sample ids, then optionally merge aliases."""
    df_meta = df_meta.copy()
    if "source_group" not in df_meta.columns:
        df_meta["source_group"] = df_meta["group"].astype(str)

    needs_prefix = df_meta["source_group"].isin(["YPAN", "YNOR"])
    df_meta.loc[needs_prefix, "sample_id"] = (
        df_meta.loc[needs_prefix, "source_group"]
        + "_"
        + df_meta.loc[needs_prefix, "sample_id"].astype(str)
    )

    if merge_aliases:
        df_meta["group"] = df_meta["source_group"].replace(GROUP_ALIASES)
    else:
        df_meta["group"] = df_meta["source_group"]

    return df_meta


def _aggregate_mean_by_sample(
    X_3ch: np.ndarray,
    df_meta: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Mean-aggregate replicate spectra by displayed group and sample_id."""
    if len(X_3ch) == 0:
        raise ValueError("No spectra to aggregate")

    agg_groups = df_meta.groupby(["group", "sample_id"]).first().reset_index()
    X_agg = []
    for _, row in agg_groups.iterrows():
        mask_s = (
            (df_meta["group"] == row["group"])
            & (df_meta["sample_id"] == row["sample_id"])
        )
        X_agg.append(X_3ch[mask_s.values].mean(axis=0))
    return np.stack(X_agg), agg_groups.reset_index(drop=True)


def load_data(data_dir=None, exclude_source_groups: set[str] | None = None):
    """Load and prepare multichannel SERS data with mean aggregation.
    
    Args:
        data_dir: Path to raw data directory. If None, uses global DATA_DIR.
        exclude_source_groups: Raw/source group codes to remove before alias
            merging (for example {"YPAN"} for external holdout).
    
    Supports two data types (set via global DATA_TYPE):
        - "raw_spectrum": Load from raw folder structure via load_raw_multichannel
        - "processed_csv": Load from preprocessed CSV with group, sample_id, x_... columns
    """
    if data_dir is None:
        data_dir = DATA_DIR
    else:
        data_dir = Path(data_dir)
    
    grid = np.linspace(402.0, 2198.0, 933)

    if DATA_TYPE == "processed_csv":
        logger.info("Loading processed CSV spectra ...")
        X_3ch, df_meta, grid = load_processed_multichannel(data_dir, target_grid=grid)
    else:
        logger.info("Loading raw multichannel spectra ...")
        X_3ch, df_meta = load_raw_multichannel(data_dir, THERMO_MAP, grid, "*.CSV")

    exclude_source_groups = (
        set(exclude_source_groups)
        if exclude_source_groups is not None
        else set(EXCLUDE_SOURCE_GROUPS)
    )
    X_3ch, df_meta = _filter_source_groups(
        X_3ch, df_meta, exclude=exclude_source_groups, label="training"
    )
    df_meta = _prepare_sample_ids_and_aliases(df_meta, merge_aliases=True)

    valid = set(CANCER_TYPES) | set(NON_CANCER_GROUPS)
    mask = df_meta["group"].isin(valid).values
    X_3ch = X_3ch[mask]
    df_meta = df_meta[mask].reset_index(drop=True)

    X_3ch_agg, df_meta_agg = _aggregate_mean_by_sample(X_3ch, df_meta)

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
            ml_s2_fit = fit_sparse_multiclass(
                ml_s2, meta_s2[itr][cancer_tr], y_type[itr][cancer_tr]
            )
            s2_pred = predict_sparse_multiclass_labels(
                ml_s2_fit, meta_s2[ival][cancer_va]
            )
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
    groups_arr=None,
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
                ml_s2_fit = fit_sparse_multiclass(
                    ml_s2, meta_s2_train[cancer_tr], y_type_outer[cancer_tr]
                )
                # Map to full class space and apply sex constraint
                s2_full_te = predict_sparse_multiclass_proba(
                    ml_s2_fit, meta_s2_test[cancer_te], n_classes
                )
                sex_te = infer_sex_from_groups(groups_arr[outer_te][cancer_te])
                s2_full_te = apply_sex_constraint(s2_full_te, CANCER_TYPES, sex_te)
                s2_pred = s2_full_te.argmax(axis=1)
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

def run_fixed_split(
    X_3ch_agg, grid, sample_ids, binary_labels, cancer_type_labels,
    n_classes, base_models, meta_learner_factories,
    groups_arr=None,
    train_idx=None, val_idx=None, test_idx=None,
):
    """
    Fixed split 학습: Train(60) → Val(20) → Test(20)
    
    Train: base model 학습
    Val:   meta-learner 선택 + 튜닝
    Test:  최종 성능 보고 (1회만 사용)
    """
    model_names = list(base_models.keys())
    meta_names = list(meta_learner_factories.keys())

    # ── Data split ──
    X_train, X_val, X_test = X_3ch_agg[train_idx], X_3ch_agg[val_idx], X_3ch_agg[test_idx]
    y_bin_train = binary_labels[train_idx]
    y_bin_val   = binary_labels[val_idx]
    y_bin_test  = binary_labels[test_idx]
    y_type_train = cancer_type_labels[train_idx]
    y_type_val   = cancer_type_labels[val_idx]
    y_type_test  = cancer_type_labels[test_idx]
    sids_train = sample_ids[train_idx]
    sids_val = sample_ids[val_idx]
    sids_test = sample_ids[test_idx]
    groups_val = groups_arr[val_idx] if groups_arr is not None else None
    groups_test = groups_arr[test_idx] if groups_arr is not None else None

    logger.info(f"\n  Train: {len(X_train)} (cancer={int(y_bin_train.sum())})")
    logger.info(f"  Val:   {len(X_val)} (cancer={int(y_bin_val.sum())})")
    logger.info(f"  Test:  {len(X_test)} (cancer={int(y_bin_test.sum())})")

    atomic_save_npz(
        OUTPUT_DIR / "fixed_split_indices.npz",
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
    )
    logger.info("  Saved fixed_split_indices.npz")

    # ── Peak features ──
    logger.info("  Extracting peak features ...")
    X_peak_all = extract_peak_features(X_3ch_agg[:, 0, :], grid)
    X_peak_train = X_peak_all[train_idx]
    X_peak_val   = X_peak_all[val_idx]
    X_peak_test  = X_peak_all[test_idx]

    # ════════════════════════════════════════════
    # Level 0: Base models — train → predict val & test
    # ════════════════════════════════════════════
    logger.info(f"\n  -- Level 0: Training {len(model_names)} base models --")

    val_s1, val_s2 = {}, {}
    test_s1, test_s2 = {}, {}

    for mi, (m_name, spec) in enumerate(base_models.items()):
        pk_tr = X_peak_train if spec["channels"] == "peak" else None
        pk_va = X_peak_val   if spec["channels"] == "peak" else None
        pk_te = X_peak_test  if spec["channels"] == "peak" else None

        # predict on val
        s1_v, s2_v = train_base_model_ext(
            spec, X_train, y_bin_train, y_type_train,
            X_val, n_classes,
            X_peak_train=pk_tr, X_peak_val=pk_va,
        )
        val_s1[m_name] = s1_v
        val_s2[m_name] = s2_v

        # predict on test
        s1_t, s2_t = train_base_model_ext(
            spec, X_train, y_bin_train, y_type_train,
            X_test, n_classes,
            X_peak_train=pk_tr, X_peak_val=pk_te,
        )
        test_s1[m_name] = s1_t
        test_s2[m_name] = s2_t

        # log
        auc_val = roc_auc_score(y_bin_val, s1_v)
        logger.info(f"    [{mi+1}/{len(model_names)}] {m_name}: val AUC={auc_val:.4f}")

    base_val_arrays = {}
    base_test_arrays = {}
    for m in model_names:
        base_val_arrays[f"{m}_s1"] = val_s1[m]
        base_val_arrays[f"{m}_s2"] = val_s2[m]
        base_test_arrays[f"{m}_s1"] = test_s1[m]
        base_test_arrays[f"{m}_s2"] = test_s2[m]
    atomic_save_npz(OUTPUT_DIR / "fixed_base_model_val.npz", **base_val_arrays)
    atomic_save_npz(OUTPUT_DIR / "fixed_base_model_test.npz", **base_test_arrays)
    logger.info("  Saved fixed_base_model_val.npz and fixed_base_model_test.npz")

    # ════════════════════════════════════════════
    # Level 1: Meta-learner selection on VAL set
    # ════════════════════════════════════════════
    logger.info(f"\n  -- Level 1: Meta-learner selection on VAL --")

    meta_s1_val  = np.column_stack([val_s1[m] for m in model_names])
    meta_s2_val  = np.hstack([val_s2[m] for m in model_names])
    meta_s1_test = np.column_stack([test_s1[m] for m in model_names])
    meta_s2_test = np.hstack([test_s2[m] for m in model_names])

    # Train meta-learners on val-set base predictions
    # (meta-learner의 train input = base model이 val에 대해 낸 예측)
    # 하지만 meta-learner도 train set의 OOF가 필요
    # → train set 내부 CV로 OOF 생성
    logger.info(f"  Generating OOF on train set for meta-learner training ...")

    oof_s1_train = {}
    oof_s2_train = {}
    inner_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=123)

    for m_name, spec in base_models.items():
        s1_oof, s2_oof = generate_oof_inner(
            m_name, spec, X_train, y_bin_train, y_type_train,
            sids_train, n_classes, grid, X_peak_train,
            inner_splits=5,
        )
        oof_s1_train[m_name] = s1_oof
        oof_s2_train[m_name] = s2_oof

    meta_s1_train_oof = np.nan_to_num(
        np.column_stack([oof_s1_train[m] for m in model_names]), nan=0.0
    )
    meta_s2_train_oof = np.nan_to_num(
        np.hstack([oof_s2_train[m] for m in model_names]), nan=0.0
    )

    oof_arrays = {
        "meta_s1": meta_s1_train_oof,
        "meta_s2": meta_s2_train_oof,
        "y_bin": y_bin_train,
        "y_type": y_type_train,
    }
    for m in model_names:
        oof_arrays[f"{m}_s1"] = oof_s1_train[m]
        oof_arrays[f"{m}_s2"] = oof_s2_train[m]
    atomic_save_npz(OUTPUT_DIR / "fixed_train_oof.npz", **oof_arrays)
    logger.info("  Saved fixed_train_oof.npz")

    # Evaluate each meta-learner: train on OOF, evaluate on VAL
    best_meta_name = None
    best_meta_score = -1.0
    meta_results = []

    for ml_name, ml_factory in meta_learner_factories.items():
        # S1
        ml_s1 = ml_factory("binary")
        ml_s1.fit(meta_s1_train_oof, y_bin_train)
        try:
            s1_pred_val = ml_s1.predict_proba(meta_s1_val)[:, 1]
        except AttributeError:
            s1_pred_val = ml_s1.decision_function(meta_s1_val)
        auc_val = roc_auc_score(y_bin_val, s1_pred_val)

        # S2
        cancer_tr = y_bin_train == 1
        cancer_va = y_bin_val == 1
        f1_val = 0.0
        if cancer_tr.sum() > 5 and cancer_va.sum() > 0:
            ml_s2 = ml_factory("multiclass")
            ml_s2_fit = fit_sparse_multiclass(
                ml_s2, meta_s2_train_oof[cancer_tr], y_type_train[cancer_tr]
            )
            s2_pred = predict_sparse_multiclass_labels(
                ml_s2_fit, meta_s2_val[cancer_va]
            )
            f1_val = f1_score(y_type_val[cancer_va], s2_pred, average="macro", zero_division=0)

        combined = auc_val * 0.6 + f1_val * 0.4
        logger.info(f"    {ml_name:<12} val AUC={auc_val:.4f}  F1={f1_val:.4f}  combined={combined:.4f}")

        meta_results.append({
            "meta_learner": ml_name, "val_s1_auc": auc_val,
            "val_s2_f1": f1_val, "combined": combined,
        })

        if combined > best_meta_score:
            best_meta_score = combined
            best_meta_name = ml_name

    logger.info(f"\n  => Best meta-learner: {best_meta_name}")

    # ════════════════════════════════════════════
    # Level 2: Final evaluation on TEST set (1회만!)
    # ════════════════════════════════════════════
    logger.info(f"\n  -- Level 2: FINAL TEST evaluation ({best_meta_name}) --")
    logger.info(f"  ⚠ This is the ONLY time test set is used!")

    # Retrain best meta-learner on train OOF
    best_factory = meta_learner_factories[best_meta_name]

    # S1
    final_s1 = best_factory("binary")
    final_s1.fit(meta_s1_train_oof, y_bin_train)
    try:
        s1_test_pred = final_s1.predict_proba(meta_s1_test)[:, 1]
    except AttributeError:
        s1_test_pred = final_s1.decision_function(meta_s1_test)
    test_auc = roc_auc_score(y_bin_test, s1_test_pred)
    try:
        s1_val_pred = final_s1.predict_proba(meta_s1_val)[:, 1]
    except AttributeError:
        s1_val_pred = final_s1.decision_function(meta_s1_val)

    # S2
    cancer_tr = y_bin_train == 1
    cancer_va = y_bin_val == 1
    cancer_te = y_bin_test == 1
    test_f1 = 0.0
    s2_val_full = np.zeros((len(y_bin_val), n_classes))
    s2_test_full = np.zeros((len(y_bin_test), n_classes))
    if cancer_tr.sum() > 5 and cancer_te.sum() > 0:
        final_s2 = best_factory("multiclass")
        final_s2_fit = fit_sparse_multiclass(
            final_s2, meta_s2_train_oof[cancer_tr], y_type_train[cancer_tr]
        )

        if cancer_va.sum() > 0:
            s2_val_full[cancer_va] = predict_sparse_multiclass_proba(
                final_s2_fit, meta_s2_val[cancer_va], n_classes
            )

        s2_test_full[cancer_te] = predict_sparse_multiclass_proba(
            final_s2_fit, meta_s2_test[cancer_te], n_classes
        )

        if groups_arr is not None:
            if cancer_va.sum() > 0:
                sex_va = infer_sex_from_groups(groups_val[cancer_va])
                s2_val_full[cancer_va] = apply_sex_constraint(
                    s2_val_full[cancer_va], CANCER_TYPES, sex_va,
                )
            sex_te = infer_sex_from_groups(groups_test[cancer_te])
            s2_test_full[cancer_te] = apply_sex_constraint(
                s2_test_full[cancer_te], CANCER_TYPES, sex_te,
            )

        s2_pred_test = s2_test_full[cancer_te].argmax(axis=1)
        test_f1 = f1_score(
            y_type_test[cancer_te], s2_pred_test,
            average="macro", zero_division=0,
        )

    logger.info(f"\n{'='*60}")
    logger.info(f"  🔥 FINAL TEST RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"  Meta-learner: {best_meta_name}")
    logger.info(f"  Stage 1 AUC:  {test_auc:.4f}")
    logger.info(f"  Stage 2 F1:   {test_f1:.4f}")
    logger.info(f"{'='*60}")

    logger.info("\n  -- Saving fixed-split artifacts for figures --")

    save_fixed_predictions(
        "fixed_test", s1_test_pred, s2_test_full,
        y_bin_test, y_type_test, sids_test, groups_test, OUTPUT_DIR,
    )
    save_fixed_predictions(
        "fixed_val", s1_val_pred, s2_val_full,
        y_bin_val, y_type_val, sids_val, groups_val, OUTPUT_DIR,
    )

    single_vs = compare_single_vs_ensemble(
        test_s1, test_s2, model_names,
        y_bin_test, y_type_test, n_classes,
        groups_arr=groups_test,
    )
    single_vs.append({
        "model": f"meta_{best_meta_name}",
        "type": "meta",
        "auc": round(float(test_auc), 6),
        "f1_type": round(float(test_f1), 6),
    })
    pd.DataFrame(single_vs).to_csv(OUTPUT_DIR / "single_vs_ensemble.csv", index=False)
    logger.info("  Saved single_vs_ensemble.csv")

    contributions = compute_base_model_contribution(
        test_s1, model_names, y_bin_test, y_type_test, n_classes,
    )
    pd.DataFrame(contributions).to_csv(OUTPUT_DIR / "base_model_contribution.csv", index=False)
    logger.info("  Saved base_model_contribution.csv")

    base_summary = []
    for m in model_names:
        base_summary.append({
            "model": m,
            "val_auc": round(float(roc_auc_score(y_bin_val, val_s1[m])), 6),
            "test_auc": round(float(roc_auc_score(y_bin_test, test_s1[m])), 6),
        })
    pd.DataFrame(base_summary).to_csv(OUTPUT_DIR / "base_model_auc_summary.csv", index=False)
    logger.info("  Saved base_model_auc_summary.csv")

    pd.DataFrame(meta_results).to_csv(OUTPUT_DIR / "meta_learner_comparison.csv", index=False)
    logger.info("  Saved meta_learner_comparison.csv")

    if cancer_te.sum() > 0:
        y_true_c = y_type_test[cancer_te]
        y_pred_c = s2_test_full[cancer_te].argmax(axis=1)
        cancer_names = list(CANCER_TYPES)
        labels_idx = list(range(n_classes))

        cm = confusion_matrix(y_true_c, y_pred_c, labels=labels_idx)
        cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)

        per_f1 = f1_score(y_true_c, y_pred_c, labels=labels_idx,
                          average=None, zero_division=0)
        per_prec = precision_score(y_true_c, y_pred_c, labels=labels_idx,
                                   average=None, zero_division=0)
        per_rec = recall_score(y_true_c, y_pred_c, labels=labels_idx,
                               average=None, zero_division=0)
        per_spec = np.zeros(n_classes)
        for i in range(n_classes):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - tp - fp - fn
            per_spec[i] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        pd.DataFrame({
            "cancer": cancer_names,
            "n": cm.sum(axis=1).astype(int),
            "sensitivity": np.round(per_rec, 6),
            "specificity": np.round(per_spec, 6),
            "precision": np.round(per_prec, 6),
            "recall": np.round(per_rec, 6),
            "f1": np.round(per_f1, 6),
        }).to_csv(OUTPUT_DIR / "cancer_type_metrics.csv",
                  index=False, encoding="utf-8-sig")
        pd.DataFrame(cm, index=cancer_names, columns=cancer_names).to_csv(
            OUTPUT_DIR / "confusion_matrix_counts.csv", encoding="utf-8-sig",
        )
        pd.DataFrame(cm_norm, index=cancer_names, columns=cancer_names).to_csv(
            OUTPUT_DIR / "confusion_matrix_normalized.csv", encoding="utf-8-sig",
        )
        logger.info("  Saved cancer_type_metrics.csv and confusion matrices")

    fpr, tpr, thresholds = roc_curve(y_bin_test, s1_test_pred)
    atomic_save_npz(
        OUTPUT_DIR / "roc_curve_data.npz",
        fpr=fpr,
        tpr=tpr,
        thresholds=thresholds,
        auc=np.array([test_auc]),
    )
    logger.info("  Saved roc_curve_data.npz")

    atomic_save_json(OUTPUT_DIR / "fixed_split_results.json", {
        "split_mode": "fixed",
        "best_meta": best_meta_name,
        "test_auc": round(float(test_auc), 6),
        "test_f1": round(float(test_f1), 6),
        "split_sizes": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "meta_results": [
            {
                "meta_learner": r["meta_learner"],
                "val_s1_auc": round(float(r["val_s1_auc"]), 6),
                "val_s2_f1": round(float(r["val_s2_f1"]), 6),
                "combined": round(float(r["combined"]), 6),
            }
            for r in meta_results
        ],
        "base_model_auc": base_summary,
        "n_classes": int(n_classes),
        "cancer_types": list(CANCER_TYPES),
        "model_names": model_names,
    })
    logger.info("  Saved fixed_split_results.json")

    return {
        "meta_results": meta_results,
        "best_meta": best_meta_name,
        "test_auc": test_auc,
        "test_f1": test_f1,
        "split_sizes": {
            "train": len(train_idx),
            "val": len(val_idx),
            "test": len(test_idx),
        },
    }
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
                               binary_labels, cancer_type_labels, n_classes,
                               groups_arr=None):
    """Compare best single model performance vs ensemble."""
    sex_arr = infer_sex_from_groups(groups_arr) if groups_arr is not None else None
    rows = []
    for m_name in model_names:
        valid = ~np.isnan(outer_oof_s1[m_name])
        if valid.sum() < 20:
            continue
        auc = roc_auc_score(binary_labels[valid], outer_oof_s1[m_name][valid])
        cancer_mask = (binary_labels == 1) & valid
        if cancer_mask.sum() > 0:
            s2_probs = outer_oof_s2[m_name][cancer_mask]
            if sex_arr is not None:
                s2_probs = apply_sex_constraint(s2_probs, CANCER_TYPES, sex_arr[cancer_mask])
            f1 = f1_score(
                cancer_type_labels[cancer_mask],
                s2_probs.argmax(axis=1),
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
        s2_avg_probs = avg_s2[cancer_mask]
        if sex_arr is not None:
            s2_avg_probs = apply_sex_constraint(s2_avg_probs, CANCER_TYPES, sex_arr[valid][cancer_mask])
        f1_avg = f1_score(
            cancer_type_labels[valid][cancer_mask],
            s2_avg_probs.argmax(axis=1),
            average="macro", zero_division=0,
        )
    rows.append({"model": "ensemble_avg", "type": "ensemble", "auc": round(auc_avg, 6), "f1_type": round(f1_avg, 6)})

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _source_groups_for_request(group_name: str) -> list[str]:
    group = group_name.upper()
    return GROUP_EXPANSIONS.get(group, [group])


def load_val_group_data(group_name, grid):
    """Load a specific source group for external val-group inference."""
    raw_groups = _source_groups_for_request(group_name)

    if DATA_TYPE == "processed_csv":
        logger.info(f"  Loading {raw_groups} from processed CSV source ...")
        X_3ch, df_meta, _ = load_processed_multichannel(DATA_DIR, target_grid=grid)
        X_3ch, df_meta = _filter_source_groups(
            X_3ch, df_meta, include=set(raw_groups), label="val-group"
        )
    else:
        data_dir = PROJECT_ROOT / "data" / "raw_data"

        # Full folder mapping (includes groups not in training, e.g. SPAN)
        import yaml
        with open(PROJECT_ROOT / "config" / "config.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        full_map = cfg["dataset"]["folder_to_group"]

        # Reverse lookup from full config mapping
        target_map = {k: v for k, v in full_map.items() if v in raw_groups}
        if not target_map:
            raise ValueError(f"No folder mapping found for group(s) {raw_groups}")

        logger.info(f"  Loading {raw_groups} from {len(target_map)} folders ...")
        X_3ch, df_meta = load_raw_multichannel(data_dir, target_map, grid, "*.CSV")
        logger.info(f"  Loaded {len(X_3ch)} spectra for {raw_groups}")

    df_meta = _prepare_sample_ids_and_aliases(df_meta, merge_aliases=False)
    X_3ch_agg, df_meta_agg = _aggregate_mean_by_sample(X_3ch, df_meta)

    logger.info(f"  Aggregated: {len(X_3ch_agg)} samples")
    return X_3ch_agg, df_meta_agg


def load_meta_training_features(model_names: list[str]):
    """Load OOF meta features from nested-CV or fixed-split development output."""
    candidates = [
        (OUTPUT_DIR / "oof_predictions.npz", "nested_cv"),
        (OUTPUT_DIR / "fixed_train_oof.npz", "fixed_train_oof"),
        (_LEGACY_OUTPUT_DIR / "oof_predictions.npz", "legacy_nested_cv"),
    ]

    for path, kind in candidates:
        if not path.exists():
            continue

        oof = np.load(path, allow_pickle=True)
        if "meta_s1" in oof.files and "meta_s2" in oof.files:
            meta_s1 = np.nan_to_num(oof["meta_s1"], nan=0.0)
            meta_s2 = np.nan_to_num(oof["meta_s2"], nan=0.0)
        else:
            missing = [
                key
                for m in model_names
                for key in (f"{m}_s1", f"{m}_s2")
                if key not in oof.files
            ]
            if missing:
                raise KeyError(f"{path} is missing OOF keys: {missing[:8]}")
            meta_s1 = np.nan_to_num(
                np.column_stack([oof[f"{m}_s1"] for m in model_names]), nan=0.0
            )
            meta_s2 = np.nan_to_num(
                np.hstack([oof[f"{m}_s2"] for m in model_names]), nan=0.0
            )

        if "binary_labels" in oof.files:
            y_bin = oof["binary_labels"].astype(int)
        elif "y_bin" in oof.files:
            y_bin = oof["y_bin"].astype(int)
        else:
            raise KeyError(f"{path} is missing binary labels")

        if "cancer_type_labels" in oof.files:
            y_type = oof["cancer_type_labels"].astype(int)
        elif "y_type" in oof.files:
            y_type = oof["y_type"].astype(int)
        else:
            raise KeyError(f"{path} is missing cancer type labels")

        if len(meta_s1) != len(y_bin) or len(meta_s2) != len(y_bin):
            raise ValueError(
                f"OOF feature/label length mismatch in {path}: "
                f"meta_s1={len(meta_s1)}, meta_s2={len(meta_s2)}, labels={len(y_bin)}"
            )

        logger.info(f"  Loaded meta-training OOF from {path} ({kind}, n={len(y_bin)})")
        return meta_s1, meta_s2, y_bin, y_type, path, kind

    searched = ", ".join(str(p) for p, _ in candidates)
    raise FileNotFoundError(f"OOF predictions not found. Searched: {searched}")


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

    logger.info(f"\n[Step 5] Loading OOF predictions for meta-learner training ...")
    try:
        meta_s1_train, meta_s2_train, y_bin_meta, y_type_meta, oof_path, oof_kind = (
            load_meta_training_features(model_names)
        )
    except Exception as e:
        logger.error(f"  Failed to load OOF predictions for meta-learner training: {e}")
        logger.error("  Run development training first, then rerun with --val-group.")
        return 1
    if oof_kind == "fixed_train_oof":
        logger.warning(
            "  Using fixed-split train OOF for meta training. "
            "For final locked external validation, prefer nested_cv OOF or a dedicated production build."
        )

    # ── 6. Train meta-learner on full OOF → predict val-group ──
    logger.info(f"\n[Step 6] Training {meta_type} meta-learner → predicting {group_name} ...")
    meta_factories = make_meta_learners(n_classes)
    if meta_type not in meta_factories:
        logger.error(f"  Unknown meta-learner: {meta_type}. Available: {list(meta_factories.keys())}")
        return 1

    factory = meta_factories[meta_type]

    # Stage 1
    ml_s1 = factory("binary")
    ml_s1.fit(meta_s1_train, y_bin_meta)
    try:
        s1_final = ml_s1.predict_proba(meta_s1_val)[:, 1]
    except AttributeError:
        s1_final = ml_s1.decision_function(meta_s1_val)

    # Stage 2
    cancer_mask_train = y_bin_meta == 1
    ml_s2 = factory("multiclass")
    ml_s2_fit = fit_sparse_multiclass(
        ml_s2, meta_s2_train[cancer_mask_train], y_type_meta[cancer_mask_train]
    )
    s2_full = predict_sparse_multiclass_proba(ml_s2_fit, meta_s2_val, n_classes)
    # Apply sex constraint
    sex_val_group = infer_sex_from_groups(val_groups)
    s2_full = apply_sex_constraint(s2_full, CANCER_TYPES, sex_val_group)
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

    from scripts.analysis.stk_v2.figures.val_group import plot_val_group_summary

    summary_fig = plot_val_group_summary(
        out_dir, group_name, s1_final, pred_counts, base_s1, model_names,
    )
    logger.info(f"  Saved {summary_fig.name}")

    # ── 10. Confusion Matrix ──
    if not args.no_cm:
        logger.info(f"\n[Step 10] Generating confusion matrices ...")
        from scripts.analysis.stk_v2.figures.val_group import plot_val_group_confusion_matrices

        # Ground truth: SPAN → determine expected label
        # Map val group to cancer type index (SPAN is pancreatic → PAN)
        GROUP_TO_EXPECTED = {
            "SPAN": "PAN", "CPAN": "PAN", "YPAN": "PAN",
        }
        expected_type = GROUP_TO_EXPECTED.get(group_name, group_name)

        cm_fig, cm_metrics = plot_val_group_confusion_matrices(
            out_dir, group_name, s1_final, s2_pred, CANCER_TYPES, expected_type=expected_type,
        )
        logger.info(f"  [Stage 1] Cancer Screening Metrics:")
        logger.info(f"    Sensitivity (TPR): {cm_metrics['s1_sensitivity']:.4f}")
        logger.info(f"    Specificity (TNR): {cm_metrics['s1_specificity']:.4f}")
        logger.info(f"    Accuracy: {cm_metrics['s1_accuracy']:.4f}")

        if "s2_accuracy" in cm_metrics:
            logger.info(f"  [Stage 2] Cancer Type Classification (GT={expected_type}):")
            logger.info(f"    Sensitivity (TPR): {cm_metrics['s2_sensitivity']:.4f}")
            logger.info(f"    Specificity (TNR): {cm_metrics['s2_specificity']:.4f}")
            logger.info(f"    Accuracy: {int(cm_metrics['s2_correct'])}/{n_val} = {cm_metrics['s2_accuracy']:.4f}")
        logger.info(f"  Saved {cm_fig.name}")

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
            from scripts.analysis.stk_v2.figures.val_group import (
                plot_meta_shap_s1,
                plot_meta_shap_s2,
                plot_spectral_shap,
            )

            # ── 11a. Meta-learner SHAP (which base models drive the decision) ──
            logger.info(f"\n[Step 11a] Meta-learner SHAP ...")
            try:
                meta_s1_bg = np.nan_to_num(meta_s1_train, nan=0.0)
                n_bg = min(args.shap_samples, len(meta_s1_bg))
                bg_idx = np.random.RandomState(42).choice(len(meta_s1_bg), n_bg, replace=False)

                explainer_s1 = shap.LinearExplainer(ml_s1, meta_s1_bg[bg_idx])
                shap_vals_s1 = explainer_s1.shap_values(meta_s1_val)

                plot_meta_shap_s1(shap_dir, group_name, model_names, shap_vals_s1)

                # Stage 2 meta SHAP
                meta_s2_bg = np.nan_to_num(meta_s2_train, nan=0.0)
                cancer_mask_bg = y_bin_meta == 1
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
                plot_meta_shap_s2(shap_dir, group_name, model_names, per_model_shap)
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

                    plot_spectral_shap(shap_dir, bm_name, wavenumbers, ch, mean_shap_bm)
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
    parser.add_argument("--project-root", type=Path, default=None,
                        help="Project root directory (default: auto-detect from script location)")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Raw spectra directory, or a processed_spectra*.csv path when --data-type processed_csv "
                             "(default: <project-root>/data/raw_data)")
    parser.add_argument("--data-type", choices=["raw_spectrum", "processed_csv"], default="raw_spectrum",
                        help="Data format: 'raw_spectrum' (load via load_raw_multichannel) or "
                             "'processed_csv' (expects group, sample_id, replicate, x_... columns)")
    parser.add_argument("--exclude-source-groups", action="append", default=[],
                        help="Comma-separated raw/source group codes to exclude before alias merging "
                             "(repeatable; e.g. --exclude-source-groups YPAN)")
    parser.add_argument("--output-name", type=str, default="stacking_v2",
                        help="Output directory name under results/training/ (default: stacking_v2)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--val-group", type=str, default=None)
    parser.add_argument("--meta-learner", type=str, default="elasticnet")
    parser.add_argument("--no-shap", action="store_true")
    parser.add_argument("--no-cm", action="store_true")
    parser.add_argument("--shap-samples", type=int, default=100)
    parser.add_argument("--split-mode", choices=["nested_cv", "fixed"],
                        default="fixed")
    parser.add_argument("--force-test-group", type=str, default="YPAN_")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Parallel jobs for sklearn/XGBoost estimators (-1 uses all CPU cores).")
    parser.add_argument("--xgb-device", choices=["cpu", "cuda"], default="cpu",
                        help="XGBoost device. Use 'cuda' only when WSL can access the GPU.")
    args = parser.parse_args()

    # ── Resolve project root, data dir, output dir ──
    global PROJECT_ROOT, DATA_DIR, DATA_TYPE, OUTPUT_DIR, FIG_OUTPUT_DIR, CHECKPOINT_DIR, _LEGACY_OUTPUT_DIR, _LEGACY_CHECKPOINT_DIR, MODEL_N_JOBS, XGB_DEVICE, EXCLUDE_SOURCE_GROUPS
    
    if args.project_root is None:
        PROJECT_ROOT = _DEFAULT_PROJECT_ROOT
    else:
        PROJECT_ROOT = args.project_root.expanduser().resolve()
    
    if args.data_dir is None:
        DATA_DIR = PROJECT_ROOT / "data" / "raw_data"
    else:
        DATA_DIR = args.data_dir.expanduser().resolve()
    
    DATA_TYPE = args.data_type
    EXCLUDE_SOURCE_GROUPS = parse_group_arg_values(args.exclude_source_groups)
    MODEL_N_JOBS = args.n_jobs
    XGB_DEVICE = args.xgb_device
    configure_runtime(n_jobs=MODEL_N_JOBS, xgb_device=XGB_DEVICE)
    
    # Update output directories based on --output-name
    OUTPUT_DIR = PROJECT_ROOT / "results" / "training" / args.output_name
    FIG_OUTPUT_DIR = PROJECT_ROOT / "results" / "figures" / "training" / args.output_name
    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
    _LEGACY_OUTPUT_DIR = PROJECT_ROOT / "results" / "weekend_experiments" / "stacking_optimization"
    _LEGACY_CHECKPOINT_DIR = _LEGACY_OUTPUT_DIR / "checkpoints"
    
    logger.info(f"Project root:     {PROJECT_ROOT}")
    logger.info(f"Data directory:   {DATA_DIR}")
    logger.info(f"Data type:        {DATA_TYPE}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Figure directory: {FIG_OUTPUT_DIR}")
    logger.info(f"Exclude source groups: {sorted(EXCLUDE_SOURCE_GROUPS) or 'none'}")
    logger.info(f"Model n_jobs:     {MODEL_N_JOBS}")
    logger.info(f"XGBoost device:   {XGB_DEVICE}")

    # ── Val-group inference (별도 경로) ──
    if args.val_group:
        return run_val_group_inference(args)

    t0 = datetime.now()

    # ── Setup ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(OUTPUT_DIR / "experiment.log", mode="a")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
    ))
    logging.getLogger().addHandler(fh)

    logger.info("=" * 64)
    logger.info(f"  STK-V2 | mode={args.split_mode} | {t0.isoformat()}")
    logger.info("=" * 64)

    # ── Load data ──
    (X_3ch_agg, df_meta_agg, grid, groups_arr,
     sample_ids, binary_labels, cancer_type_labels) = load_data(DATA_DIR)
    n_classes = len(CANCER_TYPES)
    meta_learner_factories = make_meta_learners(n_classes)

    # ════════════════════════════════════════════
    # Route A: Fixed 60/20/20
    # ════════════════════════════════════════════
    if args.split_mode == "fixed":
        train_idx, val_idx, test_idx = create_fixed_split(
            sample_ids, binary_labels,
            stratify_labels=groups_arr,
            test_ratio=0.20, val_ratio=0.20,
            force_test_prefix=args.force_test_group,
        )

        # 그룹별 분포 로깅
        for name, idx in [("Train", train_idx), ("Val", val_idx), ("Test", test_idx)]:
            counts = pd.Series(groups_arr[idx]).value_counts().sort_index()
            logger.info(f"\n  {name} set group distribution:")
            for g, c in counts.items():
                logger.info(f"    {g}: {c}")

        results = run_fixed_split(
            X_3ch_agg, grid, sample_ids, binary_labels, cancer_type_labels,
            n_classes, EXTENDED_BASE_MODELS, meta_learner_factories,
            groups_arr=groups_arr,
            train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
        )

        # Save run metadata without overwriting the detailed fixed-split figure artifact.
        atomic_save_json(OUTPUT_DIR / "fixed_split_run_summary.json", {
            "split_mode": "fixed",
            "force_test_group": args.force_test_group,
            "exclude_source_groups": sorted(EXCLUDE_SOURCE_GROUPS),
            "stratify": "group",
            "best_meta": results["best_meta"],
            "test_auc": results["test_auc"],
            "test_f1": results["test_f1"],
            "split_sizes": results["split_sizes"],
            "meta_results": results["meta_results"],
        })
        logger.info(
            "  Saved fixed-split artifacts. Render figures with: "
            f"python -m scripts.analysis.stk_v2.figures.fixed_split "
            f"--run-dir {OUTPUT_DIR} --fig-dir {FIG_OUTPUT_DIR}"
        )

        elapsed = (datetime.now() - t0).total_seconds()
        logger.info(f"\n{'='*64}")
        logger.info(f"  FIXED SPLIT COMPLETE ({elapsed:.0f}s)")
        logger.info(f"  Best meta: {results['best_meta']}")
        logger.info(f"  Test AUC:  {results['test_auc']:.4f}")
        logger.info(f"  Test F1:   {results['test_f1']:.4f}")
        logger.info(f"{'='*64}")
        return 0

    # ════════════════════════════════════════════
    # Route B: Nested CV (기존)
    # ════════════════════════════════════════════
        config = {
            "experiment": "stacking_optimization",
            "timestamp": t0.isoformat(),
            "dry_run": args.dry_run,
            "exclude_source_groups": sorted(EXCLUDE_SOURCE_GROUPS),
            "n_samples": len(X_3ch_agg),
        "n_cancer": int(binary_labels.sum()),
        "n_non_cancer": int((binary_labels == 0).sum()),
        "n_classes": n_classes,
        "cancer_types": list(CANCER_TYPES),
        "base_models": list(EXTENDED_BASE_MODELS.keys()),
        "meta_learners": list(meta_learner_factories.keys()),
    }
    atomic_save_json(OUTPUT_DIR / "config.json", config)

    (all_results, outer_oof_s1, outer_oof_s2,
     model_names, meta_names, best_meta_per_fold, X_peak_all) = run_nested_cv(
        X_3ch_agg, grid, sample_ids, binary_labels, cancer_type_labels,
        n_classes, EXTENDED_BASE_MODELS, meta_learner_factories,
        dry_run=args.dry_run, groups_arr=groups_arr,
    )

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "nested_cv_results.csv", index=False)

    oof_arrays = {
        "binary_labels": binary_labels,
        "cancer_type_labels": cancer_type_labels,
        "sample_ids": sample_ids,
    }
    for m_name in model_names:
        oof_arrays[f"{m_name}_s1"] = outer_oof_s1[m_name]
        oof_arrays[f"{m_name}_s2"] = outer_oof_s2[m_name]
    atomic_save_npz(OUTPUT_DIR / "oof_predictions.npz", **oof_arrays)

    # Best ensemble
    summary = results_df.groupby("meta_learner").agg(
        mean_s1_auc=("s1_auc", "mean"), std_s1_auc=("s1_auc", "std"),
        mean_s2_f1=("s2_f1", "mean"), std_s2_f1=("s2_f1", "std"),
    ).reset_index()
    summary["combined"] = summary["mean_s1_auc"] * 0.6 + summary["mean_s2_f1"] * 0.4
    best_row = summary.loc[summary["combined"].idxmax()]
    best_meta = best_row["meta_learner"]

    atomic_save_json(OUTPUT_DIR / "best_ensemble_config.json", {
        "best_meta_learner": best_meta,
        "mean_s1_auc": round(float(best_row["mean_s1_auc"]), 6),
        "mean_s2_f1": round(float(best_row["mean_s2_f1"]), 6),
        "base_models_used": model_names,
    })

    # Analysis
    contributions = compute_base_model_contribution(
        outer_oof_s1, model_names, binary_labels, cancer_type_labels, n_classes,
    )
    pd.DataFrame(contributions).to_csv(OUTPUT_DIR / "base_model_contribution.csv", index=False)

    single_vs = compare_single_vs_ensemble(
        outer_oof_s1, outer_oof_s2, model_names,
        binary_labels, cancer_type_labels, n_classes, groups_arr=groups_arr,
    )
    pd.DataFrame(single_vs).to_csv(OUTPUT_DIR / "single_vs_ensemble.csv", index=False)

    logger.info(
        "  Saved training artifacts. Render figures with: "
        f"python -m scripts.analysis.stk_v2.figures.training_diagnostics "
        f"--run-dir {OUTPUT_DIR} --fig-dir {FIG_OUTPUT_DIR}"
    )

    # Summary
    elapsed = (datetime.now() - t0).total_seconds()
    logger.info(f"\n{'='*64}")
    logger.info(f"  NESTED CV COMPLETE ({elapsed:.0f}s)")
    logger.info(f"  Best meta: {best_meta}  AUC={best_row['mean_s1_auc']:.4f}  F1={best_row['mean_s2_f1']:.4f}")
    logger.info(f"{'='*64}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
