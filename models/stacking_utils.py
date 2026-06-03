#!/usr/bin/env python3
"""
Stacking Ensemble — Multi-View SERS Classification
====================================================

Level 0: 9 diverse base models on different feature views
Level 1: Meta-LR combining out-of-fold predictions

Usage:
    python scripts/analysis/stacking_ensemble.py --data thermo
    python scripts/analysis/stacking_ensemble.py --data medical
"""

from __future__ import annotations
import sys, json, argparse, logging, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, f1_score

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sers.preprocessing import trim_spectrum, baseline_correction, normalize_spectrum, resample
from src.sers.io import find_spectra, read_spectrum, parse_filename
from src.sers.config import RESULTS_DIR, FIG_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(levelname)-7s │ %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

OUTPUT_DIR = RESULTS_DIR / "stacking_ensemble"
FIG_OUTPUT_DIR = FIG_DIR / "analysis" / "stacking_ensemble"

CANCER_TYPES = ("PRO", "LUN", "CRC", "PAN", "OVA", "BRE", "BLC")
NON_CANCER_GROUPS = ("NOR", "DIA", "HBP", "H.D.")
SG_WINDOW, SG_POLY = 11, 3
MODEL_N_JOBS = -1
XGB_DEVICE = "cpu"


def configure_runtime(n_jobs: int | None = None, xgb_device: str | None = None):
    """Configure CPU parallelism and optional XGBoost device for STK-V2 helpers."""
    global MODEL_N_JOBS, XGB_DEVICE
    if n_jobs is not None:
        MODEL_N_JOBS = int(n_jobs)
    if xgb_device is not None:
        XGB_DEVICE = str(xgb_device)


# =============================================================================
# Data loading (reuse from multiview_ensemble)
# =============================================================================
def preprocess_channel(x, y, grid, deriv_order):
    y_sg = savgol_filter(y, SG_WINDOW, SG_POLY, deriv=deriv_order)
    x_tr, y_tr = trim_spectrum(x.copy(), y_sg, region=(400, 2200))
    if deriv_order == 0:
        y_tr = baseline_correction(y_tr, window=101)
    y_tr = normalize_spectrum(y_tr, method="snv")
    return resample(x_tr, y_tr, grid)


def load_raw_multichannel(data_dir, folder_map, grid, pattern="*.CSV", max_rep=None):
    all_X, meta_rows, failed = [], [], 0
    for folder_name, group in folder_map.items():
        folder = data_dir / folder_name
        if not folder.is_dir():
            continue
        files = find_spectra(folder, pattern=pattern, recursive=False)
        if not files:
            for p in ["*.txt", "*.csv", "*.CSV"]:
                files = find_spectra(folder, pattern=p, recursive=False)
                if files: break
        files = [f for f in files if "_ave" not in f.stem.lower() and "zone.identifier" not in f.name.lower()]
        for fp in files:
            try:
                sid = parse_filename(fp, fallback_group=group)
                if max_rep and sid.group in max_rep and sid.replicate > max_rep[sid.group]:
                    continue
                x, y = read_spectrum(fp)
                ch0 = preprocess_channel(x, y, grid, 0)
                ch1 = preprocess_channel(x, y, grid, 1)
                ch2 = preprocess_channel(x, y, grid, 2)
                all_X.append(np.stack([ch0, ch1, ch2], axis=0))
                meta_rows.append({"group": sid.group, "sample_id": sid.sample_id, "replicate": sid.replicate})
            except Exception:
                failed += 1
    X = np.stack(all_X, axis=0).astype(np.float32)
    df = pd.DataFrame(meta_rows)
    logger.info(f"  Loaded {len(X)} spectra ({failed} failed), shape {X.shape}")
    return X, df


def _resolve_processed_csv_path(path_like) -> Path:
    path = Path(path_like)
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"processed_csv path not found: {path}")

    preferred = path / "processed_spectra.csv"
    if preferred.exists():
        return preferred

    candidates = sorted(path.glob("processed_spectra*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No processed_spectra*.csv found in {path}; pass a CSV file path via --data-dir"
        )
    if len(candidates) > 1:
        names = ", ".join(p.name for p in candidates[:8])
        if len(candidates) > 8:
            names += ", ..."
        raise ValueError(
            f"Multiple processed CSV files found in {path}: {names}. "
            "Pass the exact CSV file path via --data-dir."
        )
    return candidates[0]


def _processed_feature_columns(df: pd.DataFrame) -> list[str]:
    feature_cols = [c for c in df.columns if str(c).startswith("x_")]
    if not feature_cols:
        raise ValueError("processed CSV must contain spectrum columns prefixed with 'x_'")

    def sort_key(col: str):
        suffix = str(col)[2:]
        try:
            return (0, float(suffix))
        except ValueError:
            return (1, suffix)

    return sorted(feature_cols, key=sort_key)


def _grid_from_feature_columns(feature_cols: list[str], fallback_grid: np.ndarray | None = None) -> np.ndarray:
    parsed = []
    for col in feature_cols:
        try:
            parsed.append(float(str(col)[2:]))
        except ValueError:
            parsed = []
            break

    if len(parsed) == len(feature_cols):
        grid = np.asarray(parsed, dtype=float)
        # Columns named x_0, x_1, ... are indices, not physical wavenumbers.
        if grid.max(initial=0.0) > 300:
            return grid

    if fallback_grid is not None and len(fallback_grid) == len(feature_cols):
        return np.asarray(fallback_grid, dtype=float)
    return np.linspace(402.0, 2198.0, len(feature_cols))


def _snv_rows(X: np.ndarray) -> np.ndarray:
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    return (X - mean) / (std + 1e-8)


def load_processed_multichannel(csv_path_or_dir, target_grid=None):
    """Load preprocessed spectrum CSV as STK-V2 3-channel arrays.

    Supported columns:
      - metadata: group, sample_id, optional replicate
      - spectra: x_0...x_932 or physical columns such as x_402.00...x_2198.00

    The first channel is the processed spectrum as stored. Derivative channels
    are generated from that processed spectrum so the downstream STK-V2 feature
    views keep the same shape as raw_spectrum mode.
    """
    csv_path = _resolve_processed_csv_path(csv_path_or_dir)
    df = pd.read_csv(csv_path)
    required = {"group", "sample_id"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"processed CSV missing required columns: {missing}")

    feature_cols = _processed_feature_columns(df)
    source_grid = _grid_from_feature_columns(feature_cols, fallback_grid=target_grid)
    if target_grid is None:
        target_grid = source_grid
    target_grid = np.asarray(target_grid, dtype=float)

    X0 = df[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    X0 = np.nan_to_num(X0, nan=0.0, posinf=0.0, neginf=0.0)

    if len(source_grid) != len(target_grid) or not np.allclose(source_grid, target_grid, rtol=0, atol=1e-6):
        X0 = np.vstack([np.interp(target_grid, source_grid, row) for row in X0])

    ch0 = X0.astype(np.float32)
    window = min(SG_WINDOW, ch0.shape[1] if ch0.shape[1] % 2 == 1 else ch0.shape[1] - 1)
    if window <= SG_POLY:
        window = SG_POLY + 2 + ((SG_POLY + 2) % 2 == 0)
    ch1 = savgol_filter(ch0, window, SG_POLY, deriv=1, axis=1)
    ch2 = savgol_filter(ch0, window, SG_POLY, deriv=2, axis=1)
    ch1 = _snv_rows(ch1).astype(np.float32)
    ch2 = _snv_rows(ch2).astype(np.float32)

    X = np.stack([ch0, ch1, ch2], axis=1).astype(np.float32)
    meta = df[["group", "sample_id"]].copy()
    meta["group"] = meta["group"].astype(str)
    meta["sample_id"] = meta["sample_id"].astype(str)
    if "replicate" in df.columns:
        meta["replicate"] = df["replicate"].values
    else:
        meta["replicate"] = meta.groupby(["group", "sample_id"]).cumcount() + 1

    logger.info(f"  Loaded {len(X)} processed spectra from {csv_path}, shape {X.shape}")
    return X, meta, target_grid


# =============================================================================
# Base model definitions
# =============================================================================
BASE_MODELS = {
    "lr_raw":    {"channels": [0],       "model": "lr",    "pca": None},
    "lr_d1":     {"channels": [1],       "model": "lr",    "pca": None},
    "lr_d2":     {"channels": [2],       "model": "lr",    "pca": None},
    "lr_concat": {"channels": [0, 1, 2], "model": "lr",    "pca": None},
    "xgb_raw":   {"channels": [0],       "model": "xgb",   "pca": None},
    "xgb_d1":    {"channels": [1],       "model": "xgb",   "pca": None},
    "lr_pca":    {"channels": [0, 1, 2], "model": "lr",    "pca": 50},
    "ridge":     {"channels": [0, 1, 2], "model": "ridge",  "pca": None},
    "rf_raw":    {"channels": [0],       "model": "rf",    "pca": None},
}


def build_classifier(model_type, task="binary", n_classes=7):
    """Build a sklearn classifier for the given model type and task."""
    if model_type == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, max_iter=2000, solver="saga", class_weight="balanced",
            multi_class="multinomial" if task == "multiclass" else "auto",
            n_jobs=MODEL_N_JOBS))
    elif model_type == "ridge":
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=0.1, max_iter=2000, solver="saga", class_weight="balanced",
            multi_class="multinomial" if task == "multiclass" else "auto",
            n_jobs=MODEL_N_JOBS))
    elif model_type == "xgb":
        try:
            from xgboost import XGBClassifier
        except ImportError:
            logger.warning("XGBoost not available, falling back to LR")
            return build_classifier("lr", task, n_classes)
        if task == "binary":
            return XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                 subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                                 n_jobs=MODEL_N_JOBS, tree_method="hist",
                                 device=XGB_DEVICE, verbosity=0,
                                 objective="binary:logistic", eval_metric="logloss")
        else:
            return XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                 subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                                 n_jobs=MODEL_N_JOBS, tree_method="hist",
                                 device=XGB_DEVICE, verbosity=0,
                                 objective="multi:softprob", num_class=n_classes,
                                 eval_metric="mlogloss")
    elif model_type == "rf":
        return RandomForestClassifier(n_estimators=500, max_depth=None,
                                      min_samples_leaf=1, class_weight="balanced_subsample",
                                      n_jobs=MODEL_N_JOBS, random_state=42)
    raise ValueError(f"Unknown model type: {model_type}")


def train_base_model(spec, X_train, y_bin_train, y_type_train, X_val, n_classes):
    """Train a base model for both stages. Returns (s1_prob, s2_prob)."""
    # Prepare features
    ch = spec["channels"]
    pca_n = spec["pca"]
    model_type = spec["model"]

    X_tr = X_train[:, ch, :].reshape(len(X_train), -1)
    X_va = X_val[:, ch, :].reshape(len(X_val), -1)

    # PCA if needed (fit on train only)
    if pca_n:
        pca = PCA(n_components=pca_n, random_state=42)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)
        X_tr = pca.fit_transform(X_tr_s)
        X_va = pca.transform(X_va_s)

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


# =============================================================================
# Stacking CV
# =============================================================================
def run_stacking(X_3ch, df_meta, n_splits=5):
    """Run full stacking CV."""
    valid = set(CANCER_TYPES) | set(NON_CANCER_GROUPS)
    mask = df_meta["group"].isin(valid).values
    X_3ch = X_3ch[mask]
    df_meta = df_meta[mask].reset_index(drop=True)

    groups_arr = df_meta["group"].values
    sample_ids = (df_meta["group"] + "_" + df_meta["sample_id"].astype(str)).values
    binary_labels = np.array([1.0 if g in CANCER_TYPES else 0.0 for g in groups_arr])
    ct_map = {ct: i for i, ct in enumerate(CANCER_TYPES)}
    cancer_type_labels = np.array([ct_map.get(g, -1) for g in groups_arr])
    n_classes = len(CANCER_TYPES)
    n = len(X_3ch)
    n_models = len(BASE_MODELS)

    logger.info(f"  Data: {n} spectra, {int(binary_labels.sum())} cancer, {int((binary_labels==0).sum())} non-cancer")

    # OOF storage
    oof_s1 = {name: np.full(n, np.nan) for name in BASE_MODELS}
    oof_s2 = {name: np.full((n, n_classes), np.nan) for name in BASE_MODELS}

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = list(sgkf.split(X_3ch, binary_labels, sample_ids))

    # Level 0: collect OOF predictions
    logger.info(f"\n  ── Level 0: {n_models} base models × {n_splits} folds ──")
    for fold, (tr_idx, val_idx) in enumerate(splits):
        logger.info(f"  Fold {fold}:")
        for name, spec in BASE_MODELS.items():
            s1_p, s2_p = train_base_model(
                spec, X_3ch[tr_idx], binary_labels[tr_idx],
                cancer_type_labels[tr_idx], X_3ch[val_idx], n_classes)
            oof_s1[name][val_idx] = s1_p
            oof_s2[name][val_idx] = s2_p

        # Report fold metrics for each base model
        for name in BASE_MODELS:
            auc = roc_auc_score(binary_labels[val_idx], oof_s1[name][val_idx])
            cm = cancer_type_labels[val_idx] >= 0
            bm = binary_labels[val_idx] == 1
            f1t = f1_score(cancer_type_labels[val_idx][bm],
                          oof_s2[name][val_idx][bm].argmax(axis=1),
                          average="macro", zero_division=0) if bm.sum() > 0 else 0
            logger.info(f"    {name:<12} AUC={auc:.4f} F1-type={f1t:.4f}")

    # Level 1: meta-learner
    logger.info(f"\n  ── Level 1: Meta-LR ──")
    model_names = list(BASE_MODELS.keys())
    meta_s1 = np.column_stack([oof_s1[m] for m in model_names])  # (N, 9)
    meta_s2 = np.hstack([oof_s2[m] for m in model_names])        # (N, 63)

    final_s1 = np.full(n, np.nan)
    final_s2 = np.full((n, n_classes), np.nan)

    # Also compute simple average baseline
    avg_s1 = np.mean(meta_s1, axis=1)
    avg_s2_all = np.stack([oof_s2[m] for m in model_names], axis=0)
    avg_s2 = np.mean(avg_s2_all, axis=0)

    for fold, (tr_idx, val_idx) in enumerate(splits):
        # S1 meta-learner
        meta_lr_s1 = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        meta_lr_s1.fit(meta_s1[tr_idx], binary_labels[tr_idx])
        final_s1[val_idx] = meta_lr_s1.predict_proba(meta_s1[val_idx])[:, 1]

        # S2 meta-learner (cancer only in train)
        cancer_tr = binary_labels[tr_idx] == 1
        if cancer_tr.sum() > 10:
            meta_lr_s2 = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs",
                                            multi_class="multinomial")
            meta_lr_s2.fit(meta_s2[tr_idx][cancer_tr], cancer_type_labels[tr_idx][cancer_tr])
            raw = meta_lr_s2.predict_proba(meta_s2[val_idx])
            for i, cls in enumerate(meta_lr_s2.classes_):
                if cls < n_classes:
                    final_s2[val_idx, cls] = raw[:, i]

        auc = roc_auc_score(binary_labels[val_idx], final_s1[val_idx])
        bm = binary_labels[val_idx] == 1
        f1t = f1_score(cancer_type_labels[val_idx][bm],
                      final_s2[val_idx][bm].argmax(axis=1),
                      average="macro", zero_division=0) if bm.sum() > 0 else 0
        logger.info(f"  Fold {fold}: Stacking AUC={auc:.4f} F1-type={f1t:.4f}")

    # Compute overall metrics
    results = []

    # Each base model
    for name in model_names:
        valid_mask = ~np.isnan(oof_s1[name])
        auc = roc_auc_score(binary_labels[valid_mask], oof_s1[name][valid_mask])
        bm = (binary_labels == 1) & valid_mask
        f1t = f1_score(cancer_type_labels[bm], oof_s2[name][bm].argmax(axis=1),
                       average="macro", zero_division=0)
        s1_pred = (oof_s1[name][valid_mask] > 0.5).astype(int)
        f1b = f1_score(binary_labels[valid_mask], s1_pred, average="macro")
        results.append({"name": name, "type": "base", "auc": auc, "f1_bin": f1b, "f1_type": f1t})

    # Simple average
    auc_avg = roc_auc_score(binary_labels, avg_s1)
    bm = binary_labels == 1
    f1t_avg = f1_score(cancer_type_labels[bm], avg_s2[bm].argmax(axis=1), average="macro", zero_division=0)
    f1b_avg = f1_score(binary_labels, (avg_s1 > 0.5).astype(int), average="macro")
    results.append({"name": "simple_avg", "type": "ensemble", "auc": auc_avg, "f1_bin": f1b_avg, "f1_type": f1t_avg})

    # Stacking
    valid_mask = ~np.isnan(final_s1)
    auc_stack = roc_auc_score(binary_labels[valid_mask], final_s1[valid_mask])
    bm = (binary_labels == 1) & valid_mask
    f1t_stack = f1_score(cancer_type_labels[bm], final_s2[bm].argmax(axis=1), average="macro", zero_division=0)
    f1b_stack = f1_score(binary_labels[valid_mask], (final_s1[valid_mask] > 0.5).astype(int), average="macro")
    results.append({"name": "stacking", "type": "meta", "auc": auc_stack, "f1_bin": f1b_stack, "f1_type": f1t_stack})

    return results


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", default="thermo", choices=["thermo", "medical"])
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = datetime.now()

    logger.info("=" * 64)
    logger.info("  Stacking Ensemble — Multi-View SERS Classification")
    logger.info("=" * 64)

    THERMO_DIR = PROJECT_ROOT / "data" / "raw_data"
    MEDICAL_DIR = PROJECT_ROOT / "data" / "raw_data_medical"

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

    if args.data == "thermo":
        data_dir, folder_map, pattern, max_rep = THERMO_DIR, THERMO_MAP, "*.CSV", None
    else:
        data_dir, folder_map, pattern, max_rep = MEDICAL_DIR, MEDICAL_MAP, "*.txt", {"NOR": 6}

    logger.info(f"\n  Loading RAW spectra ({args.data})...")
    X_3ch, df_meta = load_raw_multichannel(data_dir, folder_map, grid, pattern, max_rep)
    # Disambiguate sample_ids before group merge (CPAN_4 ≠ YPAN_4)
    needs_prefix = df_meta["group"].isin(["YPAN", "YNOR"])
    df_meta.loc[needs_prefix, "sample_id"] = (
        df_meta.loc[needs_prefix, "group"] + "_" + df_meta.loc[needs_prefix, "sample_id"].astype(str)
    )
    df_meta["group"] = df_meta["group"].replace({"CPAN": "PAN", "YPAN": "PAN", "YNOR": "NOR"})

    logger.info(f"\n  Running Stacking CV...")
    results = run_stacking(X_3ch, df_meta, n_splits=args.n_splits)

    FIG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    from scripts.analysis.stk_v2.figures.stacking_results import plot_results

    plot_results(results, FIG_OUTPUT_DIR, args.data)

    report = {"timestamp": datetime.now().isoformat(), "data": args.data,
              "duration_sec": (datetime.now() - t0).total_seconds(), "results": results}
    with open(OUTPUT_DIR / f"stacking_report_{args.data}.json", "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"\n{'='*64}")
    logger.info(f"  STACKING RESULTS ({args.data.upper()})")
    logger.info(f"{'='*64}")
    logger.info(f"  {'Name':<14} {'Type':<8} {'AUC':>8} {'F1-bin':>8} {'F1-type':>8}")
    logger.info(f"  {'-'*14} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for r in results:
        logger.info(f"  {r['name']:<14} {r['type']:<8} {r['auc']:>8.4f} {r['f1_bin']:>8.4f} {r['f1_type']:>8.4f}")
    logger.info(f"{'='*64}")
    logger.info(f"  Duration: {(datetime.now() - t0).total_seconds():.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
