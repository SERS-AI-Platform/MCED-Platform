"""Train LR / RF / XGB / CNN1D 7-cancer on same medoid 5-fold split as ResNet18.

Outputs per-model OOF predictions matching `resnet18_ckpts/fold_inputs.npz`:
    val_binary_prob (n,)
    val_cancer_logits (n, 7)  — softmax probs, sex constraint applied
Plus LR coefficients per fold (for combined uSERS-Net attribution).

Stage 1: Binary cancer detection.
Stage 2: 7-class cancer-type classification on cancer samples only.
"""
from __future__ import annotations
import importlib.util as _iu
import logging
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from sers.models._legacy.resnet_v1.model import ModelConfig, SERSDataset, CNN1DCancerDetector

_spec = _iu.spec_from_file_location("legacy_train_resnet",
                                    PROJECT_ROOT / "scripts" / "training" / "_legacy" / "train_resnet.py")
_legacy = _iu.module_from_spec(_spec)
sys.modules["legacy_train_resnet"] = _legacy
_spec.loader.exec_module(_legacy)
train_fold = _legacy.train_fold

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("base_models")

CANCER_TYPES = ["PRO", "BRE", "OVA", "LUN", "CRC", "PAN", "BLC"]
NON_CANCER = ["NOR", "DIA", "HBP", "H.D."]
N_SPLITS = 5
RANDOM_STATE = 42
RUN_DIR = PROJECT_ROOT / "results" / "runs" / "2026-05-11_usersnet_alpha0.8_ovr_roc"
INPUTS = RUN_DIR / "resnet18_ckpts" / "fold_inputs.npz"
OUT_DIR = RUN_DIR / "extra_models"; OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_inputs():
    d = np.load(INPUTS, allow_pickle=True)
    return (d["X"].astype(np.float32), d["bl"].astype(int), d["ctl"].astype(int),
            d["sample_ids"], d["groups_arr"], d["wavenumbers"],
            [str(c) for c in d["cancer_types"].tolist()])


def get_fold_indices(X, bl, ctl, sample_ids):
    composite = bl * 100 + np.clip(ctl, 0, 99)
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    return [(tr, va) for tr, va in sgkf.split(X, composite, groups=sample_ids)]


def apply_sex_constraint(probs: np.ndarray, sex: np.ndarray) -> np.ndarray:
    out = probs.copy()
    pro_idx, ova_idx = CANCER_TYPES.index("PRO"), CANCER_TYPES.index("OVA")
    out[sex == 0, pro_idx] = 0.0
    out[sex == 1, ova_idx] = 0.0
    s = out.sum(axis=1, keepdims=True)
    nz = s.squeeze() > 0
    out[nz] = out[nz] / s[nz]
    return out


def load_sex_array(sample_ids):
    """Read sex per patient from raw clinical CSV through processed_spectra.csv index."""
    # processed_spectra holds sex per replicate; we used medoid so sample_ids unique per patient
    raw = pd.read_csv(PROJECT_ROOT / "results" / "processed_spectra.csv",
                      usecols=["sample_id", "group"])
    # processed_spectra does not necessarily carry sex; try clinical merge
    clin = PROJECT_ROOT / "data" / "raw_data" / "clinical.csv"
    if clin.exists():
        c = pd.read_csv(clin)
        if "sex" in c.columns and "sample_id" in c.columns:
            m = c.set_index("sample_id")["sex"].to_dict()
            sex_str = np.array([m.get(s, "U") for s in sample_ids])
            sex_arr = np.where(np.isin(sex_str, ["M", "Male", "남", 1, "1"]), 1, 0)
            return sex_arr
    # Fallback: derive sex from disease group (cancer-type-specific)
    logger.warning("Falling back to group-based sex inference (PRO=male, OVA/BRE=female)")
    sex_arr = np.zeros(len(sample_ids), dtype=int)
    return sex_arr


def train_classical(model_name, X, bl, ctl, sample_ids, fold_pairs):
    n, L = X.shape
    val_bp = np.full(n, np.nan)
    val_cl = np.full((n, len(CANCER_TYPES)), np.nan)
    lr_coefs = None
    if model_name == "logistic_regression":
        lr_coefs = {"stage1_coef": np.zeros((N_SPLITS, L)),
                    "stage1_intercept": np.zeros(N_SPLITS),
                    "stage2_coef": np.zeros((N_SPLITS, len(CANCER_TYPES), L)),
                    "stage2_intercept": np.zeros((N_SPLITS, len(CANCER_TYPES))),
                    "stage1_scaler_mean": np.zeros((N_SPLITS, L)),
                    "stage1_scaler_scale": np.zeros((N_SPLITS, L)),
                    "stage2_scaler_mean": np.zeros((N_SPLITS, L)),
                    "stage2_scaler_scale": np.zeros((N_SPLITS, L)),
                    "stage2_classes": np.zeros((N_SPLITS, len(CANCER_TYPES)), dtype=int) - 1,
                    }

    for fi, (tr, va) in enumerate(fold_pairs):
        X_tr, X_va = X[tr], X[va]
        bl_tr, ctl_tr = bl[tr], ctl[tr]
        cancer_tr = ctl_tr >= 0

        # Stage 1
        if model_name == "logistic_regression":
            s1 = make_pipeline(StandardScaler(),
                               LogisticRegression(C=1.0, max_iter=1000, solver="saga",
                                                   class_weight="balanced", random_state=RANDOM_STATE))
        elif model_name == "random_forest":
            s1 = RandomForestClassifier(n_estimators=300, n_jobs=-1, class_weight="balanced",
                                        random_state=RANDOM_STATE)
        elif model_name == "xgboost":
            from xgboost import XGBClassifier
            n_pos, n_neg = (bl_tr == 1).sum(), (bl_tr == 0).sum()
            s1 = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                               objective="binary:logistic", random_state=RANDOM_STATE,
                               n_jobs=-1, scale_pos_weight=max(n_neg, 1) / max(n_pos, 1),
                               eval_metric="logloss")
        else:
            raise ValueError(model_name)
        s1.fit(X_tr, bl_tr)
        val_bp[va] = s1.predict_proba(X_va)[:, 1]

        # Stage 2 (cancer samples only)
        present = sorted(set(ctl_tr[cancer_tr].tolist()))
        if model_name == "logistic_regression":
            s2 = make_pipeline(StandardScaler(),
                               LogisticRegression(C=1.0, max_iter=1000, solver="saga",
                                                  class_weight="balanced", random_state=RANDOM_STATE))
        elif model_name == "random_forest":
            s2 = RandomForestClassifier(n_estimators=300, n_jobs=-1, class_weight="balanced",
                                        random_state=RANDOM_STATE)
        elif model_name == "xgboost":
            from xgboost import XGBClassifier
            s2 = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                               objective="multi:softprob", num_class=len(present),
                               random_state=RANDOM_STATE, n_jobs=-1, eval_metric="mlogloss")
        s2.fit(X_tr[cancer_tr], ctl_tr[cancer_tr])
        proba = s2.predict_proba(X_va)  # (n_va, n_present)
        # Map back to full 7-class vector
        full = np.zeros((len(va), len(CANCER_TYPES)), dtype=np.float64)
        for j_local, j_global in enumerate(present):
            full[:, j_global] = proba[:, j_local]
        val_cl[va] = full

        if model_name == "logistic_regression":
            sc1 = s1.named_steps["standardscaler"]
            lg1 = s1.named_steps["logisticregression"]
            sc2 = s2.named_steps["standardscaler"]
            lg2 = s2.named_steps["logisticregression"]
            lr_coefs["stage1_coef"][fi] = lg1.coef_[0]
            lr_coefs["stage1_intercept"][fi] = lg1.intercept_[0]
            lr_coefs["stage1_scaler_mean"][fi] = sc1.mean_
            lr_coefs["stage1_scaler_scale"][fi] = sc1.scale_
            lr_coefs["stage2_scaler_mean"][fi] = sc2.mean_
            lr_coefs["stage2_scaler_scale"][fi] = sc2.scale_
            for j_local, j_global in enumerate(present):
                lr_coefs["stage2_coef"][fi, j_global] = lg2.coef_[j_local]
                lr_coefs["stage2_intercept"][fi, j_global] = lg2.intercept_[j_local]
                lr_coefs["stage2_classes"][fi, j_global] = j_global
        logger.info(f"  [{model_name}] fold {fi+1}: val AUC_S1={roc_auc_score(bl[va], val_bp[va]):.4f}")
    return val_bp, val_cl, lr_coefs


def train_cnn1d(X, bl, ctl, sample_ids, fold_pairs, mc_proto):
    import yaml
    with open(PROJECT_ROOT / "config" / "config.yaml", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)
    # base mc with overrides
    n, L = X.shape
    mc_run = ModelConfig(**{**mc_proto.__dict__,
        "n_spectral_features": L,
        "learning_rate": 5e-4, "dropout_rate": 0.3, "weight_decay": 1e-3,
        "batch_size": 32, "n_epochs": 150,
        "random_state": RANDOM_STATE,
    })
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runtime = {"num_workers": 4 if device.type == "cuda" else 0,
               "pin_memory": device.type == "cuda",
               "prefetch_factor": 2 if device.type == "cuda" else None,
               "use_amp": device.type == "cuda"}

    val_bp = np.full(n, np.nan)
    val_cl = np.full((n, len(CANCER_TYPES)), np.nan)
    for fi, (tr, va) in enumerate(fold_pairs):
        train_ds = SERSDataset(X[tr], bl[tr], ctl[tr], augment=True)
        val_ds = SERSDataset(X[va], bl[va], ctl[va], augment=False)
        model = CNN1DCancerDetector(mc_run).to(device)
        fr = train_fold(model, train_ds, val_ds, mc_run, device, fi, runtime)
        vm = fr["val_metrics"]
        val_bp[va] = vm["binary_prob"]
        val_cl[va] = torch.softmax(torch.tensor(vm["cancer_logits"]), dim=-1).numpy()
        logger.info(f"  [cnn1d] fold {fi+1}: val AUC_S1={roc_auc_score(bl[va], val_bp[va]):.4f}")
        del model; torch.cuda.empty_cache()
    return val_bp, val_cl


def get_mc_proto(L):
    import yaml
    with open(PROJECT_ROOT / "config" / "config.yaml", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)
    mc = ModelConfig.from_pipeline_config(raw_cfg, n_spectral_features=L)
    from legacy_train_resnet import apply_class_selection
    mc = apply_class_selection(mc, cancer_types=CANCER_TYPES, non_cancer_groups=NON_CANCER)
    return mc


def main():
    X, bl, ctl, sample_ids, groups_arr, wavenumbers, cancer_types = load_inputs()
    assert cancer_types == CANCER_TYPES, f"cancer order mismatch: {cancer_types}"
    n, L = X.shape
    logger.info(f"Loaded: X={X.shape}, cancers/non={(bl==1).sum()}/{(bl==0).sum()}")

    fold_pairs = get_fold_indices(X, bl, ctl, sample_ids)

    # Prefer sex_for_constraint already stored in fold_inputs.npz (built by v2_prepare_data)
    fi = np.load(INPUTS, allow_pickle=True)
    if "sex_for_constraint" in fi.files:
        sex = fi["sex_for_constraint"].astype(int)
        logger.info("Sex: loaded from fold_inputs.npz")
    else:
        sex = load_sex_array(sample_ids)
    logger.info(f"Sex: male(1)={(sex==1).sum()}, female(0)={(sex==0).sum()}")
    np.save(OUT_DIR / "sex_for_constraint.npy", sex)

    results = {}
    for m in ["logistic_regression", "random_forest", "xgboost"]:
        logger.info(f"=== Training {m} ===")
        val_bp, val_cl, lr_coefs = train_classical(m, X, bl, ctl, sample_ids, fold_pairs)
        val_cl_sx = apply_sex_constraint(val_cl, sex)
        out = {"val_binary_prob": val_bp, "val_cancer_logits": val_cl_sx,
               "val_cancer_logits_raw": val_cl}
        if lr_coefs is not None:
            out.update(lr_coefs)
        np.savez_compressed(OUT_DIR / f"{m}_oof.npz", **out)
        results[m] = roc_auc_score(bl, val_bp)
        logger.info(f"  [{m}] overall S1 AUC = {results[m]:.4f}")

    logger.info("=== Training cnn1d ===")
    mc_proto = get_mc_proto(L)
    val_bp, val_cl = train_cnn1d(X, bl, ctl, sample_ids, fold_pairs, mc_proto)
    val_cl_sx = apply_sex_constraint(val_cl, sex)
    np.savez_compressed(OUT_DIR / "cnn1d_oof.npz",
                        val_binary_prob=val_bp, val_cancer_logits=val_cl_sx,
                        val_cancer_logits_raw=val_cl)
    results["cnn1d"] = roc_auc_score(bl, val_bp)
    logger.info(f"  [cnn1d] overall S1 AUC = {results['cnn1d']:.4f}")

    pd.DataFrame([{"model": k, "binary_auc": v} for k, v in results.items()]).to_csv(
        OUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    logger.info(f"Done. Outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
