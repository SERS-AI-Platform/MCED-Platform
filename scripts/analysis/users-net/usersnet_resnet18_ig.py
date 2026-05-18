"""ResNet18 7-cancer Integrated Gradients for cancer-specific peak attribution.

Train 5-fold ResNet18 with same setup as optimize_7cancer_full (aggregate=medoid,
channels=(64,128,256,512), lr=3e-4, dropout=0.3, wd=1e-3, n_epochs=150). Save each
fold's best_state. Then for each cancer logit, compute Integrated Gradients on the
OOF val samples and aggregate per-wavenumber attribution by cancer group.

Usage:
    python scripts/analysis/usersnet_resnet18_ig.py --stage train
    python scripts/analysis/usersnet_resnet18_ig.py --stage ig
    python scripts/analysis/usersnet_resnet18_ig.py --stage all
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.signal import find_peaks
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from sers.models._legacy.resnet_v1.model import ModelConfig, SERSDataset, SERSCancerDetector

# Direct import of legacy training module (Python 3.13 dataclass quirk requires sys.modules registration)
import importlib.util as _iu
_legacy_path = PROJECT_ROOT / "scripts" / "training" / "_legacy" / "train_resnet.py"
_spec = _iu.spec_from_file_location("legacy_train_resnet", _legacy_path)
_legacy = _iu.module_from_spec(_spec)
sys.modules["legacy_train_resnet"] = _legacy
_spec.loader.exec_module(_legacy)
load_processed_spectra = _legacy.load_processed_spectra
get_feature_columns = _legacy.get_feature_columns
apply_class_selection = _legacy.apply_class_selection
resolve_aliases = _legacy.resolve_aliases
aggregate_replicates = _legacy.aggregate_replicates
train_fold = _legacy.train_fold

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("rn18_ig")

CANCER_TYPES = ["PRO", "BRE", "OVA", "LUN", "CRC", "PAN", "BLC"]
NON_CANCER = ["NOR", "DIA", "HBP", "H.D."]
TARGET_CANCERS = ["BLC", "BRE", "CRC", "LUN", "OVA", "PAN", "PRO"]
N_SPLITS = 5
RANDOM_STATE = 42

RUN_DIR = PROJECT_ROOT / "results" / "runs" / "2026-05-11_usersnet_alpha0.8_ovr_roc"
RUN_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = RUN_DIR / "resnet18_ckpts"; CKPT_DIR.mkdir(exist_ok=True)
IG_DIR = RUN_DIR / "ig_per_cancer"; IG_DIR.mkdir(exist_ok=True)


def load_config_yaml():
    import yaml
    with open(PROJECT_ROOT / "config" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def prepare_data():
    raw_cfg = load_config_yaml()
    df = load_processed_spectra()
    feat_cols = get_feature_columns(df)
    wavenumbers = np.array([float(c.split("_")[1]) for c in feat_cols])

    mc = ModelConfig.from_pipeline_config(raw_cfg, n_spectral_features=len(feat_cols))
    mc = apply_class_selection(mc, cancer_types=CANCER_TYPES, non_cancer_groups=NON_CANCER)
    df = resolve_aliases(df, mc)
    df_agg = aggregate_replicates(df, feat_cols, "medoid")
    valid_groups = set(mc.cancer_types) | set(mc.non_cancer_groups)
    df_valid = df_agg[df_agg["group"].isin(valid_groups)].copy()

    X = df_valid[feat_cols].values.astype(np.float32)
    groups_arr = df_valid["group"].to_numpy()
    sample_ids = df_valid["sample_id"].to_numpy()
    bl = np.array([1 if g in mc.cancer_types else 0 for g in groups_arr])
    ctl = np.array([mc.cancer_type_index(g) for g in groups_arr])
    return X, bl, ctl, sample_ids, groups_arr, wavenumbers, mc


def make_config(mc, n_features):
    return ModelConfig(**{**mc.__dict__,
        "n_spectral_features": n_features,
        "learning_rate": 3e-4,
        "dropout_rate": 0.3,
        "weight_decay": 1e-3,
        "batch_size": 32,
        "n_epochs": 150,
        "resnet_channels": (64, 128, 256, 512),
        "encoder_output_dim": 512,
        "random_state": RANDOM_STATE,
    })


def train_all_folds():
    # If fold_inputs.npz already exists (e.g. produced by usersnet_v2_prepare_data.py),
    # reuse it rather than re-running prepare_data (which would overwrite cohort).
    fi_path = CKPT_DIR / "fold_inputs.npz"
    if fi_path.exists():
        logger.info(f"Reusing existing {fi_path}")
        d = np.load(fi_path, allow_pickle=True)
        X = d["X"].astype(np.float32)
        bl = d["bl"].astype(np.int64); ctl = d["ctl"].astype(np.int64)
        sample_ids = d["sample_ids"]; groups_arr = d["groups_arr"]
        wavenumbers = d["wavenumbers"]
        import yaml as _yaml
        with open(PROJECT_ROOT / "config" / "config.yaml", encoding="utf-8") as f:
            raw_cfg = _yaml.safe_load(f)
        mc = ModelConfig.from_pipeline_config(raw_cfg, n_spectral_features=X.shape[1])
        mc = apply_class_selection(mc, cancer_types=CANCER_TYPES, non_cancer_groups=NON_CANCER)
        np.save(CKPT_DIR / "wavenumbers.npy", wavenumbers)
    else:
        X, bl, ctl, sample_ids, groups_arr, wavenumbers, mc = prepare_data()
        np.save(CKPT_DIR / "wavenumbers.npy", wavenumbers)
        np.savez(CKPT_DIR / "fold_inputs.npz",
                 X=X, bl=bl, ctl=ctl, sample_ids=sample_ids, groups_arr=groups_arr,
                 wavenumbers=wavenumbers, cancer_types=np.array(CANCER_TYPES))
    logger.info(f"Data: X={X.shape}, cancers/non = {(bl==1).sum()}/{(bl==0).sum()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mc_run = make_config(mc, X.shape[1])
    runtime = {
        "num_workers": 4 if device.type == "cuda" else 0,
        "pin_memory": device.type == "cuda",
        "prefetch_factor": 2 if device.type == "cuda" else None,
        "use_amp": device.type == "cuda",
    }
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    composite = bl * 100 + np.clip(ctl, 0, 99)

    fold_info = []
    for fi, (tr_idx, va_idx) in enumerate(sgkf.split(X, composite, groups=sample_ids)):
        logger.info(f"=== Fold {fi+1}/{N_SPLITS}  train={len(tr_idx)} val={len(va_idx)} ===")
        train_ds = SERSDataset(X[tr_idx], bl[tr_idx], ctl[tr_idx], augment=True)
        val_ds = SERSDataset(X[va_idx], bl[va_idx], ctl[va_idx], augment=False)
        model = SERSCancerDetector(mc_run).to(device)
        fr = train_fold(model, train_ds, val_ds, mc_run, device, fi, runtime)
        ck_path = CKPT_DIR / f"fold_{fi}.pt"
        torch.save({"fold": fi, "model_state_dict": fr["best_state"],
                    "val_idx": va_idx, "train_idx": tr_idx,
                    "epochs": fr["epochs"], "best_val_loss": fr["best_val_loss"]}, ck_path)
        val_m = fr["val_metrics"]
        auc_s1 = roc_auc_score(bl[va_idx], val_m["binary_prob"])
        logger.info(f"  Saved {ck_path}  AUC_S1={auc_s1:.4f}")
        fold_info.append({"fold": fi, "val_auc_s1": auc_s1, "epochs": fr["epochs"]})
    pd.DataFrame(fold_info).to_csv(CKPT_DIR / "fold_summary.csv", index=False, encoding="utf-8-sig")
    logger.info(f"All folds saved to {CKPT_DIR}")


def integrated_gradients(model: SERSCancerDetector, x: torch.Tensor, target_class: int,
                          baseline: torch.Tensor, steps: int = 50) -> torch.Tensor:
    """IG for cancer_logits[:, target_class]. x:(B,L) on device. Returns (B,L)."""
    model.eval()
    alphas = torch.linspace(0.0, 1.0, steps, device=x.device).view(-1, 1, 1)
    interp = baseline.unsqueeze(0) + alphas * (x - baseline).unsqueeze(0)  # (S,B,L)
    interp = interp.reshape(-1, x.shape[-1])
    interp.requires_grad_(True)
    out = model(interp)["cancer_logits"]  # (S*B, C)
    target = out[:, target_class].sum()
    grads = torch.autograd.grad(target, interp, retain_graph=False)[0]
    grads = grads.view(steps, x.shape[0], x.shape[-1]).mean(dim=0)  # (B,L)
    return (x - baseline) * grads


def run_ig():
    inp = np.load(CKPT_DIR / "fold_inputs.npz", allow_pickle=True)
    X = inp["X"]; bl = inp["bl"]; ctl = inp["ctl"]
    groups_arr = inp["groups_arr"]; wavenumbers = inp["wavenumbers"]
    n, L = X.shape

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Same model config as train_all_folds
    import yaml
    with open(PROJECT_ROOT / "config" / "config.yaml", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)
    mc = ModelConfig.from_pipeline_config(raw_cfg, n_spectral_features=L)
    mc = apply_class_selection(mc, cancer_types=CANCER_TYPES, non_cancer_groups=NON_CANCER)
    mc_run = make_config(mc, L)

    nor_mask = (groups_arr == "NOR")
    baseline_np = X[nor_mask].mean(axis=0) if nor_mask.sum() > 0 else X.mean(axis=0)
    baseline = torch.from_numpy(baseline_np.astype(np.float32)).to(device)
    logger.info(f"Baseline: mean of NOR samples (n={nor_mask.sum()})")

    attributions = np.zeros((n, len(CANCER_TYPES), L), dtype=np.float32)
    covered = np.zeros(n, dtype=bool)

    for fi in range(N_SPLITS):
        ck_path = CKPT_DIR / f"fold_{fi}.pt"
        if not ck_path.exists():
            logger.error(f"Missing checkpoint {ck_path}"); return
        ck = torch.load(ck_path, map_location=device, weights_only=False)
        model = SERSCancerDetector(mc_run).to(device)
        model.load_state_dict(ck["model_state_dict"]); model.eval()
        va_idx = ck["val_idx"]
        logger.info(f"=== Fold {fi+1}/{N_SPLITS}  IG on n_val={len(va_idx)} ===")

        BATCH = 32
        for ci, cname in enumerate(CANCER_TYPES):
            for b0 in range(0, len(va_idx), BATCH):
                idx_batch = va_idx[b0:b0+BATCH]
                xb = torch.from_numpy(X[idx_batch]).to(device)
                ig = integrated_gradients(model, xb, ci, baseline, steps=32)
                attributions[idx_batch, ci] = ig.detach().cpu().numpy()
        covered[va_idx] = True
        del model; torch.cuda.empty_cache()

    logger.info(f"Coverage: {covered.sum()}/{n} samples")
    np.savez_compressed(IG_DIR / "ig_full.npz",
                        attributions=attributions, covered=covered, X=X,
                        groups_arr=groups_arr, ctl=ctl, wavenumbers=wavenumbers,
                        cancer_types=np.array(CANCER_TYPES))

    # Per-cancer aggregation: average |IG| across samples of that cancer
    summary_rows = []
    fig, axes = plt.subplots(7, 1, figsize=(14, 22), sharex=True)
    for ax_idx, cancer in enumerate(TARGET_CANCERS):
        ci = CANCER_TYPES.index(cancer)
        sample_mask = (groups_arr == cancer) & covered
        if sample_mask.sum() == 0:
            logger.warning(f"  {cancer}: no covered samples"); continue
        attr = attributions[sample_mask, ci, :]  # (n_c, L)
        mean_abs = np.mean(np.abs(attr), axis=0)
        mean_signed = np.mean(attr, axis=0)
        mean_spec = X[sample_mask].mean(axis=0)
        nor_mean = X[nor_mask].mean(axis=0)

        # Peak detection on mean |IG|
        peaks, _ = find_peaks(mean_abs, distance=8, prominence=mean_abs.std() * 0.5)
        # top-15 by attribution magnitude
        top_idx = peaks[np.argsort(-mean_abs[peaks])][:15]
        for rank, pi in enumerate(top_idx, 1):
            summary_rows.append({"cancer": cancer, "rank": rank,
                                 "wavenumber": float(wavenumbers[pi]),
                                 "mean_abs_ig": float(mean_abs[pi]),
                                 "mean_signed_ig": float(mean_signed[pi]),
                                 "mean_intensity": float(mean_spec[pi]),
                                 "nor_intensity": float(nor_mean[pi])})

        ax = axes[ax_idx]
        ax2 = ax.twinx()
        ax.plot(wavenumbers, mean_spec, color="#444", lw=1.0, label=f"{cancer} mean spectrum")
        ax.plot(wavenumbers, nor_mean, color="#999", lw=0.8, alpha=0.6, label="NOR mean spectrum")
        ax2.fill_between(wavenumbers, 0, mean_abs, alpha=0.35, color="#d62728", label="|IG| mean")
        for pi in top_idx[:10]:
            ax2.axvline(wavenumbers[pi], color="#d62728", alpha=0.25, lw=0.8)
            ax.text(wavenumbers[pi], mean_spec[pi], f" {wavenumbers[pi]:.0f}",
                    fontsize=7, color="#d62728", rotation=90, va="bottom")
        ax.set_ylabel("Intensity (a.u.)")
        ax2.set_ylabel("|IG| attribution")
        ax.set_title(f"{cancer}  (n_samples={int(sample_mask.sum())}, top peaks shown)",
                     fontsize=11, fontweight="bold")
        ax.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Wavenumber (cm$^{-1}$)")
    fig.suptitle("Cancer-specific peaks via ResNet18 Integrated Gradients\n"
                 "(baseline: mean NOR spectrum, |IG| averaged across OOF val samples)",
                 fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    out_png = IG_DIR / "ig_per_cancer_overview.png"
    plt.savefig(out_png, dpi=200, bbox_inches="tight"); plt.close()

    df = pd.DataFrame(summary_rows)
    df.to_csv(IG_DIR / "ig_top_peaks_per_cancer.csv", index=False, encoding="utf-8-sig")
    logger.info(f"Saved {out_png}")
    logger.info(f"Saved {IG_DIR/'ig_top_peaks_per_cancer.csv'}  ({len(df)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["train", "ig", "all"], default="all")
    args = ap.parse_args()
    if args.stage in ("train", "all"):
        train_all_folds()
    if args.stage in ("ig", "all"):
        run_ig()


if __name__ == "__main__":
    main()
