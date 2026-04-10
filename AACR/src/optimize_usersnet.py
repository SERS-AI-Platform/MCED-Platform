"""
uSERS-Net Hyperparameter Optimization

Optimizes ResNet18 hyperparameters within the ensemble context.
Fusion LR predictions are fixed (already computed).
Only ResNet18 is retrained per config, then blended with Fusion LR.
Final metric: ensemble S1 AUC (primary) + S2 F1 (secondary).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import json
import itertools
from datetime import datetime
from sklearn.metrics import roc_auc_score, f1_score
from models.model import ModelConfig, SERSDataset, SERSCancerDetector
from models.train import train_fold, apply_class_selection

import yaml
import warnings
warnings.filterwarnings("ignore")

SERS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AACR_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Load fixed data ──
fus = np.load(os.path.join(AACR_DIR, "data", "early_fusion", "fold_predictions.npz"),
              allow_pickle=True)
X        = fus["X"]                   # (6200, 933) SERS features
bl       = fus["binary_labels"]
ctl      = fus["cancer_type_labels"]
fold_ids = fus["fold_ids"]
val_bp_lr = fus["val_bp_lr"]          # Fixed Fusion LR predictions
val_cl_lr = fus["val_cl_lr"]

n = len(X)
n_folds = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(os.path.join(SERS_ROOT, "config", "config.yaml"), encoding="utf-8") as f:
    raw_cfg = yaml.safe_load(f)

CANCER_TYPES = ["PRO", "LUN", "CRC", "PAN", "OVA"]

# ── Hyperparameter search space ──
SEARCH_SPACE = {
    "learning_rate":   [3e-4, 5e-4, 1e-3],
    "dropout_rate":    [0.3, 0.5, 0.7],
    "weight_decay":    [1e-4, 1e-3, 1e-2],
    "batch_size":      [32, 64],
    "n_epochs":        [150],  # fixed, early stopping handles it
    "resnet_channels": [(32, 64, 128, 256), (64, 128, 256, 512)],
}

# Smart subset: not full grid, but targeted combinations
CONFIGS = [
    # Baseline (current default)
    {"learning_rate": 5e-4, "dropout_rate": 0.5, "weight_decay": 1e-3,
     "batch_size": 32, "resnet_channels": (32, 64, 128, 256)},
    # Lower LR
    {"learning_rate": 3e-4, "dropout_rate": 0.5, "weight_decay": 1e-3,
     "batch_size": 32, "resnet_channels": (32, 64, 128, 256)},
    # Higher LR
    {"learning_rate": 1e-3, "dropout_rate": 0.5, "weight_decay": 1e-3,
     "batch_size": 32, "resnet_channels": (32, 64, 128, 256)},
    # Less dropout
    {"learning_rate": 5e-4, "dropout_rate": 0.3, "weight_decay": 1e-3,
     "batch_size": 32, "resnet_channels": (32, 64, 128, 256)},
    # More dropout
    {"learning_rate": 5e-4, "dropout_rate": 0.7, "weight_decay": 1e-3,
     "batch_size": 32, "resnet_channels": (32, 64, 128, 256)},
    # Less regularization
    {"learning_rate": 5e-4, "dropout_rate": 0.5, "weight_decay": 1e-4,
     "batch_size": 32, "resnet_channels": (32, 64, 128, 256)},
    # More regularization
    {"learning_rate": 5e-4, "dropout_rate": 0.5, "weight_decay": 1e-2,
     "batch_size": 32, "resnet_channels": (32, 64, 128, 256)},
    # Bigger model
    {"learning_rate": 5e-4, "dropout_rate": 0.5, "weight_decay": 1e-3,
     "batch_size": 32, "resnet_channels": (64, 128, 256, 512)},
    # Bigger model + less dropout
    {"learning_rate": 5e-4, "dropout_rate": 0.3, "weight_decay": 1e-3,
     "batch_size": 32, "resnet_channels": (64, 128, 256, 512)},
    # Bigger batch
    {"learning_rate": 5e-4, "dropout_rate": 0.5, "weight_decay": 1e-3,
     "batch_size": 64, "resnet_channels": (32, 64, 128, 256)},
    # Best combo candidates
    {"learning_rate": 3e-4, "dropout_rate": 0.3, "weight_decay": 1e-3,
     "batch_size": 32, "resnet_channels": (64, 128, 256, 512)},
    {"learning_rate": 1e-3, "dropout_rate": 0.3, "weight_decay": 1e-3,
     "batch_size": 64, "resnet_channels": (32, 64, 128, 256)},
    {"learning_rate": 3e-4, "dropout_rate": 0.5, "weight_decay": 1e-4,
     "batch_size": 64, "resnet_channels": (64, 128, 256, 512)},
]

BLEND_ALPHAS = [0.5, 0.6, 0.7, 0.8, 0.9]


def evaluate_ensemble(val_bp_rn, val_cl_rn, alpha):
    """Evaluate ensemble at given alpha."""
    rn_probs = torch.softmax(torch.tensor(val_cl_rn), dim=-1).numpy()
    bp = alpha * val_bp_lr + (1 - alpha) * val_bp_rn
    cl = alpha * val_cl_lr + (1 - alpha) * rn_probs

    s1_auc = roc_auc_score(bl, bp)
    cancer_mask = bl == 1
    pred_type = cl[cancer_mask].argmax(axis=1)
    s2_f1 = f1_score(ctl[cancer_mask], pred_type, average="macro", zero_division=0)
    return s1_auc, s2_f1


def run_resnet_config(config_dict, config_idx):
    """Train ResNet18 with given config across all folds, return val predictions."""
    mc = ModelConfig.from_pipeline_config(raw_cfg, n_spectral_features=933)
    mc = apply_class_selection(mc, cancer_types=CANCER_TYPES,
                                non_cancer_groups=["NOR", "DIA", "HBP", "H.D."])

    # Override with search config
    overrides = {
        "learning_rate": config_dict["learning_rate"],
        "dropout_rate": config_dict["dropout_rate"],
        "weight_decay": config_dict["weight_decay"],
        "batch_size": config_dict["batch_size"],
        "n_epochs": 150,
        "resnet_channels": config_dict["resnet_channels"],
        "random_state": 42,
    }
    mc = ModelConfig(**{**mc.__dict__, **overrides})

    runtime = {
        "num_workers": 4 if device.type == "cuda" else 0,
        "pin_memory": device.type == "cuda",
        "prefetch_factor": 2 if device.type == "cuda" else None,
        "use_amp": device.type == "cuda",
    }

    val_bp = np.full(n, np.nan)
    val_cl = np.full((n, len(CANCER_TYPES)), np.nan)

    for fold_i in range(n_folds):
        val_idx = np.where(fold_ids == fold_i)[0]
        trn_idx = np.where(fold_ids != fold_i)[0]

        train_ds = SERSDataset(X[trn_idx], bl[trn_idx], ctl[trn_idx], augment=True)
        val_ds   = SERSDataset(X[val_idx], bl[val_idx], ctl[val_idx], augment=False)

        model = SERSCancerDetector(mc).to(device)
        fold_result = train_fold(model, train_ds, val_ds, mc, device, fold_i, runtime)

        val_m = fold_result["val_metrics"]
        val_bp[val_idx] = val_m["binary_prob"]
        val_cl[val_idx] = val_m["cancer_logits"]

    # ResNet standalone
    rn_auc = roc_auc_score(bl, val_bp)

    return val_bp, val_cl, rn_auc


# ── Main sweep ──
print("=" * 70)
print(f"  uSERS-Net Hyperparameter Optimization ({len(CONFIGS)} configs)")
print(f"  Device: {device}")
print("=" * 70)

results = []
t_start = datetime.now()

for ci, cfg in enumerate(CONFIGS):
    t0 = datetime.now()
    tag = (f"lr={cfg['learning_rate']:.0e} do={cfg['dropout_rate']} "
           f"wd={cfg['weight_decay']:.0e} bs={cfg['batch_size']} "
           f"ch={cfg['resnet_channels']}")
    print(f"\n[{ci+1}/{len(CONFIGS)}] {tag}")

    val_bp_rn, val_cl_rn, rn_auc = run_resnet_config(cfg, ci)

    # Find best alpha for this ResNet config
    best_ens_auc = 0
    best_alpha = 0.8
    best_s2_f1 = 0
    for alpha in BLEND_ALPHAS:
        s1_auc, s2_f1 = evaluate_ensemble(val_bp_rn, val_cl_rn, alpha)
        if s1_auc > best_ens_auc:
            best_ens_auc = s1_auc
            best_alpha = alpha
            best_s2_f1 = s2_f1

    elapsed = (datetime.now() - t0).total_seconds()
    result = {
        **cfg,
        "resnet_channels": list(cfg["resnet_channels"]),
        "rn_standalone_auc": round(rn_auc, 5),
        "best_alpha": best_alpha,
        "ensemble_s1_auc": round(best_ens_auc, 5),
        "ensemble_s2_f1": round(best_s2_f1, 5),
        "elapsed_sec": round(elapsed, 1),
    }
    results.append(result)

    print(f"  ResNet AUC={rn_auc:.4f} | Ensemble(α={best_alpha}): "
          f"S1={best_ens_auc:.4f} S2_F1={best_s2_f1:.4f} | {elapsed:.0f}s")

# ── Summary ──
print(f"\n{'='*70}")
print("  OPTIMIZATION RESULTS")
print(f"{'='*70}")
results_sorted = sorted(results, key=lambda r: -r["ensemble_s1_auc"])
for i, r in enumerate(results_sorted):
    marker = " ★" if i == 0 else ""
    print(f"  [{i+1}] S1={r['ensemble_s1_auc']:.4f} S2_F1={r['ensemble_s2_f1']:.4f} "
          f"α={r['best_alpha']} | lr={r['learning_rate']:.0e} do={r['dropout_rate']} "
          f"wd={r['weight_decay']:.0e} bs={r['batch_size']} "
          f"ch={r['resnet_channels']}{marker}")

best = results_sorted[0]
print(f"\n  BEST CONFIG:")
print(f"  Ensemble S1 AUC: {best['ensemble_s1_auc']:.4f}")
print(f"  Ensemble S2 F1:  {best['ensemble_s2_f1']:.4f}")
print(f"  Alpha:           {best['best_alpha']}")
print(f"  ResNet standalone: {best['rn_standalone_auc']:.4f}")
for k in ["learning_rate", "dropout_rate", "weight_decay", "batch_size", "resnet_channels"]:
    print(f"  {k}: {best[k]}")

total_elapsed = datetime.now() - t_start
print(f"\n  Total time: {total_elapsed}")

# Save results
out_dir = os.path.join(AACR_DIR, "data", "early_fusion")
with open(os.path.join(out_dir, "hp_search_results.json"), "w") as f:
    json.dump({"results": results_sorted,
               "best": best,
               "timestamp": datetime.now().isoformat()}, f, indent=2)
print(f"  Saved: {out_dir}/hp_search_results.json")
