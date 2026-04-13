"""
Generate final uSERS-Net fold predictions with optimized hyperparameters.

Best config from HP search:
  lr=3e-4, dropout=0.3, wd=1e-3, bs=32, channels=(64,128,256,512), alpha=0.8
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, roc_curve

from models.model import ModelConfig, SERSDataset, SERSCancerDetector
from models.train import train_fold, apply_class_selection

import yaml
import warnings
warnings.filterwarnings("ignore")

SERS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AACR_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR   = os.path.join(AACR_DIR, "data", "early_fusion")

CANCER_TYPES = ["PRO", "LUN", "CRC", "PAN", "OVA"]

# ── Load existing data (has Fusion LR predictions) ──
fus = np.load(os.path.join(OUT_DIR, "fold_predictions.npz"), allow_pickle=True)
X        = fus["X"]
bl       = fus["binary_labels"]
ctl      = fus["cancer_type_labels"]
groups   = fus["groups"]
sids     = fus["sample_ids"]
fold_ids = fus["fold_ids"]
val_bp_lr = fus["val_bp_lr"]
val_cl_lr = fus["val_cl_lr"]
age_arr  = fus["age"]
sex_arr  = fus["sex"]
bmi_arr  = fus["bmi"]

n = len(X)
n_folds = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Optimized ResNet18 config ──
with open(os.path.join(SERS_ROOT, "config", "config.yaml"), encoding="utf-8") as f:
    raw_cfg = yaml.safe_load(f)

mc = ModelConfig.from_pipeline_config(raw_cfg, n_spectral_features=933)
mc = apply_class_selection(mc, cancer_types=CANCER_TYPES,
                            non_cancer_groups=["NOR", "DIA", "HBP", "H.D."])
mc = ModelConfig(**{**mc.__dict__,
    "learning_rate": 3e-4,
    "dropout_rate": 0.3,
    "weight_decay": 1e-3,
    "batch_size": 32,
    "n_epochs": 150,
    "resnet_channels": (64, 128, 256, 512),
    "random_state": 42,
})

runtime = {
    "num_workers": 4 if device.type == "cuda" else 0,
    "pin_memory": device.type == "cuda",
    "prefetch_factor": 2 if device.type == "cuda" else None,
    "use_amp": device.type == "cuda",
}

BEST_ALPHA = 0.8

# ── Train ResNet18 with optimized config ──
print(f"\n{'='*64}")
print(f"  uSERS-Net Final: Optimized ResNet18 + Fusion LR (alpha={BEST_ALPHA})")
print(f"  lr={mc.learning_rate}, do={mc.dropout_rate}, wd={mc.weight_decay}")
print(f"  channels={mc.resnet_channels}")
print(f"{'='*64}")

val_bp_rn = np.full(n, np.nan)
val_cl_rn = np.full((n, len(CANCER_TYPES)), np.nan)

for fold_i in range(n_folds):
    val_idx = np.where(fold_ids == fold_i)[0]
    trn_idx = np.where(fold_ids != fold_i)[0]

    print(f"\n  Fold {fold_i+1}/{n_folds} (train={len(trn_idx)}, val={len(val_idx)})")

    train_ds = SERSDataset(X[trn_idx], bl[trn_idx], ctl[trn_idx], augment=True)
    val_ds   = SERSDataset(X[val_idx], bl[val_idx], ctl[val_idx], augment=False)

    model = SERSCancerDetector(mc).to(device)
    fold_result = train_fold(model, train_ds, val_ds, mc, device, fold_i, runtime)

    val_m = fold_result["val_metrics"]
    val_bp_rn[val_idx] = val_m["binary_prob"]
    val_cl_rn[val_idx] = val_m["cancer_logits"]

    rn_auc = roc_auc_score(bl[val_idx], val_m["binary_prob"])
    print(f"  ResNet18 fold AUC: {rn_auc:.4f}")

# ── Blend ──
rn_probs = torch.softmax(torch.tensor(val_cl_rn), dim=-1).numpy()
val_bp_final = BEST_ALPHA * val_bp_lr + (1 - BEST_ALPHA) * val_bp_rn
val_cl_final = BEST_ALPHA * val_cl_lr + (1 - BEST_ALPHA) * rn_probs

# ── Summary ──
fpr, tpr, thresholds = roc_curve(bl, val_bp_final)
thr = thresholds[np.argmax(tpr - fpr)]
s1_auc = roc_auc_score(bl, val_bp_final)

cancer_all = bl == 1
pred_type = val_cl_final[cancer_all].argmax(axis=1)
true_type = ctl[cancer_all]
s2_f1 = f1_score(true_type, pred_type, average="macro", zero_division=0)

rn_standalone = roc_auc_score(bl, val_bp_rn)

print(f"\n{'='*64}")
print(f"  uSERS-Net Final Results (alpha={BEST_ALPHA})")
print(f"  ResNet18 standalone AUC: {rn_standalone:.4f}")
print(f"  Ensemble S1 AUC:        {s1_auc:.4f}")
print(f"  Ensemble S2 F1 macro:   {s2_f1:.4f}")
print(f"  Threshold (Youden):     {thr:.4f}")

print("\n  Per-cancer S2 AUC:")
for ci, cname in enumerate(["PRO", "LUN", "CRC", "CPAN", "OVA"]):
    y_t = (true_type == ci).astype(int)
    y_s = val_cl_final[cancer_all][:, ci]
    try:
        auc = roc_auc_score(y_t, y_s)
        print(f"    {cname}: {auc:.4f}")
    except Exception:
        print(f"    {cname}: N/A")

# ── Save ──
out_path = os.path.join(OUT_DIR, "fold_predictions.npz")
np.savez(out_path,
         X=X,
         binary_labels=bl,
         cancer_type_labels=ctl,
         groups=groups,
         sample_ids=sids,
         fold_ids=fold_ids,
         val_binary_prob=val_bp_final,
         val_cancer_logits=val_cl_final,
         age=age_arr, sex=sex_arr, bmi=bmi_arr,
         val_bp_lr=val_bp_lr, val_cl_lr=val_cl_lr,
         val_bp_rn=val_bp_rn, val_cl_rn=val_cl_rn,
         best_alpha=np.array([BEST_ALPHA]))

print(f"\nSaved: {out_path}")
print(f"{'='*64}")
