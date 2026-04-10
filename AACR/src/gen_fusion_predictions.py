"""
Generate uSERS-Net fold predictions: Ensemble (Fusion LR + ResNet18).

- LR: SERS(933) + age/sex/BMI(3) = 936 features (early fusion)
- ResNet18: SERS(933) only (1D CNN for spectral patterns)
- Blend: weighted average of both models' predictions (alpha search)

Uses the same fold_ids as main_5models LR benchmark for fair comparison.
Missing clinical data → imputed with global medians.
Output: AACR/data/early_fusion/fold_predictions.npz
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, roc_curve, confusion_matrix

SERS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AACR_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCHMARK_LR = os.path.join(SERS_ROOT, "models", "results", "01_benchmarks",
                             "main_5models", "logistic_regression", "v004",
                             "fold_predictions.npz")
OUT_DIR = os.path.join(AACR_DIR, "data", "early_fusion")
os.makedirs(OUT_DIR, exist_ok=True)

CANCER_TYPES = ["PRO", "LUN", "CRC", "CPAN", "OVA"]

# ── Import ResNet18 components ──
from models.model import ModelConfig, SERSDataset, SERSCancerDetector
from models.train import train_fold

import yaml
with open(os.path.join(SERS_ROOT, "config", "config.yaml"), encoding="utf-8") as f:
    raw_cfg = yaml.safe_load(f)

# ── Load benchmark base data ──
print("Loading benchmark data...")
bm = np.load(BENCHMARK_LR, allow_pickle=True)
X        = bm["X"]                   # (6200, 933)
bl       = bm["binary_labels"]       # (6200,)
ctl      = bm["cancer_type_labels"]  # (6200,)
groups   = bm["groups"]              # (6200,)
sids     = bm["sample_ids"]          # (6200,)
fold_ids = bm["fold_ids"]            # (6200,)
n = len(X)

# ── Build clinical lookup ──
print("Building clinical lookup...")

def load_clin(path, group_override=None):
    df = pd.read_csv(path)
    if group_override:
        df["disease_group"] = group_override
    df["sex_numeric"] = (df["sex"] == "M").astype(float)
    return df

pro  = load_clin(os.path.join(AACR_DIR, "data", "PRO_with_staging.csv"))
cpan = load_clin(os.path.join(AACR_DIR, "data", "PAN_with_ajcc_stage.csv"))

clin_std = pd.read_csv(os.path.join(SERS_ROOT, "data", "clinical_data", "standardized",
                                     "all_clinical_standardized.csv"))
clin_std["sex_numeric"] = (clin_std["sex"] == "M").astype(float)

lookup = {}
for df, grp_col in [(pro, "disease_group"), (cpan, "disease_group")]:
    for _, row in df.iterrows():
        pid = str(row["patient_id"]).strip()
        lookup[pid] = (row.get("age", np.nan),
                       row.get("sex_numeric", np.nan),
                       row.get("bmi", np.nan))

for _, row in clin_std.iterrows():
    pid = str(row["patient_id"]).strip()
    lookup[pid] = (row.get("age", np.nan),
                   row.get("sex_numeric", np.nan),
                   row.get("bmi", np.nan))

# OVA: positional match
ova_clin = pd.read_csv(os.path.join(SERS_ROOT, "data", "clinical_data", "standardized",
                                     "OVA_clinical_standardized.csv"))
ova_clin["sex_numeric"] = (ova_clin["sex"] == "M").astype(float)
for i, row in ova_clin.iterrows():
    lookup[f"OVA {i+1}"] = (row.get("age", np.nan),
                             row.get("sex_numeric", np.nan),
                             row.get("bmi", np.nan))

# ── Merge clinical into arrays ──
age_arr = np.full(n, np.nan)
sex_arr = np.full(n, np.nan)
bmi_arr = np.full(n, np.nan)

for i in range(n):
    key = f"{groups[i]} {sids[i]}"
    if key in lookup:
        age_arr[i], sex_arr[i], bmi_arr[i] = lookup[key]

bmi_median = np.nanmedian(bmi_arr)
age_median = np.nanmedian(age_arr)
sex_mode   = 0.0 if np.nansum(sex_arr == 0) >= np.nansum(sex_arr == 1) else 1.0

n_missing = np.isnan(age_arr).sum()
print(f"  Samples with clinical data: {n - n_missing}/{n}")
print(f"  Imputing: age_median={age_median:.1f}, sex_mode={sex_mode}, bmi_median={bmi_median:.1f}")

age_arr = np.where(np.isnan(age_arr), age_median, age_arr)
sex_arr = np.where(np.isnan(sex_arr), sex_mode,   sex_arr)
bmi_arr = np.where(np.isnan(bmi_arr), bmi_median, bmi_arr)

X_fusion = np.column_stack([X, age_arr, sex_arr, bmi_arr])  # (6200, 936)
print(f"X_fusion shape: {X_fusion.shape}")

# ── Device ──
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── ModelConfig for ResNet18 ──
from models.train import apply_class_selection
mc = ModelConfig.from_pipeline_config(raw_cfg, n_spectral_features=933)
mc = apply_class_selection(mc, cancer_types=["PRO", "LUN", "CRC", "PAN", "OVA"],
                            non_cancer_groups=["NOR", "DIA", "HBP", "H.D."])
mc = ModelConfig(**{**mc.__dict__, "n_epochs": 150})

# ── 5-fold CV ──
print("\n" + "=" * 64)
print("  uSERS-Net: Ensemble (Fusion LR + ResNet18)")
print("=" * 64)

n_folds = 5
val_bp_lr = np.full(n, np.nan)
val_cl_lr = np.full((n, len(CANCER_TYPES)), 1.0 / len(CANCER_TYPES))
val_bp_rn = np.full(n, np.nan)
val_cl_rn = np.full((n, len(CANCER_TYPES)), np.nan)

for fold_i in range(n_folds):
    val_idx = np.where(fold_ids == fold_i)[0]
    trn_idx = np.where(fold_ids != fold_i)[0]

    print(f"\n{'='*50}")
    print(f"  Fold {fold_i+1}/{n_folds} (train={len(trn_idx)}, val={len(val_idx)})")
    print(f"{'='*50}")

    # ── LR: Fusion (SERS + clinical) ──
    X_tr_fus, X_va_fus = X_fusion[trn_idx], X_fusion[val_idx]
    bl_tr, bl_va = bl[trn_idx], bl[val_idx]
    ctl_tr, ctl_va = ctl[trn_idx], ctl[val_idx]

    # Stage 1
    m1 = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=2000, solver="saga",
                           class_weight="balanced", random_state=42 + fold_i)
    )
    m1.fit(X_tr_fus, bl_tr)
    val_bp_lr[val_idx] = m1.predict_proba(X_va_fus)[:, 1]

    # Stage 2
    cancer_mask = ctl_tr >= 0
    if cancer_mask.sum() > 0:
        present = np.unique(ctl_tr[cancer_mask])
        if len(present) > 1:
            local_labels = np.searchsorted(present, ctl_tr[cancer_mask])
            m2 = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1.0, max_iter=2000, solver="saga",
                                   class_weight="balanced", random_state=142 + fold_i)
            )
            m2.fit(X_tr_fus[cancer_mask], local_labels)
            probs = m2.predict_proba(X_va_fus)
            for j, cls in enumerate(present):
                val_cl_lr[val_idx, cls] = probs[:, j]

    lr_auc = roc_auc_score(bl_va, val_bp_lr[val_idx])
    print(f"  [Fusion LR] Val S1 AUC: {lr_auc:.4f}")

    # ── ResNet18: SERS only ──
    X_tr_sers, X_va_sers = X[trn_idx], X[val_idx]

    train_ds = SERSDataset(X_tr_sers, bl_tr, ctl_tr, augment=True)
    val_ds   = SERSDataset(X_va_sers, bl_va, ctl_va, augment=False)

    runtime = {
        "num_workers": 4 if device.type == "cuda" else 0,
        "pin_memory": device.type == "cuda",
        "prefetch_factor": 2 if device.type == "cuda" else None,
        "use_amp": device.type == "cuda",
    }

    model = SERSCancerDetector(mc).to(device)
    fold_result = train_fold(model, train_ds, val_ds, mc, device, fold_i, runtime)

    val_m = fold_result["val_metrics"]
    val_bp_rn[val_idx] = val_m["binary_prob"]
    val_cl_rn[val_idx] = val_m["cancer_logits"]

    rn_auc = roc_auc_score(bl_va, val_m["binary_prob"])
    print(f"  [ResNet18]  Val S1 AUC: {rn_auc:.4f}")

# ── Blend weight search ──
print(f"\n{'='*64}")
print("  Blend Weight Search (alpha = Fusion LR weight)")
print(f"{'='*64}")

# Convert ResNet18 logits to probabilities
rn_probs = torch.softmax(torch.tensor(val_cl_rn), dim=-1).numpy()

alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
blend_results = []

for alpha in alphas:
    bp = alpha * val_bp_lr + (1 - alpha) * val_bp_rn
    cl = alpha * val_cl_lr + (1 - alpha) * rn_probs

    s1_auc = roc_auc_score(bl, bp)

    cancer_all = bl == 1
    pred_type = cl[cancer_all].argmax(axis=1)
    true_type = ctl[cancer_all]
    s2_f1 = f1_score(true_type, pred_type, average="macro", zero_division=0)

    try:
        s2_auc = roc_auc_score(true_type, cl[cancer_all], multi_class="ovr", average="macro")
    except ValueError:
        s2_auc = float("nan")

    blend_results.append({
        "alpha": alpha, "s1_auc": s1_auc, "s2_f1_macro": s2_f1, "s2_auc": s2_auc,
    })
    print(f"  alpha={alpha:.1f} | S1_AUC={s1_auc:.4f} | S2_F1={s2_f1:.4f} | S2_AUC={s2_auc:.4f}")

# Find best by S1 AUC (primary metric for screening)
best = max(blend_results, key=lambda r: r["s1_auc"])
best_alpha = best["alpha"]
print(f"\n  Best alpha: {best_alpha} (S1_AUC={best['s1_auc']:.4f}, S2_F1={best['s2_f1_macro']:.4f})")

# ── Final blended predictions ──
val_bp_final = best_alpha * val_bp_lr + (1 - best_alpha) * val_bp_rn
val_cl_final = best_alpha * val_cl_lr + (1 - best_alpha) * rn_probs

# ── Summary ──
fpr, tpr, thresholds = roc_curve(bl, val_bp_final)
thr = thresholds[np.argmax(tpr - fpr)]

s1_auc = roc_auc_score(bl, val_bp_final)
cancer_all = bl == 1
pred_type = val_cl_final[cancer_all].argmax(axis=1)
true_type = ctl[cancer_all]
s2_f1 = f1_score(true_type, pred_type, average="macro", zero_division=0)

print(f"\n{'='*64}")
print(f"  uSERS-Net Final (alpha={best_alpha})")
print(f"  S1 AUC:      {s1_auc:.4f}")
print(f"  S2 F1 macro: {s2_f1:.4f}")
print(f"  Threshold:   {thr:.4f}")

print("\n  Per-cancer S2 AUC:")
for ci, cname in enumerate(CANCER_TYPES):
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
         # Keep component predictions for analysis
         val_bp_lr=val_bp_lr, val_cl_lr=val_cl_lr,
         val_bp_rn=val_bp_rn, val_cl_rn=val_cl_rn,
         best_alpha=np.array([best_alpha]),
         blend_results=np.array(blend_results))

print(f"\nSaved: {out_path}")
print(f"{'='*64}")
