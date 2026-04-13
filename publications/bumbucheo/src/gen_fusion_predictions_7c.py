"""
Generate uSERS-Net fold predictions for 7-cancer: Ensemble (Fusion LR + ResNet18).

- LR: SERS(933) + age/sex/BMI(3) = 936 features (early fusion)
- ResNet18: SERS(933) only (1D CNN for spectral patterns)
- Blend: weighted average of both models' predictions (alpha search)
- Sex constraint: Males → OVA=0, Females → PRO=0, renormalize

Uses StratifiedGroupKFold (5-fold CV) on 7-cancer + 4 non-cancer data.
Cancer types: PRO, BRE, OVA, LUN, CRC, PAN, BLC
Output: AACR/data/early_fusion/fold_predictions_7cancer.npz
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, roc_curve

from models.model import ModelConfig, SERSDataset, SERSCancerDetector
from models.train import (
    load_processed_spectra, get_feature_columns,
    apply_class_selection, resolve_aliases, aggregate_replicates,
    create_labels, train_fold,
)

import yaml
import warnings
warnings.filterwarnings("ignore")

SERS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AACR_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR   = os.path.join(AACR_DIR, "data", "early_fusion")
os.makedirs(OUT_DIR, exist_ok=True)

CANCER_TYPES = ["PRO", "BRE", "OVA", "LUN", "CRC", "PAN", "BLC"]
NON_CANCER = ["NOR", "DIA", "HBP", "H.D."]

# ── Load config ──
with open(os.path.join(SERS_ROOT, "config", "config.yaml"), encoding="utf-8") as f:
    raw_cfg = yaml.safe_load(f)

# ── Load and prepare data ──
print("Loading processed spectra...")
df = load_processed_spectra()
feat_cols = get_feature_columns(df)
n_features = len(feat_cols)
print(f"  Features: {n_features}")

mc = ModelConfig.from_pipeline_config(raw_cfg, n_spectral_features=n_features)
mc = apply_class_selection(mc, cancer_types=CANCER_TYPES, non_cancer_groups=NON_CANCER)

df = resolve_aliases(df, mc)
df_agg = aggregate_replicates(df, feat_cols, "none")

# Filter to target groups only
df_agg = df_agg[df_agg["group"].isin(CANCER_TYPES + NON_CANCER)].reset_index(drop=True)

# ── Clinical data ──
print("Building clinical lookup...")
clin = pd.read_csv(os.path.join(SERS_ROOT, "data", "clinical_data", "standardized",
                                 "all_clinical_standardized.csv"))
clin["disease_group"] = clin["disease_group"].replace({"PAN": "CPAN"})
clin["sex_numeric"] = (clin["sex"] == "M").astype(float)

lookup = {}
for _, row in clin.iterrows():
    pid = str(row["patient_id"]).strip()
    lookup[pid] = (row.get("age", np.nan), row.get("sex_numeric", np.nan), row.get("bmi", np.nan))

# ── Create labels and arrays ──
X, bl, ctl, sample_ids, groups_arr = create_labels(df_agg, mc)
groups = df_agg["group"].values

n = len(X)
print(f"  Total: {n} spectra, {len(set(sample_ids))} subjects")
print(f"  Cancer: {(bl == 1).sum()}, Non-cancer: {(bl == 0).sum()}")
for ct in CANCER_TYPES:
    ct_count = (groups == ct).sum()
    print(f"    {ct}: {ct_count}")

# ── Merge clinical features ──
age_arr = np.full(n, np.nan)
sex_arr = np.full(n, np.nan)
bmi_arr = np.full(n, np.nan)

for i in range(n):
    key = f"{groups[i]} {sample_ids[i]}"
    if key in lookup:
        age_arr[i], sex_arr[i], bmi_arr[i] = lookup[key]
    else:
        # Try alias lookups
        aliases = {"PAN": ["CPAN", "YPAN"], "NOR": ["NOR", "YNOR"]}
        for orig in aliases.get(groups[i], []):
            key2 = f"{orig} {sample_ids[i]}"
            if key2 in lookup:
                age_arr[i], sex_arr[i], bmi_arr[i] = lookup[key2]
                break

# Impute missing
bmi_median = np.nanmedian(bmi_arr)
age_median = np.nanmedian(age_arr)
sex_mode = 0.0 if np.nansum(sex_arr == 0) >= np.nansum(sex_arr == 1) else 1.0

n_missing = np.isnan(age_arr).sum()
print(f"  With clinical data: {n - n_missing}/{n}")
print(f"  Imputing: age_median={age_median:.1f}, sex_mode={sex_mode}, bmi_median={bmi_median:.1f}")

age_arr = np.where(np.isnan(age_arr), age_median, age_arr)
sex_arr = np.where(np.isnan(sex_arr), sex_mode, sex_arr)
bmi_arr = np.where(np.isnan(bmi_arr), bmi_median, bmi_arr)

# Infer sex from cancer type for known sex-specific cancers
sex_for_constraint = sex_arr.copy()
for i, grp in enumerate(groups):
    if np.isnan(sex_for_constraint[i]) or sex_for_constraint[i] == sex_mode:
        if grp == "PRO":
            sex_for_constraint[i] = 1.0  # Male
        elif grp in ("OVA", "BRE"):
            sex_for_constraint[i] = 0.0  # Female

X_fusion = np.column_stack([X, age_arr, sex_arr, bmi_arr])
print(f"X_fusion shape: {X_fusion.shape}")

# ── 5-fold CV split ──
print("\nCreating 5-fold StratifiedGroupKFold split...")
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
fold_ids = np.full(n, -1, dtype=int)
for fold_i, (trn_idx, val_idx) in enumerate(cv.split(X, bl, sample_ids)):
    fold_ids[val_idx] = fold_i
    n_subj = len(set(sample_ids[val_idx]))
    print(f"  Fold {fold_i}: {len(val_idx)} spectra, {n_subj} subjects")

# ── Device ──
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── ModelConfig for ResNet18 ──
mc_rn = ModelConfig(**{**mc.__dict__,
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

# ── 5-fold CV ──
print("\n" + "=" * 64)
print("  uSERS-Net 7-Cancer: Ensemble (Fusion LR + ResNet18)")
print("=" * 64)

n_folds = 5
n_ct = len(CANCER_TYPES)
val_bp_lr = np.full(n, np.nan)
val_cl_lr = np.full((n, n_ct), 1.0 / n_ct)
val_bp_rn = np.full(n, np.nan)
val_cl_rn = np.full((n, n_ct), np.nan)

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
            m2 = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1.0, max_iter=2000, solver="saga",
                                   class_weight="balanced", random_state=142 + fold_i)
            )
            m2.fit(X_tr_fus[cancer_mask], ctl_tr[cancer_mask])
            probs = m2.predict_proba(X_va_fus)
            for j, cls in enumerate(m2[-1].classes_):
                val_cl_lr[val_idx, cls] = probs[:, j]

    lr_auc = roc_auc_score(bl_va, val_bp_lr[val_idx])
    print(f"  [Fusion LR] Val S1 AUC: {lr_auc:.4f}")

    # ── ResNet18: SERS only ──
    train_ds = SERSDataset(X[trn_idx], bl_tr, ctl_tr, augment=True)
    val_ds   = SERSDataset(X[val_idx], bl_va, ctl_va, augment=False)

    model = SERSCancerDetector(mc_rn).to(device)
    fold_result = train_fold(model, train_ds, val_ds, mc_rn, device, fold_i, runtime)

    val_m = fold_result["val_metrics"]
    val_bp_rn[val_idx] = val_m["binary_prob"]
    val_cl_rn[val_idx] = val_m["cancer_logits"]

    rn_auc = roc_auc_score(bl_va, val_m["binary_prob"])
    print(f"  [ResNet18]  Val S1 AUC: {rn_auc:.4f}")

# ── Apply sex constraint to LR Stage 2 predictions ──
print("\nApplying sex constraint to LR predictions...")
for i in range(n):
    sex = sex_for_constraint[i]
    if sex == 1.0:  # Male → OVA=0
        ova_idx = CANCER_TYPES.index("OVA")
        val_cl_lr[i, ova_idx] = 0.0
        row_sum = val_cl_lr[i].sum()
        if row_sum > 0:
            val_cl_lr[i] /= row_sum
    elif sex == 0.0:  # Female → PRO=0
        pro_idx = CANCER_TYPES.index("PRO")
        val_cl_lr[i, pro_idx] = 0.0
        row_sum = val_cl_lr[i].sum()
        if row_sum > 0:
            val_cl_lr[i] /= row_sum

# ── Blend weight search ──
print(f"\n{'='*64}")
print("  Blend Weight Search (alpha = Fusion LR weight)")
print(f"{'='*64}")

rn_probs = torch.softmax(torch.tensor(val_cl_rn), dim=-1).numpy()

# Apply sex constraint to ResNet probs too
for i in range(n):
    sex = sex_for_constraint[i]
    if sex == 1.0:
        ova_idx = CANCER_TYPES.index("OVA")
        rn_probs[i, ova_idx] = 0.0
        row_sum = rn_probs[i].sum()
        if row_sum > 0:
            rn_probs[i] /= row_sum
    elif sex == 0.0:
        pro_idx = CANCER_TYPES.index("PRO")
        rn_probs[i, pro_idx] = 0.0
        row_sum = rn_probs[i].sum()
        if row_sum > 0:
            rn_probs[i] /= row_sum

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

s1_sens = (val_bp_final[bl == 1] > thr).mean()
s1_spec = (val_bp_final[bl == 0] <= thr).mean()

print(f"\n{'='*64}")
print(f"  uSERS-Net 7-Cancer Final (alpha={best_alpha})")
print(f"  S1 AUC:         {s1_auc:.4f}")
print(f"  S1 Sensitivity: {s1_sens:.4f}")
print(f"  S1 Specificity: {s1_spec:.4f}")
print(f"  S2 F1 macro:    {s2_f1:.4f}")
print(f"  Threshold:      {thr:.4f}")

print("\n  Per-cancer S2 AUC:")
for ci, cname in enumerate(CANCER_TYPES):
    y_t = (true_type == ci).astype(int)
    y_s = val_cl_final[cancer_all][:, ci]
    try:
        auc = roc_auc_score(y_t, y_s)
        print(f"    {cname}: {auc:.4f}")
    except Exception:
        print(f"    {cname}: N/A")

print("\n  Per-cancer S1 Sensitivity (detection rate):")
for ct in CANCER_TYPES:
    mask = groups == ct
    if mask.sum() > 0:
        sens = (val_bp_final[mask] > thr).mean()
        print(f"    {ct}: {sens:.4f} (n={mask.sum()})")

# ── Save ──
out_path = os.path.join(OUT_DIR, "fold_predictions_7cancer.npz")
np.savez(out_path,
         X=X,
         binary_labels=bl,
         cancer_type_labels=ctl,
         groups=groups,
         sample_ids=sample_ids,
         fold_ids=fold_ids,
         val_binary_prob=val_bp_final,
         val_cancer_logits=val_cl_final,
         age=age_arr, sex=sex_arr, bmi=bmi_arr,
         sex_for_constraint=sex_for_constraint,
         val_bp_lr=val_bp_lr, val_cl_lr=val_cl_lr,
         val_bp_rn=val_bp_rn, val_cl_rn=val_cl_rn,
         best_alpha=np.array([best_alpha]),
         blend_results=np.array(blend_results),
         cancer_types=np.array(CANCER_TYPES))

print(f"\nSaved: {out_path}")
print(f"{'='*64}")
