#!/usr/bin/env python3
"""STK-V2 — ElasticNet meta SHAP over its 10 base-model inputs.

Saves meta_shap_values.csv (long format, ready for R/ggplot)."""
from pathlib import Path
import json, numpy as np, pandas as pd, joblib, shap
ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = ROOT / "results" / "training" / "stacking_v2"
LEGACY_OOF_DIR = (
    ROOT / "results" / "legacy" / "hidden_2026-05-19"
    / "top_level" / "weekend_experiments" / "stacking_optimization"
)
OOF_DIR = RUN_DIR if (RUN_DIR / "oof_predictions.npz").exists() else LEGACY_OOF_DIR
PROD_DIR = ROOT / "artifacts" / "usersnet" / "current"
OUT_DIR = RUN_DIR / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_NAMES = ["lr_raw","lr_d1","lr_d2","lr_concat","lr_peak",
              "xgb_raw","xgb_d1","rf_raw","rf_d1","ridge_concat"]

oof = np.load(OOF_DIR / "oof_predictions.npz", allow_pickle=True)
y_bin = oof["binary_labels"].astype(int)
meta_s1_X = np.column_stack([oof[f"{m}_s1"] for m in BASE_NAMES])  # (1628, 10)

# Use the deployed production meta (ElasticNet LR) — same one used in STK-V2 manifest.
meta = joblib.load(PROD_DIR / "meta_s1.joblib")
scaler = meta.named_steps["standardscaler"]
clf = meta.named_steps["logisticregression"]
X_std = scaler.transform(meta_s1_X)
print(f"meta predict AUC on OOF: {meta.score(meta_s1_X, y_bin):.4f}")
print(f"coef (10 base weights): {clf.coef_.ravel()}")

# LinearExplainer is exact for linear models on standardized features.
explainer = shap.LinearExplainer(clf, masker=shap.maskers.Independent(X_std))
shap_values = explainer(X_std)
print(f"shap shape: {shap_values.values.shape}")

# Long format for R
rows = []
for i in range(len(y_bin)):
    for j, name in enumerate(BASE_NAMES):
        rows.append({
            "sample_idx": i,
            "label": int(y_bin[i]),
            "base_model": name,
            "feature_value": float(meta_s1_X[i, j]),
            "shap_value": float(shap_values.values[i, j]),
        })
df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "meta_shap_values.csv", index=False)
print(f"wrote {OUT_DIR / 'meta_shap_values.csv'} ({len(df)} rows)")

# Per-base summary for the dashboard
summary = (df.groupby("base_model")
             .agg(mean_abs_shap=("shap_value", lambda v: float(np.mean(np.abs(v)))),
                  mean_shap=("shap_value", "mean"))
             .reset_index()
             .sort_values("mean_abs_shap", ascending=False))
summary.to_csv(OUT_DIR / "meta_shap_summary.csv", index=False)
print(summary)
