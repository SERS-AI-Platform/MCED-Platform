#!/usr/bin/env python3
"""
Export prediction / confusion / ROC data from NPZ files into tidy CSVs
for downstream R visualization.

Output directory: results/r_export/predictions/
"""

import os
import sys
import json
import glob
import warnings

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# Suppress sklearn warnings when loading old models
warnings.filterwarnings("ignore", category=UserWarning)

# ── project root ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]          # SERS-AI/
os.chdir(ROOT)

OUT = ROOT / "results" / "r_export" / "predictions"
OUT.mkdir(parents=True, exist_ok=True)

# ── helpers ───────────────────────────────────────────────────────────

def safe_load_npz(path: str) -> dict | None:
    """Load NPZ with allow_pickle=True, return dict or None on failure."""
    try:
        d = np.load(path, allow_pickle=True)
        return dict(d)
    except Exception as e:
        print(f"  [WARN] Cannot load {path}: {e}")
        return None


def print_keys(data: dict, label: str):
    """Print key summary for an NPZ."""
    print(f"\n  Keys in {label}:")
    for k, v in data.items():
        shape = v.shape if hasattr(v, "shape") else "scalar"
        dtype = v.dtype if hasattr(v, "dtype") else type(v).__name__
        print(f"    {k}: shape={shape}, dtype={dtype}")


def softmax(x):
    """Row-wise softmax for logits -> probabilities."""
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# ── 1. Fold-prediction NPZ exports ───────────────────────────────────

FOLD_PRED_SOURCES = {
    "phase_F_baseline": "results/training/experiment_002/logistic_regression/v001/fold_predictions.npz",
    "phase_T_8class":   "results/training/R_BLC_added/logistic_regression/v001/fold_predictions.npz",
    "phase_U_7class":   "results/training/7c_add_BRE/logistic_regression/v001/fold_predictions.npz",
    "phase_X_clean":    "results/training/phase_x_clean_cohort/logistic_regression/v001/fold_predictions.npz",
}

# Try to find pancreatic NPZ
pan_npz = "results/training/pancreatic/experiment/fold_predictions.npz"
if os.path.exists(pan_npz):
    FOLD_PRED_SOURCES["pancreatic"] = pan_npz


def export_fold_predictions(label: str, path: str):
    """Extract ROC, confusion matrix, probability distribution, per-class
    metrics from a fold_predictions NPZ."""

    print(f"\n{'='*60}")
    print(f"Processing: {label}  ({path})")
    print(f"{'='*60}")

    data = safe_load_npz(path)
    if data is None:
        return
    print_keys(data, label)

    subdir = OUT / label
    subdir.mkdir(exist_ok=True)

    # --- common arrays ---
    binary_labels = data.get("binary_labels")
    val_binary_prob = data.get("val_binary_prob")
    cancer_type_labels = data.get("cancer_type_labels")
    val_cancer_logits = data.get("val_cancer_logits")
    groups = data.get("groups")
    sample_ids = data.get("sample_ids")
    fold_ids = data.get("fold_ids")

    n = len(binary_labels) if binary_labels is not None else 0
    print(f"  Samples (val): {n}")

    # --- 1a. ROC curve (Stage 1 binary) ---
    if binary_labels is not None and val_binary_prob is not None:
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, thresholds = roc_curve(binary_labels, val_binary_prob)
        roc_auc = auc(fpr, tpr)
        roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})
        roc_df.to_csv(subdir / "roc_stage1.csv", index=False)
        print(f"  -> roc_stage1.csv  (AUC={roc_auc:.4f}, {len(roc_df)} pts)")
    else:
        print("  [SKIP] No binary_labels / val_binary_prob for ROC")

    # --- 1b. Stage 1 probability distribution ---
    if binary_labels is not None and val_binary_prob is not None:
        prob_df = pd.DataFrame({
            "binary_label": binary_labels,
            "pred_prob_cancer": val_binary_prob,
        })
        if groups is not None:
            prob_df["group"] = groups
        if sample_ids is not None:
            prob_df["sample_id"] = sample_ids
        if fold_ids is not None:
            prob_df["fold"] = fold_ids
        prob_df.to_csv(subdir / "stage1_prob_distribution.csv", index=False)
        print(f"  -> stage1_prob_distribution.csv  ({len(prob_df)} rows)")

    # --- 1c. Binary confusion matrix ---
    if binary_labels is not None and val_binary_prob is not None:
        from sklearn.metrics import confusion_matrix
        pred_binary = (val_binary_prob >= 0.5).astype(int)
        cm = confusion_matrix(binary_labels, pred_binary)
        cm_df = pd.DataFrame(cm,
                             index=["true_non_cancer", "true_cancer"],
                             columns=["pred_non_cancer", "pred_cancer"])
        cm_df.to_csv(subdir / "confusion_stage1.csv")
        print(f"  -> confusion_stage1.csv")

    # --- 1d. Multiclass confusion matrix & per-class metrics ---
    if cancer_type_labels is not None and val_cancer_logits is not None:
        from sklearn.metrics import (confusion_matrix, f1_score,
                                     precision_score, recall_score)

        n_classes = val_cancer_logits.shape[1]
        # Only evaluate cancer samples (binary_label==1)
        if binary_labels is not None:
            mask = binary_labels == 1
        else:
            mask = np.ones(len(cancer_type_labels), dtype=bool)

        y_true_mc = cancer_type_labels[mask]
        logits_mc = val_cancer_logits[mask]
        y_pred_mc = logits_mc.argmax(axis=1)

        # Use group labels if available for the cancer subset
        if groups is not None:
            cancer_groups = groups[mask]
            unique_groups = sorted(set(cancer_groups))
        else:
            unique_groups = [str(i) for i in range(n_classes)]

        # Build label mapping from integer to group name
        label_map = {}
        if groups is not None:
            for g, lbl in zip(cancer_groups, y_true_mc):
                label_map[int(lbl)] = g
        class_names = [label_map.get(i, str(i)) for i in range(n_classes)]

        # Confusion matrix
        cm_mc = confusion_matrix(y_true_mc, y_pred_mc)
        cm_mc_df = pd.DataFrame(cm_mc, index=class_names, columns=class_names)
        cm_mc_df.index.name = "true"
        cm_mc_df.columns.name = "predicted"
        cm_mc_df.to_csv(subdir / "confusion_stage2.csv")
        print(f"  -> confusion_stage2.csv  ({n_classes} classes)")

        # Per-class metrics
        present_classes = sorted(set(y_true_mc))
        per_class = []
        for c in present_classes:
            tp = ((y_true_mc == c) & (y_pred_mc == c)).sum()
            fn = ((y_true_mc == c) & (y_pred_mc != c)).sum()
            fp = ((y_true_mc != c) & (y_pred_mc == c)).sum()
            tn = ((y_true_mc != c) & (y_pred_mc != c)).sum()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
            per_class.append({
                "class_id": int(c),
                "class_name": label_map.get(int(c), str(c)),
                "n": int(tp + fn),
                "sensitivity": round(sens, 4),
                "specificity": round(spec, 4),
                "f1": round(f1, 4),
            })
        per_class_df = pd.DataFrame(per_class)
        per_class_df.to_csv(subdir / "per_class_metrics.csv", index=False)
        print(f"  -> per_class_metrics.csv  ({len(per_class_df)} classes)")

        # Multiclass probability table (softmax of logits)
        proba_mc = softmax(logits_mc)
        proba_df = pd.DataFrame(proba_mc, columns=[f"prob_{c}" for c in class_names])
        proba_df["true_class"] = [label_map.get(int(t), str(t)) for t in y_true_mc]
        proba_df["pred_class"] = [label_map.get(int(p), str(p)) for p in y_pred_mc]
        if groups is not None:
            proba_df["group"] = cancer_groups
        if sample_ids is not None:
            proba_df["sample_id"] = sample_ids[mask]
        proba_df.to_csv(subdir / "stage2_probabilities.csv", index=False)
        print(f"  -> stage2_probabilities.csv  ({len(proba_df)} rows)")

        # Per-class ROC (one-vs-rest)
        from sklearn.metrics import roc_curve, auc
        roc_records = []
        for c in present_classes:
            y_bin = (y_true_mc == c).astype(int)
            y_score = proba_mc[:, c]
            fpr, tpr, thr = roc_curve(y_bin, y_score)
            auc_val = auc(fpr, tpr)
            for f, t, th in zip(fpr, tpr, thr):
                roc_records.append({
                    "class_id": int(c),
                    "class_name": label_map.get(int(c), str(c)),
                    "fpr": f, "tpr": t, "threshold": th,
                    "auc": round(auc_val, 4),
                })
        roc_mc_df = pd.DataFrame(roc_records)
        roc_mc_df.to_csv(subdir / "roc_stage2_per_class.csv", index=False)
        print(f"  -> roc_stage2_per_class.csv  ({len(roc_mc_df)} rows)")
    else:
        print("  [SKIP] No multiclass data")


# ── 2. Stacking OOF predictions ──────────────────────────────────────

def export_stacking():
    stk_files = [
        ("stacking_v2", "results/weekend_experiments/stacking_optimization/oof_predictions.npz"),
        ("stacking_v1_backup", "results/weekend_experiments/stacking_optimization/_backup_v1_cpan70/oof_predictions.npz"),
    ]
    for label, path in stk_files:
        if not os.path.exists(path):
            print(f"\n[SKIP] Stacking file not found: {path}")
            continue

        print(f"\n{'='*60}")
        print(f"Processing stacking: {label}  ({path})")
        print(f"{'='*60}")

        data = safe_load_npz(path)
        if data is None:
            continue
        print_keys(data, label)

        subdir = OUT / label
        subdir.mkdir(exist_ok=True)

        binary_labels = data.get("binary_labels")
        cancer_type_labels = data.get("cancer_type_labels")
        sample_ids = data.get("sample_ids")

        # Identify base model predictions (keys ending with _s1 or _s2)
        s1_keys = sorted([k for k in data if k.endswith("_s1")])
        s2_keys = sorted([k for k in data if k.endswith("_s2")])

        # Stage 1: collect all base model binary predictions
        if binary_labels is not None and s1_keys:
            from sklearn.metrics import roc_curve, auc
            s1_df = pd.DataFrame({"binary_label": binary_labels})
            if sample_ids is not None:
                s1_df["sample_id"] = sample_ids
            for k in s1_keys:
                model_name = k.replace("_s1", "")
                s1_df[f"prob_{model_name}"] = data[k]
            s1_df.to_csv(subdir / "stacking_stage1_base_probs.csv", index=False)
            print(f"  -> stacking_stage1_base_probs.csv  ({len(s1_df)} rows, {len(s1_keys)} models)")

            # ROC for each base model
            roc_records = []
            for k in s1_keys:
                model_name = k.replace("_s1", "")
                fpr, tpr, thr = roc_curve(binary_labels, data[k])
                auc_val = auc(fpr, tpr)
                for f, t, th in zip(fpr, tpr, thr):
                    roc_records.append({
                        "model": model_name, "fpr": f, "tpr": t,
                        "threshold": th, "auc": round(auc_val, 4),
                    })
            roc_df = pd.DataFrame(roc_records)
            roc_df.to_csv(subdir / "stacking_roc_base_models.csv", index=False)
            print(f"  -> stacking_roc_base_models.csv  ({len(roc_df)} rows)")

        # Stage 2: collect multiclass predictions per base model
        if cancer_type_labels is not None and s2_keys:
            # For each base model, save the argmax prediction
            s2_summary = pd.DataFrame({"cancer_type_label": cancer_type_labels})
            if sample_ids is not None:
                s2_summary["sample_id"] = sample_ids
            for k in s2_keys:
                model_name = k.replace("_s2", "")
                logits = data[k]
                s2_summary[f"pred_{model_name}"] = logits.argmax(axis=1)
            s2_summary.to_csv(subdir / "stacking_stage2_predictions.csv", index=False)
            print(f"  -> stacking_stage2_predictions.csv  ({len(s2_summary)} rows, {len(s2_keys)} models)")


# ── 3. Cross-instrument data ─────────────────────────────────────────

def export_cross_instrument():
    ci_report = "results/training/cross_instrument/cross_instrument_report.json"
    if not os.path.exists(ci_report):
        print(f"\n[SKIP] Cross-instrument report not found: {ci_report}")
        return

    print(f"\n{'='*60}")
    print(f"Processing cross-instrument report")
    print(f"{'='*60}")

    with open(ci_report) as f:
        report = json.load(f)

    subdir = OUT / "cross_instrument"
    subdir.mkdir(exist_ok=True)

    # Dump full report as CSV-friendly records
    # The structure varies — extract whatever is there
    print(f"  Report keys: {list(report.keys())}")

    # If nested dict, flatten
    records = []
    if isinstance(report, dict):
        for top_key, val in report.items():
            if isinstance(val, dict):
                row = {"section": top_key}
                row.update({k: v for k, v in val.items() if not isinstance(v, (dict, list))})
                records.append(row)
            elif isinstance(val, (int, float, str)):
                records.append({"section": "root", "key": top_key, "value": val})
    if records:
        pd.DataFrame(records).to_csv(subdir / "cross_instrument_report.csv", index=False)
        print(f"  -> cross_instrument_report.csv  ({len(records)} rows)")

    # Also copy existing r_export cross-instrument CSVs reference
    existing = list(Path("results/r_export/06_cross_instrument").glob("*.csv"))
    if existing:
        print(f"  Note: {len(existing)} existing CSVs already in results/r_export/06_cross_instrument/")


# ── 4. Experiment logs → all_phase_metrics.csv ────────────────────────

def export_experiment_logs():
    print(f"\n{'='*60}")
    print(f"Combining experiment_log.json files")
    print(f"{'='*60}")

    log_files = sorted(glob.glob("results/training/*/logistic_regression/v*/experiment_log.json"))
    # Also catch nested dirs like phase_v6_rerun/phase_v6_rerun/...
    log_files += sorted(glob.glob("results/training/*/*/logistic_regression/v*/experiment_log.json"))
    # Add non-LR logs too
    log_files += sorted(glob.glob("results/training/*/resnet18/v*/experiment_log.json"))
    log_files += sorted(glob.glob("results/training/*/xgboost/v*/experiment_log.json"))
    # Pancreatic
    log_files += sorted(glob.glob("results/training/pancreatic/experiment/experiment_log.json"))
    # Deduplicate
    log_files = sorted(set(log_files))

    print(f"  Found {len(log_files)} experiment_log.json files")

    rows = []
    for lf in log_files:
        try:
            with open(lf) as f:
                d = json.load(f)
        except Exception as e:
            print(f"  [WARN] Cannot read {lf}: {e}")
            continue

        # Flatten metrics into top-level columns
        row = {
            "log_path": lf,
            "experiment": d.get("experiment", ""),
            "model_name": d.get("model_name", d.get("model", {}).get("name", "")),
            "model_display_name": d.get("model_display_name", ""),
            "version": d.get("version", ""),
            "aggregate": d.get("aggregate", ""),
            "n_samples": d.get("n_samples", d.get("dataset", {}).get("n_samples", "")),
            "n_features": d.get("n_features", ""),
            "n_splits": d.get("n_splits", d.get("split", {}).get("n_splits", "")),
            "cancer_types": str(d.get("cancer_types", "")),
            "non_cancer_groups": str(d.get("non_cancer_groups", "")),
            "timestamp": d.get("timestamp", ""),
        }
        # Metrics may be under 'metrics' or 'results'
        metrics = d.get("metrics", d.get("results", {}))
        if isinstance(metrics, dict):
            for mk, mv in metrics.items():
                if isinstance(mv, (int, float)):
                    row[mk] = round(mv, 6) if isinstance(mv, float) else mv
                elif isinstance(mv, dict):
                    # Nested metrics (e.g., per-fold)
                    for mk2, mv2 in mv.items():
                        if isinstance(mv2, (int, float)):
                            row[f"{mk}__{mk2}"] = round(mv2, 6) if isinstance(mv2, float) else mv2

        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(OUT / "all_phase_metrics.csv", index=False)
        print(f"  -> all_phase_metrics.csv  ({len(df)} experiments, {len(df.columns)} columns)")
    else:
        print("  [WARN] No experiment logs parsed")


# ── 5. LR coefficients ───────────────────────────────────────────────

def export_lr_coefficients():
    print(f"\n{'='*60}")
    print(f"Extracting LR coefficients from saved models")
    print(f"{'='*60}")

    # Key experiment checkpoints
    COEFF_SOURCES = {
        "phase_F_baseline": "results/training/experiment_002/logistic_regression/v001/checkpoints",
        "phase_T_8class":   "results/training/R_BLC_added/logistic_regression/v001/checkpoints",
        "phase_U_7class":   "results/training/7c_add_BRE/logistic_regression/v001/checkpoints",
        "phase_X_clean":    "results/training/phase_x_clean_cohort/logistic_regression/v001/checkpoints",
    }

    subdir = OUT / "lr_coefficients"
    subdir.mkdir(exist_ok=True)

    for label, ckpt_dir in COEFF_SOURCES.items():
        if not os.path.exists(ckpt_dir):
            print(f"  [SKIP] {label}: {ckpt_dir} not found")
            continue

        binary_files = sorted(glob.glob(os.path.join(ckpt_dir, "fold_*_binary.joblib")))
        stage2_files = sorted(glob.glob(os.path.join(ckpt_dir, "fold_*_stage2.joblib")))

        # Binary coefficients (average across folds)
        if binary_files:
            coefs = []
            for bf in binary_files:
                try:
                    pipe = joblib.load(bf)
                    lr = pipe[-1]  # last step = LogisticRegression
                    coefs.append(lr.coef_.flatten())
                except Exception as e:
                    print(f"  [WARN] Cannot load {bf}: {e}")
            if coefs:
                coef_arr = np.array(coefs)
                mean_coef = coef_arr.mean(axis=0)
                std_coef = coef_arr.std(axis=0)
                coef_df = pd.DataFrame({
                    "feature_idx": range(len(mean_coef)),
                    "coef_mean": mean_coef,
                    "coef_std": std_coef,
                })
                # Add per-fold columns
                for i, c in enumerate(coefs):
                    coef_df[f"fold_{i}"] = c
                coef_df.to_csv(subdir / f"{label}_binary_coefs.csv", index=False)
                print(f"  -> {label}_binary_coefs.csv  ({len(coef_df)} features, {len(binary_files)} folds)")

        # Stage 2 coefficients
        if stage2_files:
            coefs_per_fold = []
            class_names = None
            for sf in stage2_files:
                try:
                    pipe = joblib.load(sf)
                    lr = pipe[-1]
                    coefs_per_fold.append(lr.coef_)  # shape (n_classes, n_features)
                    if class_names is None and hasattr(lr, "classes_"):
                        class_names = lr.classes_
                except Exception as e:
                    print(f"  [WARN] Cannot load {sf}: {e}")
            if coefs_per_fold:
                # Average across folds
                mean_coef = np.mean(coefs_per_fold, axis=0)  # (n_classes, n_features)
                n_classes, n_features = mean_coef.shape
                if class_names is None:
                    class_names = [str(i) for i in range(n_classes)]
                # Long format: class x feature
                rows = []
                for ci in range(n_classes):
                    for fi in range(n_features):
                        rows.append({
                            "class_id": int(class_names[ci]) if np.issubdtype(type(class_names[ci]), np.integer) else ci,
                            "class_name": str(class_names[ci]),
                            "feature_idx": fi,
                            "coef_mean": mean_coef[ci, fi],
                        })
                coef_df = pd.DataFrame(rows)
                coef_df.to_csv(subdir / f"{label}_stage2_coefs.csv", index=False)
                print(f"  -> {label}_stage2_coefs.csv  ({n_classes} classes x {n_features} features)")


# ── 6. Spectral interpretation data ──────────────────────────────────

def export_spectral_data():
    print(f"\n{'='*60}")
    print(f"Checking spectral interpretation data")
    print(f"{'='*60}")

    subdir = OUT / "spectral_interpretation"
    subdir.mkdir(exist_ok=True)

    # Check for existing CSVs in known locations
    csv_sources = [
        "results/r_export/11_interpretation/",
        "results/training/peak_analysis/",
        "results/training/peak_validation/phase4/",
    ]
    copied = 0
    for src_dir in csv_sources:
        if os.path.exists(src_dir):
            csvs = list(Path(src_dir).glob("*.csv"))
            for csv_file in csvs:
                # Symlink or note existence
                print(f"  Found: {csv_file}")
                copied += 1

    if copied == 0:
        print("  No spectral interpretation CSVs found")
    else:
        print(f"  {copied} existing CSVs noted (already in results/r_export/11_interpretation/ etc.)")

    # Check figures directories for any hidden data
    for fig_dir in [
        "results/figures/spectral_interpretation_7cancer/",
        "results/figures/training/phase_PI_peaks/",
    ]:
        if os.path.exists(fig_dir):
            all_files = os.listdir(fig_dir)
            csvs = [f for f in all_files if f.endswith(".csv")]
            if csvs:
                for c in csvs:
                    print(f"  Found CSV in {fig_dir}: {c}")
            else:
                print(f"  {fig_dir}: {len(all_files)} files (PNGs only, no CSVs)")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  SERS-AI: Export Predictions for R")
    print(f"  Output: {OUT}")
    print("=" * 60)

    # 1. Fold predictions
    for label, path in FOLD_PRED_SOURCES.items():
        if os.path.exists(path):
            export_fold_predictions(label, path)
        else:
            print(f"\n[SKIP] {label}: file not found ({path})")

    # 2. Stacking
    export_stacking()

    # 3. Cross-instrument
    export_cross_instrument()

    # 4. Experiment logs
    export_experiment_logs()

    # 5. LR coefficients
    export_lr_coefficients()

    # 6. Spectral interpretation
    export_spectral_data()

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  EXPORT COMPLETE")
    print(f"{'='*60}")
    total_csvs = list(OUT.rglob("*.csv"))
    print(f"  Total CSVs generated: {len(total_csvs)}")
    print(f"  Output directory: {OUT}")
    print()
    for csv_file in sorted(total_csvs):
        size_kb = csv_file.stat().st_size / 1024
        print(f"  {csv_file.relative_to(OUT)}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
