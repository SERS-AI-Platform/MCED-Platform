#!/usr/bin/env python3
"""
Export all experiment phase data as tidy CSVs for R figure generation.

Reads results from SERS-AI experiment phases and writes tidy CSV files
into results/r_export/{category}/ for ~75 R-redrawn publication figures.

Usage:
    python scripts/visualization/r/export_all_for_r.py
"""

import json
import os
import sys
import glob
import shutil
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path("/home/user/SERS-AI")
RESULTS = ROOT / "results"
TRAINING = RESULTS / "training"
EXPORT = RESULTS / "r_export"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_json(path):
    """Load JSON file, return None if missing."""
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def safe_read_csv(path, **kwargs):
    """Read CSV, return None if missing."""
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_csv(p, **kwargs)


def write_csv(df, outpath, index=False):
    """Write dataframe to CSV and print confirmation."""
    ensure_dir(outpath.parent)
    df.to_csv(outpath, index=index)
    print(f"  -> {outpath.relative_to(ROOT)}  ({len(df)} rows)")


def report(found, missing, category):
    """Print summary for a category."""
    if found:
        print(f"  Found {len(found)}: {', '.join(found[:10])}"
              + (f" ... +{len(found)-10} more" if len(found) > 10 else ""))
    if missing:
        print(f"  Missing (skipped): {', '.join(missing)}")


# ---------------------------------------------------------------------------
# 01. Baseline benchmark (Phase F / experiment_002)
# ---------------------------------------------------------------------------

def export_01_baseline():
    cat = "01_baseline"
    out = EXPORT / cat
    print(f"\n{'='*60}\n[{cat}] Baseline model comparison")
    found, missing = [], []

    # benchmark_summary.csv
    df = safe_read_csv(TRAINING / "experiment_002/benchmark/benchmark_summary.csv")
    if df is not None:
        found.append("benchmark_summary")
        write_csv(df, out / "benchmark_summary.csv")
    else:
        missing.append("benchmark_summary")

    # benchmark_train_val_gaps.csv
    df = safe_read_csv(TRAINING / "experiment_002/benchmark/benchmark_train_val_gaps.csv")
    if df is not None:
        found.append("train_val_gaps")
        write_csv(df, out / "train_val_gaps.csv")
    else:
        missing.append("train_val_gaps")

    # Per-model fold metrics from experiment_002
    rows = []
    for model_dir in sorted((TRAINING / "experiment_002").iterdir()):
        if not model_dir.is_dir() or model_dir.name == "benchmark":
            continue
        fm = safe_read_csv(model_dir / "v001" / "fold_metrics.csv")
        if fm is not None:
            fm["model"] = model_dir.name
            fm["experiment"] = "experiment_002"
            rows.append(fm)
            found.append(f"fold/{model_dir.name}")
    if rows:
        write_csv(pd.concat(rows, ignore_index=True),
                  out / "fold_metrics_all_models.csv")

    # Per-model experiment_log.json -> summary
    log_rows = []
    for model_dir in sorted((TRAINING / "experiment_002").iterdir()):
        if not model_dir.is_dir() or model_dir.name == "benchmark":
            continue
        log = load_json(model_dir / "v001" / "experiment_log.json")
        if log is not None:
            row = {
                "model": model_dir.name,
                "n_samples": log.get("n_samples"),
                "n_features": log.get("n_features"),
                "aggregate": log.get("aggregate"),
                "cancer_types": ",".join(log.get("cancer_types", [])),
            }
            for k, v in log.items():
                if k.startswith(("val_", "train_")) and isinstance(v, (int, float)):
                    row[k] = v
            if "fold_summary" in log:
                for k, v in log["fold_summary"].items():
                    if isinstance(v, dict):
                        row[f"{k}_mean"] = v.get("mean")
                        row[f"{k}_std"] = v.get("std")
                    elif isinstance(v, (int, float)):
                        row[k] = v
            log_rows.append(row)
    if log_rows:
        write_csv(pd.DataFrame(log_rows), out / "experiment_log_summary.csv")

    # experiment_001 ResNet18 fold + epoch metrics
    fm001 = safe_read_csv(TRAINING / "experiment_001/resnet18/v001/fold_metrics.csv")
    if fm001 is not None:
        fm001["experiment"] = "experiment_001"
        fm001["model"] = "resnet18"
        found.append("exp001_resnet18_fold")
        write_csv(fm001, out / "experiment_001_resnet18_fold_metrics.csv")
    else:
        missing.append("exp001_resnet18_fold")

    ep001 = safe_read_csv(TRAINING / "experiment_001/resnet18/v001/epoch_metrics.csv")
    if ep001 is not None:
        found.append("exp001_resnet18_epoch")
        write_csv(ep001, out / "experiment_001_resnet18_epoch_metrics.csv")
    else:
        missing.append("exp001_resnet18_epoch")

    report(found, missing, cat)


# ---------------------------------------------------------------------------
# 02. Normalization comparison (Phase L)
# ---------------------------------------------------------------------------

def export_02_normalization():
    cat = "02_normalization"
    out = EXPORT / cat
    print(f"\n{'='*60}\n[{cat}] Normalization comparison")
    found, missing = [], []

    # Main comparison table
    df = safe_read_csv(TRAINING / "step4_normalization_comparison/comparison_table.csv")
    if df is not None:
        found.append("comparison_table")
        write_csv(df, out / "normalization_comparison.csv")
    else:
        missing.append("comparison_table")

    # Per-normalization benchmark summaries
    norm_bm_rows = []
    for norm in ["none", "l2", "minmax"]:
        bm = safe_read_csv(TRAINING / f"step4_{norm}/benchmark/benchmark_summary.csv")
        if bm is not None:
            bm["normalization"] = norm
            found.append(f"benchmark_{norm}")
            norm_bm_rows.append(bm)
        else:
            missing.append(f"benchmark_{norm}")
    # Also add SNV (experiment_002)
    bm_snv = safe_read_csv(TRAINING / "experiment_002/benchmark/benchmark_summary.csv")
    if bm_snv is not None:
        bm_snv["normalization"] = "snv"
        norm_bm_rows.append(bm_snv)
    if norm_bm_rows:
        write_csv(pd.concat(norm_bm_rows, ignore_index=True),
                  out / "normalization_benchmarks_combined.csv")

    # Per-normalization fold metrics
    norm_fold_rows = []
    for norm in ["none", "l2", "minmax"]:
        for model in ["logistic_regression", "resnet18"]:
            fm = safe_read_csv(TRAINING / f"step4_{norm}/{model}/v001/fold_metrics.csv")
            if fm is not None:
                fm["normalization"] = norm
                fm["model"] = model
                norm_fold_rows.append(fm)
    if norm_fold_rows:
        write_csv(pd.concat(norm_fold_rows, ignore_index=True),
                  out / "normalization_fold_metrics.csv")

    report(found, missing, cat)


# ---------------------------------------------------------------------------
# 03. Clinical fusion / Confounding (Phase M, N)
# ---------------------------------------------------------------------------

def export_03_clinical_fusion():
    cat = "03_clinical_fusion"
    out = EXPORT / cat
    print(f"\n{'='*60}\n[{cat}] Clinical fusion & confounding")
    found, missing = [], []

    # Confounding analysis
    d = load_json(TRAINING / "confounding_analysis/confounding_summary.json")
    if d is not None:
        rows = d.get("results", [])
        if rows:
            found.append("confounding")
            write_csv(pd.DataFrame(rows), out / "confounding_analysis.csv")
    else:
        missing.append("confounding")

    # Multimodal fusion
    d = load_json(TRAINING / "multimodal/multimodal_summary.json")
    if d is not None:
        found.append("multimodal")
        fc = d.get("full_cohort", {})
        # Main methods comparison
        rows = []
        for method in ["sers_only", "clinical_only", "early_fusion"]:
            if method in fc:
                row = {"method": method}
                row.update(fc[method])
                rows.append(row)
        if "late_fusion_best" in fc:
            row = {"method": "late_fusion_best"}
            row.update(fc["late_fusion_best"])
            rows.append(row)
        if rows:
            write_csv(pd.DataFrame(rows), out / "multimodal_fusion.csv")
        # Late fusion alpha sweep
        lfa = fc.get("late_fusion_all", [])
        if lfa:
            write_csv(pd.DataFrame(lfa), out / "late_fusion_alpha_sweep.csv")
        # Sex-stratified
        for subset in ["male_only", "female_only"]:
            if subset in d:
                sub_rows = []
                for method, vals in d[subset].items():
                    if isinstance(vals, dict):
                        row = {"method": method, "subset": subset}
                        row.update(vals)
                        sub_rows.append(row)
                if sub_rows:
                    write_csv(pd.DataFrame(sub_rows),
                              out / f"multimodal_{subset}.csv")
    else:
        missing.append("multimodal")

    # Clinical analysis per-cancer
    d = load_json(TRAINING / "clinical_analysis/clinical_analysis_summary.json")
    if d is not None:
        found.append("clinical_analysis")
        pcs = d.get("per_cancer_s1", {})
        if pcs:
            rows = [{"cancer": k, **v} for k, v in pcs.items()]
            write_csv(pd.DataFrame(rows), out / "per_cancer_s1_sensitivity.csv")
        # Overall
        ov = d.get("overall", {})
        if ov:
            write_csv(pd.DataFrame([ov]), out / "clinical_overall.csv")
    else:
        missing.append("clinical_analysis")

    # Feature study (covers clinical fusion angle too)
    df = safe_read_csv(RESULTS / "feature_study_20260403/all_ablation_results_20260403.csv")
    if df is not None:
        found.append("feature_study")
        write_csv(df, out / "feature_ablation_all.csv")

    for sub, fname in [("01_feature_representation", "ablation_results.csv"),
                       ("02_peak_fitting", "results.csv"),
                       ("03_multiview_attention", "results.csv"),
                       ("04_clinical_fusion", "results.csv")]:
        df = safe_read_csv(RESULTS / f"feature_study_20260403/{sub}/{fname}")
        if df is not None:
            tag = sub.split("_", 1)[1]
            write_csv(df, out / f"fs_{tag}.csv")

    report(found, missing, cat)


# ---------------------------------------------------------------------------
# 04. Held-out evaluation (Phase W, X, SERS-Net)
# ---------------------------------------------------------------------------

def _extract_heldout_summary(d, label):
    """Extract rows/per-group from train_val_test_summary JSON."""
    rows, pcs_rows = [], []
    for i, rep in enumerate(d.get("sers_results", [])):
        for split in ["train", "val", "test"]:
            if split not in rep:
                continue
            s = rep[split]
            row = {"repeat": i, "split": split}
            for k, v in s.items():
                if isinstance(v, (int, float, str)):
                    row[k] = v
            rows.append(row)
            for cancer, cv in s.get("per_cancer_sensitivity", {}).items():
                pcs_rows.append({"repeat": i, "split": split,
                                 "group": cancer, "metric_type": "sensitivity", **cv})
            for ctrl, cv in s.get("per_control_specificity", {}).items():
                pcs_rows.append({"repeat": i, "split": split,
                                 "group": ctrl, "metric_type": "specificity", **cv})
    return rows, pcs_rows


def export_04_heldout():
    cat = "04_heldout"
    out = EXPORT / cat
    print(f"\n{'='*60}\n[{cat}] Held-out evaluation")
    found, missing = [], []

    for exp_name, json_path in [
        ("6cancer", TRAINING / "train_val_test/train_val_test_summary.json"),
        ("7cancer", TRAINING / "train_val_test_7cancer/train_val_test_summary.json"),
        ("7cancer_clean", TRAINING / "train_val_test_7cancer_clean/train_val_test_summary.json"),
    ]:
        d = load_json(json_path)
        if d is not None:
            found.append(exp_name)
            rows, pcs = _extract_heldout_summary(d, exp_name)
            if rows:
                write_csv(pd.DataFrame(rows), out / f"heldout_{exp_name}_splits.csv")
            if pcs:
                write_csv(pd.DataFrame(pcs), out / f"heldout_{exp_name}_per_group.csv")
        else:
            missing.append(exp_name)

    # Phase X clean cohort benchmark
    bm = safe_read_csv(TRAINING / "phase_x_clean_cohort/benchmark/benchmark_summary.csv")
    if bm is not None:
        found.append("phase_x_benchmark")
        write_csv(bm, out / "phase_x_clean_benchmark.csv")
    else:
        missing.append("phase_x_benchmark")

    fm = safe_read_csv(TRAINING / "phase_x_clean_cohort/logistic_regression/v001/fold_metrics.csv")
    if fm is not None:
        found.append("phase_x_fold")
        write_csv(fm, out / "phase_x_clean_fold_metrics.csv")
    else:
        missing.append("phase_x_fold")

    # SERS-Net
    d = load_json(TRAINING / "sersnet/sersnet_summary.json")
    if d is not None:
        found.append("sersnet")
        st = d.get("summary_table", [])
        if st:
            rows = []
            for entry in st:
                row = {"model": entry.get("model")}
                for k, v in entry.items():
                    if k == "model":
                        continue
                    if isinstance(v, dict):
                        row[f"{k}_mean"] = v.get("mean")
                        row[f"{k}_std"] = v.get("std")
                    else:
                        row[k] = v
                rows.append(row)
            write_csv(pd.DataFrame(rows), out / "sersnet_summary.csv")
    else:
        missing.append("sersnet")

    report(found, missing, cat)


# ---------------------------------------------------------------------------
# 05. Single-cancer (BLC, PAN)
# ---------------------------------------------------------------------------

def export_05_single_cancer():
    cat = "05_single_cancer"
    out = EXPORT / cat
    print(f"\n{'='*60}\n[{cat}] Single-cancer experiments")
    found, missing = [], []

    # BLC MFDS model comparison
    blc_rows = []
    for model in ["baseline", "film", "xattn"]:
        d = load_json(TRAINING / f"blc_mfds/{model}/summary.json")
        if d is not None:
            found.append(f"blc_{model}")
            row = {"model": model}
            m = d.get("metrics", {})
            row.update({k: v for k, v in m.items() if isinstance(v, (int, float))})
            row["n_samples"] = d.get("n_samples")
            row["n_blc"] = d.get("n_blc")
            row["n_non_cancer"] = d.get("n_non_cancer")
            blc_rows.append(row)
        else:
            missing.append(f"blc_{model}")
    if blc_rows:
        write_csv(pd.DataFrame(blc_rows), out / "blc_model_comparison.csv")

    # Pancreatic experiment
    d = load_json(TRAINING / "pancreatic/experiment/experiment_log.json")
    if d is not None:
        found.append("pancreatic")
        ds = d.get("dataset", {})
        dataset_row = {
            "total_patients": ds.get("total_patients"),
            "total_spectra": ds.get("total_spectra"),
            "pan_patients": ds.get("PAN", {}).get("patients"),
            "nor_patients": ds.get("NOR", {}).get("patients"),
        }
        write_csv(pd.DataFrame([dataset_row]), out / "pancreatic_dataset.csv")

        # Model results
        for key in ["models", "results"]:
            data = d.get(key, {})
            if isinstance(data, dict) and data:
                model_rows = []
                for mname, mdata in data.items():
                    mr = {"model": mname}
                    if isinstance(mdata, dict):
                        for k, v in mdata.items():
                            if isinstance(v, (int, float)):
                                mr[k] = v
                    model_rows.append(mr)
                if model_rows:
                    write_csv(pd.DataFrame(model_rows),
                              out / f"pancreatic_{key}.csv")
                    break
    else:
        missing.append("pancreatic")

    report(found, missing, cat)


# ---------------------------------------------------------------------------
# 06. Cross-instrument (Phase XI)
# ---------------------------------------------------------------------------

def export_06_cross_instrument():
    cat = "06_cross_instrument"
    out = EXPORT / cat
    print(f"\n{'='*60}\n[{cat}] Cross-instrument generalization")
    found, missing = [], []

    ci = RESULTS / "cross_instrument/calibration"

    for fname in ["leaderboard.csv", "paired_held_metrics.csv"]:
        df = safe_read_csv(ci / fname)
        if df is not None:
            found.append(fname)
            write_csv(df, out / fname)
        else:
            missing.append(fname)

    # Sweep results
    for fname in ["pds_sweep.csv", "osc_sweep.csv", "method_comparison_summary.csv",
                   "method_comparison_multiseed.csv", "per_group_best.csv"]:
        df = safe_read_csv(ci / f"sweep/{fname}")
        if df is not None:
            found.append(fname)
            write_csv(df, out / fname)
        else:
            missing.append(fname)

    # Stacking validation
    df = safe_read_csv(ci / "stacking_validation/summary.csv")
    if df is not None:
        found.append("stacking_validation")
        write_csv(df, out / "stacking_validation_summary.csv")
    else:
        missing.append("stacking_validation")

    # Paired diagnosis stats
    df = safe_read_csv(ci / "diagnosis/paired_stats.csv")
    if df is not None:
        found.append("paired_diagnosis")
        write_csv(df, out / "paired_diagnosis_stats.csv")
    else:
        missing.append("paired_diagnosis")

    # Calibration report summary
    d = load_json(ci / "calibration_report.json")
    if d is not None:
        found.append("calibration_report")
        row = {
            "n_paired_samples": d.get("n_paired_samples"),
            "raw_paired_rmse": d.get("raw_paired_rmse"),
            "raw_paired_correlation": d.get("raw_paired_correlation"),
        }
        write_csv(pd.DataFrame([row]), out / "calibration_summary.csv")
    else:
        missing.append("calibration_report")

    report(found, missing, cat)


# ---------------------------------------------------------------------------
# 07. Multi-view / Feature representation (Phase MV, FR)
# ---------------------------------------------------------------------------

def export_07_multiview():
    cat = "07_multiview"
    out = EXPORT / cat
    print(f"\n{'='*60}\n[{cat}] Multi-view & feature representation")
    found, missing = [], []

    # Feature study ablation
    df = safe_read_csv(RESULTS / "feature_study_20260403/all_ablation_results_20260403.csv")
    if df is not None:
        found.append("feature_ablation_all")
        write_csv(df, out / "feature_ablation_all.csv")

    for sub, fname in [("01_feature_representation", "ablation_results.csv"),
                       ("02_peak_fitting", "results.csv"),
                       ("03_multiview_attention", "results.csv"),
                       ("04_clinical_fusion", "results.csv")]:
        df = safe_read_csv(RESULTS / f"feature_study_20260403/{sub}/{fname}")
        if df is not None:
            tag = sub.split("_", 1)[1]
            found.append(tag)
            write_csv(df, out / f"fs_{tag}.csv")

    # Channel statistics
    df = safe_read_csv(TRAINING / "multiview_derivative/channel_statistics.csv")
    if df is not None:
        found.append("channel_stats")
        write_csv(df, out / "multiview_derivative_channel_stats.csv")

    # Multiview ablation reports (JSON -> CSV)
    for fname in glob.glob(str(TRAINING / "multiview_ablation/ablation_report_*.json")):
        d = load_json(fname)
        if d is not None:
            results = d.get("results", [])
            if results:
                tag = Path(fname).stem.replace("ablation_report_", "")
                found.append(f"ablation_{tag}")
                write_csv(pd.DataFrame(results),
                          out / f"multiview_ablation_{tag}.csv")

    # Multiview ensemble
    d = load_json(TRAINING / "multiview_ensemble/ensemble_report_thermo.json")
    if d is not None:
        results = d.get("results", [])
        if results:
            found.append("multiview_ensemble")
            write_csv(pd.DataFrame(results), out / "multiview_ensemble.csv")

    # Multichannel v1 fold metrics
    mc_rows = []
    for model in ["film_resnet18", "xattn_resnet18"]:
        fm = safe_read_csv(TRAINING / f"multichannel_v1/{model}/v001/fold_metrics.csv")
        if fm is not None:
            fm["model"] = model
            mc_rows.append(fm)
            found.append(f"mc1_{model}")
    if mc_rows:
        write_csv(pd.concat(mc_rows, ignore_index=True),
                  out / "multichannel_v1_fold_metrics.csv")

    report(found, missing, cat)


# ---------------------------------------------------------------------------
# 08. Deep learning models
# ---------------------------------------------------------------------------

def export_08_dl_models():
    cat = "08_dl_models"
    out = EXPORT / cat
    print(f"\n{'='*60}\n[{cat}] Deep learning models")
    found, missing = [], []

    # FiLM / xAttn fold metrics
    dl_fold_rows = []
    for exp_dir in ["film_xattn_comparison", "film_xattn_v2"]:
        for model in ["film_resnet18", "xattn_resnet18"]:
            p = TRAINING / exp_dir / model
            if not p.is_dir():
                continue
            for vf in sorted(p.glob("v*/fold_metrics.csv")):
                fm = safe_read_csv(vf)
                if fm is not None:
                    fm["experiment"] = exp_dir
                    fm["model"] = model
                    fm["version"] = vf.parent.name
                    dl_fold_rows.append(fm)
                    found.append(f"{exp_dir}/{model}/{vf.parent.name}")
    if dl_fold_rows:
        write_csv(pd.concat(dl_fold_rows, ignore_index=True),
                  out / "film_xattn_fold_metrics.csv")

    # ResNet18 training curves
    df = safe_read_csv(TRAINING / "experiment_001/resnet18/v001/epoch_metrics.csv")
    if df is not None:
        found.append("resnet18_training_curves")
        write_csv(df, out / "resnet18_training_curves.csv")
    else:
        missing.append("resnet18_training_curves")

    # All epoch_metrics for any DL model
    for em in sorted(TRAINING.rglob("epoch_metrics.csv")):
        if em == TRAINING / "experiment_001/resnet18/v001/epoch_metrics.csv":
            continue  # already exported
        df = safe_read_csv(em)
        if df is not None and len(df) > 5:
            rel = em.relative_to(TRAINING)
            parts = list(rel.parts)
            tag = "_".join(parts[:-1])
            write_csv(df, out / f"epoch_metrics_{tag}.csv")
            found.append(f"epoch/{tag}")

    # Transformer
    d = load_json(TRAINING / "transformer/transformer_report_thermo.json")
    if d is not None:
        found.append("transformer")
        fm = d.get("fold_metrics", [])
        if fm:
            write_csv(pd.DataFrame(fm), out / "transformer_fold_metrics.csv")
        row = {k: v for k, v in d.items() if isinstance(v, (int, float, str))}
        if row:
            write_csv(pd.DataFrame([row]), out / "transformer_summary.csv")
    else:
        missing.append("transformer")

    # Contrastive
    d = load_json(TRAINING / "contrastive/contrastive_report_thermo.json")
    if d is not None:
        found.append("contrastive")
        results = d.get("results", {})
        if results:
            rows = [{"method": k, **v} for k, v in results.items()
                    if isinstance(v, dict)]
            if rows:
                write_csv(pd.DataFrame(rows), out / "contrastive_methods.csv")
    else:
        missing.append("contrastive")

    # Contrastive sweep
    df = safe_read_csv(RESULTS / "contrastive_sweep/sweep_results.csv")
    if df is not None:
        found.append("contrastive_sweep")
        write_csv(df, out / "contrastive_sweep.csv")
    else:
        missing.append("contrastive_sweep")

    report(found, missing, cat)


# ---------------------------------------------------------------------------
# 09. Stacking ensemble
# ---------------------------------------------------------------------------

def export_09_stacking():
    cat = "09_stacking"
    out = EXPORT / cat
    print(f"\n{'='*60}\n[{cat}] Stacking ensemble")
    found, missing = [], []

    # stacking_optimization_v2/r_data
    r_data = TRAINING / "stacking_optimization_v2/r_data"
    if r_data.is_dir():
        found.append("stk_v2_r_data")
        for f in sorted(r_data.glob("*.csv")):
            df = safe_read_csv(f)
            if df is not None:
                write_csv(df, out / f"stk_v2_{f.name}")
    else:
        missing.append("stk_v2_r_data")

    # Meta SHAP
    for fname in ["meta_shap_summary.csv", "meta_shap_values.csv"]:
        df = safe_read_csv(TRAINING / f"stacking_optimization_v2/{fname}")
        if df is not None:
            found.append(fname)
            write_csv(df, out / f"stk_v2_{fname}")

    # Stacking ensemble v1 reports
    for instrument in ["thermo", "medical"]:
        d = load_json(TRAINING / f"stacking_ensemble/stacking_report_{instrument}.json")
        if d is not None:
            found.append(f"stk_v1_{instrument}")
            results = d.get("results", [])
            if results:
                write_csv(pd.DataFrame(results),
                          out / f"stacking_v1_{instrument}.csv")
            base = d.get("base_models", [])
            if base:
                write_csv(pd.DataFrame(base),
                          out / f"stacking_v1_{instrument}_base.csv")

    # Stacking v2 SHAP spectral
    shap_dir = TRAINING / "stacking_v2/val_group_span/shap"
    if shap_dir.is_dir():
        for f in sorted(shap_dir.glob("*.csv")):
            df = safe_read_csv(f)
            if df is not None:
                found.append(f"shap/{f.name}")
                write_csv(df, out / f"stk_v2_shap_{f.name}")

    # Weekend stacking
    wk_stk = RESULTS / "weekend_experiments/stacking_optimization"
    for fname in ["nested_cv_results.csv", "single_vs_ensemble.csv",
                   "base_model_contribution.csv"]:
        df = safe_read_csv(wk_stk / fname)
        if df is not None:
            found.append(f"wk/{fname}")
            write_csv(df, out / f"weekend_stacking_{fname}")

    report(found, missing, cat)


# ---------------------------------------------------------------------------
# 10. Optimization (Z-opt, phases)
# ---------------------------------------------------------------------------

def export_10_optimization():
    cat = "10_optimization"
    out = EXPORT / cat
    print(f"\n{'='*60}\n[{cat}] Optimization phases")
    found, missing = [], []

    opt = TRAINING / "optimize_7cancer_full"
    for fname in ["phase1_wavenumber_ablation.csv", "phase2_aggregation_ablation.csv",
                   "phase3_architecture_hp_search.csv", "phase4_held_out_repeats.csv"]:
        df = safe_read_csv(opt / fname)
        if df is not None:
            found.append(fname)
            write_csv(df, out / fname)
        else:
            missing.append(fname)

    # Phase 4 final evaluation
    d = load_json(opt / "phase4_final_evaluation.json")
    if d is not None:
        found.append("phase4_eval")
        rows = []
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, dict):
                    row = {"key": k}
                    row.update({kk: vv for kk, vv in v.items()
                                if isinstance(vv, (int, float, str))})
                    rows.append(row)
                elif isinstance(v, (int, float, str)):
                    rows.append({"key": k, "value": v})
        if rows:
            write_csv(pd.DataFrame(rows), out / "phase4_final_evaluation.csv")

    # Phase reruns fold metrics
    for phase_dir in ["phase_v6_rerun/phase_v6_rerun",
                       "phase_v7_rerun/phase_v7_rerun",
                       "phase_u_rerun/phase_u_rerun"]:
        lr_dir = TRAINING / phase_dir / "logistic_regression"
        if not lr_dir.is_dir():
            continue
        for vdir in sorted(lr_dir.glob("v*")):
            fm = safe_read_csv(vdir / "fold_metrics.csv")
            if fm is not None:
                phase_name = phase_dir.split("/")[0]
                fm["phase"] = phase_name
                fm["version"] = vdir.name
                found.append(f"{phase_name}/{vdir.name}")
                write_csv(fm, out / f"{phase_name}_{vdir.name}_fold_metrics.csv")

    # Other phase fold metrics
    for exp_name in ["fixed_grid_6cancer", "fixed_grid_7cancer", "fixed_grid_retrain",
                      "7c_add_BRE", "R_BLC_added"]:
        fm = safe_read_csv(TRAINING / f"{exp_name}/logistic_regression/v001/fold_metrics.csv")
        if fm is not None:
            fm["experiment"] = exp_name
            found.append(exp_name)
            write_csv(fm, out / f"{exp_name}_fold_metrics.csv")

    # Hypothesis experiments (step2)
    for hyp in ["step2_hypothesis_A", "step2_hypothesis_B", "step2_hypothesis_C"]:
        fm = safe_read_csv(TRAINING / f"{hyp}/resnet18/v001/fold_metrics.csv")
        if fm is not None:
            fm["experiment"] = hyp
            found.append(hyp)
            write_csv(fm, out / f"{hyp}_fold_metrics.csv")

    report(found, missing, cat)


# ---------------------------------------------------------------------------
# 11. Interpretation (PI, peaks, coefficients)
# ---------------------------------------------------------------------------

def export_11_interpretation():
    cat = "11_interpretation"
    out = EXPORT / cat
    print(f"\n{'='*60}\n[{cat}] Model interpretation")
    found, missing = [], []

    # Permutation importance
    d = load_json(TRAINING / "permutation_importance/experiment_summary.json")
    if d is not None:
        found.append("pi_summary")
        models = d.get("models", [])
        if models:
            write_csv(pd.DataFrame(models), out / "pi_model_summary.csv")

    # Weekend PI CSVs
    pi_dir = RESULTS / "weekend_experiments/exp4_permutation_importance_20260403_204449"
    for fname in ["wavenumber_importance.csv", "importance_ranking.csv",
                   "importance_significant.csv"]:
        df = safe_read_csv(pi_dir / fname)
        if df is not None:
            found.append(f"pi_{fname}")
            write_csv(df, out / f"pi_{fname}")
        else:
            missing.append(f"pi_{fname}")

    # null distribution (large, but needed for figures)
    df = safe_read_csv(pi_dir / "null_distribution.csv")
    if df is not None:
        found.append("pi_null_distribution")
        write_csv(df, out / "pi_null_distribution.csv")

    # Peak analysis
    for fname in ["coefficient_peak_crossref.csv", "peak_discrimination_matrix.csv"]:
        df = safe_read_csv(TRAINING / f"peak_analysis/{fname}")
        if df is not None:
            found.append(fname)
            write_csv(df, out / fname)
        else:
            missing.append(fname)

    # Peak validation phases
    for phase in ["phase1", "phase2", "phase3", "phase4", "phase5"]:
        pdir = TRAINING / f"peak_validation/{phase}"
        if not pdir.is_dir():
            continue
        for f in sorted(pdir.glob("*.csv")):
            df = safe_read_csv(f)
            if df is not None:
                found.append(f"pv_{phase}/{f.name}")
                write_csv(df, out / f"peak_validation_{phase}_{f.name}")

    # Stacking peak interpretation
    d = load_json(TRAINING / "stacking_optimization_v2/peak_interpretation.json")
    if d is not None:
        found.append("stacking_peak_interp")
        rows = []
        for k, v in d.items():
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        item["category"] = k
                        rows.append(item)
        if rows:
            write_csv(pd.DataFrame(rows), out / "stacking_peak_interpretation.csv")

    # Metabolite correlations
    met_dir = RESULTS / "figures/peak_marker_direct_correlation"
    if met_dir.is_dir():
        for f in sorted(met_dir.glob("*.csv")):
            df = safe_read_csv(f)
            if df is not None:
                found.append(f"met/{f.name}")
                write_csv(df, out / f"metabolite_{f.name}")

    # NMF correlations
    df = safe_read_csv(RESULTS / "figures/nmf_tumor_marker_correlation/nmf_marker_correlations.csv")
    if df is not None:
        found.append("nmf_correlations")
        write_csv(df, out / "nmf_marker_correlations.csv")

    report(found, missing, cat)


# ---------------------------------------------------------------------------
# 12. Weekend advanced experiments
# ---------------------------------------------------------------------------

def export_12_weekend():
    cat = "12_weekend"
    out = EXPORT / cat
    print(f"\n{'='*60}\n[{cat}] Weekend advanced experiments")
    found, missing = [], []

    # LOHO validation
    loho_dir = RESULTS / "weekend_experiments/loho_validation/checkpoints"
    if loho_dir.is_dir():
        loho_rows = []
        for f in sorted(loho_dir.glob("loho_*.json")):
            d = load_json(f)
            if d and d.get("complete") and d.get("result"):
                r = d["result"]
                hospital = f.stem.replace("loho_", "").replace("_lr", "")
                loho_rows.append({
                    "hospital": hospital,
                    "s1_auc": r.get("s1_auc"),
                    "s1_sens": r.get("s1_sens"),
                    "s1_spec": r.get("s1_spec"),
                    "s2_f1": r.get("s2_f1"),
                    "n_train": r.get("n_train"),
                    "n_test": r.get("n_test"),
                })
        if loho_rows:
            found.append("loho")
            write_csv(pd.DataFrame(loho_rows), out / "loho_results.csv")

        # Permutation baseline
        perm_rows = []
        for f in sorted(loho_dir.glob("perm_*.json")):
            d = load_json(f)
            if d:
                hospital = f.stem.replace("perm_", "").replace("_lr", "")
                perm_rows.append({
                    "hospital": hospital,
                    "completed": d.get("completed"),
                })
        if perm_rows:
            write_csv(pd.DataFrame(perm_rows), out / "loho_permutation_baseline.csv")
    else:
        missing.append("loho")

    # Norm x feature search (exp6)
    nfs_dir = RESULTS / "weekend_experiments/exp6_norm_feature_search/checkpoints"
    if nfs_dir.is_dir():
        nfs_rows = []
        for f in sorted(nfs_dir.glob("*.json")):
            d = load_json(f)
            if d and "s1_auc" in d:
                nfs_rows.append(d)
        if nfs_rows:
            found.append("norm_feature_search")
            df_nfs = pd.DataFrame(nfs_rows)
            write_csv(df_nfs, out / "nfs_raw.csv")
            # Aggregated summary
            if {"norm", "transform", "model", "s1_auc", "s2_f1"}.issubset(df_nfs.columns):
                agg = df_nfs.groupby(["norm", "transform", "model"]).agg(
                    s1_auc_mean=("s1_auc", "mean"),
                    s1_auc_std=("s1_auc", "std"),
                    s2_f1_mean=("s2_f1", "mean"),
                    s2_f1_std=("s2_f1", "std"),
                    n_seeds=("seed", "nunique"),
                ).reset_index().sort_values("s2_f1_mean", ascending=False)
                write_csv(agg, out / "nfs_aggregated.csv")
    else:
        missing.append("norm_feature_search")

    report(found, missing, cat)


# ---------------------------------------------------------------------------
# 13. Preprocessing & QC validation
# ---------------------------------------------------------------------------

def export_13_preprocessing_qc():
    cat = "13_preprocessing_qc"
    out = EXPORT / cat
    print(f"\n{'='*60}\n[{cat}] Preprocessing & QC validation")
    found, missing = [], []

    # Preprocessing validation
    pv = RESULTS / "preprocessing_validation"
    for fname in ["preproc_ablation.csv", "preproc_sensitivity.csv",
                   "batch_leakage_stats.csv", "two_stage_qc_drops.csv",
                   "two_stage_qc_retention.csv", "preproc_calibration_sweep.csv",
                   "calibration_failure_summary.csv",
                   "preproc_calibration_shifts_w20.csv"]:
        df = safe_read_csv(pv / fname)
        if df is not None:
            found.append(fname)
            write_csv(df, out / fname)
        else:
            missing.append(fname)

    # QC validation thresholds
    qv = RESULTS / "qc_validation"
    for fname in ["threshold_corr_per_replicate.csv", "threshold_survival_grid.csv"]:
        df = safe_read_csv(qv / fname)
        if df is not None:
            found.append(f"qv/{fname}")
            write_csv(df, out / fname)

    # QC experiment
    qe = RESULTS / "qc/qc_experiment"
    for fname in ["sweep_results.csv", "preprocessing_stats.csv",
                   "post_preprocessing_variance.csv", "calibration_shifts.csv"]:
        df = safe_read_csv(qe / fname)
        if df is not None:
            found.append(f"qc_exp/{fname}")
            write_csv(df, out / f"qc_experiment_{fname}")

    # QC validation detailed
    qv2 = RESULTS / "qc/qc_validation"
    for fname in ["fine_sweep_results.csv", "group_qc_summary.csv",
                   "literature_comparison.csv", "per_cancer_passrate_corr.csv",
                   "per_cancer_passrate_rsd.csv", "per_cancer_sensitivity.csv",
                   "rejection_rate_comparison.csv", "prospective_simulation.csv"]:
        df = safe_read_csv(qv2 / fname)
        if df is not None:
            found.append(f"qv2/{fname}")
            write_csv(df, out / fname)

    # QC base model
    qb = RESULTS / "qc/qc_base_model"
    for fname in ["qc_stats_all_groups.csv", "gate_results_all.csv"]:
        df = safe_read_csv(qb / fname)
        if df is not None:
            found.append(f"qb/{fname}")
            write_csv(df, out / fname)

    # Raw spectrum QC
    for fname in ["spectrum_snr.csv", "replicate_qc_stats.csv"]:
        df = safe_read_csv(RESULTS / f"figures/ALL_RAW/{fname}")
        if df is not None:
            found.append(f"raw/{fname}")
            write_csv(df, out / fname)

    # Baseline comparison
    df = safe_read_csv(TRAINING / "baseline_comparison/baseline_comparison_phaseQ.csv")
    if df is not None:
        found.append("baseline_comparison")
        write_csv(df, out / "baseline_comparison_phaseQ.csv")

    report(found, missing, cat)


# ---------------------------------------------------------------------------
# 14. Metadata & cohort
# ---------------------------------------------------------------------------

def export_14_metadata():
    cat = "14_metadata"
    out = EXPORT / cat
    print(f"\n{'='*60}\n[{cat}] Metadata & cohort descriptives")
    found, missing = [], []

    for fname in ["metadata_raw.csv", "group_statistics.csv",
                   "metadata_summary.csv", "baseline_comparison.csv"]:
        df = safe_read_csv(RESULTS / f"metadata/{fname}")
        if df is not None:
            found.append(fname)
            write_csv(df, out / fname)
        else:
            missing.append(fname)

    for fname in ["group_statistics.csv", "model_comparison_with_pr_auc.csv",
                   "per_cancer_pr_auc_comparison.csv"]:
        df = safe_read_csv(RESULTS / fname)
        if df is not None:
            found.append(f"top/{fname}")
            write_csv(df, out / f"overall_{fname}")

    report(found, missing, cat)


# ---------------------------------------------------------------------------
# 15. Bootstrap CI
# ---------------------------------------------------------------------------

def export_15_bootstrap():
    cat = "15_bootstrap_ci"
    out = EXPORT / cat
    print(f"\n{'='*60}\n[{cat}] Bootstrap confidence intervals")
    found, missing = [], []

    for bs_dir in sorted(RESULTS.glob("bootstrap_ci/*/checkpoints")):
        parent_name = bs_dir.parent.name
        rows = []
        for f in sorted(bs_dir.glob("*.json")):
            d = load_json(f)
            if d is not None:
                row = {"file": f.stem}
                row.update({k: v for k, v in d.items()
                            if isinstance(v, (int, float, str))})
                rows.append(row)
        if rows:
            found.append(f"bs_{parent_name}")
            write_csv(pd.DataFrame(rows), out / f"bootstrap_{parent_name}.csv")

    if not found:
        missing.append("bootstrap_ci")

    report(found, missing, cat)


# ---------------------------------------------------------------------------
# 16. Unified phase comparison (all experiment_log + benchmark + fold)
# ---------------------------------------------------------------------------

def export_16_unified():
    cat = "16_unified_phases"
    out = EXPORT / cat
    print(f"\n{'='*60}\n[{cat}] Unified experiment phase comparison")
    found, missing = [], []

    # All experiment_log.json
    all_logs = sorted(TRAINING.rglob("experiment_log.json"))
    rows = []
    for log_path in all_logs:
        d = load_json(log_path)
        if d is None:
            continue
        rel = log_path.relative_to(TRAINING)
        parts = list(rel.parts)
        row = {
            "path": str(rel),
            "experiment": d.get("experiment", parts[0] if parts else ""),
            "model_name": d.get("model_name", parts[1] if len(parts) > 1 else ""),
            "version": d.get("version", ""),
            "aggregate": d.get("aggregate", ""),
            "n_samples": d.get("n_samples"),
            "n_features": d.get("n_features"),
            "n_splits": d.get("n_splits"),
            "cancer_types": ",".join(d.get("cancer_types", [])),
            "timestamp": d.get("timestamp", ""),
        }
        for k in ["val_s1_auc", "val_s1_accuracy", "val_s1_sensitivity",
                   "val_s1_specificity", "val_s1_f1",
                   "val_s2_accuracy", "val_s2_f1_macro", "val_s2_auc",
                   "train_s1_auc", "train_s2_f1_macro", "train_s2_auc"]:
            row[k] = d.get(k)
        rows.append(row)
    if rows:
        found.append(f"{len(rows)} experiment_logs")
        write_csv(pd.DataFrame(rows), out / "all_experiment_logs.csv")

    # All benchmark_summary.csv
    bm_all = []
    for bm_path in sorted(TRAINING.rglob("benchmark_summary.csv")):
        df = safe_read_csv(bm_path)
        if df is not None:
            df["source"] = str(bm_path.relative_to(TRAINING).parent)
            bm_all.append(df)
    if bm_all:
        found.append(f"{len(bm_all)} benchmarks")
        write_csv(pd.concat(bm_all, ignore_index=True),
                  out / "all_benchmark_summaries.csv")

    # All fold_metrics.csv
    fm_all = []
    for fm_path in sorted(TRAINING.rglob("fold_metrics.csv")):
        df = safe_read_csv(fm_path)
        if df is not None:
            rel = fm_path.relative_to(TRAINING)
            parts = list(rel.parts)
            df["experiment"] = parts[0] if parts else ""
            df["model"] = parts[1] if len(parts) > 2 else ""
            df["version"] = parts[2] if len(parts) > 3 else ""
            df["source"] = str(rel)
            fm_all.append(df)
    if fm_all:
        found.append(f"{len(fm_all)} fold_metrics")
        write_csv(pd.concat(fm_all, ignore_index=True),
                  out / "all_fold_metrics.csv")

    report(found, missing, cat)


# ---------------------------------------------------------------------------
# 17. Spectral interpretation (Phase V7)
# ---------------------------------------------------------------------------

def export_17_spectral():
    cat = "17_spectral_interpretation"
    out = EXPORT / cat
    print(f"\n{'='*60}\n[{cat}] Phase V7 spectral interpretation")
    found, missing = [], []

    v7 = RESULTS / "figures/training/phase_V7/stage2"
    if not v7.is_dir():
        missing.append("phase_V7_dir")
        report(found, missing, cat)
        return

    for fname in ["mean_spectra_overlay.csv", "peak_intensity_by_diagnosis.csv"]:
        df = safe_read_csv(v7 / fname)
        if df is not None:
            found.append(fname)
            write_csv(df, out / fname)

    # Per-diagnosis
    by_diag = v7 / "by_diagnosis"
    if by_diag.is_dir():
        all_peak_profiles = []
        all_mean_spectra = []
        all_peak_diffs = []
        for diag_dir in sorted(by_diag.iterdir()):
            if not diag_dir.is_dir():
                continue
            diag = diag_dir.name
            for f in sorted(diag_dir.glob("*.csv")):
                df = safe_read_csv(f)
                if df is not None:
                    df["diagnosis"] = diag
                    if "peak_intensity_profile" in f.name:
                        all_peak_profiles.append(df)
                    elif "mean_spectrum" in f.name:
                        all_mean_spectra.append(df)
                    elif "peak_difference" in f.name:
                        df["comparison"] = f.stem.replace("peak_difference_vs_", "")
                        all_peak_diffs.append(df)
                    elif "confusion_classes" in f.name:
                        write_csv(df, out / f"{diag}_confusion_classes.csv")
                    found.append(f"{diag}/{f.name}")

        if all_peak_profiles:
            write_csv(pd.concat(all_peak_profiles, ignore_index=True),
                      out / "all_peak_intensity_profiles.csv")
        if all_mean_spectra:
            write_csv(pd.concat(all_mean_spectra, ignore_index=True),
                      out / "all_mean_spectra.csv")
        if all_peak_diffs:
            write_csv(pd.concat(all_peak_diffs, ignore_index=True),
                      out / "all_peak_differences.csv")

    report(found, missing, cat)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("SERS-AI: Export all experiment data for R figures")
    print(f"Output directory: {EXPORT}")
    print("=" * 60)

    # Clean output directory
    if EXPORT.exists():
        shutil.rmtree(EXPORT)
    ensure_dir(EXPORT)

    export_01_baseline()
    export_02_normalization()
    export_03_clinical_fusion()
    export_04_heldout()
    export_05_single_cancer()
    export_06_cross_instrument()
    export_07_multiview()
    export_08_dl_models()
    export_09_stacking()
    export_10_optimization()
    export_11_interpretation()
    export_12_weekend()
    export_13_preprocessing_qc()
    export_14_metadata()
    export_15_bootstrap()
    export_16_unified()
    export_17_spectral()

    # Summary
    print("\n" + "=" * 60)
    total_csvs = len(list(EXPORT.rglob("*.csv")))
    print(f"DONE: {total_csvs} CSV files exported")
    print()
    for d in sorted(EXPORT.iterdir()):
        if d.is_dir():
            n = len(list(d.glob("*.csv")))
            print(f"  {d.name}: {n} files")
    print("=" * 60)


if __name__ == "__main__":
    main()
