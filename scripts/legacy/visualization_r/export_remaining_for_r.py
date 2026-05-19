#!/usr/bin/env python3
"""
export_remaining_for_r.py
=========================
Extract data for ~40 remaining R figures that need model loading or special processing.
Output: results/r_export/remaining/

Sections:
  1. DL Internal Visualization (FiLM gamma/beta, CrossAttention weights)
  2. Z-opt remaining (fig4/fig5/fig6)
  3. AB-XAI data (t-SNE, SHAP, LR coef overlay)
  4. Two-device / cross-instrument data
  5. Weekend summary card data
  6. Cross-instrument spectral comparison (paired samples)
  7. Stacking V1 data (Thermo/Medical)
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "results" / "r_export" / "remaining"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS = PROJECT_ROOT / "results"
TRAINING = RESULTS / "training"


def save_csv(df: pd.DataFrame, name: str, subdir: str | None = None):
    """Save a DataFrame as CSV and print summary."""
    if subdir:
        d = OUT_DIR / subdir
        d.mkdir(parents=True, exist_ok=True)
        path = d / name
    else:
        path = OUT_DIR / name
    df.to_csv(path, index=False)
    print(f"  -> {path.relative_to(PROJECT_ROOT)} ({df.shape[0]} rows x {df.shape[1]} cols)")


def warn_skip(msg: str):
    print(f"  [SKIP] {msg}")


# ============================================================
# 1. DL Internal Visualization (FiLM, CrossAttention)
# ============================================================
def export_dl_internals():
    print("\n=== 1. DL Internal Visualization ===")

    try:
        import torch
    except ImportError:
        warn_skip("PyTorch not available, skipping DL model loading")
        return

    # Try loading FiLM model from film_xattn_v2 (latest) or film_xattn_comparison
    film_dirs = [
        TRAINING / "film_xattn_v2" / "film_resnet18",
        TRAINING / "film_xattn_comparison" / "film_resnet18",
        TRAINING / "blc_mfds" / "film",
    ]
    xattn_dirs = [
        TRAINING / "film_xattn_v2" / "xattn_resnet18",
        TRAINING / "film_xattn_comparison" / "xattn_resnet18",
        TRAINING / "blc_mfds" / "xattn",
    ]

    # --- FiLM gamma/beta extraction ---
    film_extracted = False
    for film_dir in film_dirs:
        if not film_dir.exists():
            continue

        # Find checkpoint files
        ckpt_files = sorted(film_dir.rglob("fold_*.pt"))
        if not ckpt_files:
            continue

        # Load summary to get clinical features and config
        summary_files = list(film_dir.rglob("training_summary.json")) + list(film_dir.rglob("summary.json"))
        clinical_features = None
        n_clinical = 3
        if summary_files:
            with open(summary_files[0]) as f:
                summary = json.load(f)
            clinical_features = summary.get("clinical_features")
            n_clinical = summary.get("n_clinical", 3)

        print(f"  Loading FiLM checkpoints from {film_dir.relative_to(PROJECT_ROOT)}")
        print(f"  Found {len(ckpt_files)} fold checkpoints, n_clinical={n_clinical}")

        # Extract FiLM parameters from each fold
        all_gamma_beta = []
        for ckpt_path in ckpt_files:
            try:
                state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                    state_dict = state_dict["model_state_dict"]
                elif isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]

                fold_name = ckpt_path.stem  # e.g. fold_0

                # Extract FiLM generator weights
                for layer_idx in range(1, 5):
                    prefix = f"encoder.film{layer_idx}.mlp."
                    # The last linear layer has bias = [gamma_init; beta_init]
                    bias_key = None
                    weight_key = None
                    for k in state_dict:
                        if prefix in k and "bias" in k:
                            # Take the last bias (output layer)
                            bias_key = k
                        if prefix in k and "weight" in k:
                            weight_key = k

                    if bias_key and bias_key in state_dict:
                        bias = state_dict[bias_key].numpy()
                        n_ch = len(bias) // 2
                        gamma = bias[:n_ch]
                        beta = bias[n_ch:]

                        for feat_idx in range(min(n_ch, 32)):  # cap to 32 features for readability
                            all_gamma_beta.append({
                                "fold": fold_name,
                                "layer": f"layer{layer_idx}",
                                "feature_idx": feat_idx,
                                "gamma": float(gamma[feat_idx]),
                                "beta": float(beta[feat_idx]),
                            })
            except Exception as e:
                warn_skip(f"Could not load {ckpt_path.name}: {e}")
                continue

        if all_gamma_beta:
            df = pd.DataFrame(all_gamma_beta)
            # Add clinical variable names if available
            if clinical_features:
                # Map: FiLM bias is per-channel, not per-clinical-var
                # gamma/beta are per output channel, not per clinical feature
                pass
            save_csv(df, "film_gamma_beta_parameters.csv", "dl_internals")
            film_extracted = True

            # Also create a summary: mean gamma/beta per layer
            summary_df = df.groupby(["layer", "feature_idx"]).agg(
                gamma_mean=("gamma", "mean"),
                gamma_std=("gamma", "std"),
                beta_mean=("beta", "mean"),
                beta_std=("beta", "std"),
            ).reset_index()
            save_csv(summary_df, "film_gamma_beta_summary.csv", "dl_internals")
            break

    if not film_extracted:
        warn_skip("No FiLM model checkpoints found or loadable")

    # --- CrossAttention weights extraction ---
    xattn_extracted = False
    for xattn_dir in xattn_dirs:
        if not xattn_dir.exists():
            continue

        ckpt_files = sorted(xattn_dir.rglob("fold_*.pt"))
        if not ckpt_files:
            continue

        summary_files = list(xattn_dir.rglob("training_summary.json")) + list(xattn_dir.rglob("summary.json"))
        clinical_features = None
        n_clinical = 3
        if summary_files:
            with open(summary_files[0]) as f:
                summary = json.load(f)
            clinical_features = summary.get("clinical_features")
            n_clinical = summary.get("n_clinical", 3)

        print(f"  Loading CrossAttention checkpoints from {xattn_dir.relative_to(PROJECT_ROOT)}")
        print(f"  Found {len(ckpt_files)} fold checkpoints, n_clinical={n_clinical}")

        # Extract attention layer weights
        all_attn_params = []
        for ckpt_path in ckpt_files:
            try:
                state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                    state_dict = state_dict["model_state_dict"]
                elif isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]

                fold_name = ckpt_path.stem

                # Extract cross-attention in_proj weights per layer
                for layer_idx in range(4):  # up to 4 layers
                    prefix = f"encoder.attn_layers.{layer_idx}.cross_attn."
                    in_proj_w = None
                    in_proj_b = None
                    out_proj_w = None

                    for k in state_dict:
                        if k.startswith(prefix):
                            if "in_proj_weight" in k:
                                in_proj_w = state_dict[k].numpy()
                            elif "in_proj_bias" in k:
                                in_proj_b = state_dict[k].numpy()
                            elif "out_proj.weight" in k:
                                out_proj_w = state_dict[k].numpy()

                    if in_proj_w is not None:
                        d_model = in_proj_w.shape[1]
                        # in_proj_weight is (3*d_model, d_model) for Q, K, V
                        q_w = in_proj_w[:d_model]
                        k_w = in_proj_w[d_model:2*d_model]
                        v_w = in_proj_w[2*d_model:]

                        # Summarize with Frobenius norms
                        all_attn_params.append({
                            "fold": fold_name,
                            "layer": layer_idx,
                            "q_norm": float(np.linalg.norm(q_w)),
                            "k_norm": float(np.linalg.norm(k_w)),
                            "v_norm": float(np.linalg.norm(v_w)),
                            "out_norm": float(np.linalg.norm(out_proj_w)) if out_proj_w is not None else np.nan,
                            "d_model": d_model,
                        })

                # Extract clinical tokenizer weights
                for i in range(n_clinical):
                    w_key = f"encoder.clinical_tokenizer.projections.{i}.0.weight"
                    if w_key in state_dict:
                        w = state_dict[w_key].numpy()
                        all_attn_params.append({
                            "fold": fold_name,
                            "layer": -1,  # tokenizer
                            "clinical_var_idx": i,
                            "clinical_var": clinical_features[i] if clinical_features and i < len(clinical_features) else f"var_{i}",
                            "tokenizer_w_norm": float(np.linalg.norm(w)),
                            "tokenizer_w_mean": float(w.mean()),
                            "tokenizer_w_std": float(w.std()),
                        })

            except Exception as e:
                warn_skip(f"Could not load {ckpt_path.name}: {e}")
                continue

        if all_attn_params:
            df = pd.DataFrame(all_attn_params)
            save_csv(df, "xattn_layer_weights.csv", "dl_internals")
            xattn_extracted = True
            break

    if not xattn_extracted:
        warn_skip("No CrossAttention model checkpoints found or loadable")

    # Also export clinical features list if available from BLC models
    blc_film_summary = TRAINING / "blc_mfds" / "film" / "summary.json"
    if blc_film_summary.exists():
        with open(blc_film_summary) as f:
            s = json.load(f)
        clin = s.get("clinical_features", [])
        if clin:
            df = pd.DataFrame({"idx": range(len(clin)), "clinical_feature": clin})
            save_csv(df, "clinical_features_list.csv", "dl_internals")


# ============================================================
# 2. Z-opt remaining (fig4/fig5/fig6)
# ============================================================
def export_zopt_remaining():
    print("\n=== 2. Z-opt Remaining (fig4/fig5/fig6) ===")

    opt_dir = TRAINING / "optimize_7cancer_full"

    # fig4: phase4_held_out_repeats (check if already in r_export)
    existing_fig4 = RESULTS / "r_export" / "10_optimization" / "phase4_held_out_repeats.csv"
    src_fig4 = opt_dir / "phase4_held_out_repeats.csv"
    if existing_fig4.exists():
        print(f"  fig4 already at {existing_fig4.relative_to(PROJECT_ROOT)}")
        # Copy to remaining for completeness
        df = pd.read_csv(existing_fig4)
        save_csv(df, "zopt_fig4_held_out_repeats.csv", "zopt")
    elif src_fig4.exists():
        df = pd.read_csv(src_fig4)
        save_csv(df, "zopt_fig4_held_out_repeats.csv", "zopt")
    else:
        warn_skip("phase4_held_out_repeats.csv not found")

    # fig5: phase4_final_evaluation - convert JSON to CSV
    src_fig5_json = opt_dir / "phase4_final_evaluation.json"
    existing_fig5 = RESULTS / "r_export" / "10_optimization" / "phase4_final_evaluation.csv"
    if existing_fig5.exists():
        df = pd.read_csv(existing_fig5)
        save_csv(df, "zopt_fig5_final_evaluation.csv", "zopt")
    elif src_fig5_json.exists():
        with open(src_fig5_json) as f:
            data = json.load(f)

        rows = []
        for section in ["cv_results", "held_out_results"]:
            if section in data:
                row = {"section": section}
                row.update(data[section])
                rows.append(row)

        if data.get("config"):
            config_row = {"section": "config"}
            config = data["config"]
            for k, v in config.items():
                if isinstance(v, (str, int, float, type(None))):
                    config_row[k] = v
                elif isinstance(v, list):
                    config_row[k] = "|".join(str(x) for x in v)
                elif isinstance(v, dict):
                    for kk, vv in v.items():
                        config_row[f"{k}_{kk}"] = vv if not isinstance(vv, list) else "|".join(str(x) for x in vv)
            rows.append(config_row)

        df = pd.DataFrame(rows)
        save_csv(df, "zopt_fig5_final_evaluation.csv", "zopt")
    else:
        warn_skip("phase4_final_evaluation not found")

    # fig6: per-cancer accuracy data from optimization
    # Check for per-cancer breakdown in the held_out_repeats or other sources
    phase3 = opt_dir / "phase3_architecture_hp_search.csv"
    if phase3.exists():
        df = pd.read_csv(phase3)
        save_csv(df, "zopt_fig6_architecture_hp_search.csv", "zopt")
    else:
        warn_skip("phase3_architecture_hp_search.csv not found")

    # Also export phase1 and phase2 if available
    for fname in ["phase1_wavenumber_ablation.csv", "phase2_aggregation_ablation.csv"]:
        src = opt_dir / fname
        if src.exists():
            df = pd.read_csv(src)
            save_csv(df, f"zopt_{fname}", "zopt")


# ============================================================
# 3. AB-XAI data
# ============================================================
def export_ab_xai():
    print("\n=== 3. AB-XAI Data ===")

    xai_dir = TRAINING / "phase_ab_xai"

    # SHAP profiles
    shap_path = xai_dir / "analysis2_shap_profiles.csv"
    if shap_path.exists():
        df = pd.read_csv(shap_path)
        save_csv(df, "ab_xai_shap_profiles.csv", "ab_xai")
    else:
        warn_skip("SHAP profiles not found")

    # Top overlap wavenumbers (LR coef overlap)
    overlap_path = xai_dir / "analysis1_top_overlap_wavenumbers.csv"
    if overlap_path.exists():
        df = pd.read_csv(overlap_path)
        save_csv(df, "ab_xai_top_overlap_wavenumbers.csv", "ab_xai")
    else:
        warn_skip("Top overlap wavenumbers not found")

    # Phase AB summary
    summary_path = xai_dir / "phase_ab_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            data = json.load(f)
        # Flatten to CSV
        rows = []
        results = data.get("results", {})

        # LR coefficient analysis
        lr_coef = results.get("analysis_1_lr_coef", {})
        if lr_coef:
            rows.append({"analysis": "lr_coef_cosine_similarity", "value": lr_coef.get("cosine_similarity")})
            rows.append({"analysis": "lr_coef_pearson_correlation", "value": lr_coef.get("pearson_correlation")})

        # Misclassification counts
        misclass = results.get("analysis_3_misclassified", {})
        for k, v in misclass.items():
            if isinstance(v, (int, float)):
                rows.append({"analysis": f"misclass_{k}", "value": v})
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    rows.append({"analysis": f"misclass_{k}_{kk}", "value": vv})

        if rows:
            df = pd.DataFrame(rows)
            save_csv(df, "ab_xai_summary.csv", "ab_xai")

    # LR coefficient overlay with mean spectrum
    _create_lr_coef_overlay()


def _create_lr_coef_overlay():
    """Combine mean spectrum with LR coefficients for overlay plot."""
    mean_spec_path = RESULTS / "r_export" / "17_spectral_interpretation" / "mean_spectra_overlay.csv"
    coef_path = RESULTS / "r_export" / "predictions" / "lr_coefficients" / "phase_U_7class_binary_coefs.csv"

    if not mean_spec_path.exists():
        warn_skip("mean_spectra_overlay.csv not found")
        return
    if not coef_path.exists():
        warn_skip("LR coefficients not found")
        return

    mean_spec = pd.read_csv(mean_spec_path)
    coefs = pd.read_csv(coef_path)

    # Mean spectrum has: diagnosis, n_samples, feature, wavenumber, mean_intensity, std_intensity
    # Create a global mean spectrum (average across all diagnoses)
    global_mean = mean_spec.groupby("wavenumber").agg(
        mean_intensity=("mean_intensity", "mean"),
    ).reset_index().sort_values("wavenumber")

    # Coefs have: feature_idx, coef_mean, coef_std, fold_0..fold_4
    # feature_idx maps to wavenumber order
    # Get wavenumber grid from processed_spectra
    proc_spec = RESULTS / "processed_spectra.csv"
    if proc_spec.exists():
        spec_df = pd.read_csv(proc_spec, nrows=1)
        wn_cols = [c for c in spec_df.columns if c.startswith("x_")]
        wavenumbers = [float(c.replace("x_", "")) for c in wn_cols]
    else:
        # Fallback: use unique wavenumbers from mean_spectra
        wavenumbers = sorted(global_mean["wavenumber"].unique())

    # Allow small mismatch (e.g. 935 coefs with intercept vs 933 wavenumbers)
    n_match = min(len(coefs), len(wavenumbers))
    if n_match > 0:
        coefs = coefs.head(n_match).copy()
        coefs["wavenumber"] = wavenumbers[:n_match]

        overlay = global_mean.merge(coefs[["wavenumber", "coef_mean", "coef_std"]], on="wavenumber", how="inner")
        if not overlay.empty:
            save_csv(overlay, "lr_coef_spectrum_overlay.csv", "ab_xai")
        else:
            warn_skip("No matching wavenumbers for LR coef overlay")

    # Also create per-cancer stage2 coef overlay
    stage2_coef_path = RESULTS / "r_export" / "predictions" / "lr_coefficients" / "phase_U_7class_stage2_coefs.csv"
    if stage2_coef_path.exists():
        s2_coefs = pd.read_csv(stage2_coef_path)
        if len(s2_coefs) > 0:
            # Stage2 coefs have one row per feature per class
            # Check structure
            if "feature_idx" in s2_coefs.columns:
                s2_coefs = s2_coefs.copy()
                n_features = s2_coefs["feature_idx"].nunique() if "feature_idx" in s2_coefs.columns else len(s2_coefs)
                if n_features <= len(wavenumbers):
                    # Add wavenumber mapping
                    wn_map = {i: wavenumbers[i] for i in range(min(n_features, len(wavenumbers)))}
                    s2_coefs["wavenumber"] = s2_coefs["feature_idx"].map(wn_map)
                    save_csv(s2_coefs, "lr_stage2_coef_with_wavenumber.csv", "ab_xai")


# ============================================================
# 4. Two-device / cross-instrument data
# ============================================================
def export_cross_instrument():
    print("\n=== 4. Cross-Instrument Data ===")

    ci_report = TRAINING / "cross_instrument" / "cross_instrument_report.json"
    if ci_report.exists():
        with open(ci_report) as f:
            data = json.load(f)

        # Self-CV comparison
        rows = []
        for key in ["thermo_self_cv", "medical_self_cv"]:
            if key in data:
                row = {"scenario": key}
                row.update(data[key])
                rows.append(row)

        # Cross-device results
        for key in ["cross_thermo_to_medical", "cross_medical_to_thermo"]:
            if key in data:
                d = data[key]
                row = {"scenario": key}
                for k, v in d.items():
                    if k != "per_group":
                        row[k] = v
                rows.append(row)

        if rows:
            df = pd.DataFrame(rows)
            save_csv(df, "cross_instrument_summary.csv", "cross_instrument")

        # Per-group cross-device results
        per_group_rows = []
        for direction in ["cross_thermo_to_medical", "cross_medical_to_thermo"]:
            if direction in data and "per_group" in data[direction]:
                for group, stats in data[direction]["per_group"].items():
                    row = {"direction": direction, "group": group}
                    row.update(stats)
                    per_group_rows.append(row)

        if per_group_rows:
            df = pd.DataFrame(per_group_rows)
            save_csv(df, "cross_instrument_per_group.csv", "cross_instrument")
    else:
        warn_skip("cross_instrument_report.json not found")

    # Medical spectra mean per group
    med_spec_path = TRAINING / "cross_instrument" / "processed_spectra_medical.csv"
    med_meta_path = TRAINING / "cross_instrument" / "metadata_medical.csv"
    if med_spec_path.exists():
        med_spec = pd.read_csv(med_spec_path)
        wn_cols = [c for c in med_spec.columns if c.startswith("x_")]
        # Compute mean spectra per group
        mean_med = med_spec.groupby("group")[wn_cols].mean()
        mean_med_long = mean_med.reset_index().melt(id_vars="group", var_name="feature", value_name="mean_intensity")
        mean_med_long["wavenumber"] = mean_med_long["feature"].str.replace("x_", "").astype(float)
        mean_med_long["instrument"] = "Medical"
        mean_med_long = mean_med_long[["instrument", "group", "wavenumber", "mean_intensity"]]
        save_csv(mean_med_long, "medical_mean_spectra_per_group.csv", "cross_instrument")

    # Thermo spectra mean per group (from main processed_spectra)
    thermo_spec_path = RESULTS / "processed_spectra.csv"
    if thermo_spec_path.exists():
        thermo_spec = pd.read_csv(thermo_spec_path)
        wn_cols = [c for c in thermo_spec.columns if c.startswith("x_")]
        mean_thermo = thermo_spec.groupby("group")[wn_cols].mean()
        mean_thermo_long = mean_thermo.reset_index().melt(id_vars="group", var_name="feature", value_name="mean_intensity")
        mean_thermo_long["wavenumber"] = mean_thermo_long["feature"].str.replace("x_", "").astype(float)
        mean_thermo_long["instrument"] = "Thermo"
        mean_thermo_long = mean_thermo_long[["instrument", "group", "wavenumber", "mean_intensity"]]
        save_csv(mean_thermo_long, "thermo_mean_spectra_per_group.csv", "cross_instrument")

        # Combine both
        if med_spec_path.exists():
            combined = pd.concat([mean_thermo_long, mean_med_long], ignore_index=True)
            save_csv(combined, "two_device_mean_spectra_comparison.csv", "cross_instrument")

    # Calibration sweep results
    sweep_dir = RESULTS / "cross_instrument" / "calibration" / "sweep"
    for fname in ["method_comparison_summary.csv", "method_comparison_multiseed.csv", "per_group_best.csv"]:
        src = sweep_dir / fname
        if src.exists():
            df = pd.read_csv(src)
            save_csv(df, f"calibration_{fname}", "cross_instrument")

    # Calibration leaderboard
    lb = RESULTS / "cross_instrument" / "calibration" / "leaderboard.csv"
    if lb.exists():
        df = pd.read_csv(lb)
        save_csv(df, "calibration_leaderboard.csv", "cross_instrument")

    # Paired held metrics
    phm = RESULTS / "cross_instrument" / "calibration" / "paired_held_metrics.csv"
    if phm.exists():
        df = pd.read_csv(phm)
        save_csv(df, "calibration_paired_held_metrics.csv", "cross_instrument")

    # Diagnosis paired stats
    diag = RESULTS / "cross_instrument" / "calibration" / "diagnosis" / "paired_stats.csv"
    if diag.exists():
        df = pd.read_csv(diag)
        save_csv(df, "calibration_diagnosis_paired_stats.csv", "cross_instrument")


# ============================================================
# 5. Weekend summary card data
# ============================================================
def export_weekend_summary():
    print("\n=== 5. Weekend Summary Card Data ===")

    weekend_dir = RESULTS / "weekend_experiments"

    # Permutation importance
    pi_dir = weekend_dir / "exp4_permutation_importance_20260403_204449"
    if pi_dir.exists():
        for fname in ["importance_ranking.csv", "null_distribution.csv"]:
            src = pi_dir / fname
            if src.exists():
                df = pd.read_csv(src)
                save_csv(df, f"weekend_pi_{fname}", "weekend")

    # Stacking optimization
    stk_dir = weekend_dir / "stacking_optimization"
    if stk_dir.exists():
        for fname in ["nested_cv_results.csv", "base_model_contribution.csv", "single_vs_ensemble.csv"]:
            src = stk_dir / fname
            if src.exists():
                df = pd.read_csv(src)
                save_csv(df, f"weekend_stk_{fname}", "weekend")

        # Best ensemble config
        best_cfg = stk_dir / "best_ensemble_config.json"
        if best_cfg.exists():
            with open(best_cfg) as f:
                data = json.load(f)
            rows = []
            for k, v in data.items():
                if isinstance(v, (str, int, float, bool, type(None))):
                    rows.append({"key": k, "value": str(v)})
                elif isinstance(v, list):
                    rows.append({"key": k, "value": "|".join(str(x) for x in v)})
                elif isinstance(v, dict):
                    for kk, vv in v.items():
                        rows.append({"key": f"{k}.{kk}", "value": str(vv)})
            if rows:
                df = pd.DataFrame(rows)
                save_csv(df, "weekend_stk_best_config.csv", "weekend")

    # Norm/feature search
    norm_dir = weekend_dir / "exp6_norm_feature_search"
    if norm_dir.exists():
        for fname in os.listdir(norm_dir):
            if fname.endswith(".csv"):
                df = pd.read_csv(norm_dir / fname)
                save_csv(df, f"weekend_norm_{fname}", "weekend")

    # LOHO validation
    loho_dir = weekend_dir / "loho_validation"
    if loho_dir.exists():
        for fname in os.listdir(loho_dir):
            if fname.endswith(".csv"):
                df = pd.read_csv(loho_dir / fname)
                save_csv(df, f"weekend_loho_{fname}", "weekend")

    # Experiment logs - extract key numbers
    logs_dir = weekend_dir / "logs"
    if logs_dir.exists():
        log_summary = []
        for logfile in sorted(logs_dir.glob("*.log")):
            exp_name = logfile.stem
            lines = logfile.read_text().strip().split("\n")
            n_lines = len(lines)
            last_line = lines[-1] if lines else ""
            log_summary.append({
                "experiment": exp_name,
                "n_log_lines": n_lines,
                "last_line": last_line[:200],
            })
        if log_summary:
            df = pd.DataFrame(log_summary)
            save_csv(df, "weekend_experiment_log_summary.csv", "weekend")


# ============================================================
# 6. Cross-instrument spectral comparison (paired samples)
# ============================================================
def export_paired_spectral_comparison():
    print("\n=== 6. Cross-Instrument Paired Spectral Comparison ===")

    paired_path = RESULTS / "cross_instrument" / "calibration" / "paired_samples.csv"
    if not paired_path.exists():
        warn_skip("paired_samples.csv not found")
        return

    paired = pd.read_csv(paired_path)
    print(f"  Paired samples: {paired.shape[0]} rows, columns: {paired.columns.tolist()[:5]}")

    # paired_samples.csv has group + sample_id (1569 rows)
    # We need to match with Thermo and Medical spectra to build per-group mean comparison

    # Load Thermo spectra
    thermo_path = RESULTS / "processed_spectra.csv"
    med_path = TRAINING / "cross_instrument" / "processed_spectra_medical.csv"

    if not thermo_path.exists() or not med_path.exists():
        warn_skip("Need both Thermo and Medical spectra for paired comparison")
        return

    thermo = pd.read_csv(thermo_path)
    medical = pd.read_csv(med_path)

    # Get paired sample IDs
    paired_ids = set(zip(paired["group"], paired["sample_id"]))

    # Filter to paired samples and aggregate per sample (mean across replicates)
    wn_thermo = [c for c in thermo.columns if c.startswith("x_")]
    wn_medical = [c for c in medical.columns if c.startswith("x_")]

    thermo_paired = thermo[thermo.apply(lambda r: (r["group"], r["sample_id"]) in paired_ids, axis=1)]
    medical_paired = medical[medical.apply(lambda r: (r["group"], r["sample_id"]) in paired_ids, axis=1)]

    if thermo_paired.empty or medical_paired.empty:
        warn_skip("No matched paired samples found in spectra")
        return

    print(f"  Thermo paired: {thermo_paired['sample_id'].nunique()} unique samples")
    print(f"  Medical paired: {medical_paired['sample_id'].nunique()} unique samples")

    # Per-group mean spectra comparison
    thermo_mean = thermo_paired.groupby("group")[wn_thermo].mean()
    medical_mean = medical_paired.groupby("group")[wn_medical].mean()

    # Melt to long format
    thermo_long = thermo_mean.reset_index().melt(id_vars="group", var_name="feature", value_name="mean_intensity")
    thermo_long["wavenumber"] = thermo_long["feature"].str.replace("x_", "").astype(float)
    thermo_long["instrument"] = "Thermo"

    medical_long = medical_mean.reset_index().melt(id_vars="group", var_name="feature", value_name="mean_intensity")
    medical_long["wavenumber"] = medical_long["feature"].str.replace("x_", "").astype(float)
    medical_long["instrument"] = "Medical"

    combined = pd.concat([
        thermo_long[["instrument", "group", "wavenumber", "mean_intensity"]],
        medical_long[["instrument", "group", "wavenumber", "mean_intensity"]],
    ], ignore_index=True)

    save_csv(combined, "paired_mean_spectra_by_group.csv", "cross_instrument")

    # Overall mean spectrum per instrument (across all groups)
    thermo_overall = thermo_paired[wn_thermo].mean()
    medical_overall = medical_paired[wn_medical].mean()

    overall_rows = []
    for col, val in thermo_overall.items():
        wn = float(col.replace("x_", ""))
        overall_rows.append({"instrument": "Thermo", "wavenumber": wn, "mean_intensity": val})
    for col, val in medical_overall.items():
        wn = float(col.replace("x_", ""))
        overall_rows.append({"instrument": "Medical", "wavenumber": wn, "mean_intensity": val})

    overall_df = pd.DataFrame(overall_rows)
    save_csv(overall_df, "paired_overall_mean_spectra.csv", "cross_instrument")


# ============================================================
# 7. Stacking V1 data (Thermo/Medical)
# ============================================================
def export_stacking_v1():
    print("\n=== 7. Stacking V1 Data ===")

    stk_dir = TRAINING / "stacking_ensemble"

    for device in ["thermo", "medical"]:
        report_path = stk_dir / f"stacking_report_{device}.json"
        if not report_path.exists():
            warn_skip(f"stacking_report_{device}.json not found")
            continue

        with open(report_path) as f:
            data = json.load(f)

        # Extract per-model results
        results = data.get("results", [])
        if results:
            df = pd.DataFrame(results)
            df["device"] = device
            save_csv(df, f"stacking_v1_{device}_model_results.csv", "stacking_v1")

        # Extract metadata
        meta_rows = [{
            "device": device,
            "timestamp": data.get("timestamp"),
            "data_source": data.get("data"),
            "duration_sec": data.get("duration_sec"),
        }]
        save_csv(pd.DataFrame(meta_rows), f"stacking_v1_{device}_metadata.csv", "stacking_v1")

    # Combine both devices into one comparison table
    thermo_path = stk_dir / "stacking_report_thermo.json"
    medical_path = stk_dir / "stacking_report_medical.json"

    if thermo_path.exists() and medical_path.exists():
        with open(thermo_path) as f:
            t_data = json.load(f)
        with open(medical_path) as f:
            m_data = json.load(f)

        t_results = t_data.get("results", [])
        m_results = m_data.get("results", [])

        if t_results and m_results:
            t_df = pd.DataFrame(t_results)
            t_df["device"] = "thermo"
            m_df = pd.DataFrame(m_results)
            m_df["device"] = "medical"
            combined = pd.concat([t_df, m_df], ignore_index=True)
            save_csv(combined, "stacking_v1_combined_comparison.csv", "stacking_v1")

    # Also export multiview ensemble report if available
    mv_report = TRAINING / "multiview_ensemble" / "ensemble_report_thermo.json"
    if mv_report.exists():
        with open(mv_report) as f:
            data = json.load(f)
        results = data.get("results", data.get("channel_results", []))
        if isinstance(results, list) and results:
            df = pd.DataFrame(results)
            save_csv(df, "multiview_ensemble_thermo.csv", "stacking_v1")
        elif isinstance(results, dict):
            rows = [{"key": k, **v} if isinstance(v, dict) else {"key": k, "value": v} for k, v in results.items()]
            df = pd.DataFrame(rows)
            save_csv(df, "multiview_ensemble_thermo.csv", "stacking_v1")


# ============================================================
# BONUS: Additional useful exports
# ============================================================
def export_additional():
    print("\n=== Bonus: Additional Exports ===")

    # Contrastive learning results
    for device in ["thermo", "medical"]:
        cont_path = TRAINING / "contrastive" / f"contrastive_report_{device}.json"
        if cont_path.exists():
            with open(cont_path) as f:
                data = json.load(f)
            # Flatten
            rows = []
            for k, v in data.items():
                if isinstance(v, (str, int, float, type(None))):
                    rows.append({"key": k, "value": v})
                elif isinstance(v, dict):
                    for kk, vv in v.items():
                        rows.append({"key": f"{k}.{kk}", "value": vv})
            if rows:
                df = pd.DataFrame(rows)
                save_csv(df, f"contrastive_{device}_summary.csv", "bonus")

    # Multiview derivative channel statistics
    ch_stats = TRAINING / "multiview_derivative" / "channel_statistics.csv"
    if ch_stats.exists():
        df = pd.read_csv(ch_stats)
        save_csv(df, "multiview_channel_statistics.csv", "bonus")

    # Peak analysis
    for fname in ["coefficient_peak_crossref.csv", "peak_discrimination_matrix.csv"]:
        src = TRAINING / "peak_analysis" / fname
        if src.exists():
            df = pd.read_csv(src)
            save_csv(df, f"peak_{fname}", "bonus")

    # Confounding analysis
    conf = TRAINING / "confounding_analysis" / "confounding_summary.json"
    if conf.exists():
        with open(conf) as f:
            data = json.load(f)
        rows = []
        for k, v in data.items():
            if isinstance(v, (str, int, float, type(None))):
                rows.append({"key": k, "value": str(v)})
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    if isinstance(vv, dict):
                        for kkk, vvv in vv.items():
                            rows.append({"key": f"{k}.{kk}.{kkk}", "value": str(vvv)})
                    else:
                        rows.append({"key": f"{k}.{kk}", "value": str(vv)})
        if rows:
            df = pd.DataFrame(rows)
            save_csv(df, "confounding_analysis_summary.csv", "bonus")

    # Clinical analysis summary
    clin = TRAINING / "clinical_analysis" / "clinical_analysis_summary.json"
    if clin.exists():
        with open(clin) as f:
            data = json.load(f)
        rows = []
        for k, v in data.items():
            if isinstance(v, (str, int, float, type(None))):
                rows.append({"key": k, "value": str(v)})
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    rows.append({"key": f"{k}.{kk}", "value": str(vv)})
            elif isinstance(v, list):
                rows.append({"key": k, "value": "|".join(str(x) for x in v)})
        if rows:
            df = pd.DataFrame(rows)
            save_csv(df, "clinical_analysis_summary.csv", "bonus")


# ============================================================
# Main
# ============================================================
def main():
    print(f"Output directory: {OUT_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Project root: {PROJECT_ROOT}")

    export_dl_internals()
    export_zopt_remaining()
    export_ab_xai()
    export_cross_instrument()
    export_weekend_summary()
    export_paired_spectral_comparison()
    export_stacking_v1()
    export_additional()

    # Final summary
    print("\n" + "=" * 60)
    csv_files = sorted(OUT_DIR.rglob("*.csv"))
    print(f"Total CSVs written: {len(csv_files)}")
    for f in csv_files:
        print(f"  {f.relative_to(OUT_DIR)}")


if __name__ == "__main__":
    main()
