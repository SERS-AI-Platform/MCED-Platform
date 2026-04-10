"""
XAI Analysis for FiLM & Cross-Attention Fusion Models

Generates:
1. FiLM: Per-cancer-type gamma/beta heatmap (which layers modulate which cancer)
2. Cross-Attention: Attention weight analysis (which clinical vars matter per cancer)
3. Comparison bar chart: Baseline vs FiLM vs CrossAttn
4. FiLM: Sex-conditioned modulation analysis (male vs female gamma patterns)
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.model import ModelConfig, TwoStageLoss
from models.clinical_utils import load_clinical, merge_clinical_features, ClinicalSERSDataset, TIER1_COLS
from models.film.model import FiLMConfig, build_film_model
from models.cross_attention.model import CrossAttentionConfig, build_cross_attention_model
from models.train import (
    load_processed_spectra, get_feature_columns, apply_class_selection,
    resolve_aliases, aggregate_replicates, create_labels,
)

CANCER_NAMES = ["PRO", "BRE", "OVA", "LUN", "CRC", "PAN", "BLC"]
NON_CANCER_NAMES = ["NOR", "DIA", "HBP", "H.D."]
CLINICAL_NAMES = ["Age", "Sex", "BMI"]
LAYER_NAMES = ["Layer1\n(32ch)", "Layer2\n(64ch)", "Layer3\n(128ch)", "Layer4\n(256ch)"]


def load_data():
    """Load and prepare data matching training pipeline exactly."""
    import yaml
    df = load_processed_spectra()
    feat_cols = get_feature_columns(df)

    try:
        with open(PROJECT_ROOT / "config" / "config.yaml", encoding="utf-8") as f:
            raw_cfg = yaml.safe_load(f)
        mc = ModelConfig.from_pipeline_config(raw_cfg, n_spectral_features=len(feat_cols))
    except FileNotFoundError:
        mc = ModelConfig(n_spectral_features=len(feat_cols))

    mc = apply_class_selection(mc)
    df = resolve_aliases(df, mc)
    clin = load_clinical()
    df_agg = aggregate_replicates(df, feat_cols, "medoid")
    X, bl, ctl, sample_ids, groups_arr = create_labels(df_agg, mc)
    valid_groups = set(mc.cancer_types) | set(mc.non_cancer_groups)
    df_valid = df_agg[df_agg["group"].isin(valid_groups)].copy()
    clinical_features = merge_clinical_features(df_valid, clin, TIER1_COLS)
    return X, bl, ctl, groups_arr, clinical_features, mc


def load_fold_model(ckpt_path, model_type, mc):
    """Load a trained fold checkpoint, reconstructing config from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Use checkpoint's config to ensure shape match
    ckpt_config = ckpt.get("config", {})
    model_config = ModelConfig(**{k: v for k, v in ckpt_config.items()
                                  if k in ModelConfig.__dataclass_fields__})
    if model_type == "film":
        fc = FiLMConfig(**ckpt.get("film_config", {}))
        model = build_film_model(model_config, fc)
    else:
        xc = CrossAttentionConfig(**ckpt.get("xattn_config", {}))
        model = build_cross_attention_model(model_config, xc)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# =========================================================================
# 1. FiLM: Gamma/Beta analysis per cancer type
# =========================================================================
def analyze_film_modulation(X, clinical_features, groups_arr, ctl, mc, out_dir):
    """For each cancer type, compute average gamma/beta from FiLM generators."""
    ckpt_dir = Path("results/training/film_xattn_comparison/film_resnet18/v001/checkpoints")
    if not ckpt_dir.exists():
        print("FiLM checkpoints not found, skipping")
        return

    all_gammas = {g: [] for g in CANCER_NAMES + ["Non-cancer"]}
    all_betas = {g: [] for g in CANCER_NAMES + ["Non-cancer"]}

    for fold_i in range(5):
        ckpt_path = ckpt_dir / f"fold_{fold_i}.pt"
        if not ckpt_path.exists():
            continue
        model = load_fold_model(ckpt_path, "film", mc)

        # Get gamma/beta for each sample
        with torch.no_grad():
            clin_t = torch.FloatTensor(clinical_features)
            for layer_idx, film_gen in enumerate([
                model.encoder.film1, model.encoder.film2,
                model.encoder.film3, model.encoder.film4,
            ]):
                gamma, beta = film_gen(clin_t)
                gamma = gamma.numpy()  # (N, C_layer)
                beta = beta.numpy()

                for group_name in CANCER_NAMES:
                    ct_idx = list(mc.cancer_types).index(group_name) if group_name in mc.cancer_types else -1
                    mask = ctl == ct_idx
                    if mask.sum() > 0:
                        all_gammas[group_name].append(gamma[mask].mean(axis=0).mean())
                        all_betas[group_name].append(beta[mask].mean(axis=0).mean())

                nc_mask = ctl < 0
                all_gammas["Non-cancer"].append(gamma[nc_mask].mean(axis=0).mean())
                all_betas["Non-cancer"].append(beta[nc_mask].mean(axis=0).mean())

    # Reshape into (n_groups, 4_layers)
    group_names = CANCER_NAMES + ["Non-cancer"]
    n_groups = len(group_names)
    gamma_matrix = np.zeros((n_groups, 4))
    beta_matrix = np.zeros((n_groups, 4))

    for gi, gn in enumerate(group_names):
        vals_g = all_gammas[gn]
        vals_b = all_betas[gn]
        if vals_g:
            # 5 folds × 4 layers = 20 values, reshape to (5, 4) and average over folds
            arr_g = np.array(vals_g).reshape(-1, 4)
            arr_b = np.array(vals_b).reshape(-1, 4)
            gamma_matrix[gi] = arr_g.mean(axis=0)
            beta_matrix[gi] = arr_b.mean(axis=0)

    # Plot heatmap
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("FiLM Modulation Analysis: How Clinical Variables Condition Each Layer",
                 fontsize=14, fontweight="bold", y=1.02)

    for ax, matrix, title, cmap in [
        (axes[0], gamma_matrix, "γ (Multiplicative Scale)", "RdBu_r"),
        (axes[1], beta_matrix, "β (Additive Shift)", "PiYG"),
    ]:
        # Center colormap around 1.0 for gamma, 0.0 for beta
        if "γ" in title:
            vmin, vmax = matrix.min(), matrix.max()
            center = 1.0
        else:
            vmin, vmax = matrix.min(), matrix.max()
            center = 0.0

        abs_max = max(abs(vmin - center), abs(vmax - center))
        im = ax.imshow(matrix, cmap=cmap, aspect="auto",
                       vmin=center - abs_max, vmax=center + abs_max)
        ax.set_xticks(range(4))
        ax.set_xticklabels(LAYER_NAMES, fontsize=9)
        ax.set_yticks(range(n_groups))
        ax.set_yticklabels(group_names, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")

        # Annotate values
        for i in range(n_groups):
            for j in range(4):
                ax.text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center",
                        fontsize=8, color="black")

        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    fig.savefig(out_dir / "xai_film_gamma_beta_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir / 'xai_film_gamma_beta_heatmap.png'}")
    return gamma_matrix, beta_matrix


# =========================================================================
# 2. FiLM: Sex-conditioned modulation
# =========================================================================
def analyze_film_sex_effect(X, clinical_features, groups_arr, ctl, mc, out_dir):
    """Compare FiLM gamma for male vs female patients per layer."""
    ckpt_dir = Path("results/training/film_xattn_comparison/film_resnet18/v001/checkpoints")
    if not ckpt_dir.exists():
        return

    sex_col = clinical_features[:, 1]  # sex_numeric: M=1, F=0
    male_mask = sex_col > 0.5
    female_mask = sex_col < 0.5

    male_gammas = {l: [] for l in range(4)}
    female_gammas = {l: [] for l in range(4)}

    for fold_i in range(5):
        ckpt_path = ckpt_dir / f"fold_{fold_i}.pt"
        if not ckpt_path.exists():
            continue
        model = load_fold_model(ckpt_path, "film", mc)

        with torch.no_grad():
            clin_t = torch.FloatTensor(clinical_features)
            for layer_idx, film_gen in enumerate([
                model.encoder.film1, model.encoder.film2,
                model.encoder.film3, model.encoder.film4,
            ]):
                gamma, _ = film_gen(clin_t)
                gamma = gamma.numpy()
                # Average over channels, keep per-sample
                male_gammas[layer_idx].append(gamma[male_mask].mean(axis=0))
                female_gammas[layer_idx].append(gamma[female_mask].mean(axis=0))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("FiLM Sex-Conditioned Modulation: Male vs Female γ Distribution per Channel",
                 fontsize=13, fontweight="bold")

    channel_sizes = [32, 64, 128, 256]
    for layer_idx, ax in enumerate(axes.flatten()):
        male_avg = np.mean(male_gammas[layer_idx], axis=0)
        female_avg = np.mean(female_gammas[layer_idx], axis=0)
        diff = male_avg - female_avg

        n_ch = channel_sizes[layer_idx]
        x = np.arange(n_ch)

        ax.bar(x, diff, color=np.where(diff > 0, "#4A90D9", "#E74C3C"),
               alpha=0.8, width=1.0)
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_title(f"Layer {layer_idx+1} ({n_ch} channels)", fontweight="bold")
        ax.set_xlabel("Channel Index")
        ax.set_ylabel("γ_male − γ_female")

        # Highlight top channels
        top_pos = np.argsort(diff)[-3:]
        top_neg = np.argsort(diff)[:3]
        for idx in top_pos:
            ax.annotate(f"ch{idx}", (idx, diff[idx]), fontsize=7, ha="center",
                       va="bottom", color="#2C3E50")
        for idx in top_neg:
            ax.annotate(f"ch{idx}", (idx, diff[idx]), fontsize=7, ha="center",
                       va="top", color="#2C3E50")

    plt.tight_layout()
    fig.savefig(out_dir / "xai_film_sex_modulation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir / 'xai_film_sex_modulation.png'}")


# =========================================================================
# 3. Cross-Attention: Attention weight analysis
# =========================================================================
def analyze_cross_attention_weights(X, clinical_features, groups_arr, ctl, mc, out_dir):
    """Analyze cross-attention weights per cancer type → which clinical var matters."""
    ckpt_dir = Path("results/training/film_xattn_comparison/xattn_resnet18/v001/checkpoints")
    if not ckpt_dir.exists():
        print("CrossAttn checkpoints not found, skipping")
        return

    group_names = CANCER_NAMES + ["Non-cancer"]
    # attn_weights: (N, 1, 3) → per-sample attention over 3 clinical vars
    attn_per_group = {g: [] for g in group_names}

    for fold_i in range(5):
        ckpt_path = ckpt_dir / f"fold_{fold_i}.pt"
        if not ckpt_path.exists():
            continue
        model = load_fold_model(ckpt_path, "xattn", mc)

        with torch.no_grad():
            spec_t = torch.FloatTensor(X)
            clin_t = torch.FloatTensor(clinical_features)

            # Process in batches
            batch_size = 64
            all_attn = []
            for start in range(0, len(X), batch_size):
                end = min(start + batch_size, len(X))
                out = model(spec_t[start:end], clin_t[start:end])
                # Last layer attention: (B, 1, 3)
                attn = out["attn_weights"][-1].squeeze(1).numpy()  # (B, 3)
                all_attn.append(attn)
            all_attn = np.concatenate(all_attn, axis=0)  # (N, 3)

            for group_name in CANCER_NAMES:
                ct_idx = list(mc.cancer_types).index(group_name) if group_name in mc.cancer_types else -1
                mask = ctl == ct_idx
                if mask.sum() > 0:
                    attn_per_group[group_name].append(all_attn[mask].mean(axis=0))

            nc_mask = ctl < 0
            attn_per_group["Non-cancer"].append(all_attn[nc_mask].mean(axis=0))

    # Average over folds
    attn_matrix = np.zeros((len(group_names), 3))
    for gi, gn in enumerate(group_names):
        if attn_per_group[gn]:
            attn_matrix[gi] = np.mean(attn_per_group[gn], axis=0)

    # Plot heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                    gridspec_kw={"width_ratios": [2, 1]})
    fig.suptitle("Cross-Attention XAI: Clinical Variable Importance per Cancer Type",
                 fontsize=14, fontweight="bold", y=1.02)

    # Heatmap
    im = ax1.imshow(attn_matrix, cmap="YlOrRd", aspect="auto", vmin=0)
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(CLINICAL_NAMES, fontsize=11, fontweight="bold")
    ax1.set_yticks(range(len(group_names)))
    ax1.set_yticklabels(group_names, fontsize=10)
    ax1.set_title("Attention Weights (higher = more important)", fontsize=11)

    for i in range(len(group_names)):
        for j in range(3):
            ax1.text(j, i, f"{attn_matrix[i,j]:.3f}", ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="white" if attn_matrix[i,j] > attn_matrix.max() * 0.6 else "black")

    plt.colorbar(im, ax=ax1, shrink=0.8)

    # Bar chart of per-variable average attention
    avg_per_var = attn_matrix[:len(CANCER_NAMES)].mean(axis=0)  # cancer only
    colors = ["#3498DB", "#E74C3C", "#2ECC71"]
    bars = ax2.barh(CLINICAL_NAMES, avg_per_var, color=colors, edgecolor="white", linewidth=2)
    ax2.set_xlabel("Mean Attention (cancer types)", fontsize=10)
    ax2.set_title("Overall Clinical Variable\nImportance", fontsize=11, fontweight="bold")
    for bar, val in zip(bars, avg_per_var):
        ax2.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=10, fontweight="bold")
    ax2.set_xlim(0, avg_per_var.max() * 1.3)

    plt.tight_layout()
    fig.savefig(out_dir / "xai_cross_attention_weights.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir / 'xai_cross_attention_weights.png'}")

    return attn_matrix


# =========================================================================
# 4. Comparison chart: Baseline vs FiLM vs CrossAttn
# =========================================================================
def plot_comparison_chart(out_dir):
    """Compare metrics across models."""
    # Load summaries
    results = {}
    paths = {
        "FiLM": Path("results/training/film_xattn_comparison/film_resnet18/v001/training_summary.json"),
        "CrossAttn": Path("results/training/film_xattn_comparison/xattn_resnet18/v001/training_summary.json"),
    }

    for name, path in paths.items():
        if path.exists():
            import json
            with open(path) as f:
                summary = json.load(f)
            results[name] = summary["metrics"]

    # Add baseline from memory (Phase W-fus SERS only)
    results["Baseline\n(ResNet18)"] = {
        "val_s1_auc": 0.929,
        "val_s2_f1_macro": 0.756,
    }

    if not results:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Model Comparison: Baseline vs FiLM vs Cross-Attention Fusion",
                 fontsize=14, fontweight="bold")

    model_names = list(results.keys())
    colors = ["#95A5A6", "#3498DB", "#E74C3C"][:len(model_names)]

    # Stage 1: Detection AUC
    ax = axes[0]
    vals = [results[m].get("val_s1_auc", 0) for m in model_names]
    bars = ax.bar(model_names, vals, color=colors, edgecolor="white", linewidth=2, width=0.6)
    ax.set_ylabel("AUC", fontsize=11)
    ax.set_title("Stage 1: Cancer Detection AUC", fontsize=12, fontweight="bold")
    ax.set_ylim(0.85, 1.0)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

    # Stage 2: Type ID F1
    ax = axes[1]
    vals = [results[m].get("val_s2_f1_macro", 0) for m in model_names]
    bars = ax.bar(model_names, vals, color=colors, edgecolor="white", linewidth=2, width=0.6)
    ax.set_ylabel("F1 Macro", fontsize=11)
    ax.set_title("Stage 2: Cancer Type ID F1 Macro", fontsize=12, fontweight="bold")
    ax.set_ylim(0.6, 0.9)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{val:.3f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

    plt.tight_layout()
    fig.savefig(out_dir / "xai_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir / 'xai_model_comparison.png'}")


# =========================================================================
# 5. Cross-Attention: Per-cancer attention radar chart
# =========================================================================
def plot_attention_radar(attn_matrix, out_dir):
    """Radar/spider chart of attention distribution per cancer type."""
    if attn_matrix is None:
        return

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), subplot_kw=dict(polar=True))
    fig.suptitle("Cross-Attention: Clinical Variable Focus per Cancer Type",
                 fontsize=14, fontweight="bold", y=1.02)

    group_names = CANCER_NAMES + ["Non-cancer"]
    angles = np.linspace(0, 2 * np.pi, 3, endpoint=False).tolist()
    angles += angles[:1]  # close

    colors = ["#3498DB", "#E91E63", "#9C27B0", "#FF9800", "#4CAF50", "#795548", "#607D8B", "#95A5A6"]

    for idx, (ax, gn) in enumerate(zip(axes.flatten(), group_names)):
        values = attn_matrix[idx].tolist()
        values += values[:1]

        ax.fill(angles, values, alpha=0.25, color=colors[idx])
        ax.plot(angles, values, 'o-', linewidth=2, color=colors[idx], markersize=6)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(CLINICAL_NAMES, fontsize=9, fontweight="bold")
        ax.set_title(gn, fontsize=11, fontweight="bold", pad=15)
        ax.set_ylim(0, attn_matrix.max() * 1.2)

    # Hide unused subplot
    if len(group_names) < 8:
        axes.flatten()[-1].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_dir / "xai_attention_radar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir / 'xai_attention_radar.png'}")


def main():
    out_dir = Path("results/training/film_xattn_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    X, bl, ctl, groups_arr, clinical_features, mc = load_data()

    print("\n[1/5] FiLM gamma/beta heatmap...")
    gamma_matrix, beta_matrix = analyze_film_modulation(X, clinical_features, groups_arr, ctl, mc, out_dir)

    print("\n[2/5] FiLM sex-conditioned modulation...")
    analyze_film_sex_effect(X, clinical_features, groups_arr, ctl, mc, out_dir)

    print("\n[3/5] Cross-Attention weights analysis...")
    attn_matrix = analyze_cross_attention_weights(X, clinical_features, groups_arr, ctl, mc, out_dir)

    print("\n[4/5] Model comparison chart...")
    plot_comparison_chart(out_dir)

    print("\n[5/5] Attention radar chart...")
    plot_attention_radar(attn_matrix, out_dir)

    print("\n" + "=" * 60)
    print("XAI Analysis Complete!")
    print(f"Output directory: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
