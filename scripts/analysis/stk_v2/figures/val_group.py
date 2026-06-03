#!/usr/bin/env python3
"""Val-group inference report figures for STK-V2."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix


def plot_val_group_summary(
    out_dir: Path,
    group_name: str,
    s1_final: np.ndarray,
    pred_counts: pd.Series,
    base_s1: dict[str, np.ndarray],
    model_names: list[str],
    dpi: int = 200,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_val = len(s1_final)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(s1_final, bins=30, color="#E53935", alpha=0.7, edgecolor="black")
    axes[0].axvline(0.5, color="black", ls="--", lw=1.5, label="Threshold 0.5")
    axes[0].set(xlabel="Cancer Probability", ylabel="Count",
                title=f"{group_name} - S1 Cancer Prob (n={n_val})")
    axes[0].legend()

    colors_map = {
        "PRO": "#E91E63", "BRE": "#F06292", "OVA": "#AB47BC",
        "LUN": "#42A5F5", "CRC": "#EF5350", "PAN": "#FFA726", "BLC": "#7E57C2",
    }
    colors = [colors_map.get(str(ct), "#999999") for ct in pred_counts.index]
    axes[1].pie(
        pred_counts.values,
        labels=[f"{ct}\n({count})" for ct, count in pred_counts.items()],
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
    )
    axes[1].set_title(f"{group_name} - Predicted Cancer Types")

    base_preds = np.column_stack([base_s1[m] for m in model_names])
    im = axes[2].imshow(base_preds.T, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    axes[2].set_yticks(range(len(model_names)))
    axes[2].set_yticklabels(model_names, fontsize=8)
    axes[2].set_xlabel("Sample")
    axes[2].set_title(f"{group_name} - Base Model Cancer Probs")
    fig.colorbar(im, ax=axes[2], label="P(cancer)")

    fig.tight_layout()
    out = out_dir / "val_group_summary.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def _draw_cm(ax, cm: np.ndarray, xlabels: list[str], ylabels: list[str], title: str, cmap: str):
    im = ax.imshow(cm, cmap=cmap, aspect="auto")
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(int(value)), ha="center", va="center",
                color="white" if value > cm.max(initial=0) * 0.5 else "black")
    ax.set_xticks(range(len(xlabels)))
    ax.set_yticks(range(len(ylabels)))
    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title(title)
    return im


def plot_val_group_confusion_matrices(
    out_dir: Path,
    group_name: str,
    s1_final: np.ndarray,
    s2_pred: np.ndarray,
    cancer_types: list[str] | tuple[str, ...],
    expected_type: str | None = None,
    threshold: float = 0.5,
    dpi: int = 200,
) -> tuple[Path, dict[str, float]]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cancer_types = list(cancer_types)
    n_val = len(s1_final)

    gt_binary = np.ones(n_val, dtype=int)
    pred_binary = (s1_final > threshold).astype(int)
    cm_s1 = confusion_matrix(gt_binary, pred_binary, labels=[0, 1])
    tn_s1, fp_s1, fn_s1, tp_s1 = cm_s1.ravel()

    metrics = {
        "s1_sensitivity": tp_s1 / (tp_s1 + fn_s1) if (tp_s1 + fn_s1) > 0 else 0.0,
        "s1_specificity": tn_s1 / (tn_s1 + fp_s1) if (tn_s1 + fp_s1) > 0 else 0.0,
        "s1_accuracy": (tp_s1 + tn_s1) / cm_s1.sum() if cm_s1.sum() > 0 else 0.0,
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _draw_cm(
        axes[0], cm_s1,
        ["Non-cancer", "Cancer"], ["Non-cancer", "Cancer"],
        f"{group_name} - Stage 1 CM (GT=Cancer, t={threshold:.1f})",
        "Blues",
    )

    if expected_type in cancer_types:
        gt_type_idx = cancer_types.index(expected_type)
        gt_s2 = np.full(n_val, gt_type_idx, dtype=int)
        cm_s2 = confusion_matrix(gt_s2, s2_pred, labels=list(range(len(cancer_types))))
        _draw_cm(
            axes[1], cm_s2,
            cancer_types, cancer_types,
            f"{group_name} - Stage 2 CM (GT={expected_type})",
            "OrRd",
        )
        correct = int((s2_pred == gt_type_idx).sum())
        tp_s2 = cm_s2[gt_type_idx, gt_type_idx]
        fn_s2 = cm_s2[gt_type_idx, :].sum() - tp_s2
        fp_s2 = cm_s2[:, gt_type_idx].sum() - tp_s2
        denom_spec = cm_s2.sum() - (tp_s2 + fn_s2)
        metrics.update({
            "s2_sensitivity": tp_s2 / (tp_s2 + fn_s2) if (tp_s2 + fn_s2) > 0 else 0.0,
            "s2_specificity": (denom_spec - fp_s2) / denom_spec if denom_spec > 0 else 0.0,
            "s2_accuracy": correct / n_val if n_val > 0 else 0.0,
            "s2_correct": float(correct),
        })
    else:
        axes[1].text(0.5, 0.5, f"No GT mapping for {group_name}",
                     ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_axis_off()
        axes[1].set_title("Stage 2 CM - N/A")

    fig.tight_layout()
    out = out_dir / "confusion_matrices.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out, metrics


def plot_meta_shap_s1(
    shap_dir: Path,
    group_name: str,
    model_names: list[str],
    shap_vals_s1: np.ndarray,
    dpi: int = 200,
) -> Path:
    shap_dir = Path(shap_dir)
    shap_dir.mkdir(parents=True, exist_ok=True)
    mean_abs = np.abs(shap_vals_s1).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(model_names)), mean_abs[order], color="#1976D2", alpha=0.85)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels([model_names[i] for i in order])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Meta-Learner SHAP - Stage 1 ({group_name})")
    ax.invert_yaxis()
    fig.tight_layout()
    out = shap_dir / "meta_shap_s1.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def plot_meta_shap_s2(
    shap_dir: Path,
    group_name: str,
    model_names: list[str],
    per_model_shap: np.ndarray,
    dpi: int = 200,
) -> Path:
    shap_dir = Path(shap_dir)
    shap_dir.mkdir(parents=True, exist_ok=True)
    order = np.argsort(per_model_shap)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(model_names)), per_model_shap[order], color="#E53935", alpha=0.85)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels([model_names[i] for i in order])
    ax.set_xlabel("Mean |SHAP value| (aggregated over classes)")
    ax.set_title(f"Meta-Learner SHAP - Stage 2 ({group_name})")
    ax.invert_yaxis()
    fig.tight_layout()
    out = shap_dir / "meta_shap_s2.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def plot_spectral_shap(
    shap_dir: Path,
    bm_name: str,
    wavenumbers: np.ndarray,
    channels: list[int],
    mean_shap: np.ndarray,
    dpi: int = 200,
) -> Path:
    shap_dir = Path(shap_dir)
    shap_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 5))

    if len(channels) == 1:
        ax.plot(wavenumbers, mean_shap, color="#1976D2", lw=1.2)
        ax.fill_between(wavenumbers, mean_shap, alpha=0.3, color="#1976D2")
        top_idx = np.argsort(mean_shap)[-10:]
        threshold = mean_shap.mean() * 2
        for idx in top_idx:
            if mean_shap[idx] > threshold:
                ax.annotate(f"{wavenumbers[idx]:.0f}", xy=(wavenumbers[idx], mean_shap[idx]),
                            fontsize=7, ha="center", va="bottom")
    else:
        for ci, channel_idx in enumerate(channels):
            ch_name = ["raw", "d1", "d2"][channel_idx]
            start = ci * len(wavenumbers)
            end = start + len(wavenumbers)
            ax.plot(wavenumbers, mean_shap[start:end], lw=1.2, label=ch_name)
        ax.legend()

    ax.set_xlabel("Wavenumber (cm^-1)")
    ax.set_ylabel("Mean |SHAP value|")
    ax.set_title(f"{bm_name} - Spectral SHAP")
    fig.tight_layout()
    out = shap_dir / f"spectral_shap_{bm_name}.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out
