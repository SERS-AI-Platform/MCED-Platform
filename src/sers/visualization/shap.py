from __future__ import annotations

"""SHAP-based model explanation visualizations."""

from pathlib import Path
from typing import Any, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._common import (
    apply_publication_style,
    feature_axis_from_names,
    logger,
    normalize_shap_values,
    save_figure,
)

# ===================================================================
# SHAP computation
# ===================================================================

def compute_gradient_shap_values(
    model: Any,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    device: Any,
):
    """
    Compute SHAP values with GradientExplainer for torch models.

    The model should return:
    - binary: shape (N,) or (N, 1)
    - multiclass: shape (N, C)
    """
    try:
        import shap
        import torch
    except ImportError as exc:
        raise ImportError("shap and torch are required for SHAP computation") from exc

    X_background = np.asarray(X_background, dtype=np.float32)
    X_explain = np.asarray(X_explain, dtype=np.float32)

    explainer = shap.GradientExplainer(
        model,
        torch.as_tensor(X_background, dtype=torch.float32, device=device),
    )
    return explainer.shap_values(
        torch.as_tensor(X_explain, dtype=torch.float32, device=device),
    )


# ===================================================================
# SHAP spectrum profile builder
# ===================================================================

def build_shap_spectrum_profile(
    shap_values: Any,
    feature_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Build a full-spectrum SHAP profile ordered along the spectral axis."""
    shap_list = normalize_shap_values(shap_values)
    if len(shap_list) != 1:
        raise ValueError("SHAP spectrum profile expects a single SHAP array")

    sv = np.asarray(shap_list[0])
    n_features = sv.shape[1]
    if feature_names is None:
        feature_names = [f"x_{i}" for i in range(n_features)]

    axis = feature_axis_from_names(feature_names)
    return pd.DataFrame({
        "feature": list(feature_names),
        "wavenumber": axis,
        "mean_abs_shap": np.abs(sv).mean(axis=0),
        "mean_shap": sv.mean(axis=0),
    }).sort_values("wavenumber", ascending=True, ignore_index=True)


# ===================================================================
# SHAP importance summary
# ===================================================================

def summarize_shap_feature_importance(
    shap_values: Any,
    feature_names: Optional[Sequence[str]] = None,
    top_k: Optional[int] = None,
) -> pd.DataFrame:
    """Summarize SHAP feature importance as mean absolute contribution."""
    importance = build_shap_spectrum_profile(shap_values, feature_names)
    importance = importance.sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    if top_k is not None:
        importance = importance.head(int(top_k)).reset_index(drop=True)
    return importance


# ===================================================================
# Beeswarm summary plots
# ===================================================================

def _plot_shap_beeswarm(
    shap_values: Any,
    X_explain: np.ndarray,
    output_path: Path,
    feature_names: Optional[Sequence[str]] = None,
    top_k: int = 30,
    title: str = "SHAP Summary",
) -> np.ndarray:
    """Shared beeswarm implementation for binary / single-class SHAP."""
    try:
        import shap
    except ImportError as exc:
        raise ImportError("shap is required for SHAP visualization") from exc

    shap_list = normalize_shap_values(shap_values)
    if len(shap_list) != 1:
        raise ValueError("Beeswarm plot expects a single SHAP array")

    sv = np.asarray(shap_list[0])
    X_explain = np.asarray(X_explain)
    if X_explain.shape[0] == 0:
        raise ValueError("SHAP beeswarm requires at least one sample")
    if feature_names is None:
        feature_names = [f"x_{i}" for i in range(X_explain.shape[1])]

    mean_abs = np.abs(sv).mean(axis=0)
    top_k = min(top_k, len(mean_abs))
    top_idx = np.argsort(mean_abs)[-top_k:][::-1]

    from ._common import ensure_output_dir
    ensure_output_dir(output_path)

    shap.summary_plot(
        sv[:, top_idx],
        X_explain[:, top_idx],
        feature_names=[feature_names[i] for i in top_idx],
        show=False,
        max_display=top_k,
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {Path(output_path).name}")
    return sv


def plot_binary_shap_summary(
    shap_values: Any,
    X_explain: np.ndarray,
    output_path: Path,
    feature_names: Optional[Sequence[str]] = None,
    top_k: int = 30,
    title: str = "Binary SHAP Summary",
) -> np.ndarray:
    """Create a SHAP beeswarm summary plot for binary output."""
    return _plot_shap_beeswarm(
        shap_values, X_explain, output_path,
        feature_names=feature_names, top_k=top_k, title=title,
    )


def plot_class_shap_summary(
    shap_values: Any,
    X_explain: np.ndarray,
    output_path: Path,
    feature_names: Optional[Sequence[str]] = None,
    top_k: int = 20,
    title: str = "Class SHAP Summary",
) -> np.ndarray:
    """Create a SHAP beeswarm summary plot for a single diagnosis/class."""
    return _plot_shap_beeswarm(
        shap_values, X_explain, output_path,
        feature_names=feature_names, top_k=top_k, title=title,
    )


def plot_multiclass_shap_summary(
    shap_values: Any,
    X_explain: np.ndarray,
    class_names: Sequence[str],
    output_path: Path,
    feature_names: Optional[Sequence[str]] = None,
    top_k: int = 20,
    max_cols: int = 4,
    title: str = "Multiclass SHAP Summary",
) -> List[np.ndarray]:
    """Create one beeswarm summary subplot per class for multiclass SHAP values."""
    try:
        import shap
    except ImportError as exc:
        raise ImportError("shap is required for SHAP visualization") from exc

    shap_list = normalize_shap_values(shap_values)
    X_explain = np.asarray(X_explain)
    if feature_names is None:
        feature_names = [f"x_{i}" for i in range(X_explain.shape[1])]

    n_classes = min(len(shap_list), len(class_names))
    cols = min(max_cols, max(1, n_classes))
    rows = (n_classes + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    axes = np.atleast_1d(axes).flatten()

    for class_idx in range(n_classes):
        plt.sca(axes[class_idx])
        sv = np.asarray(shap_list[class_idx])
        mean_abs = np.abs(sv).mean(axis=0)
        top_k_class = min(top_k, len(mean_abs))
        top_idx = np.argsort(mean_abs)[-top_k_class:][::-1]
        shap.summary_plot(
            sv[:, top_idx],
            X_explain[:, top_idx],
            feature_names=[feature_names[i] for i in top_idx],
            show=False,
            max_display=top_k_class,
            plot_size=None,
        )
        axes[class_idx].set_title(str(class_names[class_idx]),
                                  fontsize=14, fontweight="bold")

    for idx in range(n_classes, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=16, fontweight="bold")
    save_figure(fig, output_path)
    return [np.asarray(v) for v in shap_list[:n_classes]]


# ===================================================================
# SHAP feature importance bar chart
# ===================================================================

def plot_shap_feature_importance_bar(
    importance_df: pd.DataFrame,
    output_path: Path,
    title: str = "SHAP Feature Importance",
    color: str = "#c0392b",
) -> None:
    """Plot mean absolute SHAP feature importance as a horizontal bar chart."""
    if importance_df.empty:
        raise ValueError("Importance dataframe is empty")

    data = importance_df.iloc[::-1].copy()
    labels = [
        f"{row.feature} ({row.wavenumber:.1f})"
        for row in data.itertuples(index=False)
    ]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(data))))
    ax.barh(labels, data["mean_abs_shap"], color=color, alpha=0.85)
    apply_publication_style(
        ax,
        xlabel="Mean |SHAP value|",
        ylabel="Feature",
        title=title,
    )
    ax.grid(True, axis="x", alpha=0.25)
    save_figure(fig, output_path)


# ===================================================================
# SHAP spectrum plots (magnitude / signed)
# ===================================================================

def _plot_shap_spectrum(
    shap_values: Any,
    output_path: Path,
    feature_names: Optional[Sequence[str]],
    title: str,
    signed: bool,
    color: str = "#c0392b",
    color_negative: str = "#2980b9",
) -> pd.DataFrame:
    """Shared implementation for magnitude and signed SHAP spectrum plots."""
    profile = build_shap_spectrum_profile(shap_values, feature_names)
    x = profile["wavenumber"].to_numpy()

    fig, ax = plt.subplots(figsize=(13, 5))

    if signed:
        y = profile["mean_shap"].to_numpy()
        ax.plot(x, y, color="#222222", linewidth=1.4)
        ax.fill_between(x, 0, y, where=y >= 0, color=color, alpha=0.2)
        ax.fill_between(x, 0, y, where=y < 0, color=color_negative, alpha=0.2)
        ax.axhline(0, color="#444444", linestyle="--", linewidth=1)
        ylabel = "Mean SHAP value"
    else:
        y = profile["mean_abs_shap"].to_numpy()
        ax.plot(x, y, color=color, linewidth=1.8)
        ax.fill_between(x, 0, y, color=color, alpha=0.15)
        ylabel = "Mean |SHAP value|"

    apply_publication_style(
        ax,
        xlabel="Raman Shift (cm\u207b\u00b9)",
        ylabel=ylabel,
        title=title,
    )
    ax.grid(True, alpha=0.25)
    save_figure(fig, output_path)
    return profile


def plot_shap_mean_magnitude_spectrum(
    shap_values: Any,
    output_path: Path,
    feature_names: Optional[Sequence[str]] = None,
    title: str = "Mean |SHAP| Spectrum",
    color: str = "#c0392b",
) -> pd.DataFrame:
    """Plot mean absolute SHAP across the spectral axis."""
    return _plot_shap_spectrum(
        shap_values, output_path, feature_names,
        title=title, signed=False, color=color,
    )


def plot_shap_mean_signed_spectrum(
    shap_values: Any,
    output_path: Path,
    feature_names: Optional[Sequence[str]] = None,
    title: str = "Mean SHAP Spectrum",
    color_positive: str = "#c0392b",
    color_negative: str = "#2980b9",
) -> pd.DataFrame:
    """Plot signed mean SHAP across the spectral axis."""
    return _plot_shap_spectrum(
        shap_values, output_path, feature_names,
        title=title, signed=True,
        color=color_positive, color_negative=color_negative,
    )
