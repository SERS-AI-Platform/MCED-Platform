from __future__ import annotations

"""Peak-difference analysis, confusion-pair summaries, and intensity profiles."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from ._common import (
    DEFAULT_CANCER_GROUPS,
    PAPER_ANNOTATION_SIZE,
    PAPER_LEGEND_SIZE,
    PAPER_LINEWIDTH,
    annotate_peak_labels,
    apply_publication_style,
    extract_feature_columns,
    feature_axis_from_names,
    logger,
    save_figure,
)
from .spectra import build_mean_spectrum_profile


# ===================================================================
# Peak-difference plots (shared core)
# ===================================================================

def _plot_peak_difference_core(
    spectra_df: pd.DataFrame,
    output_path: Path,
    target_mask: np.ndarray,
    reference_mask: np.ndarray,
    target_label: str,
    reference_label: str,
    feature_prefix: str = "x_",
    top_k: int = 15,
    title: str = "Peak Difference",
    target_color: str = "#c53030",
    reference_color: str = "#2b6cb0",
) -> pd.DataFrame:
    """Shared implementation for cancer-vs-noncancer and group-vs-reference plots."""
    feature_cols = extract_feature_columns(spectra_df, prefix=feature_prefix)
    if not feature_cols:
        raise ValueError(f"No spectral columns found with prefix '{feature_prefix}'")

    if target_mask.sum() == 0:
        raise ValueError(f"No samples found for target: {target_label}")
    if reference_mask.sum() == 0:
        raise ValueError(f"No samples found for reference: {reference_label}")

    X = spectra_df[feature_cols].to_numpy(dtype=float)
    axis = feature_axis_from_names(feature_cols)

    target_mean = X[target_mask].mean(axis=0)
    reference_mean = X[reference_mask].mean(axis=0)
    target_std = X[target_mask].std(axis=0)
    reference_std = X[reference_mask].std(axis=0)
    diff = target_mean - reference_mean

    top_k = min(top_k, len(feature_cols))
    top_idx = np.argsort(np.abs(diff))[-top_k:][::-1]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(14.5, 7.5))

    ax.plot(axis, reference_mean, color=reference_color,
            linewidth=PAPER_LINEWIDTH, label=reference_label)
    ax.fill_between(axis, reference_mean - reference_std,
                    reference_mean + reference_std,
                    color=reference_color, alpha=0.12)

    ax.plot(axis, target_mean, color=target_color,
            linewidth=PAPER_LINEWIDTH, label=target_label)
    ax.fill_between(axis, target_mean - target_std,
                    target_mean + target_std,
                    color=target_color, alpha=0.12)

    apply_publication_style(
        ax,
        xlabel="Raman Shift (cm\u207b\u00b9)",
        ylabel="Mean Intensity (a.u.)",
        title=title,
    )
    ax.grid(True, alpha=0.25)

    ax2 = ax.twinx()
    diff_label = f"{target_label} \u2212 {reference_label}"
    ax2.plot(axis, diff, color="#222222", linewidth=1.6, alpha=0.8, label=diff_label)
    ax2.axhline(0.0, color="#444444", linestyle="--", linewidth=1, alpha=0.6)
    apply_publication_style(ax2, ylabel="Difference (a.u.)")

    for idx in top_idx:
        wn = axis[idx]
        ax.axvline(wn, color="#ffb703", linestyle=":", linewidth=1, alpha=0.8)
        ax2.text(wn, diff[idx], f"{wn:.0f}",
                 fontsize=PAPER_ANNOTATION_SIZE, rotation=90,
                 va="bottom" if diff[idx] >= 0 else "top", ha="center")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
              loc="upper right", fontsize=PAPER_LEGEND_SIZE, frameon=True)

    save_figure(fig, output_path)

    return pd.DataFrame({
        "feature": [feature_cols[i] for i in top_idx],
        "wavenumber": axis[top_idx],
        "target_mean": target_mean[top_idx],
        "reference_mean": reference_mean[top_idx],
        "difference": diff[top_idx],
        "abs_difference": np.abs(diff[top_idx]),
    }).sort_values("abs_difference", ascending=False, ignore_index=True)


# ---------------------------------------------------------------------------
# Cancer vs Non-cancer
# ---------------------------------------------------------------------------

def plot_cancer_peak_difference(
    spectra_df: pd.DataFrame,
    output_path: Path,
    group_col: str = "group",
    feature_prefix: str = "x_",
    cancer_groups: Optional[Sequence[str]] = None,
    top_k: int = 15,
    title: str = "Cancer vs Non-cancer Peak Difference",
) -> pd.DataFrame:
    """Plot mean spectra for cancer vs non-cancer and highlight the largest differences."""
    if group_col not in spectra_df.columns:
        raise KeyError(f"Missing group column: {group_col}")

    cancer_groups = tuple(cancer_groups or DEFAULT_CANCER_GROUPS)
    is_cancer = spectra_df[group_col].isin(cancer_groups).to_numpy()

    return _plot_peak_difference_core(
        spectra_df=spectra_df,
        output_path=output_path,
        target_mask=is_cancer,
        reference_mask=~is_cancer,
        target_label="Cancer mean",
        reference_label="Non-cancer mean",
        feature_prefix=feature_prefix,
        top_k=top_k,
        title=title,
    )


# ---------------------------------------------------------------------------
# Target group vs Reference group(s)
# ---------------------------------------------------------------------------

def plot_group_peak_difference(
    spectra_df: pd.DataFrame,
    output_path: Path,
    target_group: str,
    reference_groups: Optional[Sequence[str]] = None,
    group_col: str = "group",
    feature_prefix: str = "x_",
    top_k: int = 15,
    title: Optional[str] = None,
    target_color: str = "#c53030",
    reference_color: str = "#2b6cb0",
) -> pd.DataFrame:
    """Plot mean spectra for one target group versus a reference set."""
    if group_col not in spectra_df.columns:
        raise KeyError(f"Missing group column: {group_col}")

    target_mask = (spectra_df[group_col].astype(str) == str(target_group)).to_numpy()
    if reference_groups is None:
        reference_mask = ~target_mask
        reference_label = "Reference mean"
    else:
        reference_mask = spectra_df[group_col].isin(reference_groups).to_numpy()
        if len(reference_groups) == 1:
            reference_label = f"{reference_groups[0]} mean"
        else:
            reference_label = "Other diagnoses mean"

    return _plot_peak_difference_core(
        spectra_df=spectra_df,
        output_path=output_path,
        target_mask=target_mask,
        reference_mask=reference_mask,
        target_label=f"{target_group} mean",
        reference_label=reference_label,
        feature_prefix=feature_prefix,
        top_k=top_k,
        title=title or f"{target_group} vs Reference Peak Difference",
        target_color=target_color,
        reference_color=reference_color,
    )


# ===================================================================
# Spectrum peak analysis
# ===================================================================

def summarize_spectrum_peaks(
    spectra: np.ndarray,
    feature_names: Optional[Sequence[str]] = None,
    top_k: int = 8,
    min_peak_distance: int = 8,
    prominence_ratio: float = 0.05,
) -> pd.DataFrame:
    """Summarize the strongest intensity peaks on a mean spectrum."""
    profile = build_mean_spectrum_profile(spectra, feature_names)
    signal = profile["mean_intensity"].to_numpy(dtype=float)
    std = profile["std_intensity"].to_numpy(dtype=float)
    axis = profile["wavenumber"].to_numpy(dtype=float)
    feature_values = profile["feature"].to_numpy()

    if top_k <= 0:
        raise ValueError("top_k must be >= 1")

    prominence = max(float(np.ptp(signal)) * float(prominence_ratio), 1e-12)
    peak_idx, properties = find_peaks(
        signal,
        distance=max(int(min_peak_distance), 1),
        prominence=prominence,
    )
    prominences = np.asarray(properties.get("prominences", np.full(len(peak_idx), np.nan)))

    if len(peak_idx) == 0:
        peak_idx = np.argsort(signal)[-min(int(top_k), len(signal)):][::-1]
        prominences = np.full(len(peak_idx), np.nan)
    else:
        order = np.argsort(signal[peak_idx])[::-1]
        peak_idx = peak_idx[order]
        prominences = prominences[order]

    peak_idx = peak_idx[:min(int(top_k), len(peak_idx))]
    prominences = prominences[:len(peak_idx)]

    return pd.DataFrame({
        "peak_rank": np.arange(1, len(peak_idx) + 1),
        "feature": feature_values[peak_idx],
        "wavenumber": axis[peak_idx],
        "peak_intensity": signal[peak_idx],
        "std_intensity": std[peak_idx],
        "prominence": prominences,
    }).sort_values("peak_rank", ignore_index=True)


# ---------------------------------------------------------------------------
# Peak intensity profile
# ---------------------------------------------------------------------------

def plot_peak_intensity_profile(
    spectra: np.ndarray,
    output_path: Path,
    feature_names: Optional[Sequence[str]] = None,
    title: str = "Peak Intensity Profile",
    color: str = "#c0392b",
    top_k: int = 8,
    min_peak_distance: int = 8,
    prominence_ratio: float = 0.05,
) -> pd.DataFrame:
    """Plot mean spectrum and annotate the most intense spectral peaks."""
    profile = build_mean_spectrum_profile(spectra, feature_names)
    peak_df = summarize_spectrum_peaks(
        spectra,
        feature_names=feature_names,
        top_k=top_k,
        min_peak_distance=min_peak_distance,
        prominence_ratio=prominence_ratio,
    )

    x = profile["wavenumber"].to_numpy()
    mean_y = profile["mean_intensity"].to_numpy()
    std_y = profile["std_intensity"].to_numpy()

    fig, ax = plt.subplots(figsize=(13, 5.8))
    ax.plot(x, mean_y, color=color, linewidth=PAPER_LINEWIDTH)
    ax.fill_between(x, mean_y - std_y, mean_y + std_y, color=color, alpha=0.14)
    ax.scatter(
        peak_df["wavenumber"], peak_df["peak_intensity"],
        s=42, color=color, edgecolor="white", linewidth=0.8, zorder=3,
        label=f"Top {len(peak_df)} peaks",
    )
    for row in peak_df.itertuples(index=False):
        ax.axvline(row.wavenumber, color=color, linestyle=":", linewidth=1, alpha=0.28)

    annotate_peak_labels(ax, peak_df, value_col="peak_intensity", color=color)
    apply_publication_style(
        ax,
        xlabel="Raman Shift (cm\u207b\u00b9)",
        ylabel="Mean Intensity (a.u.)",
        title=title,
    )
    ax.grid(True, alpha=0.22)
    ax.legend(fontsize=PAPER_LEGEND_SIZE, loc="upper right", frameon=True)
    save_figure(fig, output_path)
    return peak_df


# ---------------------------------------------------------------------------
# Peak intensity overview (horizontal bar per diagnosis)
# ---------------------------------------------------------------------------

def plot_peak_intensity_overview(
    peak_df: pd.DataFrame,
    output_path: Path,
    title: str = "Top Peak Intensity by Diagnosis",
) -> pd.DataFrame:
    """Plot the highest-intensity representative peak for each diagnosis."""
    required_cols = {"diagnosis", "peak_rank", "wavenumber", "peak_intensity"}
    missing = required_cols - set(peak_df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    top_df = (
        peak_df.loc[peak_df["peak_rank"] == 1, ["diagnosis", "wavenumber", "peak_intensity"]]
        .sort_values("peak_intensity", ascending=True, ignore_index=True)
    )
    if top_df.empty:
        raise ValueError("Peak overview requires at least one diagnosis peak")

    fig, ax = plt.subplots(figsize=(9.5, max(4.5, 0.55 * len(top_df) + 2)))
    ax.barh(top_df["diagnosis"], top_df["peak_intensity"], color="#4c78a8", alpha=0.88)
    for row in top_df.itertuples(index=False):
        ax.text(row.peak_intensity, row.diagnosis,
                f"  {row.wavenumber:.0f} cm\u207b\u00b9",
                va="center", ha="left", fontsize=PAPER_ANNOTATION_SIZE)

    apply_publication_style(
        ax, xlabel="Peak Intensity (a.u.)", ylabel="Diagnosis", title=title,
    )
    ax.grid(True, axis="x", alpha=0.22)
    save_figure(fig, output_path)
    return top_df


# ===================================================================
# Confusion-pair analysis
# ===================================================================

def summarize_confusion_pairs(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
) -> pd.DataFrame:
    """Summarize off-diagonal multiclass confusion pairs."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    rows: List[Dict[str, Any]] = []
    for class_idx in sorted(set(y_true.tolist())):
        class_mask = y_true == class_idx
        class_total = int(class_mask.sum())
        if class_total == 0:
            continue

        wrong_pred, counts = np.unique(
            y_pred[class_mask & (y_pred != class_idx)], return_counts=True,
        )
        order = np.argsort(counts)[::-1]
        for pred_idx, count in zip(wrong_pred[order], counts[order]):
            rows.append({
                "true_idx": int(class_idx),
                "predicted_idx": int(pred_idx),
                "true_class": class_names[class_idx],
                "predicted_class": class_names[pred_idx],
                "count": int(count),
                "true_total": class_total,
                "confusion_rate": float(count) / float(class_total),
            })

    columns = [
        "true_idx", "predicted_idx", "true_class", "predicted_class",
        "count", "true_total", "confusion_rate",
    ]
    if rows:
        return pd.DataFrame(rows).sort_values(
            ["count", "confusion_rate"], ascending=False, ignore_index=True,
        )
    return pd.DataFrame(columns=columns)


# ---------------------------------------------------------------------------
# Confusion summary bar chart
# ---------------------------------------------------------------------------

def plot_confusion_summary_bar(
    confusion_df: pd.DataFrame,
    output_path: Path,
    title: str = "Confused Classes",
    color: str = "#6c757d",
) -> pd.DataFrame:
    """Plot confusion-rate bars for a single true class."""
    if confusion_df.empty:
        raise ValueError("Confusion summary plot requires at least one confusion pair")

    plot_df = confusion_df.sort_values("confusion_rate", ascending=True, ignore_index=True)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.7 * len(plot_df) + 1.5)))
    ax.barh(plot_df["predicted_class"], plot_df["confusion_rate"],
            color=color, alpha=0.88)
    for row in plot_df.itertuples(index=False):
        ax.text(row.confusion_rate, row.predicted_class,
                f"  {row.count}/{row.true_total}",
                va="center", ha="left", fontsize=PAPER_ANNOTATION_SIZE)

    apply_publication_style(
        ax, xlabel="Misclassification Rate", ylabel="Predicted Class", title=title,
    )
    ax.set_xlim(0, max(0.05, plot_df["confusion_rate"].max() * 1.18))
    ax.grid(True, axis="x", alpha=0.22)
    save_figure(fig, output_path)
    return plot_df
