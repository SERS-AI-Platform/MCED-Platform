from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mapping_preprocessing_compare_core import PreprocessingComparison
from mapping_preprocessing_compare_outputs import METHODS, PLOT_METHODS, ComparisonTables

COLORS = {METHODS[0]: "#D95F02", METHODS[1]: "#2C7FB8", METHODS[2]: "#2C7FB8"}


def _zscore(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean()
    scale = float(values.std())
    return centered / scale if scale > 1e-12 else centered


def _save(fig: plt.Figure, out: Path, stem: str) -> None:
    fig.savefig(out / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(out / f"{stem}.pdf", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_shape_comparison(
    out: Path, grid: np.ndarray, results: list[PreprocessingComparison], groups: tuple[str, ...]
) -> None:
    fig, axes = plt.subplots(len(groups), 2, figsize=(14, 9), sharex="col")
    axes_array = np.atleast_2d(axes)
    for row_axes, group in zip(axes_array, groups, strict=True):
        selected = [result for result in results if result.subject.group == group]
        changed = np.vstack([result.raw_average for result in selected]).mean(axis=0)
        transformed = np.vstack(
            [result.changed_after_previous_transform for result in selected]
        ).mean(axis=0)
        previous = np.vstack([result.legacy_average for result in selected]).mean(axis=0)
        row_axes[0].plot(grid, _zscore(changed), color=COLORS[METHODS[0]], lw=1.3, label="Changed raw QC average")
        row_axes[0].plot(grid, _zscore(previous), color=COLORS[METHODS[2]], lw=1.2, label="Previous production preprocessing")
        row_axes[1].plot(grid, _zscore(transformed), color="#F39C12", lw=1.3, label="Changed average after previous transform")
        row_axes[1].plot(grid, _zscore(previous), color=COLORS[METHODS[2]], lw=1.2, label="Previous production preprocessing")
        for axis in row_axes:
            axis.set_ylabel("within-curve z-score")
            axis.set_title(f"{group} (n={len(selected)})", loc="left", fontsize=11)
            axis.grid(alpha=0.18)
            axis.legend(frameon=False, fontsize=8)
    axes_array[-1, 0].set_xlabel("Raman shift (cm$^{-1}$)")
    axes_array[-1, 1].set_xlabel("Raman shift (cm$^{-1}$)")
    axes_array[0, 0].set_title("Native outputs: raw scale vs SNV scale", loc="left", fontsize=11)
    axes_array[0, 1].set_title("Diagnostic: same previous transform applied to changed average", loc="left", fontsize=11)
    fig.suptitle("Per-group preprocessing shape comparison", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, out, "fig01_preprocessing_shape_comparison")


def plot_noise_and_peaks(
    out: Path, tables: ComparisonTables, groups: tuple[str, ...]
) -> None:
    rows = list(tables.subject_rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
    for method_index, method in enumerate(PLOT_METHODS):
        noise_key = "changed_average_hf_noise" if method == PLOT_METHODS[0] else "previous_average_hf_noise"
        peak_key = "changed_reproducible_peak_count" if method == PLOT_METHODS[0] else "previous_reproducible_peak_count"
        noise_data = [[float(row[noise_key]) for row in rows if row["group"] == group] for group in groups]
        peak_data = [[float(row[peak_key]) for row in rows if row["group"] == group] for group in groups]
        positions = np.arange(len(groups)) + (method_index - 0.5) * 0.28
        axes[0].boxplot(noise_data, positions=positions, widths=0.24, patch_artist=True, showfliers=False, boxprops={"facecolor": COLORS[method], "alpha": 0.7})
        axes[1].boxplot(peak_data, positions=positions, widths=0.24, patch_artist=True, showfliers=False, boxprops={"facecolor": COLORS[method], "alpha": 0.7})
    for axis, ylabel, title in zip(axes, ("HF noise scale", "reproducible peak count"), ("Average-spectrum noise", "Peak repeatability"), strict=True):
        axis.set_xticks(np.arange(len(groups)), groups)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.18)
    axes[0].legend([plt.Line2D([0], [0], color=COLORS[m], lw=7) for m in PLOT_METHODS], ["Changed raw QC average", "Previous production preprocessing"], frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, out, "fig02_preprocessing_noise_peak_comparison")


def plot_correlation(out: Path, tables: ComparisonTables, groups: tuple[str, ...]) -> None:
    rows = list(tables.subject_rows)
    values = [[float(row["changed_previous_shape_correlation"]) for row in rows if row["group"] == group] for group in groups]
    fig, axis = plt.subplots(figsize=(9, 5.5))
    axis.boxplot(values, positions=np.arange(len(groups)), widths=0.5, patch_artist=True, showfliers=False, boxprops={"facecolor": "#7A5195", "alpha": 0.7})
    axis.set_xticks(np.arange(len(groups)), groups)
    axis.set_ylabel("Pearson correlation")
    axis.set_title("Per-subject shape correlation: changed vs previous preprocessing")
    axis.set_ylim(-0.05, 1.05)
    axis.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    _save(fig, out, "fig03_preprocessing_subject_correlation")


def write_plots(
    out: Path,
    grid: np.ndarray,
    results: list[PreprocessingComparison],
    tables: ComparisonTables,
    groups: tuple[str, ...],
) -> None:
    plot_shape_comparison(out, grid, results, groups)
    plot_noise_and_peaks(out, tables, groups)
    plot_correlation(out, tables, groups)
