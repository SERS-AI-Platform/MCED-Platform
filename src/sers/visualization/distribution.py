from __future__ import annotations

"""Sample/spectra distribution and QC variance plots."""

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ._common import (
    PAPER_LABEL_SIZE,
    PAPER_LEGEND_SIZE,
    PAPER_TICK_SIZE,
    apply_publication_style,
    logger,
    save_figure,
)


# ---------------------------------------------------------------------------
# Sample distribution pie chart
# ---------------------------------------------------------------------------

def plot_sample_distribution_pie(
    group_stats_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Plot sample distribution pie chart.

    Saves: ``sample_distribution_pie.png``
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    groups = group_stats_df["group"].values
    n_samples = group_stats_df["n_samples"].values
    colors = plt.cm.Set3(np.linspace(0, 1, len(groups)))

    ax.pie(
        n_samples,
        labels=groups,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        textprops={"fontsize": PAPER_TICK_SIZE, "fontweight": "bold"},
    )
    apply_publication_style(ax, title="Sample Distribution by Group")
    save_figure(fig, Path(output_dir) / "sample_distribution_pie.png")


# ---------------------------------------------------------------------------
# Spectra count bar chart
# ---------------------------------------------------------------------------

def plot_spectra_count_bar(
    group_stats_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Plot total spectra count bar chart.

    Saves: ``spectra_count_bar.png``
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    groups = group_stats_df["group"].values
    n_spectra = group_stats_df["n_spectra"].values
    x_pos = np.arange(len(groups))
    colors = plt.cm.Set3(np.linspace(0, 1, len(groups)))

    ax.bar(x_pos, n_spectra, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(groups, fontsize=PAPER_TICK_SIZE)

    for i, v in enumerate(n_spectra):
        ax.text(i, v + max(n_spectra) * 0.02, str(int(v)),
                ha="center", fontsize=PAPER_TICK_SIZE, fontweight="bold")

    apply_publication_style(
        ax,
        ylabel="Number of Spectra",
        title="Total Spectra by Group",
    )
    ax.grid(True, alpha=0.3, axis="y")
    save_figure(fig, Path(output_dir) / "spectra_count_bar.png")


# ---------------------------------------------------------------------------
# Replicate variance by group (3-panel: CV, correlation, scatter)
# ---------------------------------------------------------------------------

def plot_replicate_variance_by_group(
    variance_df: pd.DataFrame,
    output_dir: Path,
    cv_threshold: float = 15.0,
    groups: Optional[List[str]] = None,
) -> None:
    """
    Plot replicate variability per group (CV distribution, correlation, quality scatter).

    Saves one PNG per group to *output_dir*.
    """
    if groups is not None:
        variance_df = variance_df[variance_df["group"].isin(groups)].copy()

    for g in sorted(variance_df["group"].unique()):
        group_data = variance_df[variance_df["group"] == g]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # CV distribution
        axes[0].hist(group_data["mean_cv"], bins=20,
                     color="#56B4E9", alpha=0.7, edgecolor="black")
        axes[0].axvline(cv_threshold, color="red", linestyle="--", linewidth=2,
                        label=f"CV threshold ({cv_threshold}%)")
        apply_publication_style(axes[0], xlabel="Mean CV (%)", ylabel="Count",
                                title=f"{g} - CV Distribution")
        axes[0].legend(fontsize=PAPER_LEGEND_SIZE)
        axes[0].grid(True, alpha=0.3)

        # Correlation distribution
        axes[1].hist(group_data["mean_pairwise_correlation"], bins=20,
                     color="#56B4E9", alpha=0.7, edgecolor="black")
        apply_publication_style(axes[1], xlabel="Mean Pairwise Correlation",
                                ylabel="Count",
                                title=f"{g} - Correlation Distribution")
        axes[1].grid(True, alpha=0.3)

        # Quality scatter
        scatter_colors = ["red" if cv > cv_threshold else "green"
                          for cv in group_data["mean_cv"]]
        axes[2].scatter(group_data["mean_pairwise_correlation"],
                        group_data["mean_cv"], c=scatter_colors, alpha=0.7, s=50)
        axes[2].axhline(cv_threshold, color="red", linestyle="--", linewidth=2,
                        label=f"CV threshold ({cv_threshold}%)")
        apply_publication_style(axes[2], xlabel="Mean Pairwise Correlation",
                                ylabel="Mean CV (%)",
                                title=f"{g} - Quality Scatter")
        axes[2].legend(fontsize=PAPER_LEGEND_SIZE)
        axes[2].grid(True, alpha=0.3)

        save_figure(fig, Path(output_dir) / f"replicate_variance_by_group_{g}.png")


# ---------------------------------------------------------------------------
# Variance heatmap
# ---------------------------------------------------------------------------

def plot_variance_heatmap(
    variance_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Plot CV heatmap (samples x groups).

    Saves: ``variance_heatmap.png``
    """
    pivot_data = variance_df.pivot_table(
        index="sample_id", columns="group", values="mean_cv", aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(12, max(8, len(pivot_data) * 0.3)))
    sns.heatmap(
        pivot_data,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn_r",
        vmin=0,
        vmax=20,
        cbar_kws={"label": "Mean CV (%)"},
        linewidths=0.5,
        ax=ax,
    )
    apply_publication_style(
        ax,
        xlabel="Group",
        ylabel="Sample ID",
        title="Replicate Variability Heatmap (CV%)",
    )
    save_figure(fig, Path(output_dir) / "variance_heatmap.png")
