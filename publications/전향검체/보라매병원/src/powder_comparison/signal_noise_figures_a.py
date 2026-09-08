from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .data import PairedSpectra
from .signal_noise_types import SignalNoiseDiagnostics

BLUE = "#2C7FB8"
ORANGE = "#D95F02"
GOLD = "#D9A21B"
INK = "#252525"
GREY = "#8A8A8A"
LIGHT_GREY = "#E5E5E5"
STAGES = ("raw", "smoothed", "baseline_corrected", "snv")
STAGE_LABELS = ("Raw", "Smoothed", "Baseline corrected", "SNV")


def _save(fig: plt.Figure, directory: Path, name: str) -> None:
    for extension in ("png", "pdf"):
        fig.savefig(directory / f"{name}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def add_labeled_boxplot(
    axis: plt.Axes,
    values: tuple[np.ndarray, ...],
    tick_labels: tuple[str, ...],
) -> None:
    """Render labeled boxes using the Matplotlib 3.11 keyword contract."""
    axis.boxplot(
        values,
        tick_labels=tick_labels,
        patch_artist=True,
        boxprops={"facecolor": BLUE, "alpha": 0.35},
        medianprops={"color": INK},
    )


def _stage_values(
    diagnostics: SignalNoiseDiagnostics,
    modality: str,
    attribute: str,
) -> list[np.ndarray]:
    return [
        np.array(
            [
                float(getattr(row, attribute))
                for row in diagnostics.repeatability.stage_rows
                if row.modality == modality and row.stage == stage
            ]
        )
        for stage in STAGES
    ]


def write_peak_reproducibility_figure(
    spectra: PairedSpectra,
    diagnostics: SignalNoiseDiagnostics,
    directory: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    positions = np.arange(len(STAGES))
    for modality, color, offset, label in (
        ("legacy_liquid", BLUE, -0.18, "Liquid"),
        ("powder", ORANGE, 0.18, "Powder"),
    ):
        counts = _stage_values(diagnostics, modality, "median_peak_count")
        consensus = _stage_values(diagnostics, modality, "consensus_fraction")
        axes[0, 0].boxplot(
            counts,
            positions=positions + offset,
            widths=0.30,
            patch_artist=True,
            boxprops={"facecolor": color, "alpha": 0.55},
            medianprops={"color": INK},
            showfliers=False,
        )
        axes[0, 1].boxplot(
            consensus,
            positions=positions + offset,
            widths=0.30,
            patch_artist=True,
            boxprops={"facecolor": color, "alpha": 0.55},
            medianprops={"color": INK},
            showfliers=False,
        )
        axes[0, 0].plot([], [], color=color, linewidth=7, alpha=0.55, label=label)
    axes[0, 0].set(title="Replicate-level apparent peak count", ylabel="Peaks per replicate")
    axes[0, 1].set(title="Peak survival under the 4-of-5 rule", ylabel="Consensus fraction")
    axes[0, 0].legend(frameon=False, ncol=2)
    legacy_noise = _stage_values(diagnostics, "legacy_liquid", "median_noise_sigma")
    powder_noise = _stage_values(diagnostics, "powder", "median_noise_sigma")
    ratios = [np.median(new) / np.median(old) for old, new in zip(legacy_noise, powder_noise)]
    axes[1, 0].bar(positions, ratios, color=ORANGE, edgecolor=INK, linewidth=0.6)
    axes[1, 0].axhline(1.0, color=INK, linestyle="--", linewidth=1)
    axes[1, 0].set(title="Residual-noise scale ratio", ylabel="Powder / liquid median")
    add_labeled_boxplot(
        axes[1, 1],
        (spectra.legacy.replicate_correlation, spectra.powder.replicate_correlation),
        ("Liquid", "Powder"),
    )
    axes[1, 1].set(
        title="Within-patient spectral-shape repeatability", ylabel="Replicate correlation"
    )
    for axis in axes.flat:
        axis.grid(axis="y", color=LIGHT_GREY, linewidth=0.8)
    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.set_xticks(positions, STAGE_LABELS, rotation=15)
    fig.suptitle("Liquid and powder peak reproducibility", fontsize=15)
    fig.text(
        0.5,
        0.95,
        r"109 paired patients; five spectra per formulation; consensus = ≥4/5 within ±12 cm$^{-1}$",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, directory, "fig19_powder_peak_reproducibility")


def write_signal_retention_figure(
    diagnostics: SignalNoiseDiagnostics,
    directory: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.0))
    colors = (BLUE, GOLD, ORANGE)
    for old, new, color in zip(
        diagnostics.separation.legacy_pairs,
        diagnostics.separation.powder_pairs,
        colors,
        strict=True,
    ):
        axes[0, 0].scatter(
            old.hedges_g, new.hedges_g, s=8, alpha=0.18, color=color, label=old.label
        )
    limit = max(
        abs(value) for axis in (axes[0, 0].get_xlim(), axes[0, 0].get_ylim()) for value in axis
    )
    axes[0, 0].plot([-limit, limit], [-limit, limit], color=INK, linestyle="--", linewidth=1)
    axes[0, 0].set(
        xlabel="Liquid Hedges' g", ylabel="Powder Hedges' g", title="Subgroup effect preservation"
    )
    axes[0, 0].legend(frameon=False, fontsize=8)
    positions = np.arange(len(diagnostics.effect_summaries))
    sign = [row.sign_agreement for row in diagnostics.effect_summaries]
    retention = [row.median_absolute_retention for row in diagnostics.effect_summaries]
    axes[0, 1].bar(positions - 0.18, sign, width=0.36, color=BLUE, label="Direction agreement")
    axes[0, 1].bar(
        positions + 0.18, retention, width=0.36, color=ORANGE, label="Median |g| retention"
    )
    axes[0, 1].set_xticks(
        positions, [row.group_pair.replace(" - ", "\nvs ") for row in diagnostics.effect_summaries]
    )
    axes[0, 1].set(title="Effect retention summary", ylabel="Proportion / ratio")
    axes[0, 1].legend(frameon=False, fontsize=8)
    selected = sorted(
        diagnostics.important_peaks, key=lambda row: row.legacy_importance, reverse=True
    )[:12]
    selected = selected[::-1]
    y = np.arange(len(selected))
    axes[1, 0].hlines(
        y,
        [row.legacy_effect for row in selected],
        [row.powder_effect for row in selected],
        color=GREY,
    )
    axes[1, 0].scatter([row.legacy_effect for row in selected], y, color=BLUE, label="Liquid")
    axes[1, 0].scatter(
        [row.powder_effect for row in selected], y, color=ORANGE, marker="s", label="Powder"
    )
    axes[1, 0].axvline(0.0, color=INK, linestyle="--", linewidth=1)
    axes[1, 0].set_yticks(y, [row.peak_name for row in selected])
    axes[1, 0].set(title="Cancer–Control effect at legacy LR peaks", xlabel="Hedges' g")
    axes[1, 0].legend(frameon=False)
    category_colors = {
        "common_stable": BLUE,
        "powder_only": ORANGE,
        "liquid_only": GOLD,
        "neither_stable": GREY,
    }
    for category, color in category_colors.items():
        rows = [row for row in diagnostics.important_peaks if row.category == category]
        axes[1, 1].scatter(
            [row.legacy_importance for row in rows],
            [row.absolute_effect_retention for row in rows],
            color=color,
            marker="o" if category != "powder_only" else "s",
            alpha=0.8,
            label=category.replace("_", " "),
        )
    axes[1, 1].axhline(1.0, color=INK, linestyle="--", linewidth=1)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set(
        title="LR importance versus biological-effect retention",
        xlabel="Legacy LR relative importance",
        ylabel="|g powder| / |g liquid| (log scale)",
    )
    axes[1, 1].legend(frameon=False, fontsize=8)
    for axis in axes.flat:
        axis.grid(color=LIGHT_GREY, linewidth=0.8)
    fig.suptitle("Biological subgroup signal change after powder conversion", fontsize=15)
    fig.text(
        0.5,
        0.95,
        "Effect size is evaluated independently of model coefficients for all three clinical comparisons",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, directory, "fig20_biological_signal_retention")
