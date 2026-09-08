from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .signal_noise_ablation import AblationResult
from .signal_noise_types import SignalNoiseDiagnostics

BLUE = "#2C7FB8"
ORANGE = "#D95F02"
GOLD = "#D9A21B"
INK = "#252525"
GREY = "#8A8A8A"
LIGHT_GREY = "#E5E5E5"
FEATURE_LABELS = {
    "full_spectrum": "Full spectrum",
    "common_stable": "Common stable",
    "powder_only": "Powder-only",
    "powder_only_excluded": "Powder-only excluded",
}


def _save(fig: plt.Figure, directory: Path, name: str) -> None:
    for extension in ("png", "pdf"):
        fig.savefig(directory / f"{name}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _metric_panel(axis: plt.Axes, rows: tuple[AblationResult, ...], title: str) -> None:
    positions = np.arange(len(rows))
    axis.bar(
        positions - 0.18,
        [row.metrics.roc_auc for row in rows],
        width=0.36,
        color=BLUE,
        label="AUROC",
    )
    axis.bar(
        positions + 0.18,
        [row.metrics.balanced_accuracy for row in rows],
        width=0.36,
        color=ORANGE,
        label="Balanced accuracy",
    )
    axis.set_xticks(
        positions, [FEATURE_LABELS[row.feature_set] for row in rows], rotation=18, ha="right"
    )
    axis.set(title=title, ylabel="OOF metric", ylim=(0, 1))
    axis.legend(frameon=False, fontsize=8)
    axis.grid(axis="y", color=LIGHT_GREY, linewidth=0.8)


def _confusion_panel(axis: plt.Axes, row: AblationResult, title: str) -> None:
    image = axis.imshow(row.confusion, cmap="Blues")
    for y, x in np.ndindex(row.confusion.shape):
        axis.text(x, y, str(row.confusion[y, x]), ha="center", va="center", color=INK)
    axis.set_xticks(np.arange(len(row.class_names)), row.class_names, rotation=18, ha="right")
    axis.set_yticks(np.arange(len(row.class_names)), row.class_names)
    axis.set(xlabel="Predicted", ylabel="True", title=title)
    image.set_clim(0, max(1, int(np.max(row.confusion))))


def write_ablation_figure(diagnostics: SignalNoiseDiagnostics, directory: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 10.0))
    _metric_panel(axes[0, 0], diagnostics.screening_ablation, "Cancer vs non-cancer")
    _metric_panel(
        axes[0, 1], diagnostics.three_group_ablation, "Control vs biopsy-negative vs cancer"
    )
    feature_counts = [np.median(row.feature_counts) for row in diagnostics.screening_ablation]
    axes[0, 2].bar(
        np.arange(len(feature_counts)),
        feature_counts,
        color=(BLUE, GOLD, ORANGE, GREY),
        edgecolor=INK,
        linewidth=0.6,
    )
    axes[0, 2].set_xticks(
        np.arange(len(feature_counts)),
        [FEATURE_LABELS[row.feature_set] for row in diagnostics.screening_ablation],
        rotation=18,
        ha="right",
    )
    axes[0, 2].set(title="Median selected features across outer folds", ylabel="Grid features")
    axes[0, 2].grid(axis="y", color=LIGHT_GREY, linewidth=0.8)
    binary_full = next(
        row for row in diagnostics.screening_ablation if row.feature_set == "full_spectrum"
    )
    binary_excluded = next(
        row for row in diagnostics.screening_ablation if row.feature_set == "powder_only_excluded"
    )
    three_full = next(
        row for row in diagnostics.three_group_ablation if row.feature_set == "full_spectrum"
    )
    _confusion_panel(axes[1, 0], binary_full, "Binary: full spectrum")
    _confusion_panel(axes[1, 1], binary_excluded, "Binary: powder-only excluded")
    _confusion_panel(axes[1, 2], three_full, "Three-class: full spectrum")
    fig.suptitle("Leakage-free powder-native LR peak ablation", fontsize=15)
    fig.text(
        0.5,
        0.95,
        "Peak masks are recomputed from training subjects inside each outer fold; exact matrices for all ablations are in CSV",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, directory, "fig22_peak_ablation_performance")


def _evidence_label(diagnostics: SignalNoiseDiagnostics) -> str:
    old = [
        row
        for row in diagnostics.repeatability.stage_rows
        if row.modality == "legacy_liquid" and row.stage == "snv"
    ]
    new = [
        row
        for row in diagnostics.repeatability.stage_rows
        if row.modality == "powder" and row.stage == "snv"
    ]
    lower_consensus = np.median([row.consensus_fraction for row in new]) < np.median(
        [row.consensus_fraction for row in old]
    )
    weakened_effect = (
        diagnostics.effect_summaries[0].sign_agreement < 0.80
        or diagnostics.effect_summaries[0].median_absolute_retention < 0.80
    )
    if lower_consensus and weakened_effect:
        return "Mixed mechanism: technical heterogeneity + class-signal alteration"
    if lower_consensus:
        return "Technical heterogeneity is the dominant current signal"
    if weakened_effect:
        return "Loss or reversal of biological contrast is the dominant current signal"
    return "No single degradation mechanism dominates"


def write_conclusion_figure(
    grid: np.ndarray,
    diagnostics: SignalNoiseDiagnostics,
    directory: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.0))
    category_colors = {
        "common_stable": BLUE,
        "powder_only": ORANGE,
        "liquid_only": GOLD,
        "neither_stable": GREY,
    }
    for category, color in category_colors.items():
        selected = diagnostics.landscape.categories == category
        axes[0, 0].scatter(
            diagnostics.landscape.powder_presence_rate[selected],
            diagnostics.landscape.powder_max_abs_effect[selected],
            color=color,
            s=10,
            alpha=0.35,
            label=category.replace("_", " "),
        )
    axes[0, 0].axvline(0.30, color=INK, linestyle="--", linewidth=1)
    axes[0, 0].set(
        title="Peak reproducibility versus clinical effect",
        xlabel="Powder consensus-subject rate",
        ylabel="Maximum |Hedges' g|",
    )
    axes[0, 0].legend(frameon=False, fontsize=8)
    for correct, color, marker, label in (
        (True, BLUE, "o", "Correct"),
        (False, ORANGE, "x", "Misclassified"),
    ):
        rows = [row for row in diagnostics.patient_links if row.native_screening_correct is correct]
        axes[0, 1].scatter(
            [row.powder_replicate_correlation for row in rows],
            [row.native_cancer_probability for row in rows],
            color=color,
            marker=marker,
            alpha=0.7,
            label=f"{label} (n={len(rows)})",
        )
    axes[0, 1].axhline(0.5, color=INK, linestyle="--", linewidth=1)
    axes[0, 1].set(
        title="Technical repeatability and LR decision",
        xlabel="Powder replicate correlation",
        ylabel="Native cancer probability",
    )
    axes[0, 1].legend(frameon=False)
    fractions = np.array([row.powder_consensus_fraction for row in diagnostics.patient_links])
    correctness = np.array([row.native_screening_correct for row in diagnostics.patient_links])
    bins = np.quantile(fractions, (0.0, 1 / 3, 2 / 3, 1.0))
    tertiles = np.digitize(fractions, bins[1:-1], right=True)
    rates = [1.0 - np.mean(correctness[tertiles == index]) for index in range(3)]
    axes[1, 0].bar(("Low", "Middle", "High"), rates, color=ORANGE, edgecolor=INK, linewidth=0.6)
    axes[1, 0].set(
        title="Misclassification by peak-consensus tertile",
        xlabel="Powder consensus fraction",
        ylabel="Error rate",
        ylim=(0, 1),
    )
    axes[1, 1].axis("off")
    axes[1, 1].text(0.0, 0.95, _evidence_label(diagnostics), fontsize=13, weight="bold", va="top")
    for index, conclusion in enumerate(diagnostics.conclusions, start=1):
        axes[1, 1].text(
            0.0,
            0.84 - 0.14 * (index - 1),
            f"{index}. {conclusion}",
            fontsize=10,
            va="top",
            wrap=True,
        )
    axes[1, 1].text(
        0.0,
        0.08,
        "Limitation: no same-run powder blank/reference is available; substrate background requires the planned randomized validation run.",
        fontsize=9,
        color=GREY,
        wrap=True,
    )
    for axis in axes.flat[:3]:
        axis.grid(color=LIGHT_GREY, linewidth=0.8)
    fig.suptitle(
        "Powder spectral change: noise, non-specific enhancement, or signal loss?", fontsize=15
    )
    fig.text(
        0.5,
        0.95,
        rf"Current paired clinical and five-lot evidence over {grid.min():.0f}–{grid.max():.0f} cm$^{{-1}}$",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, directory, "fig23_noise_vs_signal_conclusion")
