from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .signal_noise_types import SignalNoiseDiagnostics

BLUE = "#2C7FB8"
ORANGE = "#D95F02"
INK = "#252525"
GREY = "#8A8A8A"
LIGHT_GREY = "#E5E5E5"


def write_lot_variance_figure(
    grid: np.ndarray,
    diagnostics: SignalNoiseDiagnostics,
    directory: Path,
) -> None:
    variance = diagnostics.lot_variance
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    medians = [
        np.median(variance.patient_fraction),
        np.median(variance.lot_fraction),
        np.median(variance.residual_fraction),
    ]
    axes[0, 0].bar(
        ("Patient", "Lot", "Spot/residual"),
        medians,
        color=(BLUE, ORANGE, GREY),
        edgecolor=INK,
        linewidth=0.6,
    )
    axes[0, 0].set(
        title="Median variance fraction across Raman shifts",
        ylabel="Fraction of total variance",
        ylim=(0, 1),
    )
    matrix = np.vstack(
        (variance.patient_fraction, variance.lot_fraction, variance.residual_fraction)
    )
    image = axes[0, 1].imshow(
        matrix,
        aspect="auto",
        extent=(grid.min(), grid.max(), 2.5, -0.5),
        cmap="Blues",
        vmin=0,
        vmax=1,
    )
    axes[0, 1].set_yticks((0, 1, 2), ("Patient", "Lot", "Spot/residual"))
    axes[0, 1].set(title="Variance source by Raman shift", xlabel=r"Raman shift (cm$^{-1}$)")
    fig.colorbar(image, ax=axes[0, 1], label="Variance fraction")
    technical = variance.lot_fraction + variance.residual_fraction
    axes[1, 0].plot(grid, technical, color=ORANGE, linewidth=1.2)
    axes[1, 0].axhline(
        np.median(technical),
        color=INK,
        linestyle="--",
        linewidth=1,
        label=f"Median={np.median(technical):.3f}",
    )
    axes[1, 0].set(
        title="Total technical variance",
        xlabel=r"Raman shift (cm$^{-1}$)",
        ylabel="Lot + spot/residual fraction",
        ylim=(0, 1),
    )
    axes[1, 0].legend(frameon=False)
    selected = sorted(
        diagnostics.important_peaks, key=lambda row: row.legacy_importance, reverse=True
    )[:12][::-1]
    indices = [int(np.argmin(np.abs(grid - row.shift_cm_1))) for row in selected]
    axes[1, 1].barh(
        np.arange(len(selected)), technical[indices], color=ORANGE, edgecolor=INK, linewidth=0.4
    )
    axes[1, 1].set_yticks(np.arange(len(selected)), [row.peak_name for row in selected])
    axes[1, 1].set(
        title="Technical fraction at legacy LR peaks", xlabel="Variance fraction", xlim=(0, 1)
    )
    for axis in axes.flat:
        axis.grid(axis="y", color=LIGHT_GREY, linewidth=0.8)
    fig.suptitle("Powder five-lot variance decomposition", fontsize=15)
    fig.text(
        0.5,
        0.95,
        "Balanced crossed design: 5 patients × 5 powder lots × 5 replicate spots; patient–lot interaction is retained in residual",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for extension in ("png", "pdf"):
        fig.savefig(
            directory / f"fig21_powder_lot_variance_components.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)
