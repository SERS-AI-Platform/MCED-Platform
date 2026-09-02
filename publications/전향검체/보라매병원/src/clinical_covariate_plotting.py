from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure

GROUPS: Final = ("Control", "Biopsy-negative", "Cancer")
COLORS: Final = ("#2C7FB8", "#7A5195", "#D95F02")
INK: Final = "#252A31"
GRID: Final = "#D9DEE5"


@dataclass(frozen=True, slots=True)
class MetricSpec:
    column: str
    title: str
    ylabel: str
    log_scale: bool = False


@dataclass(frozen=True, slots=True)
class SignificanceComparison:
    left: int
    right: int
    symbol: str


def configure_style() -> None:
    """Apply a restrained publication figure style."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.8,
            "xtick.color": INK,
            "ytick.color": INK,
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def metric_arrays(data: pl.DataFrame, column: str) -> list[np.ndarray]:
    """Return non-missing group arrays in the fixed display order."""
    return [
        data.filter(pl.col("group") == group)[column].drop_nulls().to_numpy()
        for group in GROUPS
    ]


def draw_distribution(axis: Axes, data: pl.DataFrame, spec: MetricSpec) -> None:
    """Draw a boxplot with deterministic subject-level jitter."""
    values = metric_arrays(data, spec.column)
    positions = np.arange(1, len(GROUPS) + 1)
    boxes = axis.boxplot(
        values,
        positions=positions,
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": INK, "linewidth": 1.8},
        whiskerprops={"color": INK, "linewidth": 1.0},
        capprops={"color": INK, "linewidth": 1.0},
        boxprops={"color": INK, "linewidth": 1.0},
    )
    rng = np.random.default_rng(20260724)
    labels: list[str] = []
    for position, group, color, group_values, box in zip(
        positions, GROUPS, COLORS, values, boxes["boxes"], strict=True
    ):
        box.set_facecolor(color)
        box.set_alpha(0.24)
        jitter = rng.uniform(-0.16, 0.16, len(group_values))
        axis.scatter(
            np.full(len(group_values), position) + jitter,
            group_values,
            s=27,
            facecolor=color,
            edgecolor="white",
            linewidth=0.55,
            alpha=0.88,
            zorder=3,
        )
        labels.append(f"{group}\nn={len(group_values)}")

    axis.set_xticks(positions, labels)
    axis.set_ylabel(spec.ylabel)
    axis.set_title(spec.title, loc="left", fontweight="bold", pad=10)
    axis.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    if spec.log_scale:
        axis.set_yscale("log")


def add_significance_brackets(
    axis: Axes,
    comparisons: tuple[SignificanceComparison, ...],
    observed_maximum: float,
    *,
    log_scale: bool,
) -> None:
    """Draw non-overlapping pairwise significance brackets above all observations."""
    if log_scale:
        first_level = observed_maximum * 1.28
        levels = [first_level * 1.42**index for index in range(len(comparisons))]
        cap_levels = [level / 1.08 for level in levels]
        upper = levels[-1] * 1.3
    else:
        lower, _ = axis.get_ylim()
        span = observed_maximum - lower
        step = span * 0.075
        levels = [observed_maximum + step * (index + 1) for index in range(len(comparisons))]
        cap_levels = [level - step * 0.18 for level in levels]
        upper = levels[-1] + step * 0.75

    for comparison, level, cap_level in zip(
        comparisons,
        levels,
        cap_levels,
        strict=True,
    ):
        axis.plot(
            [comparison.left, comparison.left, comparison.right, comparison.right],
            [cap_level, level, level, cap_level],
            color=INK,
            linewidth=1.05,
            clip_on=False,
        )
        axis.text(
            (comparison.left + comparison.right) / 2,
            level,
            comparison.symbol,
            ha="center",
            va="bottom",
            color=INK,
            fontsize=11,
            fontweight="bold",
        )
    axis.set_ylim(top=upper)


def save_figure(figure: Figure, output_dir: Path, stem: str) -> None:
    """Export a figure as high-resolution PNG and vector PDF."""
    figure.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    figure.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)
