#!/usr/bin/env python3
"""Visualize the fixed wavenumber grid created before preprocessing."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.sers.config import load_config  # noqa: E402
from src.sers.io import make_fixed_grid, read_spectrum  # noqa: E402

CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
RAW_STATS_PATH = PROJECT_ROOT / "results" / "preprocessing" / "grid_analysis" / "raw_grid_stats.csv"
OUT_DIR = PROJECT_ROOT / "results" / "figures"
OUT_PATH = OUT_DIR / "preprocessing_grid_visualization.png"
SAMPLE_PATH = PROJECT_ROOT / "data" / "raw_data" / "5. Normal (100개)" / "NOR 1_1.CSV"


def _group_code(filename: str) -> str:
    match = re.match(r"^([A-Za-z.]+)", filename)
    return match.group(1).upper() if match else "UNK"


def _load_raw_grid_stats() -> pd.DataFrame:
    if not RAW_STATS_PATH.exists():
        raise FileNotFoundError(f"Missing raw grid stats: {RAW_STATS_PATH}")
    stats = pd.read_csv(RAW_STATS_PATH)
    stats = stats.dropna(subset=["x_min", "x_max", "n_points", "mean_step"]).copy()
    stats["group"] = stats["file"].map(_group_code)
    return stats


def _set_axis_style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", color="#d7dce2", linewidth=0.8, alpha=0.8)
    ax.grid(True, axis="y", color="#edf0f3", linewidth=0.6, alpha=0.8)


def _annotate_grid_boundaries(ax: plt.Axes, grid: np.ndarray, y: float | None = None) -> None:
    for x_val, label, color in [
        (400.0, "trim start 400", "#2a9d8f"),
        (grid[0], "grid start 402", "#d95f02"),
        (grid[-1], "grid end 2198", "#d95f02"),
        (2200.0, "trim end 2200", "#2a9d8f"),
    ]:
        ax.axvline(x_val, color=color, linestyle="--", linewidth=1.0, alpha=0.8)
        if y is not None:
            ax.text(
                x_val,
                y,
                label,
                rotation=90,
                va="bottom",
                ha="right",
                color=color,
                fontsize=8,
            )


def main() -> None:
    config = load_config(CONFIG_PATH)
    fixed_grid = make_fixed_grid(config)
    if fixed_grid is None:
        raise RuntimeError("preprocessing.fixed_grid is not configured.")

    raw_stats = _load_raw_grid_stats()
    raw_summary = (
        raw_stats.groupby("group")
        .agg(
            files=("file", "count"),
            x_min=("x_min", "min"),
            x_max=("x_max", "max"),
            n_points=("n_points", "median"),
            mean_step=("mean_step", "mean"),
        )
        .sort_index()
    )

    sample_path = (
        SAMPLE_PATH
        if SAMPLE_PATH.exists()
        else next((PROJECT_ROOT / "data" / "raw_data").rglob("*.CSV"))
    )
    x_raw, y_raw = read_spectrum(sample_path)
    y_on_grid = np.interp(fixed_grid, x_raw, y_raw)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 160,
        }
    )

    fig = plt.figure(figsize=(14, 9), facecolor="white")
    grid_spec = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.05, 1.45, 1.05],
        width_ratios=[1.25, 1.0],
        hspace=0.62,
        wspace=0.26,
    )

    fig.suptitle(
        "SERS-AI grid selection before preprocessing",
        fontsize=17,
        fontweight="bold",
        x=0.04,
        y=0.98,
        ha="left",
    )
    fig.text(
        0.04,
        0.945,
        "Raw spectra keep their native Raman shift axis; the pipeline defines a fixed target grid first, then preprocessing uses that grid as the model feature axis.",
        fontsize=10,
        color="#4b5563",
    )
    fig.subplots_adjust(top=0.84)

    # Panel A: raw coverage by group and fixed grid overlay.
    ax0 = fig.add_subplot(grid_spec[0, :])
    group_order = list(raw_summary.index)
    y_positions = np.arange(len(group_order))
    for i, group in enumerate(group_order):
        row = raw_summary.loc[group]
        ax0.hlines(i, row["x_min"], row["x_max"], color="#7f8c8d", linewidth=3.0, alpha=0.55)
        ax0.plot([row["x_min"], row["x_max"]], [i, i], "|", color="#34495e", markersize=10)
    fixed_y = len(group_order) + 0.45
    ax0.hlines(fixed_y, fixed_grid[0], fixed_grid[-1], color="#d95f02", linewidth=5.0)
    ax0.plot(
        fixed_grid[::25],
        np.full_like(fixed_grid[::25], fixed_y),
        "|",
        color="#d95f02",
        markersize=7,
    )
    ax0.axvspan(400, 2200, color="#2a9d8f", alpha=0.08, label="fingerprint trim window")
    ax0.axvline(400.0, color="#2a9d8f", linestyle="--", linewidth=1.0, alpha=0.8)
    ax0.axvline(2200.0, color="#2a9d8f", linestyle="--", linewidth=1.0, alpha=0.8)
    ax0.axvline(fixed_grid[0], color="#d95f02", linestyle="--", linewidth=1.0, alpha=0.8)
    ax0.axvline(fixed_grid[-1], color="#d95f02", linestyle="--", linewidth=1.0, alpha=0.8)
    ax0.text(
        fixed_grid[0],
        -0.55,
        "grid start 402",
        rotation=90,
        va="bottom",
        ha="right",
        color="#d95f02",
        fontsize=8,
    )
    ax0.text(
        fixed_grid[-1],
        -0.55,
        "grid end 2198",
        rotation=90,
        va="bottom",
        ha="right",
        color="#d95f02",
        fontsize=8,
    )
    ax0.set_yticks(list(y_positions) + [fixed_y])
    ax0.set_yticklabels(group_order + ["FIXED"])
    ax0.set_xlim(0, 3350)
    ax0.set_ylim(-0.7, fixed_y + 1.0)
    ax0.set_xlabel("Raman shift (cm^-1)")
    ax0.set_title("A. Raw spectrum coverage vs fixed common grid")
    _set_axis_style(ax0)

    # Panel B: sample raw spectrum full range.
    ax1 = fig.add_subplot(grid_spec[1, 0])
    ax1.plot(x_raw, y_raw, color="#263238", linewidth=1.0, alpha=0.8, label="raw spectrum")
    ax1.axvspan(400, 2200, color="#2a9d8f", alpha=0.10)
    ax1.plot(
        fixed_grid,
        y_on_grid,
        color="#d95f02",
        linewidth=1.0,
        alpha=0.95,
        label="values on fixed grid",
    )
    y_low, y_high = ax1.get_ylim()
    rug_base = y_low + (y_high - y_low) * 0.04
    rug_top = y_low + (y_high - y_low) * 0.13
    ax1.vlines(fixed_grid[::18], rug_base, rug_top, color="#d95f02", linewidth=0.8, alpha=0.8)
    _annotate_grid_boundaries(ax1, fixed_grid)
    ax1.set_xlim(x_raw.min(), x_raw.max())
    ax1.set_xlabel("Raman shift (cm^-1)")
    ax1.set_ylabel("Raw intensity")
    ax1.set_title(f"B. Example raw spectrum with target grid rug ({sample_path.name})")
    ax1.legend(loc="upper right", frameon=False, fontsize=9)
    _set_axis_style(ax1)

    # Panel C: zoomed fingerprint region and interpolated grid values.
    ax2 = fig.add_subplot(grid_spec[1, 1])
    zoom_mask = (x_raw >= 380) & (x_raw <= 2220)
    ax2.plot(
        x_raw[zoom_mask],
        y_raw[zoom_mask],
        color="#455a64",
        linewidth=1.1,
        alpha=0.65,
        label="native raw axis",
    )
    ax2.plot(
        fixed_grid, y_on_grid, color="#d95f02", linewidth=1.4, label="fixed grid interpolation"
    )
    ax2.scatter(
        fixed_grid[::20],
        y_on_grid[::20],
        s=14,
        color="#d95f02",
        edgecolor="white",
        linewidth=0.4,
        zorder=3,
        label="sampled grid points",
    )
    _annotate_grid_boundaries(ax2, fixed_grid)
    ax2.set_xlim(380, 2220)
    ax2.set_xlabel("Raman shift (cm^-1)")
    ax2.set_ylabel("Raw intensity")
    ax2.set_title("C. Zoom: grid points inside the fingerprint window")
    ax2.legend(loc="upper right", frameon=False, fontsize=9)
    _set_axis_style(ax2)

    # Panel D: grid spacing comparison.
    ax3 = fig.add_subplot(grid_spec[2, 0])
    ax3.hist(raw_stats["mean_step"], bins=22, color="#5c6bc0", alpha=0.72, edgecolor="white")
    fixed_step = float(np.diff(fixed_grid).mean())
    ax3.axvline(
        fixed_step, color="#d95f02", linewidth=2.2, label=f"fixed grid step = {fixed_step:.4f}"
    )
    ax3.axvline(
        raw_stats["mean_step"].mean(),
        color="#263238",
        linewidth=1.8,
        linestyle="--",
        label=f"raw mean step = {raw_stats['mean_step'].mean():.4f}",
    )
    ax3.set_xlabel("Mean spacing between adjacent x values (cm^-1)")
    ax3.set_ylabel("Raw files")
    ax3.set_title("D. Native spacing vs fixed grid spacing")
    ax3.legend(loc="upper right", frameon=False, fontsize=9)
    _set_axis_style(ax3)

    # Panel E: compact numeric summary.
    ax4 = fig.add_subplot(grid_spec[2, 1])
    ax4.axis("off")
    lines = [
        ("Raw files analyzed", f"{len(raw_stats):,}"),
        ("Raw native points", f"{int(raw_stats['n_points'].median()):,} per file"),
        ("Raw x range", f"{raw_stats['x_min'].min():.2f}-{raw_stats['x_max'].max():.2f} cm^-1"),
        ("Preprocessing trim window", "400.00-2200.00 cm^-1"),
        ("Fixed grid range", f"{fixed_grid[0]:.2f}-{fixed_grid[-1]:.2f} cm^-1"),
        ("Fixed grid points", f"{len(fixed_grid):,}"),
        ("Feature handoff", "one intensity value / fixed grid point"),
    ]
    ax4.text(
        0.0, 0.98, "E. What is fixed before preprocessing", fontsize=12, fontweight="bold", va="top"
    )
    y = 0.80
    for key, value in lines:
        ax4.text(0.02, y, key, fontsize=9.5, color="#4b5563", va="center")
        ax4.text(
            0.98,
            y,
            value,
            fontsize=9.0,
            color="#111827",
            va="center",
            ha="right",
            fontweight="bold",
        )
        ax4.hlines(y - 0.055, 0.02, 0.98, color="#e5e7eb", linewidth=0.8)
        y -= 0.115

    fig.savefig(OUT_PATH, bbox_inches="tight", dpi=200)
    print(OUT_PATH)


if __name__ == "__main__":
    main()
