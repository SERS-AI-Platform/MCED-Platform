#!/usr/bin/env python3
"""Compare native raw spectrum with its fixed-grid representation.

This intentionally does not apply smoothing, baseline correction, or
normalization. It isolates the effect of fixed-grid standardization only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.sers.config import load_config  # noqa: E402
from src.sers.io import make_fixed_grid, read_spectrum  # noqa: E402

CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
SAMPLE_PATH = PROJECT_ROOT / "data" / "raw_data" / "9. Colorectal cancer (300개)" / "CRC 1_1.CSV"
OUT_DIR = PROJECT_ROOT / "results" / "figures"
OUT_PATH = OUT_DIR / "raw_vs_fixed_grid_representation.png"


def _set_axis_style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", color="#d7dce2", linewidth=0.8, alpha=0.8)
    ax.grid(True, axis="y", color="#edf0f3", linewidth=0.6, alpha=0.8)


def _nearest_values(
    x_raw: np.ndarray, y_raw: np.ndarray, grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    idx = np.searchsorted(x_raw, grid)
    idx = np.clip(idx, 1, len(x_raw) - 1)
    left = idx - 1
    right = idx
    use_left = np.abs(grid - x_raw[left]) <= np.abs(grid - x_raw[right])
    nearest_idx = np.where(use_left, left, right)
    return x_raw[nearest_idx], y_raw[nearest_idx]


def _boundary_lines(ax: plt.Axes, fixed_grid: np.ndarray) -> None:
    ax.axvline(400.0, color="#2a9d8f", linestyle="--", linewidth=1.0, alpha=0.75)
    ax.axvline(2200.0, color="#2a9d8f", linestyle="--", linewidth=1.0, alpha=0.75)
    ax.axvline(fixed_grid[0], color="#d95f02", linestyle="-", linewidth=1.2, alpha=0.9)
    ax.axvline(fixed_grid[-1], color="#d95f02", linestyle="-", linewidth=1.2, alpha=0.9)


def main() -> None:
    config = load_config(CONFIG_PATH)
    fixed_grid = make_fixed_grid(config)
    if fixed_grid is None:
        raise RuntimeError("preprocessing.fixed_grid is not configured.")

    sample_path = (
        SAMPLE_PATH
        if SAMPLE_PATH.exists()
        else next((PROJECT_ROOT / "data" / "raw_data").rglob("*.CSV"))
    )
    x_raw, y_raw = read_spectrum(sample_path)
    y_fixed = np.interp(fixed_grid, x_raw, y_raw)
    nearest_x, nearest_y = _nearest_values(x_raw, y_raw, fixed_grid)
    delta_y = y_fixed - nearest_y
    delta_x = fixed_grid - nearest_x

    raw_fp_mask = (x_raw >= 400) & (x_raw <= 2200)
    n_raw_fp = int(raw_fp_mask.sum())

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

    fig = plt.figure(figsize=(14, 9.2), facecolor="white")
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.0, 1.45, 1.05],
        width_ratios=[1.25, 1.0],
        hspace=0.60,
        wspace=0.28,
    )

    fig.suptitle(
        "Native raw spectrum vs fixed-grid representation",
        fontsize=17,
        fontweight="bold",
        x=0.04,
        y=0.98,
        ha="left",
    )
    fig.text(
        0.04,
        0.945,
        "Only the x-axis standardization step is shown here: no smoothing, baseline correction, or SNV normalization is applied.",
        fontsize=10,
        color="#4b5563",
    )
    fig.subplots_adjust(top=0.86)

    # A. Full raw range vs what the fixed grid represents.
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(x_raw, y_raw, color="#37474f", linewidth=1.0, alpha=0.82, label="native raw spectrum")
    ax0.plot(
        fixed_grid, y_fixed, color="#d95f02", linewidth=1.25, label="same signal on fixed grid"
    )
    ax0.axvspan(
        x_raw.min(),
        fixed_grid[0],
        color="#b0bec5",
        alpha=0.22,
        label="not represented by fixed grid",
    )
    ax0.axvspan(fixed_grid[-1], x_raw.max(), color="#b0bec5", alpha=0.22)
    ax0.axvspan(400.0, 2200.0, color="#2a9d8f", alpha=0.08, label="fingerprint window")
    _boundary_lines(ax0, fixed_grid)
    ax0.set_xlim(x_raw.min(), x_raw.max())
    ax0.set_ylabel("Raw intensity")
    ax0.set_title(
        f"A. Full raw spectrum: fixed grid represents the fingerprint region only ({sample_path.name})"
    )
    ax0.legend(loc="upper right", ncol=3, frameon=False, fontsize=9)
    _set_axis_style(ax0)

    # B. Fingerprint range overlay.
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(
        x_raw[raw_fp_mask],
        y_raw[raw_fp_mask],
        color="#37474f",
        linewidth=1.0,
        alpha=0.72,
        label=f"native raw axis ({n_raw_fp} pts in 400-2200)",
    )
    ax1.plot(
        fixed_grid,
        y_fixed,
        color="#d95f02",
        linewidth=1.25,
        label=f"fixed grid ({len(fixed_grid)} pts)",
    )
    ax1.scatter(
        fixed_grid[::16],
        y_fixed[::16],
        s=16,
        color="#d95f02",
        edgecolor="white",
        linewidth=0.4,
        zorder=3,
    )
    y0, y1 = ax1.get_ylim()
    ax1.vlines(
        x_raw[raw_fp_mask][::18],
        y0,
        y0 + (y1 - y0) * 0.055,
        color="#37474f",
        linewidth=0.75,
        alpha=0.50,
    )
    ax1.vlines(
        fixed_grid[::18],
        y0 + (y1 - y0) * 0.065,
        y0 + (y1 - y0) * 0.12,
        color="#d95f02",
        linewidth=0.75,
        alpha=0.85,
    )
    ax1.text(610, y0 + (y1 - y0) * 0.14, "bottom rug: native raw x", fontsize=8.5, color="#37474f")
    ax1.text(610, y0 + (y1 - y0) * 0.21, "upper rug: fixed grid x", fontsize=8.5, color="#d95f02")
    _boundary_lines(ax1, fixed_grid)
    ax1.set_xlim(390, 2210)
    ax1.set_xlabel("Raman shift (cm^-1)")
    ax1.set_ylabel("Raw intensity")
    ax1.set_title("B. Fingerprint region: native count is 933; configured target grid is 935")
    ax1.legend(loc="upper right", frameon=False, fontsize=9)
    _set_axis_style(ax1)

    # C. Tight peak zoom where grid-point displacement is visually obvious.
    ax2 = fig.add_subplot(gs[1, 1])
    zoom_min, zoom_max = 990.0, 1002.0
    zoom_raw = (x_raw >= zoom_min) & (x_raw <= zoom_max)
    zoom_grid = (fixed_grid >= zoom_min) & (fixed_grid <= zoom_max)
    ax2.plot(x_raw[zoom_raw], y_raw[zoom_raw], color="#37474f", linewidth=1.2, alpha=0.55)
    ax2.scatter(
        x_raw[zoom_raw],
        y_raw[zoom_raw],
        s=52,
        color="#37474f",
        edgecolor="white",
        linewidth=0.7,
        label="native raw points",
    )
    ax2.plot(fixed_grid[zoom_grid], y_fixed[zoom_grid], color="#d95f02", linewidth=1.15, alpha=0.90)
    ax2.scatter(
        fixed_grid[zoom_grid],
        y_fixed[zoom_grid],
        marker="s",
        s=52,
        color="#d95f02",
        edgecolor="white",
        linewidth=0.7,
        label="fixed-grid points",
        zorder=3,
    )
    for gx, gy, nx, ny in zip(
        fixed_grid[zoom_grid], y_fixed[zoom_grid], nearest_x[zoom_grid], nearest_y[zoom_grid]
    ):
        ax2.plot([nx, gx], [ny, gy], color="#8d99ae", linewidth=1.3, alpha=0.72)
    c_y0, c_y1 = ax2.get_ylim()
    c_base = c_y0 + (c_y1 - c_y0) * 0.04
    ax2.vlines(
        x_raw[zoom_raw],
        c_base,
        c_base + (c_y1 - c_y0) * 0.055,
        color="#37474f",
        linewidth=1.2,
        alpha=0.75,
    )
    ax2.vlines(
        fixed_grid[zoom_grid],
        c_base + (c_y1 - c_y0) * 0.075,
        c_base + (c_y1 - c_y0) * 0.13,
        color="#d95f02",
        linewidth=1.2,
        alpha=0.95,
    )
    ax2.text(
        990.25,
        c_base + (c_y1 - c_y0) * 0.155,
        "gray ticks = native x, orange ticks = fixed grid x",
        fontsize=8.2,
        color="#4b5563",
    )
    ax2.set_xlim(zoom_min, zoom_max)
    ax2.set_xlabel("Raman shift (cm^-1)")
    ax2.set_ylabel("Raw intensity")
    ax2.set_title("C. Tight peak zoom: fixed grid samples at shifted x positions")
    ax2.legend(loc="upper left", frameon=False, fontsize=9)
    _set_axis_style(ax2)

    # D. Difference from nearest raw point.
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.axhline(0, color="#263238", linewidth=1.0, alpha=0.75)
    ax3.plot(fixed_grid, delta_y, color="#d95f02", linewidth=1.0)
    ax3.fill_between(fixed_grid, 0, delta_y, color="#d95f02", alpha=0.18)
    ax3.set_xlim(390, 2210)
    ax3.set_xlabel("Fixed grid Raman shift (cm^-1)")
    ax3.set_ylabel("Interpolated - nearest raw intensity")
    ax3.set_title("D. Difference introduced by moving to fixed x positions")
    _set_axis_style(ax3)

    # E. Numeric summary.
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.axis("off")
    lines = [
        ("Native raw range", f"{x_raw.min():.2f}-{x_raw.max():.2f} cm^-1"),
        ("Native raw points", f"{len(x_raw):,}"),
        ("Native points in 400-2200", f"{n_raw_fp:,}"),
        ("Fixed grid range", f"{fixed_grid[0]:.2f}-{fixed_grid[-1]:.2f} cm^-1"),
        ("Configured target points", f"{len(fixed_grid):,}"),
        ("Target - native FP count", f"{len(fixed_grid) - n_raw_fp:+d} points"),
        ("Mean |x shift| to nearest raw", f"{np.abs(delta_x).mean():.3f} cm^-1"),
        ("Max |x shift| to nearest raw", f"{np.abs(delta_x).max():.3f} cm^-1"),
    ]
    ax4.text(
        0.0, 0.98, "E. What changes before preprocessing", fontsize=12, fontweight="bold", va="top"
    )
    y = 0.82
    for key, value in lines:
        ax4.text(0.02, y, key, fontsize=9.4, color="#4b5563", va="center")
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
        ax4.hlines(y - 0.045, 0.02, 0.98, color="#e5e7eb", linewidth=0.8)
        y -= 0.095
    ax4.text(
        0.02,
        0.02,
        "935 is the configured model grid, not a native raw point count.",
        fontsize=8.6,
        color="#d95f02",
        fontweight="bold",
        va="bottom",
    )

    fig.savefig(OUT_PATH, bbox_inches="tight", dpi=200)
    print(OUT_PATH)


if __name__ == "__main__":
    main()
