#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "httpx2[http2,brotli,zstd]>=0.28",
#     "matplotlib>=3.9",
#     "numpy>=2.0",
#     "psycopg2-binary>=2.9,<3",
#     "pydantic>=2,<3",
# ]
# ///
# ─── How to run ───
# 1. Install uv (if not installed):
#      curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Run directly (no venv, no pip install needed):
#      uv run scripts/visualization/aecd_subject_preprocessing_figures.py
# ──────────────────

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

import matplotlib
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.visualization.aecd_subject_preprocessing_data import (  # noqa: E402
    RANGE_CM1,
    SubjectSeries,
    load_calibrations,
    load_spectra,
    select_subject,
    write_manifest,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUTPUT_DIR: Final = Path("figures/preprocessing_modules/aecd_subject_real")
DPI: Final = 200
NAVY: Final = "#0a306d"
GREEN: Final = "#029567"
RED: Final = "#b42318"
GRAY: Final = "#9ca3af"
LIGHT: Final = "#d6e0ef"


def axes(title: str, ylabel: str) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=DPI)
    fig.suptitle(title, fontsize=22, weight="bold", color="#111827", y=0.97)
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.17, top=0.82)
    ax.set_xlabel("Raman shift (cm⁻¹)", fontsize=16, weight="bold")
    ax.set_ylabel(ylabel, fontsize=16, weight="bold")
    ax.tick_params(labelsize=12, colors="#374151")
    ax.grid(color=LIGHT, linewidth=0.55, alpha=0.75)
    for spine in ax.spines.values():
        spine.set_color(LIGHT)
    return fig, ax


def save(fig: plt.Figure, filename: str) -> None:
    fig.savefig(OUTPUT_DIR / filename, dpi=DPI, facecolor="white")
    plt.close(fig)


def render_spectra(series: SubjectSeries) -> None:
    count = len(series.rows)
    median_raw = np.median(series.raw_values, axis=0)
    fig, ax = axes("Raw Clinical Repeat Spectra", "Intensity")
    ax.plot(series.raw_grids[0], series.raw_values.T, color=GRAY, linewidth=0.45, alpha=0.16)
    ax.plot(series.raw_grids[0], median_raw, color=NAVY, linewidth=1.7, label="Repeat median")
    ax.legend(
        frameon=False,
        fontsize=12,
        title=f"AECD API real clinical repeats; de-identified subject; n={count}",
        title_fontsize=10,
    )
    save(fig, "01_raw_repeat_spectra.png")


def render_calibration(series: SubjectSeries, representative: int) -> None:
    fig, ax = axes("PS Standard-Material Calibration", "Intensity")
    raw_grid = series.raw_grids[representative]
    corrected_grid = series.corrected_grids[representative]
    intensity = series.raw_values[representative]
    raw_mask = (raw_grid >= 970) & (raw_grid <= 1035)
    corrected_mask = (corrected_grid >= 970) & (corrected_grid <= 1035)
    ax.plot(
        raw_grid[raw_mask], intensity[raw_mask], color=GRAY, linewidth=1.5, label="Observed axis"
    )
    ax.plot(
        corrected_grid[corrected_mask],
        intensity[corrected_mask],
        color=NAVY,
        linewidth=1.8,
        label="Corrected axis",
    )
    ax.axvline(1001.4, color=GREEN, linestyle="--", linewidth=1.3, label="1001.4 cm⁻¹ reference")
    ax.set_xlim(970, 1035)
    ax.legend(
        frameon=False,
        fontsize=11,
        title=f"x_corrected = x_observed − global_shift ({series.shifts[representative]:+.3f} cm⁻¹)",
        title_fontsize=10,
    )
    save(fig, "02_ps_wavenumber_calibration.png")


def render_interpolation(series: SubjectSeries, representative: int) -> None:
    fig, ax = axes("Common-Grid Linear Interpolation", "Intensity")
    grid = series.corrected_grids[representative]
    intensity = series.raw_values[representative]
    mask = (grid >= RANGE_CM1[0]) & (grid <= RANGE_CM1[1])
    ax.plot(
        grid[mask],
        intensity[mask],
        color=GRAY,
        linewidth=1.0,
        label=f"Corrected native grid ({int(np.count_nonzero(mask))} points)",
    )
    ax.plot(
        series.common_grid,
        series.aligned_values[representative],
        color=NAVY,
        linewidth=1.3,
        label=f"400–2200 cm⁻¹ common grid ({len(series.common_grid)} points)",
    )
    ax.scatter(
        series.common_grid[::25],
        series.aligned_values[representative, ::25],
        color=GREEN,
        s=9,
        zorder=3,
    )
    ax.set_xlim(*RANGE_CM1)
    ax.legend(frameon=False, fontsize=11)
    save(fig, "03_common_grid_interpolation.png")


def render_qc(series: SubjectSeries) -> None:
    count = len(series.rows)
    passed = int(np.count_nonzero(series.keep))
    repeat_number = np.arange(1, count + 1)
    fig, ax = axes("Within-Subject Repeat QC", "Robust spectral distance")
    ax.scatter(
        repeat_number[series.keep],
        series.distances[series.keep],
        color=GREEN,
        s=22,
        label=f"QC pass (n={passed})",
    )
    ax.scatter(
        repeat_number[~series.keep],
        series.distances[~series.keep],
        color=RED,
        marker="x",
        s=42,
        linewidth=1.6,
        label=f"Excluded (n={count - passed})",
    )
    ax.axhline(
        series.qc_limit,
        color=NAVY,
        linestyle="--",
        linewidth=1.4,
        label=f"Median/MAD limit = {series.qc_limit:.2f}",
    )
    ax.set_xlabel("Repeat measurement")
    ax.legend(frameon=False, fontsize=11)
    save(fig, "04_within_subject_robust_qc.png")


def render_retained(series: SubjectSeries) -> None:
    count = len(series.rows)
    passed = int(np.count_nonzero(series.keep))
    fig, ax = axes("QC-Retained Repeat Spectra", "Intensity")
    ax.plot(
        series.common_grid,
        series.aligned_values[series.keep].T,
        color=GREEN,
        linewidth=0.45,
        alpha=0.16,
    )
    ax.plot(
        series.common_grid,
        series.aligned_values[~series.keep].T,
        color=RED,
        linewidth=0.7,
        alpha=0.5,
    )
    ax.set_xlim(*RANGE_CM1)
    ax.plot([], [], color=GREEN, label=f"Retained (n={passed})")
    ax.plot([], [], color=RED, label=f"Excluded (n={count - passed})")
    ax.legend(frameon=False, fontsize=11)
    save(fig, "05_qc_retained_spectra.png")


def render_mean(series: SubjectSeries) -> None:
    passed = int(np.count_nonzero(series.keep))
    mean = series.aligned_values[series.keep].mean(axis=0)
    fig, ax = axes("Subject Mean Spectrum", "Mean intensity")
    ax.plot(
        series.common_grid,
        series.aligned_values[series.keep].T,
        color=GRAY,
        linewidth=0.4,
        alpha=0.1,
    )
    ax.plot(
        series.common_grid,
        mean,
        color=NAVY,
        linewidth=1.8,
        label=f"Arithmetic mean of QC-passed repeats (n={passed})",
    )
    ax.set_xlim(*RANGE_CM1)
    ax.legend(
        frameon=False,
        fontsize=11,
        title="No additional spectral preprocessing after averaging",
        title_fontsize=10,
    )
    save(fig, "06_subject_mean_spectrum.png")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    page = load_spectra()
    series = select_subject(page, load_calibrations())
    representative = int(np.argmin(series.distances))
    render_spectra(series)
    render_calibration(series, representative)
    render_interpolation(series, representative)
    render_qc(series)
    render_retained(series)
    render_mean(series)
    write_manifest(series, page.total, OUTPUT_DIR)
    passed = int(np.count_nonzero(series.keep))
    print(
        f"Rendered {len(series.rows)} real repeats: {passed} QC pass, {len(series.rows) - passed} excluded"
    )


if __name__ == "__main__":
    main()
