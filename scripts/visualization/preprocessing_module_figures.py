#!/usr/bin/env python3
"""Render one 300 dpi figure per current preprocessing module.

The primary requested plotting implementation is available as the sibling R
script at scripts/visualization/r/preprocessing_module_figures.R. This Python
version exists so the figures can be rendered in environments where R is not
installed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


DEFAULT_INPUT = Path(
    "/Users/ian/Desktop/임상데이터/20260508_Urine test/1. NOR/NOR 68_1.CSV"
)
DEFAULT_OUTPUT_DIR = Path("figures/preprocessing_modules")

TARGET_WN = 1001.4
CALIBRATION_WINDOW = 10.0
TRIM_REGION = (400.0, 2200.0)
SMOOTH_WINDOW = 11
SMOOTH_POLY = 3
BASELINE_WINDOW = 101
FIXED_GRID = np.linspace(402.0, 2198.0, 935)

COLORS = {
    "gray": "#6b7280",
    "light_gray": "#9ca3af",
    "black": "#111827",
    "blue": "#2563eb",
    "green": "#059669",
    "purple": "#7c3aed",
    "red": "#dc2626",
    "orange": "#ea580c",
    "cyan": "#0891b2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SERS preprocessing module figures."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_spectrum(path: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path, header=None, names=["wn", "intensity"])
        return df.astype({"wn": float, "intensity": float})

    rng = np.random.default_rng(7)
    wn = np.linspace(50, 3300, 1700)
    baseline = 12000 * np.exp(-wn / 1900) + 1200
    peaks = (
        4500 * np.exp(-0.5 * ((wn - TARGET_WN) / 10) ** 2)
        + 2500 * np.exp(-0.5 * ((wn - 720) / 18) ** 2)
        + 3200 * np.exp(-0.5 * ((wn - 1450) / 28) ** 2)
    )
    intensity = baseline + peaks + rng.normal(0, 120, size=wn.shape)
    return pd.DataFrame({"wn": wn, "intensity": intensity})


def rolling_minimum(values: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(values)
        .rolling(window=window, center=True, min_periods=1)
        .min()
        .to_numpy()
    )


def snv(values: np.ndarray) -> np.ndarray:
    std = values.std()
    if std < 1e-10:
        return values - values.mean()
    return (values - values.mean()) / std


def detect_reference_peak(df: pd.DataFrame) -> float:
    mask = df["wn"].between(
        TARGET_WN - CALIBRATION_WINDOW, TARGET_WN + CALIBRATION_WINDOW
    )
    local = df.loc[mask, ["wn", "intensity"]].copy()
    local["smooth"] = savgol_filter(local["intensity"], 7, 2, mode="interp")
    return float(local.loc[local["smooth"].idxmax(), "wn"])


def module_axes(title: str, xlabel: str, ylabel: str) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(12, 7.6), dpi=300)
    fig.suptitle(title, fontsize=28, weight="bold", color="#1f2937", y=0.97)
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.14, top=0.76)
    ax.set_xlabel(xlabel, fontsize=23, weight="bold", labelpad=12)
    ax.set_ylabel(ylabel, fontsize=23, weight="bold", labelpad=12)
    ax.tick_params(axis="both", labelsize=20, colors="#374151")
    ax.grid(True, which="major", color="#e5e7eb", linewidth=0.65)
    ax.grid(False, which="minor")
    for spine in ax.spines.values():
        spine.set_color("#d1d5db")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    return fig, ax


def finish(fig: plt.Figure, ax: plt.Axes, output_path: Path) -> None:
    handles, labels = ax.get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.89),
        ncol=min(3, max(1, len(labels))),
        frameon=False,
        fontsize=20,
        title_fontsize=21,
    )
    if legend:
        for line in legend.get_lines():
            line.set_linewidth(3.0)
    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)


def render_figures(df: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    detected = detect_reference_peak(df)
    shift = TARGET_WN - detected
    df = df.copy()
    df["wn_cal"] = df["wn"] + shift

    trim_mask = df["wn_cal"].between(*TRIM_REGION)
    trimmed = df.loc[trim_mask, ["wn_cal", "intensity"]].copy()
    trimmed["smooth"] = savgol_filter(
        trimmed["intensity"],
        SMOOTH_WINDOW,
        SMOOTH_POLY,
        mode="interp",
    )
    trimmed["baseline"] = rolling_minimum(trimmed["smooth"].to_numpy(), BASELINE_WINDOW)
    trimmed["corrected"] = trimmed["smooth"] - trimmed["baseline"]
    trimmed["snv"] = snv(trimmed["corrected"].to_numpy())

    fixed_intensity = np.interp(FIXED_GRID, trimmed["wn_cal"], trimmed["snv"])
    fixed = pd.DataFrame({"wn": FIXED_GRID, "intensity": fixed_intensity})

    outputs: dict[str, Path] = {}

    fig, ax = module_axes(
        "Wavenumber Calibration", "Raman shift (cm-1)", "Intensity"
    )
    cal_before = df[df["wn"].between(970, 1035)]
    cal_after = df[df["wn_cal"].between(970, 1035)]
    ax.plot(
        cal_before["wn"],
        cal_before["intensity"],
        color=COLORS["gray"],
        linewidth=1.6,
        label="Before calibration",
    )
    ax.plot(
        cal_after["wn_cal"],
        cal_after["intensity"],
        color=COLORS["blue"],
        linewidth=1.6,
        label="After calibration",
    )
    ax.axvline(
        TARGET_WN,
        color=COLORS["black"],
        linestyle="--",
        linewidth=1.5,
        label="1001.4 cm-1 target",
    )
    outputs["Wavenumber Calibration"] = output_dir / "01_wavenumber_calibration.png"
    finish(fig, ax, outputs["Wavenumber Calibration"])

    fig, ax = module_axes("Fingerprint Trim", "Raman shift (cm-1)", "Intensity")
    excluded_left = df[df["wn_cal"] < TRIM_REGION[0]]
    kept = df[df["wn_cal"].between(*TRIM_REGION)]
    excluded_right = df[df["wn_cal"] > TRIM_REGION[1]]
    ax.plot(
        excluded_left["wn_cal"],
        excluded_left["intensity"],
        color=COLORS["light_gray"],
        linewidth=1.1,
        label="Excluded",
    )
    ax.plot(
        kept["wn_cal"],
        kept["intensity"],
        color=COLORS["green"],
        linewidth=1.1,
        label="Kept",
    )
    ax.plot(
        excluded_right["wn_cal"],
        excluded_right["intensity"],
        color=COLORS["light_gray"],
        linewidth=1.1,
    )
    ax.axvline(TRIM_REGION[0], color=COLORS["black"], linestyle="--", linewidth=1.4)
    ax.axvline(TRIM_REGION[1], color=COLORS["black"], linestyle="--", linewidth=1.4)
    outputs["Fingerprint Trim"] = output_dir / "02_fingerprint_trim.png"
    finish(fig, ax, outputs["Fingerprint Trim"])

    fig, ax = module_axes("Smoothing", "Raman shift (cm-1)", "Intensity")
    ax.plot(
        trimmed["wn_cal"],
        trimmed["intensity"],
        color=COLORS["light_gray"],
        linewidth=1.25,
        linestyle=(0, (2, 2)),
        alpha=0.9,
        label="Trimmed raw",
    )
    ax.plot(
        trimmed["wn_cal"],
        trimmed["smooth"],
        color=COLORS["purple"],
        linewidth=1.6,
        label="Savitzky-Golay",
    )
    outputs["Smoothing"] = output_dir / "03_smoothing.png"
    finish(fig, ax, outputs["Smoothing"])

    fig, ax = module_axes(
        "Baseline Correction", "Raman shift (cm-1)", "Corrected intensity"
    )
    ax.axhline(
        0,
        color=COLORS["gray"],
        linestyle="--",
        linewidth=1.3,
        label="Zero baseline",
    )
    ax.plot(
        trimmed["wn_cal"],
        trimmed["corrected"],
        color=COLORS["red"],
        linewidth=1.4,
        label="Baseline-corrected",
    )
    ax.set_ylim(bottom=0)
    outputs["Baseline Correction"] = output_dir / "04_baseline_correction.png"
    finish(fig, ax, outputs["Baseline Correction"])

    fig, ax = module_axes("SNV Normalization", "Raman shift (cm-1)", "SNV intensity")
    ax.axhline(
        0,
        color=COLORS["gray"],
        linestyle="--",
        linewidth=1.3,
        label="Zero mean",
    )
    ax.plot(
        trimmed["wn_cal"],
        trimmed["snv"],
        color=COLORS["orange"],
        linewidth=1.4,
        label="SNV",
    )
    outputs["SNV Normalization"] = output_dir / "05_snv_normalization.png"
    finish(fig, ax, outputs["SNV Normalization"])

    fig, ax = module_axes(
        "Fixed Grid Resampling", "Raman shift (cm-1)", "SNV intensity"
    )
    ax.plot(
        fixed["wn"],
        fixed["intensity"],
        color=COLORS["cyan"],
        linewidth=1.4,
        label="Fixed grid",
    )
    marker_idx = np.arange(0, len(fixed), 25)
    ax.scatter(
        fixed.loc[marker_idx, "wn"],
        fixed.loc[marker_idx, "intensity"],
        color=COLORS["cyan"],
        s=12,
        alpha=0.8,
        label="Grid points",
    )
    outputs["Fixed Grid Resampling"] = output_dir / "06_fixed_grid_resampling.png"
    finish(fig, ax, outputs["Fixed Grid Resampling"])

    summary = output_dir / "README.md"
    summary.write_text(
        "\n".join(
            [
                "# Preprocessing Module Figures",
                "",
                "Current repo preprocessing defaults represented here:",
                "",
                "- Wavenumber calibration: urea 1001.4 cm-1, +/-10 cm-1 search",
                "- Fingerprint trim: 400-2200 cm-1",
                "- Smoothing: Savitzky-Golay window 11, polynomial 3",
                "- Baseline correction: rolling minimum window 101",
                "  - Baseline-corrected intensity is nonnegative before SNV.",
                "- Normalization: SNV",
                "  - SNV mean-centers the baseline-corrected spectrum, so the final model input can contain negative values.",
                "- Fixed grid resampling: 402-2198 cm-1, 935 points",
                "",
                f"Example spectrum: `{df.attrs.get('source', 'synthetic')}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    outputs["README"] = summary
    return outputs


def main() -> None:
    args = parse_args()
    df = load_spectrum(args.input)
    df.attrs["source"] = str(args.input if args.input.exists() else "synthetic")
    outputs = render_figures(df, args.output_dir)
    for module, path in outputs.items():
        print(f"{module}: {path}")


if __name__ == "__main__":
    main()
